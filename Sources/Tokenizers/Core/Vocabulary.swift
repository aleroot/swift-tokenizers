// Packed, immutable token vocabulary: token UTF-8 stored back-to-back with an offsets table for
// `id → token`, and an open-addressing byte-hash index for binary-distinct `token → id` lookups
// straight from byte slices.

import Foundation

/// `@unchecked Sendable`: the tables are written during initialization and are read-only
/// afterwards, so the raw pointers are stable for the object's lifetime.
final class Vocabulary: @unchecked Sendable {
    /// Size of the dense id space (`maxId + 1`). Ids in `0..<count` may still be absent
    /// when the source vocabulary has holes; check ``contains(id:)``.
    let count: Int

    /// Number of ids actually populated.
    let populatedCount: Int

    private let bytes: UnsafeMutableBufferPointer<UInt8>
    private let storageCount: Int
    private let offsets: UnsafeMutableBufferPointer<UInt32>
    /// One bit per id.
    private let presentBits: UnsafeMutableBufferPointer<UInt64>
    private let slots: UnsafeMutableBufferPointer<UInt32>  // id + 1 ; 0 == empty

    /// Raw view of the tables for lookup loops.
    let lookup: Lookup

    /// The hash index over the token bytes.
    struct Lookup {
        let bytes: UnsafePointer<UInt8>
        let offsets: UnsafePointer<UInt32>
        let slots: UnsafePointer<UInt32>
        let mask: Int

        /// Id of the token whose UTF-8 bytes equal `key`, or `-1`.
        @inline(__always)
        func id(of key: UnsafeBufferPointer<UInt8>) -> Int32 {
            var slot = Int(truncatingIfNeeded: ByteHash.hash(key)) & mask
            let n = key.count
            while true {
                let existing = slots[slot]
                if existing == 0 { return -1 }
                let id = Int(existing - 1)
                let lo = Int(offsets[id])
                if Int(offsets[id + 1]) - lo == n, n == 0 || memcmp(bytes + lo, key.baseAddress!, n) == 0 {
                    return Int32(id)
                }
                slot = (slot + 1) & mask
            }
        }
    }

    deinit {
        bytes.deallocate()
        offsets.deallocate()
        presentBits.deallocate()
        slots.deallocate()
    }

    // MARK: - Id space

    /// Where an id's spelling comes from while the tables are being laid out.
    private enum Source {
        case absent
        case packed(Int32)
        case extra(Int32)
    }

    /// Largest dense id space the tables may span, whatever the file declares. `id → token` is
    /// a direct index, so an id costs a slot whether or not it is populated: 4 Mi ids is 16 MB
    /// of offsets and sixteen times the largest published vocabulary.
    static let maximumIdSpace = 4 << 20

    /// The id space a vocabulary of `entryCount` entries may span. Published vocabularies are
    /// contiguous or nearly so, so a wide margin over the entry count still rejects a file that
    /// declares one token at a huge id purely to force a large allocation. The floor keeps
    /// small vocabularies working when their special tokens sit at a base model's high ids.
    static func idSpace(entryCount: Int) -> Int {
        let margin = entryCount < maximumIdSpace ? entryCount * 64 : maximumIdSpace
        return min(maximumIdSpace, max(1 << 18, margin))
    }

    /// The dense id count for a vocabulary, or a `malformedVocab` error when the ids are too
    /// sparse for the dense tables to be a reasonable representation.
    private static func denseCount(maxId: Int, entryCount: Int) throws -> Int {
        guard maxId < idSpace(entryCount: entryCount) else { throw TokenizerError.malformedVocab }
        return maxId + 1
    }

    // MARK: - Construction

    convenience init(vocab: [BinaryDistinctString: Config], addedTokens: [String: Int]) throws {
        try self.init(vocab: vocab, extra: addedTokens.map { ($0.key, $0.value) })
    }

    private convenience init(vocab: [BinaryDistinctString: Config], extra: [(String, Int)]) throws {
        var entries: [(String, Int)] = []
        entries.reserveCapacity(vocab.count + extra.count)
        for (key, value) in vocab {
            guard let id = value.integer() else { continue }
            entries.append((key.string, id))
        }
        entries.append(contentsOf: extra)
        try self.init(entries: entries)
    }

    convenience init(vocab: Config, addedTokens: [String: Int], addedTokenConfig: Config = Config()) throws {
        let extra = Self.addedTokenEntries(addedTokens, config: addedTokenConfig)
        if let packed = vocab.asPackedStringMap() {
            try self.init(packed: packed, extra: extra)
        } else if let dict = vocab.dictionary() {
            try self.init(vocab: dict, extra: extra)
        } else {
            throw TokenizerError.missingVocab
        }
    }

    convenience init(packed: PackedStringMap, addedTokens: [String: Int]) throws {
        var extra: [(String, Int)] = []
        extra.reserveCapacity(addedTokens.count)
        for (token, id) in addedTokens { extra.append((token, id)) }
        try self.init(packed: packed, extra: extra)
    }

    /// Builds from a packed Unigram vocabulary (`id == index`) plus added tokens.
    convenience init(scored: PackedScoredTokens, addedTokens: [String: Int], addedTokenConfig: Config = Config()) throws
    {
        let ids = PageBuffer<Int32>(capacity: scored.count)
        for i in 0..<scored.count { ids.append(Int32(i)) }
        try self.init(
            packed: PackedStringMap(utf8: scored.utf8, offsets: scored.offsets, ids: ids),
            extra: Self.addedTokenEntries(addedTokens, config: addedTokenConfig))
    }

    /// Serialized IDs preserve distinct spellings that a Swift String dictionary can collapse.
    private static func addedTokenEntries(_ fallback: [String: Int], config: Config) -> [(String, Int)] {
        var byId: [Int: String] = [:]
        for (token, id) in fallback { byId[id] = token }
        for token in config.array(or: []) {
            guard let id = token["id"].integer(), let content = token.content.string() else { continue }
            byId[id] = content
        }
        return byId.map { ($0.value, $0.key) }
    }

    private init(packed: PackedStringMap, extra: [(String, Int)]) throws {
        var maxId = -1
        try packed.ids.withUnsafeBufferPointer { ids in
            for id in ids {
                let i = Int(id)
                guard i >= 0 else { throw TokenizerError.malformedVocab }
                if i > maxId { maxId = i }
            }
        }
        for (_, id) in extra {
            guard id >= 0 else { throw TokenizerError.malformedVocab }
            if id > maxId { maxId = id }
        }
        let count = try Self.denseCount(maxId: maxId, entryCount: packed.count + extra.count)

        let presentBits = Self.allocateBits(count)
        // Where each id's spelling comes from, so the layout pass needs no hashing and only one
        // `count`-sized table. Added tokens override a packed row with the same id.
        var sources = [Source](repeating: .absent, count: count)
        for i in 0..<packed.count {
            let id = Int(packed.ids[i])
            sources[id] = .packed(Int32(i))
            Self.set(bit: id, in: presentBits)
        }
        for (index, (_, id)) in extra.enumerated() {
            sources[id] = .extra(Int32(index))
            Self.set(bit: id, in: presentBits)
        }

        var totalBytes = 0
        for source in sources {
            switch source {
            case .absent: break
            case let .packed(i): totalBytes += Int(packed.offsets[Int(i) + 1] - packed.offsets[Int(i)])
            case let .extra(i): totalBytes += extra[Int(i)].0.utf8.count
            }
        }

        let bytes = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: max(totalBytes, 1))
        let offsets = UnsafeMutableBufferPointer<UInt32>.allocate(capacity: count + 1)
        var populated = 0
        var cursor = 0
        packed.utf8.withUnsafeBufferPointer { packedBytes in
            for id in 0..<count {
                offsets[id] = UInt32(cursor)
                switch sources[id] {
                case .absent:
                    continue
                case let .extra(index):
                    var token = extra[Int(index)].0
                    let length = token.utf8.count
                    token.withUTF8 { source in
                        (bytes.baseAddress! + cursor).update(from: source.baseAddress!, count: length)
                    }
                    cursor += length
                case let .packed(index):
                    let lo = Int(packed.offsets[Int(index)])
                    let length = Int(packed.offsets[Int(index) + 1]) - lo
                    if length > 0 {
                        (bytes.baseAddress! + cursor).update(from: packedBytes.baseAddress! + lo, count: length)
                        cursor += length
                    }
                }
                populated += 1
            }
        }
        offsets[count] = UInt32(cursor)

        var capacity = 16
        while capacity < (packed.count + extra.count) * 2 { capacity <<= 1 }
        let slots = Self.allocateSlots(capacity)
        let mask = capacity - 1
        packed.utf8.withUnsafeBufferPointer { packedBytes in
            for i in 0..<packed.count {
                let lo = Int(packed.offsets[i])
                let hi = Int(packed.offsets[i + 1])
                Self.insert(
                    key: UnsafeBufferPointer(start: packedBytes.baseAddress! + lo, count: hi - lo),
                    id: Int(packed.ids[i]),
                    bytes: bytes.baseAddress!, offsets: UnsafeBufferPointer(offsets), slots: slots, mask: mask)
            }
        }
        let offsetView = UnsafeBufferPointer(offsets)
        for (token, id) in extra {
            var copy = token
            copy.withUTF8 {
                Self.insert(
                    key: $0, id: id, bytes: bytes.baseAddress!, offsets: offsetView, slots: slots, mask: mask)
            }
        }

        self.count = count
        populatedCount = populated
        self.bytes = bytes
        storageCount = cursor
        self.offsets = offsets
        self.presentBits = presentBits
        self.slots = slots
        lookup = Lookup(bytes: bytes.baseAddress!, offsets: offsets.baseAddress!, slots: slots.baseAddress!, mask: mask)
    }

    convenience init(vocab: [String: Int]) throws {
        try self.init(entries: vocab.map { ($0.key, $0.value) })
    }

    /// Drops entries whose id the dense tables cannot represent, so a caller-supplied
    /// dictionary can never fail. Ids are embedding indices: a negative or absurdly large one
    /// cannot be honoured, and refusing the whole vocabulary would be worse than ignoring it.
    convenience init(retaining vocab: [String: Int]) {
        let limit = Self.idSpace(entryCount: vocab.count)
        var entries: [(String, Int)] = []
        entries.reserveCapacity(vocab.count)
        for (token, id) in vocab where id >= 0 && id < limit { entries.append((token, id)) }
        self.init(validated: entries, count: entries.reduce(0) { max($0, $1.1 + 1) })
    }

    /// Builds a vocabulary. Later entries win on string collisions (so added tokens can
    /// override base vocabulary ids); the id → string mapping keeps every id populated.
    convenience init(entries: [(String, Int)]) throws {
        var maxId = -1
        for (_, id) in entries {
            guard id >= 0 else { throw TokenizerError.malformedVocab }
            if id > maxId { maxId = id }
        }
        self.init(
            validated: entries, count: try Self.denseCount(maxId: maxId, entryCount: entries.count))
    }

    /// - Precondition: every id in `entries` is within `0..<count`.
    private init(validated entries: [(String, Int)], count: Int) {
        // Index of each id's spelling in `entries` (`-1`: unpopulated). Four bytes per id rather
        // than a `[String?]`, which would also retain and release once per slot.
        var entryById = [Int32](repeating: -1, count: count)
        // If several strings claim the same id, the last one wins.
        for (index, (_, id)) in entries.enumerated() { entryById[id] = Int32(index) }

        let presentBits = Self.allocateBits(count)
        var totalBytes = 0
        for id in 0..<count where entryById[id] >= 0 {
            Self.set(bit: id, in: presentBits)
            totalBytes += entries[Int(entryById[id])].0.utf8.count
        }

        let bytes = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: max(totalBytes, 1))
        let offsets = UnsafeMutableBufferPointer<UInt32>.allocate(capacity: count + 1)
        var populated = 0
        var cursor = 0
        for id in 0..<count {
            offsets[id] = UInt32(cursor)
            let index = entryById[id]
            guard index >= 0 else { continue }
            populated += 1
            var token = entries[Int(index)].0
            let length = token.utf8.count
            token.withUTF8 { source in
                (bytes.baseAddress! + cursor).update(from: source.baseAddress!, count: length)
            }
            cursor += length
        }
        offsets[count] = UInt32(cursor)

        // Hash index at load factor <= 0.5.
        var capacity = 16
        while capacity < entries.count * 2 { capacity <<= 1 }
        let slots = Self.allocateSlots(capacity)
        let mask = capacity - 1
        let offsetView = UnsafeBufferPointer(offsets)
        for (token, id) in entries {
            var copy = token
            copy.withUTF8 {
                Self.insert(
                    key: $0, id: id, bytes: bytes.baseAddress!, offsets: offsetView, slots: slots, mask: mask)
            }
        }

        self.count = count
        populatedCount = populated
        self.bytes = bytes
        storageCount = cursor
        self.offsets = offsets
        self.presentBits = presentBits
        self.slots = slots
        lookup = Lookup(bytes: bytes.baseAddress!, offsets: offsets.baseAddress!, slots: slots.baseAddress!, mask: mask)
    }

    // MARK: - Table helpers

    private static func allocateBits(_ count: Int) -> UnsafeMutableBufferPointer<UInt64> {
        let bits = UnsafeMutableBufferPointer<UInt64>.allocate(capacity: (count + 63) / 64)
        bits.initialize(repeating: 0)
        return bits
    }

    private static func allocateSlots(_ capacity: Int) -> UnsafeMutableBufferPointer<UInt32> {
        let slots = UnsafeMutableBufferPointer<UInt32>.allocate(capacity: capacity)
        slots.initialize(repeating: 0)
        return slots
    }

    @inline(__always)
    private static func set(bit id: Int, in bits: UnsafeMutableBufferPointer<UInt64>) {
        bits[id >> 6] |= 1 << UInt64(id & 63)
    }

    @inline(__always)
    private static func contains(id: Int, in bits: UnsafeMutableBufferPointer<UInt64>) -> Bool {
        bits[id >> 6] & (1 << UInt64(id & 63)) != 0
    }

    /// Inserts `id` for `key`, overwriting any existing entry with the same bytes.
    private static func insert(
        key: UnsafeBufferPointer<UInt8>, id: Int, bytes: UnsafePointer<UInt8>,
        offsets: UnsafeBufferPointer<UInt32>, slots: UnsafeMutableBufferPointer<UInt32>, mask: Int
    ) {
        var slot = Int(truncatingIfNeeded: ByteHash.hash(key)) & mask
        while true {
            let existing = slots[slot]
            if existing == 0 {
                slots[slot] = UInt32(id + 1)
                return
            }
            let existingId = Int(existing - 1)
            let lo = Int(offsets[existingId])
            if Int(offsets[existingId + 1]) - lo == key.count,
                key.count == 0 || memcmp(bytes + lo, key.baseAddress!, key.count) == 0
            {
                slots[slot] = UInt32(id + 1)
                return
            }
            slot = (slot + 1) & mask
        }
    }

    // MARK: - Lookup

    /// Runs `body` with the packed token storage and its `count + 1` offsets table.
    func withStorage<R>(_ body: (UnsafeBufferPointer<UInt8>, UnsafeBufferPointer<UInt32>) throws -> R) rethrows -> R {
        try body(
            UnsafeBufferPointer(start: bytes.baseAddress, count: storageCount),
            UnsafeBufferPointer(start: offsets.baseAddress, count: count + 1))
    }

    @inline(__always)
    func contains(id: Int) -> Bool {
        id >= 0 && id < count && Self.contains(id: id, in: presentBits)
    }

    /// The token string for `id`, or `nil` if the id is not populated.
    func token(_ id: Int) -> String? {
        guard contains(id: id) else { return nil }
        let lo = Int(offsets[id])
        let hi = Int(offsets[id + 1])
        return String(decoding: UnsafeBufferPointer(start: bytes.baseAddress! + lo, count: hi - lo), as: UTF8.self)
    }

    /// Byte length of token `id` (0 for unpopulated ids).
    @inline(__always)
    func byteCount(of id: Int) -> Int {
        Int(offsets[id + 1] - offsets[id])
    }

    /// Runs `body` with the raw UTF-8 bytes of token `id`.
    @inline(__always)
    func withBytes<R>(of id: Int, _ body: (UnsafeBufferPointer<UInt8>) throws -> R) rethrows -> R {
        let lo = Int(offsets[id])
        let hi = Int(offsets[id + 1])
        return try body(UnsafeBufferPointer(start: bytes.baseAddress! + lo, count: hi - lo))
    }

    /// Appends the raw UTF-8 bytes of token `id` to `output`.
    @inline(__always)
    func appendBytes(of id: Int, to output: inout [UInt8]) {
        let lo = Int(offsets[id])
        let hi = Int(offsets[id + 1])
        output.append(contentsOf: UnsafeBufferPointer(start: bytes.baseAddress! + lo, count: hi - lo))
    }

    /// Id of the token whose UTF-8 bytes equal `bytes`, or `-1`.
    @inline(__always)
    func id(of bytes: UnsafeBufferPointer<UInt8>) -> Int32 {
        lookup.id(of: bytes)
    }

    /// Id of `token`, or `nil`.
    func id(of token: String) -> Int? {
        var copy = token
        let result = copy.withUTF8 { lookup.id(of: $0) }
        return result < 0 ? nil : Int(result)
    }

    /// Id of `token`, or `nil`.
    func id(of token: Substring) -> Int? {
        var copy = Substring(token)
        let result = copy.withUTF8 { lookup.id(of: $0) }
        return result < 0 ? nil : Int(result)
    }

    /// Id of the token consisting of the single scalar `scalar`, or `-1`.
    func id(ofScalar scalar: Unicode.Scalar) -> Int32 {
        var buffer: (UInt8, UInt8, UInt8, UInt8) = (0, 0, 0, 0)
        let n = withUnsafeMutableBytes(of: &buffer) { raw -> Int in
            var i = 0
            for byte in UTF8.encode(scalar)! {
                raw[i] = byte
                i += 1
            }
            return i
        }
        return withUnsafeBytes(of: &buffer) { raw in
            lookup.id(of: UnsafeBufferPointer(rebasing: raw.bindMemory(to: UInt8.self)[0..<n]))
        }
    }

    /// Enumerates every populated `(id, token)` pair.
    func forEach(_ body: (Int, String) -> Void) {
        for id in 0..<count where contains(id: id) {
            body(id, token(id)!)
        }
    }
}
