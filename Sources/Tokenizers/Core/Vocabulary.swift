// Packed, immutable token vocabulary.
//
// * Token strings are stored back-to-back in one UTF-8 buffer with an offsets table, so
//   `id → token` is two reads and a String materialization (no dictionary probe,
//   no NSString bridging).
// * `token → id` is served by an open-addressing hash index over the raw UTF-8 bytes,
//   which makes lookups binary-distinct (no Unicode canonical folding) and allows
//   querying directly from a byte slice without allocating a `String`.
// * Every table is allocated with its exact size and filled once during `init`, then read
//   through raw pointers: a probe is a hash, two loads and a `memcmp`, with no reference
//   counting, exclusivity checking or copy-on-write traffic. Presence is one bit per id.

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

    // MARK: - Construction

    convenience init(vocab: [BinaryDistinctString: Config], addedTokens: [String: Int]) throws {
        var entries: [(String, Int)] = []
        entries.reserveCapacity(vocab.count + addedTokens.count)
        for (key, value) in vocab {
            guard let id = value.integer() else { continue }
            entries.append((key.string, id))
        }
        for (token, id) in addedTokens {
            entries.append((token, id))
        }
        try self.init(entries: entries)
    }

    convenience init(vocab: Config, addedTokens: [String: Int]) throws {
        if let packed = vocab.asPackedStringMap() {
            try self.init(packed: packed, addedTokens: addedTokens)
        } else if let dict = vocab.dictionary() {
            try self.init(vocab: dict, addedTokens: addedTokens)
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
    convenience init(scored: PackedScoredTokens, addedTokens: [String: Int]) throws {
        var ids = [Int32](repeating: 0, count: scored.count)
        for i in 0..<scored.count { ids[i] = Int32(i) }
        try self.init(
            packed: PackedStringMap(utf8: scored.utf8, offsets: scored.offsets, ids: ids), addedTokens: addedTokens)
    }

    private init(packed: PackedStringMap, extra: [(String, Int)]) throws {
        var maxId = -1
        for id in packed.ids {
            let i = Int(id)
            guard i >= 0, i < 64_000_000 else { throw TokenizerError.malformedVocab }
            if i > maxId { maxId = i }
        }
        var extraBytes = 0
        for (token, id) in extra {
            guard id >= 0, id < 64_000_000 else { throw TokenizerError.malformedVocab }
            if id > maxId { maxId = id }
            extraBytes += token.utf8.count
        }
        let count = maxId + 1
        guard count <= 64_000_000 else { throw TokenizerError.malformedVocab }

        let presentBits = Self.allocateBits(count)
        var packedLo = [UInt32](repeating: 0, count: count)
        var packedHi = [UInt32](repeating: 0, count: count)
        for i in 0..<packed.count {
            let id = Int(packed.ids[i])
            packedLo[id] = packed.offsets[i]
            packedHi[id] = packed.offsets[i + 1]
            Self.set(bit: id, in: presentBits)
        }
        // Index into `extra` per id (`-1`: none); dense so the layout loop performs no hashing.
        var extraIndexById = [Int32](repeating: -1, count: count)
        for (index, (_, id)) in extra.enumerated() {
            extraIndexById[id] = Int32(index)
            Self.set(bit: id, in: presentBits)
        }

        var totalBytes = extraBytes
        for id in 0..<count where Self.contains(id: id, in: presentBits) && extraIndexById[id] < 0 {
            totalBytes += Int(packedHi[id] - packedLo[id])
        }

        let bytes = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: max(totalBytes, 1))
        let offsets = UnsafeMutableBufferPointer<UInt32>.allocate(capacity: count + 1)
        var populated = 0
        var cursor = 0
        packed.utf8.withUnsafeBufferPointer { packedBytes in
            for id in 0..<count {
                offsets[id] = UInt32(cursor)
                guard Self.contains(id: id, in: presentBits) else { continue }
                populated += 1
                let extraIndex = extraIndexById[id]
                if extraIndex >= 0 {
                    var token = extra[Int(extraIndex)].0
                    let length = token.utf8.count
                    token.withUTF8 { source in
                        (bytes.baseAddress! + cursor).update(from: source.baseAddress!, count: length)
                    }
                    cursor += length
                } else if packedHi[id] > packedLo[id] {
                    let length = Int(packedHi[id] - packedLo[id])
                    (bytes.baseAddress! + cursor).update(
                        from: packedBytes.baseAddress! + Int(packedLo[id]), count: length)
                    cursor += length
                }
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

    /// Builds a vocabulary. Later entries win on string collisions (so added tokens can
    /// override base vocabulary ids); the id → string mapping keeps every id populated.
    init(entries: [(String, Int)]) throws {
        var maxId = -1
        for (_, id) in entries {
            guard id >= 0, id < 64_000_000 else { throw TokenizerError.malformedVocab }
            if id > maxId { maxId = id }
        }
        let count = maxId + 1
        guard count <= 64_000_000 else { throw TokenizerError.malformedVocab }

        // Lay out tokens by id. If several strings claim the same id, the last one wins.
        var byId = [String?](repeating: nil, count: count)
        for (token, id) in entries {
            byId[id] = token
        }
        let presentBits = Self.allocateBits(count)
        var totalBytes = 0
        for id in 0..<count {
            guard let token = byId[id] else { continue }
            Self.set(bit: id, in: presentBits)
            totalBytes += token.utf8.count
        }

        let bytes = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: max(totalBytes, 1))
        let offsets = UnsafeMutableBufferPointer<UInt32>.allocate(capacity: count + 1)
        var populated = 0
        var cursor = 0
        for id in 0..<count {
            offsets[id] = UInt32(cursor)
            guard var token = byId[id] else { continue }
            populated += 1
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
