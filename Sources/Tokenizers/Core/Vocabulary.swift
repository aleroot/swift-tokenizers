// Packed, immutable token vocabulary.
//
// * Token strings are stored back-to-back in one UTF-8 buffer with an offsets table, so
//   `id → token` is two array reads and a String materialization (no dictionary probe,
//   no NSString bridging).
// * `token → id` is served by an open-addressing hash index over the raw UTF-8 bytes,
//   which makes lookups binary-distinct (no Unicode canonical folding) and allows
//   querying directly from a byte slice without allocating a `String`.

import Foundation

final class Vocabulary: Sendable {
    /// Size of the dense id space (`maxId + 1`). Ids in `0..<count` may still be absent
    /// when the source vocabulary has holes; check ``contains(id:)``.
    let count: Int

    /// Number of ids actually populated.
    let populatedCount: Int

    private let storage: [UInt8]
    private let offsets: [UInt32]
    private let present: [Bool]
    private let slots: [UInt32]  // id + 1 ; 0 == empty
    private let mask: Int

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
        try self.init(packed: PackedStringMap(utf8: scored.utf8, offsets: scored.offsets, ids: ids), addedTokens: addedTokens)
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

        var packedLo = [UInt32](repeating: 0, count: count)
        var packedHi = [UInt32](repeating: 0, count: count)
        var present = [Bool](repeating: false, count: count)
        for i in 0..<packed.count {
            let id = Int(packed.ids[i])
            packedLo[id] = packed.offsets[i]
            packedHi[id] = packed.offsets[i + 1]
            present[id] = true
        }
        var extraById: [Int: String] = [:]
        extraById.reserveCapacity(extra.count)
        for (token, id) in extra {
            extraById[id] = token
            present[id] = true
        }

        var totalBytes = extraBytes
        for id in 0..<count where present[id] && extraById[id] == nil {
            totalBytes += Int(packedHi[id] - packedLo[id])
        }

        var storage: [UInt8] = []
        storage.reserveCapacity(totalBytes)
        var offsets = [UInt32](repeating: 0, count: count + 1)
        var populated = 0
        packed.utf8.withUnsafeBufferPointer { packedBytes in
            for id in 0..<count {
                offsets[id] = UInt32(storage.count)
                if var token = extraById[id] {
                    populated += 1
                    token.withUTF8 { storage.append(contentsOf: $0) }
                } else if present[id] {
                    populated += 1
                    let lo = Int(packedLo[id])
                    let hi = Int(packedHi[id])
                    storage.append(contentsOf: UnsafeBufferPointer(rebasing: packedBytes[lo..<hi]))
                }
            }
        }
        offsets[count] = UInt32(storage.count)

        var capacity = 16
        while capacity < (packed.count + extra.count) * 2 { capacity <<= 1 }
        var slots = [UInt32](repeating: 0, count: capacity)
        let mask = capacity - 1

        storage.withUnsafeBufferPointer { storageBuffer in
            func insert(bytes: UnsafeBufferPointer<UInt8>, id: Int) {
                var slot = Int(truncatingIfNeeded: ByteHash.hash(bytes)) & mask
                while true {
                    let existing = slots[slot]
                    if existing == 0 {
                        slots[slot] = UInt32(id + 1)
                        return
                    }
                    let existingId = Int(existing - 1)
                    let lo = Int(offsets[existingId])
                    let hi = Int(offsets[existingId + 1])
                    if hi - lo == bytes.count,
                        bytes.count == 0
                            || memcmp(storageBuffer.baseAddress! + lo, bytes.baseAddress!, bytes.count) == 0
                    {
                        slots[slot] = UInt32(id + 1)
                        return
                    }
                    slot = (slot + 1) & mask
                }
            }
            packed.utf8.withUnsafeBufferPointer { packedBytes in
                for i in 0..<packed.count {
                    let lo = Int(packed.offsets[i])
                    let hi = Int(packed.offsets[i + 1])
                    insert(bytes: UnsafeBufferPointer(rebasing: packedBytes[lo..<hi]), id: Int(packed.ids[i]))
                }
            }
            for (token, id) in extra {
                var copy = token
                copy.withUTF8 { insert(bytes: $0, id: id) }
            }
        }

        self.count = count
        self.populatedCount = populated
        self.storage = storage
        self.offsets = offsets
        self.present = present
        self.slots = slots
        self.mask = mask
    }

    convenience init(vocab: [String: Int]) throws {
        try self.init(entries: vocab.map { ($0.key, $0.value) })
    }

    /// Builds a vocabulary. Later entries win on string collisions (so added tokens can
    /// override base vocabulary ids); the id → string mapping keeps every id populated.
    init(entries: [(String, Int)]) throws {
        var maxId = -1
        var totalBytes = 0
        for (token, id) in entries {
            guard id >= 0, id < 64_000_000 else { throw TokenizerError.malformedVocab }
            if id > maxId { maxId = id }
            totalBytes += token.utf8.count
        }
        let count = maxId + 1
        guard count <= 64_000_000 else { throw TokenizerError.malformedVocab }

        // Lay out tokens by id. If several strings claim the same id, the last one wins.
        var byId = [String?](repeating: nil, count: count)
        for (token, id) in entries {
            byId[id] = token
        }

        var storage: [UInt8] = []
        storage.reserveCapacity(totalBytes)
        var offsets = [UInt32](repeating: 0, count: count + 1)
        var present = [Bool](repeating: false, count: count)
        var populated = 0
        for id in 0..<count {
            offsets[id] = UInt32(storage.count)
            if var token = byId[id] {
                present[id] = true
                populated += 1
                token.withUTF8 { storage.append(contentsOf: $0) }
            }
        }
        offsets[count] = UInt32(storage.count)

        // Hash index at load factor <= 0.5.
        var capacity = 16
        while capacity < entries.count * 2 { capacity <<= 1 }
        var slots = [UInt32](repeating: 0, count: capacity)
        let mask = capacity - 1

        storage.withUnsafeBufferPointer { storageBuffer in
            for (token, id) in entries {
                var t = token
                t.withUTF8 { bytes in
                    var slot = Int(truncatingIfNeeded: ByteHash.hash(bytes)) & mask
                    while true {
                        let existing = slots[slot]
                        if existing == 0 {
                            slots[slot] = UInt32(id + 1)
                            return
                        }
                        let existingId = Int(existing - 1)
                        let lo = Int(offsets[existingId])
                        let hi = Int(offsets[existingId + 1])
                        if hi - lo == bytes.count,
                            bytes.count == 0
                                || memcmp(storageBuffer.baseAddress! + lo, bytes.baseAddress!, bytes.count) == 0
                        {
                            // Same string inserted again: later entry wins.
                            slots[slot] = UInt32(id + 1)
                            return
                        }
                        slot = (slot + 1) & mask
                    }
                }
            }
        }

        self.count = count
        self.populatedCount = populated
        self.storage = storage
        self.offsets = offsets
        self.present = present
        self.slots = slots
        self.mask = mask
    }

    // MARK: - Lookup

    /// Runs `body` with the packed token storage and its `count + 1` offsets table.
    func withStorage<R>(_ body: (UnsafeBufferPointer<UInt8>, UnsafeBufferPointer<UInt32>) throws -> R) rethrows -> R {
        try storage.withUnsafeBufferPointer { storage in
            try offsets.withUnsafeBufferPointer { offsets in
                try body(storage, offsets)
            }
        }
    }

    @inline(__always)
    func contains(id: Int) -> Bool {
        id >= 0 && id < count && present[id]
    }

    /// The token string for `id`, or `nil` if the id is not populated.
    func token(_ id: Int) -> String? {
        guard contains(id: id) else { return nil }
        let lo = Int(offsets[id])
        let hi = Int(offsets[id + 1])
        return storage.withUnsafeBufferPointer { buffer in
            String(decoding: UnsafeBufferPointer(rebasing: buffer[lo..<hi]), as: UTF8.self)
        }
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
        return try storage.withUnsafeBufferPointer { buffer in
            try body(UnsafeBufferPointer(rebasing: buffer[lo..<hi]))
        }
    }

    /// Appends the raw UTF-8 bytes of token `id` to `output`.
    @inline(__always)
    func appendBytes(of id: Int, to output: inout [UInt8]) {
        let lo = Int(offsets[id])
        let hi = Int(offsets[id + 1])
        storage.withUnsafeBufferPointer { buffer in
            output.append(contentsOf: UnsafeBufferPointer(rebasing: buffer[lo..<hi]))
        }
    }

    /// Id of the token whose UTF-8 bytes equal `bytes`, or `-1`.
    @inline(__always)
    func id(of bytes: UnsafeBufferPointer<UInt8>) -> Int32 {
        var slot = Int(truncatingIfNeeded: ByteHash.hash(bytes)) & mask
        let n = bytes.count
        return storage.withUnsafeBufferPointer { storageBuffer -> Int32 in
            while true {
                let existing = slots[slot]
                if existing == 0 { return -1 }
                let id = Int(existing - 1)
                let lo = Int(offsets[id])
                let hi = Int(offsets[id + 1])
                if hi - lo == n {
                    if n == 0 || memcmp(storageBuffer.baseAddress! + lo, bytes.baseAddress!, n) == 0 {
                        return Int32(id)
                    }
                }
                slot = (slot + 1) & mask
            }
        }
    }

    /// Id of `token`, or `nil`.
    func id(of token: String) -> Int? {
        var copy = token
        let result = copy.withUTF8 { id(of: $0) }
        return result < 0 ? nil : Int(result)
    }

    /// Id of `token`, or `nil`.
    func id(of token: Substring) -> Int? {
        var copy = Substring(token)
        let result = copy.withUTF8 { id(of: $0) }
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
            id(of: UnsafeBufferPointer(rebasing: raw.bindMemory(to: UInt8.self)[0..<n]))
        }
    }

    /// Enumerates every populated `(id, token)` pair.
    func forEach(_ body: (Int, String) -> Void) {
        for id in 0..<count where present[id] {
            body(id, token(id)!)
        }
    }
}
