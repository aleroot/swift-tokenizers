// Compact buffers for the two large tables in `tokenizer.json`: BPE/WordPiece
// `model.vocab` (token → id) and BPE `model.merges` (ranked pairs), plus Unigram
// `model.vocab` (token, score). Parsed directly from UTF-8 so the generic Config
// tree never materializes 100k+ dictionary/array nodes. Storage is page-backed
// (``PageBuffer``) so dropping the parsed `Config` returns the memory to the OS.

import Foundation

/// Compact `token → id` map as serialized in BPE/WordPiece `model.vocab`.
public struct PackedStringMap: Sendable, Hashable {
    let utf8: PageBuffer<UInt8>
    /// `count + 1` offsets into `utf8`. Token `i` is `utf8[offsets[i]..<offsets[i + 1]]`.
    let offsets: PageBuffer<UInt32>
    let ids: PageBuffer<Int32>

    public var count: Int { ids.count }

    init(utf8: PageBuffer<UInt8>, offsets: PageBuffer<UInt32>, ids: PageBuffer<Int32>) {
        self.utf8 = utf8
        self.offsets = offsets
        self.ids = ids
    }

    init(utf8: [UInt8], offsets: [UInt32], ids: [Int32]) {
        self.init(utf8: PageBuffer(utf8), offsets: PageBuffer(offsets), ids: PageBuffer(ids))
    }

    public static func == (lhs: PackedStringMap, rhs: PackedStringMap) -> Bool {
        lhs.ids == rhs.ids && lhs.offsets == rhs.offsets && lhs.utf8 == rhs.utf8
    }

    public func hash(into hasher: inout Hasher) {
        ids.hash(into: &hasher)
        utf8.hash(into: &hasher)
    }

    @inline(__always)
    func withBytes<R>(at index: Int, _ body: (UnsafeBufferPointer<UInt8>) throws -> R) rethrows -> R {
        let lo = Int(offsets[index])
        let hi = Int(offsets[index + 1])
        return try utf8.withUnsafeBufferPointer { buffer in
            try body(UnsafeBufferPointer(rebasing: buffer[lo..<hi]))
        }
    }

    /// Linear scan; used only for rare `Config` subscripts, not the encode path.
    func id(of bytes: UnsafeBufferPointer<UInt8>) -> Int32 {
        for i in 0..<count {
            let found: Int32 = withBytes(at: i) { slice in
                if slice.count == bytes.count,
                    slice.count == 0
                        || memcmp(slice.baseAddress!, bytes.baseAddress!, slice.count) == 0
                {
                    return ids[i]
                }
                return -1
            }
            if found >= 0 { return found }
        }
        return -1
    }

    func materializeDictionary() -> [BinaryDistinctString: Config] {
        var dict = [BinaryDistinctString: Config](minimumCapacity: count)
        utf8.withUnsafeBufferPointer { buffer in
            for i in 0..<count {
                let lo = Int(offsets[i])
                let hi = Int(offsets[i + 1])
                let token = String(decoding: UnsafeBufferPointer(rebasing: buffer[lo..<hi]), as: UTF8.self)
                dict[BinaryDistinctString(token)] = Config(Int(ids[i]))
            }
        }
        return dict
    }
}

/// Compact list of string pairs as serialized in BPE `model.merges`.
public struct PackedStringPairs: Sendable, Hashable {
    let utf8: PageBuffer<UInt8>
    /// Pair `i` is left `utf8[offsets[2i]..<offsets[2i + 1]]` and
    /// right `utf8[offsets[2i + 1]..<offsets[2i + 2]]`.
    let offsets: PageBuffer<UInt32>
    public let count: Int

    init(utf8: PageBuffer<UInt8>, offsets: PageBuffer<UInt32>, count: Int) {
        self.utf8 = utf8
        self.offsets = offsets
        self.count = count
    }

    public static func == (lhs: PackedStringPairs, rhs: PackedStringPairs) -> Bool {
        lhs.count == rhs.count && lhs.offsets == rhs.offsets && lhs.utf8 == rhs.utf8
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(count)
        utf8.hash(into: &hasher)
    }

    @inline(__always)
    func withPair<R>(
        at index: Int, _ body: (UnsafeBufferPointer<UInt8>, UnsafeBufferPointer<UInt8>) throws -> R
    )
        rethrows -> R
    {
        let leftLo = Int(offsets[2 * index])
        let leftHi = Int(offsets[2 * index + 1])
        let rightHi = Int(offsets[2 * index + 2])
        return try utf8.withUnsafeBufferPointer { buffer in
            try body(
                UnsafeBufferPointer(rebasing: buffer[leftLo..<leftHi]),
                UnsafeBufferPointer(rebasing: buffer[leftHi..<rightHi]))
        }
    }

    func materializeArray() -> [Config] {
        var result: [Config] = []
        result.reserveCapacity(count)
        utf8.withUnsafeBufferPointer { buffer in
            for i in 0..<count {
                let leftLo = Int(offsets[2 * i])
                let leftHi = Int(offsets[2 * i + 1])
                let rightHi = Int(offsets[2 * i + 2])
                let a = String(decoding: UnsafeBufferPointer(rebasing: buffer[leftLo..<leftHi]), as: UTF8.self)
                let b = String(decoding: UnsafeBufferPointer(rebasing: buffer[leftHi..<rightHi]), as: UTF8.self)
                result.append(Config([Config(a), Config(b)]))
            }
        }
        return result
    }
}

/// Compact Unigram `model.vocab`: `(token, score)` rows, id = index.
public struct PackedScoredTokens: Sendable, Hashable {
    let utf8: PageBuffer<UInt8>
    let offsets: PageBuffer<UInt32>
    let scores: PageBuffer<Double>

    public var count: Int { scores.count }

    init(utf8: PageBuffer<UInt8>, offsets: PageBuffer<UInt32>, scores: PageBuffer<Double>) {
        self.utf8 = utf8
        self.offsets = offsets
        self.scores = scores
    }

    init(utf8: [UInt8], offsets: [UInt32], scores: [Double]) {
        self.init(utf8: PageBuffer(utf8), offsets: PageBuffer(offsets), scores: PageBuffer(scores))
    }

    public static func == (lhs: PackedScoredTokens, rhs: PackedScoredTokens) -> Bool {
        lhs.scores == rhs.scores && lhs.offsets == rhs.offsets && lhs.utf8 == rhs.utf8
    }

    public func hash(into hasher: inout Hasher) {
        scores.hash(into: &hasher)
        utf8.hash(into: &hasher)
    }

    func token(at index: Int) -> String {
        let lo = Int(offsets[index])
        let hi = Int(offsets[index + 1])
        return utf8.withUnsafeBufferPointer { buffer in
            String(decoding: UnsafeBufferPointer(rebasing: buffer[lo..<hi]), as: UTF8.self)
        }
    }

    func materializeArray() -> [Config] {
        var result: [Config] = []
        result.reserveCapacity(count)
        for i in 0..<count {
            result.append(Config([Config(token(at: i)), Config(scores[i])]))
        }
        return result
    }
}

extension Array {
    /// A copy with capacity equal to its count, for tables built by appending that stay alive
    /// for the tokenizer's lifetime.
    func trimmed() -> [Element] {
        guard capacity > count + count / 8 else { return self }
        return withUnsafeBufferPointer { Array($0) }
    }
}
