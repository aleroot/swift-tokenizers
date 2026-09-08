// Open-addressing hash table mapping a pair of symbol ids to the merge rank and the id of
// the merged symbol. Keys are packed as `left << 32 | right`, values as `rank << 32 | merged`,
// so a lookup is one hash, one probe sequence and no allocation.

struct MergeTable: Sendable {
    private static let emptyKey = UInt64.max

    private var keys: [UInt64]
    private var values: [UInt64]
    private let mask: Int
    private(set) var count: Int = 0

    init(expectedCount: Int) {
        var capacity = 16
        while capacity < expectedCount * 2 { capacity <<= 1 }
        keys = [UInt64](repeating: Self.emptyKey, count: capacity)
        values = [UInt64](repeating: 0, count: capacity)
        mask = capacity - 1
    }

    @inline(__always)
    private static func key(_ left: Int32, _ right: Int32) -> UInt64 {
        (UInt64(UInt32(bitPattern: left)) << 32) | UInt64(UInt32(bitPattern: right))
    }

    /// Inserts or overwrites the merge for `(left, right)`.
    mutating func insert(left: Int32, right: Int32, rank: UInt32, merged: Int32) {
        let key = Self.key(left, right)
        var slot = Int(truncatingIfNeeded: ByteHash.hash(key: key)) & mask
        while true {
            let existing = keys[slot]
            if existing == Self.emptyKey {
                keys[slot] = key
                values[slot] = (UInt64(rank) << 32) | UInt64(UInt32(bitPattern: merged))
                count += 1
                return
            }
            if existing == key {
                values[slot] = (UInt64(rank) << 32) | UInt64(UInt32(bitPattern: merged))
                return
            }
            slot = (slot + 1) & mask
        }
    }

    /// Returns `(rank, mergedId)` for the pair, or `nil` if the pair never merges.
    /// Negative ids denote symbols outside the vocabulary and never merge.
    @inline(__always)
    func lookup(left: Int32, right: Int32) -> (rank: UInt32, merged: Int32)? {
        guard left >= 0, right >= 0 else { return nil }
        let key = Self.key(left, right)
        var slot = Int(truncatingIfNeeded: ByteHash.hash(key: key)) & mask
        while true {
            let existing = keys[slot]
            if existing == key {
                let v = values[slot]
                return (UInt32(truncatingIfNeeded: v >> 32), Int32(bitPattern: UInt32(truncatingIfNeeded: v)))
            }
            if existing == Self.emptyKey { return nil }
            slot = (slot + 1) & mask
        }
    }

    /// Rank only (`UInt32.max` when absent).
    @inline(__always)
    func rank(left: Int32, right: Int32) -> UInt32 {
        lookup(left: left, right: right)?.rank ?? UInt32.max
    }
}
