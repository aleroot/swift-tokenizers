// Maps a pair of symbol ids to its merge rank and the id of the merged symbol.
//
// Entries live in dense arrays (pair, rank, merged id) and an open-addressing index of
// 32-bit entry references resolves lookups: 4 bytes per slot instead of a 16-byte
// key/value pair, which halves the footprint of a 150k-merge table (Qwen: 8.4 → 4.4 MB)
// while a lookup stays one hash, one probe sequence and no allocation.

struct MergeTable: Sendable {
    /// `left << 32 | right` per entry.
    private var pairs: [UInt64] = []
    private var ranks: [UInt32] = []
    private var mergedIds: [Int32] = []
    /// Entry index + 1; `0` marks an empty slot.
    private var slots: [UInt32]
    private let mask: Int

    var count: Int { pairs.count }

    init(expectedCount: Int) {
        var capacity = 16
        while capacity < expectedCount * 2 { capacity <<= 1 }
        slots = [UInt32](repeating: 0, count: capacity)
        mask = capacity - 1
        pairs.reserveCapacity(expectedCount)
        ranks.reserveCapacity(expectedCount)
        mergedIds.reserveCapacity(expectedCount)
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
            let reference = slots[slot]
            if reference == 0 {
                pairs.append(key)
                ranks.append(rank)
                mergedIds.append(merged)
                slots[slot] = UInt32(pairs.count)
                return
            }
            let entry = Int(reference - 1)
            if pairs[entry] == key {
                ranks[entry] = rank
                mergedIds[entry] = merged
                return
            }
            slot = (slot + 1) & mask
        }
    }

    /// Entry index for the pair, or `-1`. Negative ids denote symbols outside the vocabulary
    /// and never merge.
    @inline(__always)
    private func entry(left: Int32, right: Int32) -> Int {
        guard left >= 0, right >= 0 else { return -1 }
        let key = Self.key(left, right)
        var slot = Int(truncatingIfNeeded: ByteHash.hash(key: key)) & mask
        while true {
            let reference = slots[slot]
            if reference == 0 { return -1 }
            let entry = Int(reference - 1)
            if pairs[entry] == key { return entry }
            slot = (slot + 1) & mask
        }
    }

    /// Returns `(rank, mergedId)` for the pair, or `nil` if the pair never merges.
    @inline(__always)
    func lookup(left: Int32, right: Int32) -> (rank: UInt32, merged: Int32)? {
        let entry = entry(left: left, right: right)
        return entry < 0 ? nil : (ranks[entry], mergedIds[entry])
    }

    /// Rank only (`UInt32.max` when absent).
    @inline(__always)
    func rank(left: Int32, right: Int32) -> UInt32 {
        let entry = entry(left: left, right: right)
        return entry < 0 ? UInt32.max : ranks[entry]
    }
}
