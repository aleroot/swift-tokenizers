// Maps a pair of symbol ids to its merge rank and merged id. Entries live in dense arrays and
// an open-addressing index of 32-bit entry references resolves lookups. Once built, the tables
// are read through raw pointers so the merge loop touches no reference counts.

/// Accumulates merges in rank order directly into the storage of the ``MergeTable`` it returns,
/// so building never copies the tables.
struct MergeTableBuilder {
    private let table: MergeTable
    private let capacity: Int

    var count: Int { table.count }

    init(expectedCount: Int) {
        capacity = max(expectedCount, 1)
        table = MergeTable(capacity: capacity)
    }

    /// Inserts or overwrites the merge for `(left, right)`.
    mutating func insert(left: Int32, right: Int32, rank: UInt32, merged: Int32) {
        let key = MergeTable.key(left, right)
        let mask = table.mask
        let slots = table.slots
        var slot = Int(truncatingIfNeeded: ByteHash.hash(key: key)) & mask
        while true {
            let reference = slots[slot]
            if reference == 0 {
                // `expectedCount` is the number of `insert` calls, so the arrays never fill up.
                let entry = table.count
                precondition(entry < capacity, "MergeTableBuilder overflow")
                table.pairs[entry] = key
                table.ranks[entry] = rank
                table.mergedIds[entry] = merged
                table.count = entry + 1
                slots[slot] = UInt32(entry + 1)
                return
            }
            let entry = Int(reference - 1)
            if table.pairs[entry] == key {
                table.ranks[entry] = rank
                table.mergedIds[entry] = merged
                return
            }
            slot = (slot + 1) & mask
        }
    }

    func build() -> MergeTable { table }
}

/// `@unchecked Sendable`: the tables are written during initialization and read-only afterwards.
final class MergeTable: @unchecked Sendable {
    fileprivate let pairs: UnsafeMutablePointer<UInt64>
    fileprivate let ranks: UnsafeMutablePointer<UInt32>
    fileprivate let mergedIds: UnsafeMutablePointer<Int32>
    fileprivate let slots: UnsafeMutablePointer<UInt32>
    private let slotCount: Int
    fileprivate let mask: Int

    /// Number of entries; written only by ``MergeTableBuilder``.
    fileprivate(set) var count = 0

    /// Allocates room for `capacity` entries; the builder fills the tables in.
    fileprivate init(capacity: Int) {
        var slotCount = 16
        while slotCount < capacity * 2 { slotCount <<= 1 }
        self.slotCount = slotCount
        mask = slotCount - 1
        pairs = .allocate(capacity: capacity)
        ranks = .allocate(capacity: capacity)
        mergedIds = .allocate(capacity: capacity)
        slots = .allocate(capacity: slotCount)
        slots.initialize(repeating: 0, count: slotCount)
    }

    deinit {
        pairs.deallocate()
        ranks.deallocate()
        mergedIds.deallocate()
        slots.deallocate()
    }

    @inline(__always)
    static func key(_ left: Int32, _ right: Int32) -> UInt64 {
        (UInt64(UInt32(bitPattern: left)) << 32) | UInt64(UInt32(bitPattern: right))
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
