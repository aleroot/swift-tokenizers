// Memoises pretoken → token ids. Natural language is highly repetitive (Zipf), so most
// words in a document have already been merged once; a hit costs one hash, one byte
// compare and a memcpy, versus a full BPE merge loop.
//
// The cache is a fixed-size open-addressing table over two append-only arenas held in
// manually managed buffers (no bounds checks, no exclusivity checks, no copy-on-write). It
// performs no allocation on lookup or insert until an arena fills and the table is reset.
// Access must be serialised by the owner: callers acquire `lock` with `tryLock()` for the
// duration of one `encode` call and simply skip the cache when another thread holds it.

import Foundation

final class PretokenCache: @unchecked Sendable {
    let lock = UnfairLock()

    /// 16 bytes: the arenas are bounded, so 32-bit offsets and hashes suffice.
    private struct Slot {
        var hash: UInt32 = 0
        var keyOffset: UInt32 = 0
        var idsOffset: UInt32 = 0
        var keyLength: UInt8 = 0
        var idsCount: UInt8 = 0
        var byteLevel: Bool = false
        var used: Bool = false
    }

    private static let slotCount = 1 << 15
    private static let maxProbes = 8
    private static let maxKeyLength = 64
    private static let maxIdsCount = 64
    private static let keyArenaCapacity = 1 << 20
    private static let idsArenaCapacity = 1 << 18

    private let slots: UnsafeMutablePointer<Slot>
    private let keyArena: UnsafeMutablePointer<UInt8>
    /// Token ids are stored as 32-bit values (vocabularies are far below 2³¹ entries).
    private let idsArena: UnsafeMutablePointer<Int32>
    private var keyCount = 0
    private var idsCount = 0
    private let mask = slotCount - 1

    init() {
        slots = .allocate(capacity: Self.slotCount)
        slots.initialize(repeating: Slot(), count: Self.slotCount)
        keyArena = .allocate(capacity: Self.keyArenaCapacity)
        idsArena = .allocate(capacity: Self.idsArenaCapacity)
    }

    deinit {
        slots.deinitialize(count: Self.slotCount)
        slots.deallocate()
        keyArena.deallocate()
        idsArena.deallocate()
    }

    /// Looks up `bytes`; on a hit appends the cached ids to `ids` and returns `true`.
    @inline(__always)
    func lookup(_ bytes: UnsafeBufferPointer<UInt8>, byteLevel: Bool, into ids: inout [Int]) -> Bool {
        let n = bytes.count
        guard n <= Self.maxKeyLength, let base = bytes.baseAddress else { return false }
        let hash = ByteHash.hash(bytes)
        let tag = UInt32(truncatingIfNeeded: hash >> 32)
        var slot = Int(truncatingIfNeeded: hash) & mask
        for _ in 0..<Self.maxProbes {
            let s = slots[slot]
            if !s.used { return false }
            if s.hash == tag, s.byteLevel == byteLevel, Int(s.keyLength) == n,
                memcmp(keyArena + Int(s.keyOffset), base, n) == 0
            {
                let cached = idsArena + Int(s.idsOffset)
                for i in 0..<Int(s.idsCount) { ids.append(Int(cached[i])) }
                return true
            }
            slot = (slot + 1) & mask
        }
        return false
    }

    /// Records the ids for `bytes`.
    func insert(_ bytes: UnsafeBufferPointer<UInt8>, byteLevel: Bool, ids: ArraySlice<Int>) {
        let n = bytes.count
        guard n <= Self.maxKeyLength, ids.count <= Self.maxIdsCount, !ids.isEmpty, let base = bytes.baseAddress else {
            return
        }
        if keyCount + n > Self.keyArenaCapacity || idsCount + ids.count > Self.idsArenaCapacity {
            reset()
        }
        let hash = ByteHash.hash(bytes)
        let tag = UInt32(truncatingIfNeeded: hash >> 32)
        var slot = Int(truncatingIfNeeded: hash) & mask
        var target = slot
        for _ in 0..<Self.maxProbes {
            if !slots[slot].used {
                target = slot
                break
            }
            slot = (slot + 1) & mask
        }
        // If every probe was occupied, `target` evicts the home slot.

        let keyOffset = keyCount
        (keyArena + keyOffset).update(from: base, count: n)
        keyCount += n

        let idsOffset = idsCount
        for (i, id) in ids.enumerated() { idsArena[idsOffset + i] = Int32(truncatingIfNeeded: id) }
        idsCount += ids.count

        slots[target] = Slot(
            hash: tag,
            keyOffset: UInt32(keyOffset),
            idsOffset: UInt32(idsOffset),
            keyLength: UInt8(n),
            idsCount: UInt8(ids.count),
            byteLevel: byteLevel,
            used: true
        )
    }

    private func reset() {
        for i in 0..<Self.slotCount { slots[i].used = false }
        keyCount = 0
        idsCount = 0
    }
}
