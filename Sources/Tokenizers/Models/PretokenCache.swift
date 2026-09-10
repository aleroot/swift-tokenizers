// Memoises pretoken → token ids in a fixed-size open-addressing table over two append-only
// arenas: no allocation on lookup or insert until an arena fills and the table is reset. Access
// is serialised by the owner, which holds `lock` (`tryLock()`) for one `encode` call; a caller
// that finds it taken memoises into a smaller private table instead (``PretokenCacheLease``).

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

    /// Table and arena sizes. Bounded, so the cache never grows past a few hundred
    /// kilobytes (`.compact`) or a few megabytes (`.standard`) however much text is encoded.
    struct Configuration {
        /// `1 << slotBits` slots of 16 bytes.
        var slotBits: Int
        var keyArenaCapacity: Int
        /// In 32-bit ids.
        var idsArenaCapacity: Int

        /// Large BPE / Unigram vocabularies: ~2.5 MB.
        static let standard = Configuration(slotBits: 15, keyArenaCapacity: 1 << 20, idsArenaCapacity: 1 << 18)
        /// Short, few pre-tokens (WordPiece words): ~0.4 MB.
        static let compact = Configuration(slotBits: 12, keyArenaCapacity: 1 << 16, idsArenaCapacity: 1 << 16)
    }

    private static let maxProbes = 8
    private static let maxKeyLength = 64
    private static let maxIdsCount = 64

    private let slots: UnsafeMutablePointer<Slot>
    private let keyArena: UnsafeMutablePointer<UInt8>
    /// Token ids are stored as 32-bit values (vocabularies are far below 2³¹ entries).
    private let idsArena: UnsafeMutablePointer<Int32>
    private let slotCount: Int
    private let keyArenaCapacity: Int
    private let idsArenaCapacity: Int
    private var keyCount = 0
    private var idsCount = 0
    private let mask: Int

    init(_ configuration: Configuration = .standard) {
        slotCount = 1 << configuration.slotBits
        mask = slotCount - 1
        keyArenaCapacity = configuration.keyArenaCapacity
        idsArenaCapacity = configuration.idsArenaCapacity
        slots = .allocate(capacity: slotCount)
        slots.initialize(repeating: Slot(), count: slotCount)
        keyArena = .allocate(capacity: keyArenaCapacity)
        idsArena = .allocate(capacity: idsArenaCapacity)
    }

    deinit {
        slots.deinitialize(count: slotCount)
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
        if keyCount + n > keyArenaCapacity || idsCount + ids.count > idsArenaCapacity {
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
        for i in 0..<slotCount { slots[i].used = false }
        keyCount = 0
        idsCount = 0
    }
}

/// The cache an encoder uses for one `encode` call: the model's shared cache when no other
/// caller holds it, otherwise a smaller table private to the encoder, so concurrent callers all
/// memoise without ever blocking on each other. After a failed attempt the shared cache is left
/// alone for a number of calls: a contended `tryLock` still bounces its cache line between cores.
struct PretokenCacheLease {
    /// The table to use for the current call.
    private(set) var cache: PretokenCache?
    private var holdsShared = false
    private var backoff = 0
    private var fallback: PretokenCache?

    private static let backoffCalls = 32

    @inline(__always)
    mutating func begin(shared: PretokenCache) {
        if backoff == 0, shared.lock.tryLock() {
            cache = shared
            holdsShared = true
            return
        }
        backoff = backoff > 0 ? backoff - 1 : Self.backoffCalls
        if fallback == nil { fallback = PretokenCache(.compact) }
        cache = fallback
    }

    @inline(__always)
    mutating func finish(shared: PretokenCache) {
        if holdsShared {
            holdsShared = false
            shared.lock.unlock()
        }
        cache = nil
    }
}
