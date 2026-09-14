// Static byte-keyed trie in double-array form (Aoe 1989, as popularised by Darts).
//
// Every node is one 12-byte unit `(base, check, value)`. The transition from `node` on byte
// `c` is `child = base[node] ^ c`, valid iff `check[child]` names `node`, so a prefix walk
// costs one unit load per input byte and no hashing. Nodes are laid out in depth-first order,
// which keeps the chains that dominate real vocabularies on adjacent cache lines.
//
// An all-zero unit is a free slot: slot 0 is left unused so that no node is named by
// `check == 0`, and `value` is stored biased by one. The unit array is therefore built on
// pages that the VM system zero-fills on first touch, so a vocabulary's worth of units never
// has to be written before it is used and the slack in the upper bound the builder reserves
// costs neither time nor memory.
//
// Construction partitions the key set by its next byte one trie level at a time (an MSD
// radix sort fused with node placement), so it never compares whole keys, and allocates
// each node's children into the first 256-aligned block with all required slots free.

import Foundation

/// `@unchecked Sendable`: the builder transfers exclusive ownership of the units at init.
/// No mutable view escapes, and the allocation remains read-only until deinit.
final class DoubleArrayTrie: @unchecked Sendable {
    struct Unit {
        var base: Int32
        /// Owning node; `0` in a free slot, which no node ever names.
        var check: Int32
        /// Stored value plus one; `0` when the node ends no key.
        var value: Int32
    }

    /// Slot 0 is reserved so that a zeroed unit cannot name a node.
    static let root: Int32 = 1

    /// Immutable after `init`; manually managed so hot loops perform no retain/release or
    /// copy-on-write checks.
    private let units: UnsafePointer<Unit>
    /// Number of allocated units, always a multiple of 256 so `base ^ c` stays in range.
    let count: Int
    /// Number of distinct keys that carry a value.
    let keyCount: Int

    deinit {
        units.deallocate()
    }

    /// Builds a trie from the tokens of a vocabulary. Keys are the token strings; values the ids.
    static func make(vocabulary: Vocabulary) -> DoubleArrayTrie {
        vocabulary.withStorage { storage, offsets in
            DoubleArrayTrie(utf8: storage, offsets: offsets, count: vocabulary.count)
        }
    }

    /// Builds a trie whose value for key `i` (`utf8[offsets[i]..<offsets[i + 1]]`) is `i`.
    /// Duplicate keys keep the highest index, like HF's token-to-ID map; the empty key is ignored.
    init(utf8: UnsafeBufferPointer<UInt8>, offsets: UnsafeBufferPointer<UInt32>, count keyTotal: Int) {
        var builder = Builder(utf8: utf8, offsets: offsets, keyTotal: keyTotal)
        builder.build()
        units = UnsafePointer(builder.finish())
        count = builder.capacity
        keyCount = builder.keyCount
    }

    // MARK: - Lookup

    /// Child of `node` on `byte`, or `nil`.
    @inline(__always)
    func child(of node: Int32, byte: UInt8) -> Int32? {
        let child = units[Int(node)].base ^ Int32(byte)
        return units[Int(child)].check == node ? child : nil
    }

    /// Value stored at `node` (`-1` when the node is not a key end).
    @inline(__always)
    func value(at node: Int32) -> Int32 { units[Int(node)].value &- 1 }

    /// Calls `body(length, value)` for every key that is a prefix of `bytes[start...]`, shortest
    /// first. Stops at the first byte without a transition.
    @inline(__always)
    func forEachPrefix(of bytes: UnsafeBufferPointer<UInt8>, from start: Int, _ body: (Int, Int32) -> Void) {
        var node = Self.root
        var i = start
        let end = bytes.count
        while i < end {
            let child = units[Int(node)].base ^ Int32(bytes[i])
            let unit = units[Int(child)]
            if unit.check != node { return }
            node = child
            i += 1
            if unit.value != 0 { body(i - start, unit.value &- 1) }
        }
    }

    /// Value of the key equal to `bytes`, or `-1`.
    func value(of bytes: UnsafeBufferPointer<UInt8>) -> Int32 {
        var node = Self.root
        for byte in bytes {
            guard let next = child(of: node, byte: byte) else { return -1 }
            node = next
        }
        return units[Int(node)].value &- 1
    }

    // MARK: - Construction

    private struct Builder {
        /// Partition label for keys that end at the current depth (sorts before any byte).
        private static let terminal = -1

        let utf8: UnsafeBufferPointer<UInt8>
        let offsets: UnsafeBufferPointer<UInt32>
        let keyTotal: Int

        /// Owners of the large tables below. Pages rather than allocations: a vocabulary's
        /// unit array is tens of megabytes that the allocator would keep resident long after
        /// the trie has been trimmed to the size it ended up needing.
        private var unitPages: PageBlock<Unit>
        private var usedPages: PageBlock<UInt64>
        private let orderPages: PageBlock<UInt64>
        private let keyIndexPages: PageBlock<UInt32>
        private let labelPages: PageBlock<UInt16>

        /// Keys as `start << 32 | end` (byte offsets), permuted in place so each node's keys
        /// form a contiguous range. Carrying both bounds means a partition step touches only
        /// the key bytes themselves. `keyIndex` travels with `order` and yields the value.
        private var order: UnsafeMutablePointer<UInt64>
        private var keyIndex: UnsafeMutablePointer<UInt32>
        private var units: UnsafeMutablePointer<Unit>
        private(set) var capacity: Int
        /// Occupied slots, as a bitmap for fast base search.
        private var used: UnsafeMutablePointer<UInt64>
        private var highWater = 2  // one past the highest used slot
        private var firstFree = 2  // lowest free slot (0 is reserved, 1 is the root)
        private var frontier = 2  // search start for multi-child nodes
        private(set) var keyCount = 0

        /// Per-node scratch: bucket bookkeeping for the counting sort, the partition byte of
        /// each key in the current range, and the resulting child groups `(label, lo, hi)`.
        private let counts: UnsafeMutablePointer<Int>
        private let cursors: UnsafeMutablePointer<Int>
        /// Bucket of each key in the range being partitioned (`label + 1`), so the permutation
        /// pass never re-reads key bytes.
        private let labelOf: UnsafeMutablePointer<UInt16>
        private let groups: UnsafeMutablePointer<(label: Int, lo: Int, hi: Int)>
        /// Depth-first work list of `(node, lo, hi, depth)`.
        private var stack: UnsafeMutablePointer<(node: Int32, lo: Int, hi: Int, depth: Int)>
        private var stackCount = 0
        private var stackCapacity = 4096

        init(utf8: UnsafeBufferPointer<UInt8>, offsets: UnsafeBufferPointer<UInt32>, keyTotal: Int) {
            self.utf8 = utf8
            self.offsets = offsets
            self.keyTotal = keyTotal
            orderPages = PageBlock(count: keyTotal)
            keyIndexPages = PageBlock(count: keyTotal)
            order = orderPages.base
            keyIndex = keyIndexPages.base
            for i in 0..<keyTotal {
                order[i] = UInt64(offsets[i]) << 32 | UInt64(offsets[i + 1])
                keyIndex[i] = UInt32(i)
            }
            let totalBytes = keyTotal == 0 ? 0 : Int(offsets[keyTotal]) - Int(offsets[0])
            // Upper bound on nodes is `totalBytes + 1`; the extra blocks guarantee that a node
            // with up to 256 children can always be placed without growing.
            capacity = (totalBytes + 1 + 512 + 255) & ~255
            unitPages = PageBlock(count: capacity)
            units = unitPages.base
            usedPages = PageBlock(count: capacity / 64)
            used = usedPages.base
            counts = .allocate(capacity: 257)
            cursors = .allocate(capacity: 257)
            labelPages = PageBlock(count: max(keyTotal, 1))
            labelOf = labelPages.base
            groups = .allocate(capacity: 257)
            stack = .allocate(capacity: stackCapacity)
            markUsed(0)
            markUsed(Int(DoubleArrayTrie.root))
        }

        @inline(__always) private mutating func push(_ node: Int32, _ lo: Int, _ hi: Int, _ depth: Int) {
            if stackCount == stackCapacity {
                let grown = UnsafeMutablePointer<(node: Int32, lo: Int, hi: Int, depth: Int)>.allocate(
                    capacity: stackCapacity * 2)
                grown.moveInitialize(from: stack, count: stackCount)
                stack.deallocate()
                stack = grown
                stackCapacity *= 2
            }
            stack[stackCount] = (node, lo, hi, depth)
            stackCount += 1
        }

        /// Hands the unit array (trimmed to the last used 256-block) to the trie.
        /// Copies the trie into an allocation of exactly the size it ended up needing; the
        /// builder's pages are released with it.
        mutating func finish() -> UnsafeMutablePointer<Unit> {
            capacity = (highWater + 255) & ~255
            let compact = UnsafeMutablePointer<Unit>.allocate(capacity: capacity)
            compact.initialize(from: units, count: capacity)
            counts.deallocate()
            cursors.deallocate()
            groups.deallocate()
            stack.deallocate()
            return compact
        }

        @inline(__always) private func start(_ entry: UInt64) -> Int { Int(entry >> 32) }
        @inline(__always) private func end(_ entry: UInt64) -> Int { Int(entry & 0xFFFF_FFFF) }

        /// Partition label of `entry` at `depth`: its byte there, or `terminal` if it ends.
        @inline(__always) private func label(_ entry: UInt64, _ depth: Int) -> Int {
            let position = start(entry) + depth
            return position == end(entry) ? Self.terminal : Int(utf8[position])
        }

        @inline(__always) private func isUsed(_ slot: Int) -> Bool {
            used[slot >> 6] & (1 << UInt64(slot & 63)) != 0
        }

        @inline(__always) private mutating func markUsed(_ slot: Int) {
            used[slot >> 6] |= 1 << UInt64(slot & 63)
            if slot >= highWater { highWater = slot + 1 }
        }

        private mutating func grow() {
            let newCapacity = capacity * 2
            let grownUnits = PageBlock<Unit>(count: newCapacity)
            grownUnits.base.update(from: units, count: capacity)
            unitPages = grownUnits
            units = grownUnits.base
            let grownUsed = PageBlock<UInt64>(count: newCapacity / 64)
            grownUsed.base.update(from: used, count: capacity / 64)
            usedPages = grownUsed
            used = grownUsed.base
            capacity = newCapacity
        }

        /// Finds a `base` such that `base ^ label` is free for every child label (ascending).
        private mutating func findBase(_ groupCount: Int, firstGroup: Int) -> Int {
            while firstFree < capacity, isUsed(firstFree) { firstFree += 1 }
            let childCount = groupCount - firstGroup
            var pos = childCount == 1 ? firstFree : max(firstFree, frontier)
            while true {
                if pos + 256 > capacity { grow() }
                if !isUsed(pos) {
                    let base = pos ^ groups[firstGroup].label
                    var fits = true
                    var g = firstGroup + 1
                    while g < groupCount {
                        if isUsed(base ^ groups[g].label) {
                            fits = false
                            break
                        }
                        g += 1
                    }
                    if fits {
                        if childCount > 1 { frontier = pos }
                        return base
                    }
                }
                pos += 1
            }
        }

        /// Partitions `order[lo..<hi]` by label at `depth` (terminated keys first) and records
        /// the groups in ascending label order. Returns the group count.
        private mutating func partition(lo: Int, hi: Int, depth: Int) -> Int {
            let n = hi - lo
            if n <= 32 {
                // Insertion sort on the bucket, then scan for group boundaries. Avoids touching
                // 257 buckets for the many tiny nodes deep in the trie.
                for i in 0..<n { labelOf[i] = UInt16(label(order[lo + i], depth) + 1) }
                for i in 1..<n {
                    let key = order[lo + i]
                    let index = keyIndex[lo + i]
                    let l = labelOf[i]
                    var j = i - 1
                    while j >= 0, labelOf[j] > l {
                        order[lo + j + 1] = order[lo + j]
                        keyIndex[lo + j + 1] = keyIndex[lo + j]
                        labelOf[j + 1] = labelOf[j]
                        j -= 1
                    }
                    order[lo + j + 1] = key
                    keyIndex[lo + j + 1] = index
                    labelOf[j + 1] = l
                }
                var groupCount = 0
                var k = 0
                while k < n {
                    let l = labelOf[k]
                    var j = k + 1
                    while j < n, labelOf[j] == l { j += 1 }
                    groups[groupCount] = (Int(l) - 1, lo + k, lo + j)
                    groupCount += 1
                    k = j
                }
                return groupCount
            }

            // Counting sort over 257 buckets (bucket 0 = terminated keys).
            for b in 0..<257 { counts[b] = 0 }
            for k in 0..<n {
                let l = label(order[lo + k], depth) + 1
                labelOf[k] = UInt16(l)
                counts[l] += 1
            }
            var position = lo
            var groupCount = 0
            for b in 0..<257 where counts[b] > 0 {
                cursors[b] = position
                groups[groupCount] = (b - 1, position, position + counts[b])
                groupCount += 1
                position += counts[b]
            }
            // In-place permutation (American flag sort): cycle-lead each element home. Buckets
            // travel with their keys so no key byte is re-read.
            for g in 0..<groupCount {
                let (l, _, bucketEnd) = groups[g]
                let b = l + 1
                while cursors[b] < bucketEnd {
                    var key = order[cursors[b]]
                    var index = keyIndex[cursors[b]]
                    var target = Int(labelOf[cursors[b] - lo])
                    while target != b {
                        let slot = cursors[target]
                        cursors[target] += 1
                        swap(&key, &order[slot])
                        swap(&index, &keyIndex[slot])
                        let displaced = Int(labelOf[slot - lo])
                        labelOf[slot - lo] = UInt16(target)
                        target = displaced
                    }
                    order[cursors[b]] = key
                    keyIndex[cursors[b]] = index
                    labelOf[cursors[b] - lo] = UInt16(b)
                    cursors[b] += 1
                }
            }
            return groupCount
        }

        /// Lays out the remainder of a key that no other key shares: a chain of single-child
        /// nodes, each placed in the lowest free slot. Roughly two thirds of all nodes in a
        /// natural-language vocabulary are placed here, bypassing the partition machinery.
        private mutating func placeChain(from node: Int32, keyPosition: Int, depth: Int) {
            let key = order[keyPosition]
            var node = node
            var position = start(key) + depth
            let end = end(key)
            while position < end {
                let label = Int(utf8[position])
                while firstFree < capacity, isUsed(firstFree) { firstFree += 1 }
                if firstFree + 256 > capacity { grow() }
                let base = firstFree ^ label
                units[Int(node)].base = Int32(base)
                let child = firstFree
                markUsed(child)
                units[child].check = node
                node = Int32(child)
                position += 1
            }
            if end > start(key) {
                units[Int(node)].value = Int32(keyIndex[keyPosition]) &+ 1
                keyCount += 1
            }
        }

        mutating func build() {
            guard keyTotal > 0 else { return }
            push(DoubleArrayTrie.root, 0, keyTotal, 0)

            while stackCount > 0 {
                stackCount -= 1
                let (node, lo, hi, depth) = stack[stackCount]
                if hi - lo == 1 {
                    placeChain(from: node, keyPosition: lo, depth: depth)
                    continue
                }
                let groupCount = partition(lo: lo, hi: hi, depth: depth)
                var firstChild = 0
                if groups[0].label == Self.terminal {
                    // Keys ending at this node: the last vocabulary entry wins.
                    if depth > 0 {
                        var value: UInt32 = 0
                        for k in groups[0].lo..<groups[0].hi { value = max(value, keyIndex[k]) }
                        units[Int(node)].value = Int32(value) &+ 1
                        keyCount += 1
                    }
                    firstChild = 1
                }
                if firstChild == groupCount { continue }

                let base = findBase(groupCount, firstGroup: firstChild)
                units[Int(node)].base = Int32(base)
                for g in firstChild..<groupCount {
                    let child = base ^ groups[g].label
                    markUsed(child)
                    units[child].check = node
                }
                // Push in reverse so the first child is processed next (depth-first, in order).
                for g in (firstChild..<groupCount).reversed() {
                    let group = groups[g]
                    push(Int32(base ^ group.label), group.lo, group.hi, depth + 1)
                }
            }
        }
    }
}
