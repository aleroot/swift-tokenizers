import Foundation

// MARK: - Flat scalar trie (hot path)

/// A prefix tree over Unicode scalars with nodes stored in flat arrays. Each terminal node
/// carries a token id. Children are resolved through a single hash probe keyed by
/// `(node, scalar)`, so a common-prefix walk performs one lookup per scalar and allocates
/// nothing.
struct ScalarTrie: Sendable {
    private static let emptyKey = UInt64.max
    private var keys: [UInt64]
    private var values: [Int32]
    private var mask: Int
    private var count = 0
    private var tokenIds: [Int32] = [-1]  // root

    var nodeCount: Int { tokenIds.count }

    init() {
        keys = [UInt64](repeating: Self.emptyKey, count: 1024)
        values = [Int32](repeating: 0, count: 1024)
        mask = 1023
    }

    @inline(__always)
    private static func key(_ node: Int32, _ scalar: UInt32) -> UInt64 {
        (UInt64(UInt32(bitPattern: node)) << 32) | UInt64(scalar)
    }

    @inline(__always)
    private func child(_ node: Int32, _ scalar: UInt32) -> Int32? {
        let key = Self.key(node, scalar)
        var slot = Int(truncatingIfNeeded: ByteHash.hash(key: key)) & mask
        while true {
            let existing = keys[slot]
            if existing == key { return values[slot] }
            if existing == Self.emptyKey { return nil }
            slot = (slot + 1) & mask
        }
    }

    private mutating func setChild(_ node: Int32, _ scalar: UInt32, _ child: Int32) {
        if (count + 1) * 2 > keys.count { grow() }
        let key = Self.key(node, scalar)
        var slot = Int(truncatingIfNeeded: ByteHash.hash(key: key)) & mask
        while keys[slot] != Self.emptyKey { slot = (slot + 1) & mask }
        keys[slot] = key
        values[slot] = child
        count += 1
    }

    private mutating func grow() {
        let oldKeys = keys
        let oldValues = values
        let capacity = keys.count * 2
        keys = [UInt64](repeating: Self.emptyKey, count: capacity)
        values = [Int32](repeating: 0, count: capacity)
        mask = capacity - 1
        for (i, key) in oldKeys.enumerated() where key != Self.emptyKey {
            var slot = Int(truncatingIfNeeded: ByteHash.hash(key: key)) & mask
            while keys[slot] != Self.emptyKey { slot = (slot + 1) & mask }
            keys[slot] = key
            values[slot] = oldValues[i]
        }
    }

    mutating func insert(_ token: String, id: Int32) {
        var node: Int32 = 0
        for scalar in token.unicodeScalars {
            if let next = child(node, scalar.value) {
                node = next
            } else {
                let next = Int32(tokenIds.count)
                tokenIds.append(-1)
                setChild(node, scalar.value, next)
                node = next
            }
        }
        tokenIds[Int(node)] = id
    }

    /// Walks the trie along `scalars` starting at `start`, calling `body(length, tokenId)` for
    /// every vocabulary entry that is a prefix of the remaining input. Stops at the first
    /// scalar with no child. `length` is in scalars.
    @inline(__always)
    func forEachPrefix(of scalars: UnsafeBufferPointer<Unicode.Scalar>, from start: Int, _ body: (Int, Int32) -> Void) {
        var node: Int32 = 0
        var i = start
        while i < scalars.count {
            guard let next = child(node, scalars[i].value) else { return }
            node = next
            i += 1
            let id = tokenIds[Int(node)]
            if id >= 0 { body(i - start, id) }
        }
    }
}

// MARK: - Generic trie (compatibility)

struct Trie<T: Hashable> {
    typealias Node = TrieNode<T>

    var root: Node

    init(root: Node? = nil) {
        self.root = root ?? Node()
    }
}

extension Trie {
    func insert(_ element: any Sequence<T>) {
        var node = root
        for item in element {
            if let child = node.children[item] {
                node = child
            } else {
                let child = Node()
                node.children[item] = child
                node = child
            }
        }
        node.isLeaf = true
    }

    func append(contentsOf container: any Sequence<any Sequence<T>>) {
        for t in container {
            insert(t)
        }
    }

    /// All leaf nodes that are prefixes of `text`.
    func commonPrefixSearch(_ text: any Sequence<T>) -> [[T]] {
        var node = root
        var seqs: [[T]] = []
        var seq: [T] = []
        for item in text {
            seq.append(item)
            guard let child = node.children[item] else { return seqs }
            node = child
            if node.isLeaf {
                seqs.append(seq)
            }
        }
        return seqs
    }

    func commonPrefixSearchIterator(_ text: any Sequence<T>) -> LeavesWithCommonPrefixIterator<T> {
        LeavesWithCommonPrefixIterator(node: root, text: text)
    }

    func get(_ element: any Sequence<T>) -> Node? {
        var node = root
        for item in element {
            guard let child = node.children[item] else { return nil }
            node = child
        }
        return node
    }
}

final class TrieNode<T: Hashable> {
    var isLeaf: Bool = false
    var children: [T: TrieNode] = [:]
}

struct LeavesWithCommonPrefixIterator<T: Hashable>: Sequence, IteratorProtocol {
    var node: TrieNode<T>
    var text: any Sequence<T>
    var seq: [T] = []
    lazy var iterator = text.makeIterator() as any IteratorProtocol<T>

    mutating func next() -> [T]? {
        while true {
            guard let item = iterator.next() else { return nil }
            seq.append(item)
            guard let child = node.children[item] else { return nil }
            node = child
            if node.isLeaf {
                return seq
            }
        }
    }
}
