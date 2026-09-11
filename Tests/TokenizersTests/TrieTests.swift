//
// Created by Pedro Cuenca on 12/1/24.

import Foundation
import Testing

@testable import Tokenizers

@Suite("Trie data structure functionality")
struct TrieTests {
    @Test("Trie building and traversal")
    func trieBuilding() {
        // https://guillaume-be.github.io/2020-05-30/sentence_piece
        let trie = Trie<Character>()
        trie.insert("cat")
        trie.insert("carp")
        trie.insert("car")
        #expect(trie.root.children.count == 1)

        let c = trie.get("c")
        #expect(c != nil)
        #expect(c!.children.count == 1)  // "a"

        let ca = trie.get("ca")
        #expect(ca != nil)
        #expect(ca!.children.count == 2)  // "r", "t"

        let car = trie.get("car")
        #expect(car != nil)
        #expect(car!.isLeaf)
        #expect(!ca!.isLeaf)

        #expect(trie.get("card") == nil)
    }

    @Test("Trie common prefix search")
    func trieCommonPrefixSearch() {
        // https://guillaume-be.github.io/2020-05-30/sentence_piece
        let trie = Trie<Character>()
        trie.insert("cat")
        trie.insert("carp")
        trie.insert("car")

        // trie.commonPrefixSearch returns [Character] not String
        let leaves = trie.commonPrefixSearch("carpooling").map { String($0) }
        #expect(leaves == ["car", "carp"])
    }

    @Test("Trie common prefix search iterator")
    func trieCommonPrefixSearchIterator() {
        // https://guillaume-be.github.io/2020-05-30/sentence_piece
        let trie = Trie<Character>()
        trie.insert("cat")
        trie.insert("carp")
        trie.insert("car")

        var expected = Set(["car", "carp"])
        for leaf in trie.commonPrefixSearchIterator("carpooling").map({ String($0) }) {
            #expect(expected.contains(leaf))
            expected.remove(leaf)
        }
        #expect(expected.count == 0)
    }
}

@Suite("Double-array trie")
struct DoubleArrayTrieTests {
    private func makeTrie(_ keys: [String]) -> DoubleArrayTrie {
        var utf8: [UInt8] = []
        var offsets: [UInt32] = [0]
        for key in keys {
            utf8.append(contentsOf: Array(key.utf8))
            offsets.append(UInt32(utf8.count))
        }
        return utf8.withUnsafeBufferPointer { utf8 in
            offsets.withUnsafeBufferPointer { offsets in
                DoubleArrayTrie(utf8: utf8, offsets: offsets, count: keys.count)
            }
        }
    }

    private func prefixes(_ trie: DoubleArrayTrie, _ text: String) -> [(Int, Int32)] {
        var copy = text
        var result: [(Int, Int32)] = []
        copy.withUTF8 { bytes in trie.forEachPrefix(of: bytes, from: 0) { result.append(($0, $1)) } }
        return result
    }

    @Test("Exact lookup and common-prefix enumeration")
    func lookup() {
        let keys = ["cat", "carp", "car", "c", "▁", "▁the", "▁there", "", "日本", "日"]
        let trie = makeTrie(keys)
        #expect(trie.keyCount == keys.count - 1)  // the empty key carries no value
        for (index, key) in keys.enumerated() where !key.isEmpty {
            var copy = key
            #expect(copy.withUTF8 { trie.value(of: $0) } == Int32(index))
        }
        var missing = "card"
        #expect(missing.withUTF8 { trie.value(of: $0) } == -1)
        var empty = ""
        #expect(empty.withUTF8 { trie.value(of: $0) } == -1)

        let carpooling = prefixes(trie, "carpooling")
        #expect(carpooling.map(\.0) == [1, 3, 4])  // "c", "car", "carp"
        #expect(carpooling.map(\.1) == [3, 2, 1])
        let there = prefixes(trie, "▁thereafter")
        #expect(there.map(\.0) == [3, 6, 8])  // byte lengths of "▁", "▁the", "▁there"
        #expect(there.map(\.1) == [4, 5, 6])
        #expect(prefixes(trie, "日本語").map(\.0) == [3, 6])
        #expect(prefixes(trie, "xyz").isEmpty)
    }

    @Test("Duplicate keys keep the highest index; every unique id round-trips")
    func duplicatesAndRoundTrip() {
        let keys = ["a", "b", "a", "ab", "b"]
        let trie = makeTrie(keys)
        var a = "a"
        var b = "b"
        #expect(a.withUTF8 { trie.value(of: $0) } == 2)
        #expect(b.withUTF8 { trie.value(of: $0) } == 4)
        #expect(trie.keyCount == 3)

        // A vocabulary-sized random key set with many shared prefixes and long fan-out.
        var generator = SystemRandomNumberGenerator()
        let alphabet = Array("abcdefghijklmnopqrstuvwxyzäöü▁日本語😀".unicodeScalars)
        var random: [String] = []
        for _ in 0..<20_000 {
            let length = Int.random(in: 1...12, using: &generator)
            random.append(
                String(String.UnicodeScalarView((0..<length).map { _ in alphabet.randomElement(using: &generator)! })))
        }
        let unique = Array(Set(random))
        let big = makeTrie(unique)
        #expect(big.keyCount == unique.count)
        for (index, key) in unique.enumerated() {
            var copy = key
            #expect(copy.withUTF8 { big.value(of: $0) } == Int32(index))
        }
        // Prefix enumeration agrees with brute force.
        let lookup = Dictionary(uniqueKeysWithValues: unique.enumerated().map { ($1, Int32($0)) })
        for key in unique.prefix(500) {
            var copy = key
            let expected: [(Int, Int32)] = copy.withUTF8 { bytes in
                (1...bytes.count).compactMap { length in
                    let candidate = String(decoding: UnsafeBufferPointer(rebasing: bytes[0..<length]), as: UTF8.self)
                    guard candidate.utf8.count == length, let id = lookup[candidate] else { return nil }
                    return (length, id)
                }
            }
            let actual = prefixes(big, key)
            #expect(actual.map(\.0) == expected.map(\.0))
            #expect(actual.map(\.1) == expected.map(\.1))
        }
    }
}
