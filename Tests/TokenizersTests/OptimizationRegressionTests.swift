import Foundation
import Testing

@testable import Tokenizers

@Suite("Optimization regressions")
struct OptimizationRegressionTests {
    @Test("WordPiece accepts continuation spellings at the start and rolls back incomplete words")
    func wordPiecePrefixAndRollback() throws {
        for prefix in ["##", "▁", ""] {
            let vocabulary = try Vocabulary(entries: [("[UNK]", 0), (prefix + "a", 1), (prefix + "b", 2)])
            let tokenizer = WordpieceTokenizer(vocabulary: vocabulary, prefix: prefix)
            #expect(tokenizer.tokenize(word: prefix + "ab") == [prefix + "a", prefix + "b"])
            var scratch = WordpieceTokenizer.Scratch()
            var ids = [123, 456]
            var word = prefix + "abc"
            let encoded = word.withUTF8 { tokenizer.encode($0, into: &ids, scratch: &scratch) }
            #expect(!encoded)
            #expect(ids == [123, 456])
        }
    }

    @Test("Bounded cache preserves exact keys and ids through collisions and arena resets")
    func cacheEviction() {
        let cache = PretokenCache(.init(slotBits: 4, keyArenaCapacity: 128, idsArenaCapacity: 128))
        // A hit must be exact; eviction is always allowed. Exercise byte-distinct equivalent
        // strings, the byte-level discriminator, collisions, and both arena limits.
        var expected: [Data: [Int]] = [:]
        for index in 0..<400 {
            var key = index.isMultiple(of: 2) ? "à-\(index)" : "a\u{300}-\(index)"
            let byteLevel = index.isMultiple(of: 3)
            let value = (0..<(index % 64 + 1)).map { index * 64 + $0 }
            let tagged = Data(key.utf8) + Data([byteLevel ? 1 : 0])
            expected[tagged] = value
            key.withUTF8 { bytes in
                cache.insert(bytes, byteLevel: byteLevel, ids: value[...])
                var actual = [-1]
                #expect(cache.lookup(bytes, byteLevel: byteLevel, into: &actual))
                #expect(actual == [-1] + value)
            }
            for (tagged, value) in expected {
                tagged.withUnsafeBytes { raw in
                    let bytes = raw.bindMemory(to: UInt8.self)
                    var actual = [-1]
                    if cache.lookup(
                        UnsafeBufferPointer(rebasing: bytes.dropLast()), byteLevel: bytes.last! == 1, into: &actual)
                    {
                        #expect(actual == [-1] + value)
                    } else {
                        #expect(actual == [-1])
                    }
                }
            }
        }
    }

    @Test("Pooled WordPiece encoders agree with and without the shared cache")
    func wordPieceCacheContention() async throws {
        let data: Config = ["model": ["type": "WordPiece", "vocab": ["[UNK]": 0, "a": 1, "##b": 2]]]
        let model = try BertTokenizer(tokenizerConfig: [:], tokenizerData: data, addedTokens: [:])
        func encode() -> [Int] {
            let encoder = model.makeEncoder()
            encoder.begin()
            defer { encoder.finish() }
            var ids: [Int] = []
            for text in ["ab", "abc", "a", "ab"] {
                encoder.encode(piece: Substring(text), byteLevel: false, into: &ids)
            }
            return ids
        }
        #expect(encode() == [1, 2, 0, 1, 1, 2])
        model.cache.lock.lock()
        #expect(encode() == [1, 2, 0, 1, 1, 2])
        model.cache.lock.unlock()
        await withTaskGroup(of: [Int].self) { group in
            for _ in 0..<32 { group.addTask { encode() } }
            for await ids in group { #expect(ids == [1, 2, 0, 1, 1, 2]) }
        }
    }
}
