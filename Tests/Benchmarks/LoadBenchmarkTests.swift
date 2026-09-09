// Load-time breakdown for a large SentencePiece Unigram tokenizer (XLM-R family, 250k
// pieces): JSON parse, packed vocabulary, double-array trie. Run with `RUN_BENCHMARKS=1`.

import Foundation
import Testing

@testable import Tokenizers

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct LoadBenchmarkTests {
    static let modelId = "intfloat/multilingual-e5-small"

    @Test("Unigram load stage breakdown")
    func unigramStages() async throws {
        let folder = try await BenchmarkFixtures.modelFolder(Self.modelId)
        let data = try Data(contentsOf: folder.appendingPathComponent("tokenizer.json"))
        print("\n=== Unigram load breakdown (\(Self.modelId), \(data.count / 1024) KB) ===")

        benchmarkMeasure(label: "Config(tokenizerJSON:)", iterations: 10, warmup: 2) {
            _ = try! Config(tokenizerJSON: data)
        }
        let config = try Config(tokenizerJSON: data)
        let packed = config.model.vocab.asPackedScoredTokens()!
        print("    pieces: \(packed.count), \(packed.utf8.count) bytes")

        benchmarkMeasure(label: "Vocabulary(scored:)", iterations: 10, warmup: 2) {
            _ = try! Vocabulary(scored: packed, addedTokens: [:])
        }

        var trie: DoubleArrayTrie?
        benchmarkMeasure(label: "DoubleArrayTrie", iterations: 10, warmup: 2) {
            trie = packed.utf8.withUnsafeBufferPointer { utf8 in
                packed.offsets.withUnsafeBufferPointer { offsets in
                    DoubleArrayTrie(utf8: utf8, offsets: offsets, count: packed.count)
                }
            }
        }
        if let trie {
            print(
                "    units: \(trie.count) (\(trie.count * MemoryLayout<DoubleArrayTrie.Unit>.stride / 1024) KB), keys: \(trie.keyCount)"
            )
        }

        benchmarkMeasure(label: "PrecompiledNormalizer", iterations: 10, warmup: 2) {
            _ = try! NormalizerFactory.fromConfig(config: config.normalizer)
        }

        benchmarkMeasure(label: "AutoTokenizer.load", iterations: 10, warmup: 2) {
            _ = try! AutoTokenizer.load(from: folder)
        }
    }
}
