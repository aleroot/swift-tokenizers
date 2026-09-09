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

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct TableBenchmarkTests {
    @Test("Unicode classification tables (first access in this process)")
    func tables() {
        var t = DispatchTime.now()
        _ = ScalarClassifier.bmp
        print(
            String(
                format: "  bmp:      %.2f ms", Double(DispatchTime.now().uptimeNanoseconds - t.uptimeNanoseconds) / 1e6)
        )
        t = DispatchTime.now()
        _ = ScalarClassifier.bmpExtra
        print(
            String(
                format: "  bmpExtra: %.2f ms", Double(DispatchTime.now().uptimeNanoseconds - t.uptimeNanoseconds) / 1e6)
        )
    }
}

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct MemoryBreakdownTests {
    private func liveHeap() -> Int {
        var stats = malloc_statistics_t()
        malloc_zone_statistics(nil, &stats)
        return Int(stats.size_in_use)
    }

    @Test("Unigram retained memory by component")
    func unigramMemory() async throws {
        let folder = try await BenchmarkFixtures.modelFolder(LoadBenchmarkTests.modelId)
        let data = try Data(contentsOf: folder.appendingPathComponent("tokenizer.json"))
        let config = try Config(tokenizerJSON: data)
        let packed = config.model.vocab.asPackedScoredTokens()!
        func report(_ label: String, _ make: () throws -> AnyObject) throws {
            let before = liveHeap()
            let object = try make()
            print(String(format: "  %-22@ %6.2f MB", label, Double(liveHeap() - before) / 1_048_576))
            withExtendedLifetime(object) {}
        }
        try report("Vocabulary(scored:)") { try Vocabulary(scored: packed, addedTokens: [:]) }
        try report("DoubleArrayTrie") {
            packed.utf8.withUnsafeBufferPointer { utf8 in
                packed.offsets.withUnsafeBufferPointer { offsets in
                    DoubleArrayTrie(utf8: utf8, offsets: offsets, count: packed.count)
                }
            }
        }
        try report("PretokenCache") { PretokenCache() }
        try report("UnigramTokenizer") {
            try UnigramTokenizer(
                tokenizerConfig: Config([:] as [BinaryDistinctString: Config]), tokenizerData: config, addedTokens: [:])
        }
        try report("Normalizer") { try NormalizerFactory.fromConfig(config: config.normalizer)! as AnyObject }
        try report("PreTokenizer") { try PreTokenizerFactory.fromConfig(config: config.preTokenizer)! as AnyObject }
        try report("Decoder") { try DecoderFactory.fromConfig(config: config.decoder, addedTokens: [])! as AnyObject }
        try report("PostProcessor") { try PostProcessorFactory.fromConfig(config: config.postProcessor)! as AnyObject }
        try report("PreTrainedTokenizer") {
            try PreTrainedTokenizer(
                tokenizerConfig: Config(["tokenizer_class": Config("PreTrainedTokenizerFast")]), tokenizerData: config)
        }
        try report("LocalModelConfiguration") { try LocalModelConfiguration(modelFolder: folder) as AnyObject }
        try report("from(configuration:)") {
            let configuration = try LocalModelConfiguration(modelFolder: folder)
            return try AutoTokenizer.from(
                tokenizerConfig: configuration.tokenizerConfig!, tokenizerData: configuration.tokenizerData)
                as AnyObject
        }
        try report("AutoTokenizer.load") {
            try AutoTokenizer.load(from: folder) as AnyObject
        }
    }
}
