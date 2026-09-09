import Foundation
import Testing

@testable import Tokenizers

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct LampoWorkloadBenchmarks {
    private static let qwen = "mlx-community/Qwen3-0.6B-Base-DQ5"

    @Test("First streamed decode and token inspection")
    func decoding() async throws {
        let folder = try await BenchmarkFixtures.modelFolder(Self.qwen)
        let tokenizer = try AutoTokenizer.load(from: folder)
        let ids = tokenizer.encode(text: benchmarkParagraph, addSpecialTokens: false)
        var times: [Double] = []
        for _ in 0..<9 {
            let cold = try AutoTokenizer.load(from: folder)
            let start = DispatchTime.now().uptimeNanoseconds
            _ = cold.decode(tokens: Array(ids.prefix(16)))
            times.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000)
        }
        print("  First decode: \(benchmarkStats(times).formatted)")
        _ = tokenizer.decode(tokens: ids)
        benchmarkMeasure(label: "Inspect token IDs", iterations: 15) {
            for id in ids { _ = tokenizer.decode(tokens: [id]) }
        }
        benchmarkMeasure(label: "Stream one paragraph", iterations: 15) {
            for count in 1...ids.count { _ = tokenizer.decode(tokens: Array(ids.prefix(count))) }
        }
    }

    @Test("Jina fallback loader construction")
    func fallbackLoader() async throws {
        let folder = try await BenchmarkFixtures.modelFolder(Self.qwen)
        let data = try Data(contentsOf: folder.appendingPathComponent("tokenizer.json"))
        let config: Config = [
            "tokenizer_class": "TokenizersBackend", "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>", "model_max_length": 131_072,
        ]
        benchmarkMeasure(label: "Foundation fallback", iterations: 9) {
            let object = try! JSONSerialization.jsonObject(with: data) as! [NSString: Any]
            _ = try! AutoTokenizer.from(tokenizerConfig: config, tokenizerData: Config(object))
        }
        benchmarkMeasure(label: "Packed fallback", iterations: 9) {
            _ = try! AutoTokenizer.from(tokenizerConfig: config, tokenizerData: Config(tokenizerJSON: data))
        }
    }

    @Test("RAG query and passage batches, followed by a large document")
    func encoding() async throws {
        let document = String(repeating: benchmarkParagraph + "\n", count: 4096)
        let passages = (0..<64).map { "passage: \($0). " + benchmarkParagraph }
        for model in [Self.qwen, "intfloat/multilingual-e5-small", "google-bert/bert-base-uncased"] {
            let folder = try await BenchmarkFixtures.modelFolder(model)
            let tokenizer = try AutoTokenizer.load(from: folder)
            print("\n=== Lampo encode: \(model) ===")
            benchmarkMeasure(label: "256 short queries", iterations: 15) {
                for _ in 0..<256 { _ = tokenizer.encode(text: "query: How does local semantic search work?") }
            }
            benchmarkMeasure(label: "64 passages", iterations: 15) {
                for text in passages { _ = tokenizer.encode(text: text) }
            }
            let before = Self.liveHeap()
            autoreleasepool { _ = tokenizer.encode(text: document) }
            let after = Self.liveHeap()
            print("  Retained after \(document.utf8.count) bytes: \(after - before) bytes")
            benchmarkMeasure(label: "Repeat large document", iterations: 5, warmup: 1) {
                _ = tokenizer.encode(text: document)
            }
            withExtendedLifetime(tokenizer) {}
        }
    }

    private static func liveHeap() -> Int {
        var statistics = malloc_statistics_t()
        malloc_zone_statistics(nil, &statistics)
        return Int(statistics.size_in_use)
    }
}
