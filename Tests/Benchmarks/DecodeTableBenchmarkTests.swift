import Foundation
import Testing

@testable import Tokenizers

/// Opt-in, portable comparisons against the decoder's string pipeline. No mutable switches
/// in library code; run with RUN_BENCHMARKS=1 swift test -c release --filter DecodeTableBenchmarkTests.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct DecodeTableBenchmarkTests {
    @Test(
        "Batch and streaming decode across tokenizer families",
        arguments: [
            "Qwen/Qwen3-0.6B", "pcuenq/Llama-3.2-1B-Instruct-tokenizer", "pcuenq/gemma-tokenizer",
            "google-bert/bert-base-uncased", "FacebookAI/xlm-roberta-base",
        ])
    func decode(model: String) async throws {
        let folder = try await BenchmarkFixtures.modelFolder(model)
        let tokenizer = try #require(try AutoTokenizer.load(from: folder) as? PreTrainedTokenizer)
        let text = String(repeating: benchmarkParagraph + "\n\n", count: 400)
        let ids = tokenizer.encode(text: text, addSpecialTokens: false)
        func reference(_ ids: [Int]) -> String {
            tokenizer.cleanUp(text: tokenizer.decodeTokens(ids.compactMap { tokenizer.convertIdToToken($0) }).joined())
        }
        let expected = reference(ids)
        print("\n=== Decode \(model), \(ids.count) ids ===")
        var checksum = 0
        let shortTokenizer = try AutoTokenizer.load(from: folder)
        let query = Array(ids.prefix(12))
        benchmarkMeasure(label: "cold query", iterations: 1, warmup: 0) {
            checksum &+= shortTokenizer.decode(tokens: query).utf8.count
        }
        let streamTokenizer = try AutoTokenizer.load(from: folder)
        benchmarkMeasure(label: "cold stream creation", iterations: 1, warmup: 0) {
            let stream = streamTokenizer.makeStreamDecoder()
            checksum &+= stream.isIncremental ? 1 : 0
        }
        benchmarkMeasure(label: "cold batch", iterations: 1, warmup: 0) {
            checksum &+= tokenizer.decode(tokens: ids).utf8.count
        }
        benchmarkMeasure(label: "prepare tables", iterations: 1, warmup: 0) {
            tokenizer.prepareForDecoding()
        }
        for (label, tokens) in [("prose", ids), ("query", Array(ids.prefix(12)))] {
            benchmarkMeasure(label: "\(label) string", iterations: 20) { checksum &+= reference(tokens).utf8.count }
            benchmarkMeasure(label: "\(label) table", iterations: 20) {
                checksum &+= tokenizer.decode(tokens: tokens).utf8.count
            }
            #expect(tokenizer.decode(tokens: tokens).utf8.elementsEqual(reference(tokens).utf8))
        }
        var stream = tokenizer.makeStreamDecoder()
        var streamed = ""
        for id in ids { streamed += try stream.append(token: id) }
        streamed += stream.finish()
        #expect(streamed.utf8.elementsEqual(expected.utf8))
        benchmarkMeasure(label: "stream", iterations: 10) {
            stream.reset()
            for id in ids { checksum &+= try! stream.append(token: id).utf8.count }
            checksum &+= stream.finish().utf8.count
        }
        print("  checksum \(checksum)")
    }
    @Test("Byte-level cleanup preserves ordinary-text throughput")
    func cleanup() throws {
        let words = (0..<256).map { [UInt8($0)].withUnsafeBufferPointer { ByteLevelAlphabet.encode($0) } }
        let vocabulary = try Vocabulary(entries: words.enumerated().map { ($0.element, $0.offset) })
        let table = ByteLevelDecodeTable(vocabulary: vocabulary)
        var checksum = 0
        for (label, text) in [
            ("plain", String(repeating: "ordinary text without cleanup patterns ", count: 4_000)),
            ("artifacts", String(repeating: "Hello .  ' s  n ' t world ! ", count: 4_000)),
        ] {
            let ids = text.utf8.map(Int.init)
            let expected = TokenizationCleanup.cleanUp(table.decode(ids, skipping: []))
            #expect(table.decode(ids, skipping: [], cleanup: true).utf8.elementsEqual(expected.utf8))
            benchmarkMeasure(label: "\(label) string cleanup", iterations: 20) {
                checksum &+= TokenizationCleanup.cleanUp(table.decode(ids, skipping: [])).utf8.count
            }
            benchmarkMeasure(label: "\(label) in-place cleanup", iterations: 20) {
                checksum &+= table.decode(ids, skipping: [], cleanup: true).utf8.count
            }
        }
        print("  checksum \(checksum)")
    }
}
