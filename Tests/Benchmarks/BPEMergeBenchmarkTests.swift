// Uncached merge work isolates scratch reuse from the pre-token cache. Run in release mode
// with RUN_BENCHMARKS=1 and --filter BPEMergeBenchmarkTests on both revisions being compared.
import Foundation
import Testing

@testable import Tokenizers

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct BPEMergeBenchmarkTests {
    @Test("BPE merge scratch across the linear/heap boundary")
    func merges() async throws {
        let folder = try await BenchmarkFixtures.modelFolder(TokenizerBenchmarkTests.modelId)
        let tokenizer = try AutoTokenizer.load(from: folder) as! PreTrainedTokenizer
        let model = tokenizer.model as! BPETokenizer
        for size in [16, 96, 97, 128, 512, 4096] {
            for pattern in ["a", "HTTPServer_abc123.xyz()日本語"] {
                let bytes = Array(String(repeating: pattern, count: size).utf8.prefix(size))
                var input: [BPETokenizer.Symbol] = []
                bytes.withUnsafeBufferPointer { model.byteLevelSymbols($0, into: &input) }
                var expected = input
                var referenceScratch = BPETokenizer.MergeScratch()
                model.mergeLinear(&expected, scratch: &referenceScratch)
                var scratch = BPETokenizer.MergeScratch()
                var symbols: [BPETokenizer.Symbol] = []
                let iterations = max(200, 200_000 / size)
                var checksum = 0
                benchmarkMeasure(label: "\(size) bytes, \(pattern == "a" ? "ties" : "code")", iterations: 10) {
                    for _ in 0..<iterations {
                        symbols.removeAll(keepingCapacity: true)
                        symbols.append(contentsOf: input)
                        model.merge(&symbols, scratch: &scratch)
                        checksum &+= symbols.count
                        checksum &+= Int(symbols.first?.id ?? 0)
                    }
                }
                #expect(symbols.map(\.id) == expected.map(\.id))
                #expect(symbols.map(\.end) == expected.map(\.end))
                print("    \(iterations) merges/sample; checksum \(checksum)")
            }
        }
    }
}
