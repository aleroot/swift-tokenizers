// Per-stage breakdown of the byte-level BPE encode path. Run with `RUN_BENCHMARKS=1`.

import Foundation
import Testing

@testable import Tokenizers

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct StageBenchmarkTests {
    @Test("Encode stage breakdown")
    func stages() async throws {
        let folder = try await BenchmarkFixtures.modelFolder(TokenizerBenchmarkTests.modelId)
        let tokenizer = try AutoTokenizer.load(from: folder) as! PreTrainedTokenizer
        let model = tokenizer.model as! BPETokenizer
        let text = String(repeating: benchmarkParagraph + "\n\n", count: 200)
        let bytes = Double(text.utf8.count)
        print("\n=== Stage breakdown (\(Int(bytes)) bytes) ===")

        func report(_ label: String, _ stats: BenchmarkStats) {
            print(
                String(
                    format: "    → %.2f MB/s  (%.1f ns/byte)", (bytes / 1_048_576) / (stats.mean / 1000),
                    stats.mean * 1_000_000 / bytes))
        }

        report(
            "scanner",
            benchmarkMeasure(label: "scanner (qwen2 pattern)", iterations: 20) {
                var pieces: [Substring] = []
                pieces.reserveCapacity(text.utf8.count / 4)
                KnownSplitPattern.qwen2.split(Substring(text), into: &pieces)
            })

        var pieces: [Substring] = []
        KnownSplitPattern.qwen2.split(Substring(text), into: &pieces)
        print("    pieces: \(pieces.count)")

        report(
            "bpe (cache)",
            benchmarkMeasure(label: "bpe over pieces (cached)", iterations: 20) {
                let encoder = model.makeEncoder()
                encoder.begin()
                defer { encoder.finish() }
                var ids: [Int] = []
                ids.reserveCapacity(text.utf8.count / 3)
                for piece in pieces { encoder.encode(piece: piece, byteLevel: true, into: &ids) }
            })

        report(
            "bpe (no cache)",
            benchmarkMeasure(label: "bpe over pieces (no cache)", iterations: 20) {
                // Hold the cache lock so the encoder cannot acquire it.
                model.cache.lock.lock()
                defer { model.cache.lock.unlock() }
                let encoder = model.makeEncoder()
                encoder.begin()
                defer { encoder.finish() }
                var ids: [Int] = []
                ids.reserveCapacity(text.utf8.count / 3)
                for piece in pieces { encoder.encode(piece: piece, byteLevel: true, into: &ids) }
            })

        if let splitter = Mirror(reflecting: tokenizer).descendant("splitter") as? AddedTokenSplitter {
            print("    added tokens: \(splitter.tokens.count)")
            report(
                "splitter",
                benchmarkMeasure(label: "added-token splitter (bytes)", iterations: 20) {
                    var copy = text
                    copy.withUTF8 { bytes in
                        var sections: [AddedTokenSplitter.ByteSection] = []
                        splitter.split(bytes: bytes, into: &sections)
                    }
                })
        }

        report(
            "full encode",
            benchmarkMeasure(label: "encode(text:)", iterations: 20) {
                _ = tokenizer.encode(text: text, addSpecialTokens: false)
            })
    }
}
