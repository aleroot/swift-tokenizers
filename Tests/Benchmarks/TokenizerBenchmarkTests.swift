// End-to-end tokenizer benchmarks. Run with `RUN_BENCHMARKS=1 swift test -c release --filter Benchmarks`.
// The BPE cases mirror swift-transformers' `BPETokenizerBenchmarkTests` so results are directly
// comparable when run on the same machine.

import Foundation
import Testing

@testable import Tokenizers

/// Minimal fixture loader for the benchmark target (see `HubFixtures` in TokenizersTests).
enum BenchmarkFixtures {
    static let cacheRoot: URL = {
        if let override = ProcessInfo.processInfo.environment["SWIFT_TOKENIZERS_FIXTURES"] {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
        return base.appendingPathComponent("swift-tokenizers-tests", isDirectory: true)
    }()

    static func modelFolder(_ repo: String) async throws -> URL {
        let folder = cacheRoot.appendingPathComponent(repo, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for file in ["tokenizer.json", "tokenizer_config.json", "config.json"] {
            let destination = folder.appendingPathComponent(file)
            guard !FileManager.default.fileExists(atPath: destination.path) else { continue }
            let url = URL(string: "https://huggingface.co/\(repo)/resolve/main/\(file)")!
            let (data, response) = try await URLSession.shared.data(from: url)
            if (response as? HTTPURLResponse)?.statusCode == 200 {
                try data.write(to: destination)
            }
        }
        return folder
    }
}

let benchmarkParagraph = """
    Byte-pair encoding (BPE) is a tokenization algorithm originally proposed for data \
    compression by Philip Gage in 1994. It was later adapted for use in neural machine \
    translation by Sennrich, Haddow, and Birch in 2015, and is now the dominant \
    sub-word tokenization scheme for modern large language models including the GPT, \
    Llama, Qwen, and Mistral families. The algorithm operates by iteratively replacing \
    the most frequent adjacent pair of bytes in a corpus with a new symbol, building up \
    a vocabulary of merges that compactly represents both common words and rare strings.
    """

let benchmarkCode = """
    public final class GPT2BytePairEncoderConfiguration: Codable, Sendable {
        public let vocabularyIdentifierToTokenStringMap: [Int: String]
        public let bytePairMergeRanksByPairOfStrings: [BytePair: Int]
        public let unknownTokenIdentifierForOutOfVocabularyByteSequences: Int?
        public let beginningOfSequenceSpecialTokenIdentifier: Int?
        public let endOfSequenceSpecialTokenIdentifier: Int?
        public let shouldFuseConsecutiveUnknownTokenSequencesIntoASingleUnknownToken: Bool
    }
    """

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct TokenizerBenchmarkTests {
    /// Same model as swift-transformers' benchmark: Qwen2-style BPE with ~152K merges.
    static let modelId = "mlx-community/Qwen3-0.6B-Base-DQ5"

    let tokenizer: any Tokenizer
    let modelFolder: URL
    let shortText = "Explain the BPE tokenization algorithm in three short bullet points."
    let mediumText = String(repeating: benchmarkParagraph + "\n\n", count: 2)
    let longText = String(repeating: benchmarkParagraph + "\n\n", count: 20)
    let hugeText = String(repeating: benchmarkParagraph + "\n\n", count: 2000)

    init() async throws {
        modelFolder = try await BenchmarkFixtures.modelFolder(Self.modelId)
        tokenizer = try AutoTokenizer.load(from: modelFolder)
    }

    @Test("Tokenizer load time")
    func loadTime() throws {
        print("\n=== Tokenizer load (\(Self.modelId)) ===")
        let folder = modelFolder
        benchmarkMeasure(label: "AutoTokenizer.load", iterations: 5, warmup: 1) {
            _ = try! AutoTokenizer.load(from: folder)
        }
        let json = try Data(contentsOf: folder.appendingPathComponent("tokenizer.json"))
        print(String(format: "  tokenizer.json: %.1f MB", Double(json.count) / 1_048_576))
        benchmarkMeasure(label: "Config(jsonData:)", iterations: 5, warmup: 1) {
            _ = try! Config(jsonData: json)
        }
        benchmarkMeasure(label: "JSONSerialization", iterations: 5, warmup: 1) {
            _ = try! JSONSerialization.jsonObject(with: json)
        }
    }

    @Test("BPE encode throughput across input sizes")
    func encodeThroughput() {
        print("\n=== BPE encode throughput (\(Self.modelId)) ===")
        let cases: [(String, String, Int)] = [
            ("short (~100 B)", shortText, 200),
            ("medium (~3 KB)", mediumText, 50),
            ("long (~30 KB)", longText, 10),
            ("code (~600 B)", benchmarkCode, 100),
            ("huge (~1.2 MB)", hugeText, 3),
        ]
        for (label, text, iterations) in cases {
            let bytes = text.utf8.count
            let stats = benchmarkMeasure(label: label, iterations: iterations) {
                _ = tokenizer.encode(text: text, addSpecialTokens: false)
            }
            print(String(format: "    → %.2f MB/s", (Double(bytes) / 1_048_576.0) / (stats.mean / 1_000.0)))
        }
    }

    @Test("Decode throughput")
    func decodeThroughput() {
        print("\n=== Decode throughput (\(Self.modelId)) ===")
        let ids = tokenizer.encode(text: longText, addSpecialTokens: false)
        let bytes = longText.utf8.count
        let stats = benchmarkMeasure(label: "decode long (~30 KB)", iterations: 20) {
            _ = tokenizer.decode(tokens: ids)
        }
        print(
            String(
                format: "    → %.2f MB/s (%d tokens)", (Double(bytes) / 1_048_576.0) / (stats.mean / 1_000.0), ids.count
            ))

        // Streaming pattern used by generation loops: decode a growing prefix token by token.
        let prefix = Array(ids.prefix(200))
        benchmarkMeasure(label: "streaming 200 steps", iterations: 20) {
            for n in 1...prefix.count {
                _ = tokenizer.decode(tokens: Array(prefix[..<n]))
            }
        }
    }

    @Test("Vocabulary walk (guided generation pattern)")
    func vocabularyWalk() {
        print("\n=== convertIdToToken walk ===")
        benchmarkMeasure(label: "walk until nil", iterations: 5) {
            var id = 0
            while tokenizer.convertIdToToken(id) != nil { id += 1 }
        }
    }

    @Test("BPE merge inner loop on synthetic long words")
    func bpeMergeInnerLoop() throws {
        guard let pretrained = tokenizer as? PreTrainedTokenizer, let model = pretrained.model as? BPETokenizer else {
            Issue.record("Expected BPETokenizer model")
            return
        }
        let words = [
            "internationalization",
            "supercalifragilisticexpialidocious",
            "GPT2BytePairEncoderConfiguration",
            "vocabularyIdentifierToTokenStringMap",
            "shouldFuseConsecutiveUnknownTokenSequencesIntoASingleUnknownToken",
        ]
        print("\n=== BPE merge inner loop (per-word, 1000 iterations) ===")
        for word in words {
            let encoded = model.byteEncode(text: word).first ?? word
            let stats = benchmarkMeasure(label: "len=\(encoded.count)", iterations: 1000) {
                _ = model.bpe(token: encoded)
            }
            print(String(format: "    word=\"%@\" mean %.3f ms", word, stats.mean))
        }
    }
}

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct OtherModelBenchmarkTests {
    @Test(
        "Encode throughput for other tokenizer families",
        arguments: [
            "pcuenq/Llama-3.2-1B-Instruct-tokenizer",
            "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            "coreml-projects/Llama-2-7b-chat-coreml",
            "google-bert/bert-base-uncased",
            "t5-base",
        ])
    func familyThroughput(model: String) async throws {
        let folder = try await BenchmarkFixtures.modelFolder(model)
        let tokenizer = try AutoTokenizer.load(from: folder)
        let text = String(repeating: benchmarkParagraph + "\n\n", count: 20)
        let bytes = text.utf8.count
        print("\n=== \(model) ===")
        let stats = benchmarkMeasure(label: "long (~30 KB)", iterations: 10) {
            _ = tokenizer.encode(text: text, addSpecialTokens: false)
        }
        print(String(format: "    → %.2f MB/s", (Double(bytes) / 1_048_576.0) / (stats.mean / 1_000.0)))
    }
}
