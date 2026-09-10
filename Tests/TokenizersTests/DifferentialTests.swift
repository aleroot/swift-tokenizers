// Differential testing over ~300 adversarial inputs (`scripts/differential_corpus.py`) for 24
// tokenizers. `Resources/differential/hf__*.json` is ground truth from Hugging Face `transformers`
// (`scripts/hf_golden.py`) and must match exactly; `upstream__*.json` is swift-transformers'
// output, and every deviation from it must agree with Hugging Face.

import Foundation
import Testing

@testable import Tokenizers

@Suite("Differential vs Hugging Face & swift-transformers")
struct DifferentialTests {
    struct HFRecord: Decodable {
        let text: String
        let ids: [Int]
        let idsNoSpecial: [Int]
        let decoded: String
        let decodedSkipSpecial: String
    }

    struct SurfaceRecord: Decodable { let text: String; let tokens: [String] }
    struct SurfaceOracle: Decodable { let models: [String: [SurfaceRecord]] }

    struct UpstreamRecord: Decodable {
        let text: String
        let tokens: [String]
        let ids: [Int]
        let idsNoSpecial: [Int]
        let decoded: String
        let decodedSkipSpecial: String
    }

    static let models = [
        "coreml-projects/Llama-2-7b-chat-coreml",
        "distilbert/distilbert-base-multilingual-cased",
        "distilgpt2",
        "openai/whisper-large-v2",
        "openai/whisper-tiny.en",
        "pcuenq/Llama-3.2-1B-Instruct-tokenizer",
        "t5-base",
        "tiiuae/falcon-7b",
        "pcuenq/gemma-tokenizer",
        "microsoft/phi-4",
        "mlx-community/Phi-3-mini-4k-instruct-4bit-no-q-embed",
        "google-t5/t5-small",
        "huggyllama/llama-7b",
        "intfloat/multilingual-e5-small",
        "FacebookAI/xlm-roberta-base",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "google-bert/bert-base-uncased",
        "BAAI/bge-small-en-v1.5",
        "FacebookAI/roberta-base",
        "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
        "Qwen/Qwen3-0.6B",
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "microsoft/Phi-3-mini-128k-instruct",
    ]

    static func resourceName(_ model: String) -> String {
        model.replacingOccurrences(of: "/", with: "__")
    }

    static func loadHF(_ model: String) throws -> [HFRecord] {
        let url = try #require(Bundle.module.url(forResource: "hf__" + resourceName(model), withExtension: "json"))
        return try JSONDecoder().decode([HFRecord].self, from: Data(contentsOf: url))
    }

    static func loadUpstream(_ model: String) throws -> [UpstreamRecord] {
        guard let url = Bundle.module.url(forResource: "upstream__" + resourceName(model), withExtension: "json") else {
            throw HubFixtures.FixtureError.unsupportedTokenizer
        }
        return try JSONDecoder().decode([UpstreamRecord].self, from: Data(contentsOf: url))
    }

    // MARK: - Hugging Face ground truth

    @Test(arguments: models)
    func matchesHuggingFace(model: String) async throws {
        let records = try Self.loadHF(model)
        // These Python 4.57 goldens include class reconstruction (notably Llama's BOS
        // override). Exercise the equivalent configuration factory here; folder-loading
        // policy is covered separately by TokenizerRegressionTests and TokenizerTests.
        let configuration = try await HubFixtures.configuration(for: model)
        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: #require(configuration.tokenizerConfig), tokenizerData: configuration.tokenizerData)
        #expect(records.count > 250)

        var differences = ["ids": 0, "idsNoSpecial": 0, "decoded": 0, "decodedSkipSpecial": 0]
        var mismatches: [String] = []
        for record in records {
            let ids = tokenizer.encode(text: record.text)
            if ids != record.ids {
                mismatches.append("encode(\(record.text.debugDescription)): got \(ids) expected \(record.ids)")
                differences["ids", default: 0] += 1
            }
            let noSpecial = tokenizer.encode(text: record.text, addSpecialTokens: false)
            if noSpecial != record.idsNoSpecial {
                differences["idsNoSpecial", default: 0] += 1
                mismatches.append(
                    "encode(noSpecial)(\(record.text.debugDescription)): got \(noSpecial) expected \(record.idsNoSpecial)"
                )
            }
            let decoded = tokenizer.decode(tokens: ids)
            if !decoded.utf8.elementsEqual(record.decoded.utf8) {
                differences["decoded", default: 0] += 1
                mismatches.append(
                    "decode(\(record.text.debugDescription)): got \(decoded.debugDescription) expected \(record.decoded.debugDescription)"
                )
            }
            let decodedSkip = tokenizer.decode(tokens: ids, skipSpecialTokens: true)
            if !decodedSkip.utf8.elementsEqual(record.decodedSkipSpecial.utf8) {
                differences["decodedSkipSpecial", default: 0] += 1
                mismatches.append(
                    "decode(skip)(\(record.text.debugDescription)): got \(decodedSkip.debugDescription) expected \(record.decodedSkipSpecial.debugDescription)"
                )
            }
        }
        // One atomic file per parameterized test: parallel test processes never share a
        // mutable report. CI aggregates only after the optimized test run succeeds.
        if let directory = ProcessInfo.processInfo.environment["TOKENIZERS_PARITY_REPORT_DIR"] {
            let folder = URL(fileURLWithPath: directory, isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let report: [String: Any] = ["model": model, "cases": records.count, "differences": differences]
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: folder.appendingPathComponent(Self.resourceName(model) + ".json"), options: .atomic)
        }
        if !mismatches.isEmpty {
            let summary = Array(mismatches.prefix(10)).joined(separator: "\n")
            Issue.record(Comment(rawValue: "\(model): \(mismatches.count) mismatches vs Hugging Face\n\(summary)"))
        }
    }

    // MARK: - swift-transformers compatibility

    @Test(arguments: models)
    func matchesUpstreamOrHuggingFace(model: String) async throws {
        // The historical Swift goldens also used the configuration factory.
        let configuration = try await HubFixtures.configuration(for: model)
        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: #require(configuration.tokenizerConfig), tokenizerData: configuration.tokenizerData)
        let upstream = try Self.loadUpstream(model)
        let hf = try Self.loadHF(model)
        let surfaceURL = try #require(Bundle.module.url(forResource: "hf-token-surfaces", withExtension: "json"))
        let surfaces = try JSONDecoder().decode(SurfaceOracle.self, from: Data(contentsOf: surfaceURL))
        let surfaceRecords = try #require(surfaces.models[model])
        let surfacesByText = Dictionary(
            uniqueKeysWithValues: surfaceRecords.map { (BinaryDistinctString($0.text), $0.tokens) })
        var hfByText: [BinaryDistinctString: HFRecord] = [:]
        for record in hf { hfByText[BinaryDistinctString(record.text)] = record }
        #expect(upstream.count > 250)

        var regressions: [String] = []
        var deviations = 0

        func check<T: Equatable>(_ label: String, text: String, ours: T, upstream: T, hf: T?) {
            if ours == upstream { return }
            if let hf, ours == hf {
                deviations += 1  // we differ from swift-transformers but agree with Hugging Face
                return
            }
            regressions.append(
                "\(label)(\(text.debugDescription)): got \(ours) upstream \(upstream) hf \(hf.map { "\($0)" } ?? "n/a")"
            )
        }

        for record in upstream {
            let hfRecord = hfByText[BinaryDistinctString(record.text)]
            let ids = tokenizer.encode(text: record.text)
            check("encode", text: record.text, ours: ids, upstream: record.ids, hf: hfRecord?.ids)
            check(
                "encode(noSpecial)", text: record.text,
                ours: tokenizer.encode(text: record.text, addSpecialTokens: false),
                upstream: record.idsNoSpecial, hf: hfRecord?.idsNoSpecial
            )
            // Python/Rust serialized-pipeline tokens adjudicate surface changes. In particular,
            // upstream Swift dropped all but the first character of unknown Unigram spans.
            if ids == record.ids {
                let tokens = tokenizer.tokenize(text: record.text).map { Array($0.utf8) }
                let reference = try #require(surfacesByText[BinaryDistinctString(record.text)])
                check(
                    "tokenize", text: record.text, ours: tokens,
                    upstream: record.tokens.map { Array($0.utf8) }, hf: reference.map { Array($0.utf8) })
            }
            check(
                "decode", text: record.text, ours: Array(tokenizer.decode(tokens: ids).utf8),
                upstream: Array(record.decoded.utf8),
                hf: hfRecord.map { Array($0.decoded.utf8) })
            check(
                "decode(skip)", text: record.text,
                ours: Array(tokenizer.decode(tokens: ids, skipSpecialTokens: true).utf8),
                upstream: Array(record.decodedSkipSpecial.utf8), hf: hfRecord.map { Array($0.decodedSkipSpecial.utf8) }
            )
        }

        if !regressions.isEmpty {
            let summary = Array(regressions.prefix(10)).joined(separator: "\n")
            Issue.record(
                Comment(
                    rawValue:
                        "\(model): \(regressions.count) unexplained differences vs swift-transformers (\(deviations) HF-confirmed fixes)\n\(summary)"
                ))
        }
    }
}
