import Foundation
import Testing

@testable import Tokenizers

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct MetaspaceBenchmarkTests {
    @Test(arguments: ["t5-base", "mlx-community/Mistral-7B-Instruct-v0.3-4bit"])
    func canonicalThroughput(model: String) async throws {
        let folder = try await BenchmarkFixtures.modelFolder(model)
        let configuration = try LocalModelConfiguration(modelFolder: folder)
        let data = configuration.tokenizerData
        let tokenizerConfig = configuration.tokenizerConfig ?? Config()
        var legacy = try #require(data.dictionary())
        // Measure the text pipelines with the same model, added tokens and decoder.
        // Special-token insertion is independent of the Metaspace representation.
        legacy["post_processor"] = nil
        let canonical = try canonicalMetaspace(Config(legacy))
        let tokenizers = [
            ("legacy", try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: Config(legacy))),
            ("canonical", try PreTrainedTokenizer(tokenizerConfig: tokenizerConfig, tokenizerData: canonical)),
        ]
        let cases = [
            ("query", "Explain the BPE tokenization algorithm in three short bullet points.", 20_000),
            ("passage", String(repeating: benchmarkParagraph + "\n", count: 20), 300),
            ("document", String(repeating: benchmarkParagraph + "\n", count: 2000), 3),
            ("multilingual", String(repeating: "日本語 ภาษาไทย café e\u{301} 👩🏽‍💻 العربية\n", count: 20), 600),
        ]
        var checksum = 0
        print("\n=== Metaspace: \(model) ===")
        for (name, text, iterations) in cases {
            print("  \(name): \(text.utf8.count) bytes × \(iterations) encodes per sample")
            let expected = tokenizers[0].1.encode(text: text, addSpecialTokens: false)
            for (_, tokenizer) in tokenizers {
                try #require(tokenizer.encode(text: text, addSpecialTokens: false) == expected)
                let aligned = try tokenizer.encode(text: text, addSpecialTokens: false, withOffsets: true)
                try #require(aligned.ids == expected)
            }
            // Reverse the order on the second pass to reduce CPU clock-state bias.
            for pass in 0..<2 {
                for (variant, tokenizer) in pass == 0 ? tokenizers : Array(tokenizers.reversed()) {
                    benchmarkMeasure(label: "\(name) \(variant) #\(pass)", iterations: 7) {
                        for _ in 0..<iterations {
                            checksum &+= tokenizer.encode(text: text, addSpecialTokens: false).count
                        }
                    }
                    benchmarkMeasure(label: "offsets \(variant) #\(pass)", iterations: 7) {
                        for _ in 0..<max(1, iterations / 20) {
                            checksum &+= try! tokenizer.encode(
                                text: text, addSpecialTokens: false, withOffsets: true
                            ).ids.count
                        }
                    }
                }
            }
        }
        print("  Metaspace checksum: \(checksum)")
    }

    /// Lower only the two published Metaspace shapes under test. The independent
    /// whole-model goldens live in CanonicalModelParityTests.
    private func canonicalMetaspace(_ data: Config) throws -> Config {
        var root = try #require(data.dictionary())
        let preTokenizer: Config = data.preTokenizer
        let dropWhitespace = preTokenizer.type.string() == "Sequence"
        let metaspace: Config
        if dropWhitespace {
            let children = try #require(preTokenizer.pretokenizers.array())
            try #require(children.count == 2 && children[0].type.string() == "WhitespaceSplit")
            metaspace = children[1]
        } else {
            metaspace = preTokenizer
        }
        try #require(metaspace.type.string() == "Metaspace")
        let replacement = try #require(metaspace.replacement.string())
        let prepend =
            metaspace.prependScheme.string() ?? (metaspace.addPrefixSpace.boolean(or: true) ? "always" : "never")
        let normalizer: Config = [
            "type": "MetaspaceNormalizer", "replacement": Config(replacement),
            "prepend": Config(prepend), "drop_whitespace": Config(dropWhitespace),
        ]
        root["version"] = "2.0"
        if data.model.type.string() == "BPE" {
            var model = try #require(data.model.dictionary())
            model["byte_level"] = false
            root["model"] = Config(model)
        }
        root["normalizer"] =
            data.normalizer.isNull()
            ? normalizer : ["type": "Sequence", "normalizers": [data.normalizer, normalizer]]
        root["pre_tokenizer"] =
            metaspace.split.boolean(or: true)
            ? [
                "type": "Split", "pattern": ["String": Config(replacement)], "behavior": "MergedWithNext",
                "invert": false,
            ]
            : nil
        return Config(root)
    }
}
