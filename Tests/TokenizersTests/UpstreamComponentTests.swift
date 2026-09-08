// Behavioral ports from Hugging Face tokenizers v0.23.2 and spm_precompiled v0.1.3
// (Apache-2.0; see LICENSE). The fixture records source paths and the upstream revision.
import Foundation
import Testing

@testable import Tokenizers

@Suite("Upstream component parity")
struct UpstreamComponentTests {
    struct Fixture: Decodable, Sendable {
        let configurations: [String: Config]
        let cases: [Case]
    }
    struct Case: Decodable, Sendable {
        let name: String
        let operation: String
        let configuration: String
        let source: String
        let text: String?
        let tokens: [String]?
        let expectedText: String?
        let expectedTokens: [String]?
        let expectedIds: [Int]?
    }
    static func fixture() throws -> Fixture {
        let url = try #require(Bundle.module.url(forResource: "upstream-components", withExtension: "json"))
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
    }
    @Test(
        "Matches pinned Python/Rust reference",
        arguments: ["normalize", "decode", "preTokenize", "encode", "decodeAdded"])
    func parity(operation: String) throws {
        let fixture = try Self.fixture()
        for test in fixture.cases where test.operation == operation {
            do {
                let config = try #require(fixture.configurations[test.configuration])
                switch operation {
                case "normalize":
                    let normalizer = try #require(try NormalizerFactory.fromConfig(config: config))
                    let actual = normalizer.normalize(text: try #require(test.text))
                    #expect(Array(actual.utf8) == Array(try #require(test.expectedText).utf8), "\(test.name)")
                case "decode":
                    let decoder = try #require(try DecoderFactory.fromConfig(config: config))
                    let actual = decoder.decode(tokens: try #require(test.tokens)).joined()
                    #expect(Array(actual.utf8) == Array(try #require(test.expectedText).utf8), "\(test.name)")
                case "preTokenize":
                    let pre = try #require(try PreTokenizerFactory.fromConfig(config: config))
                    let actual = pre.preTokenize(text: try #require(test.text), options: [.firstSection])
                    #expect(
                        actual.map { Array($0.utf8) } == (try #require(test.expectedTokens)).map { Array($0.utf8) },
                        "\(test.name)")
                case "decodeAdded":
                    let tokenizer = try PreTrainedTokenizer(
                        tokenizerConfig: ["tokenizer_class": "PreTrainedTokenizerFast"], tokenizerData: config)
                    let ids = try #require(test.expectedIds)
                    #expect(
                        Array(tokenizer.decode(tokens: ids).utf8) == Array(try #require(test.expectedText).utf8),
                        "\(test.name)")
                    #expect(
                        ids.compactMap(tokenizer.convertIdToToken).map { Array($0.utf8) }
                            == (try #require(test.expectedTokens)).map { Array($0.utf8) }, "\(test.name)")
                default:
                    let tokenizer = try PreTrainedTokenizer(
                        tokenizerConfig: ["tokenizer_class": "PreTrainedTokenizerFast"], tokenizerData: config)
                    let text = try #require(test.text)
                    let expected = try #require(test.expectedIds)
                    // A second call exercises cached encoding; string tokenization is independent.
                    for _ in 0..<2 {
                        #expect(tokenizer.encode(text: text, addSpecialTokens: false) == expected, "\(test.name)")
                    }
                    if config.addedTokens.array(or: []).isEmpty {
                        #expect(
                            tokenizer.tokenize(text: text).map { Array($0.utf8) }
                                == (try #require(test.expectedTokens)).map { Array($0.utf8) }, "\(test.name)")
                    } else {
                        // Swift's string API returns vocabulary spellings, not HF's matched surfaces.
                        #expect(
                            tokenizer.tokenize(text: text).compactMap(tokenizer.convertTokenToId) == expected,
                            "\(test.name)")
                    }
                }
            } catch {
                Issue.record("\(test.name) [\(test.source)]: \(error)")
            }
        }
    }
}

@Suite("Reference configuration boundaries")
struct ReferenceConfigurationBoundaryTests {
    @Test("Malformed SentencePiece maps throw without indexing untrusted offsets")
    func invalidMaps() throws {
        for bytes: [UInt8] in [
            [], [0, 0, 0, 0], [255, 255, 255, 255, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 0], [4, 0, 0, 0, 255, 255, 255, 255, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 255, 0], [4, 0, 0, 0, 0, 0, 0, 0, 65],
        ] {
            let config: Config = [
                "type": "Precompiled", "precompiled_charsmap": Config(Data(bytes).base64EncodedString()),
            ]
            #expect(throws: TokenizerError.self) { try NormalizerFactory.fromConfig(config: config) }
        }
    }
    @Test("Serialized model type wins over absent or conflicting Python class names")
    func modelTypeAuthority() throws {
        let data: Config = [
            "model": [
                "type": "WordPiece", "vocab": ["[UNK]": 0, "A": 1],
                "unk_token": "[UNK]", "max_input_chars_per_word": 100, "continuing_subword_prefix": "##",
            ]
        ]
        for config: Config in [
            [:], ["tokenizer_class": "GPT2Tokenizer"], ["tokenizer_class": "PreTrainedTokenizerFast"],
        ] {
            let tokenizer = try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data)
            #expect(tokenizer.encode(text: "A", addSpecialTokens: false) == [1])
        }
    }
}

@Suite("Tokenizer class decode policy")
struct TokenizerClassDecodePolicyTests {
    @Test("Llama wrapper preserves declared BOS spelling while the raw pipeline normalizes it")
    func llamaAddedTokenDecode() throws {
        let data: Config = [
            "model": ["type": "BPE", "vocab": ["<s>": 0, "▁": 1], "merges": []],
            "normalizer": ["type": "Prepend", "prepend": "▁"],
            "added_tokens": [["id": 0, "content": "<s>", "normalized": true, "special": true]],
            "decoder": ["type": "Replace", "pattern": ["String": "▁"], "content": " "],
        ]
        let config: Config = ["tokenizer_class": "LlamaTokenizer", "bos_token": "<s>"]
        let wrapper = try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data)
        #expect(wrapper.decode(tokens: [0, 0]) == "<s><s>")
        let raw = try PreTrainedTokenizer(
            tokenizerConfig: ["tokenizer_class": "PreTrainedTokenizerFast"], tokenizerData: data)
        #expect(raw.decode(tokens: [0, 0]) == " <s> <s>")
    }
}
