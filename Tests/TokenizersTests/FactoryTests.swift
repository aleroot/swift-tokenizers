// Created by Pedro Cuenca on 4/8/23.

import Foundation
import Testing

@testable import Tokenizers

@Suite("Tokenizer factory")
struct FactoryTests {
    @Test
    func fromPretrained() async throws {
        let tokenizer = try await HubFixtures.tokenizer(for: "coreml-projects/Llama-2-7b-chat-coreml")
        let inputIds = tokenizer("Today she took a train to the West")
        #expect(inputIds == [1, 20628, 1183, 3614, 263, 7945, 304, 278, 3122])
    }

    @Test
    func whisper() async throws {
        let tokenizer = try await HubFixtures.tokenizer(for: "openai/whisper-large-v2")
        let inputIds = tokenizer("Today she took a train to the West")
        #expect(inputIds == [50258, 50363, 27676, 750, 1890, 257, 3847, 281, 264, 4055, 50257])
    }

    @Test
    func fromModelFolder() async throws {
        let localModelFolder = try await HubFixtures.modelFolder(
            for: "coreml-projects/Llama-2-7b-chat-coreml",
            files: ["config.json", "tokenizer_config.json", "tokenizer.json"]
        )
        let tokenizer = try await AutoTokenizer.from(modelFolder: localModelFolder)
        let inputIds = tokenizer("Today she took a train to the West")
        #expect(inputIds == [1, 20628, 1183, 3614, 263, 7945, 304, 278, 3122])
    }

    @Test
    func whisperFromModelFolder() async throws {
        let localModelFolder = try await HubFixtures.modelFolder(
            for: "openai/whisper-large-v2",
            files: ["config.json", "tokenizer_config.json", "tokenizer.json"]
        )
        let tokenizer = try await AutoTokenizer.from(modelFolder: localModelFolder)
        let inputIds = tokenizer("Today she took a train to the West")
        #expect(inputIds == [50258, 50363, 27676, 750, 1890, 257, 3847, 281, 264, 4055, 50257])
    }

    @Test
    func synchronousLoad() async throws {
        let localModelFolder = try await HubFixtures.modelFolder(for: "openai/whisper-large-v2")
        let tokenizer = try AutoTokenizer.load(from: localModelFolder)
        #expect(
            tokenizer("Today she took a train to the West") == [
                50258, 50363, 27676, 750, 1890, 257, 3847, 281, 264, 4055, 50257,
            ])
    }

    // MARK: - Malformed configurations are reported, never trapped on

    /// Builds a tokenizer from a minimal BPE `tokenizer.json` with `extra` spliced in.
    private static func tokenizer(extra: String) throws -> any Tokenizer {
        let data = """
            {"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}\(extra.isEmpty ? "" : ", " + extra)}
            """
        return try AutoTokenizer.from(
            tokenizerConfig: try Config(jsonString: #"{"tokenizer_class": "GPT2Tokenizer"}"#),
            tokenizerData: try Config(jsonString: data)
        )
    }

    @Test("Well-formed minimal configuration loads")
    func minimalConfigurationLoads() throws {
        #expect(try Self.tokenizer(extra: "").encode(text: "ab") == [2])
    }

    @Test("Unregistered tokenizer_class resolves through tokenizer.json model.type")
    func unregisteredClassUsesModelType() throws {
        let data = try Config(
            jsonString: #"{"model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": ["a b"]}}"#)
        let config = try Config(jsonString: #"{"tokenizer_class": "BrandNewHubTokenizer"}"#)
        let tokenizer = try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data, strict: true)
        #expect(tokenizer.encode(text: "ab") == [2])

        // Without a recognisable model type, strict mode still refuses to guess.
        let opaque = try Config(jsonString: #"{"model": {"type": "Mystery", "vocab": {}}}"#)
        #expect(throws: TokenizerError.self) {
            try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: opaque, strict: true)
        }
    }

    @Test("Missing folder is a thrown error")
    func missingFolder() {
        #expect(throws: TokenizerError.self) {
            try AutoTokenizer.load(from: URL(fileURLWithPath: "/nonexistent/model/folder"))
        }
    }

    @Test("Unsupported pipeline components throw")
    func unsupportedComponent() {
        #expect(throws: TokenizerError.self) {
            try Self.tokenizer(extra: #""pre_tokenizer": {"type": "FancyNewSplitter"}"#)
        }
    }

    @Test("Components missing required fields throw")
    func missingComponentFields() {
        // Strip decoder without start / stop.
        #expect(throws: TokenizerError.self) {
            try Self.tokenizer(extra: #""decoder": {"type": "Strip", "content": " "}"#)
        }
        // TemplateProcessing without `pair`.
        #expect(throws: TokenizerError.self) {
            try Self.tokenizer(extra: #""post_processor": {"type": "TemplateProcessing", "single": []}"#)
        }
        // Replace normalizer with an invalid regular expression.
        #expect(throws: TokenizerError.self) {
            try Self.tokenizer(
                extra: #""normalizer": {"type": "Replace", "pattern": {"Regex": "(unclosed"}, "content": "x"}"#)
        }
    }

    @Test("BPE model without merges throws")
    func missingMerges() throws {
        let data = try Config(jsonString: #"{"model": {"type": "BPE", "vocab": {"a": 0}}}"#)
        let config = try Config(jsonString: #"{"tokenizer_class": "GPT2Tokenizer"}"#)
        #expect(throws: TokenizerError.self) {
            try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data)
        }
    }
}

@Suite("WordLevel model")
struct WordLevelTokenizerTests {
    /// `models/wordlevel/mod.rs::test_tokenize_unk`, `::test_tokenize_missing_unk_token`.
    static func tokenizer(
        vocab: String, unk: String? = "<unk>", pre: String = "WhitespaceSplit"
    ) throws
        -> PreTrainedTokenizer
    {
        let unkField = unk.map { #""unk_token": "\#($0)","# } ?? ""
        let json = """
            {"model": {"type": "WordLevel", \(unkField) "vocab": \(vocab)},
             "pre_tokenizer": {"type": "\(pre)"}}
            """
        return try PreTrainedTokenizer(
            tokenizerConfig: ["tokenizer_class": "PreTrainedTokenizerFast"],
            tokenizerData: try Config(tokenizerJSON: Data(json.utf8)))
    }

    @Test("Whole chunks map to ids, unknown chunks to the unknown token")
    func tokenizeUnknown() throws {
        let tokenizer = try Self.tokenizer(vocab: #"{"<unk>": 0, "a": 1, "b": 2}"#)
        #expect(tokenizer.tokenize(text: "a") == ["a"])
        #expect(tokenizer.tokenize(text: "c") == ["<unk>"])
        #expect(tokenizer.encode(text: "a b c", addSpecialTokens: false) == [1, 2, 0])
        // Unknowns are not fused: one pre-token is always one token.
        #expect(tokenizer.encode(text: "c c", addSpecialTokens: false) == [0, 0])
        #expect(tokenizer.encode(text: "", addSpecialTokens: false) == [])
        #expect(tokenizer.decode(tokens: [1, 2]) == "a b")
    }

    @Test("WordLevel without a usable unknown token is rejected before encoding can lose text")
    func missingUnknownToken() throws {
        for unk: String? in [nil, "<unk>"] {
            #expect(throws: TokenizerError.self) {
                try Self.tokenizer(vocab: #"{"a": 0, "b": 1}"#, unk: unk)
            }
        }
    }

    @Test("Source offsets cover the whole chunk")
    func offsets() throws {
        let tokenizer = try Self.tokenizer(vocab: #"{"<unk>": 0, "hello": 1, "wörld": 2}"#)
        let encoding = try tokenizer.encode(text: "hello wörld nope", addSpecialTokens: false, withOffsets: true)
        #expect(encoding.ids == [1, 2, 0])
        #expect(encoding.offsets ?? [] == [0..<5, 6..<12, 13..<17])
    }

    @Test("Resolves through model.type regardless of tokenizer_class")
    func factory() throws {
        let tokenizer = try Self.tokenizer(vocab: #"{"<unk>": 0, "a": 1}"#)
        #expect(tokenizer.model is WordLevelTokenizer)
        #expect(tokenizer.unknownToken == "<unk>")
        #expect(tokenizer.unknownTokenId == 0)
    }
}
