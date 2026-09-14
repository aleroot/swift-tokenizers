import Foundation
import Testing

@testable import Tokenizers

@Suite("Special token metadata")
struct SpecialTokenMetadataTests {
    private let data: Config = [
        "model": [
            "type": "BPE", "vocab": ["<s>": 0, "</s>": 1, "<unk>": 2, "<|endoftext|>": 3, "a": 4], "merges": [],
        ],
        "added_tokens": [
            ["id": 0, "content": "<s>", "special": true],
            ["id": 1, "content": "</s>", "special": true],
            ["id": 2, "content": "<unk>", "special": true],
            ["id": 3, "content": "<|endoftext|>", "special": true],
        ],
    ]

    @Test(arguments: ["RobertaTokenizer", "RobertaTokenizerFast", "XLMRobertaTokenizerFast", "Xlm-RobertaTokenizer"])
    func robertaDefaults(name: String) throws {
        let tokenizer = try AutoTokenizer.from(tokenizerConfig: ["tokenizer_class": Config(name)], tokenizerData: data)
        #expect(tokenizer.bosTokenId == 0)
        #expect(tokenizer.eosTokenId == 1)
        #expect(tokenizer.unknownTokenId == 2)
        // Metadata does not add a BOS/EOS post-processor or a BPE unknown fallback.
        #expect(tokenizer.encode(text: "axb") == [4])
    }

    @Test("GPT-2 defaults fill Phi-4 UNK without replacing its explicit EOS")
    func gpt2Defaults() throws {
        let tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: ["tokenizer_class": "GPT2Tokenizer", "eos_token": "</s>"], tokenizerData: data)
        #expect(tokenizer.bosTokenId == 3)
        #expect(tokenizer.eosTokenId == 1)
        #expect(tokenizer.unknownTokenId == 3)
        #expect(tokenizer.encode(text: "x") == [])
    }

    @Test("Explicit null suppresses a class default; generic classes do not guess")
    func explicitNullAndGeneric() throws {
        let tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: ["tokenizer_class": "GPT2Tokenizer", "bos_token": Config(), "unk_token": Config()],
            tokenizerData: data)
        #expect(tokenizer.bosToken == nil)
        #expect(tokenizer.unknownToken == nil)
        let generic = try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: data)
        #expect(generic.bosTokenId == nil)
        #expect(generic.eosTokenId == nil)
        #expect(generic.unknownTokenId == nil)
    }

    private func withFolder(_ body: (URL) throws -> Void) throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        try Data(
            #"{"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"<s>":2,"</s>":3},"unk_token":"<unk>"}}"#.utf8
        )
        .write(to: folder.appendingPathComponent("tokenizer.json"))
        try body(folder)
    }

    @Test("A serialized tokenizer loads without companion configuration")
    func jsonOnlyFolder() throws {
        try withFolder { folder in
            let tokenizer = try AutoTokenizer.load(from: folder)
            #expect(tokenizer.encode(text: "hello") == [1])
            #expect(tokenizer.bosTokenId == nil)
            #expect(tokenizer.eosTokenId == nil)
        }
    }

    @Test("Legacy special map fills missing fields and respects explicit overrides")
    func specialMap() throws {
        try withFolder { folder in
            try Data(#"{"bos_token":{"content":"<s>","special":true},"eos_token":"</s>"}"#.utf8)
                .write(to: folder.appendingPathComponent("special_tokens_map.json"))
            let mapped = try AutoTokenizer.load(from: folder)
            #expect(mapped.bosTokenId == 2)
            #expect(mapped.eosTokenId == 3)
            try Data(#"{"bos_token":null,"eos_token":"<s>"}"#.utf8)
                .write(to: folder.appendingPathComponent("tokenizer_config.json"))
            let overridden = try AutoTokenizer.load(from: folder)
            #expect(overridden.bosTokenId == nil)
            #expect(overridden.eosTokenId == 2)
        }
    }

    @Test("Model class inference supplies defaults, not generation stop IDs")
    func modelConfiguration() throws {
        try withFolder { folder in
            try Data(#"{"model_type":"roberta","bos_token_id":1,"eos_token_id":[1,2]}"#.utf8)
                .write(to: folder.appendingPathComponent("config.json"))
            let tokenizer = try AutoTokenizer.load(from: folder)
            #expect(tokenizer.bosTokenId == 2)
            #expect(tokenizer.eosTokenId == 3)
        }
    }
}
