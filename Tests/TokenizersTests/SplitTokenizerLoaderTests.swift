import Foundation
import Testing

@testable import Tokenizers

@Suite("Split-file tokenizer loading")
struct SplitTokenizerLoaderTests {
    private func withFolder(_ files: [String: String], body: (URL) throws -> Void) throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        for (name, content) in files {
            try content.write(to: folder.appendingPathComponent(name), atomically: true, encoding: .utf8)
        }
        try body(folder)
    }

    @Test("BERT and MPNet vocab.txt preserve line IDs, normalization, special tokens and CRLF")
    func wordPiece() throws {
        for mpnet in [false, true] {
            let vocab =
                mpnet
                ? "[UNK]\r\n<s>\r\n</s>\r\n<pad>\r\n<mask>\r\nhello\r\nworld\r\n##s\r\n"
                : "[UNK]\r\n[CLS]\r\n[SEP]\r\n[PAD]\r\n[MASK]\r\nhello\r\nworld\r\n##s\r\n"
            try withFolder([
                "config.json": mpnet ? #"{"model_type":"mpnet"}"# : #"{"model_type":"bert"}"#, "vocab.txt": vocab,
            ]) { folder in
                let tokenizer = try AutoTokenizer.load(from: folder)
                #expect(tokenizer.encode(text: "Héllo worlds") == [1, 5, 6, 7, 2])
                #expect(tokenizer.decode(tokens: [1, 5, 6, 7, 2], skipSpecialTokens: true) == "hello worlds")
                #expect(try tokenizer.encode(text: "hello worlds", maxLength: 3).ids == [1, 5, 2])
                #expect(tokenizer.encode(text: mpnet ? "<mask>" : "[MASK]", addSpecialTokens: false) == [4])
            }
        }
    }

    @Test("WordPiece conversion supplies cleanup defaults and preserves explicit opt-outs")
    func cleanupDefault() throws {
        for enabled in [true, false] {
            let config =
                enabled
                ? #"{"tokenizer_class":"BertTokenizer"}"#
                : #"{"tokenizer_class":"BertTokenizer","clean_up_tokenization_spaces":false}"#
            try withFolder([
                "tokenizer_config.json": config,
                "vocab.txt": "[UNK]\n[CLS]\n[SEP]\n[PAD]\n[MASK]\ni\n'\nm\n",
            ]) { folder in
                let tokenizer = try AutoTokenizer.load(from: folder)
                #expect(tokenizer.decode(tokens: [5, 6, 7]) == (enabled ? "i'm" : "i ' m"))
                var stream = tokenizer.makeStreamDecoder()
                var output = ""
                for id in [5, 6, 7] { output += try stream.append(token: id) }
                output += stream.finish()
                #expect(output == tokenizer.decode(tokens: [5, 6, 7]))
            }
        }
    }

    @Test("Qwen NFC and one-digit splitting differ deliberately from GPT2")
    func bpeProfiles() throws {
        for qwen in [false, true] {
            let files = [
                "tokenizer_config.json": qwen
                    ? #"{"tokenizer_class":"Qwen2TokenizerFast"}"#
                    : #"{"tokenizer_class":"GPT2Tokenizer","add_bos_token":true}"#,
                "vocab.json": #"{"1":0,"2":1,"12":2,"Ã":3,"©":4,"Ã©":5,"e":6,"Ì":7,"ģ":8,"<|endoftext|>":9}"#,
                "merges.txt": "#version: 0.2\n1 2\nÃ ©\n",
            ]
            try withFolder(files) { folder in
                let tokenizer = try AutoTokenizer.load(from: folder)
                #expect(tokenizer.encode(text: "12", addSpecialTokens: false) == (qwen ? [0, 1] : [2]))
                #expect(tokenizer.encode(text: "e\u{301}", addSpecialTokens: false) == (qwen ? [5] : [6, 7, 8]))
                if !qwen { #expect(tokenizer.encode(text: "12") == [9, 2]) }
            }
        }
    }

    @Test("CLIP lowercases, splits digits, inserts BOS/EOS and removes decoded word suffixes")
    func clip() throws {
        try withFolder([
            "tokenizer_config.json": #"{"tokenizer_class":"CLIPTokenizer"}"#,
            "vocab.json":
                #"{"h":0,"i</w>":1,"hi</w>":2,"1</w>":3,"2</w>":4,"<|startoftext|>":5,"<|endoftext|>":6,"Ĝ</w>":7}"#,
            "merges.txt": "#version: 0.2\nh i</w>\n",
        ]) { folder in
            let tokenizer = try AutoTokenizer.load(from: folder)
            #expect(tokenizer.encode(text: "  HI\t12 ") == [5, 2, 3, 4, 6])
            #expect(tokenizer.decode(tokens: [5, 2, 3, 4, 6], skipSpecialTokens: true) == "hi 1 2")
            // Python's CLIP decode wrapper strips U+001C..U+001F as well as Unicode White_Space.
            #expect(tokenizer.decode(tokens: [7, 2, 7], skipSpecialTokens: true) == "hi")
        }
    }

    @Test("RoBERTa profile preserves source offsets and special-token metadata")
    func roberta() throws {
        try withFolder([
            "tokenizer_config.json": #"{"tokenizer_class":"RobertaTokenizer","add_prefix_space":true}"#,
            "vocab.json": #"{"Ġ":0,"a":1,"Ġa":2,"<s>":3,"</s>":4,"<unk>":5,"<pad>":6,"<mask>":7}"#,
            "merges.txt": "#version: 0.2\nĠ a\n",
        ]) { folder in
            let tokenizer = try AutoTokenizer.load(from: folder)
            let result = try tokenizer.encode(text: "a", withOffsets: true)
            #expect(result.ids == [3, 2, 4])
            #expect(result.offsets == [nil, 0..<1, nil])
            #expect(tokenizer.decode(tokens: result.ids, skipSpecialTokens: true) == " a")
        }
    }

    @Test("Explicit added-token flags and standalone chat templates survive conversion")
    func metadata() throws {
        try withFolder([
            "tokenizer_config.json":
                #"{"tokenizer_class":"BertTokenizer","added_tokens_decoder":{"7":{"content":"<X>","special":true,"normalized":false,"lstrip":true}}}"#,
            "vocab.txt": "[UNK]\n[CLS]\n[SEP]\n[PAD]\n[MASK]\na\n",
            "added_tokens.json": #"{"<X>":7}"#,
            "special_tokens_map.json": #"{"additional_special_tokens":["<X>"]}"#,
            "chat_template.jinja": "{% generation %}{{ messages[0].content }}{% endgeneration %}",
        ]) { folder in
            let tokenizer = try AutoTokenizer.load(from: folder)
            let result = try tokenizer.encode(text: "a   <X>", addSpecialTokens: false, withOffsets: true)
            #expect(result.ids == [5, 7])
            #expect(result.offsets == [0..<1, 1..<7])
            #expect(tokenizer.decode(tokens: [7], skipSpecialTokens: true) == "")
            #expect(
                try tokenizer.applyChatTemplateWithAssistantMask(messages: [["content": "a"]]).assistantMask == [true])
        }
    }

    @Test("Conflicts, unknown profiles and malformed merges fail; tokenizer.json remains authoritative")
    func invalidAndPrecedence() throws {
        let base = [
            "tokenizer_config.json": #"{"tokenizer_class":"GPT2Tokenizer"}"#,
            "vocab.json": #"{"a":0,"b":1,"ab":2,"<|endoftext|>":3}"#, "merges.txt": "a b\n",
        ]
        for extra in [
            ["added_tokens.json": #"{"new":0}"#], ["merges.txt": "a  b\n"],
            ["vocab.json": "{\"a\":\(Int.max)}"],
            ["vocab.json": #"{"a":1073741824}"#],
            ["tokenizer_config.json": #"{"tokenizer_class":"Unknown"}"#],
        ] {
            try withFolder(base.merging(extra) { _, new in new }) { folder in
                #expect(throws: TokenizerError.self) { try AutoTokenizer.load(from: folder) }
            }
        }
        try withFolder(
            base.merging([
                "merges.txt": "invalid",
                "tokenizer.json": #"{"model":{"type":"WordLevel","vocab":{"ab":0,"[UNK]":1},"unk_token":"[UNK]"}}"#,
            ]) { _, new in new }
        ) { folder in
            let tokenizer = try AutoTokenizer.load(from: folder)
            #expect(tokenizer.encode(text: "ab") == [0])
        }
    }
}
