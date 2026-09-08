import Foundation
import Testing

@testable import Tokenizers

@Suite("Tokenizer regressions")
struct TokenizerRegressionTests {
    private let configuration: Config = ["tokenizer_class": "GPT2Tokenizer"]

    @Test("Folder loading preserves exported BOS policy and Llama special-token decoding")
    func folderPostProcessorPolicy() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let config = #"{"tokenizer_class":"LlamaTokenizerFast","bos_token":"<s>","add_bos_token":true}"#
        let data = #"""
            {"model":{"type":"BPE","vocab":{"a":0,"<s>":1," ":2},"merges":[]},
             "normalizer":{"type":"Prepend","prepend":" "},
             "added_tokens":[{"id":1,"content":"<s>","special":true,"normalized":true}],
             "post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}}
            """#
        try Data(config.utf8).write(to: folder.appendingPathComponent("tokenizer_config.json"))
        try Data(data.utf8).write(to: folder.appendingPathComponent("tokenizer.json"))
        for strict in [true, false] {
            let tokenizer = try await AutoTokenizer.from(modelFolder: folder, strict: strict)
            #expect(tokenizer.encode(text: "a") == [2, 0])
            #expect(tokenizer.decode(tokens: [1, 1]) == "<s><s>")
            let synchronous = try AutoTokenizer.load(from: folder, strict: strict)
            #expect(synchronous.encode(text: "a") == [2, 0])
            let reconstructed = try AutoTokenizer.from(
                tokenizerConfig: Config(jsonString: config), tokenizerData: Config(jsonString: data), strict: strict)
            #expect(reconstructed.encode(text: "a") == [1, 2, 0])
        }
    }

    @Test("Chat source newlines follow Python Jinja without stripping generated output")
    func chatSourceNewlines() throws {
        let data: Config = ["model": ["type": "BPE", "vocab": ["a": 0], "merges": []]]
        let tokenizer = try PreTrainedTokenizer(tokenizerConfig: configuration, tokenizerData: data)
        for (source, expected) in [
            ("a\n", "a"), ("a\n\n", "a\n"), ("a\r\nb\r", "a\nb"),
            (#"{{ "a\n" }}"#, "a\n"),
        ] {
            let actual = try tokenizer.renderChatTemplate(messages: [], chatTemplate: .literal(source))
            #expect(actual.utf8.elementsEqual(expected.utf8))
        }
    }

    @Test("Added tokens match normalized text and Unicode word boundaries")
    func normalizedAddedTokenBoundaries() throws {
        // Expected IDs from tokenizers 0.23.2 (Lowercase + BPE + AddedToken).
        let data = try Config(
            jsonString: #"""
                {"model":{"type":"BPE","vocab":{"a":0,"b":1," ":2,"e":3,"<unk>":4},
                  "merges":[],"unk_token":"<unk>","byte_fallback":false},
                 "normalizer":{"type":"Lowercase"},
                 "added_tokens":[{"id":5,"content":"AB","normalized":true,"single_word":true,
                                  "lstrip":true,"rstrip":true,"special":false}]}
                """#)
        let tokenizer = try PreTrainedTokenizer(tokenizerConfig: configuration, tokenizerData: data)
        let cases: [(String, [Int])] = [
            ("AB", [5]), (" ab ", [5]), ("aABb", [0, 0, 1, 1]), ("éAB", [4, 0, 1]),
            ("AB\u{301}", [0, 1, 4]), ("AB!", [5, 4]), ("\n\tAB\u{a0}", [5]),
            ("AB_AB", [0, 1, 4, 0, 1]), ("AB\u{200d}", [0, 1, 4]),
        ]
        for (text, expected) in cases {
            #expect(tokenizer.encode(text: text, addSpecialTokens: false) == expected)
            #expect(tokenizer.tokenize(text: text).compactMap(tokenizer.convertTokenToId) == expected)
        }
    }

    @Test("JSONSerialization loaders preserve token IDs zero and one")
    func foundationDictionaryLoading() throws {
        let json =
            #"{"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]},"added_tokens":[{"id":3,"content":"<end>","special":true}]}"#
        let object = try JSONSerialization.jsonObject(with: Data(json.utf8))
        let dictionary = try #require(object as? [NSString: Any])
        let data = Config(dictionary)
        #expect(data.model.vocab["a"].integer() == 0)
        #expect(data.model.vocab["b"].integer() == 1)
        let tokenizer = try AutoTokenizer.from(tokenizerConfig: configuration, tokenizerData: data)
        #expect(tokenizer.encode(text: "ab") == [2])
        #expect(tokenizer.convertIdToToken(0) == "a")
        #expect(tokenizer.convertIdToToken(1) == "b")
    }

    @Test("Foundation floating scores retain Double precision")
    func foundationFloatingPrecision() throws {
        let object = try JSONSerialization.jsonObject(
            with: Data(#"{"score":-3.141592653589793,"one":1,"zero":0,"flag":true}"#.utf8))
        let config = Config(try #require(object as? [NSString: Any]))
        #expect(config.score.double() == -3.141592653589793)
        #expect(config.one.integer() == 1)
        #expect(config.zero.integer() == 0)
        #expect(config.flag.boolean() == true)
        #expect(config.flag.integer() == nil)
    }

    @Test("Fast post-processing preserves repeated and absent input sequences")
    func templateSequences() throws {
        for single in [
            #"[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"A","type_id":0}}]"#,
            #"[{"SpecialToken":{"id":"b","type_id":0}}]"#,
        ] {
            let data = try Config(
                jsonString: """
                    {"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]},
                     "post_processor":{"type":"TemplateProcessing","single":\(single),"pair":[]}}
                    """)
            let tokenizer = try PreTrainedTokenizer(tokenizerConfig: configuration, tokenizerData: data)
            let expected = tokenizer.postProcess(tokenizer.tokenize(text: "a")).compactMap(tokenizer.convertTokenToId)
            #expect(tokenizer.encode(text: "a") == expected)
        }
    }

    @Test("Training template suppresses generation suffix and accepts context")
    func trainingTemplate() throws {
        let config: Config = [
            "tokenizer_class": "GPT2Tokenizer",
            "chat_template":
                "{{ messages[0].content }}{% if enable_thinking %}b{% endif %}{% if add_generation_prompt %}c{% endif %}",
        ]
        let data: Config = ["model": ["type": "BPE", "vocab": ["a": 0, "b": 1, "c": 2], "merges": []]]
        let tokenizer = try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data)
        let messages: [Message] = [["role": "user", "content": "a"]]
        #expect(
            try tokenizer.applyChatTemplate(
                messages: messages, tools: nil, additionalContext: ["enable_thinking": true]) == [0, 1, 2])
        #expect(
            try tokenizer.applyChatTemplate(
                messages: messages, chatTemplate: nil, addGenerationPrompt: false, truncation: false, maxLength: nil,
                tools: nil, additionalContext: ["enable_thinking": true]) == [0, 1])
    }

    @Test("Invalid numeric IDs throw before indexing or allocation")
    func invalidIDs() throws {
        for id in [-1, Int.max] {
            #expect(throws: TokenizerError.self) {
                try Vocabulary(entries: [("a", id)])
            }
        }
        #expect(Config([Config("a"), Config(-1)]).token() == nil)
        #expect(Config(any: UInt.max).integer() == nil)
        let data: Config = ["model": ["type": "Unigram", "vocab": [["<unk>", -1.0]], "unk_id": -1]]
        #expect(throws: TokenizerError.self) {
            try AutoTokenizer.from(tokenizerConfig: ["tokenizer_class": "T5Tokenizer"], tokenizerData: data)
        }
    }

    @Test("Negative truncation length throws without trapping")
    func negativeTruncationLength() throws {
        let data: Config = ["model": ["vocab": ["a": 0], "merges": []]]
        let tokenizer = try PreTrainedTokenizer(tokenizerConfig: configuration, tokenizerData: data)
        #expect(throws: TokenizerError.self) {
            try tokenizer.applyChatTemplate(messages: [], chatTemplate: .literal("a"), truncation: true, maxLength: -1)
        }
    }

    @Test("Malformed JSON numbers are rejected", arguments: ["01", "1.", "1e", "1e+", "1-2", "1+2", "1e999"])
    func invalidJSONNumbers(number: String) {
        #expect(throws: JSONConfigError.self) {
            try Config(jsonString: "{\"value\":\(number)}")
        }
    }

    @Test("Malformed JSON strings are rejected")
    func invalidJSONStrings() {
        for data in [Data([0x22, 0xFF, 0x22]), Data([0x22, 0x0A, 0x22]), Data(#""\uD800""#.utf8)] {
            #expect(throws: JSONConfigError.self) { try Config(jsonData: data) }
        }
    }

    @Test("Relaxed tokenizer metadata accepted by upstream still loads")
    func relaxedMetadata() throws {
        let config = try Config(jsonString: #"{"tokenizer_class":"GPT2Tokenizer","model_max_length":Infinity,}"#)
        let data = try Config(jsonString: #"{"model":{"vocab":{"a":0,},"merges":[],},"added_tokens":[],}"#)
        let tokenizer = try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data)
        #expect(tokenizer.encode(text: "a") == [0])
        let array = try Config(jsonString: "[NaN, -Inf, 1,]")
        #expect(array[0].double()?.isNaN == true)
        #expect(array[1].double() == -Double.infinity)
        #expect(array[2].integer() == 1)
    }

    @Test("BPE without byte fallback honors model unknown-token settings")
    func unknownBPECharacters() throws {
        for fuse in [false, true] {
            let data: Config = [
                "model": [
                    "type": "BPE", "vocab": ["<unk>": 0, "a": 1], "merges": [],
                    "unk_token": "<unk>", "byte_fallback": false, "fuse_unk": Config(fuse),
                ]
            ]
            let tokenizer = try PreTrainedTokenizer(tokenizerConfig: configuration, tokenizerData: data)
            #expect(tokenizer.encode(text: "😀😀a") == (fuse ? [0, 1] : [0, 0, 1]))
            #expect(tokenizer.tokenize(text: "😀😀a") == (fuse ? ["<unk>", "a"] : ["<unk>", "<unk>", "a"]))
        }
    }

    @Test("NLLB unknown Unicode agrees with Hugging Face")
    func nllbUnknownUnicode() async throws {
        let tokenizer = try await HubFixtures.tokenizer(for: "Xenova/nllb-200-distilled-600M", strict: false)
        #expect(
            tokenizer.encode(text: "fullwidth: Ｈｅｌｌｏ　Ｗｏｒｌｄ ～", addSpecialTokens: false)
                == [14577, 17263, 419, 248144, 94124, 13855, 248059, 3])
    }

    @Test("Llama exports with a serialized processor can omit BOS metadata")
    func llamaWithoutBOSMetadata() throws {
        let data: Config = [
            "model": ["type": "BPE", "vocab": ["a": 0, "<s>": 1], "merges": []],
            "post_processor": [
                "type": "TemplateProcessing",
                "single": [
                    ["SpecialToken": ["id": "<s>", "type_id": 0]],
                    ["Sequence": ["id": "A", "type_id": 0]],
                ], "pair": [],
            ],
        ]
        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: ["tokenizer_class": "LlamaTokenizer"], tokenizerData: data)
        #expect(tokenizer.encode(text: "a") == [1, 0])
        #expect(throws: TokenizerError.self) {
            try AutoTokenizer.from(
                tokenizerConfig: ["tokenizer_class": "LlamaTokenizer", "add_bos_token": true], tokenizerData: data)
        }
    }

    @Test("JSON validation preserves BOMs within vocabulary keys")
    func embeddedBOMs() throws {
        let config = try Config(jsonString: "{\"\u{feff}#\":0,\"\\ufeff!\":1,\"#\":2}")
        #expect(config.dictionary()?.count == 3)
        #expect(config["\u{feff}#"].integer() == 0)
        #expect(config["\u{feff}!"].integer() == 1)
        #expect(config["#"].integer() == 2)
    }

    @Test("Concurrent inference, training, decoding, and template-cache access")
    func concurrentInferenceAndTraining() async throws {
        let tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: [
                "tokenizer_class": "GPT2Tokenizer",
                "chat_template": "{{ messages[0].content }}{% if add_generation_prompt %}b{% endif %}",
            ],
            tokenizerData: ["model": ["vocab": ["a": 0, "b": 1, "ab": 2], "merges": ["a b"]]]
        )
        try await withThrowingTaskGroup(of: Bool.self) { group in
            for worker in 0..<16 {
                group.addTask {
                    for iteration in 0..<64 {
                        let generation = (iteration + worker).isMultiple(of: 2)
                        let ids = try tokenizer.applyChatTemplate(
                            messages: [["role": "user", "content": "a"]], addGenerationPrompt: generation)
                        if ids != (generation ? [2] : [0]) { return false }
                        if tokenizer.decode(tokens: ids) != (generation ? "ab" : "a") { return false }
                        if tokenizer.encode(text: "abab") != [2, 2] { return false }
                    }
                    return true
                }
            }
            for try await result in group { #expect(result) }
        }
    }
}
