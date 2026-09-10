import Foundation
import Testing

@testable import Tokenizers

/// Rust reference outputs generated with tokenizers 0.23.2.
/// Strings are compared as UTF-8: Swift canonical equivalence is too permissive here.
@Suite("Hugging Face boundary audit")
struct CorrectnessAuditTests {
    @Test("An absent decoder separates tokens; Fuse concatenates explicitly")
    func defaultDecoder() throws {
        let model: Config = ["type": "BPE", "vocab": ["a": 0, "b": 1], "merges": []]
        for (decoder, expected): (Config, String) in [(Config(), "a b"), (["type": "Fuse"], "ab")] {
            let tokenizer = try PreTrainedTokenizer(
                tokenizerConfig: [:], tokenizerData: ["model": model, "decoder": decoder])
            #expect(tokenizer.decode(tokens: [0, 1]) == expected)
        }
    }

    @Test("ByteLevel decodes one byte stream with a whole-token literal fallback")
    func byteLevelDecoderComposition() throws {
        let byteLevel: Config = ["type": "ByteLevel"]
        let sequence: Config = ["type": "Sequence", "decoders": [byteLevel, ["type": "WordPiece", "prefix": "##"]]]
        for decoder in [byteLevel, sequence] {
            let tokenizer = try PreTrainedTokenizer(
                tokenizerConfig: [:],
                tokenizerData: [
                    "model": ["type": "BPE", "vocab": ["Ã": 0, "©": 1, "Ġ😀": 2, "é": 3], "merges": []],
                    "added_tokens": [["id": 1, "content": "©", "special": false, "normalized": false]],
                    "decoder": decoder,
                ])
            // The added token completes the preceding UTF-8 sequence.
            #expect(tokenizer.decode(tokens: [0, 1]) == "é")
            #expect(tokenizer.decode(tokens: [2]).utf8.elementsEqual("Ġ😀".utf8))
            #expect(tokenizer.decode(tokens: [3]) == "�")
        }
        let decoder = ByteLevelDecoder(config: [:])
        #expect(decoder.decode(tokens: ["a", "e\u{301}", "b"]) == ["ae\u{301}b"])
        #expect(decoder.decode(tokens: []) == [""])
    }

    @Test("Reference components use their own Unicode repertoires")
    func referenceCharacterClasses() throws {
        let whitespace = WhitespacePreTokenizer(config: ["type": "Whitespace"])
        #expect(whitespace.preTokenize(text: "a\u{32796}b") == ["a", "\u{32796}", "b"])
        #expect(whitespace.preTokenize(text: "a\u{F882}b") == ["a", "\u{F882}", "b"])
        let byteLevel = ByteLevelPreTokenizer(config: ["add_prefix_space": false])
        #expect(byteLevel.preTokenize(text: "\u{32796}\u{301}").count == 1)
        let punctuation = PunctuationPreTokenizer(config: [:])
        #expect(punctuation.preTokenize(text: "a\u{111C9}b") == ["a", "\u{111C9}", "b"])
        #expect(punctuation.preTokenize(text: "a\u{166D}b") == ["a", "\u{166D}", "b"])
        #expect(punctuation.preTokenize(text: "a\u{1E95E}b") == ["a\u{1E95E}b"])
        #expect(BertPreTokenizer(config: [:]).preTokenize(text: "a\u{111C9}b") == ["a", "\u{111C9}", "b"])
    }

    @Test("Metaspace first follows original positions through normalization and rewrites")
    func metaspaceOriginalStart() throws {
        let first: Config = ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": false]
        let never: Config = ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "never", "split": false]
        let rewritten = try #require(
            try PreTokenizerFactory.fromConfig(config: [
                "type": "Sequence",
                "pretokenizers": [
                    ["type": "ByteLevel", "add_prefix_space": false], ["type": "Punctuation"], first,
                ],
            ]))
        #expect(rewritten.preTokenize(text: "\u{327}") == ["▁Ì", "▁§"])
        let cases: [(Config, Config, String, [Int])] = [
            (["type": "StripAccents"], first, "\u{1734}e", [2]),
            (
                ["type": "BertNormalizer", "clean_text": false, "strip_accents": false],
                [
                    "type": "Sequence",
                    "pretokenizers": [
                        ["type": "Split", "pattern": ["Regex": "\\s+"], "behavior": "MergedWithPrevious"],
                        never, first,
                    ],
                ], "中", [0, 0, 1, 0]
            ),
        ]
        for (normalizer, preTokenizer, text, expected) in cases {
            let tokenizer = try PreTrainedTokenizer(
                tokenizerConfig: [:],
                tokenizerData: [
                    "model": ["type": "BPE", "vocab": ["▁": 0, "中": 1, "e": 2], "merges": []],
                    "normalizer": normalizer, "pre_tokenizer": preTokenizer,
                ])
            #expect(tokenizer.encode(text: text) == expected)
            #expect(try tokenizer.encode(text: text, withOffsets: true).ids == expected)
        }
    }

    @Test("Invalid Metaspace markers throw before reaching the byte scanner")
    func invalidMetaspace() {
        for marker in ["", "ab", "e\u{301}"] {
            let config: Config = ["type": "Metaspace", "replacement": Config(marker)]
            #expect(throws: TokenizerError.self) { try PreTokenizerFactory.fromConfig(config: config) }
            #expect(throws: TokenizerError.self) { try DecoderFactory.fromConfig(config: config) }
        }
    }

    @Test("Stochastic BPE configurations are rejected rather than silently encoded deterministically")
    func bpeDropout() throws {
        for dropout in [-1.0, 0.25, 1.0, 1.5] {
            let data: Config = [
                "model": [
                    "type": "BPE", "vocab": ["a": 0], "merges": [], "dropout": Config(dropout),
                ]
            ]
            #expect(throws: TokenizerError.self) {
                try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: data)
            }
        }
        let data: Config = ["model": ["type": "BPE", "vocab": ["a": 0], "merges": [], "dropout": 0.0]]
        let tokenizer = try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: data)
        #expect(tokenizer.encode(text: "a", addSpecialTokens: false) == [0])
    }

    @Test func splitRegexBehaviors() throws {
        let cases: [(String, [String], [String])] = [
            ("Removed", ["a", "b"], ["  ", "  ", "  "]),
            ("Isolated", ["  ", "a", "  ", "b", "  "], ["  ", "a", "  ", "b", "  "]),
            ("MergedWithPrevious", ["  ", "a  ", "b  "], ["  a", "  b", "  "]),
            ("MergedWithNext", ["  a", "  b", "  "], ["  ", "a  ", "b  "]),
            ("Contiguous", ["  ", "a", "  ", "b", "  "], ["  ", "a", "  ", "b", "  "]),
        ]
        for (behavior, normal, inverted) in cases {
            for invert in [false, true] {
                let splitter = try SplitPreTokenizer(config: [
                    "pattern": ["Regex": "\\s+"], "behavior": Config(behavior), "invert": Config(invert),
                ])
                #expect(splitter.preTokenize(text: "  a  b  ") == (invert ? inverted : normal))
            }
        }
    }

    @Test func emptyAndZeroWidthSplits() throws {
        let cases: [(Config, String, [String])] = [
            (["String": ""], "a😀b", ["a", "😀", "b"]),
            (["Regex": ""], "a😀b", ["a", "😀", "b"]),
            (["Regex": "(?=b)"], "ab", ["a", "b"]),
            (["String": "ab"], "xabab", ["ab", "ab"]),
        ]
        for (pattern, text, expected) in cases {
            let splitter = try SplitPreTokenizer(config: [
                "pattern": pattern, "behavior": "Removed", "invert": Config(text == "xabab"),
            ])
            #expect(splitter.preTokenize(text: text) == expected)
        }
    }

    @Test func replacementContentIsLiteral() throws {
        for content in ["$1", "${1}", "\\n", "\\$1"] {
            let normalizer = try #require(
                try NormalizerFactory.fromConfig(config: [
                    "type": "Replace", "pattern": ["Regex": "(a)"], "content": Config(content),
                ]))
            #expect(Array(normalizer.normalize(text: "a").utf8) == Array(content.utf8))
        }
        let empty = try #require(
            try NormalizerFactory.fromConfig(config: [
                "type": "Replace", "pattern": ["String": ""], "content": "X",
            ]))
        #expect(empty.normalize(text: "") == "")
        #expect(empty.normalize(text: "a😀") == "XaX😀X")
    }

    @Test func metaspaceDecodeRemovesAllFirstTokenMarkers() throws {
        for scheme in ["always", "first", "never"] {
            let decoder = try #require(
                try DecoderFactory.fromConfig(config: [
                    "type": "Metaspace", "replacement": "▁", "prepend_scheme": Config(scheme),
                ]))
            #expect(decoder.decode(tokens: ["▁▁a▁b", "▁c"]).joined() == (scheme == "never" ? "  a b c" : "ab c"))
            #expect(decoder.decode(tokens: [" a"]).joined() == " a")
        }
    }

    @Test func rewritesPreserveLossOfOriginalStart() throws {
        let splitters: [Config] = [
            ["type": "WhitespaceSplit"],
            ["type": "Split", "pattern": ["Regex": "\\s+"], "behavior": "Removed"],
        ]
        for splitter in splitters {
            let pre = try #require(
                try PreTokenizerFactory.fromConfig(config: [
                    "type": "Sequence",
                    "pretokenizers": [
                        splitter,
                        ["type": "ByteLevel", "add_prefix_space": false],
                        ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "first"],
                    ],
                ]))
            #expect(pre.preTokenize(text: "  a  b  ", options: [.firstSection]) == ["a", "b"])
            #expect(pre.preTokenize(text: "a b", options: [.firstSection]) == ["▁a", "b"])
        }
    }

    @Test func rejectedAddedTokenDoesNotRetryOverlappingContents() throws {
        for shorter in ["a", "b", "bc"] {
            let splitter = try #require(
                AddedTokenSplitter(tokens: [
                    .init(content: "ab", id: 1, lstrip: false, rstrip: false, scalarCount: 2, singleWord: true),
                    .init(
                        content: shorter, id: 2, lstrip: false, rstrip: false, scalarCount: shorter.unicodeScalars.count
                    ),
                ]))
            let ids = splitter.split("abc").compactMap { section -> Int? in
                if case .token(_, let id) = section { return id }
                return nil
            }
            #expect(ids.isEmpty)
        }
    }

    @Test func whitespaceStrippingDoesNotHideLaterAddedTokens() throws {
        let splitter = try #require(
            AddedTokenSplitter(tokens: [
                .init(content: "a", id: 1, lstrip: false, rstrip: true, scalarCount: 1),
                .init(content: " b", id: 2, lstrip: true, rstrip: false, scalarCount: 2),
            ]))
        var ids: [Int] = []
        var offsets: [Range<Int>] = []
        Array("a  b".utf8).withUnsafeBufferPointer { bytes in
            splitter.scan(
                bytes: bytes, onText: { _ in },
                onToken: { id, offset in
                    ids.append(id)
                    offsets.append(offset)
                })
        }
        #expect(ids == [1, 2])
        #expect(offsets == [0..<3, 3..<4])
    }

    @Test func addedVocabularyIsNotModelVocabulary() throws {
        let cases: [(Config, [(String, [Int])])] = [
            // wordpiece
            (
                [
                    "model": [
                        "type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                        "max_input_chars_per_word": 100, "vocab": ["[UNK]": 0, "x": 1, "##d": 2],
                    ],
                    "added_tokens": [
                        [
                            "id": 3, "content": "abc", "single_word": true, "lstrip": false, "rstrip": false,
                            "normalized": true, "special": false,
                        ]
                    ],
                ], [("abc", [3]), ("abcd", [0]), ("xabcd", [0])]
            ),
            // wordpiece-sparse
            (
                [
                    "model": [
                        "type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                        "max_input_chars_per_word": 100, "vocab": ["[UNK]": 0, "x": 4, "##d": 5],
                    ],
                    "added_tokens": [
                        [
                            "id": 3, "content": "abc", "single_word": true, "lstrip": false, "rstrip": false,
                            "normalized": true, "special": false,
                        ]
                    ],
                ], [("x", [4]), ("abc", [3]), ("abcd", [0])]
            ),
            // bpe
            (
                [
                    "model": [
                        "type": "BPE", "unk_token": "<unk>", "fuse_unk": false, "byte_fallback": false,
                        "ignore_merges": false, "vocab": ["<unk>": 0, "a": 1, "b": 2], "merges": [],
                    ],
                    "added_tokens": [
                        [
                            "id": 3, "content": "z", "single_word": true, "lstrip": false, "rstrip": false,
                            "normalized": true, "special": false,
                        ],
                        [
                            "id": 4, "content": "é", "single_word": true, "lstrip": false, "rstrip": false,
                            "normalized": true, "special": false,
                        ],
                    ],
                ], [("z", [3]), ("azb", [1, 0, 2]), ("é", [4]), ("aéb", [1, 0, 2])]
            ),
            // bpe-affix
            (
                [
                    "model": [
                        "type": "BPE", "unk_token": "<unk>", "continuing_subword_prefix": "##", "fuse_unk": false,
                        "byte_fallback": false, "ignore_merges": false, "vocab": ["<unk>": 0, "a": 1], "merges": [],
                    ],
                    "added_tokens": [
                        [
                            "id": 2, "content": "##z", "single_word": false, "lstrip": false, "rstrip": false,
                            "normalized": true, "special": false,
                        ]
                    ],
                ], [("az", [1, 0]), ("##z", [2])]
            ),
            // bpe-bytefallback
            (
                [
                    "model": [
                        "type": "BPE", "unk_token": "<unk>", "fuse_unk": false, "byte_fallback": true,
                        "ignore_merges": false, "vocab": ["<unk>": 0], "merges": [],
                    ],
                    "added_tokens": [
                        [
                            "id": 1, "content": "<0x61>", "single_word": false, "lstrip": false, "rstrip": false,
                            "normalized": true, "special": false,
                        ]
                    ],
                ], [("a", [0]), ("aaa", [0, 0, 0]), ("<0x61>", [1])]
            ),
            // unigram-bytefallback
            (
                [
                    "model": ["type": "Unigram", "unk_id": 0, "vocab": [["<unk>", 0.0]], "byte_fallback": true],
                    "added_tokens": [
                        [
                            "id": 1, "content": "<0x61>", "single_word": false, "lstrip": false, "rstrip": false,
                            "normalized": true, "special": false,
                        ]
                    ],
                ], [("a", [0]), ("aaa", [0]), ("<0x61>", [1])]
            ),
        ]
        for (config, inputs) in cases {
            let tokenizer = try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: config)
            for (text, expected) in inputs {
                // Repeat to cover the cached path as well as initial segmentation.
                for _ in 0..<2 {
                    #expect(tokenizer.encode(text: text, addSpecialTokens: false) == expected)
                }
            }
        }
    }
}

/// Untrusted `tokenizer.json` files must not be able to trade a few bytes of input for
/// unbounded work or memory, and no public entry point may terminate the process.
@Suite("Resource limits")
struct ResourceLimitTests {
    static func tokenizer(_ json: String) throws -> PreTrainedTokenizer {
        try PreTrainedTokenizer(
            tokenizerConfig: ["tokenizer_class": "PreTrainedTokenizerFast"],
            tokenizerData: try Config(tokenizerJSON: Data(json.utf8)))
    }

    @Test("A vocabulary that declares one token at a huge id is refused")
    func sparseIdSpaceIsRejected() throws {
        // `id → token` is a direct index, so this used to allocate about a gigabyte.
        for id in [Vocabulary.maximumIdSpace, 63_999_999, Int.max / 2] {
            #expect(throws: TokenizerError.self) {
                try Self.tokenizer(#"{"model":{"type":"BPE","vocab":{"a":\#(id)},"merges":[]}}"#)
            }
        }
        #expect(throws: TokenizerError.self) {
            try Self.tokenizer(#"{"model":{"type":"BPE","vocab":{"a":-1},"merges":[]}}"#)
        }
    }

    @Test("A small vocabulary may still place special tokens at a base model's high ids")
    func sparseIdSpaceWithinTheFloorIsAccepted() throws {
        let entries = (0..<8).map { #""t\#($0)": \#($0)"# } + [#""<|endoftext|>": 128255"#]
        let tokenizer = try Self.tokenizer(
            """
            {"model":{"type":"WordLevel","unk_token":"t0","vocab":{\(entries.joined(separator: ","))}},
             "pre_tokenizer":{"type":"WhitespaceSplit"}}
            """)
        #expect(tokenizer.encode(text: "t3 <|endoftext|>", addSpecialTokens: false) == [3, 128255])
    }

    @Test("The id space scales with the number of entries, between a floor and a ceiling")
    func idSpaceBounds() {
        #expect(Vocabulary.idSpace(entryCount: 0) == 1 << 18)
        #expect(Vocabulary.idSpace(entryCount: 1) == 1 << 18)
        #expect(Vocabulary.idSpace(entryCount: 200_000) == Vocabulary.maximumIdSpace)
        #expect(Vocabulary.idSpace(entryCount: Int.max) == Vocabulary.maximumIdSpace)
        // Monotonic, so growing a vocabulary never narrows the space it may span.
        var previous = 0
        for count in [0, 1, 100, 4096, 1 << 16, 1 << 20, 1 << 24] {
            let space = Vocabulary.idSpace(entryCount: count)
            #expect(space >= previous)
            previous = space
        }
    }

    @Test("A caller-supplied vocabulary never traps, whatever the ids")
    func callerSuppliedVocabularyIsSanitised() {
        // This initializer is not throwing, so unrepresentable ids are dropped instead.
        let tokenizer = BertTokenizer(
            vocab: ["ok": 1, "negative": -5, "huge": Int.max, "past-the-limit": Vocabulary.maximumIdSpace],
            merges: nil)
        #expect(tokenizer.convertTokenToId("ok") == 1)
        #expect(tokenizer.convertIdToToken(1) == "ok")
        for dropped in ["negative", "huge", "past-the-limit"] {
            #expect(tokenizer.convertTokenToId(dropped) == nil)
        }
        #expect(BertTokenizer(vocab: [:], merges: nil).convertIdToToken(0) == nil)
    }

    @Test("An implausible regular expression is refused instead of compiled")
    func regexLengthIsBounded() {
        let long = String(repeating: "a", count: maximumRegexLength + 1)
        #expect(throws: TokenizerError.self) { try compileRegex(long, component: "test") }
        #expect(throws: Never.self) {
            try compileRegex(String(repeating: "a", count: maximumRegexLength), component: "test")
        }
        #expect(throws: TokenizerError.self) {
            try Self.tokenizer(
                #"{"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"#
                    + #""pre_tokenizer":{"type":"Split","pattern":{"Regex":"\#(long)"},"behavior":"Isolated"}}"#)
        }
    }

    @Test("The compiled chat-template cache is bounded")
    func chatTemplateCacheIsBounded() throws {
        let tokenizer = try Self.tokenizer(#"{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}"#)
        // A caller that builds a fresh template per request must not retain all of them.
        for i in 0...(PreTrainedTokenizer.chatTemplateCacheLimit * 2) {
            _ = try tokenizer.renderChatTemplate(
                messages: [["role": "user", "content": "hi"]], chatTemplate: .literal("{{ \(i) }}"))
        }
        #expect(tokenizer.compiledChatTemplateCount <= PreTrainedTokenizer.chatTemplateCacheLimit)
        // The cache still serves the common case of one reused template.
        let single = try Self.tokenizer(#"{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}"#)
        for _ in 0..<32 {
            _ = try single.renderChatTemplate(
                messages: [["role": "user", "content": "hi"]], chatTemplate: .literal("{{ 1 }}"))
        }
        #expect(single.compiledChatTemplateCount == 1)
    }
}
