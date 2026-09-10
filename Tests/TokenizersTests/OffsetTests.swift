import Foundation
import Testing

@testable import Tokenizers

/// Focused expectations from Hugging Face tokenizers 0.23.2, expressed in source bytes.
@Suite("Original text offsets")
struct OffsetTests {
    @Test("ByteLevel trims Unicode whitespace in original scalar coordinates")
    func unicodeWhitespaceTrimming() throws {
        let tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: [:],
            tokenizerData: [
                "model": ["type": "BPE", "vocab": ["?": 0, "\u{2029}": 1, "[X]": 2], "merges": []],
                "added_tokens": [
                    [
                        "id": 2, "content": "[X]", "normalized": false, "special": false, "lstrip": true,
                        "rstrip": true,
                    ]
                ],
                "post_processor": ["type": "ByteLevel", "trim_offsets": true, "add_prefix_space": false],
            ])
        try check(tokenizer, text: "?\u{2029}", ids: [0, 1], offsets: [0..<1, 4..<4], utf16: [0..<1, 2..<2])
        try check(tokenizer, text: "\t\u{85}[X]\u{2029}", ids: [2], offsets: [3..<6], utf16: [2..<5])
    }

    private func check(
        _ tokenizer: any Tokenizer, text: String, special: Bool = false,
        ids: [Int], offsets: [Range<Int>?], utf16: [Range<Int>?]
    ) throws {
        let result = try tokenizer.encode(text: text, addSpecialTokens: special, withOffsets: true)
        #expect(result.ids == ids)
        #expect(result.ids == tokenizer.encode(text: text, addSpecialTokens: special))
        #expect(result.offsets == offsets)
        #expect(result.utf16Ranges() == utf16.map { $0.map { NSRange(location: $0.lowerBound, length: $0.count) } })
    }

    @Test func normalizationAndPretokenizationOffsets() throws {
        let cases: [(Config, String, [String], [Range<Int>?], [Range<Int>?])] = [
            // normalizer/lowercase
            (
                ["normalizer": ["type": "Lowercase"]], "İΣΟΣ", ["i", "\u{307}", "σ", "ο", "σ"],
                [0..<2, 0..<2, 2..<4, 4..<6, 6..<8], [0..<1, 0..<1, 1..<2, 2..<3, 3..<4]
            ),
            // normalizer/nfd
            (
                ["normalizer": ["type": "NFD"]], "café cafe\u{301}",
                ["c", "a", "f", "e", "\u{301}", " ", "c", "a", "f", "e", "\u{301}"],
                [0..<1, 1..<2, 2..<3, 3..<5, 3..<5, 5..<6, 6..<7, 7..<8, 8..<9, 9..<10, 10..<12],
                [0..<1, 1..<2, 2..<3, 3..<4, 3..<4, 4..<5, 5..<6, 6..<7, 7..<8, 8..<9, 9..<10]
            ),
            // normalizer/nfc
            (["normalizer": ["type": "NFC"]], "e\u{301}\u{327}", ["ȩ", "\u{301}"], [0..<1, 3..<5], [0..<1, 2..<3]),
            // normalizer/nfkd
            (
                ["normalizer": ["type": "NFKD"]], "ﬁﬀ①Ａ", ["f", "i", "f", "f", "1", "A"],
                [0..<3, 0..<3, 3..<6, 3..<6, 6..<9, 9..<12], [0..<1, 0..<1, 1..<2, 1..<2, 2..<3, 3..<4]
            ),
            // normalizer/nfkc
            (
                ["normalizer": ["type": "NFKC"]], "ﬁﬀ①Ａ", ["f", "i", "f", "f", "1", "A"],
                [0..<3, 0..<3, 3..<6, 3..<6, 6..<9, 9..<12], [0..<1, 0..<1, 1..<2, 1..<2, 2..<3, 3..<4]
            ),
            // normalizer/bert
            (
                [
                    "normalizer": [
                        "type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "lowercase": true,
                    ]
                ], "a\u{0}b", ["a", "b"], [0..<1, 2..<3], [0..<1, 2..<3]
            ),
            // normalizer/stripAccents
            (["normalizer": ["type": "StripAccents"]], "a\u{301}b", ["a", "b"], [0..<1, 3..<4], [0..<1, 2..<3]),
            // normalizer/strip
            (
                ["normalizer": ["type": "Strip", "strip_left": true, "strip_right": true]], "  a  b  ",
                ["a", " ", " ", "b"], [2..<3, 3..<4, 4..<5, 5..<6], [2..<3, 3..<4, 4..<5, 5..<6]
            ),
            // normalizer/prepend
            (
                ["normalizer": ["type": "Prepend", "prepend": "xx"]], "ab", ["x", "x", "a", "b"],
                [0..<1, 0..<1, 0..<1, 1..<2], [0..<1, 0..<1, 0..<1, 1..<2]
            ),
            // normalizer/replace
            (
                ["normalizer": ["type": "Replace", "pattern": ["String": "ab"], "content": "XY"]], "ab", ["X", "Y"],
                [1..<2, 1..<2], [1..<2, 1..<2]
            ),
            // normalizer/replaceEmpty
            (
                ["normalizer": ["type": "Replace", "pattern": ["String": ""], "content": "X"]], "ab",
                ["X", "a", "X", "b", "X"], [0..<0, 0..<1, 0..<1, 1..<2, 1..<2], [0..<0, 0..<1, 0..<1, 1..<2, 1..<2]
            ),
            // normalizer/replaceRegex
            (
                ["normalizer": ["type": "Replace", "pattern": ["Regex": "\\s+"], "content": " "]], "  a  b  ",
                [" ", "a", " ", "b", " "], [1..<2, 2..<3, 4..<5, 5..<6, 7..<8], [1..<2, 2..<3, 4..<5, 5..<6, 7..<8]
            ),
            // normalizer/replaceZero
            (
                ["normalizer": ["type": "Replace", "pattern": ["Regex": "(?=b)"], "content": "X"]], "ab",
                ["a", "X", "b"], [0..<1, 0..<1, 1..<2], [0..<1, 0..<1, 1..<2]
            ),
            // normalizer/sequence
            (
                [
                    "normalizer": [
                        "type": "Sequence",
                        "normalizers": [
                            ["type": "NFKC"], ["type": "Lowercase"],
                            ["type": "Replace", "pattern": ["Regex": "\\s+"], "content": " "],
                        ],
                    ]
                ], "ﬁﬀ①Ａ", ["f", "i", "f", "f", "1", "a"], [0..<3, 0..<3, 3..<6, 3..<6, 6..<9, 9..<12],
                [0..<1, 0..<1, 1..<2, 1..<2, 2..<3, 3..<4]
            ),
            // pre/byte/True/False
            (
                [
                    "pre_tokenizer": [
                        "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": false,
                    ]
                ], "A😀B", ["Ġ", "A", "ð", "Ł", "ĺ", "Ģ", "B"], [0..<1, 0..<1, 1..<5, 1..<5, 1..<5, 1..<5, 5..<6],
                [0..<1, 0..<1, 1..<3, 1..<3, 1..<3, 1..<3, 3..<4]
            ),
            // pre/meta/never/False
            (
                [
                    "pre_tokenizer": [
                        "type": "Metaspace", "replacement": "▁", "prepend_scheme": "never", "split": false,
                    ]
                ], "  a  b  ", ["▁", "▁", "a", "▁", "▁", "b", "▁", "▁"],
                [0..<1, 1..<2, 2..<3, 3..<4, 4..<5, 5..<6, 6..<7, 7..<8],
                [0..<1, 1..<2, 2..<3, 3..<4, 4..<5, 5..<6, 6..<7, 7..<8]
            ),
            // pre/meta/first/True
            (
                ["pre_tokenizer": ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": true]],
                "  a  b  ", ["▁", "▁", "a", "▁", "▁", "b", "▁", "▁"],
                [0..<1, 1..<2, 2..<3, 3..<4, 4..<5, 5..<6, 6..<7, 7..<8],
                [0..<1, 1..<2, 2..<3, 3..<4, 4..<5, 5..<6, 6..<7, 7..<8]
            ),
            // pre/meta/always/True
            (
                [
                    "pre_tokenizer": [
                        "type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true,
                    ]
                ], "  a  b  ", ["▁", "▁", "a", "▁", "▁", "b", "▁", "▁"],
                [0..<1, 1..<2, 2..<3, 3..<4, 4..<5, 5..<6, 6..<7, 7..<8],
                [0..<1, 1..<2, 2..<3, 3..<4, 4..<5, 5..<6, 6..<7, 7..<8]
            ),
            // pre/sequence
            (
                [
                    "pre_tokenizer": [
                        "type": "Sequence",
                        "pretokenizers": [
                            ["type": "WhitespaceSplit"],
                            ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true],
                            ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": true],
                        ],
                    ]
                ], "  a  b  ", ["a", "b"], [2..<3, 5..<6], [2..<3, 5..<6]
            ),
            // pre/byteByte
            (
                [
                    "pre_tokenizer": [
                        "type": "Sequence",
                        "pretokenizers": [
                            ["type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true],
                            ["type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true],
                        ],
                    ]
                ], "A😀B", ["Ġ", "Ä", "ł", "A", "Ġ", "Ã", "°", "Å", "ģ", "Ä", "º", "Ä", "¢", "Ġ", "B"],
                [
                    0..<1, 0..<1, 0..<1, 0..<1, 1..<5, 1..<5, 1..<5, 1..<5, 1..<5, 1..<5, 1..<5, 1..<5, 1..<5, 5..<6,
                    5..<6,
                ],
                [
                    0..<1, 0..<1, 0..<1, 0..<1, 1..<3, 1..<3, 1..<3, 1..<3, 1..<3, 1..<3, 1..<3, 1..<3, 1..<3, 3..<4,
                    3..<4,
                ]
            ),
            // post/byte/True/True
            (
                [
                    "pre_tokenizer": [
                        "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true,
                    ],
                    "post_processor": [
                        "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true,
                    ],
                ], "é😀\nhello", ["Ġ", "Ã", "©", "ð", "Ł", "ĺ", "Ģ", "Ċ", "h", "e", "l", "l", "o"],
                [0..<0, 0..<2, 0..<2, 2..<6, 2..<6, 2..<6, 2..<6, 6..<7, 7..<8, 8..<9, 9..<10, 10..<11, 11..<12],
                [0..<0, 0..<1, 0..<1, 1..<3, 1..<3, 1..<3, 1..<3, 3..<4, 4..<5, 5..<6, 6..<7, 7..<8, 8..<9]
            ),
        ]
        for (components, text, tokens, offsets, utf16) in cases {
            let spellings = Set(tokens).sorted()
            let vocabulary = Dictionary(
                uniqueKeysWithValues: spellings.enumerated().map { ($0.element, Config($0.offset)) })
            var data = components.dictionary(or: [:])
            data["model"] = ["type": "BPE", "vocab": Config(vocabulary), "merges": []]
            let tokenizer = try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: Config(data))
            try check(
                tokenizer, text: text, ids: tokens.compactMap(tokenizer.convertTokenToId), offsets: offsets,
                utf16: utf16)
        }
    }

    @Test(arguments: ["distilgpt2", "Qwen/Qwen3-0.6B", "google-bert/bert-base-uncased", "google-t5/t5-small"])
    func referenceModels(model: String) async throws {
        let tokenizer = try await HubFixtures.tokenizer(for: model)
        switch model {
        case "distilgpt2":
            try check(
                tokenizer, text: "A😀B", special: false,
                ids: [32, 47249, 222, 33], offsets: [0..<1, 1..<5, 1..<5, 5..<6], utf16: [0..<1, 1..<3, 1..<3, 3..<4])
            try check(
                tokenizer, text: "café cafe\u{301}", special: true,
                ids: [66, 1878, 2634, 26725, 136, 223], offsets: [0..<1, 1..<3, 3..<5, 5..<10, 10..<12, 10..<12],
                utf16: [0..<1, 1..<3, 3..<4, 4..<9, 9..<10, 9..<10])
            try check(
                tokenizer, text: "ﬁﬀ①Ａ", special: true,
                ids: [171, 105, 223, 171, 105, 222, 158, 239, 254, 171, 120, 94],
                offsets: [0..<3, 0..<3, 0..<3, 3..<6, 3..<6, 3..<6, 6..<9, 6..<9, 6..<9, 9..<12, 9..<12, 9..<12],
                utf16: [0..<1, 0..<1, 0..<1, 1..<2, 1..<2, 1..<2, 2..<3, 2..<3, 2..<3, 3..<4, 3..<4, 3..<4])
        case "Qwen/Qwen3-0.6B":
            try check(
                tokenizer, text: "A😀B", special: false,
                ids: [32, 141334, 33], offsets: [0..<1, 1..<5, 5..<6], utf16: [0..<1, 1..<3, 3..<4])
            try check(
                tokenizer, text: "café cafe\u{301}", special: true,
                ids: [924, 58858, 51950], offsets: [0..<2, 2..<5, 5..<10], utf16: [0..<2, 2..<4, 4..<9])
            try check(
                tokenizer, text: "ﬁﬀ①Ａ", special: true,
                ids: [144300, 145730, 48312, 254, 133054], offsets: [0..<3, 3..<6, 6..<9, 6..<9, 9..<12],
                utf16: [0..<1, 1..<2, 2..<3, 2..<3, 3..<4])
        case "google-bert/bert-base-uncased":
            try check(
                tokenizer, text: "A😀B", special: false,
                ids: [100], offsets: [0..<6], utf16: [0..<4])
            try check(
                tokenizer, text: "café cafe\u{301}", special: true,
                ids: [101, 7668, 7668, 102], offsets: [nil, 0..<5, 6..<10, nil], utf16: [nil, 0..<4, 5..<9, nil])
            try check(
                tokenizer, text: "ﬁﬀ①Ａ", special: true,
                ids: [101, 100, 102], offsets: [nil, 0..<12, nil], utf16: [nil, 0..<4, nil])
        case "google-t5/t5-small":
            try check(
                tokenizer, text: "A😀B", special: false,
                ids: [71, 2, 279], offsets: [0..<1, 1..<5, 5..<6], utf16: [0..<1, 1..<3, 3..<4])
            try check(
                tokenizer, text: "café cafe\u{301}", special: true,
                ids: [11949, 11949, 1], offsets: [0..<5, 6..<10, nil], utf16: [0..<4, 5..<9, nil])
            try check(
                tokenizer, text: "ﬁﬀ①Ａ", special: true,
                ids: [3, 89, 5982, 536, 188, 1], offsets: [0..<3, 0..<3, 0..<6, 6..<9, 9..<12, nil],
                utf16: [0..<1, 0..<1, 0..<2, 2..<3, 3..<4, nil])
        default: Issue.record("Unexpected model: \(model)")
        }
    }

    @Test func modelSegmentationOffsets() throws {
        let cases: [(Config, String, [Int], [Range<Int>?], [Range<Int>?])] = [
            // wordpiece
            (
                [
                    "normalizer": [
                        "type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "lowercase": true,
                    ], "pre_tokenizer": ["type": "BertPreTokenizer"],
                    "post_processor": ["type": "BertProcessing", "sep": ["[SEP]", 2], "cls": ["[CLS]", 1]],
                    "model": [
                        "type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                        "max_input_chars_per_word": 100,
                        "vocab": [
                            "[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "cafe": 3, "hello": 4, "a": 5, "b": 6, "##b": 7,
                            "##a": 8,
                        ],
                    ],
                ], "café cafe\u{301}", [3, 3], [0..<5, 6..<10], [0..<4, 5..<9]
            ),
            // bpeFallback/False
            (
                [
                    "model": [
                        "type": "BPE", "unk_token": "[UNK]", "fuse_unk": true, "byte_fallback": false,
                        "ignore_merges": false, "vocab": ["[UNK]": 0, "a": 1, "b": 2, "ab": 3], "merges": [["a", "b"]],
                    ]
                ], "A😀B", [0], [0..<6], [0..<4]
            ),
            // unigram/False
            (
                [
                    "model": [
                        "type": "Unigram", "unk_id": 0,
                        "vocab": [["[UNK]", 0.0], ["a", -1.0], ["b", -1.0], ["ab", -0.1]], "byte_fallback": false,
                    ]
                ], "A😀B", [0], [0..<6], [0..<4]
            ),
            // unigramByteLevel/False
            (
                [
                    "pre_tokenizer": [
                        "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false,
                    ],
                    "post_processor": [
                        "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true,
                    ],
                    "model": [
                        "type": "Unigram", "unk_id": 0, "vocab": [["[UNK]", 0.0], ["a", -1.0], ["b", -1.0]],
                        "byte_fallback": false,
                    ],
                ], "  a  b  ", [0, 1, 0, 2, 0], [2..<2, 2..<3, 5..<5, 5..<6, 8..<8],
                [2..<2, 2..<3, 5..<5, 5..<6, 8..<8]
            ),
            // ignoreMerges
            (
                [
                    "model": [
                        "type": "BPE", "unk_token": "[UNK]", "fuse_unk": false, "byte_fallback": false,
                        "ignore_merges": true, "vocab": ["[UNK]": 0, "a": 1, "b": 2, "ab": 3], "merges": [],
                    ]
                ], "ab", [3], [0..<2], [0..<2]
            ),
            // affixes
            (
                [
                    "model": [
                        "type": "BPE", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                        "end_of_word_suffix": "</w>", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                        "vocab": [
                            "[UNK]": 0, "a": 1, "b": 2, "##a": 3, "##b": 4, "a</w>": 5, "b</w>": 6, "##a</w>": 7,
                            "##b</w>": 8, "ab</w>": 9,
                        ], "merges": [["a", "##b</w>"]],
                    ]
                ], "ab", [9], [0..<2], [0..<2]
            ),
        ]
        for (config, text, ids, offsets, utf16) in cases {
            let tokenizer = try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: config)
            try check(tokenizer, text: text, ids: ids, offsets: offsets, utf16: utf16)
        }
    }

    @Test func addedTokenOffsetsIncludeConsumedWhitespace() throws {
        let cases: [(Bool, Bool, Bool, [Int], [Range<Int>?])] = [
            (false, false, false, [0, 0, 1, 2, 0, 0], [0..<1, 1..<2, 2..<5, 5..<10, 10..<11, 11..<12]),
            (false, false, true, [0, 0, 1, 2, 0, 0], [0..<1, 1..<2, 2..<5, 5..<10, 10..<11, 11..<12]),
            (false, true, false, [1, 2, 0, 0], [0..<7, 7..<10, 10..<11, 11..<12]),
            (false, true, true, [1, 2, 0, 0], [0..<7, 7..<10, 10..<11, 11..<12]),
            (true, false, false, [0, 0, 1, 2, 0, 0], [0..<1, 1..<2, 2..<5, 5..<10, 10..<11, 11..<12]),
            (true, false, true, [0, 0, 1, 2, 0, 0], [0..<1, 1..<2, 2..<5, 5..<10, 10..<11, 11..<12]),
            (true, true, false, [1, 2, 0, 0], [0..<7, 7..<10, 10..<11, 11..<12]),
            (true, true, true, [1, 2, 0, 0], [0..<7, 7..<10, 10..<11, 11..<12]),
        ]
        for (singleWord, strip, normalized, ids, offsets) in cases {
            let tokenizer = try PreTrainedTokenizer(
                tokenizerConfig: [:],
                tokenizerData: [
                    "model": ["type": "BPE", "vocab": ["Ġ": 0], "merges": []],
                    "normalizer": ["type": "Lowercase"],
                    "pre_tokenizer": ["type": "ByteLevel", "add_prefix_space": true],
                    "added_tokens": [
                        [
                            "id": 1, "content": "[X]", "single_word": Config(singleWord), "lstrip": Config(strip),
                            "rstrip": Config(strip), "normalized": Config(normalized), "special": false,
                        ],
                        [
                            "id": 2, "content": "[Y]", "single_word": false, "lstrip": true,
                            "rstrip": false, "normalized": Config(normalized), "special": false,
                        ],
                    ],
                ])
            try check(tokenizer, text: "  [X]  [Y]  ", ids: ids, offsets: offsets, utf16: offsets)
        }
    }

    @Test func byteFallbackOffsets() throws {
        for unigram in [false, true] {
            let model: Config
            if unigram {
                var vocabulary: [Config] = [["[UNK]", 0.0], ["a", -1.0], ["b", -1.0], ["ab", -0.1]]
                vocabulary += (0..<256).map { [Config(String(format: "<0x%02X>", $0)), -10.0] }
                model = ["type": "Unigram", "unk_id": 0, "vocab": Config(vocabulary), "byte_fallback": true]
            } else {
                var vocabulary: [String: Config] = ["[UNK]": 0, "a": 1, "b": 2, "ab": 3]
                for byte in 0..<256 { vocabulary[String(format: "<0x%02X>", byte)] = Config(byte + 4) }
                model = [
                    "type": "BPE", "unk_token": "[UNK]", "vocab": Config(vocabulary),
                    "merges": [["a", "b"]], "byte_fallback": true, "fuse_unk": true,
                ]
            }
            let tokenizer = try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: ["model": model])
            // Unigram's fused unknown span belongs to every emitted byte token.
            try check(
                tokenizer, text: "A😀B", ids: [69, 244, 163, 156, 132, 70],
                offsets: unigram ? Array(repeating: 0..<6, count: 6) : [0..<1, 1..<5, 1..<5, 1..<5, 1..<5, 5..<6],
                utf16: unigram ? Array(repeating: 0..<4, count: 6) : [0..<1, 1..<3, 1..<3, 1..<3, 1..<3, 3..<4])
        }
    }

    @Test func insertedSpecialTokensHaveNoSourceRange() throws {
        let template: Config = [
            "type": "TemplateProcessing",
            "single": [
                ["SpecialToken": ["id": "[CLS]"]], ["Sequence": ["id": "A"]], ["SpecialToken": ["id": "[SEP]"]],
            ],
            "pair": [],
        ]
        let processors: [Config] = [
            ["type": "BertProcessing", "cls": ["[CLS]", 0], "sep": ["[SEP]", 1]],
            ["type": "RobertaProcessing", "cls": ["[CLS]", 0], "sep": ["[SEP]", 1], "trim_offsets": false],
            template,
            ["type": "Sequence", "processors": [["type": "ByteLevel", "trim_offsets": false], template]],
        ]
        for processor in processors {
            let tokenizer = try PreTrainedTokenizer(
                tokenizerConfig: [:],
                tokenizerData: [
                    "model": ["type": "BPE", "vocab": ["[CLS]": 0, "[SEP]": 1, "a": 2], "merges": []],
                    "pre_tokenizer": ["type": "WhitespaceSplit"], "post_processor": processor,
                    "added_tokens": [["id": 0, "content": "[CLS]", "special": true, "normalized": false]],
                ])
            try check(
                tokenizer, text: "[CLS] a", special: true, ids: [0, 0, 2, 1],
                offsets: [nil, 0..<5, 6..<7, nil], utf16: [nil, 0..<5, 6..<7, nil])
            try check(tokenizer, text: "[CLS] a", ids: [0, 2], offsets: [0..<5, 6..<7], utf16: [0..<5, 6..<7])
            try check(tokenizer, text: "", special: true, ids: [0, 1], offsets: [nil, nil], utf16: [nil, nil])
        }
    }

    @Test func graphemeRanges() {
        let result = TokenEncoding(
            text: "e\u{301}👩🏽‍💻",
            tokens: [
                AlignedToken(id: 1, offset: 0..<1), AlignedToken(id: 2, offset: 3..<7),
                AlignedToken(id: 3, offset: nil), AlignedToken(id: 4, offset: 0..<0),
            ])
        #expect(
            result.utf16Ranges() == [
                NSRange(location: 0, length: 1), NSRange(location: 2, length: 2), nil, NSRange(location: 0, length: 0),
            ])
        #expect(
            result.utf16Ranges(expandingToGraphemeClusters: true) == [
                NSRange(location: 0, length: 2), NSRange(location: 2, length: 7), nil, NSRange(location: 0, length: 0),
            ])
    }

    @Test func existentialConvenienceDispatch() throws {
        let tokenizer: any Tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: [:],
            tokenizerData: [
                "model": ["type": "BPE", "vocab": ["a": 0], "merges": []]
            ])
        #expect(try tokenizer.encode(text: "a", withOffsets: true).ids == [0])
        let idsOnly = try tokenizer.encode(text: "a", withOffsets: false)
        #expect(idsOnly.ids == [0])
        #expect(idsOnly.offsets == nil)
        #expect(idsOnly.utf16Ranges() == nil)
    }

    @Test func customEncodingValidation() throws {
        #expect(throws: TokenizerError.self) { try TokenEncoding(text: "😀", ids: [1], offsets: [0..<1]) }
        #expect(throws: TokenizerError.self) { try TokenEncoding(text: "a", ids: [1], offsets: []) }
        #expect(throws: TokenizerError.self) { try TokenEncoding(text: "a", ids: [1], offsets: [-1..<0]) }
        #expect(throws: TokenizerError.self) { try TokenEncoding(text: "a", ids: [1], offsets: [0..<2]) }
        #expect(
            try TokenEncoding(text: "😀", ids: [1], offsets: [0..<4]).utf16Ranges() == [NSRange(location: 0, length: 2)])
    }

    @Test func legacyWordPiece() throws {
        let tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: ["tokenizer_class": "BertTokenizer"],
            tokenizerData: [
                "model": ["vocab": ["[UNK]": 0, "[CLS]": 1, "cafe": 2, "hello": 3, "中": 4, "!": 5, "a": 6, "##b": 7]]
            ])
        for text in ["Café HELLO", "Cafe\u{301} HELLO", "中Cafe\u{301}!", "[CLS] HELLO", "ab", "a\nb", "a\u{a0}b"] {
            #expect(try tokenizer.encode(text: text, withOffsets: true).ids == tokenizer.encode(text: text))
        }
        #expect(
            try tokenizer.encode(text: "Café HELLO", withOffsets: true).utf16Ranges() == [
                NSRange(location: 0, length: 4), NSRange(location: 5, length: 5),
            ])
    }

    @Test func concurrentOffsets() async throws {
        let tokenizer = try await HubFixtures.tokenizer(for: "Qwen/Qwen3-0.6B")
        let text = String(repeating: "A😀B Café cafe\u{301}\n", count: 32)
        let expected = try tokenizer.encode(text: text, withOffsets: true)
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<32 {
                group.addTask {
                    let result = try tokenizer.encode(text: text, withOffsets: true)
                    #expect(result.ids == expected.ids)
                    #expect(result.offsets == expected.offsets)
                    #expect(tokenizer.encode(text: text) == expected.ids)
                }
            }
            try await group.waitForAll()
        }
    }
    @Test func tracedNormalizationMatchesIDsPipelineUnicode() throws {
        let normalizers: [any ByteNormalizer] = [
            NFDNormalizer(config: [:]), NFCNormalizer(config: [:]), NFKDNormalizer(config: [:]),
            NFKCNormalizer(config: [:]), LowercaseNormalizer(config: [:]), BertNormalizer(config: [:]),
        ]
        var seed: UInt64 = 20260909
        for _ in 0..<160 {
            var text = ""
            for _ in 0..<24 {
                seed = seed &* 6364136223846793005 &+ 1
                let value = UInt32((seed >> 32) % 0x110000)
                if let scalar = Unicode.Scalar(value) { text.unicodeScalars.append(scalar) }
                text += "e\u{301}\u{327} "
            }
            for normalizer in normalizers {
                let aligned = try AlignedText(text).normalized(by: normalizer)
                #expect(aligned.bytes.elementsEqual(normalizer.normalize(text: text).utf8), "\(type(of: normalizer))")
                #expect(
                    aligned.units.allSatisfy { $0.origin.lowerBound >= 0 && $0.origin.upperBound <= text.utf8.count })
            }
        }
    }

    @Test func graphemeIndexMatchesFoundation() throws {
        for text in ["e\u{301}👩🏽‍💻🇮🇹", "हिन्दी 한국어 العربية", "\u{301}a\u{327}\u{301}"] {
            var boundaries = [0]
            for scalar in text.unicodeScalars { boundaries.append(boundaries.last! + scalar.utf8.count) }
            var offsets: [Range<Int>?] = [nil]
            for start in boundaries {
                for end in boundaries where end >= start { offsets.append(start..<end) }
            }
            let encoding = try TokenEncoding(
                text: text, ids: Array(repeating: 1, count: offsets.count), offsets: offsets)
            let raw = try #require(encoding.utf16Ranges())
            let expected = raw.map {
                $0.map { $0.length == 0 ? $0 : (text as NSString).rangeOfComposedCharacterSequences(for: $0) }
            }
            #expect(encoding.utf16Ranges(expandingToGraphemeClusters: true) == expected)
        }
    }

    @Test func crlfIsOneSwiftGrapheme() throws {
        let result = try TokenEncoding(text: "a\r\nb", ids: [1, 2], offsets: [1..<2, 2..<3])
        #expect(
            result.utf16Ranges(expandingToGraphemeClusters: true) == [
                NSRange(location: 1, length: 2), NSRange(location: 1, length: 2),
            ])
    }

    @Test func droppedSymbolsDoNotShiftLaterSourceRanges() throws {
        let tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: [:],
            tokenizerData: [
                "model": ["type": "BPE", "vocab": ["b": 0], "merges": []]
            ])
        let result = try tokenizer.encode(text: "😀ab", withOffsets: true)
        #expect(result.ids == [0])
        #expect(result.offsets == [5..<6])
        #expect(result.utf16Ranges() == [NSRange(location: 3, length: 1)])
    }

}
