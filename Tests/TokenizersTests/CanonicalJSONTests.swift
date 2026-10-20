import Dispatch
// Canonical tokenizer.json 2.0 behavior from huggingface/tokenizers @ fb49a292
// (Apache-2.0). Metaspace cases include the upstream normalizer's reference table.
import Foundation
import Testing

@testable import Tokenizers

@Suite("Canonical tokenizer.json 2.0")
struct CanonicalJSONTests {
    private func tokenizer(_ data: Config, packed: Bool = true) throws -> PreTrainedTokenizer {
        let json = try JSONEncoder().encode(data)
        let parsed = try packed ? Config(tokenizerJSON: json) : JSONDecoder().decode(Config.self, from: json)
        return try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: parsed)
    }

    private func byteLevelConfig() -> Config {
        var vocab: [String: Config] = [:]
        for byte in 0..<256 { vocab[byteEncoderTable[byte]] = Config(byte) }
        vocab["Ġa"] = 256
        vocab["<bos>"] = 257
        vocab["<sep>"] = 258
        vocab["<eos>"] = 259
        return [
            "version": "2.0",
            "model": ["type": "BPE", "byte_level": true, "vocab": Config(vocab), "merges": [["Ġ", "a"]]],
            "decoder": ["type": "ByteLevel"],
        ]
    }

    @Test("The model's byte-level flag serves IDs, token strings, offsets and decoding", arguments: [false, true])
    func byteLevel(packed: Bool) throws {
        let canonical = byteLevelConfig()
        var legacy = canonical.dictionary(or: [:])
        legacy["version"] = "1.0"
        legacy["pre_tokenizer"] = ["type": "ByteLevel", "add_prefix_space": false, "use_regex": false]
        let current = try tokenizer(canonical, packed: packed)
        let reference = try tokenizer(Config(legacy), packed: packed)
        for (text, expected): (String, [Int]) in [
            (" a", [256]), ("é", [195, 169]), ("Ġa", [196, 160, 97]),
            ("a😀", [97, 240, 159, 152, 128]), ("\t\n a", [9, 10, 256]), ("", []),
        ] {
            for _ in 0..<3 { #expect(current.encode(text: text, addSpecialTokens: false) == expected) }
            #expect(
                current.tokenize(text: text).map { Array($0.utf8) }
                    == reference.tokenize(text: text).map { Array($0.utf8) })
            #expect(current.decode(tokens: expected) == text)
            let aligned = try current.encode(text: text, addSpecialTokens: false, withOffsets: true)
            let old = try reference.encode(text: text, addSpecialTokens: false, withOffsets: true)
            #expect(aligned.ids == expected)
            #expect(aligned.offsets == old.offsets)
        }
    }

    @Test("Byte-level mapping follows the declared split instead of materializing alphabet text")
    func splitAndNormalize() throws {
        var config = byteLevelConfig().dictionary(or: [:])
        config["normalizer"] = ["type": "Lowercase"]
        config["pre_tokenizer"] = [
            "type": "Sequence",
            "pretokenizers": [
                ["type": "Split", "pattern": ["String": "|"], "behavior": "Removed", "invert": false],
                ["type": "FixedLength", "length": 2],
            ],
        ]
        config["added_tokens"] = [["id": 260, "content": "<X>", "special": true, "normalized": false]]
        let tokenizer = try tokenizer(Config(config))
        let expected = [256, 195, 169, 260, 97]
        #expect(tokenizer.encode(text: " A|É<X>A", addSpecialTokens: false) == expected)
        let aligned = try tokenizer.encode(text: " A|É<X>A", addSpecialTokens: false, withOffsets: true)
        #expect(aligned.ids == expected)
        #expect(aligned.offsets == [0..<2, 3..<5, 3..<5, 5..<8, 8..<9])
        let stages = try PreTokenizerFactory.stages(
            PreTokenizerFactory.fromConfig(config: config["pre_tokenizer"]), model: Config(config).model)
        #expect(stages.map(\.shape) == [.split, .split, .byteLevel])
    }

    @Test("An explicit false byte-level flag preserves scalar BPE")
    func scalarModel() throws {
        var config = byteLevelConfig().dictionary(or: [:])
        var model = try #require(config["model"]?.dictionary())
        model["byte_level"] = false
        config["model"] = Config(model)
        #expect(try tokenizer(Config(config)).encode(text: "é", addSpecialTokens: false) == [233])
    }

    private var template: Config {
        [
            "type": "TemplateProcessing",
            "single": [["ids": [257, 258]], ["seq": "A"], ["ids": [259]]],
            "pair": [
                ["ids": [257], "type_id": 5], ["seq": "A", "type_id": 2],
                ["ids": [258], "type_id": 7], ["seq": "B", "type_id": 3], ["ids": [259], "type_id": 8],
            ],
        ]
    }

    @Test("Flat templates compile authoritative IDs and pair segments", arguments: [false, true])
    func flatTemplates(packed: Bool) throws {
        var config = byteLevelConfig().dictionary(or: [:])
        config["post_processor"] = template
        let tokenizer = try tokenizer(Config(config), packed: packed)
        #expect(tokenizer.encode(text: " a") == [257, 258, 256, 259])
        #expect(tokenizer.encode(text: " a", addSpecialTokens: false) == [256])
        #expect(tokenizer.postProcess(["Ġa"]) == ["<bos>", "<sep>", "Ġa", "<eos>"])
        let offsets = try tokenizer.encode(text: " a", withOffsets: true)
        #expect(offsets.ids == [257, 258, 256, 259])
        #expect(offsets.offsets == [nil, nil, 0..<2, nil])
        #expect(try tokenizer.encode(text: " aa", maxLength: 4).ids == [257, 258, 256, 259])
        #expect(try tokenizer.encode(text: " aa", maxLength: 4, withOffsets: true).ids == [257, 258, 256, 259])
        let pair = try tokenizer.encode(text: " a", textPair: "é")
        #expect(pair.ids == [257, 256, 258, 195, 169, 259])
        #expect(pair.typeIds == [5, 2, 7, 3, 3, 8])
        let plainPair = try tokenizer.encode(text: " a", textPair: "é", addSpecialTokens: false)
        #expect(plainPair.ids == [256, 195, 169])
        #expect(plainPair.typeIds == [2, 3, 3])
    }

    @Test("Nested processors bind flat spellings and retain direct IDs")
    func nestedTemplates() throws {
        var config = byteLevelConfig().dictionary(or: [:])
        config["post_processor"] = [
            "type": "Sequence", "processors": [["type": "ByteLevel", "trim_offsets": false], template],
        ]
        let tokenizer = try tokenizer(Config(config))
        #expect(tokenizer.encode(text: " a") == [257, 258, 256, 259])
        #expect(tokenizer.postProcess(["Ġa"]) == ["<bos>", "<sep>", "Ġa", "<eos>"])
        #expect(try tokenizer.encode(text: " a", withOffsets: true).ids == [257, 258, 256, 259])
    }

    @Test("Flat IDs do not require a vocabulary match; templates can repeat a sequence")
    func authoritativeIds() throws {
        let processor = try TemplateProcessing(config: [
            "single": [["ids": [9000]], ["seq": "A"], ["ids": []], ["seq": "A"]],
            "pair": [["seq": "A"], ["seq": "B", "type_id": 1]],
        ])
        var ids = [1, 2]
        processor.bound { _ in 999 }.postProcess(ids: &ids, addSpecialTokens: true)
        #expect(ids == [9000, 1, 2, 1, 2])
    }

    @Test("Canonical normalization and templates work across all model families", arguments: [false, true])
    func modelFamilies(packed: Bool) throws {
        let models: [(Config, [Int], [Range<Int>])] = [
            (
                ["type": "BPE", "byte_level": false, "vocab": ["a": 0, "b": 1, "ab": 2], "merges": [["a", "b"]]],
                [2, 2], [0..<2, 3..<5]
            ),
            (
                ["type": "Unigram", "unk_id": 0, "vocab": [["?", 0.0], ["a", -1.0], ["b", -1.0], ["ab", -1.1]]],
                [3, 3], [0..<2, 3..<5]
            ),
            (
                [
                    "type": "WordPiece", "unk_token": "?", "continuing_subword_prefix": "##",
                    "max_input_chars_per_word": 100, "vocab": ["?": 0, "a": 1, "##b": 2],
                ],
                [1, 2, 1, 2], [0..<1, 1..<2, 3..<4, 4..<5]
            ),
            (
                ["type": "WordLevel", "unk_token": "?", "vocab": ["?": 0, "ab": 1]],
                [1, 1], [0..<2, 3..<5]
            ),
        ]
        for (model, plain, ranges) in models {
            let data: Config = [
                "version": "2.0", "model": model,
                "normalizer": CanonicalMetaspaceTests.config("never"),
                "pre_tokenizer": ["type": "Split", "pattern": ["String": "▁"], "behavior": "Removed", "invert": false],
                "post_processor": [
                    "type": "TemplateProcessing",
                    "single": [["ids": [99]], ["seq": "A"]], "pair": [["seq": "A"], ["seq": "B", "type_id": 1]],
                ],
            ]
            let tokenizer = try tokenizer(data, packed: packed)
            #expect(tokenizer.encode(text: "ab ab") == [99] + plain)
            let aligned = try tokenizer.encode(text: "ab ab", withOffsets: true)
            #expect(aligned.ids == [99] + plain)
            #expect(aligned.offsets == [nil] + ranges.map { Optional($0) })
        }
    }

    @Test("Canonical byte models and positional normalizers remain safe when shared")
    func concurrentEncoding() throws {
        var config = byteLevelConfig().dictionary(or: [:])
        config["normalizer"] = CanonicalMetaspaceTests.config("first", drop: true, replacement: " ")
        config["post_processor"] = template
        let tokenizer = try tokenizer(Config(config))
        let texts = ["a b", " a b", "é 😀", "▁a\t中文", ""]
        let expected = try texts.map { try tokenizer.encode(text: $0, withOffsets: true) }
        DispatchQueue.concurrentPerform(iterations: 16) { worker in
            for iteration in 0..<40 {
                let index = (worker + iteration) % texts.count
                let text = texts[index]
                #expect(tokenizer.encode(text: text) == expected[index].ids)
                do {
                    let actual = try tokenizer.encode(text: text, withOffsets: true)
                    #expect(actual.ids == expected[index].ids)
                    #expect(actual.offsets == expected[index].offsets)
                } catch { Issue.record(error) }
            }
        }
    }

    @Test("Malformed canonical fields throw instead of dropping input")
    func malformedFields() throws {
        let pieces: [Config] = [
            ["seq": "C"], ["seq": false], ["seq": "A", "ids": []], ["ids": "1"],
            ["ids": [-1]], ["ids": [4294967296]], ["ids": [true]],
            ["seq": "A", "type_id": -1], ["seq": "A", "type_id": false],
            ["seq": "A", "type_id": 4294967296], ["seq": "B"],
        ]
        for piece in pieces {
            #expect(throws: TokenizerError.self) {
                _ = try TemplateProcessing(config: ["single": Config([piece]), "pair": [["seq": "A"], ["seq": "B"]]])
            }
        }
        for flag: Config in [1, "true", []] {
            #expect(throws: TokenizerError.self) {
                _ = try PreTokenizerFactory.stages(nil, model: ["type": "BPE", "byte_level": flag])
            }
        }
        #expect(throws: TokenizerError.self) {
            _ = try PreTokenizerFactory.stages(nil, model: ["type": "WordLevel", "byte_level": true])
        }
        for bad: Config in [
            ["replacement": "", "prepend": "first", "drop_whitespace": false],
            ["replacement": "ab", "prepend": "first", "drop_whitespace": false],
            ["replacement": "▁", "prepend": true, "drop_whitespace": false],
            ["replacement": "▁", "prepend": "sometimes", "drop_whitespace": false],
            ["replacement": "▁", "prepend": "first", "drop_whitespace": "false"],
        ] {
            #expect(throws: TokenizerError.self) { _ = try MetaspaceNormalizer(config: bad) }
        }
    }

    @Test("Canonical JSON loads from a folder without tokenizer_config.json")
    func folderLoading() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        var config = byteLevelConfig().dictionary(or: [:])
        config["post_processor"] = template
        try JSONEncoder().encode(Config(config)).write(to: folder.appendingPathComponent("tokenizer.json"))
        #expect(try AutoTokenizer.load(from: folder).encode(text: " a") == [257, 258, 256, 259])
    }
}

@Suite("Canonical Metaspace normalization")
struct CanonicalMetaspaceTests {
    struct Case: Sendable {
        let scheme: String
        let drop: Bool
        let offset: Int
        let text: String
        let expected: String
    }
    static let cases = [
        Case(scheme: "always", drop: false, offset: 0, text: "aa bb cc", expected: "▁aa▁bb▁cc"),
        Case(scheme: "always", drop: false, offset: 0, text: " aa bb", expected: "▁aa▁bb"),
        Case(scheme: "always", drop: false, offset: 0, text: "aa\tbb  cc", expected: "▁aa\tbb▁▁cc"),
        Case(scheme: "always", drop: false, offset: 7, text: "aa bb cc", expected: "▁aa▁bb▁cc"),
        Case(scheme: "never", drop: false, offset: 0, text: "aa bb cc", expected: "aa▁bb▁cc"),
        Case(scheme: "never", drop: false, offset: 0, text: " aa bb", expected: "▁aa▁bb"),
        Case(scheme: "first", drop: false, offset: 0, text: "aa bb cc", expected: "▁aa▁bb▁cc"),
        Case(scheme: "first", drop: false, offset: 7, text: "aa bb cc", expected: "aa▁bb▁cc"),
        Case(scheme: "first", drop: false, offset: 0, text: " aa bb", expected: "▁aa▁bb"),
        Case(scheme: "always", drop: true, offset: 0, text: "aa bb cc", expected: "▁aa▁bb▁cc"),
        Case(scheme: "always", drop: true, offset: 0, text: "aa\tbb  cc", expected: "▁aa▁bb▁cc"),
        Case(scheme: "never", drop: true, offset: 0, text: "aa bb cc", expected: "aabbcc"),
        Case(scheme: "first", drop: true, offset: 0, text: "aa bb cc", expected: "▁aabbcc"),
        Case(scheme: "first", drop: true, offset: 7, text: "aa bb cc", expected: "aabbcc"),
        Case(scheme: "first", drop: true, offset: 0, text: " aa bb", expected: "aabb"),
        Case(scheme: "first", drop: true, offset: 0, text: "\u{85}a\u{A0}b\u{3000}", expected: "ab"),
        Case(scheme: "always", drop: true, offset: 0, text: "▁aa\t▁bb", expected: "▁aa▁bb"),
        Case(scheme: "never", drop: true, offset: 0, text: "a\u{200B}b", expected: "a\u{200B}b"),
        Case(scheme: "always", drop: true, offset: 0, text: " \t\n", expected: ""),
        Case(scheme: "first", drop: false, offset: 0, text: "", expected: ""),
    ]

    static func config(_ scheme: String, drop: Bool = false, replacement: String = "▁") -> Config {
        [
            "type": "MetaspaceNormalizer", "replacement": Config(replacement),
            "prepend": Config(scheme), "drop_whitespace": Config(drop),
        ]
    }

    @Test("Matches the pinned upstream table and preserves source traces", arguments: cases)
    func reference(test: Case) throws {
        let normalizer = try #require(
            try NormalizerFactory.fromConfig(config: Self.config(test.scheme, drop: test.drop)))
        var input = test.text
        var output: [UInt8] = []
        input.withUTF8 { normalizer.normalize($0, at: test.offset, into: &output, scratch: ScratchBuffers()) }
        #expect(output == Array(test.expected.utf8))
        let aligned = try AlignedText(test.text).normalized(by: normalizer, at: test.offset)
        #expect(Array(aligned.bytes) == output)
        #expect(aligned.origins == nil || aligned.origins?.count == output.count)
    }

    @Test(
        "Bulk rewrites append dense markers without overwriting existing output", arguments: ["▁", "😀", "é", "$", " "])
    func appendingRewrites(marker: String) throws {
        let count = 4096
        var cases: [(String, Bool, Int, String, String)] = [
            ("always", false, 0, String(repeating: " ", count: count), String(repeating: marker, count: count)),
            (
                "always", false, 0, String(repeating: "a ", count: count),
                marker + String(repeating: "a" + marker, count: count)
            ),
            (
                "always", true, 0, String(repeating: "a\t\u{3000}", count: count),
                String(repeating: marker + "a", count: count)
            ),
            ("always", true, 0, String(repeating: "a ", count: count), String(repeating: marker + "a", count: count)),
            ("first", false, 7, "é 😀", "é" + marker + "😀"),
            ("first", true, 0, " a\tb", "ab"),
            ("never", true, 0, "\t\n\u{3000}", ""),
            ("always", true, 0, marker + "a " + marker + "b", marker + "a" + marker + "b"),
            ("always", false, 0, "", ""),
        ]
        for length in [15, 16, 17, 31, 32, 33] {
            let word = String(repeating: "a", count: length)
            cases.append(("always", true, 0, word + " b", marker + word + marker + "b"))
            cases.append(("always", true, 0, word + "\u{3000}b", marker + word + marker + "b"))
        }
        for (scheme, drop, offset, text, expected) in cases {
            let config = Self.config(scheme, drop: drop, replacement: marker)
            let normalizers: [any ByteNormalizer] = [
                try MetaspaceNormalizer(config: config),
                try NormalizerSequence(config: ["normalizers": [["type": "Lowercase"], config]]),
            ]
            for normalizer in normalizers {
                let prefix: [UInt8] = [0xDE, 0xAD]
                var output = prefix
                var input = text
                let scratch = ScratchBuffers()
                input.withUTF8 { bytes in
                    normalizer.normalize(bytes, at: offset, into: &output, scratch: scratch)
                    normalizer.normalize(bytes, at: offset, into: &output, scratch: scratch)
                }
                #expect(output == prefix + Array((expected + expected).utf8))
            }
        }
    }

    @Test("First sees the section's original offset through sequences and added tokens", arguments: [false, true])
    func sectionPositions(packed: Bool) throws {
        let config: Config = [
            "version": "2.0",
            "model": ["type": "WordLevel", "unk_token": "?", "vocab": ["?": 0, "▁a": 1, "a": 2, "▁b": 3, "b": 4]],
            "added_tokens": [["id": 5, "content": "<s>", "special": true, "normalized": false]],
            "normalizer": [
                "type": "Sequence",
                "normalizers": [
                    ["type": "Strip", "strip_left": true, "strip_right": true], ["type": "Lowercase"],
                    Self.config("first"),
                ],
            ],
            "pre_tokenizer": [
                "type": "Split", "pattern": ["String": "▁"], "behavior": "MergedWithNext", "invert": false,
            ],
        ]
        let json = try JSONEncoder().encode(config)
        let parsed = try packed ? Config(tokenizerJSON: json) : JSONDecoder().decode(Config.self, from: json)
        let tokenizer = try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: parsed)
        for (text, ids): (String, [Int]) in [(" A B ", [1, 3]), ("A<s>B", [1, 5, 4]), ("<s>A B", [5, 2, 3])] {
            #expect(tokenizer.encode(text: text, addSpecialTokens: false) == ids)
            #expect(try tokenizer.encode(text: text, addSpecialTokens: false, withOffsets: true).ids == ids)
            #expect(tokenizer.tokenize(text: text).compactMap(tokenizer.convertTokenToId) == ids)
        }
        #expect(try tokenizer.encode(text: "A<s>B", textPair: "A", addSpecialTokens: false).ids == [1, 5, 4, 1])
    }

    @Test("Canonical normalization precedes normalized added-token matching")
    func normalizedTokens() throws {
        let config: Config = [
            "version": "2.0",
            "model": ["type": "WordLevel", "unk_token": "?", "vocab": ["?": 0, "▁a": 1, "▁b": 2]],
            "added_tokens": [["id": 3, "content": "X", "special": false, "normalized": true]],
            "normalizer": ["type": "Sequence", "normalizers": [["type": "Lowercase"], Self.config("always")]],
            "pre_tokenizer": [
                "type": "Split", "pattern": ["String": "▁"], "behavior": "MergedWithNext", "invert": false,
            ],
        ]
        let tokenizer = try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: config)
        #expect(tokenizer.encode(text: "A X B", addSpecialTokens: false) == [1, 3, 2])
        #expect(try tokenizer.encode(text: "A X B", addSpecialTokens: false, withOffsets: true).ids == [1, 3, 2])
    }

    @Test("Whitespace removal and custom markers have exact byte offsets")
    func droppedWhitespaceOffsets() throws {
        for marker in ["▁", "😀", "$", " "] {
            let data: Config = [
                "version": "2.0",
                "model": [
                    "type": "WordLevel", "unk_token": "?",
                    "vocab": Config([
                        "?": Config(0), marker + "a": Config(1), marker + "b": Config(2),
                    ]),
                ],
                "normalizer": Self.config("always", drop: true, replacement: marker),
                "pre_tokenizer": [
                    "type": "Split", "pattern": ["String": Config(marker)],
                    "behavior": "MergedWithNext", "invert": false,
                ],
            ]
            let tokenizer = try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: data)
            let aligned = try tokenizer.encode(text: " a\tb\u{3000}", addSpecialTokens: false, withOffsets: true)
            #expect(aligned.ids == [1, 2])
            #expect(aligned.offsets == [1..<2, 3..<4])
            #expect(tokenizer.encode(text: " a\tb\u{3000}", addSpecialTokens: false) == aligned.ids)
        }
    }
}
