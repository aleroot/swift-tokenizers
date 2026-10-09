import Foundation
import Testing

@testable import Tokenizers

@Suite("Piece decode table parity")
struct PieceDecodeTableTests {
    private let entries: [(String, Int)] = [
        ("", 0), ("hello", 1), ("##s", 2), ("▁a▁b", 3), (" .", 4),
        ("<0xC3>", 5), ("<0xA9>", 6), ("<0xFF>", 7), ("<0x20>", 8),
        (" ' ", 9), ("do not", 10), ("é", 12), ("e\u{301}", 13),
    ]

    @Test("Decoded slices match the decoder for empty, sparse, skipped and fallback ids")
    func parity() throws {
        let configs: [Config] = [
            ["type": "WordPiece", "prefix": "##", "cleanup": false],
            ["type": "WordPiece", "prefix": "##", "cleanup": true],
            ["type": "WordPiece", "prefix": "", "cleanup": true],
            ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "always"],
            ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "never"],
            [
                "type": "Sequence",
                "decoders": [
                    ["type": "Replace", "pattern": ["String": "▁"], "content": " "],
                    ["type": "ByteFallback"], ["type": "Fuse"],
                    ["type": "Strip", "content": " ", "start": 2, "stop": 0],
                ],
            ],
            ["type": "Sequence", "decoders": [["type": "ByteFallback"]]],
            [
                "type": "Sequence",
                "decoders": [
                    ["type": "Replace", "pattern": ["String": "hello"], "content": ""]
                ],
            ],
        ]
        let vocabulary = try Vocabulary(entries: entries)
        let spellings = [12: "▁normalized"]
        var sequences = [[Int](), [0], [1, 0], [0, 1], [1, 0, 2], [5, 0, 6], [5, 6, 7], [8, 8, 1]]
        // Exhaustive three-token combinations include a hole, invalid ids, special tokens,
        // empty pieces between fallback bytes, and canonically equivalent byte spellings.
        let ids = [-1] + Array(0...13) + [Int.max]
        for a in ids {
            for b in ids {
                for c in ids { sequences.append([a, b, c]) }
            }
        }
        for config in configs {
            let decoder = try #require(try DecoderFactory.fromConfig(config: config))
            let shape = try #require(PieceDecodeShape.classify(decoder))
            let table = PieceDecodeTable(vocabulary: vocabulary, normalizedSpellings: spellings, shape: shape)
            for skipped: Set<Int> in [[], [0, 3, 7]] {
                for sequence in sequences {
                    let strings = sequence.compactMap { id -> String? in
                        skipped.contains(id) ? nil : spellings[id] ?? vocabulary.token(id)
                    }
                    let raw = decoder.decode(tokens: strings).joined()
                    for cleanup in [false, true] {
                        let expected = cleanup ? TokenizationCleanup.cleanUp(raw) : raw
                        let actual = table.decode(sequence, skipping: skipped, cleanup: cleanup)
                        #expect(
                            actual.utf8.elementsEqual(expected.utf8),
                            "\(config.type) ids=\(sequence), cleanup=\(cleanup)")
                        // Stop after a failure so a broken shape does not flood test output.
                        if !actual.utf8.elementsEqual(expected.utf8) { return }
                    }
                }
            }
        }
    }

    @Test("An entirely empty vocabulary storage decodes safely")
    func emptyStorage() throws {
        let vocabulary = try Vocabulary(entries: [("", 0)])
        for shape: PieceDecodeShape in [
            .spm(replacements: [], byteFallback: false, strip: nil),
            .spm(replacements: [], byteFallback: true, strip: nil),
            .metaspace(pattern: Array("▁".utf8), removeLeading: true),
            .wordPiece(prefix: Array("##".utf8), cleanup: false),
        ] {
            let table = PieceDecodeTable(vocabulary: vocabulary, normalizedSpellings: [:], shape: shape)
            #expect(table.decode([0], skipping: []).isEmpty)
        }
    }

    @Test("Strict UTF-8 fallback validation agrees with the decoder for arbitrary runs")
    func fallbackRuns() throws {
        let entries = (0..<256).map { (String(format: "<0x%02X>", $0), $0) } + [("", 256), ("text", 257)]
        let vocabulary = try Vocabulary(entries: entries)
        let decoder = ByteFallbackDecoder(config: [:])
        let table = PieceDecodeTable(
            vocabulary: vocabulary, normalizedSpellings: [:],
            shape: .spm(replacements: [], byteFallback: true, strip: nil))
        var rng: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next() -> UInt64 {
            rng ^= rng << 13
            rng ^= rng >> 7
            rng ^= rng << 17
            return rng
        }
        var runs = [
            [0xE3, 0x81, 0x82], [0xF0, 0x9F, 0x98, 0x80], [0xEF, 0xBF, 0xBD],
            [0xE0, 0x80, 0x80], [0xED, 0xA0, 0x80], [0xF4, 0x90, 0x80, 0x80],
            [0xC3, 0xA9, 0xFF], [0xC3, 256, 0xA9], [0xE2, 257, 0x82, 0xAC],
        ]
        for _ in 0..<10_000 {
            let length = Int(next() % 64)
            runs.append((0..<length).map { _ in Int(next() % 260) - 1 })
        }
        for ids in runs {
            let expected = decoder.decode(tokens: ids.compactMap { vocabulary.token($0) }).joined()
            #expect(table.decode(ids, skipping: []).utf8.elementsEqual(expected.utf8), "ids=\(ids)")
        }
    }

    @Test(
        "Concurrent first decode and independent streams agree across model types",
        arguments: [
            "BPE", "Unigram", "WordPiece", "WordLevel",
        ])
    func concurrentDecode(type: String) async throws {
        let model: Config
        if type == "Unigram" {
            model = [
                "type": "Unigram", "vocab": [["<unk>", 0.0], ["", -1.0], ["hello", -2.0], ["##s", -3.0]], "unk_id": 0,
            ]
        } else {
            model = [
                "type": Config(type), "vocab": ["<unk>": 0, "": 1, "hello": 2, "##s": 3], "unk_token": "<unk>",
                "merges": [],
            ]
        }
        let tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: ["clean_up_tokenization_spaces": true],
            tokenizerData: ["model": model, "decoder": ["type": "WordPiece", "prefix": "##", "cleanup": true]])
        let ids = Array(repeating: [2, 1, 3, -1, Int.max], count: 32).flatMap { $0 }
        let expected = tokenizer.cleanUp(
            text: tokenizer.decodeTokens(ids.compactMap { tokenizer.convertIdToToken($0) }).joined())
        try await withThrowingTaskGroup(of: Void.self) { group in
            for worker in 0..<32 {
                group.addTask {
                    if worker.isMultiple(of: 2) { tokenizer.prepareForDecoding() }
                    #expect(tokenizer.decode(tokens: ids).utf8.elementsEqual(expected.utf8))
                    var stream = tokenizer.makeStreamDecoder()
                    var result = ""
                    for id in ids { result += try stream.append(token: id) }
                    result += stream.finish()
                    #expect(result.utf8.elementsEqual(expected.utf8))
                }
            }
            try await group.waitForAll()
        }
    }
}
