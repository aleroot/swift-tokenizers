import Foundation
import Testing

@testable import Tokenizers

@Suite("Stateful streaming decoding")
struct TokenStreamDecoderTests {
    private func tokenizer(vocab: [String], decoder: Config, cleanup: Bool = false) throws -> any Tokenizer {
        try AutoTokenizer.from(
            tokenizerConfig: ["clean_up_tokenization_spaces": Config(cleanup)],
            tokenizerData: [
                "model": [
                    "type": "BPE",
                    "vocab": Config(
                        Dictionary(
                            uniqueKeysWithValues: vocab.enumerated().map {
                                (BinaryDistinctString($0.element), Config($0.offset))
                            })), "merges": [],
                ],
                "decoder": decoder,
            ])
    }

    private func check(_ tokenizer: any Tokenizer, ids: [Int], incremental: Bool = true) throws {
        let pretrained = try #require(tokenizer as? PreTrainedTokenizer)
        let strings = ids.compactMap { tokenizer.convertIdToToken($0) }
        let expected = pretrained.cleanUp(text: pretrained.decodeTokens(strings).joined())
        for prepared in [false, true] {
            if prepared { tokenizer.prepareForDecoding() }
            var stream = tokenizer.makeStreamDecoder()
            #expect(stream.isIncremental == incremental)
            var result = ""
            for id in ids {
                result += try stream.append(token: id)
                #expect(expected.utf8.starts(with: result.utf8))
            }
            result += stream.finish()
            #expect(result.utf8.elementsEqual(expected.utf8))
            #expect(stream.finish().isEmpty)
            #expect(throws: TokenizerError.self) { try stream.append(token: 0) }
            stream.reset()
            var repeated = ""
            for id in ids { repeated += try stream.append(token: id) }
            repeated += stream.finish()
            #expect(repeated.utf8.elementsEqual(expected.utf8))
            let batch = tokenizer.decode(tokens: ids)
            #expect(batch.utf8.elementsEqual(expected.utf8))
        }
    }

    @Test("Byte-level streams hold partial scalars and preserve genuine replacement characters")
    func byteLevel() throws {
        let words = (0..<256).map { [UInt8($0)].withUnsafeBufferPointer { ByteLevelAlphabet.encode($0) } }
        let tokenizer = try tokenizer(vocab: words, decoder: ["type": "ByteLevel"])
        for bytes in [
            [0xF0, 0x9F, 0x98, 0x80], [0xEF, 0xBF, 0xBD], [0xE0, 0x80, 0x80], [0xF4, 0x90],
            [0xED, 0xA0, 0x80], [0xE2, 0x82], [0x80, 0xC0, 0xAF], [0xC2, 65], [0xF0, 0x9F, 65], [],
        ] {
            try check(tokenizer, ids: bytes)
        }
        var random: UInt64 = 912_513
        for _ in 0..<256 {
            var ids: [Int] = []
            for _ in 0..<24 {
                random = random &* 6_364_136_223_846_793_005 &+ 1
                ids.append(Int(random >> 32) & 255)
            }
            try check(tokenizer, ids: ids)
        }
        var stream = tokenizer.makeStreamDecoder()
        #expect(try stream.append(token: 0xF0) == "")
        #expect(try stream.append(token: 0x9F) == "")
        #expect(try stream.append(token: 0x98) == "")
        #expect(try stream.append(token: 0x80) == "😀")
    }

    @Test("Cleanup across token boundaries is ordered and never retracts text")
    func cleanup() throws {
        let words = (0..<256).map { [UInt8($0)].withUnsafeBufferPointer { ByteLevelAlphabet.encode($0) } }
        let tokenizer = try tokenizer(vocab: words, decoder: ["type": "ByteLevel"], cleanup: true)
        for text in [
            "I ' m happy . You 're n't !", " ' 'm 's 've 're ? ! ,", "hello ", " . ? ! , ' n't 'm 's 've 're", "do not",
            "  ' s", " n ' t", "  ' ve", "  ' re", "   ' m", " ' .", " ' ,", " ' !", " ' ?",
        ] {
            try check(tokenizer, ids: text.utf8.map(Int.init))
        }
        let artifacts = Array("  ' s . n ' t ? ".utf8).map(Int.init)
        for malformed in [[0xFF], [0xC3], [0xE0, 0x80, 0x80], [0xF0, 0x9F], [0xED, 0xA0, 0x80]] {
            try check(tokenizer, ids: malformed + artifacts + malformed)
        }
    }

    @Test("WordPiece and Metaspace retain first-token semantics, including empty tokens")
    func tokenDecoders() throws {
        for decoder: Config in [
            ["type": "WordPiece", "prefix": "##", "cleanup": true],
            ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "always"],
        ] {
            let tokenizer = try tokenizer(
                vocab: ["", "Hello", "##s", "▁a▁b", "▁world", ".", "do not"], decoder: decoder)
            try check(tokenizer, ids: [-1, 0, 1, 2, 3, 4, 5, Int.max, 6])
            try check(tokenizer, ids: [2, 1, 5])
            try check(tokenizer, ids: [1, 0, 2, 0])
        }
    }

    @Test("Byte fallback validates whole runs before emitting, including late invalid bytes")
    func byteFallback() throws {
        let vocab = ["▁hello", "▁", "<0xF0>", "<0x9F>", "<0x98>", "<0x80>", "<0x41>", "<0xFF>", "end", "<0x20>"]
        let decoder: Config = [
            "type": "Sequence",
            "decoders": [
                ["type": "Replace", "pattern": ["String": "▁"], "content": " "],
                ["type": "ByteFallback"], ["type": "Fuse"], ["type": "Strip", "content": " ", "start": 1, "stop": 0],
            ],
        ]
        let tokenizer = try tokenizer(vocab: vocab, decoder: decoder)
        for ids in [[1, 0, 2, 3, 4, 5, 8], [6, 6, 7, 8], [2, 3], [9, 6, 8], [2, 3, 4, 5], [0, 7, 8]] {
            try check(tokenizer, ids: ids)
        }
    }

    @Test("Preparation preserves normalized added tokens, skipped specials, and active streams")
    func preparationAndSpecialTokens() throws {
        let data: Config = [
            "model": [
                "type": "BPE", "vocab": ["": 0, "<s>": 1, "▁hello": 2, "<0xC3>": 3, "<0xA9>": 4, "end": 5],
                "merges": [],
            ],
            "normalizer": ["type": "Prepend", "prepend": "▁"],
            "added_tokens": [["id": 1, "content": "<s>", "special": true, "normalized": true]],
            "decoder": [
                "type": "Sequence",
                "decoders": [
                    ["type": "Replace", "pattern": ["String": "▁"], "content": " "],
                    ["type": "ByteFallback"], ["type": "Fuse"],
                    ["type": "Strip", "content": " ", "start": 1, "stop": 0],
                ],
            ],
        ]
        let ids = [1, 2, 3, 1, 4, 0, 5]
        for skip in [false, true] {
            let tokenizer: any Tokenizer = try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: data)
            let expected = tokenizer.decode(tokens: ids, skipSpecialTokens: skip)
            var original = tokenizer.makeStreamDecoder(skipSpecialTokens: skip)
            var result = ""
            for id in ids.prefix(3) { result += try original.append(token: id) }
            tokenizer.prepareForDecoding()
            tokenizer.prepareForDecoding()
            for id in ids.dropFirst(3) { result += try original.append(token: id) }
            result += original.finish()
            #expect(result.utf8.elementsEqual(expected.utf8))
            #expect(tokenizer.decode(tokens: ids, skipSpecialTokens: skip).utf8.elementsEqual(expected.utf8))
            var prepared = tokenizer.makeStreamDecoder(skipSpecialTokens: skip)
            var preparedResult = ""
            for id in ids { preparedResult += try prepared.append(token: id) }
            preparedResult += prepared.finish()
            #expect(preparedResult.utf8.elementsEqual(expected.utf8))
        }
    }

    @Test("Global replacement pipelines buffer once instead of guessing a stable prefix")
    func buffered() throws {
        let tokenizer = try tokenizer(
            vocab: ["a", "b", "c"],
            decoder: [
                "type": "Sequence",
                "decoders": [
                    ["type": "Fuse"], ["type": "Replace", "pattern": ["Regex": "^a.*c$"], "content": "replacement"],
                ],
            ])
        try check(tokenizer, ids: [0, 1, 2], incremental: false)
    }

    @Test("Independent copies, skipped special IDs and invalid IDs do not disturb UTF-8 state")
    func copiesAndSkippedTokens() throws {
        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: [:],
            tokenizerData: [
                "model": ["type": "BPE", "vocab": ["Ã": 0, "©": 1, "x": 2], "merges": []],
                "decoder": ["type": "ByteLevel"], "added_tokens": [["id": 2, "content": "x", "special": true]],
            ])
        var first = tokenizer.makeStreamDecoder(skipSpecialTokens: true)
        #expect(try first.append(token: 0) == "")
        var second = first
        #expect(try first.append(token: 2) == "")
        #expect(try first.append(token: Int.max) == "")
        #expect(try first.append(token: 1) == "é")
        #expect(second.finish() == "�")
        #expect(first.finish() == "")
    }
}
