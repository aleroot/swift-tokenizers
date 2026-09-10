import Foundation
import Testing

@testable import Tokenizers

@Suite("Lampo tokenizer workloads")
struct LampoWorkloadTests {
    @Test("Direct byte-level decoding repairs UTF-8 only after joining tokens")
    func directByteLevelDecoding() throws {
        let raw: [[UInt8]] = [
            [], [0], [0x41], [0xF0], [0x9F], [0x98], [0x80], [0xC0, 0xAF],
            Array(String(repeating: "long token ", count: 128).utf8),
        ]
        let vocabulary = try Vocabulary(entries: raw.enumerated().map { index, bytes in
            (bytes.withUnsafeBufferPointer { ByteLevelAlphabet.encode($0) }, index * 2)
        } + [("literal😀", 20)])
        let table = ByteLevelDecodeTable(vocabulary: vocabulary)
        let decoder = ByteLevelDecoder(config: [:])
        let cases = [
            [], [0], [1], [-1], [Int.max], [20], [2], [6], [16],
            [6, 8, 10, 12], [6, 8, 10], [14, 2, 4], [0, 1, -1, Int.max],
            [16, 20, 16], [6, 20, 8, 10, 12],
        ]
        for ids in cases {
            for skipped: Set<Int> in [[], [6, 20], Set(ids)] {
                let strings = ids.filter { !skipped.contains($0) }.compactMap { vocabulary.token($0) }
                let expected = decoder.decode(tokens: strings).joined()
                #expect(table.decode(ids, skipping: skipped).utf8.elementsEqual(expected.utf8))
            }
        }
        // Exercise every one- and two-byte sequence, including invalid UTF-8 and split scalars.
        let bytesVocabulary = try Vocabulary(entries: (0..<256).map { byte in
            ([UInt8(byte)].withUnsafeBufferPointer { ByteLevelAlphabet.encode($0) }, byte)
        })
        let bytesTable = ByteLevelDecodeTable(vocabulary: bytesVocabulary)
        for first in 0..<256 {
            #expect(bytesTable.decode([first], skipping: []) == String(decoding: [UInt8(first)], as: UTF8.self))
            for second in 0..<256 {
                let expected = String(decoding: [UInt8(first), UInt8(second)], as: UTF8.self)
                #expect(bytesTable.decode([first, second], skipping: []) == expected)
            }
        }
    }

    @Test("Byte-level decoding distinguishes added token IDs from equivalent Unicode spellings")
    func addedTokenIdentity() throws {
        let json = Data(
            #"""
            {
                "model": {"type":"BPE", "vocab":{"Ġ":0,"Ġ":1,"x":2}, "merges":[]},
                "added_tokens":[{"id":1,"content":"Ġ","special":true}],
                "decoder":{"type":"ByteLevel"}
            }
            """#.utf8)
        let tokenizer = try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: Config(tokenizerJSON: json))
        #expect(tokenizer.decode(tokens: []).isEmpty)
        #expect(tokenizer.decode(tokens: [0, 2]).utf8.elementsEqual(" x".utf8))
        #expect(tokenizer.decode(tokens: [1, 2]).utf8.elementsEqual("G\u{0307}x".utf8))
        #expect(tokenizer.decode(tokens: [0, 1, 2], skipSpecialTokens: true).utf8.elementsEqual(" x".utf8))
    }

    @Test("Distinct added token IDs retain canonically equivalent spellings", arguments: [false, true])
    func equivalentAddedTokens(packed: Bool) throws {
        let json = Data(
            #"""
            {
                "model": {"type":"BPE", "vocab":{"Ġ":0,"Ġ":1,"x":2}, "merges":[]},
                "added_tokens":[{"id":0,"content":"Ġ","special":true},{"id":1,"content":"Ġ","special":true}],
                "decoder":{"type":"ByteLevel"}
            }
            """#.utf8)
        let data = try packed ? Config(tokenizerJSON: json) : Config(jsonData: json)
        let tokenizer = try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: data)
        #expect(tokenizer.encode(text: "ĠG\u{0307}x", addSpecialTokens: false) == [0, 1, 2])
        #expect(tokenizer.convertIdToToken(0)?.utf8.elementsEqual("Ġ".utf8) == true)
        #expect(tokenizer.convertIdToToken(1)?.utf8.elementsEqual("G\u{0307}".utf8) == true)
        #expect(tokenizer.decode(tokens: [1]).utf8.elementsEqual("G\u{0307}".utf8))
        #expect(tokenizer.decode(tokens: [0, 1, 2], skipSpecialTokens: true) == "x")
    }

    @Test("Scratch pooling reuses ordinary buffers and releases document-sized outliers")
    func scratchLifetime() {
        let pool = EncodeScratchPool()
        let scratch = pool.take()
        let pipeline = EncodePipeline(
            splitter: nil, normalizer: IdentityNormalizer(), normalizedSplitter: nil,
            preTokenizer: PreTokenizationRunner(stages: []), fuseUnknownId: nil)
        func run(_ count: Int) {
            [UInt8](repeating: 0x61, count: count).withUnsafeBufferPointer {
                pipeline.run($0, scratch: scratch, onToken: { _ in }, onPiece: { _, _ in })
            }
        }
        run(EncodeScratch.maximumReusableInputBytes)
        pool.recycle(scratch)
        #expect(pool.take() === scratch)
        run(EncodeScratch.maximumReusableInputBytes + 1)
        pool.recycle(scratch)
        #expect(pool.take() !== scratch)
    }

    @Test("A large document followed by small queries preserves token IDs")
    func documentThenQueries() throws {
        let data: Config = [
            "model": ["type": "WordPiece", "vocab": ["[UNK]": 0, "a": 1]],
            "pre_tokenizer": ["type": "WhitespaceSplit"],
        ]
        let tokenizer = try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: data)
        let count = EncodeScratch.maximumReusableInputBytes / 2 + 1
        let ids = tokenizer.encode(text: String(repeating: "a ", count: count), addSpecialTokens: false)
        #expect(ids.count == count)
        #expect(ids.allSatisfy { $0 == 1 })
        #expect(tokenizer.encode(text: "a a", addSpecialTokens: false) == [1, 1])
    }
}
