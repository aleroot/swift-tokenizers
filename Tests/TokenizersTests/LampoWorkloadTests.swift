import Foundation
import Testing

@testable import Tokenizers

@Suite("Lampo tokenizer workloads")
struct LampoWorkloadTests {
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
            splitter: nil, normalizer: nil, normalizedSplitter: nil, preTokenizer: nil, fuseUnknownId: nil)
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
