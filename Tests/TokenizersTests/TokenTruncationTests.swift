import Testing

@testable import Tokenizers

@Suite("Special-token-aware truncation")
struct TokenTruncationTests {
    private func tokenizer(processor: Config) throws -> any Tokenizer {
        try AutoTokenizer.from(
            tokenizerConfig: [:],
            tokenizerData: [
                "model": ["type": "BPE", "vocab": ["a": 0, "b": 1, "é": 2, "[CLS]": 3, "[SEP]": 4], "merges": []],
                "post_processor": processor,
            ])
    }
    private var bert: Config { ["type": "BertProcessing", "cls": ["[CLS]", 3], "sep": ["[SEP]", 4]] }

    @Test("Both sides preserve CLS/SEP and original Unicode source ranges")
    func sides() throws {
        let tokenizer = try tokenizer(processor: bert)
        let right = try tokenizer.encode(text: "abé", maxLength: 4, withOffsets: true)
        #expect(right.ids == [3, 0, 1, 4])
        #expect(right.offsets == [nil, 0..<1, 1..<2, nil])
        let left = try tokenizer.encode(text: "abé", maxLength: 4, truncationSide: .left, withOffsets: true)
        #expect(left.ids == [3, 1, 2, 4])
        #expect(left.offsets == [nil, 1..<2, 2..<4, nil])
        #expect(try tokenizer.encode(text: "abé", maxLength: 4).ids == right.ids)
        #expect(try tokenizer.encode(text: "abé", maxLength: 4, truncationSide: .left).ids == left.ids)
        #expect(try tokenizer.encode(text: "abé", maxLength: 2).ids == [3, 4])
        #expect(try tokenizer.encode(text: "", maxLength: 2).ids == [3, 4])
        #expect(try tokenizer.encode(text: "abé", addSpecialTokens: false, maxLength: 0).ids.isEmpty)
        #expect(
            try tokenizer.encode(text: "abé", addSpecialTokens: false, maxLength: 1, truncationSide: .left).ids == [2])
        #expect(throws: TokenizerError.self) { try tokenizer.encode(text: "a", maxLength: 1) }
        #expect(throws: TokenizerError.self) { try tokenizer.encode(text: "a", maxLength: -1) }
    }

    @Test("Composed processors, repeated sequences and multi-ID specials use the total output budget")
    func repeatedSequence() throws {
        let template: Config = [
            "type": "TemplateProcessing",
            "single": [
                ["SpecialToken": ["id": "pair", "type_id": 0]], ["Sequence": ["id": "A", "type_id": 0]],
                ["Sequence": ["id": "A", "type_id": 0]],
            ], "pair": [], "special_tokens": ["pair": ["ids": [3, 4], "tokens": ["[CLS]", "[SEP]"]]],
        ]
        let tokenizer = try tokenizer(processor: ["type": "Sequence", "processors": [template, bert]])
        for offsets in [false, true] {
            #expect(try tokenizer.encode(text: "abé", maxLength: 7, withOffsets: offsets).ids == [3, 3, 4, 0, 0, 4])
            #expect(
                try tokenizer.encode(text: "abé", maxLength: 8, withOffsets: offsets).ids == [3, 3, 4, 0, 1, 0, 1, 4])
            #expect(
                try tokenizer.encode(text: "abé", addSpecialTokens: false, maxLength: 3, withOffsets: offsets).ids == [
                    0, 0,
                ])
        }
    }
}
