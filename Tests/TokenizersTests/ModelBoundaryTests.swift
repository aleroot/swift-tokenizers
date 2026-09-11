import Foundation
import Testing

@testable import Tokenizers

/// Reproduced with the Rust-backed tokenizers 0.23.2 package. Exercise both JSON paths:
/// the packed tokenizer parser and the generic Config decoder used by callers.
@Suite("Model boundary regressions")
struct ModelBoundaryTests {
    private func tokenizer(_ json: String, packed: Bool) throws -> PreTrainedTokenizer {
        let bytes = Data(json.utf8)
        let data = try packed ? Config(tokenizerJSON: bytes) : JSONDecoder().decode(Config.self, from: bytes)
        return try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: data)
    }

    @Test("Unigram uses the true minimum and resolves fused surfaces back to model IDs", arguments: [false, true])
    func positiveScores(packed: Bool) throws {
        let tokenizer = try tokenizer(
            #"""
            {"model":{"type":"Unigram","unk_id":0,"byte_fallback":false,
             "vocab":[["<unk>",1000.0],["ax",1979.0]]}}
            """#, packed: packed)
        // Unknown scalars win the lattice (990 + 990 > 1979), but upstream looks up
        // the fused surface again. "ax" is a model piece; "axx" and "axax" are not.
        for (text, expected): (String, [Int]) in [("ax", [1]), ("axx", [0]), ("axax", [0])] {
            for _ in 0..<2 {
                #expect(tokenizer.encode(text: text, addSpecialTokens: false) == expected)
            }
            #expect(tokenizer.tokenize(text: text) == [text])
            let encoding = try tokenizer.encode(text: text, addSpecialTokens: false, withOffsets: true)
            #expect(encoding.ids == expected)
            #expect(encoding.offsets == [0..<text.utf8.count])
        }
    }

    @Test("The last duplicate Unigram piece supplies both its ID and its score", arguments: [false, true])
    func duplicatePieces(packed: Bool) throws {
        let tokenizer = try tokenizer(
            #"""
            {"model":{"type":"Unigram","unk_id":0,"vocab":
             [["<unk>",0.0],["a",-1.0],["a",-4.0],["b",-1.0],["ab",-3.0]]}}
            """#, packed: packed)
        #expect(tokenizer.convertTokenToId("a") == 2)
        #expect(tokenizer.convertIdToToken(1) == "a")
        #expect(tokenizer.convertIdToToken(2) == "a")
        #expect(tokenizer.encode(text: "a", addSpecialTokens: false) == [2])
        // Using the first duplicate's score would select a + b instead of ab.
        #expect(tokenizer.encode(text: "ab", addSpecialTokens: false) == [4])
        #expect(tokenizer.tokenize(text: "ab") == ["ab"])
        #expect(try tokenizer.encode(text: "ab", addSpecialTokens: false, withOffsets: true).ids == [4])
    }

    @Test("A literal Unigram unknown piece is a vocabulary match before byte fallback", arguments: [false, true])
    func literalUnknownByteFallback(packed: Bool) throws {
        let tokenizer = try tokenizer(
            #"""
            {"model":{"type":"Unigram","unk_id":0,"byte_fallback":true,
             "vocab":[["?",0.0],["<0x3F>",-1.0],["<0x78>",-1.0]]}}
            """#, packed: packed)
        #expect(tokenizer.encode(text: "?", addSpecialTokens: false) == [0])
        #expect(tokenizer.tokenize(text: "?") == ["?"])
        #expect(try tokenizer.encode(text: "?", addSpecialTokens: false, withOffsets: true).ids == [0])
        // Once fused with an unknown scalar, "?x" has no vocabulary entry.
        #expect(tokenizer.encode(text: "?x", addSpecialTokens: false) == [1, 2])
    }

    @Test("BPE merge operands and products must belong to the model vocabulary", arguments: [false, true])
    func missingMergeVocabulary(packed: Bool) throws {
        for vocab in [#"{"b":0,"ab":1}"#, #"{"a":0,"ab":1}"#, #"{"a":0,"b":1}"#] {
            for merges in [#"[["a","b"]]"#, #"["a b"]"#] {
                let json = """
                    {"model":{"type":"BPE","vocab":\(vocab),"merges":\(merges)}}
                    """
                #expect(throws: TokenizerError.self) { try tokenizer(json, packed: packed) }
            }
        }
        // An added token cannot repair a missing model merge product.
        #expect(throws: TokenizerError.self) {
            try tokenizer(
                #"""
                {"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[["a","b"]]},
                 "added_tokens":[{"id":2,"content":"ab","special":false,"normalized":false,"single_word":false,"lstrip":false,"rstrip":false}]}
                """#, packed: packed)
        }
    }

    @Test("Valid BPE merges strip continuing prefixes and retain end suffixes", arguments: [false, true])
    func validAffixedMerges(packed: Bool) throws {
        let tokenizer = try tokenizer(
            #"""
            {"model":{"type":"BPE","continuing_subword_prefix":"##","end_of_word_suffix":"</w>",
             "vocab":{"a":0,"##b</w>":1,"ab</w>":2},"merges":[["a","##b</w>"]]}}
            """#, packed: packed)
        #expect(tokenizer.encode(text: "ab", addSpecialTokens: false) == [2])
    }

    @Test("An added token cannot supply WordLevel's missing model unknown", arguments: [false, true])
    func wordLevelAddedUnknown(packed: Bool) throws {
        #expect(throws: TokenizerError.self) {
            try tokenizer(
                #"""
                {"model":{"type":"WordLevel","unk_token":"<unk>","vocab":{"a":0}},
                 "added_tokens":[{"id":1,"content":"<unk>","special":true,"normalized":false,"single_word":false,"lstrip":false,"rstrip":false}]}
                """#, packed: packed)
        }
    }
}
