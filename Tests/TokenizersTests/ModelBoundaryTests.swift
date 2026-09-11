import Foundation
import Testing

@testable import Tokenizers

/// Reproduced with the Rust-backed tokenizers 0.23.2 package. Exercise both JSON paths:
/// the packed tokenizer parser and the generic Config decoder used by callers.
@Suite("Model boundary regressions")
struct ModelBoundaryTests {
    private func tokenizer(_ json: String, packed: Bool, config: Config = [:]) throws -> PreTrainedTokenizer {
        let bytes = Data(json.utf8)
        let data = try packed ? Config(tokenizerJSON: bytes) : JSONDecoder().decode(Config.self, from: bytes)
        return try PreTrainedTokenizer(tokenizerConfig: config, tokenizerData: data)
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

    @Test("BPE and WordPiece unknown tokens must belong to the model", arguments: [false, true])
    func modelUnknownVocabulary(packed: Bool) throws {
        for type in ["BPE", "WordPiece"] {
            let settings = type == "BPE" ? #""merges":[]"#
                : ###""continuing_subword_prefix":"##","max_input_chars_per_word":100"###
            for added in ["[]", #"[{"id":1,"content":"<unk>","special":true,"normalized":false,"single_word":false,"lstrip":false,"rstrip":false}]"#] {
                let json = """
                    {"model":{"type":"\(type)","unk_token":"<unk>","vocab":{"a":0},\(settings)},
                     "added_tokens":\(added)}
                    """
                // HF loads this, then errors on "x". Swift rejects at load because encode is nonthrowing.
                #expect(throws: TokenizerError.self) { try tokenizer(json, packed: packed) }
            }
        }
    }

    @Test("BPE uses its serialized unknown token and fusion policy", arguments: [false, true])
    func bpeModelUnknownPolicy(packed: Bool) throws {
        for field in [#""unk_token":"<unk>","#, #""unk_token":null,"#, ""] {
            let tokenizer = try tokenizer(
                """
                {"model":{"type":"BPE",\(field)"fuse_unk":false,
                 "vocab":{"a":0,"<unk>":1,"<other>":2},"merges":[]}}
                """, packed: packed, config: ["unk_token": "<other>", "fuse_unk": true])
            // Wrapper metadata stays compatible even though segmentation uses the model.
            #expect(tokenizer.unknownToken == "<other>")
            #expect(tokenizer.unknownTokenId == 2)
            let hasUnknown = field.contains("<unk>")
            let ids = hasUnknown ? [1, 1, 0] : [0]
            for _ in 0..<2 { #expect(tokenizer.encode(text: "xxa", addSpecialTokens: false) == ids) }
            #expect(tokenizer.tokenize(text: "xxa") == (hasUnknown ? ["<unk>", "<unk>", "a"] : ["a"]))
            let encoding = try tokenizer.encode(text: "xxa", addSpecialTokens: false, withOffsets: true)
            #expect(encoding.ids == ids)
            // HF 0.23.2 incorrectly shifts "a" to (0, 1) after dropping "xx" when
            // no fallback is configured. Preserve its actual source location (2, 3).
            #expect(encoding.offsets == (hasUnknown ? [0..<1, 1..<2, 2..<3] : [2..<3]))
        }
    }

    @Test("Model unknowns coexist with added tokens without changing cached IDs or offsets", arguments: [false, true])
    func validModelUnknown(packed: Bool) throws {
        for type in ["BPE", "WordPiece"] {
            let settings = type == "BPE" ? #""merges":[]"#
                : ###""continuing_subword_prefix":"##","max_input_chars_per_word":100"###
            let tokenizer = try tokenizer(
                """
                {"model":{"type":"\(type)","unk_token":"<unk>","vocab":{"a":0,"<unk>":1},\(settings)},
                 "added_tokens":[{"id":2,"content":"<added>","special":true,"normalized":false,
                 "single_word":false,"lstrip":false,"rstrip":false}]}
                """, packed: packed)
            for _ in 0..<2 {
                #expect(tokenizer.encode(text: "x<added>a", addSpecialTokens: false) == [1, 2, 0])
            }
            #expect(tokenizer.tokenize(text: "x<added>a") == ["<unk>", "<added>", "a"])
            let encoding = try tokenizer.encode(text: "x<added>a", addSpecialTokens: false, withOffsets: true)
            #expect(encoding.ids == [1, 2, 0])
            #expect(encoding.offsets == [0..<1, 1..<8, 8..<9])
        }
    }

    @Test("Malformed and mixed BPE merge formats are rejected", arguments: [false, true])
    func malformedMerges(packed: Bool) throws {
        // All referenced operands/products exist: failures must come from the merge format.
        for merges in [#"["a  b"]"#, #"["foo bar baz"]"#, #"["a"]"#, #"[""]"#,
                       #"[["a"]]"#, #"[["a","b","c"]]"#, #"[["a",1]]"#, #"[null]"#,
                       #"[["a","b"],"b a"]"#, #"["a b",["b","a"]]"#] {
            #expect(throws: TokenizerError.self) {
                try tokenizer(
                    """
                    {"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2," b":3,"a b":4,
                     "foo":5,"bar baz":6,"foobar baz":7,"ba":8},"merges":\(merges)}}
                    """, packed: packed)
            }
        }
    }

    @Test("Legacy BPE headers preserve rank; empty operands and tuple spaces remain valid", arguments: [false, true])
    func validLegacyMerges(packed: Bool) throws {
        for merges in [##"["#version: 0.2","b c","#version ignored","a b"]"##,
                       #"[["b","c"],["a","b"]]"#] {
            let tokenizer = try tokenizer(
                """
                {"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"bc":4},"merges":\(merges)}}
                """, packed: packed)
            #expect(tokenizer.encode(text: "abc", addSpecialTokens: false) == [0, 4])
        }
        for merges in [#"[" a","a "]"#, #"[["","a"],["a",""]]"#] {
            let tokenizer = try tokenizer(
                """
                {"model":{"type":"BPE","vocab":{"":0,"a":1},"merges":\(merges)}}
                """, packed: packed)
            #expect(tokenizer.encode(text: "a", addSpecialTokens: false) == [1])
        }
        // A combining mark can extend the delimiter's grapheme; split on the space scalar.
        let combining = try tokenizer(
            #"{"model":{"type":"BPE","vocab":{"a":0,"\u0301":1,"a\u0301":2},"merges":["a \u0301"]}}"#,
            packed: packed)
        #expect(combining.encode(text: "a\u{301}", addSpecialTokens: false) == [2])
        let tokenizer = try tokenizer(
            #"{"model":{"type":"BPE","vocab":{"a":0," ":1,"b":2," b":3,"a b":4},"merges":[[" ","b"],["a"," b"]]}}"#,
            packed: packed)
        #expect(tokenizer.encode(text: "a b", addSpecialTokens: false) == [4])
    }
}
