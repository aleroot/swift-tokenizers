// Post-processing behaviour, ported from Hugging Face `tokenizers` 0.23.2:
// `processors/roberta.rs`, `processors/bert.rs`, `processors/template.rs` and
// `pre_tokenizers/byte_level.rs::processor_trims_offsets`.
//
// Two invariants distinguish these processors from the swift-transformers implementation
// they replace: token *content* is never rewritten (only offsets are trimmed), and the ids
// declared beside each special token are authoritative, not a vocabulary lookup.
import Foundation
import Testing

@testable import Tokenizers

@Suite("Post-processor functionality tests")
struct PostProcessorTests {
    static func robertaConfig(trimOffsets: Bool = true, addPrefixSpace: Bool = true) -> Config {
        Config([
            "cls": ["<s>", 0 as UInt],
            "sep": ["</s>", 1 as UInt],
            "trim_offsets": Config(trimOffsets),
            "add_prefix_space": Config(addPrefixSpace),
        ])
    }

    @Suite("RoBERTa post-processing behavior")
    struct RoBERTaProcessingTests {
        @Test("Wraps a single sequence, and a pair with a doubled separator")
        func wrapsSequences() throws {
            let processor = try RobertaProcessing(config: PostProcessorTests.robertaConfig())
            #expect(
                processor.postProcess(tokens: ["Hello", "there"], tokensPair: nil)
                    == ["<s>", "Hello", "there", "</s>"])
            #expect(
                processor.postProcess(tokens: ["Hello"], tokensPair: ["pair"])
                    == ["<s>", "Hello", "</s>", "</s>", "pair", "</s>"])
        }

        @Test("Is a no-op without special tokens")
        func noSpecialTokens() throws {
            let processor = try RobertaProcessing(config: PostProcessorTests.robertaConfig())
            #expect(
                processor.postProcess(tokens: ["Hello"], tokensPair: ["pair"], addSpecialTokens: false)
                    == ["Hello", "pair"])
        }

        @Test("Ignores an empty token pair")
        func ignoresEmptyTokensPair() throws {
            let processor = try RobertaProcessing(config: PostProcessorTests.robertaConfig())
            #expect(
                processor.postProcess(tokens: ["The", "sun"], tokensPair: [])
                    == ["<s>", "The", "sun", "</s>"])
        }

        @Test(
            "Never rewrites token content, whatever the trim settings",
            arguments: [(true, true), (true, false), (false, true), (false, false)])
        func preservesTokenContent(trim: Bool, prefixSpace: Bool) throws {
            // `trim_offsets` moves offset boundaries only; upstream leaves the strings alone.
            let tokens = [" The ", " sun", "sets ", "  in ", "  the    ", "west"]
            let processor = try RobertaProcessing(
                config: PostProcessorTests.robertaConfig(trimOffsets: trim, addPrefixSpace: prefixSpace))
            #expect(processor.postProcess(tokens: tokens, tokensPair: nil) == ["<s>"] + tokens + ["</s>"])
        }

        @Test("Inserts the ids declared in the configuration")
        func usesDeclaredIds() throws {
            let processor = try RobertaProcessing(config: PostProcessorTests.robertaConfig())
            var ids = [7, 8]
            // A vocabulary lookup that would answer differently must not be consulted.
            processor.bound { _ in 999 }.postProcess(ids: &ids, addSpecialTokens: true)
            #expect(ids == [0, 7, 8, 1])
        }
    }

    @Suite("BERT post-processing behavior")
    struct BertProcessingTests {
        static let config = Config(["cls": ["[CLS]", 101 as UInt], "sep": ["[SEP]", 102 as UInt]])

        @Test("Wraps single and pair sequences")
        func wrapsSequences() throws {
            let processor = try BertProcessing(config: Self.config)
            #expect(
                processor.postProcess(tokens: ["Hello"], tokensPair: nil) == ["[CLS]", "Hello", "[SEP]"])
            #expect(
                processor.postProcess(tokens: ["Hello"], tokensPair: ["pair"])
                    == ["[CLS]", "Hello", "[SEP]", "pair", "[SEP]"])
            #expect(
                processor.postProcess(tokens: ["Hello"], tokensPair: ["pair"], addSpecialTokens: false)
                    == ["Hello", "pair"])
        }

        @Test("Inserts the ids declared in the configuration")
        func usesDeclaredIds() throws {
            let processor = try BertProcessing(config: Self.config)
            var ids = [5]
            processor.bound { _ in 999 }.postProcess(ids: &ids, addSpecialTokens: true)
            #expect(ids == [101, 5, 102])
        }
    }

    @Suite("Template post-processing behavior")
    struct TemplateProcessingTests {
        /// `[CLS] $A [SEP]` where the `special_tokens` map disagrees with the vocabulary.
        static func config(single: [Config], specialTokens: Config) -> Config {
            Config([
                "type": "TemplateProcessing",
                "single": Config(single),
                "pair": [
                    ["SpecialToken": ["id": "[CLS]", "type_id": 0]],
                    ["Sequence": ["id": "A", "type_id": 0]],
                    ["Sequence": ["id": "B", "type_id": 1]],
                ],
                "special_tokens": specialTokens,
            ])
        }

        static let specials: Config = [
            "[CLS]": ["id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]],
            "[SEP]": ["id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]],
            "[PAIR]": ["id": "[PAIR]", "ids": [7, 8], "tokens": ["[A]", "[B]"]],
        ]

        static let cls: Config = ["SpecialToken": ["id": "[CLS]", "type_id": 0]]
        static let sep: Config = ["SpecialToken": ["id": "[SEP]", "type_id": 0]]
        static let sequenceA: Config = ["Sequence": ["id": "A", "type_id": 0]]

        @Test("Uses the ids of the special_tokens map, not the vocabulary")
        func declaredIds() throws {
            let processor = try TemplateProcessing(
                config: Self.config(single: [Self.cls, Self.sequenceA, Self.sep], specialTokens: Self.specials))
            var ids = [5, 6]
            processor.bound { _ in 999 }.postProcess(ids: &ids, addSpecialTokens: true)
            #expect(ids == [101, 5, 6, 102])
            #expect(
                processor.postProcess(tokens: ["a", "b"], tokensPair: nil) == ["[CLS]", "a", "b", "[SEP]"])
        }

        @Test("A special piece can expand to several tokens")
        func multiTokenPiece() throws {
            let pair: Config = ["SpecialToken": ["id": "[PAIR]", "type_id": 0]]
            let processor = try TemplateProcessing(
                config: Self.config(single: [pair, Self.sequenceA], specialTokens: Self.specials))
            var ids = [5]
            processor.bound { _ in 999 }.postProcess(ids: &ids, addSpecialTokens: true)
            #expect(ids == [7, 8, 5])
            #expect(processor.postProcess(tokens: ["a"], tokensPair: nil) == ["[A]", "[B]", "a"])
        }

        @Test("Repeating the sequence keeps every copy")
        func repeatedSequence() throws {
            let processor = try TemplateProcessing(
                config: Self.config(
                    single: [Self.cls, Self.sequenceA, Self.sep, Self.sequenceA, Self.sep],
                    specialTokens: Self.specials))
            var ids = [5]
            processor.bound { _ in 999 }.postProcess(ids: &ids, addSpecialTokens: true)
            #expect(ids == [101, 5, 102, 5, 102])
        }

        @Test("Falls back to the vocabulary when the file has no special_tokens map")
        func vocabularyFallback() throws {
            let processor = try TemplateProcessing(
                config: Self.config(single: [Self.cls, Self.sequenceA], specialTokens: Config()))
            var ids = [5]
            processor.bound { $0 == "[CLS]" ? 42 : nil }.postProcess(ids: &ids, addSpecialTokens: true)
            #expect(ids == [42, 5])
        }

        @Test("Special tokens are dropped when they are not requested")
        func withoutSpecialTokens() throws {
            let processor = try TemplateProcessing(
                config: Self.config(single: [Self.cls, Self.sequenceA, Self.sep], specialTokens: Self.specials))
            var ids = [5, 6]
            processor.bound { _ in 999 }.postProcess(ids: &ids, addSpecialTokens: false)
            #expect(ids == [5, 6])
        }

        @Test("A single template may not reference sequence B")
        func rejectsSequenceB() {
            let single: Config = ["Sequence": ["id": "B", "type_id": 0]]
            #expect(throws: TokenizerError.self) {
                try TemplateProcessing(config: Self.config(single: [single], specialTokens: Self.specials))
            }
        }
    }

    @Suite("Byte-level offset trimming")
    struct ByteLevelOffsetTests {
        /// `pre_tokenizers/byte_level.rs::processor_trims_offsets`, with byte offsets.
        static let tokens = ["Ġ", "ĠĠĠĠHelloĠĠ", "ĠĠHello", "HelloĠĠ", "ĠĠĠĠ"]
        static let offsets = [0..<1, 0..<11, 11..<18, 18..<25, 25..<29]

        static func trim(_ processor: any PostProcessor, addSpecialTokens: Bool = false) throws -> [Range<Int>?] {
            let aligned = zip(tokens.indices, offsets).map { index, offset in
                AlignedToken(id: index, offset: offset, spelling: tokens[index])
            }
            return try processor.processOffsets(
                aligned, addSpecialTokens: addSpecialTokens, resolve: { _ in nil },
                spelling: { tokens.indices.contains($0) ? tokens[$0] : nil }
            ).map(\.offset)
        }

        @Test("Trims leading and trailing byte-level spaces")
        func trimsOffsets() throws {
            let processor = ByteLevelPostProcessor(config: ["trim_offsets": true, "add_prefix_space": true])
            #expect(try Self.trim(processor) == [0..<0, 4..<9, 13..<18, 18..<23, 29..<29])
        }

        @Test("Keeps the offsets untouched when trimming is off")
        func keepsOffsets() throws {
            let processor = ByteLevelPostProcessor(config: ["trim_offsets": false])
            #expect(try Self.trim(processor) == Self.offsets)
        }

        @Test("Never rewrites token content")
        func contentUnchanged() throws {
            let processor = ByteLevelPostProcessor(config: ["trim_offsets": true])
            #expect(processor.postProcess(tokens: Self.tokens, tokensPair: nil) == Self.tokens)
        }
    }
}
