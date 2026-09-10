// Created by Jan Krukowski on 23/11/2023.

import Foundation
import Testing

@testable import Tokenizers

@Suite("Pre-Tokenizer Tests")
struct PreTokenizerTests {
    @Test("Whitespace pre-tokenizer splits text by whitespace")
    func whitespacePreTokenizer() {
        let preTokenizer = WhitespacePreTokenizer(config: Config([String: Config]()))

        #expect(
            preTokenizer.preTokenize(text: "Hey friend!") == ["Hey", "friend!"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", "friend!", "How", "are", "you?!?",
            ]
        )
        #expect(
            preTokenizer.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                "Hey,", "friend,", "what's", "up?",
            ]
        )
    }

    @Test("Punctuation pre-tokenizer separates punctuation")
    func punctuationPreTokenizer() {
        let preTokenizer = PunctuationPreTokenizer(config: Config([String: Config]()))

        #expect(
            preTokenizer.preTokenize(text: "Hey friend!") == ["Hey friend", "!"]
        )
        // Default behaviour is `Isolated` (each punctuation scalar on its own), as in
        // `tokenizers.pre_tokenizers.Punctuation()`.
        #expect(
            preTokenizer.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey friend", "!", "     How are you", "?", "!", "?",
            ]
        )
        let contiguous = PunctuationPreTokenizer(config: Config(["behavior": Config("Contiguous")]))
        #expect(
            contiguous.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey friend", "!", "     How are you", "?!?",
            ]
        )
        #expect(
            preTokenizer.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                "   Hey", ",", "    friend", ",", "    what", "'", "s up", "?", "  ",
            ]
        )
    }

    @Test("Split pre-tokenizer with a literal pattern honours every behaviour")
    func splitLiteralBehaviors() throws {
        // Expectations from `tokenizers.pre_tokenizers.Split(' ', behavior)` / `Split('ab', behavior)`.
        let cases: [(String, [String], [String])] = [
            ("Isolated", ["a", " ", " ", "b", " ", "c", " ", " "], ["x", "ab", "ab"]),
            ("Removed", ["a", "b", "c"], ["x"]),
            ("MergedWithPrevious", ["a ", " ", "b ", "c ", " "], ["xab", "ab"]),
            ("MergedWithNext", ["a", " ", " b", " c", " ", " "], ["x", "ab", "ab"]),
            ("Contiguous", ["a", "  ", "b", " ", "c", "  "], ["x", "abab"]),
        ]
        for (behavior, spaces, abs) in cases {
            let space = try SplitPreTokenizer(
                config: Config(["pattern": Config(["String": Config(" ")]), "behavior": Config(behavior)]))
            #expect(space.preTokenize(text: "a  b c  ") == spaces, Comment(rawValue: behavior))
            let ab = try SplitPreTokenizer(
                config: Config(["pattern": Config(["String": Config("ab")]), "behavior": Config(behavior)]))
            #expect(ab.preTokenize(text: "xabab") == abs, Comment(rawValue: behavior))
        }
        // No match at all: the text is a single chunk (gemma-4 after its space → ▁ normalizer).
        let none = try SplitPreTokenizer(
            config: Config(["pattern": Config(["String": Config(" ")]), "behavior": Config("MergedWithPrevious")]))
        #expect(none.preTokenize(text: "▁Hello▁world") == ["▁Hello▁world"])
    }

    @Test("Byte-level pre-tokenizer with various configurations")
    func byteLevelPreTokenizer() {
        let preTokenizer1 = ByteLevelPreTokenizer(config: Config([String: Config]()))

        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!") == ["Hey", "Ġfriend", "!"]
        )
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", "Ġfriend", "!", "ĠĠĠĠ", "ĠHow", "Ġare", "Ġyou", "?!?",
            ]
        )
        #expect(
            preTokenizer1.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                "ĠĠ", "ĠHey", ",", "ĠĠĠ", "Ġfriend", ",", "ĠĠĠ", "Ġwhat", "'s", "Ġup", "?", "ĠĠ",
            ]
        )

        // `add_prefix_space` prepends to the section before the regex split, so only the first
        // piece gains a space — verified with `tokenizers`:
        // ByteLevel(add_prefix_space=True).pre_tokenize_str("Hey friend!") == ["ĠHey", "Ġfriend", "!"]
        let preTokenizer2 = ByteLevelPreTokenizer(config: Config(["addPrefixSpace": true]))

        #expect(
            preTokenizer2.preTokenize(text: "Hey friend!") == ["ĠHey", "Ġfriend", "!"]
        )
        #expect(
            preTokenizer2.preTokenize(text: "Hey friend!     How are you?!?") == [
                "ĠHey", "Ġfriend", "!", "ĠĠĠĠ", "ĠHow", "Ġare", "Ġyou", "?!?",
            ]
        )
        #expect(
            preTokenizer2.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                "ĠĠ", "ĠHey", ",", "ĠĠĠ", "Ġfriend", ",", "ĠĠĠ", "Ġwhat", "'s", "Ġup", "?",
                "ĠĠ",
            ]
        )

        let preTokenizer3 = ByteLevelPreTokenizer(config: Config(["useRegex": false]))

        #expect(
            preTokenizer3.preTokenize(text: "Hey friend!") == ["HeyĠfriend!"]
        )
        #expect(
            preTokenizer3.preTokenize(text: "Hey friend!     How are you?!?") == [
                "HeyĠfriend!ĠĠĠĠĠHowĠareĠyou?!?"
            ]
        )
        #expect(
            preTokenizer3.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                "ĠĠĠHey,ĠĠĠĠfriend,ĠĠĠĠwhat'sĠup?ĠĠ"
            ]
        )
    }

    @Test("Digits pre-tokenizer handles numeric content")
    func digitsPreTokenizer() {
        let preTokenizer1 = DigitsPreTokenizer(config: Config([String: Config]()))

        #expect(
            preTokenizer1.preTokenize(text: "1 12 123! 1234abc") == [
                "1", " ", "12", " ", "123", "! ", "1234", "abc",
            ]
        )

        let preTokenizer2 = DigitsPreTokenizer(config: Config(["individualDigits": true]))

        #expect(
            preTokenizer2.preTokenize(text: "1 12 123! 1234abc") == [
                "1", " ", "1", "2", " ", "1", "2", "3", "! ", "1", "2", "3", "4", "abc",
            ]
        )
    }

    @Test("Split pre-tokenizer with string and regex patterns")
    func splitPreTokenizer() throws {
        let preTokenizer1 = try SplitPreTokenizer(config: Config(["pattern": ["String": " "]]))
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!") == ["Hey", " ", "friend!"]
        )
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", " ", "friend!", " ", " ", " ", " ", " ", "How", " ", "are", " ", "you?!?",
            ]
        )
        #expect(
            preTokenizer1.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                " ", " ", " ", "Hey,", " ", " ", " ", " ", "friend,", " ", " ", " ", " ", "what's",
                " ", "up?", " ", " ",
            ]
        )

        let preTokenizer2 = try SplitPreTokenizer(config: Config(["pattern": ["Regex": "\\s"]]))
        #expect(
            preTokenizer2.preTokenize(text: "Hey friend!") == ["Hey", " ", "friend!"]
        )
        #expect(
            preTokenizer2.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", " ", "friend!", " ", " ", " ", " ", " ", "How", " ", "are", " ", "you?!?",
            ]
        )
        #expect(
            preTokenizer2.preTokenize(text: "   Hey,    friend,    what's up?  ") == [
                " ", " ", " ", "Hey,", " ", " ", " ", " ", "friend,", " ", " ", " ", " ", "what's",
                " ", "up?", " ", " ",
            ]
        )

        let preTokenizer3 = try SplitPreTokenizer(
            config: Config([
                "pattern": [
                    "Regex":
                        "(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                ], "invert": true,
            ]))
        #expect(
            preTokenizer3.preTokenize(text: "Hello") == ["Hello"]
        )

        #expect(
            preTokenizer3.preTokenize(text: "Hey friend!") == ["Hey", " friend", "!"]
        )
        #expect(
            preTokenizer3.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", " friend", "!", "    ", " How", " are", " you", "?!?",
            ]
        )
    }

    @Test("Split behavior merged with previous")
    func splitBehaviorMergedWithPrevious() {
        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) == [
                "the-", "final-", "-", "countdown",
            ]
        )

        #expect(
            "the-final--countdown-".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) == [
                "the-", "final-", "-", "countdown-",
            ]
        )

        #expect(
            "the-final--countdown--".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) == [
                "the-", "final-", "-", "countdown-", "-",
            ]
        )

        #expect(
            "-the-final--countdown--".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) == [
                "-", "the-", "final-", "-", "countdown-", "-",
            ]
        )

        #expect(
            "--the-final--countdown--".split(by: "-", options: .caseInsensitive, behavior: .mergedWithPrevious) == [
                "-", "-", "the-", "final-", "-", "countdown-", "-",
            ]
        )
    }

    @Test("Split behavior merged with next")
    func splitBehaviorMergedWithNext() {
        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext) == [
                "the", "-final", "-", "-countdown",
            ]
        )

        #expect(
            "-the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext) == [
                "-the", "-final", "-", "-countdown",
            ]
        )

        #expect(
            "--the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext) == [
                "-", "-the", "-final", "-", "-countdown",
            ]
        )

        #expect(
            "--the-final--countdown-".split(by: "-", options: .caseInsensitive, behavior: .mergedWithNext) == [
                "-", "-the", "-final", "-", "-countdown", "-",
            ]
        )
    }

    @Test("Split behavior other")
    func splitBehaviorOther() {
        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .isolated) == [
                "the", "-", "final", "-", "-", "countdown",
            ]
        )

        #expect(
            "the-final--countdown".split(by: "-", options: .caseInsensitive, behavior: .removed) == [
                "the", "final", "countdown",
            ]
        )
    }

    /// https://github.com/huggingface/tokenizers/pull/1357
    @Test("Metaspace pre-tokenizer with prefix space handling")
    func metaspacePreTokenizer() {
        // Prepend "always"
        let preTokenizer = MetaspacePreTokenizer(
            config: Config([
                "add_prefix_space": true,
                "replacement": "▁",
                "prepend_scheme": "always",
            ]))

        let text = "Hey my friend <s>how▁are you"
        let tokens =
            text
            .split(by: "<s>", includeSeparators: true)
            .flatMap { preTokenizer.preTokenize(text: $0) }

        #expect(
            tokens == ["▁Hey", "▁my", "▁friend", "▁", "▁<s>", "▁how", "▁are", "▁you"]
        )
    }

    @Test("Metaspace prepend_scheme 'always' without add_prefix_space (XLM-RoBERTa case)")
    func metaspacePrependSchemeAlwaysWithoutAddPrefixSpace() {
        let preTokenizer = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "prepend_scheme": "always",
            ]))

        #expect(
            preTokenizer.preTokenize(text: "Hello") == ["▁Hello"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hello world") == ["▁Hello", "▁world"]
        )
        // Already starts with replacement — no double prepend
        #expect(
            preTokenizer.preTokenize(text: "▁Hello") == ["▁Hello"]
        )
    }

    @Test("Metaspace prepend_scheme 'first' only prepends on first section")
    func metaspacePrependSchemeFirst() {
        let preTokenizer = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "prepend_scheme": "first",
            ]))

        // First section (default options include .firstSection)
        #expect(
            preTokenizer.preTokenize(text: "Hello") == ["▁Hello"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hello world") == ["▁Hello", "▁world"]
        )

        // Non-first section (empty options)
        #expect(
            preTokenizer.preTokenize(text: "Hello", options: []) == ["Hello"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hello world", options: []) == ["Hello", "▁world"]
        )
    }

    @Test("Metaspace prepend_scheme 'never' never prepends")
    func metaspacePrependSchemeNever() {
        let preTokenizer = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "prepend_scheme": "never",
            ]))

        #expect(
            preTokenizer.preTokenize(text: "Hello") == ["Hello"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hello world") == ["Hello", "▁world"]
        )
    }

    @Test("Metaspace legacy add_prefix_space without prepend_scheme")
    func metaspaceLegacyAddPrefixSpace() {
        // add_prefix_space: true, no prepend_scheme → behaves like "always"
        let alwaysTokenizer = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "add_prefix_space": true,
            ]))

        #expect(
            alwaysTokenizer.preTokenize(text: "Hello") == ["▁Hello"]
        )
        #expect(
            alwaysTokenizer.preTokenize(text: "Hello world") == ["▁Hello", "▁world"]
        )

        // add_prefix_space: false, no prepend_scheme → behaves like "never"
        let neverTokenizer = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "add_prefix_space": false,
            ]))

        #expect(
            neverTokenizer.preTokenize(text: "Hello") == ["Hello"]
        )
        #expect(
            neverTokenizer.preTokenize(text: "Hello world") == ["Hello", "▁world"]
        )
    }

    @Test("Metaspace default config (no add_prefix_space, no prepend_scheme) defaults to always")
    func metaspaceDefaultConfig() {
        let preTokenizer = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁"
            ]))

        #expect(
            preTokenizer.preTokenize(text: "Hello") == ["▁Hello"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hello world") == ["▁Hello", "▁world"]
        )
    }

    @Test("Metaspace prepend_scheme supersedes add_prefix_space")
    func metaspacePrependSchemeSupersedesAddPrefixSpace() {
        // prepend_scheme: "always" wins even when add_prefix_space is false
        let preTokenizer = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "add_prefix_space": false,
                "prepend_scheme": "always",
            ]))

        #expect(
            preTokenizer.preTokenize(text: "Hello") == ["▁Hello"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hello world") == ["▁Hello", "▁world"]
        )
    }

    @Test("Metaspace prepend_scheme 'never' supersedes add_prefix_space true")
    func metaspacePrependSchemeNeverSupersedesAddPrefixSpace() {
        // prepend_scheme: "never" wins even when add_prefix_space is true
        let preTokenizer = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "add_prefix_space": true,
                "prepend_scheme": "never",
            ]))

        #expect(
            preTokenizer.preTokenize(text: "Hello") == ["Hello"]
        )
        #expect(
            preTokenizer.preTokenize(text: "Hello world") == ["Hello", "▁world"]
        )
    }

    @Test("Metaspace handles empty string input")
    func metaspaceEmptyString() {
        let always = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "prepend_scheme": "always",
            ]))
        // `tokenizers` never prepends to an empty string (`NormalizedString::prepend` is a no-op),
        // so an empty input yields no pieces.
        #expect(
            always.preTokenize(text: "") == []
        )

        let never = MetaspacePreTokenizer(
            config: Config([
                "replacement": "▁",
                "prepend_scheme": "never",
            ]))
        #expect(
            never.preTokenize(text: "") == []
        )
    }

    @Test("BERT pre-tokenizer performs basic splitting")
    func bertPreTokenizer() {
        let preTokenizer1 = BertPreTokenizer(config: Config([String: Config]()))
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!") == ["Hey", "friend", "!"]
        )
        #expect(
            preTokenizer1.preTokenize(text: "Hey friend!     How are you?!?") == [
                "Hey", "friend", "!", "How", "are", "you", "?", "!", "?",
            ]
        )
        #expect(
            preTokenizer1.preTokenize(text: "   Hey,    friend ,    what's up?  ") == [
                "Hey", ",", "friend", ",", "what", "\'", "s", "up", "?",
            ]
        )
        #expect(
            preTokenizer1.preTokenize(text: "   Hey,    friend ,  0 99  what's up?  ") == [
                "Hey", ",", "friend", ",", "0", "99", "what", "\'", "s", "up", "?",
            ]
        )
    }
}

@Suite("Metaspace prepend_scheme first inside a Sequence")
struct MetaspaceFirstInSequenceTests {
    /// `tokenizers` prepends only to the split whose original offset is 0, so leading
    /// whitespace removed by an earlier stage suppresses the prefix:
    /// Sequence([WhitespaceSplit(), Metaspace(prepend_scheme="first")]).pre_tokenize_str(" hello world")
    /// == [("hello", (1, 6)), ("world", (7, 12))]
    @Test("Prefix follows the original offset, not the piece index")
    func originalOffset() throws {
        let sequence = try PreTokenizerSequence(
            config: Config([
                "pretokenizers": Config([
                    Config(["type": Config("WhitespaceSplit")]),
                    Config(["type": Config("Metaspace"), "replacement": Config("▁"), "prepend_scheme": Config("first")]
                    ),
                ])
            ]))
        #expect(sequence.preTokenize(text: " hello world") == ["hello", "world"])
        #expect(sequence.preTokenize(text: "hello world") == ["▁hello", "world"])
    }
}

@Suite("Upstream pre-tokenizers added for parity")
struct AddedPreTokenizerTests {
    /// `pre_tokenizers/unicode_scripts/pre_tokenizer.rs::basic` and
    /// `::spaces_are_included_in_every_script`.
    @Test("UnicodeScripts splits at script changes")
    func unicodeScripts() {
        let pre = UnicodeScriptsPreTokenizer(config: [:])
        #expect(pre.preTokenize(text: "どこで生れ。Yes") == ["どこで生れ", "。", "Yes"])
        #expect(pre.preTokenize(text: "Apples are りんご 林檎") == ["Apples are ", "りんご 林檎"])
        // Hiragana, Katakana and the prolonged sound mark all count as Han.
        #expect(pre.preTokenize(text: "グッド") == ["グッド"])
        #expect(pre.preTokenize(text: "コーヒー") == ["コーヒー"])
        #expect(pre.preTokenize(text: "") == [])
        // A leading run that belongs to every script starts no piece, as upstream.
        #expect(pre.preTokenize(text: " abc") == ["abc"])
        #expect(pre.preTokenize(text: "   ") == [])
    }

    /// `pre_tokenizers/fixed_length.rs::basic`, `::custom_length`, `::utf8_characters`.
    @Test("FixedLength chunks by Unicode scalar count")
    func fixedLength() throws {
        let five = try FixedLengthPreTokenizer(config: ["length": 5])
        #expect(five.preTokenize(text: "Hello world") == ["Hello", " worl", "d"])
        #expect(five.preTokenize(text: "Short") == ["Short"])
        #expect(five.preTokenize(text: "") == [])
        let three = try FixedLengthPreTokenizer(config: ["length": 3])
        #expect(three.preTokenize(text: "Hello world") == ["Hel", "lo ", "wor", "ld"])
        #expect(three.preTokenize(text: "Hello 👋 world") == ["Hel", "lo ", "👋 w", "orl", "d"])
        // The default matches `default_length` upstream.
        #expect(try FixedLengthPreTokenizer(config: [:]).preTokenize(text: "Hello world") == ["Hello", " worl", "d"])
        #expect(throws: TokenizerError.self) { try FixedLengthPreTokenizer(config: ["length": 0]) }
    }

    /// `pre_tokenizers/delimiter.rs`: a `Removed` split, so empty pieces disappear.
    @Test("CharDelimiterSplit removes the delimiter and empty pieces")
    func charDelimiterSplit() throws {
        let space = try CharDelimiterSplitPreTokenizer(config: ["delimiter": " "])
        #expect(space.preTokenize(text: "Hey friend!") == ["Hey", "friend!"])
        #expect(space.preTokenize(text: "  Hey   friend!  ") == ["Hey", "friend!"])
        #expect(space.preTokenize(text: "") == [])
        #expect(space.preTokenize(text: "   ") == [])
        let marker = try CharDelimiterSplitPreTokenizer(config: ["delimiter": "▁"])
        #expect(marker.preTokenize(text: "▁Hello▁there") == ["Hello", "there"])
        #expect(throws: TokenizerError.self) { try CharDelimiterSplitPreTokenizer(config: ["delimiter": "ab"]) }
    }

    /// `NormalizedString::prepend` cannot attach to an empty string, so `add_prefix_space`
    /// leaves an empty chunk empty instead of emitting a lone `Ġ`.
    @Test("ByteLevel add_prefix_space is a no-op on empty input")
    func byteLevelEmptyInput() {
        for useRegex in [true, false] {
            let pre = ByteLevelPreTokenizer(config: ["add_prefix_space": true, "use_regex": Config(useRegex)])
            #expect(pre.preTokenize(text: "") == [])
            #expect(pre.preTokenize(text: "Hello") == ["ĠHello"])
        }
    }
}

@Suite("Oniguruma dialect translation")
struct OnigurumaDialectTests {
    /// `tokenizers` compiles `Split` and `Replace` patterns with Oniguruma, whose `\\w` is
    /// `[\\p{Alphabetic}\\p{M}\\p{N}\\p{Pc}]`; ICU's adds ZWJ/ZWNJ and drops `Nl`/`No`.
    @Test("Word classes follow Oniguruma, not ICU")
    func wordClass() throws {
        let split = try SplitPreTokenizer(
            config: ["pattern": ["Regex": #"\w+"#], "behavior": "Isolated", "invert": false])
        // ½ (No), ² (No), Ⅰ (Nl) and ⓒ (So but Alphabetic) are all word characters upstream.
        #expect(split.preTokenize(text: "½²Ⅰ letters") == ["½²Ⅰ", " ", "letters"])
        #expect(split.preTokenize(text: "Hi ⓒ here") == ["Hi", " ", "ⓒ", " ", "here"])
        // ZWNJ / ZWJ are word characters for ICU but not for Oniguruma.
        #expect(split.preTokenize(text: "x\u{200C}y") == ["x", "\u{200C}", "y"])
    }

    @Test("Anchors match at every line boundary")
    func anchors() throws {
        let normalizer = try ReplaceNormalizer(
            config: ["pattern": ["Regex": #"^\s*"#], "content": ""])
        #expect(normalizer.normalize(text: "a\n  b\n  c") == "a\nb\nc")
    }

    @Test("Only \\n is a line terminator, as in Ruby syntax")
    func lineTerminators() throws {
        // `$` matches between `\r` and `\n`; `.` matches `\r`, NEL and the Unicode separators.
        let anchored = try SplitPreTokenizer(
            config: ["pattern": ["Regex": #"^\s+|\s+$|\s+"#], "behavior": "Isolated", "invert": false])
        #expect(anchored.preTokenize(text: "crlf\r\nend") == ["crlf", "\r", "\n", "end"])
        let dot = try SplitPreTokenizer(
            config: ["pattern": ["Regex": #"[a-z]+|."#], "behavior": "Isolated", "invert": false])
        #expect(
            dot.preTokenize(text: "a\rb\u{85}c\u{2028}d\ne") == [
                "a", "\r", "b", "\u{85}", "c", "\u{2028}", "d", "\n", "e",
            ])
        // Ruby's `m` option is ICU's `s`: `.` also matches `\n`.
        let dotall = try SplitPreTokenizer(
            config: ["pattern": ["Regex": #"(?m)[a-z]+|."#], "behavior": "Isolated", "invert": false])
        #expect(dotall.preTokenize(text: "ab\r\n\ncd") == ["ab", "\r", "\n", "\n", "cd"])
        #expect(OnigurumaDialect.translate(#"(?m)a."#) == #"a[\s\S]"#)
        #expect(OnigurumaDialect.translate(#"(?mi:a.)|(?-m)b."#) == #"(?i:a[\s\S])|b[^\n]"#)
        #expect(OnigurumaDialect.translate(#"(?m)(a.)."#) == #"(a[\s\S])[\s\S]"#)
        #expect(OnigurumaDialect.translate(#"(?i-m:a.)."#) == #"(?i:a[^\n])[^\n]"#)
        #expect(OnigurumaDialect.translate(#"(?i:'s|'m)"#) == #"(?i:'s|'m)"#)
        #expect(OnigurumaDialect.translate(#"[(?m).]\."#) == #"[(?m).]\."#)
        #expect(OnigurumaDialect.translate(#"(?=.)(?!.)(?<=.)"#) == #"(?=[^\n])(?![^\n])(?<=[^\n])"#)
    }

    @Test("ASCII digit groups take the scanner path with regex results")
    func asciiDigitGroups() throws {
        // Expectations from `tokenizers.pre_tokenizers.Split(Regex(pattern), "isolated")`.
        for pattern in [#"[0-9][0-9][0-9]"#, #"[0-9]{3}"#] {
            let split = try SplitPreTokenizer(
                config: ["pattern": ["Regex": Config(pattern)], "behavior": "Isolated", "invert": false])
            #expect(split.preTokenize(text: "12345") == ["123", "45"])
            #expect(split.preTokenize(text: "a1234567b") == ["a", "123", "456", "7b"])
            #expect(split.preTokenize(text: "12 345 6789") == ["12 ", "345", " ", "678", "9"])
            #expect(split.preTokenize(text: "١٢٣45") == ["١٢٣45"])
            #expect(split.preTokenize(text: "123") == ["123"])
            #expect(split.preTokenize(text: "") == [])
        }
        let single = try SplitPreTokenizer(
            config: ["pattern": ["Regex": #"[0-9]"#], "behavior": "Isolated", "invert": false])
        #expect(single.preTokenize(text: "١٢٣45") == ["١٢٣", "4", "5"])
        #expect(SplitPreTokenizer.asciiDigitGroup(of: "[0-9]") == 1)
        #expect(SplitPreTokenizer.asciiDigitGroup(of: "[0-9]{12}") == 12)
        #expect(SplitPreTokenizer.asciiDigitGroup(of: "[0-9]+") == nil)
        #expect(SplitPreTokenizer.asciiDigitGroup(of: "[0-9]{1,3}") == nil)
        #expect(SplitPreTokenizer.asciiDigitGroup(of: "[0-9][0-9]{3}") == nil)
    }

    @Test("Escapes, nested classes and complements survive translation")
    func translation() {
        let members = OnigurumaDialect.wordMembers
        let pua = OnigurumaDialect.applePrivateUse
        #expect(OnigurumaDialect.translate(#"abc"#) == #"abc"#)
        #expect(OnigurumaDialect.translate(#"\\w"#) == #"\\w"#)  // an escaped backslash, then a literal `w`
        #expect(OnigurumaDialect.translate(#"\w"#) == "[[\(members)]--\(pua)]")
        #expect(OnigurumaDialect.translate(#"[\w]"#) == "[[[\(members)]--\(pua)]]")
        #expect(OnigurumaDialect.translate(#"[\w-]"#) == "[[[\(members)]--\(pua)]\\-]")
        #expect(OnigurumaDialect.translate(#"\W"#) == "[[^\(members)]\(pua)]")
        #expect(OnigurumaDialect.translate(#"\d\s"#) == #"\d\s"#)
        #expect(OnigurumaDialect.translate(#"\p{L}"#) == "[\\p{L}--\(pua)]")
        #expect(OnigurumaDialect.translate(#"\P{L}"#) == "[\\P{L}\(pua)]")
        #expect(OnigurumaDialect.translate(#"\p{^L}"#) == "[\\P{L}\(pua)]")
        #expect(OnigurumaDialect.translate(#"\pL"#) == "[\\p{L}--\(pua)]")
        #expect(OnigurumaDialect.translate(#"\p{Co}"#) == "[\\p{Co}\(pua)]")
        #expect(OnigurumaDialect.translate(#"\P{Co}"#) == "[\\P{Co}--\(pua)]")
        #expect(OnigurumaDialect.translate(#"[^\r\n\p{L}\p{N}]"#) == "[^\\r\\n[\\p{L}--\(pua)][\\p{N}--\(pua)]]")
    }

    @Test("Apple private-use code points keep their Unicode category")
    func applePrivateUse() throws {
        // U+F8E1 is `\p{S}` and U+F8C1 is `\p{L}` for Apple's ICU; Unicode says `Co`.
        let pattern = #"[\p{P}\p{S}]+|[\p{L}\p{M}]+|\p{N}+|\w+|[^\s\p{L}\p{N}\p{P}\p{S}]+"#
        let split = try SplitPreTokenizer(
            config: ["pattern": ["Regex": Config(pattern)], "behavior": "Isolated", "invert": false])
        #expect(split.preTokenize(text: "a\u{F8E1}*\u{F8C1}b") == ["a", "\u{F8E1}", "*", "\u{F8C1}", "b"])
        let other = try SplitPreTokenizer(
            config: ["pattern": ["Regex": #"\p{Co}+"#], "behavior": "Isolated", "invert": false])
        #expect(other.preTokenize(text: "a\u{F8E1}\u{E000}b") == ["a", "\u{F8E1}\u{E000}", "b"])
    }

    @Test("Every translated pattern still compiles")
    func compiles() throws {
        for pattern in [#"\w+"#, #"[\w-]+"#, #"[^\w\s]+"#, #"\W"#, #"(?i:\w)"#, #"^\w$"#, #"[\W]"#] {
            #expect(throws: Never.self) { try compileRegex(pattern, component: "test") }
        }
    }
}
