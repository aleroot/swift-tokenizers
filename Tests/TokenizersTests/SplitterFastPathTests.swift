// The chunked ASCII fast paths of the byte splitters must agree with the scalar definition
// on every alignment: pieces crossing chunk boundaries, boundaries at lane 0 / lane 15, and
// non-ASCII bytes forcing the scalar path mid-text.

import Testing

@testable import Tokenizers

@Suite("Splitter fast paths")
struct SplitterFastPathTests {
    /// Words, punctuation, whitespace runs, digits, underscores and non-ASCII letters, arranged
    /// so that every lane position sees every class over the set of prefixes.
    static let texts: [String] = {
        let base =
            "Hello, world!  This is a test_case of 12 items; café naïve über 東京 (yes) — done.\tTab\nNew 'quote' a.b.c   x"
        var texts: [String] = []
        for shift in 0..<20 {
            texts.append(String(repeating: "a", count: shift) + base)
            texts.append(String(repeating: " ", count: shift) + base)
            texts.append(String(repeating: "!", count: shift) + base)
        }
        texts.append("")
        texts.append(String(repeating: " ", count: 40))
        texts.append(String(repeating: "x", count: 40))
        texts.append(String(repeating: "!", count: 40))
        texts.append(String(repeating: "a!", count: 20))
        texts.append(String(repeating: "a ", count: 20))
        texts.append(String(repeating: "é", count: 20) + " abc")
        return texts
    }()

    static func referenceBert(_ text: String) -> [String] {
        var pieces: [String] = []
        var current = ""
        for scalar in text.unicodeScalars {
            let flags = ScalarClassifier.flags(value: scalar.value)
            if flags & ScalarFlags.whitespace != 0 {
                if !current.isEmpty { pieces.append(current) }
                current = ""
            } else if flags & ScalarFlags.punctuation != 0 {
                if !current.isEmpty { pieces.append(current) }
                pieces.append(String(scalar))
                current = ""
            } else {
                current.unicodeScalars.append(scalar)
            }
        }
        if !current.isEmpty { pieces.append(current) }
        return pieces
    }

    static func referenceWhitespace(_ text: String, splitWords: Bool) -> [String] {
        var pieces: [String] = []
        var current = ""
        var currentIsWord = false
        for scalar in text.unicodeScalars {
            let isWhitespace = ScalarClassifier.flags(value: scalar.value) & ScalarFlags.whitespace != 0
            if isWhitespace {
                if !current.isEmpty { pieces.append(current) }
                current = ""
                continue
            }
            let isWord = WhitespacePreTokenizer.isWord(scalar.value)
            if splitWords, !current.isEmpty, isWord != currentIsWord {
                pieces.append(current)
                current = ""
            }
            current.unicodeScalars.append(scalar)
            currentIsWord = isWord
        }
        if !current.isEmpty { pieces.append(current) }
        return pieces
    }

    @Test("BertPreTokenizer")
    func bert() throws {
        let splitter = try BertPreTokenizer(config: Config([String: Config]()))
        for text in Self.texts {
            #expect(
                splitter.preTokenize(text: text) == Self.referenceBert(text), Comment(rawValue: text.debugDescription))
        }
    }

    @Test("WhitespaceSplit and Whitespace")
    func whitespace() throws {
        let split = try WhitespacePreTokenizer(config: Config(["type": Config("WhitespaceSplit")]))
        let words = try WhitespacePreTokenizer(config: Config(["type": Config("Whitespace")]))
        for text in Self.texts {
            #expect(
                split.preTokenize(text: text) == Self.referenceWhitespace(text, splitWords: false),
                Comment(rawValue: text.debugDescription))
            #expect(
                words.preTokenize(text: text) == Self.referenceWhitespace(text, splitWords: true),
                Comment(rawValue: text.debugDescription))
        }
    }
}
