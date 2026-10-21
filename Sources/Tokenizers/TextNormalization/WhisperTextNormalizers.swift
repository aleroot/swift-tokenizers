import Foundation

/// Whisper's language-independent text normalizer, which puts transcripts in a canonical form
/// before computing word or character error rates. It matches `transformers`'
/// `BasicTextNormalizer` and `WhisperTokenizer.basic_normalize`.
///
/// It lowercases, removes spans in brackets and parentheses, turns marks, symbols and
/// punctuation into spaces and collapses whitespace:
///
/// ```swift
/// let normalize = BasicTextNormalizer()
/// normalize("Hello, World! [music]")  // "hello world "
/// ```
public struct BasicTextNormalizer: Sendable {
    /// Whether to decompose characters and drop their diacritics, rather than keep them.
    public var removesDiacritics: Bool
    /// Whether to separate every grapheme cluster with a space, for character error rates.
    public var splitsLetters: Bool

    public init(removeDiacritics: Bool = false, splitLetters: Bool = false) {
        removesDiacritics = removeDiacritics
        splitsLetters = splitLetters
    }

    public func callAsFunction(_ text: String) -> String {
        let text = PythonUnicode.removingBracketedSpans(PythonUnicode.lowercased(text))
        var scalars = PythonUnicode.removingSymbols(text, removingDiacritics: removesDiacritics)
        // Compatibility mappings can produce capitals ("ℍ" → "H").
        if splitsLetters || scalars.contains(where: PythonUnicode.changesWhenLowercased) {
            var text = PythonUnicode.lowercased(scalars.string)
            if splitsLetters { text = text.map(String.init).joined(separator: " ") }
            scalars = [Unicode.Scalar](scalarsOf: text)
        }
        scalars.collapseWhitespace()
        return scalars.string
    }
}

/// Whisper's English text normalizer, the one English speech recognition benchmarks such as
/// the Open ASR Leaderboard score with. It matches `transformers`' `EnglishTextNormalizer` and
/// `WhisperTokenizer.normalize`.
///
/// On top of ``BasicTextNormalizer`` it drops hesitations ("um", "hmm"), expands contractions
/// and titles ("won't" → "will not", "dr" → "doctor"), writes numbers and currencies in digits
/// ("twenty five dollars" → "$25") and applies British to American spellings:
///
/// ```swift
/// let normalize = try EnglishTextNormalizer(contentsOf: modelFolder.appending(path: "normalizer.json"))
/// normalize("Mr. Smith won't pay twenty five dollars.")  // "mister smith will not pay $25"
/// ```
public struct EnglishTextNormalizer: Sendable {
    /// British to American spellings, applied word by word.
    public let spellings: [String: String]

    private static let numbers = EnglishNumberNormalizer()
    private static let hesitations: Set<String> = ["hmm", "mm", "mhm", "mmm", "uh", "um"]
    private static let kept: [Unicode.Scalar] = [".", "%", "$", "¢", "€", "£"]

    /// Contractions and abbreviated titles, rewritten in order; each must end at a word
    /// boundary and, when `isWord`, start at one.
    private static let replacements: [(pattern: [Unicode.Scalar], isWord: Bool, replacement: [Unicode.Scalar])] =
        [
            ("won't", "will not"), ("can't", "can not"), ("let's", "let us"), ("ain't", "aint"),
            ("y'all", "you all"), ("wanna", "want to"), ("gotta", "got to"), ("gonna", "going to"),
            ("i'ma", "i am going to"), ("imma", "i am going to"), ("woulda", "would have"),
            ("coulda", "could have"), ("shoulda", "should have"), ("ma'am", "madam"), ("mr", "mister "),
            ("mrs", "missus "), ("st", "saint "), ("dr", "doctor "), ("prof", "professor "), ("capt", "captain "),
            ("gov", "governor "), ("ald", "alderman "), ("gen", "general "), ("sen", "senator "),
            ("rep", "representative "), ("pres", "president "), ("rev", "reverend "), ("hon", "honorable "),
            ("asst", "assistant "), ("assoc", "associate "), ("lt", "lieutenant "), ("col", "colonel "),
            ("jr", "junior "), ("sr", "senior "), ("esq", "esquire "),
        ].map { (Array($0.unicodeScalars), true, Array($1.unicodeScalars)) }
        + [
            ("'d been", " had been"), ("'s been", " has been"), ("'d gone", " had gone"), ("'s gone", " has gone"),
            ("'d done", " had done"), ("'s got", " has got"), ("n't", " not"), ("'re", " are"), ("'s", " is"),
            ("'d", " would"), ("'ll", " will"), ("'t", " not"), ("'ve", " have"), ("'m", " am"),
        ].map { (Array($0.unicodeScalars), false, Array($1.unicodeScalars)) }

    /// - Parameter spellings: The British to American mapping, the `normalizer.json` of Whisper
    ///   model repositories.
    public init(spellings: [String: String]) {
        self.spellings = spellings
    }

    /// Reads the spelling mapping from a `normalizer.json` file.
    public init(contentsOf url: URL) throws {
        self.init(spellings: try JSONDecoder().decode([String: String].self, from: Data(contentsOf: url)))
    }

    public func callAsFunction(_ text: String) -> String {
        var text = [Unicode.Scalar](scalarsOf: PythonUnicode.removingBracketedSpans(PythonUnicode.lowercased(text)))
        text.substitute { text, index in
            guard ["h", "m", "u"].contains(text[index]), text.isBoundary(at: index) else { return nil }
            let end = text.end(from: index, while: PythonUnicode.isWord)
            return end - index <= 3 && Self.hesitations.contains(text[index..<end].string) ? (end - index, []) : nil
        }
        if text.contains("'") {
            // No space before an apostrophe.
            text.substitute { text, index in
                guard PythonText.isSpace(text[index]) else { return nil }
                let end = text.end(from: index, while: PythonText.isSpace)
                return text.contains("'", at: end) ? (end - index + 1, ["'"]) : nil
            }
        }
        Self.expandContractions(&text)
        // Commas between digits go; periods stay only before digits.
        if text.contains(",") {
            text.substitute { text, index in
                text.contains(",", at: index + 1) && PythonUnicode.isDecimal(text[index]) && index + 2 < text.count
                    && PythonUnicode.isDecimal(text[index + 2]) ? (3, [text[index], text[index + 2]]) : nil
            }
        }
        text.substitute { text, index in
            guard text[index] == ".", !text.isDigit(at: index + 1) else { return nil }
            return index + 1 < text.count ? (2, [" ", text[index + 1]]) : (1, [" "])
        }
        text = PythonUnicode.removingSymbols(text.string, removingDiacritics: true, keep: Self.kept)
        text = [Unicode.Scalar](scalarsOf: Self.numbers(text).map { spellings[$0] ?? $0 }.joined(separator: " "))
        // Currency signs and periods not before a digit, percent signs not after one.
        text.substitute { text, index in
            guard text[index] != "%", Self.kept.contains(text[index]), index + 1 < text.count,
                !PythonUnicode.isDigit(text[index + 1])
            else { return nil }
            return (2, [" ", text[index + 1]])
        }
        if text.contains("%") {
            text.substitute { text, index in
                text.contains("%", at: index + 1) && !PythonUnicode.isDigit(text[index]) ? (2, [text[index], " "]) : nil
            }
        }
        text.collapseWhitespace()
        return text.string
    }

    /// ``replacements`` by the ASCII value of their first scalar.
    private static let replacementsByFirstScalar: [[Int]] = {
        var table = [[Int]](repeating: [], count: 128)
        for (index, replacement) in replacements.enumerated() { table[Int(replacement.pattern[0].value)].append(index) }
        return table
    }()

    private static func expandContractions(_ text: inout [Unicode.Scalar]) {
        // One scan finds the patterns that occur; a rewrite can create later ones ("gotta" →
        // "got to" before "'s got"), so the text is scanned again after each rewrite.
        var pending = occurringReplacements(in: text)
        while pending != 0 {
            let next = pending.trailingZeroBitCount
            let (pattern, isWord, replacement) = replacements[next]
            let rewritten = text.substitute { text, index in
                text.contains(pattern, at: index) && (!isWord || text.isBoundary(at: index))
                    && text.isBoundary(at: index + pattern.count) ? (pattern.count, replacement) : nil
            }
            let later = ~UInt64(0) << (next + 1)
            pending = (rewritten ? occurringReplacements(in: text) : pending) & later
        }
    }

    /// Bit `i` is set when the pattern of `replacements[i]` occurs in `text`.
    private static func occurringReplacements(in text: [Unicode.Scalar]) -> UInt64 {
        var occurring: UInt64 = 0
        for index in text.indices where text[index].isASCII {
            for candidate in replacementsByFirstScalar[Int(text[index].value)]
            where text.contains(replacements[candidate].pattern, at: index) {
                occurring |= 1 << candidate
            }
        }
        return occurring
    }
}
