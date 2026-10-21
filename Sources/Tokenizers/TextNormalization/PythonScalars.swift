import Foundation

/// Python `str` and `re` semantics over Unicode scalars, the unit Python strings index by. Each
/// `re.sub` of the Whisper normalizers is one `substitute` pass.
extension [Unicode.Scalar] {
    /// The scalars of `text`, decoded from its UTF-8 in one pass.
    init(scalarsOf text: String) {
        self = []
        reserveCapacity(text.utf8.count)
        for scalar in text.unicodeScalars { append(scalar) }
    }

    /// `\w`.
    @inline(__always)
    func isWord(at index: Int) -> Bool { indices.contains(index) && PythonUnicode.isWord(self[index]) }

    /// `\b`.
    @inline(__always)
    func isBoundary(at index: Int) -> Bool { isWord(at: index - 1) != isWord(at: index) }

    /// `\s`.
    @inline(__always)
    func isSpace(at index: Int) -> Bool { indices.contains(index) && PythonText.isSpace(self[index]) }

    /// `[0-9]`.
    @inline(__always)
    func isDigit(at index: Int) -> Bool { indices.contains(index) && PythonUnicode.isDigit(self[index]) }

    @inline(__always)
    func contains(_ literal: Self, at index: Int) -> Bool {
        index >= 0 && index + literal.count <= count && self[index..<index + literal.count].elementsEqual(literal)
    }

    @inline(__always)
    func contains(_ scalar: Unicode.Scalar, at index: Int) -> Bool { indices.contains(index) && self[index] == scalar }

    /// The end of the run of scalars from `index` that satisfy `predicate`.
    @inline(__always)
    func end(from index: Int, while predicate: (Unicode.Scalar) -> Bool) -> Int {
        var end = index
        while end < count, predicate(self[end]) { end += 1 }
        return end
    }

    /// `re.sub`: tries `match` at each index left to right, from `start` where no match can begin
    /// before. A match replaces `length` scalars and scanning resumes after them; the text is
    /// copied only once something matches.
    @inline(__always) @discardableResult
    mutating func substitute(from start: Int = 0, _ match: (Self, Int) -> (length: Int, replacement: Self)?) -> Bool {
        var output: Self = []
        var copied = 0
        var index = start
        while index < count {
            guard let (length, replacement) = match(self, index) else {
                index += 1
                continue
            }
            if output.isEmpty { output.reserveCapacity(count) }
            output += self[copied..<index]
            output += replacement
            index += length
            copied = index
        }
        guard copied > 0 else { return false }
        output += self[copied...]
        self = output
        return true
    }

    fileprivate mutating func removeBracketedSpans() {
        removeSpans(opening: { $0 == "<" || $0 == "[" }, closing: { $0 == ">" || $0 == "]" }, allowsEmpty: true)
        removeSpans(opening: { $0 == "(" }, closing: { $0 == ")" }, allowsEmpty: false)
    }

    @inline(__always)
    private mutating func removeSpans(
        opening: (Unicode.Scalar) -> Bool, closing: (Unicode.Scalar) -> Bool, allowsEmpty: Bool
    ) {
        guard var next = firstIndex(where: opening) else { return }
        substitute { text, index in
            // Once no closing scalar follows an opening one, none follows any later one either.
            guard index >= next, opening(text[index]) else { return nil }
            guard let close = text[(index + 1)...].firstIndex(where: closing) else {
                next = text.count
                return nil
            }
            return allowsEmpty || close > index + 1 ? (close - index + 1, []) : nil
        }
    }

    /// `re.sub(r"\s+", " ", s)`.
    mutating func collapseWhitespace() {
        substitute { text, index in
            guard PythonText.isSpace(text[index]) else { return nil }
            let end = text.end(from: index, while: PythonText.isSpace)
            return end == index + 1 && text[index] == " " ? nil : (end - index, [" "])
        }
    }

    /// `str.split()`.
    func words() -> [String] {
        split(whereSeparator: PythonText.isSpace).map(\.string)
    }
}

extension Collection<Unicode.Scalar> {
    /// The scalars as a string, encoded directly.
    var string: String {
        String(unsafeUninitializedCapacity: count * 4) { output in
            var written = 0
            for scalar in self {
                let value = scalar.value
                if value < 0x80 {
                    output[written] = UInt8(value)
                    written += 1
                } else if value < 0x800 {
                    output[written] = UInt8(0xC0 | value >> 6)
                    output[written + 1] = UInt8(0x80 | value & 0x3F)
                    written += 2
                } else if value < 0x10000 {
                    output[written] = UInt8(0xE0 | value >> 12)
                    output[written + 1] = UInt8(0x80 | value >> 6 & 0x3F)
                    output[written + 2] = UInt8(0x80 | value & 0x3F)
                    written += 3
                } else {
                    output[written] = UInt8(0xF0 | value >> 18)
                    output[written + 1] = UInt8(0x80 | value >> 12 & 0x3F)
                    output[written + 2] = UInt8(0x80 | value >> 6 & 0x3F)
                    output[written + 3] = UInt8(0x80 | value & 0x3F)
                    written += 4
                }
            }
            return written
        }
    }
}

/// The text transformations the Whisper normalizers share.
enum PythonUnicode {
    /// What the normalizers ask of a scalar, read from the running Unicode data. The data
    /// matches Python 3.12 (Unicode 15) on every scalar assigned by then, except the one
    /// patched below.
    private enum Flag {
        /// `\w`: `str.isalnum()` or `_`.
        static let word: UInt8 = 1
        /// General category `M*`, `S*` or `P*`.
        static let markSymbolOrPunctuation: UInt8 = 2
        /// General category `Mn`.
        static let nonspacingMark: UInt8 = 4
        static let changesWhenLowercased: UInt8 = 8
    }

    /// ``Flag``s of the Basic Multilingual Plane (64 KiB), built on first use.
    private static let bmpFlags: [UInt8] = (0..<0x10000).map { Unicode.Scalar($0).map(computeFlags) ?? 0 }

    @inline(__always)
    private static func flags(_ scalar: Unicode.Scalar) -> UInt8 {
        scalar.value < 0x10000 ? bmpFlags[Int(scalar.value)] : computeFlags(scalar)
    }

    private static func computeFlags(_ scalar: Unicode.Scalar) -> UInt8 {
        let properties = scalar.properties
        // U+1171E was a nonspacing mark until Unicode 15.
        let category = scalar.value == 0x1171E ? .nonspacingMark : properties.generalCategory
        var flags: UInt8 = properties.changesWhenLowercased ? Flag.changesWhenLowercased : 0
        switch category {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter:
            flags |= Flag.word
        case .nonspacingMark:
            flags |= Flag.markSymbolOrPunctuation | Flag.nonspacingMark
        case .spacingMark, .enclosingMark, .mathSymbol, .currencySymbol, .modifierSymbol, .otherSymbol,
            .dashPunctuation, .openPunctuation, .closePunctuation, .initialPunctuation, .finalPunctuation,
            .otherPunctuation:
            flags |= Flag.markSymbolOrPunctuation
        case .connectorPunctuation:
            flags |= Flag.markSymbolOrPunctuation | (scalar == "_" ? Flag.word : 0)
        default:
            if properties.numericType != nil { flags |= Flag.word }
        }
        return flags
    }

    /// `\w`.
    @inline(__always)
    static func isWord(_ scalar: Unicode.Scalar) -> Bool { flags(scalar) & Flag.word != 0 }

    /// `[0-9]`.
    @inline(__always)
    static func isDigit(_ scalar: Unicode.Scalar) -> Bool { ("0"..."9").contains(scalar) }

    /// `\d`: a Unicode decimal digit.
    @inline(__always)
    static func isDecimal(_ scalar: Unicode.Scalar) -> Bool {
        scalar.isASCII ? isDigit(scalar) : scalar.properties.generalCategory == .decimalNumber
    }

    /// Whether `str.lower()` changes `scalar`.
    @inline(__always)
    static func changesWhenLowercased(_ scalar: Unicode.Scalar) -> Bool {
        flags(scalar) & Flag.changesWhenLowercased != 0
    }

    /// Removes spans in brackets and parentheses: `re.sub(r"[<\[][^>\]]*[>\]]", "", s)` then
    /// `re.sub(r"\(([^)]+?)\)", "", s)`.
    static func removingBracketedSpans(_ text: String) -> String {
        var text = text
        let openings: [UInt8] = [UInt8(ascii: "<"), UInt8(ascii: "["), UInt8(ascii: "(")]
        guard text.withUTF8({ $0.contains(where: openings.contains) }) else { return text }
        var scalars = [Unicode.Scalar](scalarsOf: text)
        scalars.removeBracketedSpans()
        return scalars.string
    }

    /// `str.lower()`. `String.lowercased()` maps the same characters, but has no final sigma rule.
    static func lowercased(_ text: String) -> String {
        guard text.unicodeScalars.contains("Σ") else { return text.lowercased() }
        let scalars = [Unicode.Scalar](scalarsOf: text)
        var output = ""
        var start = 0
        for index in scalars.indices where scalars[index] == "Σ" {
            output += scalars[start..<index].string.lowercased()
            output.unicodeScalars.append(isFinalSigma(scalars, at: index) ? "ς" : "σ")
            start = index + 1
        }
        return output + scalars[start...].string.lowercased()
    }

    /// Preceded by a cased letter and not followed by one, skipping case-ignorable scalars.
    private static func isFinalSigma(_ scalars: [Unicode.Scalar], at index: Int) -> Bool {
        let before = scalars[..<index].last { !$0.properties.isCaseIgnorable }
        let after = scalars[(index + 1)...].first { !$0.properties.isCaseIgnorable }
        return before?.properties.isCased == true && after?.properties.isCased != true
    }

    /// Letters NFKD leaves whole that `remove_symbols_and_diacritics` still spells in ASCII.
    private static let additionalDiacritics: [Unicode.Scalar: String] = [
        "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ß": "ss", "ẞ": "SS",
        "đ": "d", "Đ": "D", "ð": "d", "Ð": "D", "þ": "th", "Þ": "th", "ł": "l", "Ł": "L",
    ]

    /// `remove_symbols(s)`, or `remove_symbols_and_diacritics(s, keep)` when `removingDiacritics`:
    /// marks, symbols and punctuation become spaces after NFKC, or after NFKD with nonspacing
    /// marks dropped.
    static func removingSymbols(
        _ text: String, removingDiacritics: Bool, keep: [Unicode.Scalar] = []
    ) -> [Unicode.Scalar] {
        var text = text
        let mask = removingDiacritics ? UnicodeNormalization.Property.notNFKD : UnicodeNormalization.Property.notNFKC
        if !text.withUTF8({ UnicodeNormalization.isNormalized($0, mask: mask) }) {
            // ICU: CFStringNormalize drops the plane of a supplementary scalar that composes
            // with a following mark (U+103B9 U+0301 → U+03AF on macOS 26).
            text = text.applyingTransform(StringTransform(removingDiacritics ? "NFKD" : "NFKC"), reverse: false)!
        }
        var output: [Unicode.Scalar] = []
        output.reserveCapacity(text.utf8.count)
        for scalar in text.unicodeScalars {
            if removingDiacritics, !scalar.isASCII, let spelled = additionalDiacritics[scalar] {
                output += spelled.unicodeScalars
                continue
            }
            let flags = flags(scalar)
            if removingDiacritics, flags & Flag.nonspacingMark != 0 { continue }
            output.append(flags & Flag.markSymbolOrPunctuation != 0 && !keep.contains(scalar) ? " " : scalar)
        }
        return output
    }
}
