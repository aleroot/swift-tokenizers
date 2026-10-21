/// Whisper's `EnglishNumberNormalizer`: spelled-out numbers become digits ("twenty one" → "21",
/// "one oh one" → "101", "$20 million" → "$20000000"), keeping suffixes ("1960s", "32nd") and
/// turning currency words into symbols. Values are decimal strings, since Python's integers
/// are unbounded ("one decillion").
struct EnglishNumberNormalizer {
    private enum Word {
        case zero
        /// One to nineteen; with a suffix, "fifth" or "sixes".
        case one(Int, suffix: String?)
        /// Twenty to ninety; with a suffix, "twentieth" or "twenties".
        case ten(Int, suffix: String?)
        /// A power of ten; with a suffix, "thousandth" or "millions".
        case multiplier(Int, suffix: String?)
        case precedingPrefix(String), followingPrefix(String)
        case per, percent
        case special

        var isZero: Bool { if case .zero = self { true } else { false } }
        var isOne: Bool { if case .one(_, nil) = self { true } else { false } }
        var isTen: Bool { if case .ten(_, nil) = self { true } else { false } }
        var isMultiplier: Bool { if case .multiplier(_, nil) = self { true } else { false } }
    }

    private static let words: [String: Word] = {
        var words: [String: Word] = ["o": .zero, "oh": .zero, "zero": .zero]
        let ones = [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
            "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        ]
        for (offset, name) in ones.enumerated() {
            let value = offset + 1
            words[name] = .one(value, suffix: nil)
            words[name == "six" ? "sixes" : name + "s"] = .one(value, suffix: "s")
            if value > 3, value != 5, value != 12 {
                words[name + (name.hasSuffix("t") ? "h" : "th")] = .one(value, suffix: "th")
            }
        }
        for (name, value, suffix) in [
            ("zeroth", 0, "th"), ("first", 1, "st"), ("second", 2, "nd"), ("third", 3, "rd"), ("fifth", 5, "th"),
            ("twelfth", 12, "th"),
        ] {
            words[name] = .one(value, suffix: suffix)
        }
        for (offset, name) in ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
            .enumerated()
        {
            let value = (offset + 2) * 10
            words[name] = .ten(value, suffix: nil)
            words[name.replacingOccurrences(of: "y", with: "ies")] = .ten(value, suffix: "s")
            words[name.replacingOccurrences(of: "y", with: "ieth")] = .ten(value, suffix: "th")
        }
        for (name, power) in [
            ("hundred", 2), ("thousand", 3), ("million", 6), ("billion", 9), ("trillion", 12), ("quadrillion", 15),
            ("quintillion", 18), ("sextillion", 21), ("septillion", 24), ("octillion", 27), ("nonillion", 30),
            ("decillion", 33),
        ] {
            words[name] = .multiplier(power, suffix: nil)
            words[name + "s"] = .multiplier(power, suffix: "s")
            words[name + "th"] = .multiplier(power, suffix: "th")
        }
        for (name, symbol) in [("minus", "-"), ("negative", "-"), ("plus", "+"), ("positive", "+")] {
            words[name] = .precedingPrefix(symbol)
        }
        for (name, symbol) in [("pound", "£"), ("euro", "€"), ("dollar", "$"), ("cent", "¢")] {
            words[name] = .followingPrefix(symbol)
            words[name + "s"] = .followingPrefix(symbol)
        }
        words["per"] = .per
        words["percent"] = .percent
        for name in ["and", "double", "triple", "point"] { words[name] = .special }
        return words
    }()

    private static let prefixes: Set<Unicode.Scalar> = ["-", "+", "£", "€", "$", "¢"]

    /// The words of the normalized text.
    func callAsFunction(_ text: [Unicode.Scalar]) -> [String] {
        let words = process(Self.preprocess(text).words())
        // Only currency signs and ones have anything to postprocess.
        guard words.contains(where: { $0.unicodeScalars.contains { $0 == "1" || Self.currencies.contains($0) } }) else {
            return words
        }
        var text = [Unicode.Scalar](scalarsOf: words.joined(separator: " "))
        Self.postprocess(&text)
        return text.words()
    }

    // MARK: - Words

    /// A value being read: an integer as canonical decimal digits, or digits joined as text
    /// ("0" for "oh", "1.5", "192.168").
    private enum Value {
        case integer(String)
        case text(String)

        var string: String {
            switch self {
            case let .integer(digits), let .text(digits): digits
            }
        }
    }

    private func process(_ words: [String]) -> [String] {
        let kinds = words.map { Self.words[$0] }
        var output: [String] = []
        var prefix = ""
        var value: Value?
        var skip = false

        func emit(_ result: String) {
            output.append(prefix + result)
            prefix = ""
            value = nil
        }
        func read(_ number: Value, suffix: String?) {
            if let suffix { emit(number.string + suffix) } else { value = number }
        }
        /// Python's `str(value or "")`.
        var valueOrEmpty: String {
            switch value {
            case nil, .integer("0"): ""
            case let value?: value.string
            }
        }

        for (index, current) in words.enumerated() {
            if skip {
                skip = false
                continue
            }
            let previous = index > 0 ? kinds[index - 1] : nil
            let next = index + 1 < words.count ? words[index + 1] : nil
            let nextKind = index + 1 < words.count ? kinds[index + 1] : nil
            let nextIsNumeric = next.map(Self.isNumeric) ?? false
            let first = current.unicodeScalars.first!
            let hasPrefix = Self.prefixes.contains(first)
            let bare = hasPrefix ? String(current.unicodeScalars.dropFirst()) : current

            if Self.isNumeric(bare) {
                if let pending = value {
                    if case let .text(text) = pending, text.unicodeScalars.last == "." {
                        // Decimals and IP address components.
                        value = .text(text + current)
                        continue
                    }
                    emit(pending.string)
                }
                if hasPrefix { prefix = String(first) }
                let fraction = Self.fraction(bare)!
                value = fraction.decimals.allSatisfy { $0 == "0" } ? .integer(fraction.digits) : .text(bare)
                continue
            }
            guard let kind = kinds[index] else {
                if let pending = value { emit(pending.string) }
                emit(current)
                continue
            }
            switch kind {
            case .zero:
                value = .text(valueOrEmpty + "0")

            // A suffixed word reads like its plain form, then ends the number: "twenty fifth".
            case let .one(ones, suffix):
                let number: Value =
                    switch value {
                    case nil: .integer(String(ones))
                    case let .text(text)?: .text(Self.appending(ones, to: text, after: previous))
                    case let .integer(digits)? where previous?.isOne == true: .text(digits + String(ones))
                    case let .integer(digits)?: Self.adding(ones, to: digits)
                    }
                read(number, suffix: suffix)

            case let .ten(tens, suffix):
                let number: Value =
                    switch value {
                    case nil: .integer(String(tens))
                    case let .text(text)?: .text(text + String(tens))
                    case let .integer(digits)?: Self.adding(tens, to: digits)
                    }
                read(number, suffix: suffix)

            case let .multiplier(power, suffix):
                switch value {
                case nil:
                    read(.integer(Self.power(power)), suffix: suffix)
                case let .text(text)?:
                    if let product = Self.fraction(text).flatMap({ Self.scaled($0, by: power) }) {
                        read(.integer(product), suffix: suffix)
                    } else {
                        emit(text)
                        read(.integer(Self.power(power)), suffix: suffix)
                    }
                case let .integer(digits)?:
                    read(.integer(Self.multiplied(digits, by: power)), suffix: suffix)
                }

            case let .precedingPrefix(symbol):
                if let pending = value { emit(pending.string) }
                if nextKind != nil || nextIsNumeric { prefix = symbol } else { emit(current) }

            case let .followingPrefix(symbol):
                if let pending = value {
                    prefix = symbol
                    emit(pending.string)
                } else {
                    emit(current)
                }

            case .per, .percent:
                guard let pending = value else {
                    emit(current)
                    break
                }
                if case .percent = kind {
                    emit(pending.string + "%")
                } else if next == "cent" {
                    emit(pending.string + "%")
                    skip = true
                } else {
                    emit(pending.string)
                    emit(current)
                }

            case .special:
                if nextKind == nil, !nextIsNumeric {
                    if let pending = value { emit(pending.string) }
                    emit(current)
                } else if current == "and" {
                    // "and" after hundreds, thousands, ... joins the number.
                    guard previous?.isMultiplier != true else { break }
                    if let pending = value { emit(pending.string) }
                    emit(current)
                } else if current == "double" || current == "triple" {
                    if let nextKind, nextKind.isOne || nextKind.isZero {
                        let digit = if case let .one(ones, _) = nextKind { String(ones) } else { "0" }
                        value = .text(valueOrEmpty + String(repeating: digit, count: current == "double" ? 2 : 3))
                        skip = true
                    } else {
                        if let pending = value { emit(pending.string) }
                        emit(current)
                    }
                } else if nextIsNumeric || nextKind.map({ $0.isZero || $0.isOne || $0.isTen }) == true {
                    // "point" before a digit word; otherwise it is dropped.
                    value = .text(valueOrEmpty + ".")
                }
            }
        }
        if let pending = value { emit(pending.string) }
        return output
    }

    /// `ones` after text, or after an integer read from single digits: "oh five", "one two".
    /// After a ten the digit replaces its trailing zero.
    private static func appending(_ ones: Int, to text: String, after previous: Word?) -> String {
        let replacesZero = previous?.isTen == true && ones < 10
        return (replacesZero ? String(text.unicodeScalars.dropLast()) : text) + String(ones)
    }

    // MARK: - Decimal arithmetic

    /// `digits` + `addend` (below 100) when the digits it fills are zero, as in "twenty one" or
    /// "two hundred twelve"; otherwise the digits side by side, as in "nineteen eighty".
    private static func adding(_ addend: Int, to digits: String) -> Value {
        let fills = addend < 10 ? digits.hasSuffix("0") : digits == "0" || digits.hasSuffix("00")
        return fills ? .integer(sum(digits, String(addend))) : .text(digits + String(addend))
    }

    /// `^\d+(\.\d+)?$`.
    private static func isNumeric(_ word: String) -> Bool {
        var digits = 0
        var point: Int?
        for scalar in word.unicodeScalars {
            if PythonUnicode.isDecimal(scalar) {
                digits += 1
            } else if scalar == ".", point == nil, digits > 0 {
                point = digits
            } else {
                return false
            }
        }
        return digits > (point ?? 0)
    }

    /// `Fraction(text)` for a decimal number: canonical integer digits and the decimal digits
    /// in ASCII, or nil where Python raises `ValueError`.
    private static func fraction(_ text: String) -> (digits: String, decimals: String)? {
        let parts = text.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.allSatisfy({ $0.unicodeScalars.allSatisfy(PythonUnicode.isDecimal) }),
            !parts[0].isEmpty || parts.count == 2 && !parts[1].isEmpty
        else { return nil }
        let ascii = parts.map { part in
            part.unicodeScalars.map { Unicode.Scalar(0x30 + UInt8($0.properties.numericValue!)) }.string
        }
        return (canonical(ascii[0]), parts.count == 2 ? ascii[1] : "")
    }

    /// The fraction times 10^`power` when that is an integer.
    private static func scaled(_ fraction: (digits: String, decimals: String), by power: Int) -> String? {
        let decimals = fraction.decimals.reversed().drop { $0 == "0" }.reversed()
        guard decimals.count <= power else { return nil }
        return canonical(fraction.digits + String(decimals) + String(repeating: "0", count: power - decimals.count))
    }

    /// Python's `value // 1000 * 1000 + value % 1000 * multiplier`: "two thousand five hundred".
    private static func multiplied(_ digits: String, by power: Int) -> String {
        let before = digits.count > 3 ? digits.dropLast(3) + "000" : "0"
        let residual = canonical(String(digits.suffix(3)))
        return sum(before, residual == "0" ? "0" : residual + String(repeating: "0", count: power))
    }

    private static func power(_ power: Int) -> String { "1" + String(repeating: "0", count: power) }

    private static func canonical(_ digits: String) -> String {
        let trimmed = digits.drop { $0 == "0" }
        return trimmed.isEmpty ? "0" : String(trimmed)
    }

    private static func sum(_ lhs: String, _ rhs: String) -> String {
        var result: [UInt8] = []
        var carry: UInt8 = 0
        var left = lhs.utf8.reversed().makeIterator()
        var right = rhs.utf8.reversed().makeIterator()
        while true {
            let a = left.next(), b = right.next()
            guard a != nil || b != nil || carry > 0 else { break }
            let total = (a ?? 0x30) - 0x30 + (b ?? 0x30) - 0x30 + carry
            result.append(0x30 + total % 10)
            carry = total / 10
        }
        return String(decoding: result.reversed(), as: UTF8.self)
    }

    // MARK: - Text passes

    private static let and = Array("and".unicodeScalars), half = Array("half".unicodeScalars)
    private static let ordinalSuffixes = ["st", "nd", "rd", "th", "s"].map { Array($0.unicodeScalars) }
    private static let currencies: [Unicode.Scalar] = ["€", "£", "$"]

    private static func preprocess(_ text: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var text = text.indices.contains { andAHalfEnd(text, at: $0) != nil } ? replacingHalves(text) : text
        guard text.contains(where: PythonUnicode.isDigit) else { return text }

        // A space at each number/letter boundary, except before a suffix.
        text.substitute { text, index in
            isLetter(text[index]) && text.isDigit(at: index + 1) ? (2, [text[index], " ", text[index + 1]]) : nil
        }
        text.substitute { text, index in
            PythonUnicode.isDigit(text[index]) && index + 1 < text.count && isLetter(text[index + 1])
                ? (2, [text[index], " ", text[index + 1]]) : nil
        }
        text.substitute { text, index in
            guard PythonUnicode.isDigit(text[index]), text.isSpace(at: index + 1) else { return nil }
            let start = text.end(from: index + 1, while: PythonText.isSpace)
            guard
                let suffix = ordinalSuffixes.first(where: {
                    text.contains($0, at: start) && text.isBoundary(at: start + $0.count)
                })
            else { return nil }
            return (start + suffix.count - index, [text[index]] + suffix)
        }
        return text
    }

    /// "<number> and a half" → "<number> point five".
    private static func replacingHalves(_ text: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var segments: [ArraySlice<Unicode.Scalar>] = []
        var start = 0
        var index = 0
        while index < text.count {
            if let end = andAHalfEnd(text, at: index) {
                segments.append(text[start..<index])
                index = end
                start = end
            } else {
                index += 1
            }
        }
        segments.append(text[start...])
        var results: [Unicode.Scalar] = []
        for (offset, segment) in segments.enumerated() where !segment.allSatisfy(PythonText.isSpace) {
            if !results.isEmpty { results.append(" ") }
            results += segment
            guard offset < segments.count - 1 else { continue }
            let last = segment.split(whereSeparator: PythonText.isSpace).last.map(\.string)
            let word = last.flatMap { words[$0] }
            let isNumber = word.map { $0.isZero || $0.isOne || $0.isTen || $0.isMultiplier } ?? false
            results += (isNumber ? " point five" : " and a half").unicodeScalars
        }
        return results
    }

    /// The end of `\band\s+a\s+half\b` at `index`.
    private static func andAHalfEnd(_ text: [Unicode.Scalar], at index: Int) -> Int? {
        guard text[index] == "a", text.contains(and, at: index), text.isBoundary(at: index) else { return nil }
        let article = text.end(from: index + 3, while: PythonText.isSpace)
        guard article > index + 3, text.contains("a", at: article) else { return nil }
        let halfStart = text.end(from: article + 1, while: PythonText.isSpace)
        guard halfStart > article + 1, text.contains(half, at: halfStart), text.isBoundary(at: halfStart + 4) else {
            return nil
        }
        return halfStart + 4
    }

    private static func postprocess(_ text: inout [Unicode.Scalar]) {
        if text.contains(where: currencies.contains) {
            // "$2 and ¢7" → "$2.07".
            text.substitute { text, index in
                guard currencies.contains(text[index]), text.isDigit(at: index + 1) else { return nil }
                let integerEnd = text.end(from: index + 1, while: PythonUnicode.isDigit)
                guard text.contains(" ", at: integerEnd) else { return nil }
                var cent = integerEnd + 1
                if text.contains(Array("and ".unicodeScalars), at: cent) { cent += 4 }
                guard text.contains("¢", at: cent), let cents = centDigits(text, at: cent + 1) else { return nil }
                let padded = cents.count == 1 ? ["0"] + cents : cents
                return (cent + 1 + cents.count - index, Array(text[index..<integerEnd]) + ["."] + padded)
            }
            // "$0.07" → "¢7".
            text.substitute { text, index in
                guard currencies.contains(text[index]), text.contains("0", at: index + 1),
                    index + 2 < text.count, text[index + 2] != "\n", let cents = centDigits(text, at: index + 3)
                else { return nil }
                return (3 + cents.count, ["¢"] + (cents.count == 2 && cents[0] == "0" ? [cents[1]] : cents))
            }
        }
        // "1" → "one", "1s" → "ones".
        guard text.contains("1") else { return }
        text.substitute { text, index in
            guard text[index] == "1", text.isBoundary(at: index) else { return nil }
            if text.contains("s", at: index + 1), text.isBoundary(at: index + 2) { return (2, ["o", "n", "e", "s"]) }
            return text.isBoundary(at: index + 1) ? (1, ["o", "n", "e"]) : nil
        }
    }

    /// `([0-9]{1,2})\b` at `index`.
    private static func centDigits(_ text: [Unicode.Scalar], at index: Int) -> [Unicode.Scalar]? {
        let end = text.end(from: index, while: PythonUnicode.isDigit)
        return (1...2).contains(end - index) && text.isBoundary(at: end) ? Array(text[index..<end]) : nil
    }

    @inline(__always)
    private static func isLetter(_ scalar: Unicode.Scalar) -> Bool { ("a"..."z").contains(scalar) }
}
