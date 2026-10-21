/// Python `str` semantics that response templates depend on.
enum PythonText {
    /// `str.isspace()`: the Unicode `White_Space` characters plus the ASCII separators
    /// U+001C...U+001F.
    @inline(__always)
    static func isSpace(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x09...0x0D, 0x1C...0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F,
            0x3000:
            true
        default: false
        }
    }

    /// `\w` in a Python `str` pattern: letters, marks, decimal digits, connector punctuation
    /// and the joiners.
    @inline(__always)
    static func isWord(_ scalar: Unicode.Scalar) -> Bool {
        if scalar.isASCII {
            let value = UInt8(scalar.value)
            return (0x30...0x39).contains(value) || (0x41...0x5A).contains(value) || (0x61...0x7A).contains(value)
                || value == 0x5F
        }
        let properties = scalar.properties
        if properties.isAlphabetic || properties.isJoinControl { return true }
        switch properties.generalCategory {
        case .nonspacingMark, .spacingMark, .enclosingMark, .decimalNumber, .connectorPunctuation: return true
        default: return false
        }
    }

    /// `str.strip()`.
    static func strip(_ text: Substring) -> Substring {
        let scalars = text.unicodeScalars
        guard let first = scalars.firstIndex(where: { !isSpace($0) }) else { return text[text.endIndex...] }
        let last = scalars.lastIndex(where: { !isSpace($0) })!
        return Substring(scalars[first...last])
    }

    static func strip(_ text: String) -> String { String(strip(text[...])) }

    /// `int(text)`: optional surrounding whitespace and sign, decimal digits of any script, and
    /// single underscores between digits. `nil` where Python raises `ValueError`, or when the
    /// value does not fit in `Int`.
    static func int(_ text: String) -> Int? {
        guard let (negative, digits) = signedDigits(strip(text[...])), !digits.isEmpty else { return nil }
        guard let magnitude = Int(digits) else { return nil }
        return negative ? -magnitude : magnitude
    }

    /// `float(text)`: a decimal literal with optional exponent, `inf`, `infinity` or `nan` in
    /// any case, digits of any script and underscores between digits.
    static func float(_ text: String) -> Double? {
        let stripped = strip(text[...])
        var body = stripped
        var negative = false
        if let sign = body.unicodeScalars.first, sign == "+" || sign == "-" {
            negative = sign == "-"
            body = body.dropFirst()
        }
        switch body.lowercased() {
        case "inf", "infinity": return negative ? -.infinity : .infinity
        case "nan": return .nan
        default: break
        }
        // Mantissa digits, optional fraction, optional exponent; underscores only between digits.
        var ascii = negative ? "-" : ""
        var sawDigit = false
        var previousWasDigit = false
        var pendingUnderscore = false
        var state = 0  // 0 integer part, 1 fraction, 2 exponent sign, 3 exponent digits
        var exponentDigits = false
        for scalar in body.unicodeScalars {
            if let digit = decimalValue(scalar) {
                ascii.unicodeScalars.append(Unicode.Scalar(UInt8(ascii: "0") + UInt8(digit)))
                if state == 2 { state = 3 }
                if state == 3 { exponentDigits = true } else { sawDigit = true }
                previousWasDigit = true
                pendingUnderscore = false
                continue
            }
            if scalar == "_" {
                guard previousWasDigit else { return nil }
                previousWasDigit = false
                pendingUnderscore = true
                continue
            }
            if pendingUnderscore { return nil }
            previousWasDigit = false
            switch (scalar, state) {
            case (".", 0):
                state = 1
                ascii += "."
            case ("e", 0), ("E", 0), ("e", 1), ("E", 1):
                guard sawDigit else { return nil }
                state = 2
                ascii += "e"
            case ("+", 2), ("-", 2):
                state = 3
                ascii.unicodeScalars.append(scalar)
            default:
                return nil
            }
        }
        guard sawDigit, !pendingUnderscore, state < 2 || exponentDigits else { return nil }
        return Double(ascii)
    }

    /// The sign and ASCII digits of an integer literal.
    private static func signedDigits(_ text: Substring) -> (Bool, String)? {
        var scalars = Substring.UnicodeScalarView(text.unicodeScalars)
        var negative = false
        if let sign = scalars.first, sign == "+" || sign == "-" {
            negative = sign == "-"
            scalars = scalars.dropFirst()
        }
        var digits = ""
        var previousWasDigit = false
        for scalar in scalars {
            if let digit = decimalValue(scalar) {
                digits.unicodeScalars.append(Unicode.Scalar(UInt8(ascii: "0") + UInt8(digit)))
                previousWasDigit = true
            } else if scalar == "_", previousWasDigit {
                previousWasDigit = false
            } else {
                return nil
            }
        }
        guard previousWasDigit else { return nil }
        return (negative, digits)
    }

    /// The value of a decimal digit (`Nd`) in any script.
    @inline(__always)
    private static func decimalValue(_ scalar: Unicode.Scalar) -> Int? {
        if scalar.isASCII {
            let value = scalar.value
            return (0x30...0x39).contains(value) ? Int(value - 0x30) : nil
        }
        guard scalar.properties.numericType == .decimal, let value = scalar.properties.numericValue else {
            return nil
        }
        return Int(value)
    }

    /// `str.split(separator)` on code points. `separator` must not be empty.
    static func split(_ text: String, separator: String) -> [Substring] {
        let scalars = text.unicodeScalars
        let needle = Array(separator.unicodeScalars)
        var pieces: [Substring] = []
        var pieceStart = scalars.startIndex
        var index = scalars.startIndex
        while index < scalars.endIndex {
            if scalars[index...].starts(with: needle) {
                pieces.append(Substring(scalars[pieceStart..<index]))
                index = scalars.index(index, offsetBy: needle.count)
                pieceStart = index
            } else {
                index = scalars.index(after: index)
            }
        }
        pieces.append(Substring(scalars[pieceStart...]))
        return pieces
    }
}
