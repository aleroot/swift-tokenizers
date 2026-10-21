/// Decodes JSON text exactly as Python's `json.loads` does: `NaN`, `Infinity` and `-Infinity`
/// are accepted, control characters inside strings are rejected, a repeated key keeps its first
/// position and takes the last value, and only ASCII whitespace may surround the value.
enum ResponseJSON {
    enum Failure: Error, Equatable {
        /// The text is not JSON (`json.JSONDecodeError`).
        case invalid(String)
        /// Nesting deeper than Python's recursion limit (`RecursionError`), which `allow_non_json`
        /// does not absorb.
        case tooDeep
    }

    /// Nesting depth at which CPython's scanner runs out of recursion.
    static let maximumDepth = 1_000

    static func decode(_ text: String) throws(Failure) -> ResponseValue {
        var text = text
        let result: Result<ResponseValue, Failure> = text.withUTF8 { bytes in
            var parser = Parser(bytes: bytes)
            do throws(Failure) { return .success(try parser.document()) } catch { return .failure(error) }
        }
        return try result.get()
    }

    private struct Parser {
        let bytes: UnsafeBufferPointer<UInt8>
        var index = 0
        var depth = 0

        init(bytes: UnsafeBufferPointer<UInt8>) { self.bytes = bytes }

        mutating func document() throws(Failure) -> ResponseValue {
            if bytes.starts(with: [0xEF, 0xBB, 0xBF]) { throw .invalid("Unexpected UTF-8 BOM") }
            skipWhitespace()
            let value = try value()
            skipWhitespace()
            guard index == bytes.count else { throw .invalid("Extra data at byte \(index)") }
            return value
        }

        @inline(__always)
        mutating func skipWhitespace() {
            while index < bytes.count {
                switch bytes[index] {
                case 0x20, 0x09, 0x0A, 0x0D: index += 1
                default: return
                }
            }
        }

        @inline(__always)
        func peek() -> UInt8? { index < bytes.count ? bytes[index] : nil }

        mutating func consume(_ literal: StaticString) -> Bool {
            let count = literal.utf8CodeUnitCount
            guard index + count <= bytes.count else { return false }
            for offset in 0..<count where bytes[index + offset] != literal.utf8Start[offset] { return false }
            index += count
            return true
        }

        mutating func value() throws(Failure) -> ResponseValue {
            guard let byte = peek() else { throw .invalid("Expecting value at byte \(index)") }
            switch byte {
            case UInt8(ascii: "\""):
                return .string(try string())
            case UInt8(ascii: "{"):
                return try object()
            case UInt8(ascii: "["):
                return try array()
            case UInt8(ascii: "n") where consume("null"):
                return .null
            case UInt8(ascii: "t") where consume("true"):
                return .bool(true)
            case UInt8(ascii: "f") where consume("false"):
                return .bool(false)
            case UInt8(ascii: "N") where consume("NaN"):
                return .double(.nan)
            case UInt8(ascii: "I") where consume("Infinity"):
                return .double(.infinity)
            case UInt8(ascii: "-") where index + 1 < bytes.count && bytes[index + 1] == UInt8(ascii: "I"):
                index += 1
                guard consume("Infinity") else { throw .invalid("Expecting value at byte \(index - 1)") }
                return .double(-.infinity)
            case UInt8(ascii: "-"), UInt8(ascii: "0")...UInt8(ascii: "9"):
                return try number()
            default:
                throw .invalid("Expecting value at byte \(index)")
            }
        }

        mutating func enter() throws(Failure) {
            depth += 1
            if depth > ResponseJSON.maximumDepth { throw .tooDeep }
        }

        mutating func object() throws(Failure) -> ResponseValue {
            try enter()
            defer { depth -= 1 }
            index += 1
            var object = ResponseObject()
            skipWhitespace()
            if peek() == UInt8(ascii: "}") {
                index += 1
                return .object(object)
            }
            while true {
                guard peek() == UInt8(ascii: "\"") else {
                    throw .invalid("Expecting property name enclosed in double quotes at byte \(index)")
                }
                let key = try string()
                skipWhitespace()
                guard peek() == UInt8(ascii: ":") else { throw .invalid("Expecting ':' delimiter at byte \(index)") }
                index += 1
                skipWhitespace()
                object[key] = try value()
                skipWhitespace()
                switch peek() {
                case UInt8(ascii: ","):
                    index += 1
                    skipWhitespace()
                case UInt8(ascii: "}"):
                    index += 1
                    return .object(object)
                default:
                    throw .invalid("Expecting ',' delimiter at byte \(index)")
                }
            }
        }

        mutating func array() throws(Failure) -> ResponseValue {
            try enter()
            defer { depth -= 1 }
            index += 1
            var values: [ResponseValue] = []
            skipWhitespace()
            if peek() == UInt8(ascii: "]") {
                index += 1
                return .array(values)
            }
            while true {
                values.append(try value())
                skipWhitespace()
                switch peek() {
                case UInt8(ascii: ","):
                    index += 1
                    skipWhitespace()
                case UInt8(ascii: "]"):
                    index += 1
                    return .array(values)
                default:
                    throw .invalid("Expecting ',' delimiter at byte \(index)")
                }
            }
        }

        mutating func number() throws(Failure) -> ResponseValue {
            let start = index
            if peek() == UInt8(ascii: "-") { index += 1 }
            // `-?(0|[1-9][0-9]*)`, ASCII digits only, as in CPython's C scanner.
            guard let first = peek(), isDigit(first) else { throw .invalid("Expecting value at byte \(start)") }
            index += 1
            if first != UInt8(ascii: "0") { skipDigits() }
            var isInteger = true
            if peek() == UInt8(ascii: "."), index + 1 < bytes.count, isDigit(bytes[index + 1]) {
                index += 1
                skipDigits()
                isInteger = false
            }
            if let marker = peek(), marker == UInt8(ascii: "e") || marker == UInt8(ascii: "E") {
                var cursor = index + 1
                if cursor < bytes.count, bytes[cursor] == UInt8(ascii: "+") || bytes[cursor] == UInt8(ascii: "-") {
                    cursor += 1
                }
                if cursor < bytes.count, isDigit(bytes[cursor]) {
                    index = cursor
                    skipDigits()
                    isInteger = false
                }
            }
            let literal = String(decoding: UnsafeBufferPointer(rebasing: bytes[start..<index]), as: UTF8.self)
            if isInteger, let value = Int(literal) { return .int(value) }
            // Integers beyond `Int` are arbitrary precision in Python; the nearest double is kept.
            return .double(Double(literal)!)
        }

        @inline(__always)
        func isDigit(_ byte: UInt8) -> Bool { byte &- UInt8(ascii: "0") < 10 }

        @inline(__always)
        mutating func skipDigits() {
            while index < bytes.count, isDigit(bytes[index]) { index += 1 }
        }

        mutating func string() throws(Failure) -> String {
            index += 1
            var output: [UInt8] = []
            var runStart = index
            while index < bytes.count {
                let byte = bytes[index]
                if byte == UInt8(ascii: "\"") {
                    output.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[runStart..<index]))
                    index += 1
                    return String(decoding: output, as: UTF8.self)
                }
                if byte < 0x20 { throw .invalid("Invalid control character at byte \(index)") }
                if byte != UInt8(ascii: "\\") {
                    index += 1
                    continue
                }
                output.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[runStart..<index]))
                index += 1
                guard let escape = peek() else { break }
                index += 1
                switch escape {
                case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"): output.append(escape)
                case UInt8(ascii: "b"): output.append(0x08)
                case UInt8(ascii: "f"): output.append(0x0C)
                case UInt8(ascii: "n"): output.append(0x0A)
                case UInt8(ascii: "r"): output.append(0x0D)
                case UInt8(ascii: "t"): output.append(0x09)
                case UInt8(ascii: "u"):
                    var scalar = try hex4()
                    if (0xD800...0xDBFF).contains(scalar), index + 1 < bytes.count,
                        bytes[index] == UInt8(ascii: "\\"), bytes[index + 1] == UInt8(ascii: "u")
                    {
                        let save = index
                        index += 2
                        let low = try hex4()
                        if (0xDC00...0xDFFF).contains(low) {
                            scalar = 0x10000 + ((scalar - 0xD800) << 10) + (low - 0xDC00)
                        } else {
                            index = save
                        }
                    }
                    // Python keeps a lone surrogate; a Swift string cannot hold one.
                    let decoded = Unicode.Scalar(scalar) ?? "\u{FFFD}"
                    output.append(contentsOf: String(decoded).utf8)
                default:
                    throw .invalid("Invalid \\escape at byte \(index - 2)")
                }
                runStart = index
            }
            throw .invalid("Unterminated string")
        }

        mutating func hex4() throws(Failure) -> UInt32 {
            guard index + 4 <= bytes.count else { throw .invalid("Invalid \\uXXXX escape at byte \(index)") }
            var value: UInt32 = 0
            for _ in 0..<4 {
                let byte = bytes[index]
                let digit: UInt8
                switch byte {
                case UInt8(ascii: "0")...UInt8(ascii: "9"): digit = byte - UInt8(ascii: "0")
                case UInt8(ascii: "a")...UInt8(ascii: "f"): digit = byte - UInt8(ascii: "a") + 10
                case UInt8(ascii: "A")...UInt8(ascii: "F"): digit = byte - UInt8(ascii: "A") + 10
                default: throw .invalid("Invalid \\uXXXX escape at byte \(index)")
                }
                value = value << 4 | UInt32(digit)
                index += 1
            }
            return value
        }
    }
}
