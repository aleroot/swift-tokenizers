// A small, allocation-conscious JSON parser that produces `Config` trees directly from
// UTF-8 bytes. `tokenizer.json` files for modern models are 5–20 MB with 100K+ vocabulary
// entries; going through `JSONSerialization` and then re-walking an `NSDictionary` tree
// is several times slower than building the target representation in a single pass.

import Foundation

public enum JSONConfigError: Error, CustomStringConvertible, Sendable {
    case unexpectedEnd
    case unexpectedCharacter(UInt8, offset: Int)
    case invalidNumber(offset: Int)
    case invalidEscape(offset: Int)
    case invalidUTF8(offset: Int)
    case nestingTooDeep
    case trailingGarbage(offset: Int)
    case unsupportedEncoding

    public var description: String {
        switch self {
        case .unexpectedEnd: "Unexpected end of JSON input"
        case let .unexpectedCharacter(c, offset):
            "Unexpected character '\(Character(UnicodeScalar(c)))' at offset \(offset)"
        case let .invalidNumber(offset): "Invalid number at offset \(offset)"
        case let .invalidEscape(offset): "Invalid escape sequence at offset \(offset)"
        case let .invalidUTF8(offset): "Invalid UTF-8 at offset \(offset)"
        case .nestingTooDeep: "JSON nesting too deep"
        case let .trailingGarbage(offset): "Trailing characters after JSON value at offset \(offset)"
        case .unsupportedEncoding: "Unsupported text encoding"
        }
    }
}

public extension Config {
    /// Parses a JSON document. Accepts UTF-8 (with or without BOM) as well as UTF-16 LE/BE
    /// documents that carry a byte order mark. Like swift-transformers' file loader,
    /// accepts trailing commas and non-finite numeric metadata (Inf/Infinity/NaN).
    init(jsonData data: Foundation.Data) throws {
        let utf8 = try Config.normalizeToUTF8(data)
        self = try utf8.withUnsafeBytes { raw -> Config in
            var parser = JSONConfigParser(bytes: raw.bindMemory(to: UInt8.self))
            return try parser.parseDocument()
        }
    }

    /// Parses a JSON document held in a `String`.
    init(jsonString string: String) throws {
        var copy = string
        self = try copy.withUTF8 { buffer -> Config in
            var parser = JSONConfigParser(bytes: buffer)
            return try parser.parseDocument()
        }
    }

    /// Reads and parses a JSON file.
    init(jsonFile url: URL) throws {
        let data = try Foundation.Data(contentsOf: url, options: .mappedIfSafe)
        try self.init(jsonData: data)
    }

    private static func normalizeToUTF8(_ data: Foundation.Data) throws -> Foundation.Data {
        guard data.count >= 2 else { return data }
        let b0 = data[data.startIndex]
        let b1 = data[data.startIndex + 1]
        if b0 == 0xFF, b1 == 0xFE {
            guard let s = String(data: data, encoding: .utf16LittleEndian) else {
                throw JSONConfigError.unsupportedEncoding
            }
            return Foundation.Data(s.utf8)
        }
        if b0 == 0xFE, b1 == 0xFF {
            guard let s = String(data: data, encoding: .utf16BigEndian) else {
                throw JSONConfigError.unsupportedEncoding
            }
            return Foundation.Data(s.utf8)
        }
        if data.count >= 3, b0 == 0xEF, b1 == 0xBB, data[data.startIndex + 2] == 0xBF {
            return data.dropFirst(3)
        }
        return data
    }
}

struct JSONConfigParser {
    private let bytes: UnsafeBufferPointer<UInt8>
    private var pos: Int = 0
    private var depth: Int = 0
    private static let maxDepth = 512

    /// Scratch buffer reused while unescaping strings.
    private var scratch: [UInt8] = []

    init(bytes: UnsafeBufferPointer<UInt8>) {
        self.bytes = bytes
    }

    mutating func parseDocument() throws -> Config {
        skipWhitespace()
        let value = try parseValue()
        skipWhitespace()
        if pos < bytes.count {
            throw JSONConfigError.trailingGarbage(offset: pos)
        }
        return value
    }

    // MARK: - Values

    private mutating func parseValue() throws -> Config {
        guard pos < bytes.count else { throw JSONConfigError.unexpectedEnd }
        switch bytes[pos] {
        case UInt8(ascii: "{"):
            return try parseObject()
        case UInt8(ascii: "["):
            return try parseArray()
        case UInt8(ascii: "\""):
            return Config(BinaryDistinctString(try parseString()))
        case UInt8(ascii: "t"):
            try expectLiteral("true")
            return Config(true)
        case UInt8(ascii: "f"):
            try expectLiteral("false")
            return Config(false)
        case UInt8(ascii: "n"):
            try expectLiteral("null")
            return Config()
        case UInt8(ascii: "I"), UInt8(ascii: "N"):
            return try parseNonFiniteNumber()
        case UInt8(ascii: "-"), UInt8(ascii: "0")...UInt8(ascii: "9"):
            if bytes[pos] == UInt8(ascii: "-"), pos + 1 < bytes.count, bytes[pos + 1] == UInt8(ascii: "I") {
                return try parseNonFiniteNumber()
            }
            return try parseNumber()
        default:
            throw JSONConfigError.unexpectedCharacter(bytes[pos], offset: pos)
        }
    }

    private mutating func parseObject() throws -> Config {
        depth += 1
        defer { depth -= 1 }
        guard depth <= Self.maxDepth else { throw JSONConfigError.nestingTooDeep }

        pos += 1  // '{'
        var dict = [BinaryDistinctString: Config]()
        skipWhitespace()
        if pos < bytes.count, bytes[pos] == UInt8(ascii: "}") {
            pos += 1
            return Config(dict)
        }
        while true {
            skipWhitespace()
            guard pos < bytes.count else { throw JSONConfigError.unexpectedEnd }
            guard bytes[pos] == UInt8(ascii: "\"") else {
                throw JSONConfigError.unexpectedCharacter(bytes[pos], offset: pos)
            }
            let key = try parseString()
            skipWhitespace()
            guard pos < bytes.count else { throw JSONConfigError.unexpectedEnd }
            guard bytes[pos] == UInt8(ascii: ":") else {
                throw JSONConfigError.unexpectedCharacter(bytes[pos], offset: pos)
            }
            pos += 1
            skipWhitespace()
            let value = try parseValue()
            dict[BinaryDistinctString(key)] = value
            skipWhitespace()
            guard pos < bytes.count else { throw JSONConfigError.unexpectedEnd }
            let c = bytes[pos]
            if c == UInt8(ascii: ",") {
                pos += 1
                skipWhitespace()
                if pos < bytes.count, bytes[pos] == UInt8(ascii: "}") {
                    pos += 1
                    return Config(dict)
                }
                continue
            }
            if c == UInt8(ascii: "}") {
                pos += 1
                return Config(dict)
            }
            throw JSONConfigError.unexpectedCharacter(c, offset: pos)
        }
    }

    private mutating func parseArray() throws -> Config {
        depth += 1
        defer { depth -= 1 }
        guard depth <= Self.maxDepth else { throw JSONConfigError.nestingTooDeep }

        pos += 1  // '['
        var array: [Config] = []
        skipWhitespace()
        if pos < bytes.count, bytes[pos] == UInt8(ascii: "]") {
            pos += 1
            return Config(array)
        }
        while true {
            skipWhitespace()
            array.append(try parseValue())
            skipWhitespace()
            guard pos < bytes.count else { throw JSONConfigError.unexpectedEnd }
            let c = bytes[pos]
            if c == UInt8(ascii: ",") {
                pos += 1
                skipWhitespace()
                if pos < bytes.count, bytes[pos] == UInt8(ascii: "]") {
                    pos += 1
                    return Config(array)
                }
                continue
            }
            if c == UInt8(ascii: "]") {
                pos += 1
                return Config(array)
            }
            throw JSONConfigError.unexpectedCharacter(c, offset: pos)
        }
    }

    private mutating func expectLiteral(_ literal: StaticString) throws {
        let count = literal.utf8CodeUnitCount
        guard pos + count <= bytes.count else { throw JSONConfigError.unexpectedEnd }
        let matches = literal.withUTF8Buffer { lit -> Bool in
            for i in 0..<count where bytes[pos + i] != lit[i] { return false }
            return true
        }
        guard matches else { throw JSONConfigError.unexpectedCharacter(bytes[pos], offset: pos) }
        pos += count
    }

    // MARK: - Numbers

    private mutating func parseNonFiniteNumber() throws -> Config {
        let start = pos
        if bytes[pos] == UInt8(ascii: "-") { pos += 1 }
        while pos < bytes.count, (0x41...0x5A).contains(bytes[pos]) || (0x61...0x7A).contains(bytes[pos]) {
            pos += 1
        }
        let text = String(decoding: bytes[start..<pos], as: UTF8.self)
        switch text {
        case "Inf", "Infinity": return Config(Double.infinity)
        case "-Inf", "-Infinity": return Config(-Double.infinity)
        case "NaN": return Config(Double.nan)
        default: throw JSONConfigError.invalidNumber(offset: start)
        }
    }

    private mutating func parseNumber() throws -> Config {
        let start = pos
        var negative = false
        if bytes[pos] == UInt8(ascii: "-") {
            negative = true
            pos += 1
        }
        guard pos < bytes.count else { throw JSONConfigError.unexpectedEnd }

        let integerStart = pos
        guard bytes[pos] >= 0x30, bytes[pos] <= 0x39 else {
            throw JSONConfigError.invalidNumber(offset: start)
        }
        if bytes[pos] == 0x30 {
            pos += 1
            if pos < bytes.count, bytes[pos] >= 0x30, bytes[pos] <= 0x39 {
                throw JSONConfigError.invalidNumber(offset: start)
            }
        } else {
            while pos < bytes.count, bytes[pos] >= 0x30, bytes[pos] <= 0x39 { pos += 1 }
        }
        let integerEnd = pos
        var isFloat = false
        if pos < bytes.count, bytes[pos] == UInt8(ascii: ".") {
            isFloat = true
            pos += 1
            let digitsStart = pos
            while pos < bytes.count, bytes[pos] >= 0x30, bytes[pos] <= 0x39 { pos += 1 }
            guard pos > digitsStart else { throw JSONConfigError.invalidNumber(offset: start) }
        }
        if pos < bytes.count, bytes[pos] == UInt8(ascii: "e") || bytes[pos] == UInt8(ascii: "E") {
            isFloat = true
            pos += 1
            if pos < bytes.count, bytes[pos] == UInt8(ascii: "+") || bytes[pos] == UInt8(ascii: "-") { pos += 1 }
            let digitsStart = pos
            while pos < bytes.count, bytes[pos] >= 0x30, bytes[pos] <= 0x39 { pos += 1 }
            guard pos > digitsStart else { throw JSONConfigError.invalidNumber(offset: start) }
        }

        if !isFloat {
            var magnitude: UInt64 = 0
            for i in integerStart..<integerEnd {
                let (product, multiplicationOverflow) = magnitude.multipliedReportingOverflow(by: 10)
                let (sum, additionOverflow) = product.addingReportingOverflow(UInt64(bytes[i] - 0x30))
                if multiplicationOverflow || additionOverflow { return try floatingFallback(start) }
                magnitude = sum
            }
            if negative {
                guard magnitude <= UInt64(Int.max) + 1 else { return try floatingFallback(start) }
                return Config(magnitude == UInt64(Int.max) + 1 ? Int.min : -Int(magnitude))
            }
            guard magnitude <= UInt64(Int.max) else { return try floatingFallback(start) }
            return Config(Int(magnitude))
        }
        return try floatingFallback(start)
    }

    private func floatingFallback(_ start: Int) throws -> Config {
        let slice = UnsafeBufferPointer(rebasing: bytes[start..<pos])
        let text = String(decoding: slice, as: UTF8.self)
        guard let d = Double(text), d.isFinite else { throw JSONConfigError.invalidNumber(offset: start) }
        return Config(d)
    }

    // MARK: - Strings

    /// Parses a JSON string starting at the opening quote and returns its contents.
    private mutating func parseString() throws -> String {
        pos += 1  // opening quote
        let start = pos

        // Fast scan: find closing quote, detect escapes and non-ASCII bytes.
        var hasEscape = false
        var i = pos
        let n = bytes.count
        while i < n {
            let c = bytes[i]
            if c == UInt8(ascii: "\"") { break }
            if c == UInt8(ascii: "\\") {
                hasEscape = true
                break
            }
            guard c >= 0x20 else { throw JSONConfigError.unexpectedCharacter(c, offset: i) }
            i += 1
        }

        if !hasEscape {
            guard i < n else { throw JSONConfigError.unexpectedEnd }
            let slice = UnsafeBufferPointer(rebasing: bytes[start..<i])
            pos = i + 1
            let string = String(decoding: slice, as: UTF8.self)
            // Foundation's encoding initializer consumes a leading BOM even inside a
            // vocabulary key. Preserve every scalar and reject repaired UTF-8 instead.
            guard string.utf8.elementsEqual(slice) else {
                throw JSONConfigError.invalidUTF8(offset: start)
            }
            return string
        }

        // Slow path with escapes: copy into scratch buffer.
        scratch.removeAll(keepingCapacity: true)
        scratch.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[start..<i]))
        pos = i
        while pos < n {
            let c = bytes[pos]
            if c == UInt8(ascii: "\"") {
                pos += 1
                let string = String(decoding: scratch, as: UTF8.self)
                guard string.utf8.elementsEqual(scratch) else {
                    throw JSONConfigError.invalidUTF8(offset: start)
                }
                return string
            }
            if c == UInt8(ascii: "\\") {
                pos += 1
                guard pos < n else { throw JSONConfigError.unexpectedEnd }
                let e = bytes[pos]
                pos += 1
                switch e {
                case UInt8(ascii: "\""): scratch.append(0x22)
                case UInt8(ascii: "\\"): scratch.append(0x5C)
                case UInt8(ascii: "/"): scratch.append(0x2F)
                case UInt8(ascii: "b"): scratch.append(0x08)
                case UInt8(ascii: "f"): scratch.append(0x0C)
                case UInt8(ascii: "n"): scratch.append(0x0A)
                case UInt8(ascii: "r"): scratch.append(0x0D)
                case UInt8(ascii: "t"): scratch.append(0x09)
                case UInt8(ascii: "u"):
                    var scalar = try parseHex4()
                    if scalar >= 0xD800, scalar <= 0xDBFF {
                        // High surrogate: expect a low surrogate.
                        if pos + 1 < n, bytes[pos] == UInt8(ascii: "\\"), bytes[pos + 1] == UInt8(ascii: "u") {
                            pos += 2
                            let low = try parseHex4()
                            if low >= 0xDC00, low <= 0xDFFF {
                                scalar = 0x10000 + ((scalar - 0xD800) << 10) + (low - 0xDC00)
                            } else {
                                throw JSONConfigError.invalidEscape(offset: pos - 4)
                            }
                        } else {
                            throw JSONConfigError.invalidEscape(offset: pos)
                        }
                    } else if scalar >= 0xDC00, scalar <= 0xDFFF {
                        throw JSONConfigError.invalidEscape(offset: pos - 4)
                    }
                    appendUTF8(scalar)
                default:
                    throw JSONConfigError.invalidEscape(offset: pos - 1)
                }
            } else {
                guard c >= 0x20 else { throw JSONConfigError.unexpectedCharacter(c, offset: pos) }
                scratch.append(c)
                pos += 1
            }
        }
        throw JSONConfigError.unexpectedEnd
    }

    private mutating func parseHex4() throws -> UInt32 {
        guard pos + 4 <= bytes.count else { throw JSONConfigError.unexpectedEnd }
        var value: UInt32 = 0
        for _ in 0..<4 {
            let c = bytes[pos]
            let digit: UInt32
            switch c {
            case UInt8(ascii: "0")...UInt8(ascii: "9"): digit = UInt32(c - UInt8(ascii: "0"))
            case UInt8(ascii: "a")...UInt8(ascii: "f"): digit = UInt32(c - UInt8(ascii: "a") + 10)
            case UInt8(ascii: "A")...UInt8(ascii: "F"): digit = UInt32(c - UInt8(ascii: "A") + 10)
            default: throw JSONConfigError.invalidEscape(offset: pos)
            }
            value = (value << 4) | digit
            pos += 1
        }
        return value
    }

    private mutating func appendUTF8(_ scalar: UInt32) {
        switch scalar {
        case 0..<0x80:
            scratch.append(UInt8(scalar))
        case 0x80..<0x800:
            scratch.append(UInt8(0xC0 | (scalar >> 6)))
            scratch.append(UInt8(0x80 | (scalar & 0x3F)))
        case 0x800..<0x10000:
            scratch.append(UInt8(0xE0 | (scalar >> 12)))
            scratch.append(UInt8(0x80 | ((scalar >> 6) & 0x3F)))
            scratch.append(UInt8(0x80 | (scalar & 0x3F)))
        default:
            scratch.append(UInt8(0xF0 | (scalar >> 18)))
            scratch.append(UInt8(0x80 | ((scalar >> 12) & 0x3F)))
            scratch.append(UInt8(0x80 | ((scalar >> 6) & 0x3F)))
            scratch.append(UInt8(0x80 | (scalar & 0x3F)))
        }
    }

    // MARK: - Whitespace

    @inline(__always)
    private mutating func skipWhitespace() {
        while pos < bytes.count {
            switch bytes[pos] {
            case 0x20, 0x09, 0x0A, 0x0D: pos += 1
            default: return
            }
        }
    }
}
