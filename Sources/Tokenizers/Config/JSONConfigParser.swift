// An allocation-conscious JSON parser that produces `Config` trees directly from UTF-8 bytes.

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
        try self.init(jsonData: data, packTokenizerTables: false)
    }

    /// Parses `tokenizer.json`, packing `model.vocab` / `model.merges` into compact buffers
    /// instead of a Config tree of 100k+ nodes.
    init(tokenizerJSON data: Foundation.Data) throws {
        try self.init(jsonData: data, packTokenizerTables: true)
    }

    /// Reads and parses a `tokenizer.json` file with packed vocab/merge tables.
    init(tokenizerJSONFile url: URL) throws {
        let data = try Foundation.Data(contentsOf: url, options: .mappedIfSafe)
        try self.init(tokenizerJSON: data)
    }

    private init(jsonData data: Foundation.Data, packTokenizerTables: Bool) throws {
        let utf8 = try Config.normalizeToUTF8(data)
        self = try utf8.withUnsafeBytes { raw -> Config in
            var parser = JSONConfigParser(
                bytes: raw.bindMemory(to: UInt8.self), packTokenizerTables: packTokenizerTables)
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
    /// Grow strategy for vocab objects / merge arrays: reserve once the collection is
    /// clearly large so 100k+ `BinaryDistinctString` keys are not rehashed on every
    /// power-of-two resize. Small objects never hit the threshold.
    private static let largeCollectionThreshold = 64

    /// Scratch buffer reused while unescaping strings.
    private var scratch: [UInt8] = []
    /// When true, `vocab` / `merges` keys are parsed into packed tables.
    private let packTokenizerTables: Bool

    init(bytes: UnsafeBufferPointer<UInt8>, packTokenizerTables: Bool = false) {
        self.bytes = bytes
        self.packTokenizerTables = packTokenizerTables
    }

    /// Remaining-byte heuristic used after a collection proves large. `bytesPerEntry` is
    /// a conservative lower bound (`"a":0,` is 6 bytes; merge lists are larger).
    @inline(__always)
    private func estimatedRemainingEntries(bytesPerEntry: Int) -> Int {
        let remaining = bytes.count - pos
        let estimate = remaining / max(bytesPerEntry, 1)
        return min(max(estimate, 0), 2_000_000)
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

    private mutating func parseValue(key: String? = nil) throws -> Config {
        guard pos < bytes.count else { throw JSONConfigError.unexpectedEnd }
        switch bytes[pos] {
        case UInt8(ascii: "{"):
            if packTokenizerTables, key == "vocab" {
                return try parsePackedStringMapOrObject()
            }
            return try parseObject()
        case UInt8(ascii: "["):
            if packTokenizerTables, key == "vocab" {
                return try parsePackedScoredTokensOrArray()
            }
            if packTokenizerTables, key == "merges" {
                return try parsePackedStringPairsOrArray()
            }
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
            let value = try parseValue(key: key)
            dict[BinaryDistinctString(key)] = value
            if dict.count == Self.largeCollectionThreshold {
                dict.reserveCapacity(
                    Self.largeCollectionThreshold + estimatedRemainingEntries(bytesPerEntry: 24))
            }
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
            if array.count == Self.largeCollectionThreshold {
                array.reserveCapacity(
                    Self.largeCollectionThreshold + estimatedRemainingEntries(bytesPerEntry: 16))
            }
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
            let digits = integerEnd - integerStart
            // Vocab ids are small; skip overflow tracking for the common 1–9 digit case.
            if digits <= 9 {
                var value = 0
                for i in integerStart..<integerEnd {
                    value = value * 10 + Int(bytes[i] - 0x30)
                }
                return Config(negative ? -value : value)
            }
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

        let (i, isASCII) = ByteKernels.jsonStringEnd(bytes, from: pos)
        let n = bytes.count
        if i < n, bytes[i] < 0x20 {
            throw JSONConfigError.unexpectedCharacter(bytes[i], offset: i)
        }
        let hasEscape = i < n && bytes[i] == 0x5C

        if !hasEscape {
            guard i < n else { throw JSONConfigError.unexpectedEnd }
            let slice = UnsafeBufferPointer(rebasing: bytes[start..<i])
            pos = i + 1
            return try Self.makeString(from: slice, offset: start, ascii: isASCII)
        }

        return try parseStringFromEscape(start: start, firstEscape: i)
    }

    private enum PackedShapeError: Error { case mismatch }

    private mutating func parsePackedStringMapOrObject() throws -> Config {
        let saved = pos
        do {
            return try parsePackedStringMap()
        } catch PackedShapeError.mismatch {
            pos = saved
            return try parseObject()
        }
    }

    private mutating func parsePackedScoredTokensOrArray() throws -> Config {
        let saved = pos
        do {
            return try parsePackedScoredTokens()
        } catch PackedShapeError.mismatch {
            pos = saved
            return try parseArray()
        }
    }

    private mutating func parsePackedStringPairsOrArray() throws -> Config {
        let saved = pos
        do {
            return try parsePackedStringPairs()
        } catch PackedShapeError.mismatch {
            pos = saved
            return try parseArray()
        }
    }

    private mutating func parsePackedStringMap() throws -> Config {
        depth += 1
        defer { depth -= 1 }
        guard depth <= Self.maxDepth else { throw JSONConfigError.nestingTooDeep }
        pos += 1  // '{'
        skipWhitespace()
        var utf8: [UInt8] = []
        var offsets: [UInt32] = [0]
        var ids: [Int32] = []
        utf8.reserveCapacity(min(bytes.count - pos, 8_000_000))
        ids.reserveCapacity(estimatedRemainingEntries(bytesPerEntry: 24))
        if pos < bytes.count, bytes[pos] == UInt8(ascii: "}") {
            pos += 1
            return Config(data: .stringMap(PackedStringMap(utf8: utf8, offsets: offsets, ids: ids)))
        }
        while true {
            skipWhitespace()
            guard pos < bytes.count else { throw JSONConfigError.unexpectedEnd }
            guard bytes[pos] == UInt8(ascii: "\"") else {
                throw JSONConfigError.unexpectedCharacter(bytes[pos], offset: pos)
            }
            try appendJSONString(to: &utf8)
            offsets.append(UInt32(utf8.count))
            skipWhitespace()
            guard pos < bytes.count, bytes[pos] == UInt8(ascii: ":") else {
                throw JSONConfigError.unexpectedCharacter(pos < bytes.count ? bytes[pos] : 0, offset: pos)
            }
            pos += 1
            skipWhitespace()
            guard let id = try parseIntegerIfPossible() else { throw PackedShapeError.mismatch }
            ids.append(Int32(clamping: id))
            skipWhitespace()
            guard pos < bytes.count else { throw JSONConfigError.unexpectedEnd }
            let c = bytes[pos]
            if c == UInt8(ascii: ",") {
                pos += 1
                skipWhitespace()
                if pos < bytes.count, bytes[pos] == UInt8(ascii: "}") {
                    pos += 1
                    break
                }
                continue
            }
            if c == UInt8(ascii: "}") {
                pos += 1
                break
            }
            throw JSONConfigError.unexpectedCharacter(c, offset: pos)
        }
        return Config(data: .stringMap(PackedStringMap(utf8: utf8, offsets: offsets, ids: ids)))
    }

    private mutating func parsePackedStringPairs() throws -> Config {
        depth += 1
        defer { depth -= 1 }
        guard depth <= Self.maxDepth else { throw JSONConfigError.nestingTooDeep }
        pos += 1  // '['
        skipWhitespace()
        var utf8: [UInt8] = []
        var offsets: [UInt32] = [0]
        var count = 0
        utf8.reserveCapacity(min(bytes.count - pos, 8_000_000))
        offsets.reserveCapacity(estimatedRemainingEntries(bytesPerEntry: 16) * 2)
        if pos < bytes.count, bytes[pos] == UInt8(ascii: "]") {
            pos += 1
            return Config(data: .stringPairs(PackedStringPairs(utf8: utf8, offsets: offsets, count: 0)))
        }
        while true {
            skipWhitespace()
            guard pos < bytes.count else { throw JSONConfigError.unexpectedEnd }
            if bytes[pos] == UInt8(ascii: "\"") {
                let start = utf8.count
                try appendJSONString(to: &utf8)
                if let space = utf8[start...].firstIndex(of: UInt8(ascii: " ")) {
                    let leftCount = space - start
                    let right = Array(utf8[(space + 1)...])
                    utf8.removeSubrange(space..<utf8.count)
                    offsets.append(UInt32(utf8.count))
                    utf8.append(contentsOf: right)
                    offsets.append(UInt32(utf8.count))
                    _ = leftCount
                } else {
                    throw PackedShapeError.mismatch
                }
            } else if bytes[pos] == UInt8(ascii: "[") {
                pos += 1
                skipWhitespace()
                guard pos < bytes.count, bytes[pos] == UInt8(ascii: "\"") else { throw PackedShapeError.mismatch }
                try appendJSONString(to: &utf8)
                offsets.append(UInt32(utf8.count))
                skipWhitespace()
                if pos < bytes.count, bytes[pos] == UInt8(ascii: ",") {
                    pos += 1
                    skipWhitespace()
                }
                guard pos < bytes.count, bytes[pos] == UInt8(ascii: "\"") else { throw PackedShapeError.mismatch }
                try appendJSONString(to: &utf8)
                offsets.append(UInt32(utf8.count))
                skipWhitespace()
                if pos < bytes.count, bytes[pos] == UInt8(ascii: ",") {
                    pos += 1
                    skipWhitespace()
                }
                guard pos < bytes.count, bytes[pos] == UInt8(ascii: "]") else { throw PackedShapeError.mismatch }
                pos += 1
            } else {
                throw PackedShapeError.mismatch
            }
            count += 1
            skipWhitespace()
            guard pos < bytes.count else { throw JSONConfigError.unexpectedEnd }
            let c = bytes[pos]
            if c == UInt8(ascii: ",") {
                pos += 1
                skipWhitespace()
                if pos < bytes.count, bytes[pos] == UInt8(ascii: "]") {
                    pos += 1
                    break
                }
                continue
            }
            if c == UInt8(ascii: "]") {
                pos += 1
                break
            }
            throw JSONConfigError.unexpectedCharacter(c, offset: pos)
        }
        return Config(data: .stringPairs(PackedStringPairs(utf8: utf8, offsets: offsets, count: count)))
    }

    private mutating func parsePackedScoredTokens() throws -> Config {
        depth += 1
        defer { depth -= 1 }
        guard depth <= Self.maxDepth else { throw JSONConfigError.nestingTooDeep }
        pos += 1  // '['
        skipWhitespace()
        var utf8: [UInt8] = []
        var offsets: [UInt32] = [0]
        var scores: [Double] = []
        utf8.reserveCapacity(min(bytes.count - pos, 8_000_000))
        scores.reserveCapacity(estimatedRemainingEntries(bytesPerEntry: 24))
        if pos < bytes.count, bytes[pos] == UInt8(ascii: "]") {
            pos += 1
            return Config(data: .scoredTokens(PackedScoredTokens(utf8: utf8, offsets: offsets, scores: scores)))
        }
        while true {
            skipWhitespace()
            guard pos < bytes.count, bytes[pos] == UInt8(ascii: "[") else { throw PackedShapeError.mismatch }
            pos += 1
            skipWhitespace()
            guard pos < bytes.count, bytes[pos] == UInt8(ascii: "\"") else { throw PackedShapeError.mismatch }
            try appendJSONString(to: &utf8)
            offsets.append(UInt32(utf8.count))
            skipWhitespace()
            if pos < bytes.count, bytes[pos] == UInt8(ascii: ",") {
                pos += 1
                skipWhitespace()
            }
            let scoreConfig = try parseNumber()
            guard let score = scoreConfig.double(), score.isFinite else { throw PackedShapeError.mismatch }
            scores.append(score)
            skipWhitespace()
            if pos < bytes.count, bytes[pos] == UInt8(ascii: ",") {
                pos += 1
                skipWhitespace()
            }
            guard pos < bytes.count, bytes[pos] == UInt8(ascii: "]") else { throw PackedShapeError.mismatch }
            pos += 1
            skipWhitespace()
            guard pos < bytes.count else { throw JSONConfigError.unexpectedEnd }
            let c = bytes[pos]
            if c == UInt8(ascii: ",") {
                pos += 1
                skipWhitespace()
                if pos < bytes.count, bytes[pos] == UInt8(ascii: "]") {
                    pos += 1
                    break
                }
                continue
            }
            if c == UInt8(ascii: "]") {
                pos += 1
                break
            }
            throw JSONConfigError.unexpectedCharacter(c, offset: pos)
        }
        return Config(data: .scoredTokens(PackedScoredTokens(utf8: utf8, offsets: offsets, scores: scores)))
    }

    /// Parses a JSON integer in place. Returns `nil` (without consuming) when the next value is not an integer.
    private mutating func parseIntegerIfPossible() throws -> Int? {
        guard pos < bytes.count else { throw JSONConfigError.unexpectedEnd }
        let c = bytes[pos]
        if c != UInt8(ascii: "-") && (c < 0x30 || c > 0x39) { return nil }
        let start = pos
        let parsed = try parseNumber()
        if let value = parsed.integer() { return value }
        pos = start
        return nil
    }

    private mutating func appendJSONString(to buffer: inout [UInt8]) throws {
        pos += 1  // opening quote
        let start = pos
        let (i, isASCII) = ByteKernels.jsonStringEnd(bytes, from: pos)
        let n = bytes.count
        if i < n, bytes[i] < 0x20 {
            throw JSONConfigError.unexpectedCharacter(bytes[i], offset: i)
        }
        let hasEscape = i < n && bytes[i] == 0x5C
        if !hasEscape {
            guard i < n else { throw JSONConfigError.unexpectedEnd }
            let slice = UnsafeBufferPointer(rebasing: bytes[start..<i])
            pos = i + 1
            if !isASCII, !Self.isValidUTF8(slice) {
                throw JSONConfigError.invalidUTF8(offset: start)
            }
            buffer.append(contentsOf: slice)
            return
        }
        let string = try parseStringFromEscape(start: start, firstEscape: i)
        var copy = string
        copy.withUTF8 { buffer.append(contentsOf: $0) }
    }

    /// Shared escape decoder for ordinary strings and packed tokenizer tables.
    private mutating func parseStringFromEscape(start: Int, firstEscape: Int) throws -> String {
        scratch.removeAll(keepingCapacity: true)
        scratch.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[start..<firstEscape]))
        pos = firstEscape
        let n = bytes.count
        while pos < n {
            let c = bytes[pos]
            if c == UInt8(ascii: "\"") {
                pos += 1
                return try scratch.withUnsafeBufferPointer {
                    try Self.makeString(from: $0, offset: start, ascii: ByteKernels.isASCII($0))
                }
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

    private static func isValidUTF8(_ bytes: UnsafeBufferPointer<UInt8>) -> Bool {
        var i = 0
        let n = bytes.count
        while i < n {
            let b = bytes[i]
            if b < 0x80 {
                i += 1
                continue
            }
            let remaining = n - i
            if b < 0xC2 || b > 0xF4 { return false }
            if b < 0xE0 {
                guard remaining >= 2, bytes[i + 1] & 0xC0 == 0x80 else { return false }
                i += 2
            } else if b < 0xF0 {
                guard remaining >= 3, bytes[i + 1] & 0xC0 == 0x80, bytes[i + 2] & 0xC0 == 0x80 else {
                    return false
                }
                if b == 0xE0, bytes[i + 1] < 0xA0 { return false }
                if b == 0xED, bytes[i + 1] >= 0xA0 { return false }
                i += 3
            } else {
                guard remaining >= 4, bytes[i + 1] & 0xC0 == 0x80, bytes[i + 2] & 0xC0 == 0x80,
                    bytes[i + 3] & 0xC0 == 0x80
                else {
                    return false
                }
                if b == 0xF0, bytes[i + 1] < 0x90 { return false }
                if b == 0xF4, bytes[i + 1] >= 0x90 { return false }
                i += 4
            }
        }
        return true
    }

    /// Validate bytes before decoding, preserving a leading BOM without a String round trip.
    private static func makeString(
        from slice: UnsafeBufferPointer<UInt8>, offset: Int, ascii: Bool
    ) throws -> String {
        guard ascii || isValidUTF8(slice) else {
            throw JSONConfigError.invalidUTF8(offset: offset)
        }
        return String(decoding: slice, as: UTF8.self)
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
