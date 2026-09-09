// A JSON-like configuration tree with dynamic member lookup, source-compatible with
// swift-transformers' `Config`.

import Foundation
import Jinja

@dynamicMemberLookup
public struct Config: Hashable, Sendable,
    ExpressibleByStringLiteral,
    ExpressibleByIntegerLiteral,
    ExpressibleByBooleanLiteral,
    ExpressibleByFloatLiteral,
    ExpressibleByDictionaryLiteral,
    ExpressibleByArrayLiteral,
    ExpressibleByExtendedGraphemeClusterLiteral,
    CustomStringConvertible
{
    public typealias Key = BinaryDistinctString
    public typealias Value = Config

    let value: Data

    /// The underlying value kinds a configuration node can hold.
    public enum Data: Sendable {
        case null
        case string(BinaryDistinctString)
        case integer(Int)
        case boolean(Bool)
        /// Stored at full precision; `floating()` narrows to `Float` for API compatibility.
        case floating(Double)
        case dictionary([BinaryDistinctString: Config])
        case array([Config])
        case token((UInt, BinaryDistinctString))
        /// Packed BPE/WordPiece `model.vocab` (`token → id`).
        case stringMap(PackedStringMap)
        /// Packed BPE `model.merges` (ranked pairs).
        case stringPairs(PackedStringPairs)
        /// Packed Unigram `model.vocab` (`[token, score]` rows).
        case scoredTokens(PackedScoredTokens)

        public func string() -> String? {
            if case let .string(v) = self { return v.string }
            return nil
        }

        public func boolean() -> Bool? {
            switch self {
            case let .boolean(v): return v
            case let .integer(v): return v == 1
            case let .string(v):
                switch v.string.lowercased() {
                case "true", "t", "1": return true
                case "false", "f", "0": return false
                default: return nil
                }
            default: return nil
            }
        }

        public func integer() -> Int? {
            if case let .integer(v) = self { return v }
            return nil
        }

        public func floating() -> Float? {
            switch self {
            case let .floating(v): return Float(v)
            case let .integer(v): return Float(v)
            default: return nil
            }
        }

        public func double() -> Double? {
            switch self {
            case let .floating(v): return v
            case let .integer(v): return Double(v)
            default: return nil
            }
        }

        public var description: String {
            switch self {
            case .null: "null"
            case let .string(v): "\"\(v)\""
            case let .integer(v): "\(v)"
            case let .boolean(v): "\(v)"
            case let .floating(v): "\(v)"
            case let .array(a): "[\(a)]"
            case let .dictionary(d): "{\(d)}"
            case let .token(t): "(\(t.0), \(t.1))"
            case let .stringMap(m): "{<\(m.count) tokens>}"
            case let .stringPairs(p): "[<\(p.count) pairs>]"
            case let .scoredTokens(t): "[<\(t.count) pieces>]"
            }
        }

        // Loose equality: numeric / string kinds compare by converted value, so that
        // `Config(1) == Config(true)` and `Config("1") == Config(1)` hold.
        public static func == (lhs: Data, rhs: Data) -> Bool {
            switch (lhs, rhs) {
            case (.null, .null):
                return true
            case let (.string(l), _):
                if let r = rhs.string() { return l == BinaryDistinctString(r) }
            case let (.integer(l), _):
                if let r = rhs.integer() { return l == r }
            case let (.boolean(l), _):
                if let r = rhs.boolean() { return l == r }
            case let (.floating(l), _):
                // Floating values compare at Float precision.
                if let r = rhs.double() { return Float(l) == Float(r) }
            case let (.dictionary(l), .dictionary(r)):
                return l == r
            case let (.array(l), .array(r)):
                return l == r
            case let (.token(l), .token(r)):
                return l == r
            case let (.stringMap(l), .stringMap(r)):
                return l == r
            case let (.stringPairs(l), .stringPairs(r)):
                return l == r
            case let (.scoredTokens(l), .scoredTokens(r)):
                return l == r
            default:
                return false
            }
            switch rhs {
            case let .string(r):
                if let l = lhs.string() { return BinaryDistinctString(l) == r }
            case let .integer(r):
                if let l = lhs.integer() { return l == r }
            case let .boolean(r):
                if let l = lhs.boolean() { return l == r }
            case let .floating(r):
                if let l = lhs.double() { return Float(l) == Float(r) }
            default:
                return false
            }
            return false
        }
    }

    // MARK: - Construction

    public init() { value = .null }

    init(data: Data) { value = data }

    public init(_ value: BinaryDistinctString) { self.value = .string(value) }
    public init(_ value: String) { self.value = .string(BinaryDistinctString(value)) }
    public init(_ value: Int) { self.value = .integer(value) }
    public init(_ value: Bool) { self.value = .boolean(value) }
    public init(_ value: Float) { self.value = .floating(Double(value)) }
    public init(_ value: Double) { self.value = .floating(value) }
    public init(_ value: [Config]) { self.value = .array(value) }
    public init(_ value: [BinaryDistinctString: Config]) { self.value = .dictionary(value) }
    public init(_ token: (UInt, BinaryDistinctString)) { self.value = .token(token) }

    public init(_ values: (BinaryDistinctString, Config)...) {
        var dict = [BinaryDistinctString: Config](minimumCapacity: values.count)
        for (k, v) in values { dict[k] = v }
        value = .dictionary(dict)
    }

    public init(_ dictionary: [String: Config]) {
        var dict = [BinaryDistinctString: Config](minimumCapacity: dictionary.count)
        for (k, v) in dictionary { dict[BinaryDistinctString(k)] = v }
        value = .dictionary(dict)
    }

    public init(_ dictionary: [NSString: Config]) {
        var dict = [BinaryDistinctString: Config](minimumCapacity: dictionary.count)
        for (k, v) in dictionary { dict[BinaryDistinctString(k)] = v }
        value = .dictionary(dict)
    }

    /// Builds a configuration from a loosely-typed dictionary (e.g. the output of
    /// `JSONSerialization`). Unknown leaf types are converted to `null`.
    public init(_ dictionary: [NSString: Any]) {
        value = Config.convert(dictionary as Any).value
    }

    /// Builds a configuration from a loosely-typed value tree.
    public init(any object: Any) {
        value = Config.convert(object).value
    }

    private static func convert(_ object: Any) -> Config {
        switch object {
        case let dict as [NSString: Any]:
            var out = [BinaryDistinctString: Config](minimumCapacity: dict.count)
            for (k, v) in dict { out[BinaryDistinctString(k)] = convert(v) }
            return Config(out)
        case let dict as [String: Any]:
            var out = [BinaryDistinctString: Config](minimumCapacity: dict.count)
            for (k, v) in dict { out[BinaryDistinctString(k)] = convert(v) }
            return Config(out)
        case let array as [Any]:
            return Config(array.map { convert($0) })
        case let obj as Config:
            return obj
        case let obj as BinaryDistinctString:
            return Config(obj)
        case let obj as String:
            return Config(obj)
        case let obj as NSNumber where CFGetTypeID(obj) == CFBooleanGetTypeID():
            return Config(obj.boolValue)
        case let obj as NSNumber:
            // JSONSerialization bridges numeric 0/1 to Bool as well. Inspect the CF type
            // before Swift casts so vocabulary IDs never become boolean nodes.
            let type = String(cString: obj.objCType)
            if type == "f" || type == "d" { return Config(obj.doubleValue) }
            if type == "Q", obj.uint64Value > UInt64(Int.max) { return Config(obj.doubleValue) }
            return Config(obj.intValue)
        case let obj as Bool:
            return Config(obj)
        case let obj as Int:
            return Config(obj)
        case let obj as UInt:
            return Int(exactly: obj).map { Config($0) } ?? Config(Double(obj))
        case let obj as Float:
            return Config(obj)
        case let obj as Double:
            return Config(obj)
        case let obj as (UInt, String):
            return Config((obj.0, BinaryDistinctString(obj.1)))
        case let obj as (UInt, BinaryDistinctString):
            return Config(obj)
        case is NSNull:
            return Config()
        default:
            return Config()
        }
    }

    // MARK: Literals

    public init(stringLiteral value: String) { self.value = .string(BinaryDistinctString(value)) }
    public init(integerLiteral value: Int) { self.value = .integer(value) }
    public init(booleanLiteral value: Bool) { self.value = .boolean(value) }
    public init(floatLiteral value: Double) { self.value = .floating(value) }

    public init(dictionaryLiteral elements: (BinaryDistinctString, Config)...) {
        var dict = [BinaryDistinctString: Config](minimumCapacity: elements.count)
        for (k, v) in elements { dict[k] = v }
        value = .dictionary(dict)
    }

    public init(arrayLiteral elements: Config...) { value = .array(elements) }

    // MARK: - Inspection

    public func isNull() -> Bool {
        if case .null = value { return true }
        return false
    }

    public var description: String { value.description }

    // MARK: Getters — string

    public func get() -> String? { string() }
    public func get(or: String) -> String? { string(or: or) }
    public func string() -> String? { value.string() }
    public func string(or: String) -> String { string() ?? or }

    public func get() -> BinaryDistinctString? { binaryDistinctString() }
    public func get(or: BinaryDistinctString) -> BinaryDistinctString? { binaryDistinctString(or: or) }

    public func binaryDistinctString() -> BinaryDistinctString? {
        if case let .string(v) = value { return v }
        return nil
    }

    public func binaryDistinctString(or: BinaryDistinctString) -> BinaryDistinctString {
        binaryDistinctString() ?? or
    }

    // MARK: Getters — boolean

    public func get() -> Bool? { boolean() }
    public func get(or: Bool) -> Bool? { boolean(or: or) }
    public func boolean() -> Bool? { value.boolean() }
    public func boolean(or: Bool) -> Bool { boolean() ?? or }

    // MARK: Getters — integer

    public func get() -> Int? { integer() }
    public func get(or: Int) -> Int? { integer(or: or) }
    public func integer() -> Int? { value.integer() }
    public func integer(or: Int) -> Int { integer() ?? or }

    // MARK: Getters — floating

    public func get() -> Float? { floating() }
    public func get(or: Float) -> Float? { floating(or: or) }
    public func floating() -> Float? { value.floating() }
    public func floating(or: Float) -> Float { floating() ?? or }
    public func double() -> Double? { value.double() }
    public func double(or: Double) -> Double { double() ?? or }

    // MARK: Getters — dictionary

    public func get() -> [BinaryDistinctString: Int]? {
        guard let dict = dictionary() else { return nil }
        var out = [BinaryDistinctString: Int](minimumCapacity: dict.count)
        for (k, v) in dict {
            if let i = v.value.integer() { out[k] = i }
        }
        return out
    }

    public func get() -> [BinaryDistinctString: Config]? { dictionary() }
    public func get(or: [BinaryDistinctString: Config]) -> [BinaryDistinctString: Config] { dictionary(or: or) }

    public func dictionary() -> [BinaryDistinctString: Config]? {
        switch value {
        case let .dictionary(v): return v
        case let .stringMap(packed): return packed.materializeDictionary()
        default: return nil
        }
    }

    func asPackedStringMap() -> PackedStringMap? {
        if case let .stringMap(v) = value { return v }
        return nil
    }

    func asPackedStringPairs() -> PackedStringPairs? {
        if case let .stringPairs(v) = value { return v }
        return nil
    }

    func asPackedScoredTokens() -> PackedScoredTokens? {
        if case let .scoredTokens(v) = value { return v }
        return nil
    }

    public func dictionary(or: [BinaryDistinctString: Config]) -> [BinaryDistinctString: Config] {
        dictionary() ?? or
    }

    // MARK: Getters — array

    public func get() -> [String]? {
        guard let arr = array() else { return nil }
        var out: [String] = []
        out.reserveCapacity(arr.count)
        for element in arr {
            if let s = element.value.string() { out.append(s) }
        }
        return out
    }

    public func get(or: [String]) -> [String] {
        if let arr: [String] = get() { return arr }
        return or
    }

    public func get() -> [BinaryDistinctString]? {
        guard let arr = array() else { return nil }
        var out: [BinaryDistinctString] = []
        out.reserveCapacity(arr.count)
        for element in arr {
            if let s = element.binaryDistinctString() { out.append(s) }
        }
        return out
    }

    public func get(or: [BinaryDistinctString]) -> [BinaryDistinctString] {
        if let arr: [BinaryDistinctString] = get() { return arr }
        return or
    }

    public func get() -> [Config]? { array() }
    public func get(or: [Config]) -> [Config] { array(or: or) }

    public func array() -> [Config]? {
        switch value {
        case let .array(v): return v
        case let .stringPairs(packed): return packed.materializeArray()
        case let .scoredTokens(packed): return packed.materializeArray()
        default: return nil
        }
    }

    public func array(or: [Config]) -> [Config] { array() ?? or }

    // MARK: Getters — token

    public func get() -> (UInt, String)? { token() }
    public func get(or: (UInt, String)) -> (UInt, String) { token(or: or) }

    /// A `(id, content)` token tuple. Accepts either a native token node or a two element
    /// `[content, id]` array as serialized by `tokenizers` for `sep` / `cls` entries.
    public func token() -> (UInt, String)? {
        if case let .token(v) = value { return (v.0, v.1.string) }
        if case let .array(arr) = value {
            guard arr.count == 2, let token = arr[0].string(), let id = arr[1].integer(), id >= 0 else { return nil }
            return (UInt(id), token)
        }
        return nil
    }

    public func token(or: (UInt, String)) -> (UInt, String) { token() ?? or }

    // MARK: Subscripts

    public subscript(index: BinaryDistinctString) -> Config {
        switch value {
        case let .dictionary(dict):
            return dict[index] ?? dict[Config.uncamelCase(index)] ?? Config()
        case let .stringMap(packed):
            var copy = index.string
            let id = copy.withUTF8 { packed.id(of: $0) }
            return id >= 0 ? Config(Int(id)) : Config()
        default:
            return Config()
        }
    }

    public subscript(index: Int) -> Config {
        guard case let .array(arr) = value, index >= 0, index < arr.count else { return Config() }
        return arr[index]
    }

    public subscript(dynamicMember member: String) -> Config? {
        guard case let .dictionary(dict) = value else { return nil }
        let key = BinaryDistinctString(member)
        return dict[key] ?? dict[Config.uncamelCase(key)] ?? Config()
    }

    public subscript(dynamicMember member: String) -> Config {
        guard case let .dictionary(dict) = value else { return Config() }
        let key = BinaryDistinctString(member)
        return dict[key] ?? dict[Config.uncamelCase(key)] ?? Config()
    }

    /// `addPrefixSpace` → `add_prefix_space`
    static func uncamelCase(_ key: BinaryDistinctString) -> BinaryDistinctString {
        let utf8 = key.string.utf8
        // Fast path: no ASCII uppercase means nothing to convert.
        var hasUpper = false
        for b in utf8 where b >= 0x41 && b <= 0x5A {
            hasUpper = true
            break
        }
        guard hasUpper else { return key }

        var result = ""
        result.reserveCapacity(key.string.utf8.count + 4)
        var previousWasLower = false
        for scalar in key.string.unicodeScalars {
            if scalar.properties.isUppercase {
                if previousWasLower { result += "_" }
                result += String(scalar).lowercased()
                previousWasLower = false
            } else {
                result.unicodeScalars.append(scalar)
                previousWasLower = true
            }
        }
        return BinaryDistinctString(result)
    }

    // MARK: Jinja

    /// Converts the configuration tree into a Jinja template value.
    public func jinjaValue() -> Jinja.Value {
        switch value {
        case let .array(arr):
            return .array(arr.map { $0.jinjaValue() })
        case let .dictionary(dict):
            var result = OrderedDictionary<String, Jinja.Value>()
            for (k, v) in dict { result[k.string] = v.jinjaValue() }
            return .object(result)
        case let .boolean(b): return .boolean(b)
        case let .floating(f): return .double(f)
        case let .integer(i): return .int(i)
        case let .string(s): return .string(s.string)
        case let .token(t): return .object([String(t.0): .string(t.1.string)])
        case let .stringMap(m):
            return Config(m.materializeDictionary()).jinjaValue()
        case let .stringPairs(p):
            return Config(p.materializeArray()).jinjaValue()
        case let .scoredTokens(t):
            return Config(t.materializeArray()).jinjaValue()
        case .null: return .null
        }
    }
}

// MARK: - Equatable / Hashable

extension Config: Equatable {
    public static func == (lhs: Config, rhs: Config) -> Bool { lhs.value == rhs.value }
}

extension Config.Data: Hashable {
    public func hash(into hasher: inout Hasher) {
        switch self {
        case .null:
            hasher.combine(0)
        case let .string(s):
            hasher.combine(1)
            hasher.combine(s)
        case let .integer(i):
            hasher.combine(2)
            hasher.combine(i)
        case let .boolean(b):
            hasher.combine(3)
            hasher.combine(b)
        case let .floating(f):
            hasher.combine(4)
            hasher.combine(Float(f))
        case let .dictionary(d):
            hasher.combine(5)
            d.hash(into: &hasher)
        case let .array(a):
            hasher.combine(6)
            for e in a { e.hash(into: &hasher) }
        case let .token(t):
            hasher.combine(7)
            hasher.combine(t.0)
            hasher.combine(t.1)
        case let .stringMap(m):
            hasher.combine(8)
            m.hash(into: &hasher)
        case let .stringPairs(p):
            hasher.combine(9)
            p.hash(into: &hasher)
        case let .scoredTokens(t):
            hasher.combine(10)
            t.hash(into: &hasher)
        }
    }
}

// MARK: - Deprecated accessors kept for source compatibility

public extension Config {
    @available(*, deprecated, message: "Use string() instead")
    var stringValue: String? { string() }

    @available(*, deprecated, message: "Use integer() instead")
    var intValue: Int? { integer() }

    @available(*, deprecated, message: "Use boolean() instead")
    var boolValue: Bool? { boolean() }

    @available(*, deprecated, message: "Use array() instead")
    var arrayValue: [Config]? { array() }

    @available(*, deprecated, message: "Use token() instead")
    var tokenValue: (UInt, String)? { token() }
}

// MARK: - Codable

extension Config: Codable {
    public init(from decoder: any Swift.Decoder) throws {
        if let container = try? decoder.singleValueContainer() {
            if container.decodeNil() {
                value = .null
                return
            }
            if let i = try? container.decode(Int.self) {
                value = .integer(i)
                return
            }
            if let f = try? container.decode(Double.self) {
                value = .floating(f)
                return
            }
            if let b = try? container.decode(Bool.self) {
                value = .boolean(b)
                return
            }
            if let s = try? container.decode(String.self) {
                value = .string(BinaryDistinctString(s))
                return
            }
        }

        if let token = Self.decodeToken(decoder) {
            value = token
            return
        }

        if var unkeyed = try? decoder.unkeyedContainer() {
            var elements: [Config] = []
            while !unkeyed.isAtEnd {
                elements.append(try unkeyed.decode(Config.self))
            }
            value = .array(elements)
            return
        }

        let keyed = try decoder.container(keyedBy: CodingKeys.self)
        var dict = [BinaryDistinctString: Config]()
        for key in keyed.allKeys {
            dict[BinaryDistinctString(key.stringValue)] = try keyed.decode(Config.self, forKey: key)
        }
        value = .dictionary(dict)
    }

    private static func decodeToken(_ decoder: any Swift.Decoder) -> Data? {
        guard var container = try? decoder.unkeyedContainer(), container.count == 2 else { return nil }
        guard let id = try? container.decode(UInt.self), let content = try? container.decode(String.self) else {
            return nil
        }
        return .token((id, BinaryDistinctString(content)))
    }

    public func encode(to encoder: any Swift.Encoder) throws {
        switch value {
        case .null:
            var c = encoder.singleValueContainer()
            try c.encodeNil()
        case let .integer(v):
            var c = encoder.singleValueContainer()
            try c.encode(v)
        case let .floating(v):
            var c = encoder.singleValueContainer()
            try c.encode(v)
        case let .boolean(v):
            var c = encoder.singleValueContainer()
            try c.encode(v)
        case let .string(v):
            var c = encoder.singleValueContainer()
            try c.encode(v.string)
        case let .dictionary(v):
            var c = encoder.container(keyedBy: CodingKeys.self)
            for (k, val) in v {
                try c.encode(val, forKey: CodingKeys(stringValue: k.string)!)
            }
        case let .array(v):
            var c = encoder.unkeyedContainer()
            try c.encode(contentsOf: v)
        case let .token(v):
            var c = encoder.unkeyedContainer()
            try c.encode(v.0)
            try c.encode(v.1.string)
        case let .stringMap(m):
            try Config(m.materializeDictionary()).encode(to: encoder)
        case let .stringPairs(p):
            try Config(p.materializeArray()).encode(to: encoder)
        case let .scoredTokens(t):
            try Config(t.materializeArray()).encode(to: encoder)
        }
    }

    private struct CodingKeys: CodingKey {
        var stringValue: String
        init?(stringValue: String) { self.stringValue = stringValue }
        var intValue: Int? { nil }
        init?(intValue: Int) { nil }
    }
}

public enum ConfigError: Error {
    case typeMismatch(expected: Config.Data, actual: Config.Data)
    case typeConversionFailed(value: any Sendable, targetType: any Sendable.Type)
}
