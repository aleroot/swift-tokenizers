import Foundation

/// A JSON-like value produced by response parsing.
///
/// Objects keep their keys in insertion order, as Python dictionaries do, so a parsed tool call
/// renders its arguments in the order the model wrote them when it is fed back through a chat
/// template.
public enum ResponseValue: Sendable, Equatable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([ResponseValue])
    case object(ResponseObject)

    /// The string, when this value is one.
    public var stringValue: String? {
        if case let .string(value) = self { return value }
        return nil
    }

    /// The object, when this value is one.
    public var objectValue: ResponseObject? {
        if case let .object(value) = self { return value }
        return nil
    }

    /// The integer, when this value is one.
    public var intValue: Int? {
        if case let .int(value) = self { return value }
        return nil
    }

    /// The array, when this value is one.
    public var arrayValue: [ResponseValue]? {
        if case let .array(value) = self { return value }
        return nil
    }

    /// Python truthiness: `None`, `False`, zero and empty containers are false.
    var isTruthy: Bool {
        switch self {
        case .null: false
        case let .bool(value): value
        case let .int(value): value != 0
        case let .double(value): value != 0
        case let .string(value): !value.isEmpty
        case let .array(value): !value.isEmpty
        case let .object(value): !value.isEmpty
        }
    }

    /// Whether `finalize` drops this value from the message: `None` and empty containers.
    var isEmptyForMessage: Bool {
        switch self {
        case .null: true
        case let .string(value): value.isEmpty
        case let .array(value): value.isEmpty
        case let .object(value): value.isEmpty
        default: false
        }
    }

    /// Python's type name, for error messages that mirror `transformers`.
    var pythonTypeName: String {
        switch self {
        case .null: "NoneType"
        case .bool: "bool"
        case .int: "int"
        case .double: "float"
        case .string: "str"
        case .array: "list"
        case .object: "dict"
        }
    }
}

extension ResponseValue: ExpressibleByNilLiteral, ExpressibleByBooleanLiteral, ExpressibleByIntegerLiteral,
    ExpressibleByFloatLiteral, ExpressibleByStringLiteral, ExpressibleByArrayLiteral, ExpressibleByDictionaryLiteral
{
    public init(nilLiteral: ()) { self = .null }
    public init(booleanLiteral value: Bool) { self = .bool(value) }
    public init(integerLiteral value: Int) { self = .int(value) }
    public init(floatLiteral value: Double) { self = .double(value) }
    public init(stringLiteral value: String) { self = .string(value) }
    public init(arrayLiteral elements: ResponseValue...) { self = .array(elements) }
    public init(dictionaryLiteral elements: (String, ResponseValue)...) {
        self = .object(ResponseObject(elements))
    }
}

extension ResponseValue: CustomStringConvertible {
    /// Compact JSON, with object keys in their stored order.
    public var description: String {
        var output = ""
        write(to: &output)
        return output
    }

    func write(to output: inout String) {
        switch self {
        case .null: output += "null"
        case let .bool(value): output += value ? "true" : "false"
        case let .int(value): output += String(value)
        case let .double(value):
            if value.isNaN {
                output += "NaN"
            } else if value.isInfinite {
                output += value < 0 ? "-Infinity" : "Infinity"
            } else {
                output += "\(value)"
            }
        case let .string(value): Self.writeString(value, to: &output)
        case let .array(values):
            output += "["
            for (index, value) in values.enumerated() {
                if index > 0 { output += ", " }
                value.write(to: &output)
            }
            output += "]"
        case let .object(object):
            output += "{"
            for (index, entry) in object.enumerated() {
                if index > 0 { output += ", " }
                Self.writeString(entry.key, to: &output)
                output += ": "
                entry.value.write(to: &output)
            }
            output += "}"
        }
    }

    /// A JSON string literal, escaping only what JSON requires.
    static func writeString(_ value: String, to output: inout String) {
        output += "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": output += "\\\""
            case "\\": output += "\\\\"
            case "\n": output += "\\n"
            case "\r": output += "\\r"
            case "\t": output += "\\t"
            case "\u{08}": output += "\\b"
            case "\u{0C}": output += "\\f"
            case _ where scalar.value < 0x20:
                output +=
                    "\\u" + String(repeating: "0", count: 4 - String(scalar.value, radix: 16).count)
                    + String(scalar.value, radix: 16)
            default: output.unicodeScalars.append(scalar)
            }
        }
        output += "\""
    }
}

/// An object whose keys keep their insertion order.
///
/// Assigning to an existing key replaces its value in place, as a Python dictionary does.
public struct ResponseObject: Sendable, RandomAccessCollection, ExpressibleByDictionaryLiteral {
    public typealias Element = (key: String, value: ResponseValue)

    private var entries: [Element] = []
    private var positions: [String: Int] = [:]

    public init() {}

    public init(_ elements: some Sequence<(String, ResponseValue)>) {
        for (key, value) in elements { self[key] = value }
    }

    public init(dictionaryLiteral elements: (String, ResponseValue)...) {
        self.init(elements)
    }

    public var startIndex: Int { 0 }
    public var endIndex: Int { entries.count }
    public subscript(position: Int) -> Element { entries[position] }

    /// The keys, in insertion order.
    public var keys: [String] { entries.map(\.key) }

    public subscript(key: String) -> ResponseValue? {
        get { positions[key].map { entries[$0].value } }
        set {
            if let index = positions[key] {
                if let newValue {
                    entries[index].value = newValue
                } else {
                    entries.remove(at: index)
                    positions[key] = nil
                    for position in index..<entries.count { positions[entries[position].key] = position }
                }
            } else if let newValue {
                positions[key] = entries.count
                entries.append((key, newValue))
            }
        }
    }

    /// Applies `update` to the value stored under `key`, inserting `defaultValue` first when the
    /// key is missing.
    mutating func update(
        _ key: String, default defaultValue: ResponseValue, _ update: (inout ResponseValue) throws -> Void
    )
        rethrows
    {
        if positions[key] == nil { self[key] = defaultValue }
        try update(&entries[positions[key]!].value)
    }
}

extension ResponseObject: Equatable {
    /// Key order does not matter, as for Python dictionaries.
    public static func == (lhs: ResponseObject, rhs: ResponseObject) -> Bool {
        lhs.count == rhs.count && lhs.allSatisfy { rhs[$0.key] == $0.value }
    }
}

extension ResponseObject: CustomStringConvertible {
    public var description: String { ResponseValue.object(self).description }
}
