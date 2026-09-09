// Converts chat-template inputs (`[String: any Sendable]` messages, tools and extra context)
// into Jinja values. `Dictionary` keys are laid out in the canonical order of
// `transformers.utils.get_json_schema` (user-named keys alphabetically); `KeyValuePairs` and
// pre-built `Jinja.Value`s render in the order given.

import Foundation
import Jinja

enum ChatTemplateValue {
    /// Converts a message, tool, or context value into a Jinja value.
    static func make(_ value: Any?) throws -> Value {
        try convert(value, userKeyed: false)
    }

    /// Keys whose object values are keyed by *user-chosen* names (parameter names, argument
    /// names, definition names) rather than by schema keywords.
    private static let userKeyedContainers: Set<String> = [
        "properties", "arguments", "patternProperties", "$defs", "definitions", "dependentRequired",
    ]

    /// Canonical position of every schema keyword. Order follows
    /// `transformers.utils.get_json_schema` and the OpenAI function-calling format:
    ///
    /// * tool envelope: `type`, `function`; tool call: `id`, `type`, `function` (+ `index`)
    /// * function: `name`, `description`, `parameters` | `arguments`, `return`, `strict`
    /// * schema node: `type`, `items` | `prefixItems`, `nullable`, `enum`, `description`,
    ///   `properties`, `required`, `additionalProperties`, …
    /// * chat message: `role`, `content`, `name`, `tool_calls`, `tool_call_id`, …
    private static let canonicalRank: [String: Int] = {
        let order: [String] = [
            "role", "id", "index", "type", "function",
            "name", "items", "prefixItems", "nullable", "enum", "const", "description",
            "parameters", "arguments", "return", "returns", "strict",
            "content", "reasoning_content", "reasoning", "thinking", "tool_calls", "tool_call_id", "tool_calls_id",
            "title", "properties", "required", "additionalProperties", "default", "examples",
            "format", "pattern", "minLength", "maxLength",
            "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
            "minItems", "maxItems", "uniqueItems", "minProperties", "maxProperties",
            "anyOf", "oneOf", "allOf", "not", "$ref", "$defs", "definitions", "$schema", "$id",
        ]
        var rank: [String: Int] = [:]
        for (i, key) in order.enumerated() { rank[key] = i }
        return rank
    }()

    private static func convert(_ value: Any?, userKeyed: Bool) throws -> Value {
        switch value {
        case nil:
            return .null
        case is NSNull:
            return .null
        case let value as Value:
            return value
        case let string as String:
            return .string(string)
        case let number as NSNumber where CFGetTypeID(number) == CFBooleanGetTypeID():
            // A Swift `Bool` (or a JSON boolean from JSONSerialization); must precede the
            // integer cases because `NSNumber(true) as? Int` succeeds.
            return .boolean(number.boolValue)
        case let number as NSNumber:
            let type = String(cString: number.objCType)
            if type == "f" || type == "d" { return .double(number.doubleValue) }
            if type == "Q", number.uint64Value > UInt64(Int.max) { return .double(number.doubleValue) }
            return .int(number.intValue)
        case let bool as Bool:
            return .boolean(bool)
        case let int as Int:
            return .int(int)
        case let int as any BinaryInteger:
            return Int(exactly: int).map(Value.int) ?? .double(Double(int))
        case let double as Double:
            return .double(double)
        case let float as Float:
            return .double(Double(float))

        case let pairs as KeyValuePairs<String, any Sendable>:
            return try object(pairs.map { ($0.key, $0.value as Any?) })
        case let pairs as KeyValuePairs<String, Any>:
            return try object(pairs.map { ($0.key, $0.value as Any?) })
        case let pairs as KeyValuePairs<String, Any?>:
            return try object(pairs.map { ($0.key, $0.value) })

        // Unordered containers: canonical schema order, or alphabetical for user-named keys.
        case let dict as [String: any Sendable]:
            return try convertDictionary(dict.mapValues { $0 as Any? }, userKeyed: userKeyed)
        case let dict as [String: Any]:
            return try convertDictionary(dict.mapValues { $0 as Any? }, userKeyed: userKeyed)
        case let dict as [String: Any?]:
            return try convertDictionary(dict, userKeyed: userKeyed)

        case let array as [any Sendable]:
            return .array(try array.map { try convert($0, userKeyed: false) })
        case let array as [Any]:
            return .array(try array.map { try convert($0, userKeyed: false) })
        case let array as [Any?]:
            return .array(try array.map { try convert($0, userKeyed: false) })

        case let .some(wrapped):
            // A nested `Optional` erased into `Any` (e.g. `Optional<Any>.some(.none)`): unwrap it.
            let mirror = Mirror(reflecting: wrapped)
            guard mirror.displayStyle == .optional else {
                throw TokenizerError.chatTemplate(
                    "cannot convert a value of type \(type(of: wrapped)) to a template value")
            }
            return try convert(mirror.children.first?.value, userKeyed: userKeyed)
        }
    }

    private static func convertDictionary(_ dict: [String: Any?], userKeyed: Bool) throws -> Value {
        let keys = userKeyed ? dict.keys.sorted() : dict.keys.sorted(by: canonicalLess)
        return try object(keys.map { ($0, dict[$0] ?? nil) })
    }

    /// Builds an object from already-ordered pairs, converting each value with the right
    /// keying context for its key.
    private static func object(_ pairs: [(String, Any?)]) throws -> Value {
        var storage = emptyObjectStorage
        for (key, value) in pairs {
            storage[.string(key)] = try convert(value, userKeyed: userKeyedContainers.contains(key))
        }
        return .object(storage)
    }

    /// An empty instance of the insertion-ordered storage behind `Value.object`, obtained
    /// through `Value` itself so this module never names — or depends on — swift-jinja's
    /// collection dependency. Copy-on-write makes copying it free.
    private static let emptyObjectStorage = {
        guard case let .object(storage) = Value.object([:] as [String: Value]) else {
            preconditionFailure("Value.object must produce an object value")
        }
        return storage
    }()

    /// Canonical keywords first (in schema order), then everything else alphabetically.
    private static func canonicalLess(_ a: String, _ b: String) -> Bool {
        switch (canonicalRank[a], canonicalRank[b]) {
        case let (ra?, rb?): return ra < rb
        case (.some, nil): return true
        case (nil, .some): return false
        case (nil, nil): return a < b
        }
    }
}
