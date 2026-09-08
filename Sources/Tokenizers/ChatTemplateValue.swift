// Conversion of chat-template inputs (`[String: any Sendable]` messages, tools and extra
// context) into Jinja values, with deterministic and *schema-aware* key ordering.
//
// Why this exists
// ---------------
// Python `transformers` renders `tools | tojson` in dictionary insertion order — the order
// in which the caller wrote the JSON. A Swift `Dictionary` has no insertion order, so a
// naïve conversion (such as swift-jinja's `Value(any:)`) sorts keys
// alphabetically and renders `{"function": {"description": …, "name": …}, "type": …}` —
// a shape no model was trained on.
//
// Tool specifications are not arbitrary JSON, though: they follow the OpenAI
// function-calling schema, and `transformers.utils.get_json_schema` — the generator whose
// output the chat templates were trained against — emits its keywords in one fixed order.
// `Dictionary` inputs are therefore laid out in that canonical order, which reproduces the
// Python rendering byte for byte for every standard tool definition. Keys that are not
// schema keywords, and the user-named children of `properties` / `arguments` / `$defs`,
// have no canonical order; they keep the caller's order when it is available (see below)
// and fall back to alphabetical otherwise.
//
// Callers who need full control can nest order-preserving values anywhere inside a
// message or tool: `KeyValuePairs<String, any Sendable>` (standard library; dictionary-
// literal syntax, keeps literal order, `Sendable`) or a pre-built `Jinja.Value`. These
// render exactly in the order given.

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

        // Order-preserving containers.
        case let pairs as KeyValuePairs<String, any Sendable>:
            return try object(pairs.map { ($0.key, $0.value as Any?) })
        case let pairs as KeyValuePairs<String, Any>:
            return try object(pairs.map { ($0.key, $0.value as Any?) })
        case let pairs as KeyValuePairs<String, Any?>:
            return try object(pairs.map { ($0.key, $0.value) })

        // Unordered containers: canonical schema order, or alphabetical for user-named keys.
        case let dict as [String: Any?]:
            let keys = userKeyed ? dict.keys.sorted() : dict.keys.sorted(by: canonicalLess)
            return try object(keys.map { ($0, dict[$0] ?? nil) })

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
