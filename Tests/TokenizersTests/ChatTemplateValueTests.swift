// Key ordering of dictionaries rendered through chat templates. Expected strings are
// Python `json.dumps` output: `transformers.utils.get_json_schema` for the tool, and an
// OpenAI-format tool-call message — i.e. exactly what `tools | tojson` renders in
// `transformers`.

import Foundation
import Jinja
import Testing

@testable import Tokenizers

private func tojson(_ value: Any) throws -> String {
    try Template("{{ v | tojson }}").render(["v": ChatTemplateValue.make(value)])
}

@Suite("Chat template value ordering")
struct ChatTemplateValueTests {
    /// The HF `get_json_schema` tool, written as a plain Swift dictionary literal (whose key
    /// order is *not* preserved by `Dictionary`).
    static let tool: [String: any Sendable] = [
        "function": [
            "parameters": [
                "required": ["city"],
                "type": "object",
                "properties": [
                    "unit": ["description": "Temperature unit", "enum": ["celsius", "fahrenheit"], "type": "string"]
                        as [String: any Sendable],
                    "tags": [
                        "description": "Extra tags", "items": ["type": "string"] as [String: any Sendable],
                        "type": "array",
                    ] as [String: any Sendable],
                    "city": ["description": "The city name", "type": "string"] as [String: any Sendable],
                    "days": ["description": "Forecast horizon", "nullable": true, "type": "integer"]
                        as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
            "return": ["description": "The forecast", "type": "object"] as [String: any Sendable],
            "description": "Get the current weather in a city.",
            "name": "get_weather",
        ] as [String: any Sendable],
        "type": "function",
    ]

    @Test("Dictionaries render in the canonical get_json_schema order")
    func canonicalToolOrder() throws {
        // Parameter names (children of `properties`) have no canonical order: alphabetical.
        let expected = """
            {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather in a city.", \
            "parameters": {"type": "object", "properties": {\
            "city": {"type": "string", "description": "The city name"}, \
            "days": {"type": "integer", "nullable": true, "description": "Forecast horizon"}, \
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Extra tags"}, \
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}}, \
            "required": ["city"]}, "return": {"type": "object", "description": "The forecast"}}}
            """
        #expect(try tojson(Self.tool) == expected)
    }

    @Test("Tool-call messages render in OpenAI order")
    func toolCallMessageOrder() throws {
        let message: [String: any Sendable] = [
            "tool_calls": [
                [
                    "function": [
                        "arguments": ["unit": "celsius", "city": "Paris"] as [String: any Sendable],
                        "name": "get_weather",
                    ] as [String: any Sendable],
                    "type": "function",
                    "id": "call_1",
                ] as [String: any Sendable]
            ],
            "content": NSNull(),
            "role": "assistant",
        ]
        // json.dumps of the OpenAI message: arguments (user-named) alphabetical.
        let expected = """
            {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", \
            "function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}}]}
            """
        #expect(try tojson(message) == expected)
    }

    @Test("KeyValuePairs and Jinja values preserve the caller's order anywhere")
    func orderPreservingContainers() throws {
        // Parameters declared in a deliberate, non-alphabetical order.
        let properties: KeyValuePairs<String, any Sendable> = [
            "unit": ["type": "string"] as [String: any Sendable],
            "city": ["type": "string"] as [String: any Sendable],
        ]
        let tool: [String: any Sendable] = [
            "type": "function",
            "function": [
                "name": "get_weather",
                "parameters": ["type": "object", "properties": properties, "required": ["unit", "city"]]
                    as [String: any Sendable],
            ] as [String: any Sendable],
        ]
        #expect(
            try tojson(tool) == """
                {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", \
                "properties": {"unit": {"type": "string"}, "city": {"type": "string"}}, "required": ["unit", "city"]}}}
                """
        )

        // An ordered literal overrides the canonical order too, at any level.
        let ordered: KeyValuePairs<String, any Sendable> = ["zeta": 1, "type": "x", "alpha": true]
        #expect(try tojson(ordered) == #"{"zeta": 1, "type": "x", "alpha": true}"#)
        #expect(
            try tojson(["function": ordered, "type": "function"] as [String: any Sendable])
                == #"{"type": "function", "function": {"zeta": 1, "type": "x", "alpha": true}}"#)

        // A pre-built Jinja value passes through untouched.
        let value: Value = .object(["b": .int(2), "a": .int(1)])
        #expect(try tojson(value) == #"{"b": 2, "a": 1}"#)
    }

    @Test("Scalars: booleans stay booleans, integers stay integers")
    func scalars() throws {
        let dict: [String: any Sendable] = ["flag": true, "n": 1, "x": 2.5, "s": "t", "none": NSNull()]
        #expect(try tojson(dict) == #"{"flag": true, "n": 1, "none": null, "s": "t", "x": 2.5}"#)
    }

    @Test("Unknown keys follow canonical keywords, alphabetically")
    func unknownKeys() throws {
        let node: [String: any Sendable] = ["x-custom": 1, "type": "string", "description": "d", "a-custom": 2]
        #expect(try tojson(node) == #"{"type": "string", "description": "d", "a-custom": 2, "x-custom": 1}"#)
    }
}
