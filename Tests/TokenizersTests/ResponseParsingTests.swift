import Foundation
import Testing

@testable import Tokenizers

@Suite("Response parsing parity with transformers")
struct ResponseParsingTests {
    struct Case: CustomTestStringConvertible, Sendable {
        let index: Int
        let record: ResponseObject
        var testDescription: String { "#\(index) \(record["source"]?.stringValue ?? "")" }
    }

    /// Calls recorded from `transformers.utils.chat_parsing`: its own test suite plus seeded
    /// fuzzing over real-model templates.
    static let cases: [Case] = {
        let url =
            ProcessInfo.processInfo.environment["RESPONSE_PARSING_GOLDEN"].map(URL.init(fileURLWithPath:))
            ?? Bundle.module.url(forResource: "response_parsing", withExtension: "json")!
        let golden = try! ResponseJSON.decode(String(contentsOf: url, encoding: .utf8))
        return golden.objectValue!["cases"]!.arrayValue!.enumerated().map { Case(index: $0, record: $1.objectValue!) }
    }()

    @Test("Events, messages and errors match transformers", arguments: cases)
    func matchesTransformers(_ test: Case) throws {
        let record = test.record
        let tools = record["tools"].flatMap { $0 == .null ? nil : $0.arrayValue }?.map { $0.sendable as! ToolSpec }
        var parser: ResponseParser
        do {
            let template = try ResponseTemplate(jsonString: record["template"]!.stringValue!)
            parser = try ResponseParser(template: template, prefix: record["prefix"]!.stringValue!, tools: tools)
        } catch {
            #expect(record["init_error"] != nil, "unexpected error \(error)")
            return
        }
        #expect(record["init_error"] == nil, "transformers raised \(record["init_error"]!)")
        guard record["init_error"] == nil else { return }

        #expect(Self.same(parser.initialEvents, try events(record["initial_events"]!)), "initial events")
        let steps = zip(record["chunks"]!.arrayValue!, record["steps"]!.arrayValue!).map {
            ($0.stringValue!, $1.objectValue!)
        }
        let finalizedAfter = record["finalized_after"]?.intValue ?? steps.count
        for (index, (chunk, step)) in steps.enumerated() {
            if index == finalizedAfter, !(try finalize(&parser, record)) { return }
            do {
                let produced = try parser.feed(chunk)
                guard let wanted = step["events"] else {
                    Issue.record("transformers raised \(step["error"]!) on \(chunk.debugDescription)")
                    return
                }
                #expect(Self.same(produced, try events(wanted)), "feed \(chunk.debugDescription)")
            } catch {
                #expect(step["error"] != nil, "unexpected error \(error)")
                return
            }
        }
        if finalizedAfter == steps.count { _ = try finalize(&parser, record) }
        if record["finalized_again"] != nil {
            #expect(throws: ResponseParsingError.alreadyFinalized) { try parser.finalize() }
        }
    }

    /// Finalizes as transformers did; whether the parser is still usable afterwards.
    private func finalize(_ parser: inout ResponseParser, _ record: ResponseObject) throws -> Bool {
        guard let final = record["final"]?.objectValue else { return true }
        do {
            let (message, produced) = try parser.finalize()
            guard let wantedMessage = final["message"] else {
                Issue.record("transformers raised \(final["error"]!) on finalize")
                return false
            }
            #expect(Self.same(.object(message), wantedMessage), "message \(message)")
            #expect(Self.same(produced, try events(final["events"]!)), "final events")
        } catch {
            #expect(final["error"] != nil, "unexpected error \(error)")
            return false
        }
        return true
    }

    // MARK: Behaviour beyond the recorded calls

    @Test("Gemma 4's published template parses from tokenizer_config.json")
    func gemma4FromConfiguration() throws {
        // `response_template` as google/gemma-4-E2B-it ships it.
        let configuration = try Config(
            jsonString: #"""
                {"tokenizer_class": "GemmaTokenizer", "response_template": {
                  "defaults": {"role": "assistant"},
                  "fields": {
                    "content": {"close": ["<turn|>", "<|tool_response>", "<eos>"], "content": "text"},
                    "thinking": {"close": "<channel|>", "content": "text", "open": "<|channel>thought\n"},
                    "tool_calls": {"close": "<tool_call|>", "content": "json",
                      "content_args": {"string_delims": [["<|\"|>", "<|\"|>"]], "unquoted_keys": true},
                      "open_pattern": "<\\|tool_call>call:(?P<name>\\w+)", "repeats": true,
                      "transform": {"function": {"arguments": "{content}", "name": "{name}"}, "type": "function"}}},
                  "start_anchor": ["<|turn>model\n", "<tool_response|>"]}}
                """#)
        let tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: configuration, tokenizerData: ["model": ["type": "BPE", "vocab": ["a": 0], "merges": []]])
        let days: [String: any Sendable] = ["days": ["type": "integer"] as [String: any Sendable]]
        let function: [String: any Sendable] = [
            "name": "get_weather", "parameters": ["properties": days] as [String: any Sendable],
        ]
        let tools: [ToolSpec] = [["type": "function", "function": function]]
        let message = try tokenizer.parseResponse(
            #"<|channel>thought\#nCheck the tool.<channel|><|tool_call>call:get_weather{location:<|"|>Paris, FR<|"|>,days:"3"}<tool_call|>"#,
            prefix: "<|turn>user\nWeather?<turn|>\n<|turn>model\n", tools: tools)
        #expect(
            message
                == [
                    "role": "assistant", "thinking": "Check the tool.",
                    "tool_calls": [
                        [
                            "type": "function",
                            "function": ["name": "get_weather", "arguments": ["location": "Paris, FR", "days": 3]],
                        ]
                    ],
                ])
        // Arguments keep the order the model wrote them in when rendered by a chat template.
        let rendered = try tokenizer.renderChatTemplate(
            messages: [message.message],
            chatTemplate: .literal("{{ messages[0].tool_calls[0].function.arguments | tojson }}"))
        #expect(rendered == #"{"location": "Paris, FR", "days": 3}"#)
    }

    @Test("Tokenizers without a response template say so")
    func missingTemplate() throws {
        let tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: [:], tokenizerData: ["model": ["type": "BPE", "vocab": ["a": 0], "merges": []]])
        #expect(try tokenizer.responseTemplate == nil)
        #expect(throws: ResponseParsingError.missingTemplate) { try tokenizer.parseResponse("a", prefix: "") }
    }

    @Test("Python regex syntax compiles for ICU")
    func regexTranslation() throws {
        let regex = try PythonRegex(#"<(?P<tool_name>\w+)>(?P<body>.*?)</(?P=tool_name)>\Z"#)
        #expect(regex.groupNames == ["tool_name", "body"])
        let match = try #require(regex.matches(in: "<get_x>a\nb</get_x>").first)
        #expect(match.captures.map(\.value) == ["get_x", "a\nb"])
        #expect(regex.matches(in: "<a>b</a>\n").isEmpty)  // `\Z` is the very end, unlike ICU's
        #expect(try PythonRegex("[[]{x}").matches(in: "[{x}").count == 1)  // literal `[` and `{`
    }

    @Test("Required literals are found without misreading the pattern")
    func requiredLiterals() throws {
        func literal(_ pattern: String) throws -> (String, Bool) {
            let regex = try PythonRegex(pattern)
            return (String(decoding: regex.requiredLiteral, as: UTF16.self), regex.startsWithRequiredLiteral)
        }
        #expect(try literal(#"<\|tool_call>call:(?P<name>\w+)"#) == ("<|tool_call>call:", true))
        #expect(try literal(#"(?:<\|message_model\|>)?[^<]*<\|content_text\|>"#) == ("<|content_text|>", false))
        #expect(try literal(#"\n?</response>"#) == ("</response>", false))
        #expect(try literal(#"ab+c"#) == ("ab", true))
        #expect(try literal(#"abc|abd"#) == ("", false))
        #expect(try literal(#"(?i)abc"#) == ("", false))
    }

    // MARK: Helpers

    private func events(_ value: ResponseValue) throws -> [ResponseEvent] {
        try value.arrayValue!.map { event in
            let event = event.objectValue!
            let field = event["field"]!.stringValue!
            switch event["type"]!.stringValue! {
            case "region_open": return .regionOpen(field: field)
            case "region_chunk":
                return .regionChunk(
                    field: field, text: event["text"]!.stringValue!, isDirty: event["dirty"] == .bool(true))
            case "region_close": return .regionClose(field: field, value: event["value"]!)
            case let other: throw ResponseParsingError.content("unknown event \(other)")
            }
        }
    }

    /// Equality that treats NaN as equal to itself, as the recorded JSON does.
    static func same(_ lhs: ResponseValue, _ rhs: ResponseValue) -> Bool {
        switch (lhs, rhs) {
        case let (.double(a), .double(b)): return a == b || (a.isNaN && b.isNaN)
        case let (.array(a), .array(b)): return a.count == b.count && zip(a, b).allSatisfy(same)
        case let (.object(a), .object(b)):
            return a.count == b.count && a.allSatisfy { entry in b[entry.key].map { same(entry.value, $0) } ?? false }
        default: return lhs == rhs
        }
    }

    static func same(_ lhs: [ResponseEvent], _ rhs: [ResponseEvent]) -> Bool {
        lhs.count == rhs.count
            && zip(lhs, rhs).allSatisfy { pair in
                if case let (.regionClose(a, x), .regionClose(b, y)) = pair { return a == b && same(x, y) }
                return pair.0 == pair.1
            }
    }
}

extension ResponseValue {
    /// The value as the dictionaries and arrays `ToolSpec` and `Message` hold.
    fileprivate var sendable: any Sendable {
        switch self {
        case .null: return Optional<String>.none as any Sendable
        case let .bool(value): return value
        case let .int(value): return value
        case let .double(value): return value
        case let .string(value): return value
        case let .array(values): return values.map(\.sendable)
        case let .object(object): return Dictionary(uniqueKeysWithValues: object.map { ($0.key, $0.value.sendable) })
        }
    }
}
