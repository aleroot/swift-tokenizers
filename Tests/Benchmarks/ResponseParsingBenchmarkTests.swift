import Foundation
import Testing

@testable import Tokenizers

/// Opt-in response parsing timings on realistic generations: a long reasoning trace followed by
/// tool calls. Run with RUN_BENCHMARKS=1 swift test -c release --filter ResponseParsingBenchmarkTests.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct ResponseParsingBenchmarkTests {
    struct Workload: Sendable, CustomTestStringConvertible {
        let name: String
        let template: String
        let prefix: String
        let response: String
        var testDescription: String { name }
    }

    static let reasoning = String(repeating: benchmarkParagraph + " ", count: 8)

    static let workloads = [
        Workload(
            name: "Gemma 4 (literal and regex delimiters)",
            template: #"""
                {"defaults": {"role": "assistant"}, "start_anchor": ["<|turn>model\n", "<tool_response|>"],
                 "fields": {
                   "thinking": {"open": "<|channel>thought\n", "close": "<channel|>"},
                   "tool_calls": {"open_pattern": "<\\|tool_call>call:(?P<name>\\w+)", "close": "<tool_call|>",
                     "repeats": true, "content": "json",
                     "content_args": {"unquoted_keys": true, "string_delims": [["<|\"|>", "<|\"|>"]]},
                     "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}}},
                   "content": {"close": ["<turn|>", "<|tool_response>", "<eos>"]}}}
                """#,
            prefix: "<|turn>user\nPlan my trip.<turn|>\n<|turn>model\n",
            response: "<|channel>thought\n" + reasoning + "<channel|>"
                + #"<|tool_call>call:get_weather{location:<|"|>Paris, FR<|"|>,days:3}<tool_call|>"#
                + #"<|tool_call>call:search{query:<|"|>museums open on Monday<|"|>,limit:5}<tool_call|>"#
                + "Here is the plan.<turn|>"),
        Workload(
            name: "Qwen3 (regex open, XML arguments)",
            template: #"""
                {"defaults": {"role": "assistant"}, "start_anchor": "<|im_start|>assistant\n",
                 "fields": {
                   "reasoning_content": {"open": "<think>", "close": "</think>"},
                   "tool_calls": {"open_pattern": "<tool_call>\\s*<function=(?P<name>\\w+)>", "close": "</tool_call>",
                     "repeats": true, "content": "xml-inline",
                     "content_args": {"tag_pattern": "<parameter=(?P<key>\\w+)>\\s*(?P<value>.*?)\\s*</parameter>",
                       "value_parser": {"name": "json", "args": {"allow_non_json": true}}},
                     "transform": {"type": "function", "function": {"name": "{name}", "arguments": "{content}"}}},
                   "content": {"close": "<|im_end|>"}}}
                """#,
            prefix: "<|im_start|>user\nPlan my trip.<|im_end|>\n<|im_start|>assistant\n",
            response: "<think>" + reasoning + "</think>\n"
                + "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n"
                + "<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>\nHere is the plan.<|im_end|>"),
        Workload(
            name: "Inkling (prefixless regex opens)",
            template: #"""
                {"defaults": {"role": "assistant"}, "start_anchor": "<|message_model|>",
                 "fields": {
                   "reasoning_content": {"open_pattern": "(?:<\\|message_model\\|>)?[^<]*<\\|content_thinking\\|>",
                     "close": "<|end_message|>", "repeats": true, "join": "", "content_args": {"strip": false}},
                   "content": {"open_pattern": "(?:<\\|message_model\\|>)?[^<]*<\\|content_text\\|>",
                     "close": "<|end_message|>", "repeats": true, "join": "", "content_args": {"strip": false}}}}
                """#,
            prefix: "<|message_model|>",
            response: "model<|content_thinking|>" + reasoning + "<|end_message|><|message_model|>model<|content_text|>"
                + "Here is the plan.<|end_message|>"),
    ]

    @Test("Parse and stream realistic generations", arguments: workloads)
    func parse(_ workload: Workload) throws {
        let template = try ResponseTemplate(jsonString: workload.template)
        let expected = try ResponseParser.parse(workload.response, template: template, prefix: workload.prefix)
        // Token-sized pieces: about four characters, as a model emits them.
        let pieces = stride(from: 0, to: workload.response.count, by: 4).map { start in
            String(workload.response.dropFirst(start).prefix(4))
        }
        print("\n=== \(workload.name): \(workload.response.utf8.count) bytes, \(pieces.count) pieces ===")
        var checksum = 0
        benchmarkMeasure(label: "compile template", iterations: 50) {
            checksum &+= (try? ResponseTemplate(jsonString: workload.template)) == nil ? 0 : 1
        }
        let oneShot = benchmarkMeasure(label: "parse whole response", iterations: 50) {
            checksum &+=
                (try? ResponseParser.parse(workload.response, template: template, prefix: workload.prefix))?.count ?? 0
        }
        let streamed = benchmarkMeasure(label: "stream piece by piece", iterations: 20) {
            var parser = try! ResponseParser(template: template, prefix: workload.prefix)
            for piece in pieces { checksum &+= try! parser.feed(piece).count }
            let (message, _) = try! parser.finalize()
            checksum &+= message.count
        }
        let megabytesPerSecond = Double(workload.response.utf8.count) / oneShot.p50 / 1e3
        let microsecondsPerPiece = streamed.p50 * 1e3 / Double(pieces.count)
        print(
            String(format: "  whole %.1f MB/s, streaming %.2f µs per piece", megabytesPerSecond, microsecondsPerPiece))
        var parser = try ResponseParser(template: template, prefix: workload.prefix)
        for piece in pieces { _ = try parser.feed(piece) }
        #expect(try parser.finalize().message == expected)
        print("  checksum \(checksum)")
    }
}
