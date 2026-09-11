import Testing

@testable import Tokenizers

@Suite("Assistant generation masks")
struct ChatTemplateEncodingTests {
    private func tokenizer() throws -> any Tokenizer {
        try AutoTokenizer.from(
            tokenizerConfig: [:],
            tokenizerData: [
                "model": [
                    "type": "BPE",
                    "vocab": ["a": 0, "b": 1, "ab": 2, "é": 3, " ": 4, "\n": 5, "<eos>": 6, "e": 7, "\u{301}": 8],
                    "merges": ["a b"],
                ],
                "added_tokens": [["id": 6, "content": "<eos>", "special": true, "normalized": false]],
            ])
    }

    @Test("Repeated content uses source positions, nested generations and assistant EOS are included")
    func repeatedContent() throws {
        let tokenizer = try tokenizer()
        let template = """
            {% for m in messages %}
            {% if m.role == 'assistant' %}{% generation %}{{ m.content }}{% generation %}<eos>{% endgeneration %}{% endgeneration %}{% else %}{{ m.content }}{% endif %}{{ '\n' -}}
            {% endfor %}
            """
        let messages: [Message] = [
            ["role": "user", "content": "é"], ["role": "assistant", "content": "é"], ["role": "user", "content": "é"],
        ]
        let result = try tokenizer.applyChatTemplateWithAssistantMask(
            messages: messages, chatTemplate: .literal(template))
        #expect(result.text == "é\né<eos>\né\n")
        #expect(result.ids == [3, 5, 3, 6, 5, 3, 5])
        #expect(result.assistantMask == [false, false, true, true, false, false, false])
        #expect(result.ids == tokenizer.encode(text: result.text, addSpecialTokens: false))
        let truncated = try tokenizer.applyChatTemplateWithAssistantMask(
            messages: messages, chatTemplate: .literal(template), maxLength: 3)
        #expect(truncated.ids == [3, 5, 3])
        #expect(truncated.assistantMask == [false, false, true])
        #expect(truncated.encoding.offsets == [0..<2, 2..<3, 3..<5])
    }

    @Test("BPE tokens crossing a generation boundary overlap the mask; empty and inactive blocks do not")
    func boundaries() throws {
        let tokenizer = try tokenizer()
        let template =
            "a{% generation %}b{% endgeneration %}{% generation %}{% endgeneration %}{% if false %}{% generation %}a{% endgeneration %}{% endif %}é"
        let result = try tokenizer.applyChatTemplateWithAssistantMask(messages: [], chatTemplate: .literal(template))
        #expect(result.ids == [2, 3])
        #expect(result.assistantMask == [true, false])
        let inactive = try tokenizer.applyChatTemplateWithAssistantMask(
            messages: [], chatTemplate: .literal("{% if false %}{% generation %}a{% endgeneration %}{% endif %}b"))
        #expect(inactive.assistantMask == [false])
        let empty = try tokenizer.applyChatTemplateWithAssistantMask(
            messages: [], chatTemplate: .literal(template), maxLength: 0)
        #expect(empty.ids.isEmpty && empty.assistantMask.isEmpty)
    }

    @Test("Whitespace control and tools/additional context have identical ordinary and annotated rendering")
    func contextAndWhitespace() throws {
        let tokenizer = try tokenizer()
        let template = "a \r\n{#- comment -#} {% generation -%}{{ tools[0].name }}{{ suffix }}{%- endgeneration %}\r\n"
        let result = try tokenizer.applyChatTemplateWithAssistantMask(
            messages: [], chatTemplate: .literal(template), tools: [["name": "b"]], additionalContext: ["suffix": "é"])
        #expect(result.text == "abé")
        #expect(result.assistantMask == [true, true])
    }

    @Test("Template caches preserve byte-distinct canonically equivalent Unicode sources")
    func unicodeTemplateKeys() throws {
        let tokenizer = try tokenizer()
        for (text, ids) in [("é", [3]), ("e\u{301}", [7, 8]), ("é", [3])] {
            let result = try tokenizer.applyChatTemplateWithAssistantMask(
                messages: [], chatTemplate: .literal("{% generation %}" + text + "{% endgeneration %}"))
            #expect(result.text.utf8.elementsEqual(text.utf8))
            #expect(result.ids == ids)
            #expect(result.assistantMask == Array(repeating: true, count: ids.count))
        }
    }

    @Test("Generation templates share a bounded cache safely across callers")
    func concurrentCache() async throws {
        let tokenizer = try #require(tokenizer() as? PreTrainedTokenizer)
        let template = "a{% generation %}é{% endgeneration %}"
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<32 {
                group.addTask {
                    let result = try tokenizer.applyChatTemplateWithAssistantMask(
                        messages: [], chatTemplate: .literal(template))
                    #expect(result.ids == [0, 3])
                    #expect(result.assistantMask == [false, true])
                }
            }
            try await group.waitForAll()
        }
        #expect(tokenizer.compiledChatTemplateCount == 1)
        for index in 0..<32 {
            _ = try tokenizer.applyChatTemplateWithAssistantMask(
                messages: [], chatTemplate: .literal(template + String(index)))
        }
        #expect(tokenizer.compiledChatTemplateCount <= PreTrainedTokenizer.chatTemplateCacheLimit)
    }

    @Test("Missing annotations, captured annotations and interrupted blocks fail explicitly")
    func unsupported() throws {
        let tokenizer = try tokenizer()
        for template in [
            "a", "{{ '{% generation %}' }}", "{% set x %}{% generation %}a{% endgeneration %}{% endset %}{{x}}",
            "{% filter upper %}{% generation %}a{% endgeneration %}{% endfilter %}",
            "{% for m in messages %}{% generation %}a{% break %}{% endgeneration %}{% endfor %}",
        ] {
            #expect(throws: (any Error).self) {
                try tokenizer.applyChatTemplateWithAssistantMask(
                    messages: [["role": "assistant", "content": "a"]], chatTemplate: .literal(template))
            }
        }
        #expect(throws: TokenizerError.self) {
            try tokenizer.applyChatTemplateWithAssistantMask(
                messages: [], chatTemplate: .literal("{% generation %}a{% endgeneration %}"), maxLength: -1)
        }
    }
}
