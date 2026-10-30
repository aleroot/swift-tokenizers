import Foundation
import Testing

@testable import Tokenizers

@Suite("Chat Template Continuation Tests")
struct ChatTemplateContinuationTests {
    /// The key that asks a chat template to continue its final message.
    private static let continuation: [String: any Sendable] = [ChatTemplateContinuation.key: true]

    private static let conversation: [Message] = [
        ["role": "user", "content": "Describe the Swift programming language."],
        ["role": "assistant", "content": "Swift is a"],
    ]

    @MainActor
    static let qwenTask = Task { try await HubFixtures.preTrainedTokenizer(for: "Qwen/Qwen3-0.6B") }

    static func qwen() async throws -> PreTrainedTokenizer { try await qwenTask.value }

    @MainActor
    static let phiTask = Task {
        try await HubFixtures.preTrainedTokenizer(for: "microsoft/Phi-3-mini-128k-instruct")
    }

    static func phi() async throws -> PreTrainedTokenizer { try await phiTask.value }

    @MainActor
    static let gemmaTask = Task { try await HubFixtures.preTrainedTokenizer(for: "pcuenq/gemma-tokenizer") }

    static func gemma() async throws -> PreTrainedTokenizer { try await gemmaTask.value }

    @Test("The prompt ends inside the final message")
    func continuesTheFinalMessage() async throws {
        let tokenizer = try await Self.qwen()
        // Ground truth from `transformers` 4.57.1 with `continue_final_message=True`.
        let prompt = try tokenizer.renderChatTemplate(
            messages: Self.conversation, additionalContext: Self.continuation)
        #expect(
            prompt
                == "<|im_start|>user\nDescribe the Swift programming language.<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\nSwift is a")
    }

    @Test("The continued prompt encodes the same tokens as the reference")
    func encodesTheContinuedPrompt() async throws {
        let tokenizer = try await Self.qwen()
        let ids = try tokenizer.applyChatTemplate(
            messages: Self.conversation, additionalContext: Self.continuation)
        #expect(
            ids
                == [151644, 872, 198, 74785, 279, 23670, 15473, 4128, 13, 151645, 198, 151644, 77091, 198,
                    151667, 271, 151668, 271, 55336, 374, 264])
    }

    @Test("A template that trims the message ends the prompt at the text")
    func trimsTheTrailingSpace() async throws {
        let tokenizer = try await Self.gemma()
        let messages: [Message] = [
            ["role": "user", "content": "Describe the Swift programming language."],
            ["role": "assistant", "content": "Swift is a "],
        ]
        let prompt = try tokenizer.renderChatTemplate(
            messages: messages, additionalContext: Self.continuation)
        // The Gemma template trims the message, so the space the marker carried is gone.
        #expect(
            prompt
                == "<bos><start_of_turn>user\nDescribe the Swift programming language.<end_of_turn>\n"
                + "<start_of_turn>model\nSwift is a")
    }

    @Test("A template that writes its own header ends the prompt at the text")
    func continuesWithATemplateHeader() async throws {
        let tokenizer = try await Self.phi()
        let prompt = try tokenizer.renderChatTemplate(
            messages: Self.conversation, additionalContext: Self.continuation)
        #expect(
            prompt
                == "<|user|>\nDescribe the Swift programming language.<|end|>\n<|assistant|>\nSwift is a")
    }

    @Test("The conversation keeps its earlier turns")
    func continuesTheLastTurn() async throws {
        let tokenizer = try await Self.qwen()
        let messages: [Message] = [
            ["role": "user", "content": "First question."],
            ["role": "assistant", "content": "First answer."],
            ["role": "user", "content": "Second question."],
            ["role": "assistant", "content": "Second answer"],
        ]
        let prompt = try tokenizer.renderChatTemplate(
            messages: messages, additionalContext: Self.continuation)
        #expect(
            prompt
                == "<|im_start|>user\nFirst question.<|im_end|>\n<|im_start|>assistant\nFirst answer.<|im_end|>\n"
                + "<|im_start|>user\nSecond question.<|im_end|>\n<|im_start|>assistant\n"
                + "<think>\n\n</think>\n\nSecond answer")
    }

    @Test("The last text part of a message continues")
    func continuesTheLastTextPart() async throws {
        let tokenizer = try await Self.qwen()
        let template =
            "{% for message in messages %}{{ message['role'] }}:"
            + "{% for part in message['content'] %}{{ part['text'] }}{% endfor %} {% endfor %}"
        let messages: [Message] = [
            [
                "role": "user",
                "content": [["type": "text", "text": "What is in this picture?"]] as [[String: any Sendable]],
            ],
            [
                "role": "assistant",
                "content": [
                    ["type": "text", "text": "A cat"],
                    ["type": "text", "text": "on a desk"],
                ] as [[String: any Sendable]],
            ],
        ]
        let prompt = try tokenizer.renderChatTemplate(
            messages: messages, chatTemplate: .literal(template), additionalContext: Self.continuation)
        #expect(prompt == "user:What is in this picture? assistant:A caton a desk")
    }

    @Test("A template that changes the message cannot be continued")
    func failsOnAChangedMessage() async throws {
        let tokenizer = try await Self.qwen()
        let template =
            "{% for message in messages %}{{ message['role'] }}: {{ message['content'] | upper }}{% endfor %}"
        #expect(throws: TokenizerError.self) {
            try tokenizer.renderChatTemplate(
                messages: Self.conversation, chatTemplate: .literal(template),
                additionalContext: Self.continuation)
        }
    }

    @Test("The key never reaches the template")
    func hidesTheKeyFromTheTemplate() async throws {
        let tokenizer = try await Self.qwen()
        let template =
            "{% if continue_final_message %}leaked{% else %}"
            + "{% for message in messages %}{{ message['content'] }}{% endfor %}{% endif %}"
        let prompt = try tokenizer.renderChatTemplate(
            messages: Self.conversation, chatTemplate: .literal(template),
            additionalContext: Self.continuation)
        #expect(prompt == "Describe the Swift programming language.Swift is a")
    }

    @Test("A conversation that ends with a tool result renders as usual")
    func rendersAToolResultAsUsual() async throws {
        let tokenizer = try await Self.qwen()
        let messages: [Message] = [
            ["role": "user", "content": "Describe the Swift programming language."],
            ["role": "assistant", "content": "Swift is a"],
            ["role": "tool", "content": "a tool result"],
        ]
        let prompt = try tokenizer.renderChatTemplate(
            messages: messages, additionalContext: Self.continuation)
        // The ChatML template writes a tool result as a user turn.
        #expect(
            prompt
                == "<|im_start|>user\nDescribe the Swift programming language.<|im_end|>\n"
                + "<|im_start|>assistant\nSwift is a<|im_end|>\n"
                + "<|im_start|>user\n<tool_response>\na tool result\n</tool_response><|im_end|>\n")
    }

    @Test("The key never reaches the template of a conversation that does not continue")
    func hidesTheKeyFromAnOrdinaryTemplate() async throws {
        let tokenizer = try await Self.qwen()
        let template =
            "{% if continue_final_message %}leaked{% else %}{{ messages[0]['content'] }}{% endif %}"
        let prompt = try tokenizer.renderChatTemplate(
            messages: [["role": "user", "content": "Describe the Swift programming language."]],
            chatTemplate: .literal(template), additionalContext: Self.continuation)
        #expect(prompt == "Describe the Swift programming language.")
    }

    @Test("A conversation without the key renders as usual")
    func rendersWithoutTheKey() async throws {
        let tokenizer = try await Self.qwen()
        let prompt = try tokenizer.renderChatTemplate(messages: Self.conversation)
        #expect(
            prompt
                == "<|im_start|>user\nDescribe the Swift programming language.<|im_end|>\n"
                + "<|im_start|>assistant\n<think>\n\n</think>\n\nSwift is a<|im_end|>\n")
    }

    @Test("A context that does not ask for a continuation reaches the template")
    func keepsAnInactiveRequestInTheContext() async throws {
        let tokenizer = try await Self.qwen()
        let template = "{% if continue_final_message is defined %}defined{% else %}undefined{% endif %}"
        let prompt = try tokenizer.renderChatTemplate(
            messages: [["role": "user", "content": "Describe the Swift programming language."]],
            chatTemplate: .literal(template),
            additionalContext: [ChatTemplateContinuation.key: false])
        #expect(prompt == "defined")
    }

    @Test("A value that is not a request reaches the template")
    func keepsAnyOtherValueInTheContext() async throws {
        let tokenizer = try await Self.qwen()
        let prompt = try tokenizer.renderChatTemplate(
            messages: [["role": "user", "content": "Describe the Swift programming language."]],
            chatTemplate: .literal("{{ continue_final_message }}"),
            additionalContext: [ChatTemplateContinuation.key: "kept"])
        #expect(prompt == "kept")
    }

    @Test("An assistant mask refuses a continued prompt")
    func refusesAMaskedContinuation() throws {
        let tokenizer = try Self.smallTokenizer()
        let template =
            "{% for m in messages %}{% if m.role == 'assistant' %}{% generation %}{{ m.content }}"
            + "{% endgeneration %}{% else %}{{ m.content }}{% endif %}\n{% endfor %}"
        let messages: [Message] = [["role": "user", "content": "a"], ["role": "assistant", "content": "b"]]
        #expect(throws: TokenizerError.self) {
            try tokenizer.applyChatTemplateWithAssistantMask(
                messages: messages, chatTemplate: .literal(template),
                additionalContext: [ChatTemplateContinuation.key: true])
        }
        // A context that does not ask for a continuation leaves the mask alone.
        let masked = try tokenizer.applyChatTemplateWithAssistantMask(
            messages: messages, chatTemplate: .literal(template),
            additionalContext: [ChatTemplateContinuation.key: false])
        let plain = try tokenizer.applyChatTemplateWithAssistantMask(
            messages: messages, chatTemplate: .literal(template))
        #expect(masked.ids == plain.ids)
        #expect(masked.assistantMask == plain.assistantMask)
    }

    /// A tokenizer with a tiny vocabulary, for the paths that need no model.
    private static func smallTokenizer() throws -> any Tokenizer {
        try AutoTokenizer.from(
            tokenizerConfig: [:],
            tokenizerData: [
                "model": [
                    "type": "BPE",
                    "vocab": ["a": 0, "b": 1, "ab": 2, " ": 3, "\n": 4],
                    "merges": ["a b"],
                ],
            ])
    }
}
