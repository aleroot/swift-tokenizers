import Foundation
import Jinja

/// Tokens and loss-mask annotations from actual `{% generation %}` blocks.
public struct ChatTemplateEncoding: Sendable {
    public let encoding: TokenEncoding
    /// True when a token's source range overlaps generated assistant content. A token
    /// crossing a generation boundary is included; inserted post-processor tokens are not.
    public let assistantMask: [Bool]
    public var ids: [Int] { encoding.ids }
    public var text: String { encoding.text }

    /// Creates a result for a custom tokenizer, validating one mask entry per token.
    public init(encoding: TokenEncoding, assistantMask: [Bool]) throws {
        guard encoding.ids.count == assistantMask.count else {
            throw TokenizerError.invalidConfiguration("Assistant mask and token IDs must have the same count")
        }
        self.encoding = encoding
        self.assistantMask = assistantMask
    }

    init(encoding: TokenEncoding, generationRanges: [Range<Int>]) {
        self.encoding = encoding
        assistantMask = (encoding.offsets ?? []).map { offset in
            guard let offset, !offset.isEmpty else { return false }
            var low = 0
            var high = generationRanges.count
            while low < high {
                let mid = (low + high) / 2
                if generationRanges[mid].upperBound <= offset.lowerBound { low = mid + 1 } else { high = mid }
            }
            return low < generationRanges.count && generationRanges[low].lowerBound < offset.upperBound
        }
    }
}

public extension Tokenizer {
    /// Renders a training conversation and aligns an assistant loss mask to its tokens.
    /// The template must annotate assistant content with `{% generation %}`. Templates
    /// without annotations throw instead of silently masking out every training token.
    /// `maxLength` optionally truncates the result on the right, including its mask.
    func applyChatTemplateWithAssistantMask(
        messages: [Message], chatTemplate: ChatTemplateArgument? = nil,
        addGenerationPrompt: Bool = false, maxLength: Int? = nil,
        tools: [ToolSpec]? = nil, additionalContext: [String: any Sendable]? = nil
    ) throws -> ChatTemplateEncoding {
        try encodeChatTemplateWithAssistantMask(
            messages: messages, chatTemplate: chatTemplate, addGenerationPrompt: addGenerationPrompt,
            maxLength: maxLength, tools: tools, additionalContext: additionalContext)
    }

    /// Primitive for custom conformers, used by the convenience `apply` overload.
    func encodeChatTemplateWithAssistantMask(
        messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool,
        maxLength: Int?, tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> ChatTemplateEncoding {
        throw TokenizerError.unsupportedComponent("assistant masks for this tokenizer")
    }
}

/// swift-jinja exposes its AST but does not yet expose generation-span callbacks. Annotate
/// output at AST boundaries, after whitespace control, and verify byte-for-byte render parity.
/// Restrict annotations to direct output: capturing/filtering a generation block can change
/// string operations before the markers are removed and cannot be tracked safely this way.
struct GenerationTemplate: Sendable {
    let nodes: [Node]
    let start: String
    let end: String

    init(source: String) throws {
        let nonce = UUID().uuidString
        let start = "\u{0}generation-\(nonce)-start\u{0}"
        let end = "\u{0}generation-\(nonce)-end\u{0}"
        let parsed = try Parser.parse(Lexer.tokenize(ChatTemplatePreprocessor.preparedSource(source)))
        var found = false
        func annotate(_ nodes: [Node], captured: Bool = false) throws -> [Node] {
            try nodes.flatMap { node -> [Node] in
                guard case let .statement(statement) = node else { return [node] }
                let result: Statement
                switch statement {
                case let .generation(body):
                    guard !captured else {
                        throw TokenizerError.chatTemplate(
                            "Assistant masks require generation blocks in direct output, outside set, macro, call and filter blocks"
                        )
                    }
                    found = true
                    return [.text(start)] + (try annotate(body)) + [.text(end)]
                case let .program(body): result = .program(try annotate(body, captured: captured))
                case let .if(test, body, alternate):
                    result = .if(
                        test, try annotate(body, captured: captured), try annotate(alternate, captured: captured))
                case let .for(variable, expression, body, alternate, test):
                    result = .for(
                        variable, expression, try annotate(body, captured: captured),
                        try annotate(alternate, captured: captured), test: test)
                case let .set(target, value, body):
                    result = .set(target: target, value: value, body: try annotate(body, captured: true))
                case let .macro(name, args, defaults, body):
                    result = .macro(name, args, defaults, try annotate(body, captured: true))
                case let .call(callable, args, body):
                    result = .call(callable: callable, callerArgs: args, body: try annotate(body, captured: true))
                case let .filter(expression, body):
                    result = .filter(filterExpr: expression, body: try annotate(body, captured: true))
                case .break, .continue: result = statement
                }
                return [.statement(result)]
            }
        }
        self.start = start
        self.end = end
        nodes = try annotate(parsed)
        guard found else { throw TokenizerError.chatTemplate("Assistant masks require {% generation %} annotations") }
    }

    func ranges(context: [String: Jinja.Value], expected: String) throws -> [Range<Int>] {
        let environment = Environment()
        for (key, value) in context { environment[key] = value }
        let rendered = try Interpreter.interpret(nodes, environment: environment)
        let bytes = Array(rendered.utf8)
        let opening = Array(start.utf8)
        let closing = Array(end.utf8)
        var output: [UInt8] = []
        output.reserveCapacity(bytes.count)
        var depth = 0
        var beginning = 0
        var ranges: [Range<Int>] = []
        var index = 0
        var copied = 0
        while index < bytes.count {
            guard bytes[index] == 0 else { index += 1; continue }
            func matches(_ marker: [UInt8]) -> Bool {
                index + marker.count <= bytes.count && bytes[index..<index + marker.count].elementsEqual(marker)
            }
            let isOpening = matches(opening)
            guard isOpening || matches(closing) else { index += 1; continue }
            output.append(contentsOf: bytes[copied..<index])
            if isOpening {
                if depth == 0 { beginning = output.count }
                depth += 1
            } else {
                guard depth > 0 else { throw TokenizerError.chatTemplate("Unbalanced generation output") }
                depth -= 1
                if depth == 0, output.count > beginning { ranges.append(beginning..<output.count) }
            }
            index += isOpening ? opening.count : closing.count
            copied = index
        }
        output.append(contentsOf: bytes[copied...])
        guard depth == 0, output.elementsEqual(expected.utf8),
            !expected.contains(start), !expected.contains(end)
        else {
            throw TokenizerError.chatTemplate(
                "Generation annotations changed template output; this template requires native generation tracking")
        }
        return ranges
    }
}
