import Jinja

extension ResponseObject {
    /// The parsed message in the form chat templates accept, to append to a conversation.
    /// Nested objects keep their key order when rendered.
    public var message: Message {
        Dictionary(map { ($0.key, $0.value as any Sendable) }, uniquingKeysWith: { _, last in last })
    }
}

extension ResponseValue {
    /// A chat-template value as a response value.
    init(jinja value: Jinja.Value) {
        switch value {
        case .null, .undefined, .function, .macro: self = .null
        case let .boolean(value): self = .bool(value)
        case let .int(value): self = .int(value)
        case let .double(value): self = .double(value)
        case let .string(value): self = .string(value)
        case let .array(values): self = .array(values.map(ResponseValue.init(jinja:)))
        case let .object(storage):
            var object = ResponseObject()
            for (key, value) in storage {
                switch key {
                case let .string(name): object[name] = ResponseValue(jinja: value)
                case let .int(number): object[String(number)] = ResponseValue(jinja: value)
                }
            }
            self = .object(object)
        }
    }
}

public extension Tokenizer {
    /// Parses a complete generated response into an assistant message, as `transformers`'
    /// `parse_response` does.
    ///
    /// - Parameters:
    ///   - response: The generated text, without the prompt.
    ///   - prefix: The prompt the model continued; see ``ResponseParser/init(template:prefix:tools:)``.
    ///   - tools: The tools offered to the model, to type tool-call arguments.
    ///   - template: A template to use instead of the tokenizer's own.
    func parseResponse(
        _ response: String, prefix: String, tools: [ToolSpec]? = nil, template: ResponseTemplate? = nil
    ) throws -> ResponseObject {
        try ResponseParser.parse(
            response, template: try resolvedResponseTemplate(template), prefix: prefix, tools: tools)
    }

    /// Parses generated token ids, decoding them and the prompt ids first.
    func parseResponse(
        tokens: [Int], prefix: [Int], tools: [ToolSpec]? = nil, template: ResponseTemplate? = nil
    ) throws -> ResponseObject {
        try parseResponse(
            decode(tokens: tokens, skipSpecialTokens: false), prefix: decode(tokens: prefix, skipSpecialTokens: false),
            tools: tools, template: template)
    }

    /// A parser for a response that is still being generated.
    func makeResponseParser(
        prefix: String, tools: [ToolSpec]? = nil, template: ResponseTemplate? = nil
    ) throws -> ResponseParser {
        try ResponseParser(template: try resolvedResponseTemplate(template), prefix: prefix, tools: tools)
    }

    /// A parser for a response that is still being generated, after the prompt ids `prefix`.
    func makeResponseParser(
        prefix: [Int], tools: [ToolSpec]? = nil, template: ResponseTemplate? = nil
    ) throws -> ResponseParser {
        try makeResponseParser(
            prefix: decode(tokens: prefix, skipSpecialTokens: false), tools: tools, template: template)
    }

    private func resolvedResponseTemplate(_ template: ResponseTemplate?) throws -> ResponseTemplate {
        if let template { return template }
        guard let template = try responseTemplate else { throw ResponseParsingError.missingTemplate }
        return template
    }
}
