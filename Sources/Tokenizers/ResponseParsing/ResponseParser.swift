import Foundation

/// A change in a streamed response, reported by ``ResponseParser``.
public enum ResponseEvent: Sendable, Equatable {
    /// The model started writing `field`.
    case regionOpen(field: String)
    /// More text of `field`. `isDirty` is `true` for structured regions (`json`, `xml-inline`,
    /// `kv-lines`), whose raw text only becomes a value when the region closes.
    case regionChunk(field: String, text: String, isDirty: Bool)
    /// `field` is complete, with its parsed value.
    case regionClose(field: String, value: ResponseValue)
}

/// Turns generated text into a structured assistant message, incrementally.
///
/// Feed the text as it is generated and render the events; call ``finalize()`` when generation
/// ends to get the message. The algorithm is the one `transformers` uses, event for event: a
/// delimiter is never emitted as text, so text that could still turn out to be the start of one
/// is held back until the next chunk decides.
///
/// ```swift
/// var parser = try tokenizer.makeResponseParser(prefix: prompt)
/// for event in parser.initialEvents { render(event) }
/// for try await piece in generatedText {
///     for event in try parser.feed(piece) { render(event) }
/// }
/// let (message, events) = try parser.finalize()
/// ```
public struct ResponseParser: Sendable {
    /// Events produced while reading the prefix, for regions the chat template already opened.
    public private(set) var initialEvents: [ResponseEvent] = []

    private let template: ResponseTemplate
    /// Tool name to the `properties` of its parameters, for typing tool-call arguments.
    private let toolParameters: [String: ResponseObject]

    private var buffer: [UInt16] = []
    private var position = 0
    private var output: ResponseObject
    /// The field being written, or `nil` for text outside every field.
    private var current: Int?
    private var captures: [(name: String, value: String)] = []
    private var body: [UInt16] = []
    private var opened = false
    private var finalized = false

    /// Creates a parser positioned after `prefix`, the prompt the model continues.
    ///
    /// - Parameters:
    ///   - template: How the model lays out its reply.
    ///   - prefix: The prompt. It is required because chat templates often write the start of
    ///     the assistant message themselves (an opening `<think>`, say); only the part after the
    ///     template's start anchor is read. Pass `""` when the generated text holds the whole
    ///     message.
    ///   - tools: The tools offered to the model, as passed to the chat template. Tool-call
    ///     arguments are then typed from each tool's JSON Schema: `"7"` becomes `7` for an
    ///     integer parameter and stays `"7"` for a string one.
    public init(template: ResponseTemplate, prefix: String, tools: [ToolSpec]? = nil) throws {
        self.template = template
        self.toolParameters = try Self.parameters(of: tools ?? [])
        output = template.defaults
        current = template.implicitField
        guard !prefix.isEmpty else { return }
        var prefixUnits: [UInt16] = []
        UTF16Text.append(prefix, to: &prefixUnits)
        let truncated = template.truncatePastLastAnchor(prefixUnits)
        guard !truncated.isEmpty else { return }
        buffer = Array(truncated)
        var events: [ResponseEvent] = []
        try process(&events, endOfStream: false)
        initialEvents = events
    }

    /// Parses a complete response in one call.
    public static func parse(
        _ text: String, template: ResponseTemplate, prefix: String, tools: [ToolSpec]? = nil
    ) throws -> ResponseObject {
        var parser = try ResponseParser(template: template, prefix: prefix, tools: tools)
        _ = try parser.feed(text)
        return try parser.finalize().message
    }

    /// Adds generated text and returns the events it completes.
    public mutating func feed(_ text: String) throws -> [ResponseEvent] {
        guard !finalized else { throw ResponseParsingError.alreadyFinalized }
        UTF16Text.append(text, to: &buffer)
        var events: [ResponseEvent] = []
        try process(&events, endOfStream: false)
        return events
    }

    /// Ends the stream: flushes held text, closes the open region and returns the message.
    ///
    /// Fields with an empty value are left out of the message unless the template gives them a
    /// default.
    public mutating func finalize() throws -> (message: ResponseObject, events: [ResponseEvent]) {
        guard !finalized else { throw ResponseParsingError.alreadyFinalized }
        var events: [ResponseEvent] = []
        try process(&events, endOfStream: true)
        let missing = template.fields.filter { !$0.optional && output[$0.name] == nil }.map(\.name)
        guard missing.isEmpty else { throw ResponseParsingError.missingRequiredFields(missing) }
        finalized = true
        let defaults = template.defaults
        let message = ResponseObject(
            output.filter { defaults[$0.key] != nil || !$0.value.isEmptyForMessage }.map { ($0.key, $0.value) })
        return (message, events)
    }

    // MARK: - State machine

    private enum Kind { case open, close }

    private struct Candidate {
        var kind: Kind
        var field: Int
        var match: ResponseDelimiter.Match
    }

    private mutating func process(_ events: inout [ResponseEvent], endOfStream: Bool) throws {
        while true {
            let (best, holdStart) = buffer.withUnsafeBufferPointer { scan($0, endOfStream: endOfStream) }
            if let best {
                let start = best.match.range.lowerBound
                if start > position { accumulate(position..<start, &events) }
                position = best.match.range.upperBound
                switch best.kind {
                case .open:
                    try closeCurrent(&events)
                    current = best.field
                    captures = best.match.captures
                    body.removeAll(keepingCapacity: true)
                    opened = true
                    events.append(.regionOpen(field: template.fields[best.field].name))
                case .close:
                    let hadContent = opened
                    try closeCurrent(&events)
                    // A zero-width close of an empty region would fire again forever.
                    if !hadContent, best.match.range.isEmpty { return }
                }
                continue
            }
            if endOfStream {
                if position < buffer.count {
                    accumulate(position..<buffer.count, &events)
                    position = buffer.count
                }
                try closeCurrent(&events)
                return
            }
            // Emit everything before the earliest delimiter that is still undecided.
            if holdStart > position {
                accumulate(position..<holdStart, &events)
                position = holdStart
            }
            return
        }
    }

    /// The delimiters that matter now: the close of the open explicit region, or every open plus
    /// the implicit region's close. Returns the earliest delimiter that can be acted on, and the
    /// first position whose text must wait for more input.
    private func scan(_ text: UnsafeBufferPointer<UInt16>, endOfStream: Bool) -> (Candidate?, Int) {
        var best: Candidate?
        var holdStart = text.count
        var string: NSString?
        // An `NSString` over the buffer without copying it, made only if a regex needs it.
        func view() -> NSString {
            if let string { return string }
            let created: NSString =
                text.baseAddress.map {
                    NSString(
                        charactersNoCopy: UnsafeMutablePointer(mutating: $0), length: text.count, freeWhenDone: false)
                } ?? ""
            string = created
            return created
        }
        func consider(_ kind: Kind, _ field: Int, _ delimiter: ResponseDelimiter) {
            // Only a delimiter starting no later than the best candidate, or than the first
            // undecided text, can change the outcome.
            let limit = min(best?.match.range.lowerBound ?? text.count, holdStart)
            guard
                let match = delimiter.search(
                    text, from: position, through: limit, partial: !endOfStream, text: view)
            else { return }
            if !endOfStream, match.isPartial || delimiter.canGrow(match.range, count: text.count) {
                holdStart = min(holdStart, match.range.lowerBound)
                return
            }
            let candidate = Candidate(kind: kind, field: field, match: match)
            if let current = best, !precedes(candidate, current) { return }
            best = candidate
        }

        var watched: [(Kind, Int, ResponseDelimiter)] = []
        if let current, current != template.implicitField {
            if let close = template.fields[current].close { watched.append((.close, current, close)) }
        } else {
            for (index, field) in template.fields.enumerated() {
                if let open = field.open { watched.append((.open, index, open)) }
            }
            if let implicit = template.implicitField, let close = template.fields[implicit].close {
                watched.append((.close, implicit, close))
            }
        }
        // Literals first: they are cheap, and what they find narrows the regex searches.
        for (kind, field, delimiter) in watched where delimiter.isLiteral { consider(kind, field, delimiter) }
        for (kind, field, delimiter) in watched where !delimiter.isLiteral { consider(kind, field, delimiter) }
        // A delimiter that is still undecided and starts no later might be the real one.
        if let candidate = best, candidate.match.range.lowerBound >= holdStart { best = nil }
        return (best, holdStart)
    }

    /// Earliest start, then longest, then opens before closes, then field name.
    private func precedes(_ lhs: Candidate, _ rhs: Candidate) -> Bool {
        let (l, r) = (lhs.match.range, rhs.match.range)
        if l.lowerBound != r.lowerBound { return l.lowerBound < r.lowerBound }
        if l.count != r.count { return l.count > r.count }
        if lhs.kind != rhs.kind { return lhs.kind == .open }
        return template.fields[lhs.field].name.unicodeScalars.lexicographicallyPrecedes(
            template.fields[rhs.field].name.unicodeScalars)
    }

    /// Routes buffer text into the current region; text outside every region is dropped.
    private mutating func accumulate(_ range: Range<Int>, _ events: inout [ResponseEvent]) {
        guard !range.isEmpty, let current else { return }
        let field = template.fields[current]
        if !opened {
            events.append(.regionOpen(field: field.name))
            opened = true
        }
        let slice = buffer[range]
        body.append(contentsOf: slice)
        events.append(
            .regionChunk(
                field: field.name, text: UTF16Text.string(slice), isDirty: !field.content.isStreamable))
    }

    /// Closes the current region, stores its value and returns to the implicit region.
    private mutating func closeCurrent(_ events: inout [ResponseEvent]) throws {
        defer {
            current = template.implicitField
            captures = []
            body.removeAll(keepingCapacity: true)
            opened = false
        }
        guard let current, opened else { return }
        let field = template.fields[current]
        var value = try Self.value(of: field, body: UTF16Text.string(body[...]), captures: captures)
        if !toolParameters.isEmpty { value = typedToolCalls(value) }
        if let join = field.join {
            guard case let .string(string) = value else {
                throw ResponseParsingError.content(
                    "Field '\(field.name)': 'join' requires each match to parse to a string, got \(value.pythonTypeName)."
                )
            }
            switch output[field.name] {
            case nil, .null?: output[field.name] = value
            case let .string(previous)?: output[field.name] = .string(previous + join + string)
            case let previous?:
                throw ResponseParsingError.content(
                    "Field '\(field.name)': cannot join a string onto \(previous.pythonTypeName)")
            }
        } else if field.repeats {
            try output.update(field.name, default: .array([])) { stored throws in
                guard case var .array(values) = stored else {
                    throw ResponseParsingError.content(
                        "Field '\(field.name)': cannot append to \(stored.pythonTypeName)")
                }
                values.append(value)
                stored = .array(values)
            }
        } else {
            output[field.name] = value
        }
        events.append(.regionClose(field: field.name, value: value))
    }

    /// The parsed content of a region, reshaped by the field's transform.
    private static func value(
        of field: ResponseTemplate.Field, body: String, captures: [(name: String, value: String)]
    )
        throws -> ResponseValue
    {
        let parsed = try field.content.parse(body)
        guard let transform = field.transform else { return parsed }
        var scope: [String: ResponseValue] = [:]
        for (name, text) in captures { scope[name] = .string(text) }
        guard field.transformEach else {
            scope["content"] = parsed
            return try transform.apply(scope)
        }
        guard case let .array(items) = parsed else {
            throw ResponseParsingError.content(
                "Field '\(field.name)': transform_each requires the parsed content to be a list, got \(parsed.pythonTypeName)."
            )
        }
        var values: [ResponseValue] = []
        values.reserveCapacity(items.count)
        for item in items {
            guard case let .object(object) = item else {
                throw ResponseParsingError.content(
                    "Field '\(field.name)': transform_each requires each list element to be a dict, got \(item.pythonTypeName)."
                )
            }
            var itemScope = scope
            for (key, value) in object { itemScope[key] = value }
            values.append(try transform.apply(itemScope))
        }
        return .array(values)
    }

    // MARK: - Tool-call argument types

    private static func parameters(of tools: [ToolSpec]) throws -> [String: ResponseObject] {
        var parameters: [String: ResponseObject] = [:]
        for tool in tools {
            guard case let .object(tool) = ResponseValue(jinja: try ChatTemplateValue.make(tool)) else { continue }
            let function = tool["function"] ?? .object(tool)
            guard case let .object(function) = function, let name = function["name"]?.stringValue else { continue }
            parameters[name] = function["parameters"]?.objectValue?["properties"]?.objectValue ?? ResponseObject()
        }
        return parameters
    }

    /// Casts string arguments of tool calls to the types their schema declares.
    private func typedToolCalls(_ value: ResponseValue) -> ResponseValue {
        switch value {
        case let .array(items):
            return .array(items.map(typedToolCalls))
        case .object(var call):
            guard case var .object(function)? = call["function"], let name = function["name"]?.stringValue,
                case var .object(arguments)? = function["arguments"]
            else { return value }
            guard let properties = toolParameters[name], !properties.isEmpty else { return value }
            for (key, argument) in arguments {
                guard let schema = properties[key] else { continue }
                let types = Self.schemaTypes(schema)
                guard !types.isEmpty else { continue }
                switch argument {
                case let .string(raw):
                    arguments[key] = Self.coerce(raw, types)
                case let .array(items) where !types.contains("array"):
                    // Repeated keys collected by `merge_duplicates`.
                    arguments[key] = .array(
                        items.map { item in item.stringValue.map { Self.coerce($0, types) } ?? item })
                default:
                    break
                }
            }
            function["arguments"] = .object(arguments)
            call["function"] = .object(function)
            return .object(call)
        default:
            return value
        }
    }

    /// The JSON Schema types a parameter accepts, in declaration order.
    static func schemaTypes(_ schema: ResponseValue) -> [String] {
        guard case let .object(schema) = schema else { return [] }
        var types: [String] = []
        switch schema["type"] {
        case let .string(type)?: types.append(type)
        case let .array(declared)?: types.append(contentsOf: declared.compactMap(\.stringValue))
        default: break
        }
        for union in ["anyOf", "oneOf"] {
            if case let .array(choices)? = schema[union] {
                for choice in choices { types.append(contentsOf: schemaTypes(choice)) }
            }
        }
        if schema["nullable"]?.isTruthy == true, !types.contains("null") { types.append("null") }
        return types
    }

    /// The first of `types` that `raw` converts to; `raw` itself when none does.
    static func coerce(_ raw: String, _ types: [String]) -> ResponseValue {
        for type in types {
            switch type {
            case "integer":
                if let value = PythonText.int(raw) { return .int(value) }
                // JSON Schema integers include whole numbers written as floats, e.g. "3.0".
                guard let number = PythonText.float(raw), number.isFinite, number.rounded() == number else { continue }
                return Int(exactly: number).map(ResponseValue.int) ?? .double(number)
            case "number":
                guard let number = PythonText.float(raw), number.isFinite else { continue }
                if number.rounded() == number, !raw.contains("."), let integer = Int(exactly: number) {
                    return .int(integer)
                }
                return .double(number)
            case "boolean":
                switch PythonText.strip(raw).lowercased() {
                case "true", "1": return .bool(true)
                case "false", "0": return .bool(false)
                default: continue
                }
            case "null":
                let stripped = PythonText.strip(raw)
                if stripped == "null" || stripped == "None" { return .null }
            case "object", "array":
                guard let decoded = try? ResponseJSON.decode(raw) else { continue }
                if type == "object", case .object = decoded { return decoded }
                if type == "array", case .array = decoded { return decoded }
            default:
                continue
            }
        }
        return .string(raw)
    }
}
