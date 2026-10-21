import Foundation

/// A failure while loading a response template or parsing a response with one.
public enum ResponseParsingError: Error, LocalizedError, Sendable, Equatable {
    /// The tokenizer has no `response_template`, and none was passed.
    case missingTemplate
    /// The template is malformed.
    case invalidTemplate(String)
    /// A region's text could not be turned into its value.
    case content(String)
    /// Fields the template marks as required (`"optional": false`) never appeared.
    case missingRequiredFields([String])
    /// The parser was used after ``ResponseParser/finalize()``.
    case alreadyFinalized

    public var errorDescription: String? {
        switch self {
        case .missingTemplate:
            "This tokenizer does not have a response_template for parsing chat responses."
        case let .invalidTemplate(message):
            "Invalid response_template: \(message)"
        case let .content(message):
            "Response parsing failed: \(message)"
        case let .missingRequiredFields(fields):
            "Required response_template fields missing from parsed output: \(fields)"
        case .alreadyFinalized:
            "ResponseParser already finalized."
        }
    }
}

/// A compiled `response_template`: the declarative description of how a model lays out the
/// regions of its reply (reasoning, content, tool calls) that turns generated text back into a
/// structured assistant message.
///
/// This is the format `transformers` reads from the `response_template` key of
/// `tokenizer_config.json` and uses in `parse_response`, and it is parsed with the same rules.
/// Regular expressions in the template are compiled once, here.
public struct ResponseTemplate: Sendable {
    struct Field: Sendable {
        let name: String
        let open: ResponseDelimiter?
        let close: ResponseDelimiter?
        let content: ResponseContentParser
        let repeats: Bool
        let join: String?
        let optional: Bool
        let transform: ResponseTransform?
        let transformEach: Bool
    }

    let defaults: ResponseObject
    let fields: [Field]
    let fieldIndex: [String: Int]
    /// The field without an opening delimiter, which collects text outside every other region.
    let implicitField: Int?
    let startAnchor: ResponseDelimiter

    /// Compiles a template from its JSON text, keeping the key order the text uses.
    public init(jsonString: String) throws {
        let value: ResponseValue
        do { value = try ResponseJSON.decode(jsonString) } catch {
            throw ResponseParsingError.invalidTemplate("the template is not valid JSON")
        }
        try self.init(value)
    }

    /// Compiles a template held as a value, for instance one built in code.
    public init(_ spec: ResponseValue) throws {
        guard let spec = spec.objectValue else {
            throw ResponseParsingError.invalidTemplate("response_template must be a dict, got \(spec.pythonTypeName)")
        }
        try self.init(object: spec)
    }

    /// Compiles the `response_template` of a tokenizer configuration. `Config` does not keep
    /// key order, so keys are taken in sorted order, which is how `transformers` writes
    /// `tokenizer_config.json`.
    init(config: Config) throws {
        try self.init(ResponseValue(config: config))
    }

    private init(object spec: ResponseObject) throws(ResponseParsingError) {
        let version = spec["version"] ?? .int(1)
        guard version == .int(1) || version == .double(1) || version == .bool(true) else {
            throw .invalidTemplate("Unsupported response_template version: \(version)")
        }
        let allowed: Set = ["version", "defaults", "fields", "start_anchor", "start_anchor_pattern"]
        let unknown = spec.keys.filter { !allowed.contains($0) }
        guard unknown.isEmpty else { throw .invalidTemplate("Unknown keys in response_template: \(unknown.sorted())") }
        let defaults = spec["defaults"] ?? .object([:])
        guard let defaultsObject = defaults.objectValue else {
            throw .invalidTemplate("response_template.defaults must be a dict")
        }
        guard let fieldSpecs = spec["fields"]?.objectValue, !fieldSpecs.isEmpty else {
            throw .invalidTemplate("response_template.fields must be a non-empty dict")
        }

        var fields: [Field] = []
        var fieldIndex: [String: Int] = [:]
        for (name, value) in fieldSpecs {
            fieldIndex[name] = fields.count
            fields.append(try Self.field(name: name, value))
        }
        let implicit = fields.indices.filter { fields[$0].open == nil }
        if implicit.count > 1 {
            throw .invalidTemplate(
                "At most one field may omit 'open'/'open_pattern' (that field becomes the implicit-open / "
                    + "leftover sink). Found: \(implicit.map { fields[$0].name }.joined(separator: ", "))")
        }
        guard
            let startAnchor = try ResponseDelimiter.compile(
                spec, literal: "start_anchor", pattern: "start_anchor_pattern", scope: "response_template")
        else {
            throw .invalidTemplate("response_template must define 'start_anchor' or 'start_anchor_pattern'.")
        }

        self.defaults = defaultsObject
        self.fields = fields
        self.fieldIndex = fieldIndex
        self.implicitField = implicit.first
        self.startAnchor = startAnchor
    }

    private static func field(name: String, _ value: ResponseValue) throws(ResponseParsingError) -> Field {
        let scope = "Field '\(name)'"
        guard let spec = value.objectValue else { throw .invalidTemplate("\(scope) must be a dict") }
        let allowed: Set = [
            "open", "open_pattern", "close", "close_pattern", "content", "content_args", "repeats", "join",
            "optional", "transform", "transform_each",
        ]
        let unknown = spec.keys.filter { !allowed.contains($0) }
        guard unknown.isEmpty else { throw .invalidTemplate("\(scope): unknown keys \(unknown.sorted())") }

        let contentName = spec["content"] ?? "text"
        guard let parserName = contentName.stringValue, ResponseContentParser.names.contains(parserName) else {
            throw .invalidTemplate(
                "\(scope): unknown content parser '\(contentName)'. Available: \(ResponseContentParser.names.sorted())")
        }
        let open = try ResponseDelimiter.compile(spec, literal: "open", pattern: "open_pattern", scope: scope)
        let close = try ResponseDelimiter.compile(spec, literal: "close", pattern: "close_pattern", scope: scope)

        var join: String?
        if let value = spec["join"], value != .null {
            guard let string = value.stringValue else {
                throw .invalidTemplate("\(scope): 'join' must be a string, got \(value.pythonTypeName)")
            }
            join = string
        }
        let repeats = spec["repeats"]?.isTruthy ?? false
        if join != nil, !repeats { throw .invalidTemplate("\(scope): 'join' requires 'repeats': true") }

        let transformValue = spec["transform"].flatMap { $0 == .null ? nil : $0 }
        let transformEach: Bool
        switch spec["transform_each"] ?? .bool(false) {
        case let .bool(value): transformEach = value
        case let other:
            throw .invalidTemplate("\(scope): transform_each must be a bool, got \(other.pythonTypeName)")
        }
        if transformEach, transformValue == nil {
            throw .invalidTemplate("\(scope): transform_each is set but no transform was provided")
        }
        let transform = try transformValue.map { value throws(ResponseParsingError) in
            try ResponseTransform(value, scope: scope)
        }
        if transform == nil {
            let captured = Set((open?.groupNames ?? []) + (close?.groupNames ?? [])).sorted()
            if !captured.isEmpty {
                throw .invalidTemplate(
                    "\(scope): open_pattern/close_pattern declares named group(s) \(captured), but the field has no "
                        + "'transform'. Named captures are only surfaced through a 'transform' template.")
            }
        }

        let arguments: ResponseObject
        switch spec["content_args"] ?? .object([:]) {
        case let .object(object): arguments = object
        case let other: throw .invalidTemplate("\(scope): content_args must be a dict, got \(other.pythonTypeName)")
        }

        return Field(
            name: name, open: open, close: close,
            content: try ResponseContentParser(name: parserName, arguments: arguments, scope: scope),
            repeats: repeats, join: join, optional: spec["optional"]?.isTruthy ?? true,
            transform: transform, transformEach: transformEach)
    }

    /// The part of `prefix` after the last start anchor, where the assistant message begins
    /// (`truncate_past_last_anchor`). The whole prefix when the anchor never occurs.
    func truncatePastLastAnchor(_ prefix: [UInt16]) -> ArraySlice<UInt16> {
        let end = prefix.withUnsafeBufferPointer { startAnchor.lastMatchEnd(in: $0) }
        return prefix[(end ?? 0)...]
    }
}

extension ResponseValue {
    /// A configuration value as a response value, with object keys in sorted order.
    init(config: Config) {
        switch config.value {
        case .null: self = .null
        case let .string(value): self = .string(value.string)
        case let .integer(value): self = .int(value)
        case let .boolean(value): self = .bool(value)
        case let .floating(value): self = .double(value)
        case let .array(values): self = .array(values.map(ResponseValue.init(config:)))
        case let .dictionary(dictionary):
            let keys = dictionary.keys.sorted { $0.string.utf8.lexicographicallyPrecedes($1.string.utf8) }
            self = .object(ResponseObject(keys.map { ($0.string, ResponseValue(config: dictionary[$0]!)) }))
        case let .token((id, content)): self = .array([.int(Int(id)), .string(content.string)])
        case .stringMap, .stringPairs, .scoredTokens:
            // Packed vocabulary tables never occur inside a tokenizer configuration.
            self = .null
        }
    }
}
