import Foundation

/// Turns the text of a closed region into its value (`content` and `content_args`).
indirect enum ResponseContentParser: Sendable {
    case text(strip: Bool)
    case int
    case float
    case bool(strip: Bool)
    case json(JSONDialect)
    case xmlInline(tag: PythonRegex, value: ResponseContentParser?, mergeDuplicates: Bool)
    case kvLines(lineSeparator: String, keyValueSeparator: String, strip: Bool, value: ResponseContentParser?)

    struct JSONDialect: Sendable {
        var stringDelimiters: [(open: String, close: String)] = []
        var unquotedKeys = false
        var allowNonJSON = false
        var strip = true
    }

    static let names: Set<String> = ["text", "int", "float", "bool", "json", "xml-inline", "kv-lines"]

    /// Regions whose chunks are already part of the final value; the others stream raw text
    /// marked dirty and produce their value on close.
    var isStreamable: Bool {
        switch self {
        case .text, .int, .float, .bool: true
        default: false
        }
    }

    init(name: String, arguments: ResponseObject, scope: String) throws(ResponseParsingError) {
        let strip = arguments["strip"]?.isTruthy ?? true
        func valueParser() throws(ResponseParsingError) -> ResponseContentParser? {
            guard let spec = arguments["value_parser"], spec != .null else { return nil }
            guard let object = spec.objectValue else {
                throw .invalidTemplate("\(scope): 'value_parser' must be a dict")
            }
            let name = object["name"]?.stringValue ?? "text"
            guard Self.names.contains(name) else {
                throw .invalidTemplate("\(scope): unknown value_parser '\(name)'")
            }
            let arguments = object["args"]?.objectValue ?? ResponseObject()
            return try ResponseContentParser(name: name, arguments: arguments, scope: scope)
        }
        switch name {
        case "text": self = .text(strip: strip)
        case "int": self = .int
        case "float": self = .float
        case "bool": self = .bool(strip: strip)
        case "json":
            var dialect = JSONDialect(strip: strip)
            dialect.unquotedKeys = arguments["unquoted_keys"]?.isTruthy ?? false
            dialect.allowNonJSON = arguments["allow_non_json"]?.isTruthy ?? false
            for pair in arguments["string_delims"]?.arrayValue ?? [] {
                guard let pair = pair.arrayValue, pair.count == 2, let open = pair[0].stringValue,
                    let close = pair[1].stringValue
                else { throw .invalidTemplate("\(scope): 'string_delims' must hold [open, close] string pairs") }
                dialect.stringDelimiters.append((open, close))
            }
            self = .json(dialect)
        case "xml-inline":
            guard let source = arguments["tag_pattern"]?.stringValue else {
                throw .invalidTemplate("\(scope): xml-inline: 'tag_pattern' content_arg is required")
            }
            let tag: PythonRegex
            do { tag = try PythonRegex(source) } catch {
                throw .invalidTemplate("\(scope): invalid tag_pattern regex: \(source)")
            }
            self = .xmlInline(
                tag: tag, value: try valueParser(), mergeDuplicates: arguments["merge_duplicates"]?.isTruthy ?? false)
        case "kv-lines":
            let lineSeparator = arguments["line_sep"]?.stringValue ?? "\n"
            let keyValueSeparator = arguments["kv_sep"]?.stringValue ?? ":"
            guard !lineSeparator.isEmpty, !keyValueSeparator.isEmpty else {
                throw .invalidTemplate("\(scope): kv-lines separators cannot be empty")
            }
            self = .kvLines(
                lineSeparator: lineSeparator, keyValueSeparator: keyValueSeparator, strip: strip,
                value: try valueParser())
        default:
            throw .invalidTemplate(
                "\(scope): unknown content parser '\(name)'. Available: \(Self.names.sorted())")
        }
    }

    func parse(_ text: String) throws(ResponseParsingError) -> ResponseValue {
        switch self {
        case let .text(strip):
            return .string(strip ? PythonText.strip(text) : text)
        case .int:
            guard let value = PythonText.int(text) else {
                throw .content("invalid literal for int(): \(text.debugDescription)")
            }
            return .int(value)
        case .float:
            guard let value = PythonText.float(text) else {
                throw .content("could not convert string to float: \(text.debugDescription)")
            }
            return .double(value)
        case let .bool(strip):
            let value = (strip ? PythonText.strip(text) : text).lowercased()
            return .bool(value == "true" || value == "1")
        case let .json(dialect):
            return try Self.json(text, dialect)
        case let .xmlInline(tag, valueParser, merge):
            var output = ResponseObject()
            let hasValue = tag.groupNames.contains("value")
            for match in tag.matches(in: text) {
                guard let key = match.captures.first(where: { $0.name == "key" })?.value else {
                    throw .content("xml-inline: tag_pattern must have a named group 'key'")
                }
                let raw = hasValue ? match.captures.first(where: { $0.name == "value" })?.value : ""
                let value: ResponseValue
                if let raw {
                    value = try valueParser?.parse(raw) ?? .string(raw)
                } else if valueParser == nil {
                    value = .null
                } else {
                    throw .content("xml-inline: the 'value' group did not participate in the match")
                }
                if merge, let existing = output[key] {
                    if case var .array(values) = existing {
                        values.append(value)
                        output[key] = .array(values)
                    } else {
                        output[key] = .array([existing, value])
                    }
                } else {
                    output[key] = value
                }
            }
            return .object(output)
        case let .kvLines(lineSeparator, keyValueSeparator, strip, valueParser):
            var output = ResponseObject()
            for rawLine in PythonText.split(text, separator: lineSeparator) {
                let line = strip ? PythonText.strip(rawLine) : rawLine
                guard !line.isEmpty, let separator = line.range(of: keyValueSeparator, options: .literal) else {
                    continue
                }
                var key = line[..<separator.lowerBound]
                var value = line[separator.upperBound...]
                if strip {
                    key = PythonText.strip(key)
                    value = PythonText.strip(value)
                }
                output[String(key)] = try valueParser?.parse(String(value)) ?? .string(String(value))
            }
            return .object(output)
        }
    }

    /// `json.loads` behind the dialect knobs for model-emitted quirks.
    private static func json(_ text: String, _ dialect: JSONDialect) throws(ResponseParsingError) -> ResponseValue {
        var working = text
        var captured: [String] = []
        if !dialect.stringDelimiters.isEmpty {
            if text.unicodeScalars.contains(where: { $0 == "\u{01}" || $0 == "\u{02}" }) {
                throw .content("json: input contains reserved sentinel characters (\\x01/\\x02); cannot parse safely.")
            }
            // `re.sub(escape(open) + "(.*?)" + escape(close), …, flags=DOTALL)`, without a regex.
            for (open, close) in dialect.stringDelimiters {
                var result = ""
                var rest = working[...]
                while let start = rest.range(of: open, options: .literal),
                    let end = rest[start.upperBound...].range(of: close, options: .literal)
                {
                    result += rest[..<start.lowerBound]
                    captured.append(String(rest[start.upperBound..<end.lowerBound]))
                    result += "\u{01}\(captured.count - 1)\u{02}"
                    rest = rest[end.upperBound...]
                }
                result += rest
                working = result
            }
        }
        if dialect.unquotedKeys { working = quotingBareKeys(working) }
        for (index, string) in captured.enumerated() {
            var literal = ""
            ResponseValue.writeString(string, to: &literal)
            working = working.replacingOccurrences(of: "\u{01}\(index)\u{02}", with: literal, options: .literal)
        }
        do {
            return try ResponseJSON.decode(working)
        } catch .invalid(let message) {
            if dialect.allowNonJSON { return .string(dialect.strip ? PythonText.strip(text) : text) }
            if working == text {
                throw .content(
                    "json parser could not parse region as JSON.\nContent: \(text.debugDescription)\nError: \(message)")
            }
            throw .content(
                "json: could not parse after dialect transforms.\nOriginal: \(text.debugDescription)\n"
                    + "Transformed: \(working.debugDescription)\nError: \(message)")
        } catch {
            throw .content("json: maximum nesting depth exceeded")
        }
    }

    /// `re.sub(r"(?<=[{,])(\w+):", r'"\1":', text)`: quotes the bare keys after `{` and `,`.
    private static func quotingBareKeys(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        var output = String.UnicodeScalarView()
        var index = 0
        while index < scalars.count {
            if index > 0, scalars[index - 1] == "{" || scalars[index - 1] == ",", PythonText.isWord(scalars[index]) {
                var end = index
                while end < scalars.count, PythonText.isWord(scalars[end]) { end += 1 }
                if end < scalars.count, scalars[end] == ":" {
                    output.append("\"")
                    output.append(contentsOf: scalars[index..<end])
                    output.append(contentsOf: "\":".unicodeScalars)
                    index = end + 1
                    continue
                }
            }
            output.append(scalars[index])
            index += 1
        }
        return String(output)
    }
}

/// A `transform` template with its placeholders resolved at load time.
indirect enum ResponseTransform: Sendable {
    case literal(ResponseValue)
    /// `{root.key.key}`.
    case placeholder(path: [String], source: String)
    case object([(String, ResponseTransform)])
    case array([ResponseTransform])

    init(_ value: ResponseValue, scope: String) throws(ResponseParsingError) {
        switch value {
        case let .object(object):
            var entries: [(String, ResponseTransform)] = []
            for (key, value) in object { entries.append((key, try ResponseTransform(value, scope: scope))) }
            self = .object(entries)
        case let .array(values):
            var elements: [ResponseTransform] = []
            for value in values { elements.append(try ResponseTransform(value, scope: scope)) }
            self = .array(elements)
        case let .string(string):
            let scalars = Array(string.unicodeScalars)
            if Self.placeholderEnd(in: scalars, at: 0) == scalars.count {
                let path = String(String.UnicodeScalarView(scalars.dropFirst().dropLast()))
                self = .placeholder(path: path.components(separatedBy: "."), source: path)
            } else if scalars.indices.contains(where: { Self.placeholderEnd(in: scalars, at: $0) != nil }) {
                throw .invalidTemplate(
                    "\(scope): transform string \(string.debugDescription) mixes a {placeholder} with literal text. "
                        + "Use either a whole-string placeholder (e.g. \"{content}\") or a plain literal; "
                        + "string interpolation is not supported.")
            } else {
                self = .literal(value)
            }
        default:
            self = .literal(value)
        }
    }

    func apply(_ scope: [String: ResponseValue]) throws(ResponseParsingError) -> ResponseValue {
        switch self {
        case let .literal(value):
            return value
        case let .array(elements):
            var values: [ResponseValue] = []
            values.reserveCapacity(elements.count)
            for element in elements { values.append(try element.apply(scope)) }
            return .array(values)
        case let .object(entries):
            var object = ResponseObject()
            for (key, transform) in entries { object[key] = try transform.apply(scope) }
            return .object(object)
        case let .placeholder(path, source):
            guard var value = scope[path[0]] else {
                throw .content(
                    "transform placeholder '{\(source)}' is not defined. Available: \(scope.keys.sorted())")
            }
            for key in path.dropFirst() {
                guard let object = value.objectValue else {
                    throw .content(
                        "transform placeholder '{\(source)}' cannot index into \(value.pythonTypeName) at '\(key)'")
                }
                guard let next = object[key] else {
                    throw .content(
                        "transform placeholder '{\(source)}' is missing key '\(key)'. Available: \(object.keys.sorted())"
                    )
                }
                value = next
            }
            return value
        }
    }

    /// The end of a `{name.key}` placeholder (`\{\w+(?:\.\w+)*\}`) starting at `start`.
    private static func placeholderEnd(in scalars: [Unicode.Scalar], at start: Int) -> Int? {
        guard scalars[start] == "{" else { return nil }
        var index = start + 1
        while true {
            let wordStart = index
            while index < scalars.count, PythonText.isWord(scalars[index]) { index += 1 }
            guard index > wordStart, index < scalars.count else { return nil }
            if scalars[index] == "}" { return index + 1 }
            guard scalars[index] == "." else { return nil }
            index += 1
        }
    }
}
