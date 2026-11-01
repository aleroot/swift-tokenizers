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

/// Annotates generation blocks in source, then verifies the unmarked render byte for byte.
struct GenerationTemplate: Sendable {
    let template: Template
    let start: String
    let end: String
    let startName: String
    let endName: String

    init(source: String) throws {
        let nonce = UUID().uuidString.replacingOccurrences(of: "-", with: "")
        start = "\u{0}generation-\(nonce)-start\u{0}"
        end = "\u{0}generation-\(nonce)-end\u{0}"
        startName = "_generation_\(nonce)_start"
        endName = "_generation_\(nonce)_end"
        let normalized = source.replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        let annotated = try Self.annotate(normalized, startName: startName, endName: endName)
        template = try Template(annotated, with: .init(lstripBlocks: true, trimBlocks: true))
    }

    private static func annotate(_ source: String, startName: String, endName: String) throws -> String {
        var result = ""
        var cursor = source.startIndex
        var captures: [String] = []
        var inRaw = false
        var found = false
        while cursor < source.endIndex {
            if inRaw {
                guard let close = source[cursor...].firstRange(of: /\{%[-]?\s*endraw\s*[-]?%\}/) else {
                    result += source[cursor...]
                    break
                }
                result += source[cursor..<close.upperBound]
                cursor = close.upperBound
                inRaw = false
                continue
            }
            guard let open = source[cursor...].range(of: "{") else {
                result += source[cursor...]
                break
            }
            result += source[cursor..<open.lowerBound]
            let suffix = source[open.lowerBound...]
            let delimiter: String
            if suffix.hasPrefix("{{") {
                delimiter = "}}"
            } else if suffix.hasPrefix("{%") {
                delimiter = "%}"
            } else if suffix.hasPrefix("{#") {
                delimiter = "#}"
            } else {
                result += "{"
                cursor = source.index(after: open.lowerBound)
                continue
            }
            let start = open.lowerBound
            var i = source.index(start, offsetBy: 2)
            var quote: Character?
            var escaped = false
            var close: String.Index?
            while i < source.endIndex {
                let c = source[i]
                if delimiter != "#}", let current = quote {
                    if escaped {
                        escaped = false
                    } else if c == "\\" {
                        escaped = true
                    } else if c == current {
                        quote = nil
                    }
                } else if delimiter != "#}", c == "'" || c == "\"" {
                    quote = c
                } else if source[i...].hasPrefix(delimiter) {
                    close = i
                    break
                }
                i = source.index(after: i)
            }
            guard let close else {
                result += source[start...]
                cursor = source.endIndex
                break
            }
            let after = source.index(close, offsetBy: 2)
            let tag = source[start..<after]
            cursor = after
            guard delimiter == "%}" else { result += tag; continue }
            var body = source[source.index(start, offsetBy: 2)..<close]
            if body.first == "-" { body = body.dropFirst() }
            if body.last == "-" { body = body.dropLast() }
            let statement = body.trimmingCharacters(in: .whitespacesAndNewlines)
            let keyword = String(statement.prefix { $0.isLetter || $0 == "_" })
            if keyword == "raw" { inRaw = true; result += tag; continue }
            if ["macro", "call", "filter"].contains(keyword)
                || (keyword == "set" && !Self.isSetAssignment(statement))
            {
                captures.append(keyword)
            } else if keyword.hasPrefix("end"), captures.last == String(keyword.dropFirst(3)) {
                captures.removeLast()
            }
            if keyword == "generation" {
                guard statement == "generation", captures.isEmpty else {
                    throw TokenizerError.chatTemplate(
                        "Assistant masks require generation blocks in direct output, outside set, macro, call and filter blocks"
                    )
                }
                found = true
                result += tag
                let stripRight = source[source.index(before: close)] == "-"
                if stripRight {
                    while cursor < source.endIndex, source[cursor].isWhitespace {
                        result.append(source[cursor]); cursor = source.index(after: cursor)
                    }
                } else if cursor < source.endIndex, source[cursor] == "\n" || source[cursor] == "\r\n" {
                    result.append(source[cursor]); cursor = source.index(after: cursor)
                }
                result += "{{ \(startName) }}"
            } else if keyword == "endgeneration" {
                guard statement == "endgeneration" else {
                    throw TokenizerError.chatTemplate("Invalid generation block")
                }
                if !tag.hasPrefix("{%-") {
                    let prefix = source[..<start]
                    let lineStart = prefix.lastIndex(of: "\n").map { source.index(after: $0) } ?? source.startIndex
                    let indentation = source[lineStart..<start]
                    if indentation.allSatisfy({ $0 == " " || $0 == "\t" }) {
                        result.removeLast(indentation.count)
                    }
                }
                result += tag.hasPrefix("{%-") ? "{{- \(endName) }}" : "{{ \(endName) }}"
                result += tag
            } else {
                result += tag
            }
        }
        guard found else { throw TokenizerError.chatTemplate("Assistant masks require {% generation %} annotations") }
        return result
    }

    private static func isSetAssignment(_ statement: String) -> Bool {
        var quote: Character?
        var escaped = false
        var depth = 0
        for c in statement {
            if let current = quote {
                if escaped { escaped = false } else if c == "\\" { escaped = true } else if c == current { quote = nil }
            } else if c == "'" || c == "\"" {
                quote = c
            } else if c == "(" || c == "[" || c == "{" {
                depth += 1
            } else if c == ")" || c == "]" || c == "}" {
                depth -= 1
            } else if c == "=" && depth == 0 {
                return true
            }
        }
        return false
    }

    func ranges(context: [String: Jinja.Value], expected: String) throws -> [Range<Int>] {
        var annotatedContext = context
        annotatedContext[startName] = .string(start)
        annotatedContext[endName] = .string(end)
        let rendered = try template.render(annotatedContext)
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
