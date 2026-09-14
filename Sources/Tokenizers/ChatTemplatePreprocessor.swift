import Foundation

/// Preserve jinja2 whitespace-control semantics around comments and literal braces.
///
/// swift-jinja applies whitespace control (`{{-`, `{%-`, `-}}`, `-%}`) with a
/// source-level regex pass before lexing. Two cases exercised by the Gemma 4 chat
/// template motivated this pre-pass:
///
/// 1. **Comment strip markers were ignored before 2.5.1.** `{#-` strips whitespace before the
///    comment and `-#}` all whitespace after it (comments themselves never produce
///    output). Version 2.5.1 handles these markers, but comment rewriting remains
///    necessary for the literal-brace interaction described below.
/// 2. **A literal `{` before a stripped tag keeps one space.** To avoid merging the
///    literal brace with the tag delimiter, swift-jinja rewrites `{\s+{{-` as `{ {{`,
///    emitting a spurious space. jinja2 strips the whitespace entirely: `{ {{- x -}}`
///    renders `{` followed by `x`.
///
/// This pre-pass rewrites the template so that swift-jinja's own (correct for the
/// remaining cases) pass produces jinja2-identical output:
///
/// * Comments are removed outright, applying their strip markers to the surrounding
///   text. Removing the whole comment side-steps the brace-merging problem for
///   `{ {#-`.
/// * A literal `{` that precedes a whitespace-controlled `{{-`/`{%-` is rewritten as
///   the expression `{{ '{' }}`, which renders the same character but cannot merge
///   with the following delimiter.
///
/// The scanner respects quoted strings inside `{{ … }}` / `{% … %}` tags. Hashes
/// inside those strings are escaped for swift-jinja's lexer so its source-level
/// whitespace pass cannot interpret literal comment markers as template syntax.
enum ChatTemplatePreprocessor {
    /// Apply the same template options for normal rendering and public-AST generation tracking.
    static func preparedSource(_ source: String) -> String {
        let source = preprocess(source)
        let lines = source.components(separatedBy: .newlines).map { line in
            line.contains(/^[ \t]*{[#%]/) ? line.replacing(/^[ \t]*/, with: "") : line
        }
        return lines.joined(separator: "\n").replacing(/%}\n/, with: "%}").replacing(/#}\n/, with: "#}")
    }

    static func preprocess(_ source: String) -> String {
        // Jinja2's lexer normalizes source line endings and defaults
        // keep_trailing_newline to false. Strip exactly one source newline;
        // newlines emitted by expressions must remain untouched.
        var source = source.replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        if source.hasSuffix("\n") { source.removeLast() }
        guard
            source.contains("{#") || source.contains("-#}") || source.contains("{ ") || source.contains("{\n")
                || source.contains("{\t")
        else {
            return source
        }
        var scalars = Substring(source)
        var out = String()
        out.reserveCapacity(source.count)

        while !scalars.isEmpty {
            if scalars.hasPrefix("{{") || scalars.hasPrefix("{%") {
                copyTag(&scalars, into: &out)
            } else if scalars.hasPrefix("{#") {
                skipComment(&scalars, into: &out)
            } else if scalars.hasPrefix("{"), let (delimiter, consumed) = strippedTagAfterBrace(scalars) {
                // Literal `{`, whitespace, then a whitespace-controlled tag:
                // jinja2 strips the whitespace; swift-jinja would keep one space.
                out += "{{ '{' }}" + delimiter
                scalars = scalars.dropFirst(consumed)
            } else {
                out.append(scalars.removeFirst())
            }
        }
        return out
    }

    /// If `scalars` is `{ <whitespace>+ {{-` or `{ <whitespace>+ {%-`, returns the
    /// tag delimiter (`{{-` / `{%-`) and the number of characters covering the
    /// brace, the whitespace, and the delimiter. Otherwise returns nil.
    private static func strippedTagAfterBrace(_ scalars: Substring) -> (String, Int)? {
        var rest = scalars.dropFirst()  // past `{`
        var sawWhitespace = false
        while let c = rest.first, c == " " || c == "\t" || c == "\n" || c == "\r" {
            sawWhitespace = true
            rest = rest.dropFirst()
        }
        guard sawWhitespace else { return nil }
        let consumed = scalars.count - rest.count + 3
        if rest.hasPrefix("{{-") { return ("{{-", consumed) }
        if rest.hasPrefix("{%-") { return ("{%-", consumed) }
        return nil
    }

    /// Copies a tag while shielding quoted hashes from swift-jinja's regex pass.
    private static func copyTag(_ scalars: inout Substring, into out: inout String) {
        let close: String = scalars.hasPrefix("{{") ? "}}" : "%}"
        out.append(scalars.removeFirst())
        out.append(scalars.removeFirst())
        var quote: Character? = nil
        var escaped = false
        while !scalars.isEmpty {
            let c = scalars.removeFirst()
            if quote != nil, !escaped, c == "#" {
                // swift-jinja 2.5.1 strips literal `{#-` / `-#}` before lexing.
                // Its string lexer decodes `\#` back to `#`, preserving the value.
                out.append("\\")
            }
            out.append(c)
            if let q = quote {
                if escaped {
                    escaped = false
                } else if c == "\\" {
                    escaped = true
                } else if c == q {
                    quote = nil
                }
            } else if c == "'" || c == "\"" {
                quote = c
            } else if quote == nil, scalars.hasPrefix(close.dropFirst()), String(c) + close.dropFirst() == close {
                out.append(scalars.removeFirst())
                return
            }
        }
    }

    /// Replaces a `{# … #}` comment with an empty comment `{##}`, applying
    /// `{#-` / `-#}` whitespace stripping.
    ///
    /// Comments never produce output, but they still *segment* the text in jinja2:
    /// a `{%-` after a comment never strips whitespace that precedes the comment.
    /// Emitting an empty comment preserves that segmentation while the strip
    /// markers are applied here. Plain (undashed) comments keep their surroundings;
    /// swift-jinja's own `lstrip_blocks` / `trim_blocks` (always enabled, as in
    /// transformers) then apply to the empty comment exactly as jinja2 applies
    /// them to the original one.
    private static func skipComment(_ scalars: inout Substring, into out: inout String) {
        scalars = scalars.dropFirst(2)  // past `{#`
        let stripLeft = scalars.hasPrefix("-")
        if stripLeft { scalars = scalars.dropFirst() }

        // Find the closing `#}`; check for a `-` strip marker before it.
        var rest = scalars
        var stripRight = false
        var previous: Character? = nil
        var closed = false
        while !rest.isEmpty {
            let c = rest.removeFirst()
            if c == "#", rest.hasPrefix("}") {
                stripRight = previous == "-"
                rest = rest.dropFirst()
                closed = true
                break
            }
            previous = c
        }
        guard closed else {
            // Unterminated comment: emit the consumed text and leave the rest to swift-jinja's
            // own lexer error.
            out += stripLeft ? "{#-" : "{#"
            out += scalars
            scalars = Substring()
            return
        }
        scalars = rest

        if stripLeft {
            while let c = out.last, c == " " || c == "\t" || c == "\n" || c == "\r" {
                out.removeLast()
            }
        }
        out += "{##}"
        if stripRight {
            while let c = scalars.first, c == " " || c == "\t" || c == "\n" || c == "\r" {
                scalars = scalars.dropFirst()
            }
        }
    }
}
