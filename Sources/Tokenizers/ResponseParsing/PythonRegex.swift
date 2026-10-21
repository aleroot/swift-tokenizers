import Foundation

/// A regular expression written for Python's `regex` module, the engine `transformers` uses for
/// response templates, compiled for ICU (`NSRegularExpression`).
///
/// ICU is used rather than Swift `Regex`: it matches code points as Python does, where `Regex`
/// matches grapheme clusters by default, it reports when an attempt runs out of text (which
/// streaming needs), and it is several times faster on these patterns.
///
/// Patterns are compiled as `transformers` compiles them: `DOTALL`, no `MULTILINE`, Unicode
/// classes. Python-only syntax is translated: `(?P<name>…)`, `(?P=name)`, `\Z`, and a `[`, `&&`
/// or `--` inside a class, or a `{` that is not a quantifier, which are literals in Python.
struct PythonRegex: @unchecked Sendable {
    /// A match in UTF-16 offsets, with the named groups that took part, in pattern order.
    struct Match {
        var range: Range<Int>
        var captures: [(name: String, value: String)] = []
    }

    /// How an attempt starting at one position ends.
    enum Attempt {
        case match(Match)
        /// The text ran out before the attempt could succeed or fail.
        case reachedEnd
        case failed
    }

    let expression: NSRegularExpression
    /// The same pattern as `(?:P)|(?!)`, for anchored attempts. The never-matching alternative
    /// brings the minimum match length to zero, which stops ICU from reporting `hitEnd` for any
    /// attempt with less text left than that minimum, without reading the text.
    private let attemptExpression: NSRegularExpression
    /// Python group names, in pattern order; ICU knows them as `g0`, `g1`, …
    let groupNames: [String]
    /// Text every match contains, in UTF-16 units, or empty when none is known.
    let requiredLiteral: [UInt16]
    /// Whether every match starts with ``requiredLiteral``.
    let startsWithRequiredLiteral: Bool

    init(_ pattern: String) throws {
        let source = Array(pattern.unicodeScalars)
        let translated = Self.translate(source)
        let options: NSRegularExpression.Options = [.dotMatchesLineSeparators, .useUnixLineSeparators]
        expression = try NSRegularExpression(pattern: translated.pattern, options: options)
        attemptExpression = try NSRegularExpression(pattern: "(?:\(translated.pattern))|(?!)", options: options)
        groupNames = translated.groupNames
        (requiredLiteral, startsWithRequiredLiteral) = Self.requiredLiteral(source)
    }

    // MARK: Matching

    /// The leftmost match at or after `start`, as `pattern.search(text, start)` finds it:
    /// lookbehinds see the text before `start`, and `^` matches only at the very beginning.
    func firstMatch(in text: NSString, units: UnsafeBufferPointer<UInt16>, from start: Int) -> Match? {
        if !requiredLiteral.isEmpty, units.firstRange(of: requiredLiteral, from: start) == nil { return nil }
        let range = NSRange(location: start, length: text.length - start)
        let result = expression.firstMatch(
            in: text as String, options: [.withTransparentBounds, .withoutAnchoringBounds], range: range)
        return result.map { match($0, in: text) }
    }

    /// An attempt anchored at `start`, as Python's partial search makes it at each position: a
    /// match wins, and otherwise reaching the end of the text makes it a partial match.
    func attempt(in text: NSString, at start: Int) -> Attempt {
        var outcome = Attempt.failed
        attemptExpression.enumerateMatches(
            in: text as String,
            options: [.anchored, .reportCompletion, .withTransparentBounds, .withoutAnchoringBounds],
            range: NSRange(location: start, length: text.length - start)
        ) { result, flags, stop in
            if let result {
                outcome = .match(match(result, in: text))
                stop.pointee = true
            } else if flags.contains(.hitEnd) {
                outcome = .reachedEnd
            }
        }
        return outcome
    }

    /// Every non-overlapping match, as `re.finditer` reports them.
    func matches(in text: String) -> [Match] {
        let string = text as NSString
        return expression.matches(in: text, range: NSRange(location: 0, length: string.length))
            .map { match($0, in: string) }
    }

    private func match(_ result: NSTextCheckingResult, in text: NSString) -> Match {
        var match = Match(range: result.range.location..<NSMaxRange(result.range))
        for (index, name) in groupNames.enumerated() {
            let range = result.range(withName: "g\(index)")
            if range.location != NSNotFound { match.captures.append((name, text.substring(with: range))) }
        }
        return match
    }

    // MARK: Translation

    private static func translate(_ source: [Unicode.Scalar]) -> (pattern: String, groupNames: [String]) {
        var output = String.UnicodeScalarView()
        var names: [String] = []
        var index = 0
        var classStart: Int?

        func emit(_ text: String) { output.append(contentsOf: text.unicodeScalars) }
        func starts(_ prefix: String) -> Bool { source[index...].starts(with: prefix.unicodeScalars) }
        /// The name between `from` and `terminator`, and the index after the terminator.
        func name(from start: Int, until terminator: Unicode.Scalar) -> (String, Int)? {
            guard let end = source[start...].firstIndex(of: terminator) else { return nil }
            return (String(String.UnicodeScalarView(source[start..<end])), end + 1)
        }

        while index < source.count {
            let scalar = source[index]
            if scalar == "\\", index + 1 < source.count {
                if classStart == nil, source[index + 1] == "Z" {
                    emit("\\z")
                } else {
                    output.append(scalar)
                    output.append(source[index + 1])
                }
                index += 2
                continue
            }
            if let start = classStart {
                switch scalar {
                case "]" where index > start: classStart = nil
                case "[": emit("\\")
                case "&" where starts("&&"), "-" where starts("--"): emit("\\")
                default: break
                }
                output.append(scalar)
                index += 1
                continue
            }
            switch scalar {
            case "[":
                output.append(scalar)
                index += 1
                if index < source.count, source[index] == "^" {
                    output.append("^")
                    index += 1
                }
                classStart = index
                // A `]` right after the opening bracket is a literal in Python.
                if index < source.count, source[index] == "]" {
                    emit("\\]")
                    index += 1
                }
                continue
            case "(" where starts("(?P<") || (starts("(?<") && !starts("(?<=") && !starts("(?<!")):
                let nameStart = index + (starts("(?P<") ? 4 : 3)
                if let (groupName, next) = name(from: nameStart, until: ">") {
                    emit("(?<g\(names.count)>")
                    names.append(groupName)
                    index = next
                    continue
                }
            case "(" where starts("(?P="):
                if let (groupName, next) = name(from: index + 4, until: ")"),
                    let position = names.firstIndex(of: groupName)
                {
                    emit("\\k<g\(position)>")
                    index = next
                    continue
                }
            case "{" where isQuantifier(source, at: index):
                // Copied whole, so its `}` is not taken for a literal one below.
                let close = source[index...].firstIndex(of: "}")!
                output.append(contentsOf: source[index...close])
                index = close + 1
                continue
            case "{", "}":
                emit("\\")  // literals in Python, syntax errors in ICU
            default:
                break
            }
            output.append(scalar)
            index += 1
        }
        return (String(output), names)
    }

    /// Whether the `{` at `index` opens a `{m}`, `{m,}`, `{,n}` or `{m,n}` quantifier.
    private static func isQuantifier(_ source: [Unicode.Scalar], at index: Int) -> Bool {
        guard let close = source[(index + 1)...].firstIndex(of: "}") else { return false }
        let body = source[(index + 1)..<close]
        return body.contains { ("0"..."9").contains($0) } && body.allSatisfy { ("0"..."9").contains($0) || $0 == "," }
            && body.count(where: { $0 == "," }) <= 1
    }

    // MARK: Required literal

    /// The longest run of literal characters every match contains, read from the top level of
    /// the pattern, and whether it starts every match. Anything not understood ends a run, so
    /// the answer is conservative: a shorter or empty literal, never a wrong one.
    private static func requiredLiteral(_ source: [Unicode.Scalar]) -> ([UInt16], Bool) {
        var best: [Unicode.Scalar] = []
        var bestIsPrefix = false
        var run: [Unicode.Scalar] = []
        var runIsPrefix = true
        var consumed = false
        var index = 0

        func endRun() {
            if run.count > best.count {
                best = run
                bestIsPrefix = runIsPrefix
            }
            if !run.isEmpty { consumed = true }
            run = []
            runIsPrefix = !consumed
        }
        func consumingAtom() {
            endRun()
            consumed = true
            runIsPrefix = false
        }

        while index < source.count {
            let scalar = source[index]
            var literal: Unicode.Scalar?
            switch scalar {
            case "|":
                return ([], false)  // a top-level alternative: no literal is required
            case "(":
                // Inline flags (`(?i)`, `(?x)`) change what literals mean: give up.
                if index + 2 < source.count, source[index + 1] == "?", source[index + 2].properties.isAlphabetic,
                    source[index + 2] != "P"
                {
                    return ([], false)
                }
                guard let end = closingParenthesis(source, from: index) else { return ([], false) }
                index = end + 1
                consumingAtom()
                skipQuantifier(source, &index)
                continue
            case "[":
                guard let end = closingBracket(source, from: index) else { return ([], false) }
                index = end + 1
                consumingAtom()
                skipQuantifier(source, &index)
                continue
            case "^", "$":
                endRun()
                index += 1
                continue
            case ".":
                index += 1
                consumingAtom()
                skipQuantifier(source, &index)
                continue
            case "\\":
                guard index + 1 < source.count else { return ([], false) }
                let escaped = source[index + 1]
                index += 2
                switch escaped {
                case "n": literal = "\n"
                case "t": literal = "\t"
                case "r": literal = "\r"
                case "A", "Z", "b", "B", "G":
                    endRun()
                    continue
                case _ where escaped.properties.isAlphabetic || ("0"..."9").contains(escaped):
                    consumingAtom()
                    skipQuantifier(source, &index)
                    continue
                default: literal = escaped
                }
            case "*", "+", "?", "{", ")":
                return ([], false)  // malformed at the top level
            default:
                literal = scalar
                index += 1
            }
            guard let literal else { continue }
            // A quantifier makes the character optional or repeated: it cannot extend a run.
            if index < source.count, ["?", "*", "{", "+"].contains(source[index]) {
                let required = source[index] == "+"
                if required { run.append(literal) }
                endRun()
                consumed = true
                runIsPrefix = false
                skipQuantifier(source, &index)
                continue
            }
            run.append(literal)
        }
        endRun()
        return (Array(String(String.UnicodeScalarView(best)).utf16), bestIsPrefix)
    }

    private static func skipQuantifier(_ source: [Unicode.Scalar], _ index: inout Int) {
        guard index < source.count else { return }
        switch source[index] {
        case "?", "*", "+": index += 1
        case "{" where isQuantifier(source, at: index): index = source[index...].firstIndex(of: "}")! + 1
        default: return
        }
        if index < source.count, source[index] == "?" || source[index] == "+" { index += 1 }
    }

    private static func closingParenthesis(_ source: [Unicode.Scalar], from start: Int) -> Int? {
        var depth = 0
        var index = start
        while index < source.count {
            switch source[index] {
            case "\\": index += 1
            case "[":
                guard let end = closingBracket(source, from: index) else { return nil }
                index = end
            case "(": depth += 1
            case ")":
                depth -= 1
                if depth == 0 { return index }
            default: break
            }
            index += 1
        }
        return nil
    }

    private static func closingBracket(_ source: [Unicode.Scalar], from start: Int) -> Int? {
        var index = start + 1
        if index < source.count, source[index] == "^" { index += 1 }
        if index < source.count, source[index] == "]" { index += 1 }
        while index < source.count {
            switch source[index] {
            case "\\": index += 1
            case "]": return index
            default: break
            }
            index += 1
        }
        return nil
    }
}

extension UnsafeBufferPointer<UInt16> {
    /// The first occurrence of `needle` at or after `start`.
    func firstRange(of needle: [UInt16], from start: Int) -> Range<Int>? {
        guard let first = needle.first, needle.count <= count else { return nil }
        var position = start
        while position + needle.count <= count {
            if self[position] == first, needle.indices.dropFirst().allSatisfy({ self[position + $0] == needle[$0] }) {
                return position..<position + needle.count
            }
            position += 1
        }
        return nil
    }
}
