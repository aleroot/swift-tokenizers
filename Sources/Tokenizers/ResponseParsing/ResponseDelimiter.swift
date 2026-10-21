import Foundation

/// Where a region of a response template starts or ends: one or more literal strings, or a
/// regular expression.
struct ResponseDelimiter: Sendable {
    enum Matcher: Sendable {
        /// Literals, longest first, so the longest one present wins as in a Python alternation.
        case literals([[UInt16]])
        case regex(PythonRegex)
    }

    let matcher: Matcher
    /// Whether one literal is a strict prefix of another, so a match at the end of the text
    /// could still grow into a longer one.
    let literalCanExtend: Bool

    var groupNames: [String] {
        if case let .regex(regex) = matcher { return regex.groupNames }
        return []
    }

    /// Compiles `spec[literalKey]` (a string or a list of strings) or `spec[patternKey]`.
    static func compile(
        _ spec: ResponseObject, literal literalKey: String, pattern patternKey: String, scope: String
    )
        throws(ResponseParsingError) -> ResponseDelimiter?
    {
        let literal = spec[literalKey]
        let pattern = spec[patternKey]
        if literal != nil, pattern != nil {
            throw .invalidTemplate("\(scope): cannot specify both '\(literalKey)' and '\(patternKey)'")
        }
        if let literal {
            var literals: [String]
            switch literal {
            case let .string(value):
                literals = [value]
            case let .array(values):
                guard !values.isEmpty else {
                    throw .invalidTemplate("\(scope): '\(literalKey)' list must contain at least one literal")
                }
                literals = []
                for value in values {
                    guard let string = value.stringValue else {
                        throw .invalidTemplate("\(scope): '\(literalKey)' list must contain only strings")
                    }
                    if !literals.contains(string) { literals.append(string) }
                }
            default:
                throw .invalidTemplate(
                    "\(scope): '\(literalKey)' must be a string or list of strings, got \(literal.pythonTypeName)")
            }
            if literals.contains("") {
                throw .invalidTemplate("\(scope): '\(literalKey)' literals cannot be empty strings")
            }
            let units = literals.map { Array($0.utf16) }
            let canExtend = units.contains { a in units.contains { b in a != b && a.starts(with: b) } }
            // Stable longest-first order, as `sorted(literals, key=len, reverse=True)`; lengths are
            // compared in code points.
            let ordered = zip(literals, units).enumerated().sorted { lhs, rhs in
                let (l, r) = (lhs.element.0.unicodeScalars.count, rhs.element.0.unicodeScalars.count)
                return l != r ? l > r : lhs.offset < rhs.offset
            }.map(\.element.1)
            return ResponseDelimiter(matcher: .literals(ordered), literalCanExtend: canExtend)
        }
        if let pattern {
            guard let source = pattern.stringValue else {
                throw .invalidTemplate("\(scope): invalid \(patternKey) regex: expected a string")
            }
            do {
                return ResponseDelimiter(matcher: .regex(try PythonRegex(source)), literalCanExtend: false)
            } catch {
                throw .invalidTemplate("\(scope): invalid \(patternKey) regex: \(source)")
            }
        }
        return nil
    }

    /// A match found by ``search(_:from:through:partial:text:)``.
    struct Match {
        var range: Range<Int>
        var isPartial = false
        var captures: [(name: String, value: String)] = []
    }

    var isLiteral: Bool {
        if case .literals = matcher { return true }
        return false
    }

    /// `pattern.search(text, start, partial=partial)` with the semantics of Python's `regex`,
    /// limited to matches that start at or before `limit`: the leftmost position where an attempt
    /// matches or, when `partial`, runs out of text. A partial match ends at the end of the text.
    func search(
        _ units: UnsafeBufferPointer<UInt16>, from start: Int, through limit: Int, partial: Bool,
        text: () -> NSString
    ) -> Match? {
        let last = Swift.min(limit, units.count - 1)
        guard start <= last else { return nil }
        switch matcher {
        case let .literals(literals):
            return Self.searchLiterals(literals, in: units, positions: start...last, partial: partial)
        case let .regex(regex):
            if !partial, last == units.count - 1 {
                // Unbounded: ICU's own search loop is the fastest way to the leftmost match.
                return regex.firstMatch(in: text(), units: units, from: start).map {
                    Match(range: $0.range, captures: $0.captures)
                }
            }
            return Self.searchAttempts(regex, in: units, text: text(), positions: start...last, partial: partial)
        }
    }

    /// Whether a complete match could still change as more text arrives (`_can_grow`).
    func canGrow(_ range: Range<Int>, count: Int) -> Bool {
        guard range.upperBound == count else { return false }
        if range.isEmpty { return true }
        if case .literals = matcher { return literalCanExtend }
        return true
    }

    /// The end of the last non-overlapping match, as the last `finditer` result.
    func lastMatchEnd(in units: UnsafeBufferPointer<UInt16>) -> Int? {
        switch matcher {
        case let .literals(literals):
            var position = 0
            var last: Int?
            while position < units.count,
                let match = Self.searchLiterals(
                    literals, in: units, positions: position...(units.count - 1), partial: false)
            {
                last = match.range.upperBound
                position = match.range.upperBound
            }
            return last
        case let .regex(regex):
            return regex.matches(in: UTF16Text.string(units)).last?.range.upperBound
        }
    }

    /// The alternation `lit1|lit2|…`, longest first: at each position the first literal present
    /// matches; otherwise a literal the text ends inside makes the match partial.
    private static func searchLiterals(
        _ literals: [[UInt16]], in units: UnsafeBufferPointer<UInt16>, positions: ClosedRange<Int>, partial: Bool
    ) -> Match? {
        for position in positions {
            let unit = units[position]
            var endsInside = false
            for literal in literals where literal[0] == unit {
                let available = min(literal.count, units.count - position)
                var offset = 1
                while offset < available, units[position + offset] == literal[offset] { offset += 1 }
                if offset == literal.count { return Match(range: position..<position + offset) }
                if offset == available { endsInside = true }
            }
            if partial, endsInside { return Match(range: position..<units.count, isPartial: true) }
        }
        return nil
    }

    /// One anchored attempt per position that can start a match, as Python's search makes them.
    /// When the pattern starts with a literal, only positions holding it are tried, and text
    /// that ends inside it is a partial match without asking ICU.
    private static func searchAttempts(
        _ regex: PythonRegex, in units: UnsafeBufferPointer<UInt16>, text: NSString, positions: ClosedRange<Int>,
        partial: Bool
    ) -> Match? {
        let literal = regex.startsWithRequiredLiteral ? regex.requiredLiteral : []
        var position = positions.lowerBound
        while position <= positions.upperBound {
            defer { position += 1 }
            if let first = literal.first {
                guard units[position] == first else { continue }
                let available = min(literal.count, units.count - position)
                var offset = 1
                while offset < available, units[position + offset] == literal[offset] { offset += 1 }
                guard offset == available else { continue }
                if available < literal.count {
                    return partial ? Match(range: position..<units.count, isPartial: true) : nil
                }
            } else if UTF16.isTrailSurrogate(units[position]) {
                continue  // matches start on code points, never inside a surrogate pair
            }
            switch regex.attempt(in: text, at: position) {
            case let .match(match): return Match(range: match.range, captures: match.captures)
            case .reachedEnd where partial: return Match(range: position..<units.count, isPartial: true)
            default: continue
            }
        }
        return nil
    }
}
