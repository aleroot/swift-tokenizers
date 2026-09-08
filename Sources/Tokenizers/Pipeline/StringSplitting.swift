// Generic split helpers used by the configurable (non-fast-path) pre-tokenizers and
// normalizers. These mirror the `tokenizers` library's `SplitDelimiterBehavior`.

import Foundation

enum StringSplitPattern {
    case regexp(regexp: NSRegularExpression)
    case string(pattern: String)

    func split(_ text: String, invert: Bool = true) -> [String] {
        switch self {
        case let .regexp(regexp):
            return text.split(matching: regexp, includeSeparators: true)
        case let .string(substring):
            return text.split(by: substring, options: [], includeSeparators: !invert)
        }
    }

    static func from(config: Config) throws -> StringSplitPattern? {
        if let pattern = config.pattern.String.string() {
            return .string(pattern: pattern)
        }
        if let pattern = config.pattern.Regex.string() {
            return .regexp(regexp: try compileRegex(pattern, component: "Split pre-tokenizer"))
        }
        return nil
    }

    /// The raw regex source, if this is a regex pattern.
    var regexSource: String? {
        if case let .regexp(regexp) = self { return regexp.pattern }
        return nil
    }
}

/// Compiles a regular expression from a `tokenizer.json` field, reporting failures as a
/// configuration error rather than crashing.
func compileRegex(_ pattern: String, component: String) throws -> NSRegularExpression {
    do {
        return try NSRegularExpression(pattern: pattern, options: [])
    } catch {
        throw TokenizerError.invalidConfiguration(
            "\(component): invalid regular expression \(pattern.debugDescription)")
    }
}

enum SplitDelimiterBehavior {
    case removed
    case isolated
    case mergedWithPrevious
    case mergedWithNext
}

extension String {
    /// All non-overlapping ranges of `string` (interpreted per `options`).
    func ranges(of string: String, options: CompareOptions = .regularExpression) -> [Range<Index>] {
        var result: [Range<Index>] = []
        var start = startIndex
        while let range = range(of: string, options: options, range: start..<endIndex) {
            result.append(range)
            start =
                range.lowerBound < range.upperBound
                ? range.upperBound
                : index(range.lowerBound, offsetBy: 1, limitedBy: endIndex) ?? endIndex
        }
        return result
    }

    func split(
        by string: String,
        options: CompareOptions = .regularExpression,
        includeSeparators: Bool = false,
        omittingEmptySubsequences: Bool = true
    ) -> [String] {
        var result: [String] = []
        var start = startIndex
        while let range = range(of: string, options: options, range: start..<endIndex) {
            if omittingEmptySubsequences, start < range.lowerBound {
                result.append(String(self[start..<range.lowerBound]))
            }
            if includeSeparators {
                result.append(String(self[range]))
            }
            if range.upperBound == start {
                // Zero-width match; step forward to avoid an infinite loop.
                guard range.upperBound < endIndex else { break }
                start = index(after: range.upperBound)
            } else {
                start = range.upperBound
            }
        }
        if omittingEmptySubsequences, start < endIndex {
            result.append(String(self[start...]))
        }
        return result
    }

    /// Splits on every match of a compiled regex, keeping the matches as separators.
    func split(matching regex: NSRegularExpression, includeSeparators: Bool) -> [String] {
        let ns = self as NSString
        let full = NSRange(location: 0, length: ns.length)
        var result: [String] = []
        var start = 0
        regex.enumerateMatches(in: self, options: [], range: full) { match, _, _ in
            guard let match else { return }
            let r = match.range
            if r.location > start {
                result.append(ns.substring(with: NSRange(location: start, length: r.location - start)))
            }
            if includeSeparators, r.length > 0 {
                result.append(ns.substring(with: r))
            }
            start = r.location + r.length
        }
        if start < ns.length {
            result.append(ns.substring(from: start))
        }
        return result
    }

    /// Splits on a capturing regex, emitting the innermost participating capture group as
    /// the separator (used for added-token splitting).
    func split(by captureRegex: NSRegularExpression) -> [String] {
        let selfRange = NSRange(startIndex..<endIndex, in: self)
        let matches = captureRegex.matches(in: self, options: [], range: selfRange)
        if matches.isEmpty { return [self] }

        var result: [String] = []
        var start = startIndex
        for match in matches {
            guard let matchRange = Range(match.range, in: self) else { continue }
            if start < matchRange.lowerBound {
                result.append(String(self[start..<matchRange.lowerBound]))
            }
            start = matchRange.upperBound
            for r in (0..<match.numberOfRanges).reversed() {
                if let sepRange = Range(match.range(at: r), in: self) {
                    result.append(String(self[sepRange]))
                    break
                }
            }
        }
        if start < endIndex {
            result.append(String(self[start...]))
        }
        return result
    }

    func split(
        by string: String, options: CompareOptions = .regularExpression, behavior: SplitDelimiterBehavior
    ) -> [String] {
        func mergedWithNext(ranges: [Range<String.Index>]) -> [Range<String.Index>] {
            var merged: [Range<String.Index>] = []
            var currentStart = startIndex
            for range in ranges {
                if range.lowerBound == startIndex { continue }
                merged.append(currentStart..<range.lowerBound)
                currentStart = range.lowerBound
            }
            if currentStart < endIndex {
                merged.append(currentStart..<endIndex)
            }
            return merged
        }

        func mergedWithPrevious(ranges: [Range<String.Index>]) -> [Range<String.Index>] {
            var merged: [Range<String.Index>] = []
            var currentStart = startIndex
            for range in ranges {
                merged.append(currentStart..<range.upperBound)
                currentStart = range.upperBound
            }
            if currentStart < endIndex {
                merged.append(currentStart..<endIndex)
            }
            return merged
        }

        switch behavior {
        case .removed:
            return split(by: string, options: options, includeSeparators: false)
        case .isolated:
            return split(by: string, options: options, includeSeparators: true)
        case .mergedWithNext:
            return mergedWithNext(ranges: ranges(of: string, options: options)).map { String(self[$0]) }
        case .mergedWithPrevious:
            return mergedWithPrevious(ranges: ranges(of: string, options: options)).map { String(self[$0]) }
        }
    }
}

extension StringProtocol {
    /// Byte-wise prefix test. Unlike `hasPrefix`, this does not consider grapheme clusters, so
    /// `"##\u{0301}".hasBytePrefix("##")` is `true` (matching Rust's `starts_with`).
    @inline(__always)
    func hasBytePrefix(_ prefix: String) -> Bool {
        utf8.starts(with: prefix.utf8)
    }

    /// Literal, byte-wise replacement of every occurrence of `target`. Foundation's
    /// `replacingOccurrences` respects grapheme clusters and canonical equivalence, so it will
    /// not match `" "` in `" \u{0308}"`; `tokenizers` replaces on raw code points.
    func replacingBytes(of target: String, with replacement: String) -> String {
        let pattern = Array(target.utf8)
        guard !pattern.isEmpty else { return String(self) }
        let repl = Array(replacement.utf8)
        var copy = Substring(self)
        let result: String? = copy.withUTF8 { bytes -> String? in
            let n = bytes.count
            let m = pattern.count
            let first = pattern[0]
            // Find the first occurrence with memchr; most inputs of most normalizers have none.
            guard let base = bytes.baseAddress else { return nil }
            @inline(__always)
            func matches(at p: Int) -> Bool {
                guard p + m <= n else { return false }
                for k in 1..<m where base[p + k] != pattern[k] { return false }
                return true
            }
            @inline(__always)
            func nextCandidate(from p: Int) -> Int? {
                guard p < n, let hit = memchr(base + p, Int32(first), n - p) else { return nil }
                return UnsafeRawPointer(hit) - UnsafeRawPointer(base)
            }
            var i = 0
            var start: Int? = nil
            while let c = nextCandidate(from: i) {
                if matches(at: c) {
                    start = c
                    break
                }
                i = c + 1
            }
            guard let firstMatch = start else { return nil }
            // Count the matches so the result can be written straight into its final storage.
            var count = 0
            i = firstMatch
            while let c = nextCandidate(from: i) {
                if matches(at: c) {
                    count += 1
                    i = c + m
                } else {
                    i = c + 1
                }
            }
            let capacity = n + count * (repl.count - m)
            return String(unsafeUninitializedCapacity: capacity) { out -> Int in
                guard let dst = out.baseAddress else { return 0 }
                var w = 0
                @inline(__always)
                func copy(_ lo: Int, _ hi: Int) {
                    if hi > lo {
                        UnsafeMutableRawPointer(dst + w).copyMemory(from: base + lo, byteCount: hi - lo)
                        w += hi - lo
                    }
                }
                copy(0, firstMatch)
                var r = firstMatch
                while r < n {
                    if base[r] == first, matches(at: r) {
                        for b in repl {
                            dst[w] = b
                            w += 1
                        }
                        r += m
                    } else if let c = nextCandidate(from: r + 1) {
                        copy(r, c)
                        r = c
                    } else {
                        copy(r, n)
                        r = n
                    }
                }
                return w
            }
        }
        return result ?? String(self)
    }

    /// Removes a byte-wise prefix if present.
    func droppingBytePrefix(_ prefix: String) -> Substring {
        guard hasBytePrefix(prefix) else { return Substring(self) }
        let index = utf8.index(utf8.startIndex, offsetBy: prefix.utf8.count)
        return Substring(self[index...])
    }
}

extension Substring {
    /// Splits a substring at every occurrence of `separator` (a literal string), merging each
    /// separator with the text that follows it. Equivalent to
    /// `split(by:behavior: .mergedWithNext)` but works on UTF-8 and avoids regex machinery.
    func splitMergedWithNext(separator: String) -> [Substring] {
        let sep = Array(separator.utf8)
        guard !sep.isEmpty else { return [self] }
        var pieces: [Substring] = []
        let utf8 = self.utf8
        var pieceStart = utf8.startIndex
        var i = utf8.startIndex
        let end = utf8.endIndex
        let first = sep[0]
        while i < end {
            if utf8[i] == first, matches(sep, at: i) {
                // A separator at the very start never yields an empty leading piece.
                if i != pieceStart {
                    pieces.append(self[pieceStart..<i])
                    pieceStart = i
                }
                i = utf8.index(i, offsetBy: sep.count)
            } else {
                utf8.formIndex(after: &i)
            }
        }
        if pieceStart < end {
            pieces.append(self[pieceStart..<end])
        }
        return pieces
    }

    private func matches(_ bytes: [UInt8], at start: String.UTF8View.Index) -> Bool {
        let utf8 = self.utf8
        var i = start
        for b in bytes {
            guard i < utf8.endIndex, utf8[i] == b else { return false }
            utf8.formIndex(after: &i)
        }
        return true
    }
}
