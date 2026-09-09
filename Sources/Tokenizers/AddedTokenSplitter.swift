// Splits input text around added / special tokens before normalization and pre-tokenization,
// reproducing the reference `\s*(<tok>)\s*|…` alternation semantics (leftmost match, longest
// content wins, `lstrip`/`rstrip` swallow whitespace) with a double-array trie and one
// allocation-free pass.

import Foundation

final class AddedTokenSplitter: Sendable {
    struct Token: Sendable {
        let content: String
        let id: Int
        let lstrip: Bool
        let rstrip: Bool
        let scalarCount: Int
        var singleWord: Bool = false
    }

    /// A section of the input: plain text, or an added token.
    enum Section {
        case text(Substring)
        case token(Substring, id: Int)
    }

    /// A section expressed in byte offsets of the scanned buffer.
    enum ByteSection {
        case text(Range<Int>)
        case token(id: Int)
    }

    let tokens: [Token]
    /// Token contents keyed by bytes; value = index into `tokens`.
    private let trie: DoubleArrayTrie
    /// Bytes that start some token's content.
    private let tokenFirstBytes: [Bool]
    /// Some token's content begins with whitespace (so whitespace runs cannot be skipped wholesale).
    private let tokenStartsWithWhitespace: Bool
    /// When every token starts with the same byte, `memchr` skips straight to candidates.
    private let singleFirstByte: UInt8?
    private let hasLstrip: Bool
    /// Some `lstrip` token's content itself begins with whitespace, so a match may start
    /// inside a whitespace run rather than right after it.
    private let lstripStartsWithWhitespace: Bool
    private let hasSingleWord: Bool

    init?(tokens: [Token]) {
        guard !tokens.isEmpty else { return nil }
        self.tokens = tokens

        var utf8: [UInt8] = []
        var offsets: [UInt32] = [0]
        var firstBytes = [Bool](repeating: false, count: 256)
        var hasLstrip = false
        var lstripStartsWithWhitespace = false
        var tokenStartsWithWhitespace = false
        for token in tokens {
            utf8.append(contentsOf: token.content.utf8)
            offsets.append(UInt32(utf8.count))
            guard let firstByte = token.content.utf8.first, let first = token.content.unicodeScalars.first else {
                continue
            }
            firstBytes[Int(firstByte)] = true
            let startsWithWhitespace = ScalarClassifier.flags(value: first.value) & ScalarFlags.whitespace != 0
            if startsWithWhitespace { tokenStartsWithWhitespace = true }
            if token.lstrip {
                hasLstrip = true
                if startsWithWhitespace { lstripStartsWithWhitespace = true }
            }
        }
        tokenFirstBytes = firstBytes
        self.tokenStartsWithWhitespace = tokenStartsWithWhitespace
        trie = utf8.withUnsafeBufferPointer { utf8 in
            offsets.withUnsafeBufferPointer { offsets in
                DoubleArrayTrie(utf8: utf8, offsets: offsets, count: tokens.count)
            }
        }
        self.hasLstrip = hasLstrip
        self.lstripStartsWithWhitespace = lstripStartsWithWhitespace
        hasSingleWord = tokens.contains(where: \.singleWord)
        let candidates = firstBytes.enumerated().filter(\.element).map { UInt8($0.offset) }
        singleFirstByte = candidates.count == 1 ? candidates[0] : nil
    }

    /// Byte-offset variant of ``split(_:)`` operating on well-formed UTF-8.
    func split(bytes: UnsafeBufferPointer<UInt8>, into sections: inout [ByteSection]) {
        let end = bytes.count
        var sectionStart = 0
        var i = 0

        while i < end {
            // Jump to the next byte that starts some token's content …
            var candidate: Int
            if let single = singleFirstByte {
                candidate = ByteKernels.firstIndex(of: single, in: bytes, from: i)
            } else {
                candidate = i
                while candidate < end, !tokenFirstBytes[Int(bytes[candidate])] {
                    candidate += UTF8Cursor.width(bytes[candidate])
                }
            }
            guard candidate < end else { break }
            // … then, for `\s*(<tok>)`, back up to the start of the whitespace run before it:
            // whitespace matters only when a token follows it, so every other run is skipped.
            if hasLstrip {
                var runStart = candidate
                while runStart > i {
                    var previous = runStart - 1
                    while previous > i, bytes[previous] & 0xC0 == 0x80 { previous -= 1 }
                    guard whitespace(bytes, at: previous).0 else { break }
                    runStart = previous
                }
                candidate = runStart
            }
            i = candidate

            var best = tokenFirstBytes[Int(bytes[i])] ? longestMatch(bytes, from: i, requireLstrip: false) : nil
            var skipTo = -1

            if hasLstrip {
                let (isWhitespace, width) = whitespace(bytes, at: i)
                if isWhitespace {
                    // `\s*(<tok>)`: the token may start right after the whitespace run — or, when a
                    // token's own content starts with whitespace, anywhere inside it. Later starts
                    // are checked first, so ties keep the leftmost regex alternative's behaviour.
                    var runEnd = i + width
                    while runEnd < end {
                        let (ws, w) = whitespace(bytes, at: runEnd)
                        guard ws else { break }
                        runEnd += w
                    }
                    var start = runEnd
                    while start > i {
                        if start < end, tokenFirstBytes[Int(bytes[start])],
                            let match = longestMatch(bytes, from: start, requireLstrip: true),
                            best == nil || tokens[Int(match.token)].scalarCount > tokens[Int(best!.token)].scalarCount
                        {
                            best = match
                        }
                        guard lstripStartsWithWhitespace else { break }
                        start -= 1
                        while start > i, bytes[start] & 0xC0 == 0x80 { start -= 1 }
                        if start == i { break }
                    }
                    // Nothing can start inside the run unless a token itself begins with whitespace.
                    if !tokenStartsWithWhitespace { skipTo = runEnd }
                }
            }

            guard let match = best else {
                i = skipTo > i ? skipTo : i + UTF8Cursor.width(bytes[i])
                continue
            }

            if sectionStart < i {
                sections.append(.text(sectionStart..<i))
            }
            let token = tokens[Int(match.token)]
            sections.append(.token(id: token.id))

            var next = match.contentEnd
            if token.rstrip {
                while next < end {
                    let (ws, w) = whitespace(bytes, at: next)
                    guard ws else { break }
                    next += w
                }
            }
            sectionStart = next
            i = next
        }

        if sectionStart < end {
            sections.append(.text(sectionStart..<end))
        }
    }

    /// Longest token whose content starts at `start` (and satisfies `lstrip` / `single_word`
    /// constraints), with the offset just past it.
    @inline(__always)
    private func longestMatch(
        _ bytes: UnsafeBufferPointer<UInt8>, from start: Int, requireLstrip: Bool
    ) -> (token: Int32, contentEnd: Int)? {
        var best: (token: Int32, contentEnd: Int)?
        trie.forEachPrefix(of: bytes, from: start) { length, index in
            let token = tokens[Int(index)]
            if requireLstrip, !token.lstrip { return }
            if hasSingleWord, token.singleWord, !isolatedWord(bytes, start: start, end: start + length) { return }
            best = (index, start + length)
        }
        return best
    }

    /// `single_word`: the match must not be adjacent to word characters.
    private func isolatedWord(_ bytes: UnsafeBufferPointer<UInt8>, start: Int, end: Int) -> Bool {
        if start > 0 {
            var previous = start - 1
            while previous > 0, bytes[previous] & 0xC0 == 0x80 { previous -= 1 }
            if Self.isWord(UTF8Cursor.decode(bytes, at: previous).0) { return false }
        }
        return end == bytes.count || !Self.isWord(UTF8Cursor.decode(bytes, at: end).0)
    }

    @inline(__always)
    private func whitespace(_ bytes: UnsafeBufferPointer<UInt8>, at p: Int) -> (Bool, Int) {
        let b0 = bytes[p]
        if b0 < 0x80 { return (b0 == 0x20 || (b0 >= 0x09 && b0 <= 0x0D), 1) }
        let (v, w) = UTF8Cursor.decode(bytes, at: p)
        return (ScalarClassifier.flags(value: v) & ScalarFlags.whitespace != 0, w)
    }

    /// Rust's Unicode word boundary: alphabetic, marks, decimal numbers, connector
    /// punctuation, and join controls. WordPiece punctuation rules are different.
    private static func isWord(_ value: UInt32) -> Bool {
        WhitespacePreTokenizer.isWord(value)
    }

    /// String variant: sections as substrings of `text`.
    func split(_ text: String) -> [Section] {
        var copy = text
        return copy.withUTF8 { bytes in
            var sections: [ByteSection] = []
            split(bytes: bytes, into: &sections)
            return sections.map { section in
                switch section {
                case .text(let range):
                    let start = text.utf8.index(text.utf8.startIndex, offsetBy: range.lowerBound)
                    let end = text.utf8.index(start, offsetBy: range.count)
                    return .text(text[start..<end])
                case .token(let id):
                    let token = tokens.first { $0.id == id }!
                    return .token(Substring(token.content), id: id)
                }
            }
        }
    }
}
