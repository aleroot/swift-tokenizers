// Splits input text around added / special tokens before normalization and pre-tokenization,
// matching contents leftmost-longest before applying single-word and whitespace rules,
// as in Hugging Face added_vocabulary.rs. Rejected single-word matches still consume their
// content in the matcher; whitespace stripping does not hide later content matches.

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
    /// When every token starts with the same byte, `memchr` skips straight to candidates.
    private let singleFirstByte: UInt8?
    private let hasSingleWord: Bool

    init?(tokens: [Token]) {
        guard !tokens.isEmpty else { return nil }
        self.tokens = tokens

        var utf8: [UInt8] = []
        var offsets: [UInt32] = [0]
        var firstBytes = [Bool](repeating: false, count: 256)
        for token in tokens {
            utf8.append(contentsOf: token.content.utf8)
            offsets.append(UInt32(utf8.count))
            if let firstByte = token.content.utf8.first { firstBytes[Int(firstByte)] = true }
        }
        tokenFirstBytes = firstBytes
        trie = utf8.withUnsafeBufferPointer { utf8 in
            offsets.withUnsafeBufferPointer { offsets in
                DoubleArrayTrie(utf8: utf8, offsets: offsets, count: tokens.count)
            }
        }
        hasSingleWord = tokens.contains(where: \.singleWord)
        let candidates = firstBytes.enumerated().filter(\.element).map { UInt8($0.offset) }
        singleFirstByte = candidates.count == 1 ? candidates[0] : nil
    }

    /// Byte-offset variant of ``split(_:)`` operating on well-formed UTF-8.
    func split(bytes: UnsafeBufferPointer<UInt8>, into sections: inout [ByteSection]) {
        scan(bytes: bytes, onText: { sections.append(.text($0)) },
             onToken: { id, _ in sections.append(.token(id: id)) })
    }

    /// Shared matcher; only the opt-in caller retains matched source ranges.
    func scan(bytes: UnsafeBufferPointer<UInt8>, onText: (Range<Int>) -> Void,
              onToken: (Int, Range<Int>) -> Void) {
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
            guard let match = longestMatch(bytes, from: candidate) else {
                i = candidate + UTF8Cursor.width(bytes[candidate])
                continue
            }
            // Content matching advances independently of acceptance and whitespace stripping.
            i = match.contentEnd
            let token = tokens[Int(match.token)]
            if hasSingleWord, token.singleWord,
                !isolatedWord(bytes, start: candidate, end: match.contentEnd)
            {
                continue
            }
            var start = candidate
            if token.lstrip {
                while start > sectionStart {
                    var previous = start - 1
                    while previous > sectionStart, bytes[previous] & 0xC0 == 0x80 { previous -= 1 }
                    guard whitespace(bytes, at: previous).0 else { break }
                    start = previous
                }
                start = max(start, sectionStart)
            }
            if sectionStart < start { onText(sectionStart..<start) }
            var next = match.contentEnd
            if token.rstrip {
                while next < end {
                    let (ws, width) = whitespace(bytes, at: next)
                    guard ws else { break }
                    next += width
                }
            }
            onToken(token.id, min(start, next)..<next)
            sectionStart = next
        }

        if sectionStart < end {
            onText(sectionStart..<end)
        }
    }

    /// Select content before checking flags: a rejected longest match must not fall back
    /// to a shorter or overlapping added token.
    @inline(__always)
    private func longestMatch(
        _ bytes: UnsafeBufferPointer<UInt8>, from start: Int
    ) -> (token: Int32, contentEnd: Int)? {
        var best: (token: Int32, contentEnd: Int)?
        trie.forEachPrefix(of: bytes, from: start) { length, index in
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
