// Splits input text around added / special tokens before normalization and pre-tokenization.
//
// The reference implementation compiles every added token into one alternation regex
// (`\s*(<tok1>)\s*|(<tok2>)|…`, longest content first) and splits on it. This type
// reproduces those semantics — leftmost match, longest content wins, `lstrip`/`rstrip`
// swallow adjacent whitespace, only the content is emitted — with a byte trie and a single
// pass over the input.

import Foundation

struct AddedTokenSplitter: Sendable {
    struct Token: Sendable {
        let content: String
        let id: Int
        let lstrip: Bool
        let rstrip: Bool
        let scalarCount: Int
    }

    /// A section of the input: plain text, or an added token.
    enum Section {
        case text(Substring)
        case token(Substring, id: Int)
    }

    let tokens: [Token]
    private let trie: ByteTrie
    /// Bytes that can start a token (or whitespace, when any token has `lstrip`).
    private let candidateFirstBytes: [Bool]
    /// When every token starts with the same byte, `memchr` skips straight to candidates.
    private let singleCandidateByte: UInt8?
    private let hasLstrip: Bool

    init?(tokens: [Token]) {
        guard !tokens.isEmpty else { return nil }
        self.tokens = tokens
        var trie = ByteTrie()
        var firstBytes = [Bool](repeating: false, count: 256)
        var hasLstrip = false
        for (index, token) in tokens.enumerated() {
            guard let firstByte = token.content.utf8.first else { continue }
            trie.insert(Array(token.content.utf8), index: Int32(index))
            firstBytes[Int(firstByte)] = true
            if token.lstrip { hasLstrip = true }
        }
        self.trie = trie
        self.hasLstrip = hasLstrip
        if hasLstrip {
            // Whitespace may precede an lstrip token; ASCII whitespace plus lead bytes of
            // non-ASCII whitespace (U+0085, U+00A0, U+1680, U+2000–U+200A, U+2028/9, U+202F, U+205F, U+3000).
            for b in [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0xC2, 0xE1, 0xE2, 0xE3] { firstBytes[b] = true }
        }
        candidateFirstBytes = firstBytes
        let candidates = firstBytes.enumerated().filter(\.element).map { UInt8($0.offset) }
        singleCandidateByte = candidates.count == 1 ? candidates[0] : nil
    }

    /// A section expressed in byte offsets of the scanned buffer.
    enum ByteSection {
        case text(Range<Int>)
        case token(id: Int)
    }

    /// Byte-offset variant of ``split(_:)`` operating on well-formed UTF-8.
    func split(bytes: UnsafeBufferPointer<UInt8>, into sections: inout [ByteSection]) {
        let end = bytes.count
        var sectionStart = 0
        var i = 0

        @inline(__always)
        func isWhitespace(_ p: Int) -> (Bool, Int) {
            let (v, w) = UTF8Cursor.decode(bytes, at: p)
            return (ScalarClassifier.classify(value: v) == .whitespace, w)
        }

        while i < end {
            if let single = singleCandidateByte {
                // Jump to the next possible token start.
                guard let hit = memchr(bytes.baseAddress! + i, Int32(single), end - i) else { break }
                i = UnsafePointer<UInt8>(hit.assumingMemoryBound(to: UInt8.self)) - bytes.baseAddress!
            } else {
                let firstByte = bytes[i]
                guard candidateFirstBytes[Int(firstByte)] else {
                    i += UTF8Cursor.width(firstByte)
                    continue
                }
            }

            var best: (token: Int32, contentEnd: Int)?
            if let (index, matchEnd) = trie.longestMatch(bytes, from: i, filter: nil) {
                best = (index, matchEnd)
            }

            if hasLstrip {
                let (ws, w) = isWhitespace(i)
                if ws {
                    var probes: [Int] = []
                    var p = i + w
                    while p < end {
                        probes.append(p)
                        let (pws, pw) = isWhitespace(p)
                        guard pws else { break }
                        p += pw
                    }
                    for start in probes.reversed() {
                        if let (index, matchEnd) = trie.longestMatch(
                            bytes, from: start, filter: { self.tokens[Int($0)].lstrip })
                        {
                            if best == nil || tokens[Int(index)].scalarCount > tokens[Int(best!.token)].scalarCount {
                                best = (index, matchEnd)
                            }
                        }
                    }
                }
            }

            guard let match = best else {
                i += UTF8Cursor.width(bytes[i])
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
                    let (ws, w) = isWhitespace(next)
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

    /// Splits `text` into alternating text / token sections (empty text sections are omitted).
    func split(_ text: String) -> [Section] {
        var sections: [Section] = []
        let utf8 = text.utf8
        let scalars = text.unicodeScalars
        var sectionStart = text.startIndex
        var i = text.startIndex
        let end = text.endIndex

        while i < end {
            let firstByte = utf8[i]
            guard candidateFirstBytes[Int(firstByte)] else {
                scalars.formIndex(after: &i)
                continue
            }

            var best: (token: Int32, contentStart: String.Index, contentEnd: String.Index)?

            // Direct match at i.
            if let (index, matchEnd) = trie.longestMatch(text, from: i, filter: nil) {
                best = (index, i, matchEnd)
            }

            // lstrip tokens may start after a run of whitespace beginning at i.
            if hasLstrip, ScalarClassifier.classify(scalars[i]) == .whitespace {
                var p = scalars.index(after: i)
                var probes: [String.Index] = []
                while p < end {
                    probes.append(p)
                    guard ScalarClassifier.classify(scalars[p]) == .whitespace else { break }
                    scalars.formIndex(after: &p)
                }
                // Greedy `\s*`: prefer the latest start (longest whitespace prefix).
                for start in probes.reversed() {
                    if let (index, matchEnd) = trie.longestMatch(
                        text, from: start, filter: { self.tokens[Int($0)].lstrip })
                    {
                        if best == nil || tokens[Int(index)].scalarCount > tokens[Int(best!.token)].scalarCount {
                            best = (index, start, matchEnd)
                        }
                    }
                }
            }

            guard let match = best else {
                scalars.formIndex(after: &i)
                continue
            }

            if sectionStart < i {
                sections.append(.text(text[sectionStart..<i]))
            }
            let token = tokens[Int(match.token)]
            sections.append(.token(text[match.contentStart..<match.contentEnd], id: token.id))

            var next = match.contentEnd
            if token.rstrip {
                while next < end, ScalarClassifier.classify(scalars[next]) == .whitespace {
                    scalars.formIndex(after: &next)
                }
            }
            sectionStart = next
            i = next
        }

        if sectionStart < end {
            sections.append(.text(text[sectionStart..<end]))
        }
        return sections
    }
}

/// A byte trie whose nodes live in flat arrays; children resolved via a hash keyed on
/// `(node, byte)`.
struct ByteTrie: Sendable {
    private var children: [UInt64: Int32] = [:]
    private var terminal: [Int32] = [-1]

    @inline(__always)
    private static func key(_ node: Int32, _ byte: UInt8) -> UInt64 {
        (UInt64(UInt32(bitPattern: node)) << 32) | UInt64(byte)
    }

    mutating func insert(_ bytes: [UInt8], index: Int32) {
        var node: Int32 = 0
        for b in bytes {
            let k = Self.key(node, b)
            if let child = children[k] {
                node = child
            } else {
                let child = Int32(terminal.count)
                terminal.append(-1)
                children[k] = child
                node = child
            }
        }
        terminal[Int(node)] = index
    }

    /// Byte-offset variant of ``longestMatch(_:from:filter:)``.
    @inline(__always)
    func longestMatch(_ bytes: UnsafeBufferPointer<UInt8>, from start: Int, filter: ((Int32) -> Bool)?) -> (Int32, Int)?
    {
        var node: Int32 = 0
        var i = start
        var best: (Int32, Int)?
        while i < bytes.count {
            guard let child = children[Self.key(node, bytes[i])] else { break }
            node = child
            i += 1
            let t = terminal[Int(node)]
            if t >= 0, filter?(t) ?? true {
                best = (t, i)
            }
        }
        return best
    }

    /// Longest token starting at `start` whose index satisfies `filter`. Returns the token
    /// index and the index just past the match.
    func longestMatch(_ text: String, from start: String.Index, filter: ((Int32) -> Bool)?) -> (Int32, String.Index)? {
        let utf8 = text.utf8
        var node: Int32 = 0
        var i = start
        var best: (Int32, String.Index)?
        while i < utf8.endIndex {
            guard let child = children[Self.key(node, utf8[i])] else { break }
            node = child
            utf8.formIndex(after: &i)
            let t = terminal[Int(node)]
            if t >= 0, filter?(t) ?? true {
                best = (t, i)
            }
        }
        return best
    }
}
