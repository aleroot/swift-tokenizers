// Regex-free implementations of the three `Split` patterns that form the DeepSeek
// (V2 / V3 / R1 / V4) byte-level pre-tokenizer sequence. Each reproduces the leftmost-first
// semantics of its regular expression exactly, running as a single linear pass over raw UTF-8.
//
// These patterns do not match every scalar, so the scanners also emit the gaps between
// matches: that is what `Isolated` behaviour keeps as pieces.

import Foundation

enum DeepSeekScanner {

    // MARK: `\p{N}{1,3}`

    /// Cuts runs of numeric scalars into groups of at most three, left to right.
    static func scanNumbers(_ bytes: UnsafeBufferPointer<UInt8>, into pieces: inout [Range<Int>]) {
        guard !bytes.isEmpty else { return }
        ScalarClassifier.bmp.withUnsafeBufferPointer { table in
            let end = bytes.count
            let number = ScalarFlags.number
            var i = 0
            var cursor = 0
            while i < end {
                // Skip to the next numeric scalar, sixteen ASCII bytes at a time.
                if let v = ByteKernels.asciiChunk(bytes, at: i) {
                    let digits = ByteKernels.bits(ByteKernels.digits(v))
                    if digits == 0 {
                        i += ByteKernels.width
                        continue
                    }
                    i += digits.trailingZeroBitCount
                } else {
                    let (_, w, f) = ByteLevelScanner.decodeClassified(bytes, i, table)
                    if f & number == 0 {
                        i += w
                        continue
                    }
                }
                if cursor < i { pieces.append(cursor..<i) }
                var j = i
                var count = 0
                while j < end, count < 3 {
                    let (_, w, f) = ByteLevelScanner.decodeClassified(bytes, j, table)
                    if f & number == 0 { break }
                    j += w
                    count += 1
                }
                pieces.append(i..<j)
                cursor = j
                i = j
            }
            if cursor < end { pieces.append(cursor..<end) }
        }
    }

    // MARK: `[一-龥぀-ゟ゠-ヿ]+`

    /// `true` for U+4E00...U+9FA5 (Han) and U+3040...U+30FF (hiragana and katakana).
    @inline(__always)
    static func isCJK(_ v: UInt32) -> Bool {
        (v >= 0x4E00 && v <= 0x9FA5) || (v >= 0x3040 && v <= 0x30FF)
    }

    /// Isolates maximal runs of Han, hiragana and katakana.
    static func scanCJK(_ bytes: UnsafeBufferPointer<UInt8>, into pieces: inout [Range<Int>]) {
        guard !bytes.isEmpty else { return }
        let end = bytes.count
        var i = 0
        var cursor = 0

        // Every scalar of the class encodes as three bytes whose lead byte is in 0xE3...0xE9.
        // No continuation byte and no other lead byte falls in that range, so a raw byte scan
        // for one can never stop inside another scalar.
        @inline(__always)
        func candidate(_ b: UInt8) -> Bool { b >= 0xE3 && b <= 0xE9 }

        while i < end {
            if i + ByteKernels.width <= end {
                let v = ByteKernels.load(bytes.baseAddress!, i)
                let lanes = ByteKernels.bits(ByteKernels.range(v &- 0xE3, 0, 7))
                if lanes == 0 {
                    i += ByteKernels.width
                    continue
                }
                i += lanes.trailingZeroBitCount
            } else if !candidate(bytes[i]) {
                i += 1
                continue
            }
            let (value, width) = UTF8Cursor.decode(bytes, at: i)
            if !isCJK(value) {
                i += width
                continue
            }
            if cursor < i { pieces.append(cursor..<i) }
            var j = i + width
            while j < end, candidate(bytes[j]) {
                let (next, w) = UTF8Cursor.decode(bytes, at: j)
                if !isCJK(next) { break }
                j += w
            }
            pieces.append(i..<j)
            cursor = j
            i = j
        }
        if cursor < end { pieces.append(cursor..<end) }
    }

    // MARK: The main alternation

    /// `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+`
    /// `|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`
    /// `| ?[\p{P}\p{S}]+[\r\n]*`
    /// `|\s*[\r\n]+|\s+(?!\S)|\s+`
    static func scan(_ bytes: UnsafeBufferPointer<UInt8>, into pieces: inout [Range<Int>]) {
        guard !bytes.isEmpty else { return }
        ScalarClassifier.bmpWithSymbols.withUnsafeBufferPointer { table in
            scan(bytes, table: table, into: &pieces)
        }
    }

    /// Decodes and classifies the scalar at `p` with `\p{S}` folded into
    /// ``ScalarFlags/punctuation``, so a single flag stands for the `[\p{P}\p{S}]` class.
    @inline(__always)
    private static func classify(
        _ bytes: UnsafeBufferPointer<UInt8>, _ p: Int, _ table: UnsafeBufferPointer<UInt8>
    ) -> (UInt32, Int, UInt8) {
        let b0 = bytes[p]
        if b0 < 0x80 { return (UInt32(b0), 1, table[Int(b0)]) }
        let (v, w) = UTF8Cursor.decode(bytes, at: p)
        if v < 0x10000 { return (v, w, table[Int(v)]) }
        return (v, w, ScalarClassifier.symbolFlags(value: v))
    }

    @inline(__always)
    private static func isASCIILetter(_ b: UInt8) -> Bool {
        (b | 0x20) >= UInt8(ascii: "a") && (b | 0x20) <= UInt8(ascii: "z")
    }

    private static func scan(
        _ bytes: UnsafeBufferPointer<UInt8>, table: UnsafeBufferPointer<UInt8>, into pieces: inout [Range<Int>]
    ) {
        let end = bytes.count
        // `[\p{L}\p{M}]` and `[\p{P}\p{S}]`, the two classes the alternation runs over.
        let letterMask = ScalarFlags.letter | ScalarFlags.mark
        let punctuationMask = ScalarFlags.punctuation
        var i = 0
        var cursor = 0

        @inline(__always)
        func at(_ p: Int) -> (UInt32, Int, UInt8) { classify(bytes, p, table) }

        /// Emits the gap before `match`, then `match`.
        @inline(__always)
        func emit(_ match: Range<Int>) {
            if cursor < match.lowerBound { pieces.append(cursor..<match.lowerBound) }
            pieces.append(match)
            cursor = match.upperBound
        }

        /// Advances over scalars whose flags intersect `mask`, sixteen bytes at a time while
        /// the run is ASCII.
        @inline(__always)
        func consumeRun(from p: Int, _ mask: UInt8) -> Int {
            var j = p
            while j < end {
                if j + ByteKernels.width <= end {
                    let v = ByteKernels.load(bytes.baseAddress!, j)
                    let lanes =
                        mask & ScalarFlags.letter != 0 ? ByteKernels.letters(v) : ByteKernels.punctuation(v)
                    // Lanes that end the run: ASCII bytes outside the class, plus non-ASCII
                    // bytes, which the scalar step below classifies.
                    let stop = ~ByteKernels.bits(lanes)
                    if stop == 0 {
                        j += ByteKernels.width
                        continue
                    }
                    j += stop.trailingZeroBitCount
                }
                let b0 = bytes[j]
                if b0 < 0x80 {
                    if table[Int(b0)] & mask == 0 { break }
                    j += 1
                } else {
                    let (_, w, f) = at(j)
                    if f & mask == 0 { break }
                    j += w
                }
            }
            return j
        }

        /// `[\p{P}\p{S}]+[\r\n]*` starting at `start`, the run's first scalar.
        @inline(__always)
        func punctuationRunEnd(from start: Int) -> Int {
            var j = consumeRun(from: start + UTF8Cursor.width(bytes[start]), punctuationMask)
            while j < end, ScalarClassifier.isNewline(UInt32(bytes[j])) { j += 1 }
            return j
        }

        while i < end {
            let (c, cw, f) = at(i)

            // 1. `[!"#$…~][A-Za-z]+`: ASCII punctuation glued to an ASCII word.
            if c < 0x80, f & punctuationMask != 0, i + 1 < end, isASCIILetter(bytes[i + 1]) {
                var j = i + 2
                while j < end, isASCIILetter(bytes[j]) { j += 1 }
                emit(i..<j)
                i = j
                continue
            }

            // 2. `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`. A leading mark is in both the prefix
            // and the run, and either reading spans the same bytes, so the run covers it.
            if f & letterMask != 0 {
                let j = consumeRun(from: i + cw, letterMask)
                emit(i..<j)
                i = j
                continue
            }
            if f & punctuationMask == 0, !ScalarClassifier.isNewline(c) {
                let n = i + cw
                if n < end, at(n).2 & letterMask != 0 {
                    let j = consumeRun(from: n + UTF8Cursor.width(bytes[n]), letterMask)
                    emit(i..<j)
                    i = j
                    continue
                }
            }

            // 3. ` ?[\p{P}\p{S}]+[\r\n]*`.
            if f & punctuationMask != 0 {
                let j = punctuationRunEnd(from: i)
                emit(i..<j)
                i = j
                continue
            }
            if c == 0x20, i + 1 < end, at(i + 1).2 & punctuationMask != 0 {
                let j = punctuationRunEnd(from: i + 1)
                emit(i..<j)
                i = j
                continue
            }

            // 4-6. `\s*[\r\n]+`, `\s+(?!\S)`, `\s+`.
            if f & ScalarFlags.whitespace != 0 {
                let match = whitespaceMatch(bytes, from: i, table: table)
                emit(match)
                i = match.upperBound
                continue
            }

            // No alternative applies: the scalar stays in the gap. Numbers reach this point
            // unless they prefix a letter run, which is why the pattern needs `\p{N}{1,3}`
            // as a separate split.
            i += cw
        }
        if cursor < end { pieces.append(cursor..<end) }
    }

    /// The whitespace alternatives, starting at a whitespace scalar. `\s*[\r\n]+` wins when the
    /// run contains a newline and ends at the last one; otherwise `\s+(?!\S)` takes the run
    /// except its final scalar, which `\s+` leaves for the next match to absorb.
    @inline(__always)
    private static func whitespaceMatch(
        _ bytes: UnsafeBufferPointer<UInt8>, from i: Int, table: UnsafeBufferPointer<UInt8>
    ) -> Range<Int> {
        let end = bytes.count
        let whitespace = ScalarFlags.whitespace
        let (first, fw, _) = classify(bytes, i, table)
        var k = i + fw
        var lastNewlineEnd = ScalarClassifier.isNewline(first) ? k : -1
        var previousStart = i
        var runLength = 1
        while k < end {
            let (v, w, f) = classify(bytes, k, table)
            if f & whitespace == 0 { break }
            previousStart = k
            k += w
            if ScalarClassifier.isNewline(v) { lastNewlineEnd = k }
            runLength += 1
        }
        if lastNewlineEnd >= 0 { return i..<lastNewlineEnd }
        if k == end || runLength == 1 { return i..<k }
        return i..<previousStart
    }
}
