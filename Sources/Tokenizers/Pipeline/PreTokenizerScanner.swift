// Regex-free implementations of the byte-level BPE pre-tokenization patterns used by the
// GPT-2, Llama-3 / cl100k, Qwen-2 / Qwen-3.5, o200k (GPT-4o, gpt-oss, Muse) and Falcon-H1
// model families. Each scanner reproduces the leftmost-first alternation semantics of the
// original regular expression exactly, but runs as a single linear pass over raw UTF-8 with
// table-driven classification.
//
// Patterns that are not recognised fall back to `NSRegularExpression` (see `SplitPreTokenizer`).

import Foundation

/// A well-known pre-tokenization regex, matched by its exact source text.
enum KnownSplitPattern: Sendable {
    /// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
    case gpt2
    /// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
    case llama3
    /// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
    case qwen2
    /// Qwen-3.5: `qwen2` with combining marks (`\p{M}`) treated as letters:
    /// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
    case qwen3
    /// o200k (GPT-4o, gpt-oss, Muse-Glimmer): case-aware letter runs with attached contractions,
    /// `\p{N}{1,3}`, punctuation swallowing trailing `[\r\n/]`.
    case o200k
    /// Falcon-H1: `o200k` without contraction suffixes and with single-digit numbers.
    case falcon

    static let gpt2Source = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
    static let llama3Source =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    static let qwen2Source =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    static let qwen3Source =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    static let o200kSource =
        #"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    static let falconSource =
        #"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    init?(regexSource: String) {
        switch regexSource {
        case Self.gpt2Source: self = .gpt2
        case Self.llama3Source: self = .llama3
        case Self.qwen2Source: self = .qwen2
        case Self.qwen3Source: self = .qwen3
        case Self.o200kSource: self = .o200k
        case Self.falconSource: self = .falcon
        default: return nil
        }
    }

    var source: String {
        switch self {
        case .gpt2: Self.gpt2Source
        case .llama3: Self.llama3Source
        case .qwen2: Self.qwen2Source
        case .qwen3: Self.qwen3Source
        case .o200k: Self.o200kSource
        case .falcon: Self.falconSource
        }
    }

    var rules: ByteLevelScanner.Rules {
        switch self {
        case .gpt2: .gpt2
        case .llama3: .llama3
        case .qwen2: .qwen2
        case .qwen3: .qwen3
        case .o200k: .o200k
        case .falcon: .falcon
        }
    }

    /// Splits `text` into the sequence of regex matches. Every scalar of the input belongs to
    /// exactly one piece for these patterns.
    func split(_ text: Substring, into pieces: inout [Substring]) {
        var copy = text
        var ranges: [Range<Int>] = []
        copy.withUTF8 { bytes in
            ranges.reserveCapacity(bytes.count / 4 + 1)
            ByteLevelScanner.scan(bytes, rules: rules, into: &ranges)
        }
        let utf8 = text.utf8
        let base = utf8.startIndex
        pieces.reserveCapacity(pieces.count + ranges.count)
        for range in ranges {
            let lower = utf8.index(base, offsetBy: range.lowerBound)
            let upper = utf8.index(base, offsetBy: range.upperBound)
            pieces.append(text[lower..<upper])
        }
    }

    /// Byte-range variant used by the fast encode path.
    @inline(__always)
    func split(_ bytes: UnsafeBufferPointer<UInt8>, into ranges: inout [Range<Int>]) {
        ByteLevelScanner.scan(bytes, rules: rules, into: &ranges)
    }

    /// Appends the matches in `text` as `PreToken`s, reusing `scratch` for the byte ranges so
    /// repeated calls on many small chunks do not allocate.
    func split(_ text: Substring, scratch: inout [Range<Int>], byteLevel: Bool, into output: inout [PreToken]) {
        var copy = text
        scratch.removeAll(keepingCapacity: true)
        copy.withUTF8 { bytes in
            ByteLevelScanner.scan(bytes, rules: rules, into: &scratch)
        }
        if scratch.count == 1 {
            output.append(PreToken(text: text, byteLevel: byteLevel))
            return
        }
        let utf8 = text.utf8
        let base = utf8.startIndex
        for range in scratch {
            let lower = utf8.index(base, offsetBy: range.lowerBound)
            let upper = utf8.index(base, offsetBy: range.upperBound)
            output.append(PreToken(text: text[lower..<upper], byteLevel: byteLevel))
        }
    }
}

// MARK: - Scalar classification

/// Coarse class of a scalar, as the classic GPT-2 / Llama-3 patterns see it (marks are
/// punctuation there).
@usableFromInline
enum ScalarClass: UInt8, Sendable {
    case letter = 0
    case number = 1
    case whitespace = 2
    case other = 3
}

/// Bit flags describing a scalar. Exactly one of `letter`, `number`, `whitespace`, `other`,
/// `mark` is set; `upper` / `lower` refine `letter`; `punctuation` refines `other`.
@usableFromInline
enum ScalarFlags {
    @usableFromInline static let letter: UInt8 = 1
    @usableFromInline static let number: UInt8 = 2
    @usableFromInline static let whitespace: UInt8 = 4
    @usableFromInline static let other: UInt8 = 8
    @usableFromInline static let mark: UInt8 = 16
    /// `Lu` / `Lt` (with `letter`).
    @usableFromInline static let upper: UInt8 = 32
    /// `Ll` (with `letter`).
    @usableFromInline static let lower: UInt8 = 64
    /// `P*` or ASCII punctuation as `tokenizers` defines it (with `other`).
    @usableFromInline static let punctuation: UInt8 = 128
}

@usableFromInline
enum ScalarClassifier {
    /// Precomputed flags for the Basic Multilingual Plane (64 KiB), built on first use.
    @usableFromInline
    static let bmp: [UInt8] = {
        var table = [UInt8](repeating: ScalarFlags.other, count: 0x10000)
        for v in 0..<0x10000 {
            guard let scalar = Unicode.Scalar(UInt32(v)) else { continue }
            table[v] = flagsSlow(scalar)
        }
        return table
    }()

    @inlinable
    static func classify(_ scalar: Unicode.Scalar) -> ScalarClass {
        classify(value: scalar.value)
    }

    @inlinable
    static func classify(value v: UInt32) -> ScalarClass {
        let f = flags(value: v)
        if f & ScalarFlags.letter != 0 { return .letter }
        if f & ScalarFlags.number != 0 { return .number }
        if f & ScalarFlags.whitespace != 0 { return .whitespace }
        return .other
    }

    @inlinable
    static func flags(value v: UInt32) -> UInt8 {
        if v < 0x10000 { return bmp[Int(v)] }
        guard let scalar = Unicode.Scalar(v) else { return ScalarFlags.other }
        return flagsSlow(scalar)
    }

    @usableFromInline
    static func flagsSlow(_ scalar: Unicode.Scalar) -> UInt8 {
        let v = scalar.value
        if v < 0x80 {
            switch v {
            case 0x41...0x5A: return ScalarFlags.letter | ScalarFlags.upper
            case 0x61...0x7A: return ScalarFlags.letter | ScalarFlags.lower
            case 0x30...0x39: return ScalarFlags.number
            case 0x09...0x0D, 0x20: return ScalarFlags.whitespace
            case 0x21...0x2F, 0x3A...0x40, 0x5B...0x60, 0x7B...0x7E:
                return ScalarFlags.other | ScalarFlags.punctuation
            default: return ScalarFlags.other
            }
        }
        let props = scalar.properties
        switch props.generalCategory {
        case .uppercaseLetter, .titlecaseLetter:
            return ScalarFlags.letter | ScalarFlags.upper
        case .lowercaseLetter:
            return ScalarFlags.letter | ScalarFlags.lower
        case .modifierLetter, .otherLetter:
            return ScalarFlags.letter
        case .nonspacingMark, .spacingMark, .enclosingMark:
            return ScalarFlags.mark
        case .decimalNumber, .letterNumber, .otherNumber:
            return ScalarFlags.number
        case .connectorPunctuation, .dashPunctuation, .openPunctuation, .closePunctuation,
            .initialPunctuation, .finalPunctuation, .otherPunctuation:
            return ScalarFlags.other | ScalarFlags.punctuation
        default:
            return props.isWhitespace ? ScalarFlags.whitespace : ScalarFlags.other
        }
    }

    @inlinable
    static func isNewline(_ v: UInt32) -> Bool {
        v == 0x0A || v == 0x0D
    }
}

// MARK: - UTF-8 decoding

@usableFromInline
enum UTF8Cursor {
    /// Decodes the scalar starting at `i` (assumes well-formed UTF-8, as produced by `String`).
    /// Returns the scalar value and its encoded width.
    @inlinable
    static func decode(_ bytes: UnsafeBufferPointer<UInt8>, at i: Int) -> (value: UInt32, width: Int) {
        let b0 = bytes[i]
        if b0 < 0x80 { return (UInt32(b0), 1) }
        if b0 < 0xE0 {
            let b1 = bytes[i + 1]
            return ((UInt32(b0 & 0x1F) << 6) | UInt32(b1 & 0x3F), 2)
        }
        if b0 < 0xF0 {
            let b1 = bytes[i + 1]
            let b2 = bytes[i + 2]
            return ((UInt32(b0 & 0x0F) << 12) | (UInt32(b1 & 0x3F) << 6) | UInt32(b2 & 0x3F), 3)
        }
        let b1 = bytes[i + 1]
        let b2 = bytes[i + 2]
        let b3 = bytes[i + 3]
        return ((UInt32(b0 & 0x07) << 18) | (UInt32(b1 & 0x3F) << 12) | (UInt32(b2 & 0x3F) << 6) | UInt32(b3 & 0x3F), 4)
    }

    /// Width of the scalar starting at `i`.
    @inlinable
    static func width(_ b0: UInt8) -> Int {
        if b0 < 0x80 { return 1 }
        if b0 < 0xE0 { return 2 }
        if b0 < 0xF0 { return 3 }
        return 4
    }
}

// MARK: - Scanner

enum ByteLevelScanner {
    struct Rules: Sendable {
        /// The pattern starts with a standalone contraction alternative (`'s|'t|…`).
        let standaloneContractions: Bool
        /// Contractions are matched case-insensitively (`(?i:'s|...)`).
        let caseInsensitiveContractions: Bool
        /// Letter runs may be preceded by any single non-letter/non-number/non-newline scalar
        /// (`[^\r\n\p{L}\p{N}]?\p{L}+`) instead of only an optional space (` ?\p{L}+`).
        let letterPrefixAnyNonLetter: Bool
        /// Maximum digits per number piece (`Int.max` for `\p{N}+`).
        let maxDigits: Int
        /// Punctuation runs swallow trailing CR/LF (`[^\s\p{L}\p{N}]+[\r\n]*`).
        let punctuationTrailingNewlines: Bool
        /// Punctuation runs also swallow `/` among the trailing characters (`[\r\n/]*`).
        let punctuationTrailingSlash: Bool
        /// Whitespace runs that contain a newline end at the last newline (`\s*[\r\n]+`).
        let newlineRun: Bool
        /// Flags that make a scalar part of a letter run (`letter`, optionally `mark`).
        let letterMask: UInt8
        /// Flags that make a scalar part of a punctuation run (`other`, optionally `mark`).
        let punctuationMask: UInt8
        /// Letter runs follow the o200k case-aware alternation
        /// `[Lu Lt Lm Lo M]*[Ll Lm Lo M]+ | [Lu Lt Lm Lo M]+[Ll Lm Lo M]*`.
        let caseSplit: Bool
        /// A contraction directly after a letter run belongs to it (`…(?i:'s|'t|…)?`).
        let contractionSuffix: Bool

        static let gpt2 = Rules(
            standaloneContractions: true, caseInsensitiveContractions: false, letterPrefixAnyNonLetter: false,
            maxDigits: Int.max, punctuationTrailingNewlines: false, punctuationTrailingSlash: false, newlineRun: false,
            letterMask: ScalarFlags.letter, punctuationMask: ScalarFlags.other | ScalarFlags.mark,
            caseSplit: false, contractionSuffix: false
        )
        static let llama3 = Rules(
            standaloneContractions: true, caseInsensitiveContractions: true, letterPrefixAnyNonLetter: true,
            maxDigits: 3, punctuationTrailingNewlines: true, punctuationTrailingSlash: false, newlineRun: true,
            letterMask: ScalarFlags.letter, punctuationMask: ScalarFlags.other | ScalarFlags.mark,
            caseSplit: false, contractionSuffix: false
        )
        static let qwen2 = Rules(
            standaloneContractions: true, caseInsensitiveContractions: true, letterPrefixAnyNonLetter: true,
            maxDigits: 1, punctuationTrailingNewlines: true, punctuationTrailingSlash: false, newlineRun: true,
            letterMask: ScalarFlags.letter, punctuationMask: ScalarFlags.other | ScalarFlags.mark,
            caseSplit: false, contractionSuffix: false
        )
        /// `[\p{L}\p{M}]+` letter runs; `[^\s\p{L}\p{M}\p{N}]+` punctuation excludes marks.
        static let qwen3 = Rules(
            standaloneContractions: true, caseInsensitiveContractions: true, letterPrefixAnyNonLetter: true,
            maxDigits: 1, punctuationTrailingNewlines: true, punctuationTrailingSlash: false, newlineRun: true,
            letterMask: ScalarFlags.letter | ScalarFlags.mark, punctuationMask: ScalarFlags.other,
            caseSplit: false, contractionSuffix: false
        )
        /// Marks join letter runs *and* punctuation runs (`[^\s\p{L}\p{N}]+` keeps them).
        static let o200k = Rules(
            standaloneContractions: false, caseInsensitiveContractions: true, letterPrefixAnyNonLetter: true,
            maxDigits: 3, punctuationTrailingNewlines: true, punctuationTrailingSlash: true, newlineRun: true,
            letterMask: ScalarFlags.letter | ScalarFlags.mark, punctuationMask: ScalarFlags.other | ScalarFlags.mark,
            caseSplit: true, contractionSuffix: true
        )
        static let falcon = Rules(
            standaloneContractions: false, caseInsensitiveContractions: true, letterPrefixAnyNonLetter: true,
            maxDigits: 1, punctuationTrailingNewlines: true, punctuationTrailingSlash: true, newlineRun: true,
            letterMask: ScalarFlags.letter | ScalarFlags.mark, punctuationMask: ScalarFlags.other | ScalarFlags.mark,
            caseSplit: true, contractionSuffix: false
        )
    }

    /// Splits `bytes` (well-formed UTF-8) into match ranges (byte offsets).
    static func scan(_ bytes: UnsafeBufferPointer<UInt8>, rules: Rules, into pieces: inout [Range<Int>]) {
        ScalarClassifier.bmp.withUnsafeBufferPointer { table in
            scan(bytes, rules: rules, table: table, into: &pieces)
        }
    }

    /// Decodes and classifies the scalar at `p`. Returns (value, width, flags).
    @inline(__always)
    static func decodeClassified(
        _ bytes: UnsafeBufferPointer<UInt8>, _ p: Int, _ table: UnsafeBufferPointer<UInt8>
    ) -> (UInt32, Int, UInt8) {
        let b0 = bytes[p]
        if b0 < 0x80 { return (UInt32(b0), 1, table[Int(b0)]) }
        let (v, w) = UTF8Cursor.decode(bytes, at: p)
        if v < 0x10000 { return (v, w, table[Int(v)]) }
        return (v, w, ScalarClassifier.flags(value: v))
    }

    private static func scan(
        _ bytes: UnsafeBufferPointer<UInt8>, rules: Rules, table: UnsafeBufferPointer<UInt8>,
        into pieces: inout [Range<Int>]
    ) {
        let end = bytes.count
        var i = 0
        let number = ScalarFlags.number
        let letterMask = rules.letterMask
        let punctuationMask = rules.punctuationMask

        @inline(__always)
        func at(_ p: Int) -> (UInt32, Int, UInt8) { decodeClassified(bytes, p, table) }

        /// Advances over scalars whose flags intersect `mask`.
        @inline(__always)
        func consumeRun(from p: Int, _ mask: UInt8) -> Int {
            var j = p
            while j < end {
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

        /// End of the letter run that starts at `s` (a letter-class scalar).
        @inline(__always)
        func letterRunEnd(from s: Int) -> Int {
            var e: Int
            if rules.caseSplit {
                e = caseSplitRunEnd(bytes, from: s, end: end, letterMask: letterMask, table: table)
            } else {
                e = consumeRun(from: s + UTF8Cursor.width(bytes[s]), letterMask)
            }
            if rules.contractionSuffix, e < end, bytes[e] == 0x27,
                let k = contractionEnd(bytes, after: e, caseInsensitive: rules.caseInsensitiveContractions)
            {
                e = k
            }
            return e
        }

        while i < end {
            let (c, cw, f) = at(i)

            // 1. Standalone contractions: 's 't 're 've 'm 'll 'd
            if rules.standaloneContractions, c == 0x27 {
                if let j = contractionEnd(bytes, after: i, caseInsensitive: rules.caseInsensitiveContractions) {
                    pieces.append(i..<j)
                    i = j
                    continue
                }
            }

            // 2. Letter runs with optional prefix.
            if f & letterMask != 0 {
                let j = letterRunEnd(from: i)
                pieces.append(i..<j)
                i = j
                continue
            }

            let prefixAllowsLetters: Bool
            if rules.letterPrefixAnyNonLetter {
                prefixAllowsLetters = f & number == 0 && !ScalarClassifier.isNewline(c)
            } else {
                prefixAllowsLetters = c == 0x20
            }
            if prefixAllowsLetters {
                let n = i + cw
                if n < end, at(n).2 & letterMask != 0 {
                    let j = letterRunEnd(from: n)
                    pieces.append(i..<j)
                    i = j
                    continue
                }
            }

            // 3. Number runs.
            if f & number != 0 {
                var j = i + cw
                var count = 1
                while j < end, count < rules.maxDigits {
                    let (_, w, g) = at(j)
                    if g & number == 0 { break }
                    j += w
                    count += 1
                }
                pieces.append(i..<j)
                i = j
                continue
            }
            if !rules.letterPrefixAnyNonLetter, c == 0x20 {
                // GPT-2: ` ?\p{N}+`
                let n = i + 1
                if n < end {
                    let (_, nw, g) = at(n)
                    if g & number != 0 {
                        let j = consumeRun(from: n + nw, number)
                        pieces.append(i..<j)
                        i = j
                        continue
                    }
                }
            }

            // 4. Punctuation runs with optional leading space (and optional trailing newlines).
            if f & punctuationMask != 0 || c == 0x20 {
                var start = i
                if c == 0x20 {
                    let n = i + 1
                    if n < end, at(n).2 & punctuationMask != 0 {
                        start = n
                    } else {
                        // A lone space that is not followed by punctuation: whitespace rules apply.
                        i = scanWhitespace(bytes, from: i, rules: rules, table: table, into: &pieces)
                        continue
                    }
                }
                var j = consumeRun(from: start + UTF8Cursor.width(bytes[start]), punctuationMask)
                if rules.punctuationTrailingNewlines {
                    if rules.punctuationTrailingSlash {
                        while j < end, ScalarClassifier.isNewline(UInt32(bytes[j])) || bytes[j] == 0x2F { j += 1 }
                    } else {
                        while j < end, ScalarClassifier.isNewline(UInt32(bytes[j])) { j += 1 }
                    }
                }
                pieces.append(i..<j)
                i = j
                continue
            }

            // 5–7. Whitespace.
            i = scanWhitespace(bytes, from: i, rules: rules, table: table, into: &pieces)
        }
    }

    /// o200k letter run starting at `s`:
    /// `[Lu Lt Lm Lo M]*[Ll Lm Lo M]+ | [Lu Lt Lm Lo M]+[Ll Lm Lo M]*`, leftmost-first with
    /// backtracking. Let U = `Lu`/`Lt`, L = `Ll`, N = `Lm`/`Lo`/marks. The first alternative
    /// greedily takes U/N scalars; if a L/N scalar follows it continues over L/N. Otherwise
    /// it backtracks to the last N in the U/N run and stops right after it (the U scalars
    /// that follow start the next match). Only when the run contains no N at all — it is
    /// pure upper case — does the second alternative apply and take the whole run.
    @inline(__always)
    private static func caseSplitRunEnd(
        _ bytes: UnsafeBufferPointer<UInt8>, from s: Int, end: Int, letterMask: UInt8, table: UnsafeBufferPointer<UInt8>
    ) -> Int {
        let upper = ScalarFlags.upper
        let lower = ScalarFlags.lower
        var j = s
        var lastNeutralEnd = -1
        // Phase 1: `[Lu Lt Lm Lo M]*`
        while j < end {
            let (_, w, f) = decodeClassified(bytes, j, table)
            if f & letterMask == 0 || f & lower != 0 { break }
            if f & upper == 0 { lastNeutralEnd = j + w }
            j += w
        }
        if j < end {
            let (_, w, f) = decodeClassified(bytes, j, table)
            if f & letterMask != 0 {
                // A lower-case letter follows: `[Ll Lm Lo M]+`.
                var e = j + w
                while e < end {
                    let (_, w2, g) = decodeClassified(bytes, e, table)
                    if g & letterMask == 0 || g & upper != 0 { break }
                    e += w2
                }
                return e
            }
        }
        if lastNeutralEnd >= 0 { return lastNeutralEnd }
        return j
    }

    /// Handles the `\s*[\r\n]+`, `\s+(?!\S)` and `\s+` alternatives starting at a whitespace
    /// scalar. Returns the offset after the emitted piece.
    @inline(__always)
    private static func scanWhitespace(
        _ bytes: UnsafeBufferPointer<UInt8>,
        from i: Int,
        rules: Rules,
        table: UnsafeBufferPointer<UInt8>,
        into pieces: inout [Range<Int>]
    ) -> Int {
        let end = bytes.count
        let whitespace = ScalarFlags.whitespace
        let (first, fw, _) = decodeClassified(bytes, i, table)
        var k = i + fw
        var lastNewlineEnd = ScalarClassifier.isNewline(first) ? k : -1
        var previousStart = i
        var runLength = 1
        while k < end {
            let (v, w, f) = decodeClassified(bytes, k, table)
            if f & whitespace == 0 { break }
            previousStart = k
            k += w
            if ScalarClassifier.isNewline(v) { lastNewlineEnd = k }
            runLength += 1
        }

        if rules.newlineRun, lastNewlineEnd >= 0 {
            pieces.append(i..<lastNewlineEnd)
            return lastNewlineEnd
        }

        if k == end || runLength == 1 {
            // `\s+(?!\S)` at end of input, or a single whitespace char via `\s+`.
            pieces.append(i..<k)
            return k
        }
        // `\s+(?!\S)`: leave the last whitespace to be attached to the following token.
        pieces.append(i..<previousStart)
        return previousStart
    }

    /// If a contraction (`'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`) starts at `apostrophe`,
    /// returns the offset just past it.
    @inline(__always)
    private static func contractionEnd(
        _ bytes: UnsafeBufferPointer<UInt8>, after apostrophe: Int, caseInsensitive: Bool
    ) -> Int? {
        let end = bytes.count
        let first = apostrophe + 1
        guard first < end else { return nil }

        @inline(__always)
        func fold(_ p: Int) -> (UInt32, Int) {
            let (v, w) = UTF8Cursor.decode(bytes, at: p)
            if caseInsensitive {
                if v >= 0x41, v <= 0x5A { return (v + 0x20, w) }
                if v == 0x17F { return (UInt32(UInt8(ascii: "s")), w) }  // ſ folds to s
            }
            return (v, w)
        }

        let (f, fw) = fold(first)
        switch f {
        case UInt32(UInt8(ascii: "s")), UInt32(UInt8(ascii: "t")), UInt32(UInt8(ascii: "m")), UInt32(UInt8(ascii: "d")):
            return first + fw
        case UInt32(UInt8(ascii: "r")), UInt32(UInt8(ascii: "v")), UInt32(UInt8(ascii: "l")):
            let second = first + fw
            guard second < end else { return nil }
            let (g, gw) = fold(second)
            if f == UInt32(UInt8(ascii: "l")) {
                return g == UInt32(UInt8(ascii: "l")) ? second + gw : nil
            }
            return g == UInt32(UInt8(ascii: "e")) ? second + gw : nil
        default:
            return nil
        }
    }
}
