// Pre-tokenization: splitting normalized text into the chunks the model tokenizes independently.
// Built-in pre-tokenizers work on raw UTF-8 as a list of ``PreTokenizationStage``s (splitters,
// rewriters and the `byteLevel` marker) executed by ``PreTokenizationRunner``; the public
// `[String]` API is derived from the same stages.

import Foundation

/// Options passed to pre-tokenization.
public enum PreTokenizerOption: String, Sendable {
    /// The text being processed starts the input (before any added token or other piece).
    case firstSection
}

public typealias PreTokenizerOptions = Set<PreTokenizerOption>

/// Splits text into chunks that are tokenized independently by the model.
public protocol PreTokenizer: Sendable {
    func preTokenize(text: String, options: PreTokenizerOptions) -> [String]
    func preTokenize(texts: [String], options: PreTokenizerOptions) -> [String]
    func callAsFunction(texts: [String], options: PreTokenizerOptions) -> [String]
    func callAsFunction(text: String, options: PreTokenizerOptions) -> [String]
    /// Creates the pre-tokenizer from its `tokenizer.json` entry.
    /// - Throws: ``TokenizerError/invalidConfiguration(_:)`` when required fields are missing.
    init(config: Config) throws
}

public extension PreTokenizer {
    func preTokenize(texts: [String], options: PreTokenizerOptions = [.firstSection]) -> [String] {
        texts.flatMap { preTokenize(text: $0, options: options) }
    }

    func callAsFunction(texts: [String], options: PreTokenizerOptions = [.firstSection]) -> [String] {
        preTokenize(texts: texts, options: options)
    }

    func callAsFunction(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        preTokenize(text: text, options: options)
    }
}

// MARK: - Byte-level contracts

/// One step of byte-level pre-tokenization.
enum PreTokenizationStage: Sendable {
    /// Splits each piece into sub-pieces (ranges of the same text).
    case split(any ByteSplitter)
    /// Rewrites each piece's text, then splits the rewritten text.
    case rewrite(any ByteRewriter)
    /// Marks every piece as byte-level (`ByteLevel` as the final stage).
    case byteLevel
}

/// A pre-tokenizer expressed as byte-level stages. Built-in pre-tokenizers all conform.
protocol StagedPreTokenizer: PreTokenizer {
    var stages: [PreTokenizationStage] { get }
}

extension StagedPreTokenizer {
    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        var copy = text
        var pieces: [String] = []
        let runner = PreTokenizationRunner(stages: stages)
        copy.withUTF8 { bytes in
            runner.run(bytes, options: options, scratch: ScratchBuffers()) { piece, byteLevel in
                pieces.append(
                    byteLevel ? ByteLevelAlphabet.encode(piece) : String(decoding: piece, as: UTF8.self))
            }
        }
        return pieces
    }
}

/// Splits a chunk into pieces given as byte ranges of the chunk.
protocol ByteSplitter: StagedPreTokenizer {
    func split(_ bytes: UnsafeBufferPointer<UInt8>, options: PreTokenizerOptions, into pieces: inout [Range<Int>])
}

extension ByteSplitter {
    var stages: [PreTokenizationStage] { [.split(self)] }
}

/// Rewrites a chunk's text and splits the result.
protocol ByteRewriter: StagedPreTokenizer {
    /// Appends the rewritten form of `bytes` to `output`, and the pieces it splits into as
    /// ranges relative to the appended text.
    func rewrite(
        _ bytes: UnsafeBufferPointer<UInt8>, options: PreTokenizerOptions, into output: inout [UInt8],
        pieces: inout [Range<Int>]
    )
}

extension ByteRewriter {
    var stages: [PreTokenizationStage] { [.rewrite(self)] }
}

// MARK: - Factory

enum PreTokenizerType: String {
    case Sequence
    case ByteLevel
    case Punctuation
    case Digits
    case Split
    case Whitespace
    case WhitespaceSplit
    case Metaspace
    case BertPreTokenizer
    case Unknown = ""
}

struct PreTokenizerFactory {
    static func fromConfig(config: Config?) throws -> (any StagedPreTokenizer)? {
        guard let config, let typeName = config.type.string() else { return nil }
        switch PreTokenizerType(rawValue: typeName) {
        case .Sequence: return try PreTokenizerSequence(config: config)
        case .ByteLevel: return ByteLevelPreTokenizer(config: config)
        case .Punctuation: return PunctuationPreTokenizer(config: config)
        case .Digits: return DigitsPreTokenizer(config: config)
        case .Split: return try SplitPreTokenizer(config: config)
        case .Whitespace, .WhitespaceSplit: return WhitespacePreTokenizer(config: config)
        case .Metaspace:
            try MetaspacePreTokenizer.validate(config)
            return MetaspacePreTokenizer(config: config)
        case .BertPreTokenizer: return BertPreTokenizer(config: config)
        default: throw TokenizerError.unsupportedComponent("pre-tokenizer `\(typeName)`")
        }
    }
}

// MARK: - Implementations

/// Splits on whitespace (dropped) and punctuation (isolated), as `tokenizers::BertPreTokenizer`.
final class BertPreTokenizer: ByteSplitter {
    required init(config: Config) {}

    func split(_ bytes: UnsafeBufferPointer<UInt8>, options: PreTokenizerOptions, into pieces: inout [Range<Int>]) {
        ScalarClassifier.bmp.withUnsafeBufferPointer { table in
            var i = 0
            var start = 0
            let end = bytes.count
            while i < end {
                // All-ASCII chunk: classify 16 bytes at once and visit only the boundaries
                // (whitespace transitions and punctuation, which is always its own piece).
                if let v = ByteKernels.asciiChunk(bytes, at: i) {
                    let whitespace = ByteKernels.bits(ByteKernels.whitespace(v))
                    let punctuation = ByteKernels.bits(ByteKernels.punctuation(v))
                    var boundaries = whitespace | punctuation
                    while boundaries != 0 {
                        let lane = boundaries.trailingZeroBitCount
                        boundaries &= boundaries - 1
                        let position = i + lane
                        if start < position { pieces.append(start..<position) }
                        if punctuation & (1 << lane) != 0 { pieces.append(position..<position + 1) }
                        start = position + 1
                    }
                    i += ByteKernels.width
                    continue
                }
                let (value, width, flags) = ByteLevelScanner.decodeClassified(bytes, i, table)
                if flags & ScalarFlags.whitespace != 0 {
                    if start < i { pieces.append(start..<i) }
                    start = i + width
                } else if TokenizerUnicode.isPunctuation(value) {
                    if start < i { pieces.append(start..<i) }
                    pieces.append(i..<i + width)
                    start = i + width
                }
                i += width
            }
            if start < end { pieces.append(start..<end) }
        }
    }
}

final class PreTokenizerSequence: StagedPreTokenizer {
    let preTokenizers: [any StagedPreTokenizer]
    let stages: [PreTokenizationStage]

    required init(config: Config) throws {
        let configs = try require(config.pretokenizers.array(), "Sequence pre-tokenizer", field: "pretokenizers")
        preTokenizers = try configs.compactMap { try PreTokenizerFactory.fromConfig(config: $0) }
        stages = preTokenizers.flatMap(\.stages)
    }
}

/// `Whitespace` (`\w+|[^\w\s]+`) and `WhitespaceSplit` (`\S+`).
final class WhitespacePreTokenizer: ByteSplitter {
    let splitWords: Bool
    required init(config: Config) { splitWords = config.type.string() == "Whitespace" }

    func split(_ bytes: UnsafeBufferPointer<UInt8>, options: PreTokenizerOptions, into pieces: inout [Range<Int>]) {
        let n = bytes.count
        var i = 0
        var start = -1
        var previousWord = false
        while i < n {
            if let v = ByteKernels.asciiChunk(bytes, at: i) {
                splitASCIIChunk(v, at: i, start: &start, previousWord: &previousWord, into: &pieces)
                i += ByteKernels.width
                continue
            }
            let b0 = bytes[i]
            let value: UInt32
            let width: Int
            let isWhitespace: Bool
            if b0 < 0x80 {
                value = UInt32(b0)
                width = 1
                isWhitespace = b0 == 0x20 || (b0 >= 0x09 && b0 <= 0x0D)
            } else {
                (value, width) = UTF8Cursor.decode(bytes, at: i)
                isWhitespace = ScalarClassifier.flags(value: value) & ScalarFlags.whitespace != 0
            }
            if isWhitespace {
                if start >= 0 {
                    pieces.append(start..<i)
                    start = -1
                }
            } else if splitWords {
                let isWord = Self.isWord(value)
                if start >= 0, isWord != previousWord {
                    pieces.append(start..<i)
                    start = -1
                }
                if start < 0 { start = i }
                previousWord = isWord
            } else if start < 0 {
                start = i
            }
            i += width
        }
        if start >= 0 { pieces.append(start..<n) }
    }

    /// Sixteen ASCII bytes at `offset`: the class of every lane comes from three lane masks,
    /// and only lanes where the class changes are visited. `start` / `previousWord` carry the
    /// scanner state across chunks exactly as the scalar loop does.
    @inline(__always)
    private func splitASCIIChunk(
        _ v: ByteKernels.Vector, at offset: Int, start: inout Int, previousWord: inout Bool,
        into pieces: inout [Range<Int>]
    ) {
        let whitespace = ByteKernels.bits(ByteKernels.whitespace(v))
        let word = splitWords ? ByteKernels.bits(ByteKernels.word(v)) : ~whitespace
        // Lanes whose class differs from the previous lane's (the lane before the chunk being
        // the scanner state: inside a piece of the `previousWord` class, or in whitespace).
        let inside: UInt16 = start >= 0 ? 1 : 0
        let previousIsWord: UInt16 = start >= 0 && (previousWord || !splitWords) ? 1 : 0
        var changes = (whitespace ^ (whitespace << 1 | (inside ^ 1))) | (word ^ (word << 1 | previousIsWord))
        while changes != 0 {
            let lane = changes.trailingZeroBitCount
            changes &= changes - 1
            let position = offset + lane
            if start >= 0 {
                pieces.append(start..<position)
                start = -1
            }
            if whitespace & (1 << lane) == 0 {
                start = position
                previousWord = word & (1 << lane) != 0
            }
        }
    }

    /// Regex `\w`: alphabetic, marks, decimal numbers, connector punctuation, join controls.
    @inline(__always)
    static func isWord(_ value: UInt32) -> Bool {
        if value < 128 {
            return (value >= 0x61 && value <= 0x7A) || (value >= 0x41 && value <= 0x5A)
                || (value >= 0x30 && value <= 0x39)
                || value == 0x5F
        }
        return ScalarClassifier.extraFlags(value: value) & ScalarExtraFlags.word != 0
    }
}

/// Replaces spaces with a replacement character (SentencePiece's `▁`) and optionally
/// prepends it, then splits so every chunk starts with the replacement.
final class MetaspacePreTokenizer: ByteRewriter {
    static func validate(_ config: Config) throws {
        if !config.replacement.isNull(), config.replacement.string()?.unicodeScalars.count != 1 {
            throw TokenizerError.invalidConfiguration("Metaspace replacement must be one Unicode scalar")
        }
    }
    let addPrefixSpace: Bool
    let replacement: String
    let stringReplacement: String
    private let replacementBytes: [UInt8]
    private let stringReplacementBytes: [UInt8]
    /// `str_rep` differs from a plain space, so spaces are substituted rather than copied.
    private let substitutesSpace: Bool
    /// `str_rep` equals the multi-byte `replacement`: substitution and split fuse into one pass.
    private let fusesMarker: Bool

    enum PrependScheme: String {
        case first
        case never
        case always

        static var defaultScheme: PrependScheme { .always }
        static func from(rawValue value: String?) -> PrependScheme {
            guard let value else { return defaultScheme }
            return PrependScheme(rawValue: value) ?? defaultScheme
        }
    }

    let prependScheme: PrependScheme

    /// When `false` (Mistral v0.3 and other recent SentencePiece exports), the text is not
    /// split on the replacement character: the whole chunk is one BPE word, so runs of
    /// whitespace can merge into multi-space tokens.
    let split: Bool

    required init(config: Config) {
        addPrefixSpace = config.addPrefixSpace.boolean(or: false)
        replacement = config.replacement.string(or: " ")
        stringReplacement = config.strRep.string(or: replacement)
        split = config.split.boolean(or: true)
        replacementBytes = Array(replacement.utf8)
        stringReplacementBytes = Array(stringReplacement.utf8)
        substitutesSpace = stringReplacementBytes != [0x20]
        fusesMarker = stringReplacementBytes.elementsEqual(replacementBytes) && replacementBytes.count > 1

        // `prepend_scheme` supersedes `add_prefix_space` (tokenizers PR #1357).
        if let scheme = config.prependScheme.string() {
            prependScheme = PrependScheme(rawValue: scheme) ?? .always
        } else {
            prependScheme = config.addPrefixSpace.boolean(or: true) ? .always : .never
        }
    }

    func rewrite(
        _ bytes: UnsafeBufferPointer<UInt8>, options: PreTokenizerOptions, into output: inout [UInt8],
        pieces: inout [Range<Int>]
    ) {
        // `NormalizedString::prepend` is a no-op on empty input.
        guard !bytes.isEmpty else { return }
        let base = output.count
        let needsPrefix: Bool
        switch prependScheme {
        case .always: needsPrefix = true
        case .first: needsPrefix = options.contains(.firstSection)
        case .never: needsPrefix = false
        }
        // The reference substitutes spaces first, then prepends unless the text already
        // starts with the replacement character.
        let startsWithReplacement =
            bytes.starts(with: replacementBytes)
            || (bytes[0] == 0x20 && stringReplacementBytes.starts(with: replacementBytes))
        let prefix = needsPrefix && !startsWithReplacement
        if fusesMarker {
            rewriteFused(bytes, prefix: prefix, into: &output, pieces: &pieces)
            return
        }
        if prefix {
            output.append(contentsOf: stringReplacementBytes)
        }
        if !substitutesSpace {
            output.append(contentsOf: bytes)
        } else {
            StringReplacePattern.replaceLiteral(bytes, pattern: [0x20], with: stringReplacementBytes, into: &output)
        }
        let length = output.count - base
        guard split else {
            pieces.append(0..<length)
            return
        }
        // Split on the replacement, merged with the following text.
        output.withUnsafeBufferPointer { output in
            let text = UnsafeBufferPointer(rebasing: output[base..<base + length])
            let marker = replacementBytes
            let m = marker.count
            var start = 0
            var i = 0
            while i + m <= length {
                if text[i] == marker[0], m == 1 || memcmp(text.baseAddress! + i, marker, m) == 0 {
                    if i > start { pieces.append(start..<i) }
                    start = i
                    i += m
                } else {
                    i += 1
                }
            }
            if start < length { pieces.append(start..<length) }
        }
    }

    /// Substitution and split in one pass for the usual configuration (`str_rep` equal to the
    /// multi-byte `replacement`, e.g. `▁`): a SIMD scan finds the next space or marker lead
    /// byte, the gap is copied, and every emitted marker starts a piece. Output goes straight
    /// into reserved storage instead of through one `append` per fragment.
    private func rewriteFused(
        _ bytes: UnsafeBufferPointer<UInt8>, prefix: Bool, into output: inout [UInt8], pieces: inout [Range<Int>]
    ) {
        let n = bytes.count
        let m = replacementBytes.count
        // Nothing to substitute (the rule after a `WhitespaceSplit` stage, and for most words):
        // copy the piece as it stands instead of reserving and re-trimming a worst-case buffer.
        if ByteKernels.firstIndex(of: 0x20, or: replacementBytes[0], in: bytes, from: 0) == n {
            let base = output.count
            if prefix { output.append(contentsOf: replacementBytes) }
            output.append(contentsOf: bytes)
            pieces.append(0..<(output.count - base))
            return
        }
        replacementBytes.withUnsafeBufferPointer { marker in
            let lead = marker[0]
            let split = self.split
            output.appendUninitialized(maximum: (prefix ? m : 0) + n * m) { out in
                var written = 0
                var pieceStart = 0
                @inline(__always) func emitMarker() {
                    if split, written > pieceStart {
                        pieces.append(pieceStart..<written)
                    }
                    if split { pieceStart = written }
                    (out + written).update(from: marker.baseAddress!, count: m)
                    written += m
                }
                if prefix { emitMarker() }
                var i = 0
                while i < n {
                    let j = ByteKernels.firstIndex(of: 0x20, or: lead, in: bytes, from: i)
                    if j > i {
                        (out + written).update(from: bytes.baseAddress! + i, count: j - i)
                        written += j - i
                        i = j
                    }
                    if i >= n { break }
                    if bytes[i] == 0x20 {
                        emitMarker()
                        i += 1
                    } else if i + m <= n, memcmp(bytes.baseAddress! + i, marker.baseAddress!, m) == 0 {
                        emitMarker()
                        i += m
                    } else {
                        out[written] = bytes[i]
                        written += 1
                        i += 1
                    }
                }
                if split {
                    if written > pieceStart { pieces.append(pieceStart..<written) }
                } else {
                    pieces.append(0..<written)
                }
                return written
            }
        }
    }
}

/// Byte-level pre-tokenizer: optionally prepends a space and splits with the GPT-2 regex,
/// then maps every byte to the byte-level alphabet.
final class ByteLevelPreTokenizer: StagedPreTokenizer {
    let addPrefixSpace: Bool
    let trimOffsets: Bool
    let useRegex: Bool

    required init(config: Config) {
        addPrefixSpace = config.addPrefixSpace.boolean(or: false)
        trimOffsets = config.trimOffsets.boolean(or: true)
        useRegex = config.useRegex.boolean(or: true)
    }

    var stages: [PreTokenizationStage] {
        var stages: [PreTokenizationStage] = []
        if addPrefixSpace { stages.append(.rewrite(PrefixSpaceRewriter())) }
        if useRegex { stages.append(.split(KnownSplitPattern.gpt2)) }
        stages.append(.byteLevel)
        return stages
    }

    /// `add_prefix_space`: a leading space unless the chunk already starts with one.
    struct PrefixSpaceRewriter: ByteRewriter {
        init() {}
        init(config: Config) { self.init() }

        func rewrite(
            _ bytes: UnsafeBufferPointer<UInt8>, options: PreTokenizerOptions, into output: inout [UInt8],
            pieces: inout [Range<Int>]
        ) {
            let base = output.count
            if bytes.first != UInt8(ascii: " ") { output.append(UInt8(ascii: " ")) }
            output.append(contentsOf: bytes)
            pieces.append(0..<(output.count - base))
        }
    }
}

extension KnownSplitPattern: ByteSplitter {
    init(config: Config) throws {
        guard let source = config.pattern.Regex.string(), let known = KnownSplitPattern(regexSource: source) else {
            throw TokenizerError.invalidConfiguration("Not a known split pattern")
        }
        self = known
    }

    func split(_ bytes: UnsafeBufferPointer<UInt8>, options: PreTokenizerOptions, into pieces: inout [Range<Int>]) {
        split(bytes, into: &pieces)
    }
}

/// Splits on punctuation (`P*` plus ASCII punctuation, as `tokenizers::is_punc`) according
/// to the configured `SplitDelimiterBehavior` (default `Isolated`, like `tokenizers`).
final class PunctuationPreTokenizer: ByteSplitter {
    enum Behavior: String {
        case removed = "Removed"
        case isolated = "Isolated"
        case mergedWithPrevious = "MergedWithPrevious"
        case mergedWithNext = "MergedWithNext"
        case contiguous = "Contiguous"
    }

    let behavior: Behavior

    required init(config: Config) {
        behavior = config.behavior.string().flatMap(Behavior.init(rawValue:)) ?? .isolated
    }

    func split(_ bytes: UnsafeBufferPointer<UInt8>, options: PreTokenizerOptions, into pieces: inout [Range<Int>]) {
        Self.splitRanges(bytes, behavior: behavior, into: &pieces)
    }

    /// Appends the byte ranges of the chunks `bytes` splits into.
    static func splitRanges(_ bytes: UnsafeBufferPointer<UInt8>, behavior: Behavior, into ranges: inout [Range<Int>]) {
        let end = bytes.count
        // Punctuation matches: single scalars, or runs when contiguous. Emitted per behaviour
        // as they are found so no intermediate array is needed.
        var cursor = 0
        var i = 0
        while i < end {
            let (value, w) = UTF8Cursor.decode(bytes, at: i)
            guard TokenizerUnicode.isPunctuation(value) else {
                i += w
                continue
            }
            var j = i + w
            if behavior == .contiguous {
                while j < end {
                    let (next, w2) = UTF8Cursor.decode(bytes, at: j)
                    if !TokenizerUnicode.isPunctuation(next) { break }
                    j += w2
                }
            }
            Self.emit(match: i..<j, behavior: behavior, cursor: &cursor, into: &ranges)
            i = j
        }
        if cursor < end { ranges.append(cursor..<end) }
    }

    /// Emits the pieces implied by a delimiter match at `match` for `behavior`, advancing `cursor`.
    @inline(__always)
    static func emit(match: Range<Int>, behavior: Behavior, cursor: inout Int, into ranges: inout [Range<Int>]) {
        switch behavior {
        case .isolated, .contiguous:
            if match.lowerBound > cursor { ranges.append(cursor..<match.lowerBound) }
            ranges.append(match)
            cursor = match.upperBound
        case .removed:
            if match.lowerBound > cursor { ranges.append(cursor..<match.lowerBound) }
            cursor = match.upperBound
        case .mergedWithPrevious:
            ranges.append(cursor..<match.upperBound)
            cursor = match.upperBound
        case .mergedWithNext:
            if match.lowerBound > cursor {
                ranges.append(cursor..<match.lowerBound)
                cursor = match.lowerBound
            }
        }
    }
}

/// `Digits`: isolates runs of numeric characters (`char::is_numeric`, i.e. `N*`), or every
/// numeric character when `individual_digits`.
final class DigitsPreTokenizer: ByteSplitter {
    let individualDigits: Bool

    required init(config: Config) {
        individualDigits = config.individualDigits.boolean(or: false)
    }

    func split(_ bytes: UnsafeBufferPointer<UInt8>, options: PreTokenizerOptions, into pieces: inout [Range<Int>]) {
        let end = bytes.count
        var cursor = 0
        var i = 0
        while i < end {
            let (value, width) = UTF8Cursor.decode(bytes, at: i)
            guard Self.isDigit(value) else {
                i += width
                continue
            }
            var j = i + width
            if !individualDigits {
                while j < end {
                    let (next, nextWidth) = UTF8Cursor.decode(bytes, at: j)
                    guard Self.isDigit(next) else { break }
                    j += nextWidth
                }
            }
            if i > cursor { pieces.append(cursor..<i) }
            pieces.append(i..<j)
            cursor = j
            i = j
        }
        if cursor < end { pieces.append(cursor..<end) }
    }

    /// Rust `char::is_numeric` (general categories Nd, Nl, No).
    @inline(__always)
    private static func isDigit(_ value: UInt32) -> Bool {
        if value < 0x80 { return value >= 0x30 && value <= 0x39 }
        switch Unicode.Scalar(value)?.properties.generalCategory {
        case .decimalNumber, .letterNumber, .otherNumber: return true
        default: return false
        }
    }
}

/// `Split`: a regex or literal delimiter with a `SplitDelimiterBehavior`. Well-known
/// byte-level regexes run through the hand-written scanners; literals are matched on raw
/// UTF-8; anything else goes through `NSRegularExpression` on a materialised string.
final class SplitPreTokenizer: ByteSplitter {
    let pattern: StringSplitPattern?
    let invert: Bool
    /// Set when the regex is one of the hand-optimised byte-level patterns.
    let known: KnownSplitPattern?
    /// `[0-9]` with `Isolated` behaviour (Falcon-H1): every ASCII digit becomes its own chunk.
    let asciiDigitsIsolated: Bool
    /// A literal (`pattern.String`) delimiter, split on raw UTF-8 with `behavior`.
    let literal: [UInt8]?
    let behavior: PunctuationPreTokenizer.Behavior
    private let fallbackRegex: NSRegularExpression?

    required init(config: Config) throws {
        pattern = try StringSplitPattern.from(config: config)
        invert = config.invert.boolean(or: false)
        behavior = config.behavior.string().flatMap(PunctuationPreTokenizer.Behavior.init(rawValue:)) ?? .isolated
        let source = config.pattern.Regex.string()
        if let source, !invert, behavior == .isolated {
            known = KnownSplitPattern(regexSource: source)
        } else {
            known = nil
        }
        asciiDigitsIsolated =
            source == "[0-9]" && !config.invert.boolean(or: false)
            && (config.behavior.string() ?? "Isolated") == "Isolated"
        if let string = config.pattern.String.string(), !string.isEmpty, !config.invert.boolean(or: false) {
            literal = Array(string.utf8)
        } else {
            literal = nil
        }
        if known == nil, !asciiDigitsIsolated, literal == nil, let pattern {
            switch pattern {
            case .regexp(let regex): fallbackRegex = regex
            case .string(let string):
                fallbackRegex = try compileRegex(
                    NSRegularExpression.escapedPattern(for: string), component: "Split pre-tokenizer")
            }
        } else {
            fallbackRegex = nil
        }
    }

    var stages: [PreTokenizationStage] {
        [.split(self)]
    }

    func split(_ bytes: UnsafeBufferPointer<UInt8>, options: PreTokenizerOptions, into pieces: inout [Range<Int>]) {
        if let known {
            known.split(bytes, into: &pieces)
        } else if asciiDigitsIsolated {
            Self.splitASCIIDigits(bytes, into: &pieces)
        } else if let literal {
            splitLiteral(bytes, literal, into: &pieces)
        } else if let fallbackRegex {
            splitRegex(bytes, regex: fallbackRegex, into: &pieces)
        } else {
            pieces.append(0..<bytes.count)
        }
    }

    /// Generic patterns partition the original text into matches and gaps. Inversion flips
    /// their flags, then delimiter behavior merges or removes partitions. Keeping ranges in
    /// the original buffer also preserves Metaspace's first-section check after removed gaps.
    private func splitRegex(
        _ bytes: UnsafeBufferPointer<UInt8>, regex: NSRegularExpression, into pieces: inout [Range<Int>]
    ) {
        guard !bytes.isEmpty else { return }
        let text = String(decoding: bytes, as: UTF8.self)
        // ICU reports UTF-16 offsets; convert all boundaries in one scalar pass.
        var byteOffsets = [Int](repeating: -1, count: text.utf16.count + 1)
        var utf16Offset = 0
        var byteOffset = 0
        for scalar in text.unicodeScalars {
            byteOffsets[utf16Offset] = byteOffset
            utf16Offset += scalar.value > 0xFFFF ? 2 : 1
            byteOffset += scalar.utf8.count
        }
        byteOffsets[utf16Offset] = byteOffset
        var partitions: [(range: Range<Int>, matched: Bool)] = []
        var cursor = 0
        for match in regex.matches(in: text, range: NSRange(location: 0, length: utf16Offset)) {
            let start = byteOffsets[match.range.location]
            let end = byteOffsets[NSMaxRange(match.range)]
            guard start >= 0, end >= 0 else { continue }
            if start > cursor { partitions.append((cursor..<start, invert)) }
            partitions.append((start..<end, !invert))
            cursor = end
        }
        if cursor < bytes.count { partitions.append((cursor..<bytes.count, invert)) }
        if behavior == .mergedWithNext { partitions.reverse() }
        var result: [Range<Int>] = []
        var previousMatch = false
        for (range, matched) in partitions {
            if behavior == .removed {
                if !matched { result.append(range) }
            } else if let last = result.last,
                (behavior == .contiguous && matched == previousMatch)
                    || ((behavior == .mergedWithPrevious || behavior == .mergedWithNext)
                        && matched && !previousMatch)
            {
                result[result.count - 1] =
                    min(last.lowerBound, range.lowerBound)..<max(last.upperBound, range.upperBound)
            } else {
                result.append(range)
            }
            previousMatch = matched
        }
        if behavior == .mergedWithNext { result.reverse() }
        pieces.append(contentsOf: result.filter { !$0.isEmpty })
    }

    /// Splits on a literal delimiter, byte-wise, honouring `behavior` exactly like
    /// `tokenizers::pre_tokenizers::Split` with a `String` pattern (gemma-4 uses
    /// `Split(" ", MergedWithPrevious)` after its space → `▁` normalizer).
    private func splitLiteral(_ bytes: UnsafeBufferPointer<UInt8>, _ literal: [UInt8], into pieces: inout [Range<Int>])
    {
        let end = bytes.count
        let m = literal.count
        let first = literal[0]
        var cursor = 0
        var i = 0
        @inline(__always)
        func matches(at p: Int) -> Bool {
            guard p + m <= end, bytes[p] == first else { return false }
            for k in 1..<m where bytes[p + k] != literal[k] { return false }
            return true
        }
        while i < end {
            guard matches(at: i) else {
                i += 1
                continue
            }
            var j = i + m
            if behavior == .contiguous {
                while matches(at: j) { j += m }
            }
            PunctuationPreTokenizer.emit(match: i..<j, behavior: behavior, cursor: &cursor, into: &pieces)
            i = j
        }
        if cursor < end { pieces.append(cursor..<end) }
    }

    /// Isolates ASCII digits. The byte-level alphabet maps `0`–`9` to themselves, so the split
    /// is identical on raw and on mapped text.
    static func splitASCIIDigits(_ bytes: UnsafeBufferPointer<UInt8>, into pieces: inout [Range<Int>]) {
        var start = 0
        for i in 0..<bytes.count where bytes[i] >= 0x30 && bytes[i] <= 0x39 {
            if i > start { pieces.append(start..<i) }
            pieces.append(i..<i + 1)
            start = i + 1
        }
        if start < bytes.count { pieces.append(start..<bytes.count) }
    }
}
