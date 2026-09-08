import Foundation

/// Options passed to pre-tokenization.
public enum PreTokenizerOption: String, Sendable {
    /// The text being processed is the first section (before any added token).
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

// MARK: - Fast path

/// A pre-tokenized chunk handed to the model.
struct PreToken {
    /// The chunk text. When `byteLevel` is `true` this is *raw* text whose UTF-8 bytes must
    /// each be mapped through the byte-level alphabet before vocabulary lookup.
    var text: Substring
    var byteLevel: Bool
}

/// Internal protocol implemented by pre-tokenizers that can emit chunks without building
/// intermediate `[String]` arrays.
protocol FastPreTokenizer: PreTokenizer {
    func preTokenize(_ text: Substring, options: PreTokenizerOptions, into output: inout [PreToken])
}

extension PreTokenizer {
    /// Fast-path entry point. Falls back to the `[String]` API for pre-tokenizers that do
    /// not implement `FastPreTokenizer`.
    func preTokenizeFast(_ text: Substring, options: PreTokenizerOptions, into output: inout [PreToken]) {
        if let fast = self as? any FastPreTokenizer {
            fast.preTokenize(text, options: options, into: &output)
        } else {
            for piece in preTokenize(text: String(text), options: options) {
                output.append(PreToken(text: Substring(piece), byteLevel: false))
            }
        }
    }
}

/// Materialises a byte-level chunk into its alphabet string (e.g. `" hi"` → `"Ġhi"`).
func materializeByteLevel(_ token: PreToken) -> String {
    guard token.byteLevel else { return String(token.text) }
    return ByteLevelAlphabet.encode(token.text.utf8)
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
    static func fromConfig(config: Config?) throws -> (any PreTokenizer)? {
        guard let config, let typeName = config.type.string() else { return nil }
        switch PreTokenizerType(rawValue: typeName) {
        case .Sequence: return try PreTokenizerSequence(config: config)
        case .ByteLevel: return ByteLevelPreTokenizer(config: config)
        case .Punctuation: return PunctuationPreTokenizer(config: config)
        case .Digits: return DigitsPreTokenizer(config: config)
        case .Split: return try SplitPreTokenizer(config: config)
        case .Whitespace, .WhitespaceSplit: return WhitespacePreTokenizer(config: config)
        case .Metaspace: return MetaspacePreTokenizer(config: config)
        case .BertPreTokenizer: return BertPreTokenizer(config: config)
        default: throw TokenizerError.unsupportedComponent("pre-tokenizer `\(typeName)`")
        }
    }
}

// MARK: - Shared regexes

// These patterns are compile-time constants exercised by the test-suite, so force-unwrapping
// their compilation cannot fail at runtime.

private let punctuationClass = #"\p{P}\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"#

private let digitsIndividualRegex: NSRegularExpression = try! NSRegularExpression(pattern: "[^\\d]+|\\d")
private let digitsGroupedRegex: NSRegularExpression = try! NSRegularExpression(pattern: "[^\\d]+|\\d+")

/// Returns every match of `regex` in `text`.
func splitMatches(in text: String, with regex: NSRegularExpression) -> [String] {
    let ns = text as NSString
    var result: [String] = []
    regex.enumerateMatches(in: text, range: NSRange(location: 0, length: ns.length)) { match, _, _ in
        guard let match else { return }
        result.append(ns.substring(with: match.range))
    }
    return result
}

/// Calls `body` with each match of `regex` in `text`.
func enumerateRegexTokens(in text: String, with regex: NSRegularExpression, _ body: (String) -> Void) {
    let ns = text as NSString
    withoutActuallyEscaping(body) { escapable in
        regex.enumerateMatches(in: text, range: NSRange(location: 0, length: ns.length)) { match, _, _ in
            guard let match else { return }
            escapable(ns.substring(with: match.range))
        }
    }
}

// MARK: - Implementations

final class BertPreTokenizer: PreTokenizer, FastPreTokenizer {
    required init(config: Config) {}

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        var output: [PreToken] = []
        preTokenize(Substring(text), options: options, into: &output)
        return output.map { String($0.text) }
    }

    func preTokenize(_ text: Substring, options: PreTokenizerOptions, into output: inout [PreToken]) {
        var copy = text
        copy.withUTF8 { bytes in
            var i = 0
            var start = 0
            func append(_ lo: Int, _ hi: Int) {
                guard lo < hi else { return }
                let utf8 = text.utf8
                let lower = utf8.index(utf8.startIndex, offsetBy: lo)
                let upper = utf8.index(utf8.startIndex, offsetBy: hi)
                output.append(PreToken(text: text[lower..<upper], byteLevel: false))
            }
            while i < bytes.count {
                let (value, width) = UTF8Cursor.decode(bytes, at: i)
                let whitespace = ScalarClassifier.classify(value: value) == .whitespace
                let punctuation: Bool
                if value < 128 {
                    punctuation =
                        (33...47).contains(value) || (58...64).contains(value)
                        || (91...96).contains(value) || (123...126).contains(value)
                } else {
                    punctuation = CharacterSet.punctuationCharacters.contains(Unicode.Scalar(value)!)
                }
                if whitespace || punctuation {
                    append(start, i)
                    if punctuation { append(i, i + width) }
                    start = i + width
                }
                i += width
            }
            append(start, bytes.count)
        }
    }
}

final class PreTokenizerSequence: PreTokenizer, FastPreTokenizer {
    let preTokenizers: [any PreTokenizer]

    /// How each stage after the first is applied on the fast path, resolved once so the
    /// per-chunk loop performs no dynamic casts.
    private enum Stage {
        /// `ByteLevel` without regex: only flags chunks.
        case markByteLevel(ByteLevelPreTokenizer)
        /// Known byte-level regex: scanner applied per chunk with a shared scratch buffer.
        case knownSplit(KnownSplitPattern)
        /// `[0-9]` isolation (Falcon-H1).
        case asciiDigits(SplitPreTokenizer)
        case punctuation(PunctuationPreTokenizer)
        case fast(any FastPreTokenizer)
        case generic(any PreTokenizer)
    }
    private let stages: [Stage]

    required init(config: Config) throws {
        let configs = try require(config.pretokenizers.array(), "Sequence pre-tokenizer", field: "pretokenizers")
        preTokenizers = try configs.compactMap { try PreTokenizerFactory.fromConfig(config: $0) }
        stages = preTokenizers.dropFirst().map { stage in
            if let byteLevel = stage as? ByteLevelPreTokenizer, !byteLevel.useRegex { return .markByteLevel(byteLevel) }
            if let split = stage as? SplitPreTokenizer {
                if let known = split.known { return .knownSplit(known) }
                if split.asciiDigitsIsolated { return .asciiDigits(split) }
            }
            if let punctuation = stage as? PunctuationPreTokenizer { return .punctuation(punctuation) }
            if let fast = stage as? any FastPreTokenizer { return .fast(fast) }
            return .generic(stage)
        }
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        var current = [text]
        for preTokenizer in preTokenizers {
            current = preTokenizer.preTokenize(texts: current, options: options)
        }
        return current
    }

    func preTokenize(_ text: Substring, options: PreTokenizerOptions, into output: inout [PreToken]) {
        guard let first = preTokenizers.first else {
            output.append(PreToken(text: text, byteLevel: false))
            return
        }
        if preTokenizers.count == 1 {
            first.preTokenizeFast(text, options: options, into: &output)
            return
        }

        var current: [PreToken] = []
        first.preTokenizeFast(text, options: options, into: &current)
        var next: [PreToken] = []
        var scratch: [Range<Int>] = []
        for stage in stages {
            next.removeAll(keepingCapacity: true)
            next.reserveCapacity(current.count)
            switch stage {
            case let .markByteLevel(byteLevel):
                // Common tail stage (Llama 3, Qwen…): only flags chunks as byte-level.
                for token in current {
                    byteLevel.markByteLevel(token, into: &next)
                }
            case let .knownSplit(known):
                for token in current {
                    if token.byteLevel {
                        Self.materializedStage(token, options: options, into: &next) { text, options, next in
                            known.split(text, scratch: &scratch, byteLevel: false, into: &next)
                        }
                    } else {
                        known.split(token.text, scratch: &scratch, byteLevel: false, into: &next)
                    }
                }
            case let .asciiDigits(split):
                // Falcon-H1: digit isolation after the byte-level stage, without materialising.
                for token in current {
                    split.splitASCIIDigits(token, into: &next)
                }
            case let .punctuation(punctuation):
                for token in current {
                    punctuation.split(token, into: &next)
                }
            case let .fast(fast):
                for token in current {
                    if token.byteLevel {
                        Self.materializedStage(token, options: options, into: &next) { text, options, next in
                            fast.preTokenize(text, options: options, into: &next)
                        }
                    } else {
                        fast.preTokenize(token.text, options: options, into: &next)
                    }
                }
            case let .generic(stage):
                for token in current {
                    let text = token.byteLevel ? materializeByteLevel(token) : String(token.text)
                    for piece in stage.preTokenize(text: text, options: options) {
                        next.append(PreToken(text: Substring(piece), byteLevel: false))
                    }
                }
            }
            swap(&current, &next)
        }
        output.append(contentsOf: current)
    }

    /// Rare: a byte-level chunk feeding a further stage. Materialise it into the alphabet
    /// string and run the stage on that (the results are then plain text, like `tokenizers`).
    @inline(never)
    private static func materializedStage(
        _ token: PreToken, options: PreTokenizerOptions, into next: inout [PreToken],
        _ body: (Substring, PreTokenizerOptions, inout [PreToken]) -> Void
    ) {
        let materialized = materializeByteLevel(token)
        body(Substring(materialized), options, &next)
    }
}

/// `Whitespace` / `WhitespaceSplit`: runs of non-whitespace scalars (`\S+`).
final class WhitespacePreTokenizer: PreTokenizer, FastPreTokenizer {
    let splitWords: Bool
    required init(config: Config) { splitWords = config.type.string() == "Whitespace" }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        var tokens: [PreToken] = []
        preTokenize(Substring(text), options: options, into: &tokens)
        return tokens.map { String($0.text) }
    }

    func preTokenize(_ text: Substring, options: PreTokenizerOptions, into output: inout [PreToken]) {
        var copy = text
        var ranges: [Range<Int>] = []
        copy.withUTF8 { bytes in
            let n = bytes.count
            var i = 0
            var start = -1
            var previousWord = false
            while i < n {
                let (value, width) = UTF8Cursor.decode(bytes, at: i)
                let isWhitespace = ScalarClassifier.classify(value: value) == .whitespace
                let isWord = Self.isWord(value)
                if isWhitespace {
                    if start >= 0 { ranges.append(start..<i); start = -1 }
                } else {
                    if start >= 0, splitWords, isWord != previousWord {
                        ranges.append(start..<i)
                        start = -1
                    }
                    if start < 0 { start = i }
                    previousWord = isWord
                }
                i += width
            }
            if start >= 0 { ranges.append(start..<n) }
        }
        let utf8 = text.utf8
        for range in ranges {
            let lower = utf8.index(utf8.startIndex, offsetBy: range.lowerBound)
            let upper = utf8.index(utf8.startIndex, offsetBy: range.upperBound)
            output.append(PreToken(text: text[lower..<upper], byteLevel: false))
        }
    }
    private static func isWord(_ value: UInt32) -> Bool {
        if value < 128 {
            return (65...90).contains(value) || (97...122).contains(value) || (48...57).contains(value) || value == 95
        }
        guard let scalar = Unicode.Scalar(value) else { return false }
        if scalar.properties.isAlphabetic || value == 0x200C || value == 0x200D { return true }
        switch scalar.properties.generalCategory {
        case .nonspacingMark, .spacingMark, .enclosingMark, .decimalNumber, .connectorPunctuation: return true
        default: return false
        }
    }

}

/// Replaces spaces with a replacement character (SentencePiece's `▁`) and optionally
/// prepends it, then splits so every chunk starts with the replacement.
final class MetaspacePreTokenizer: PreTokenizer, FastPreTokenizer {
    let addPrefixSpace: Bool
    let replacement: String
    let stringReplacement: String

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

        // `prepend_scheme` supersedes `add_prefix_space` (tokenizers PR #1357).
        if let scheme = config.prependScheme.string() {
            prependScheme = PrependScheme(rawValue: scheme) ?? .always
        } else {
            prependScheme = config.addPrefixSpace.boolean(or: true) ? .always : .never
        }
    }

    private func transformed(_ text: Substring, options: PreTokenizerOptions) -> String {
        // `NormalizedString::prepend` is a no-op on empty input.
        guard !text.isEmpty else { return "" }
        var normalized: String
        if stringReplacement == " " {
            normalized = String(text)
        } else {
            normalized = text.replacingBytes(of: " ", with: stringReplacement)
        }
        if !normalized.hasBytePrefix(replacement) {
            switch prependScheme {
            case .always:
                normalized = stringReplacement + normalized
            case .first:
                if options.contains(.firstSection) {
                    normalized = stringReplacement + normalized
                }
            case .never:
                break
            }
        }
        return normalized
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        let normalized = transformed(Substring(text), options: options)
        guard split else { return normalized.isEmpty ? [] : [normalized] }
        return normalized.split(by: replacement, behavior: .mergedWithNext)
    }

    func preTokenize(_ text: Substring, options: PreTokenizerOptions, into output: inout [PreToken]) {
        guard !text.isEmpty else { return }

        // Fast path: a chunk without spaces or replacement characters (the common case after a
        // whitespace split) is emitted as-is, or with the prefix prepended in one allocation.
        var copy = text
        let replacementBytes = Array(replacement.utf8)
        let trivial = copy.withUTF8 { bytes -> Bool in
            for (i, b) in bytes.enumerated() {
                if b == 0x20 { return false }
                if b == replacementBytes[0], bytes[i...].starts(with: replacementBytes) { return false }
            }
            return true
        }
        if trivial {
            let needsPrefix: Bool
            switch prependScheme {
            case .always: needsPrefix = true
            case .first: needsPrefix = options.contains(.firstSection)
            case .never: needsPrefix = false
            }
            if needsPrefix {
                var prefixed = stringReplacement
                prefixed.reserveCapacity(stringReplacement.utf8.count + text.utf8.count)
                prefixed.append(contentsOf: text)
                output.append(PreToken(text: Substring(prefixed), byteLevel: false))
            } else {
                output.append(PreToken(text: text, byteLevel: false))
            }
            return
        }

        let normalized = transformed(text, options: options)
        guard split else {
            if !normalized.isEmpty { output.append(PreToken(text: Substring(normalized), byteLevel: false)) }
            return
        }
        for piece in Substring(normalized).splitMergedWithNext(separator: replacement) {
            output.append(PreToken(text: piece, byteLevel: false))
        }
    }
}

/// Byte-level pre-tokenizer: optionally splits with the GPT-2 regex and maps every byte to
/// the byte-level alphabet.
final class ByteLevelPreTokenizer: PreTokenizer, FastPreTokenizer {
    let addPrefixSpace: Bool
    let trimOffsets: Bool
    let useRegex: Bool

    required init(config: Config) {
        addPrefixSpace = config.addPrefixSpace.boolean(or: false)
        trimOffsets = config.trimOffsets.boolean(or: true)
        useRegex = config.useRegex.boolean(or: true)
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        var tokens: [PreToken] = []
        preTokenize(Substring(text), options: options, into: &tokens)
        return tokens.map { materializeByteLevel($0) }
    }

    func preTokenize(_ text: Substring, options: PreTokenizerOptions, into output: inout [PreToken]) {
        guard useRegex else {
            markByteLevel(PreToken(text: text, byteLevel: false), into: &output)
            return
        }
        var pieces: [Substring] = []
        pieces.reserveCapacity(text.utf8.count / 4 + 1)
        KnownSplitPattern.gpt2.split(text, into: &pieces)
        for piece in pieces {
            markByteLevel(PreToken(text: piece, byteLevel: false), into: &output)
        }
    }

    /// Flags a chunk as byte-level, applying `add_prefix_space` if configured.
    @inline(__always)
    func markByteLevel(_ token: PreToken, into output: inout [PreToken]) {
        if token.byteLevel {
            output.append(token)
            return
        }
        if addPrefixSpace, token.text.utf8.first != UInt8(ascii: " ") {
            output.append(PreToken(text: Substring(" " + token.text), byteLevel: true))
        } else {
            output.append(PreToken(text: token.text, byteLevel: true))
        }
    }
}

/// Splits on punctuation (`P*` plus ASCII punctuation, as `tokenizers::is_punc`) according
/// to the configured `SplitDelimiterBehavior` (default `Isolated`, like `tokenizers`).
final class PunctuationPreTokenizer: PreTokenizer, FastPreTokenizer {
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

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        var tokens: [PreToken] = []
        preTokenize(Substring(text), options: options, into: &tokens)
        return tokens.map { String($0.text) }
    }

    func preTokenize(_ text: Substring, options: PreTokenizerOptions, into output: inout [PreToken]) {
        split(PreToken(text: text, byteLevel: false), into: &output)
    }

    /// Splits a chunk, preserving its byte-level flag (punctuation classification is by
    /// scalar, and the byte-level alphabet maps ASCII punctuation to itself).
    func split(_ token: PreToken, into output: inout [PreToken]) {
        var copy = token.text
        var ranges: [Range<Int>] = []
        copy.withUTF8 { bytes in
            Self.splitRanges(bytes, behavior: behavior, into: &ranges)
        }
        if ranges.count == 1 {
            output.append(token)
        } else {
            appendByteRanges(ranges, of: token, into: &output)
        }
    }

    /// Appends the byte ranges of the chunks `bytes` splits into.
    static func splitRanges(_ bytes: UnsafeBufferPointer<UInt8>, behavior: Behavior, into ranges: inout [Range<Int>]) {
        ScalarClassifier.bmp.withUnsafeBufferPointer { table in
            let end = bytes.count
            // Punctuation matches: single scalars, or runs when contiguous. Emitted per behaviour
            // as they are found so no intermediate array is needed.
            var cursor = 0
            var i = 0
            while i < end {
                let (_, w, f) = ByteLevelScanner.decodeClassified(bytes, i, table)
                guard f & ScalarFlags.punctuation != 0 else {
                    i += w
                    continue
                }
                var j = i + w
                if behavior == .contiguous {
                    while j < end {
                        let (_, w2, g) = ByteLevelScanner.decodeClassified(bytes, j, table)
                        if g & ScalarFlags.punctuation == 0 { break }
                        j += w2
                    }
                }
                switch behavior {
                case .isolated, .contiguous:
                    if i > cursor { ranges.append(cursor..<i) }
                    ranges.append(i..<j)
                    cursor = j
                case .removed:
                    if i > cursor { ranges.append(cursor..<i) }
                    cursor = j
                case .mergedWithPrevious:
                    ranges.append(cursor..<j)
                    cursor = j
                case .mergedWithNext:
                    if i > cursor {
                        ranges.append(cursor..<i)
                        cursor = i
                    }
                }
                i = j
            }
            if cursor < end { ranges.append(cursor..<end) }
        }
    }
}

/// Appends the sub-chunks of `token` delimited by `ranges` (UTF-8 offsets).
@inline(__always)
func appendByteRanges(_ ranges: [Range<Int>], of token: PreToken, into output: inout [PreToken]) {
    let utf8 = token.text.utf8
    let base = utf8.startIndex
    output.reserveCapacity(output.count + ranges.count)
    for range in ranges where !range.isEmpty {
        let lower = utf8.index(base, offsetBy: range.lowerBound)
        let upper = utf8.index(base, offsetBy: range.upperBound)
        output.append(PreToken(text: token.text[lower..<upper], byteLevel: token.byteLevel))
    }
}

final class DigitsPreTokenizer: PreTokenizer {
    let regex: NSRegularExpression

    required init(config: Config) {
        regex = config.individualDigits.boolean(or: false) ? digitsIndividualRegex : digitsGroupedRegex
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        splitMatches(in: text, with: regex)
    }
}

final class SplitPreTokenizer: PreTokenizer, FastPreTokenizer {
    let pattern: StringSplitPattern?
    let invert: Bool
    /// Set when the regex is one of the hand-optimised byte-level patterns.
    let known: KnownSplitPattern?
    /// `[0-9]` with `Isolated` behaviour (Falcon-H1): every ASCII digit becomes its own chunk.
    let asciiDigitsIsolated: Bool
    /// A literal (`pattern.String`) delimiter, split on raw UTF-8 with `behavior`.
    let literal: [UInt8]?
    let behavior: PunctuationPreTokenizer.Behavior

    required init(config: Config) throws {
        pattern = try StringSplitPattern.from(config: config)
        invert = config.invert.boolean(or: false)
        behavior = config.behavior.string().flatMap(PunctuationPreTokenizer.Behavior.init(rawValue:)) ?? .isolated
        let source = config.pattern.Regex.string()
        if let source, !config.invert.boolean(or: false) {
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
    }

    func preTokenize(text: String, options: PreTokenizerOptions = [.firstSection]) -> [String] {
        if let known {
            var pieces: [Substring] = []
            known.split(Substring(text), into: &pieces)
            return pieces.map(String.init)
        }
        if asciiDigitsIsolated || literal != nil {
            var tokens: [PreToken] = []
            preTokenize(Substring(text), options: options, into: &tokens)
            return tokens.map { String($0.text) }
        }
        guard let pattern else { return [text] }
        return pattern.split(text, invert: invert)
    }

    /// Splits on a literal delimiter, byte-wise, honouring `behavior` exactly like
    /// `tokenizers::pre_tokenizers::Split` with a `String` pattern (gemma-4 uses
    /// `Split(" ", MergedWithPrevious)` after its space → `▁` normalizer).
    func splitLiteral(_ token: PreToken, _ literal: [UInt8], into output: inout [PreToken]) {
        var copy = token.text
        var ranges: [Range<Int>] = []
        copy.withUTF8 { bytes in
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
                switch behavior {
                case .isolated, .contiguous:
                    if i > cursor { ranges.append(cursor..<i) }
                    ranges.append(i..<j)
                    cursor = j
                case .removed:
                    if i > cursor { ranges.append(cursor..<i) }
                    cursor = j
                case .mergedWithPrevious:
                    ranges.append(cursor..<j)
                    cursor = j
                case .mergedWithNext:
                    if i > cursor {
                        ranges.append(cursor..<i)
                        cursor = i
                    }
                }
                i = j
            }
            if cursor < end { ranges.append(cursor..<end) }
        }
        if ranges.count == 1, ranges[0].count == token.text.utf8.count {
            output.append(token)
        } else {
            appendByteRanges(ranges, of: token, into: &output)
        }
    }

    /// Isolates ASCII digits, preserving the byte-level flag: the byte-level alphabet maps
    /// `0`–`9` to themselves, so the split is identical on raw and on mapped text.
    func splitASCIIDigits(_ token: PreToken, into output: inout [PreToken]) {
        var copy = token.text
        let firstDigit: Int? = copy.withUTF8 { bytes in
            for i in 0..<bytes.count where bytes[i] >= 0x30 && bytes[i] <= 0x39 { return i }
            return nil
        }
        guard let firstDigit else {
            // Common case: no digits at all.
            output.append(token)
            return
        }
        let utf8 = token.text.utf8
        let base = utf8.startIndex
        var start = 0
        copy.withUTF8 { bytes in
            @inline(__always)
            func emit(_ lo: Int, _ hi: Int) {
                output.append(
                    PreToken(
                        text: token.text[utf8.index(base, offsetBy: lo)..<utf8.index(base, offsetBy: hi)],
                        byteLevel: token.byteLevel))
            }
            for i in firstDigit..<bytes.count where bytes[i] >= 0x30 && bytes[i] <= 0x39 {
                if i > start { emit(start, i) }
                emit(i, i + 1)
                start = i + 1
            }
            if start < bytes.count { emit(start, bytes.count) }
        }
    }

    func preTokenize(_ text: Substring, options: PreTokenizerOptions, into output: inout [PreToken]) {
        if asciiDigitsIsolated {
            splitASCIIDigits(PreToken(text: text, byteLevel: false), into: &output)
            return
        }
        if let literal {
            splitLiteral(PreToken(text: text, byteLevel: false), literal, into: &output)
            return
        }
        if let known {
            var pieces: [Substring] = []
            pieces.reserveCapacity(text.utf8.count / 4 + 1)
            known.split(text, into: &pieces)
            for piece in pieces {
                output.append(PreToken(text: piece, byteLevel: false))
            }
            return
        }
        for piece in preTokenize(text: String(text), options: options) {
            output.append(PreToken(text: Substring(piece), byteLevel: false))
        }
    }
}
