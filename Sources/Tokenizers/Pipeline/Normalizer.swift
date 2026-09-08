import Foundation

/// Text normalization applied before pre-tokenization (lowercasing, Unicode normalization,
/// accent stripping, …).
public protocol Normalizer: Sendable {
    func normalize(text: String) -> String
    func callAsFunction(text: String) -> String
    /// Creates the normalizer from its `tokenizer.json` entry.
    /// - Throws: ``TokenizerError/invalidConfiguration(_:)`` when required fields are missing.
    init(config: Config) throws
}

public extension Normalizer {
    func callAsFunction(text: String) -> String { normalize(text: text) }
}

enum NormalizerType: String {
    case Sequence
    case Prepend
    case Replace
    case Lowercase
    case NFD
    case NFC
    case NFKD
    case NFKC
    case Bert
    case BertNormalizer
    case Precompiled
    case StripAccents
    case Strip
    case Unknown = ""
}

struct NormalizerFactory {
    static func fromConfig(config: Config?) throws -> (any Normalizer)? {
        guard let config, let typeName = config.type.string() else { return nil }
        switch NormalizerType(rawValue: typeName) {
        case .Sequence: return try NormalizerSequence(config: config)
        case .Prepend: return PrependNormalizer(config: config)
        case .Replace: return try ReplaceNormalizer(config: config)
        case .Lowercase: return LowercaseNormalizer(config: config)
        case .NFD: return NFDNormalizer(config: config)
        case .NFC: return NFCNormalizer(config: config)
        case .NFKD: return NFKDNormalizer(config: config)
        case .NFKC: return NFKCNormalizer(config: config)
        case .Bert, .BertNormalizer: return BertNormalizer(config: config)
        case .Precompiled: return try PrecompiledNormalizer(config: config)
        case .StripAccents: return StripAccentsNormalizer(config: config)
        case .Strip: return StripNormalizer(config: config)
        default: throw TokenizerError.unsupportedComponent("normalizer `\(typeName)`")
        }
    }
}

final class NormalizerSequence: Normalizer {
    let normalizers: [any Normalizer]

    required init(config: Config) throws {
        let configs = try require(config.normalizers.array(), "Sequence normalizer", field: "normalizers")
        normalizers = try configs.compactMap { try NormalizerFactory.fromConfig(config: $0) }
    }

    func normalize(text: String) -> String {
        var current = text
        for normalizer in normalizers {
            current = normalizer.normalize(text: current)
        }
        return current
    }
}

final class PrependNormalizer: Normalizer {
    let prepend: String

    required init(config: Config) {
        prepend = config.prepend.string(or: "")
    }

    func normalize(text: String) -> String { text.isEmpty ? text : prepend + text }
}

final class ReplaceNormalizer: Normalizer {
    let pattern: StringReplacePattern?

    required init(config: Config) throws {
        pattern = try StringReplacePattern.from(config: config)
    }

    func normalize(text: String) -> String {
        guard let pattern else { return text }
        return pattern.replace(text)
    }
}

final class LowercaseNormalizer: Normalizer {
    required init(config: Config) {}
    func normalize(text: String) -> String { text.lowercased() }
}

final class NFDNormalizer: Normalizer {
    required init(config: Config) {}
    func normalize(text: String) -> String { text.decomposedStringWithCanonicalMapping }
}

final class NFCNormalizer: Normalizer {
    required init(config: Config) {}
    func normalize(text: String) -> String { text.precomposedStringWithCanonicalMapping }
}

final class NFKDNormalizer: Normalizer {
    required init(config: Config) {}
    func normalize(text: String) -> String { text.decomposedStringWithCompatibilityMapping }
}

final class NFKCNormalizer: Normalizer {
    required init(config: Config) {}
    func normalize(text: String) -> String { text.precomposedStringWithCompatibilityMapping }
}

final class BertNormalizer: Normalizer {
    let shouldCleanText: Bool
    let shouldHandleChineseChars: Bool
    let shouldStripAccents: Bool
    let shouldLowercase: Bool

    required init(config: Config) {
        shouldCleanText = config.cleanText.boolean(or: true)
        shouldHandleChineseChars = config.handleChineseChars.boolean(or: true)
        shouldLowercase = config.lowercase.boolean(or: true)
        shouldStripAccents = config.stripAccents.boolean(or: shouldLowercase)
    }

    func normalize(text: String) -> String {
        var copy = text
        let isASCII = copy.withUTF8 { bytes in bytes.allSatisfy { $0 < 0x80 } }
        if isASCII {
            return normalizeASCII(text)
        }
        var output = text
        if shouldCleanText || shouldHandleChineseChars {
            var bytes: [UInt8] = []
            bytes.reserveCapacity(text.utf8.count + 16)
            for scalar in text.unicodeScalars {
                if shouldCleanText {
                    if scalar.value == 0 || scalar.value == 0xFFFD || isControl(scalar) { continue }
                    if isWhitespace(scalar) { bytes.append(32); continue }
                }
                let chinese = shouldHandleChineseChars && scalar.isCJKUnifiedIdeograph
                if chinese { bytes.append(32) }
                bytes.append(contentsOf: scalar.utf8)
                if chinese { bytes.append(32) }
            }
            output = String(decoding: bytes, as: UTF8.self)
        }
        if shouldStripAccents { output = Self.stripAccents(output) }
        if shouldLowercase { output = output.lowercased() }
        return output
    }

    /// ASCII has no CJK ideographs or combining marks, so only cleaning and lowercasing apply.
    private func normalizeASCII(_ text: String) -> String {
        var copy = text
        return copy.withUTF8 { bytes -> String in
            var out: [UInt8] = []
            out.reserveCapacity(bytes.count)
            for b in bytes {
                var byte = b
                if shouldCleanText {
                    switch byte {
                    case 0x09, 0x0A, 0x0D: byte = 0x20
                    case 0x00...0x1F, 0x7F: continue
                    default: break
                    }
                }
                if shouldLowercase, byte >= 0x41, byte <= 0x5A { byte |= 0x20 }
                out.append(byte)
            }
            return String(decoding: out, as: UTF8.self)
        }
    }

    private func isWhitespace(_ c: Unicode.Scalar) -> Bool {
        if c.value == 0x09 || c.value == 0x0A || c.value == 0x0D { return true }
        if c.value < 0x80 { return c.value == 0x20 }
        return c.properties.generalCategory == .spaceSeparator
    }

    /// Cc/Cf/Cs/Co (but not unassigned code points, which `tokenizers` leaves untouched).
    private func isControl(_ c: Unicode.Scalar) -> Bool {
        if c.value == 0x09 || c.value == 0x0A || c.value == 0x0D { return false }
        if c.value < 0x80 { return c.value < 0x20 || c.value == 0x7F }
        switch c.properties.generalCategory {
        case .control, .format, .surrogate, .privateUse: return true
        default: return false
        }
    }

    /// NFD-decompose then drop every nonspacing mark (general category Mn), matching HF's
    /// `_run_strip_accents`.
    static func stripAccents(_ text: String) -> String {
        let decomposed = text.decomposedStringWithCanonicalMapping
        var hasMark = false
        for scalar in decomposed.unicodeScalars
        where scalar.value >= 0x300 && scalar.properties.generalCategory == .nonspacingMark {
            hasMark = true
            break
        }
        guard hasMark else { return decomposed }
        var bytes: [UInt8] = []
        bytes.reserveCapacity(decomposed.utf8.count)
        for scalar in decomposed.unicodeScalars where scalar.properties.generalCategory != .nonspacingMark {
            bytes.append(contentsOf: scalar.utf8)
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}

extension Unicode.Scalar {
    /// https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    var isCJKUnifiedIdeograph: Bool {
        (value >= 0x4E00 && value <= 0x9FFF)
            || (value >= 0x3400 && value <= 0x4DBF)
            || (value >= 0x20000 && value <= 0x2A6DF)
            || (value >= 0x2A700 && value <= 0x2B73F)
            || (value >= 0x2B740 && value <= 0x2B81F)
            || (value >= 0x2B820 && value <= 0x2CEAF)
            || (value >= 0xF900 && value <= 0xFAFF)
            || (value >= 0x2F800 && value <= 0x2FA1F)
    }
}

/// SentencePiece's serialized Darts map, following spm_precompiled 0.1.3.
/// The map is model data: substituting Foundation NFKC changes token IDs.
final class PrecompiledNormalizer: Normalizer {
    private let trie: [UInt32]
    private let replacements: [UInt8]
    private let asciiReplacements: [[UInt8]?]
    private let crlfReplacement: [UInt8]?

    required init(config: Config) throws {
        guard let encoded = config.precompiledCharsmap.string(), let data = Data(base64Encoded: encoded),
            data.count >= 8
        else {
            throw TokenizerError.invalidConfiguration("Precompiled normalizer requires a valid base64 charsmap")
        }
        let bytes = Array(data)
        func uint32(at i: Int) -> UInt32 {
            UInt32(bytes[i]) | UInt32(bytes[i + 1]) << 8 | UInt32(bytes[i + 2]) << 16 | UInt32(bytes[i + 3]) << 24
        }
        let size = Int(uint32(at: 0))
        guard size >= 4, size % 4 == 0, size <= bytes.count - 4 else {
            throw TokenizerError.invalidConfiguration("Invalid Precompiled trie size")
        }
        trie = stride(from: 4, to: size + 4, by: 4).map { uint32(at: $0) }
        replacements = Array(bytes[(size + 4)...])
        guard replacements.last == 0, String(bytes: replacements, encoding: .utf8) != nil else {
            throw TokenizerError.invalidConfiguration("Invalid Precompiled replacement table")
        }
        guard Self.offset(trie[0]) < trie.count else {
            throw TokenizerError.invalidConfiguration("Invalid Precompiled root offset")
        }
        let trie = self.trie
        let replacements = self.replacements
        asciiReplacements = (0..<128).map { value in
            Self.replacement(for: [UInt8(value)], trie: trie, replacements: replacements).map(Array.init)
        }
        crlfReplacement = Self.replacement(for: [UInt8(13), 10], trie: trie, replacements: replacements).map(Array.init)
    }

    @inline(__always) private static func offset(_ unit: UInt32) -> Int {
        Int(unit >> 10) << Int((unit & (1 << 9)) >> 6)
    }

    /// Returns the first prefix match, as the reference does (not the longest match).
    private static func replacement<C: Collection>(
        for bytes: C, trie: [UInt32], replacements: [UInt8]
    ) -> ArraySlice<UInt8>? where C.Element == UInt8 {
        var node = Self.offset(trie[0])
        for byte in bytes {
            if byte == 0 { break }
            node ^= Int(byte)
            guard node < trie.count else { return nil }
            let unit = trie[node]
            guard unit & 0x800000ff == UInt32(byte) else { return nil }
            node ^= Self.offset(unit)
            if unit & 0x100 != 0 {
                guard node < trie.count else { return nil }
                let start = Int(trie[node] & 0x7fffffff)
                guard start < replacements.count else { return nil }
                var end = start
                while end < replacements.count, replacements[end] != 0 { end += 1 }
                return replacements[start..<end]
            }
        }
        return nil
    }

    private func replacement<C: Collection>(for bytes: C) -> ArraySlice<UInt8>? where C.Element == UInt8 {
        Self.replacement(for: bytes, trie: trie, replacements: replacements)
    }

    func normalize(text: String) -> String {
        // Most prompts are ASCII. Reuse model-specific mappings without constructing a
        // Character/String per scalar; unchanged text needs no output allocation.
        var input = text
        if let ascii = input.withUTF8({ bytes -> String? in
            guard bytes.allSatisfy({ $0 < 128 }) else { return nil }
            guard bytes.contains(where: { asciiReplacements[Int($0)] != nil }) else { return text }
            var output: [UInt8] = []
            output.reserveCapacity(bytes.count)
            var i = 0
            while i < bytes.count {
                if bytes[i] == 13, i + 1 < bytes.count, bytes[i + 1] == 10, let crlfReplacement {
                    output.append(contentsOf: crlfReplacement)
                    i += 2
                } else {
                    if let mapped = asciiReplacements[Int(bytes[i])] {
                        output.append(contentsOf: mapped)
                    } else {
                        output.append(bytes[i])
                    }
                    i += 1
                }
            }
            return String(decoding: output, as: UTF8.self)
        }) {
            return ascii
        }
        var output: [UInt8] = []
        output.reserveCapacity(text.utf8.count)
        for grapheme in text {
            let chunk = String(grapheme)
            if chunk.utf8.count < 6, let mapped = replacement(for: chunk.utf8) {
                output.append(contentsOf: mapped)
            } else {
                for scalar in chunk.unicodeScalars {
                    let bytes = scalar.utf8
                    output.append(contentsOf: replacement(for: bytes) ?? ArraySlice(bytes))
                }
            }
        }
        return String(decoding: output, as: UTF8.self)
    }
}

final class StripAccentsNormalizer: Normalizer {
    required init(config: Config) {}
    func normalize(text: String) -> String {
        // Standalone StripAccents removes marks without decomposing precomposed letters.
        String(
            String.UnicodeScalarView(
                text.unicodeScalars.filter {
                    switch $0.properties.generalCategory {
                    case .nonspacingMark, .spacingMark, .enclosingMark: return false
                    default: return true
                    }
                }))
    }
}

final class StripNormalizer: Normalizer {
    let leftStrip: Bool
    let rightStrip: Bool

    required init(config: Config) {
        leftStrip = config.stripLeft.boolean(or: true)
        rightStrip = config.stripRight.boolean(or: true)
    }

    func normalize(text: String) -> String {
        // Rust trims Unicode scalar values. A Swift Character may combine whitespace
        // with an accent; removing that whole grapheme would silently delete the accent.
        var result = text.unicodeScalars[...]
        if leftStrip {
            result = result.drop(while: { $0.properties.isWhitespace })
        }
        if rightStrip {
            while let last = result.last, last.properties.isWhitespace {
                result.removeLast()
            }
        }
        return String(result)
    }
}

// MARK: - Replace patterns

enum StringReplacePattern: Sendable {
    case regexp(regexp: NSRegularExpression, replacement: String)
    case string(pattern: String, replacement: String)

    func replace(_ text: String) -> String {
        switch self {
        case let .regexp(regexp, replacement):
            let range = NSRange(text.startIndex..., in: text)
            return regexp.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: replacement)
        case let .string(toReplace, replacement):
            return text.replacingBytes(of: toReplace, with: replacement)
        }
    }

    static func from(config: Config) throws -> StringReplacePattern? {
        guard let replacement = config.content.string() else { return nil }
        if let pattern = config.pattern.String.string() {
            return .string(pattern: pattern, replacement: replacement)
        }
        if let pattern = config.pattern.Regex.string() {
            // Many SentencePiece configs express a literal via a regex with a single-scalar
            // pattern such as " ". Treat trivial patterns as literals to skip the regex engine.
            if Self.isLiteral(pattern), !replacement.contains("$"), !replacement.contains("\\") {
                return .string(pattern: pattern, replacement: replacement)
            }
            return .regexp(regexp: try compileRegex(pattern, component: "Replace normalizer"), replacement: replacement)
        }
        return nil
    }

    private static func isLiteral(_ pattern: String) -> Bool {
        let metacharacters: Set<Character> = ["\\", "^", "$", ".", "|", "?", "*", "+", "(", ")", "[", "]", "{", "}"]
        return !pattern.isEmpty && !pattern.contains(where: { metacharacters.contains($0) })
    }
}
