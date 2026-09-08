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
        case .Precompiled: return PrecompiledNormalizer(config: config)
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

    func normalize(text: String) -> String { prepend + text }
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
        if shouldCleanText { output = cleanText(output) }
        if shouldHandleChineseChars { output = handleChineseChars(output) }
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

    /// Mirrors `tokenizers`' `do_clean_text`: drops NUL, U+FFFD and control/format characters
    /// (Cc/Cf/Cs/Co, except tab/newline/CR) and maps every whitespace to a plain space.
    private func cleanText(_ text: String) -> String {
        var out = ""
        out.reserveCapacity(text.utf8.count)
        for scalar in text.unicodeScalars {
            let v = scalar.value
            if v == 0 || v == 0xFFFD || isControl(scalar) { continue }
            out.unicodeScalars.append(isWhitespace(scalar) ? " " : scalar)
        }
        return out
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

    private func handleChineseChars(_ text: String) -> String {
        var hasCJK = false
        for scalar in text.unicodeScalars where scalar.isCJKUnifiedIdeograph {
            hasCJK = true
            break
        }
        guard hasCJK else { return text }
        var out = ""
        out.reserveCapacity(text.utf8.count + 16)
        for scalar in text.unicodeScalars {
            if scalar.isCJKUnifiedIdeograph {
                out.unicodeScalars.append(" ")
                out.unicodeScalars.append(scalar)
                out.unicodeScalars.append(" ")
            } else {
                out.unicodeScalars.append(scalar)
            }
        }
        return out
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
        var out = ""
        out.reserveCapacity(decomposed.utf8.count)
        for scalar in decomposed.unicodeScalars where scalar.properties.generalCategory != .nonspacingMark {
            out.unicodeScalars.append(scalar)
        }
        return out
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

final class PrecompiledNormalizer: Normalizer {
    // Simplified implementation (mirrors transformers.js): NFKC plus SentencePiece's
    // control/separator handling. The precompiled charsmap itself is not interpreted.
    required init(config: Config) {}

    func normalize(text: String) -> String {
        // SentencePiece's nmt_nfkc charsmap leaves U+FF5E (FULLWIDTH TILDE) untouched, while NFKC
        // would fold it to "~". Normalize the segments between tildes independently.
        var result = ""
        var segment = ""
        func flush() {
            if !segment.isEmpty {
                result += segment.precomposedStringWithCompatibilityMapping
                segment = ""
            }
        }
        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x0001...0x0008, 0x000B, 0x000E...0x001F, 0x007F, 0x008F, 0x009F:
                break  // non-printing control characters
            case 0x0009, 0x000A, 0x000C, 0x000D, 0x1680, 0x200B...0x200F, 0x2028, 0x2029, 0x2581, 0xFEFF, 0xFFFD:
                segment.append(" ")  // separators
            case 0xFF5E:
                flush()
                result.unicodeScalars.append(scalar)
            default:
                segment.unicodeScalars.append(scalar)
            }
        }
        flush()
        return result
    }
}

final class StripAccentsNormalizer: Normalizer {
    required init(config: Config) {}
    func normalize(text: String) -> String { text.precomposedStringWithCompatibilityMapping }
}

final class StripNormalizer: Normalizer {
    let leftStrip: Bool
    let rightStrip: Bool

    required init(config: Config) {
        leftStrip = config.stripLeft.boolean(or: true)
        rightStrip = config.stripRight.boolean(or: true)
    }

    func normalize(text: String) -> String {
        var result = Substring(text)
        if leftStrip {
            result = result.drop(while: { $0.isWhitespace })
        }
        if rightStrip {
            while let last = result.last, last.isWhitespace {
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
