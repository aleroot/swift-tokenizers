import Foundation

/// Converts token strings back into text fragments.
public protocol Decoder: Sendable {
    func decode(tokens: [String]) -> [String]
    func callAsFunction(tokens: [String]) -> [String]
    /// Creates the decoder from its `tokenizer.json` entry.
    /// - Throws: ``TokenizerError/invalidConfiguration(_:)`` when required fields are missing.
    init(config: Config) throws
}

public extension Decoder {
    func callAsFunction(tokens: [String]) -> [String] { decode(tokens: tokens) }
}

enum DecoderType: String {
    case Sequence
    case WordPiece
    case ByteLevel
    case Replace
    case ByteFallback
    case Fuse
    case Strip
    case Metaspace
    case Unknown = ""
}

struct DecoderFactory {
    static func fromConfig(config: Config?, addedTokens: Set<String>? = nil) throws -> (any Decoder)? {
        guard let config, let typeName = config.type.string() else { return nil }
        switch DecoderType(rawValue: typeName) {
        case .Sequence: return try DecoderSequence(config: config, addedTokens: addedTokens)
        case .ByteLevel: return ByteLevelDecoder(config: config, addedTokens: addedTokens)
        case .Replace: return try ReplaceDecoder(config: config)
        case .ByteFallback: return ByteFallbackDecoder(config: config)
        case .Fuse: return FuseDecoder(config: config)
        case .Strip: return try StripDecoder(config: config)
        case .Metaspace: return MetaspaceDecoder(config: config)
        case .WordPiece: return try WordPieceDecoder(config: config)
        default: throw TokenizerError.unsupportedComponent("decoder `\(typeName)`")
        }
    }
}

final class WordPieceDecoder: Decoder {
    let prefix: String
    let cleanup: Bool

    /// https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/decoders/wordpiece.rs#L31
    /// (A compile-time constant pattern: compilation cannot fail.)
    private static let cleanupRegex = try! NSRegularExpression(
        pattern: "\\s(\\.|\\?|\\!|\\,|'\\s|n't|'m|'s|'ve|'re)", options: [])

    required init(config: Config) throws {
        prefix = try require(config.prefix.string(), "WordPiece decoder", field: "prefix")
        cleanup = config.cleanup.boolean(or: false)
    }

    func decode(tokens: [String]) -> [String] {
        guard let first = tokens.first else { return [] }
        var result: [String] = []
        result.reserveCapacity(tokens.count)
        result.append(cleanup ? cleanUpTokenization(first) : first)
        for token in tokens.dropFirst() {
            let piece = token.hasBytePrefix(prefix) ? String(token.droppingBytePrefix(prefix)) : " \(token)"
            result.append(cleanup ? cleanUpTokenization(piece) : piece)
        }
        return result
    }

    private func cleanUpTokenization(_ token: String) -> String {
        let range = NSRange(location: 0, length: token.utf16.count)
        return Self.cleanupRegex.stringByReplacingMatches(in: token, options: [], range: range, withTemplate: "$1")
            .replacingOccurrences(of: " do not", with: " don't")
    }
}

final class DecoderSequence: Decoder {
    let decoders: [any Decoder]

    required convenience init(config: Config) throws {
        try self.init(config: config, addedTokens: nil)
    }

    init(config: Config, addedTokens: Set<String>?) throws {
        let configs = try require(config.decoders.array(), "Sequence decoder", field: "decoders")
        decoders = try configs.compactMap { try DecoderFactory.fromConfig(config: $0, addedTokens: addedTokens) }
    }

    func decode(tokens: [String]) -> [String] {
        var current = tokens
        for decoder in decoders {
            current = decoder.decode(tokens: current)
        }
        return current
    }
}

final class ByteLevelDecoder: Decoder {
    let addedTokens: Set<String>

    required init(config: Config) {
        addedTokens = []
    }

    init(config: Config, addedTokens: Set<String>?) {
        self.addedTokens = addedTokens ?? []
    }

    func decode(tokens: [String]) -> [String] {
        var subTexts: [String] = []
        var bytes: [UInt8] = []

        func flush() {
            if !bytes.isEmpty {
                subTexts.append(String(decoding: bytes, as: UTF8.self))
                bytes.removeAll(keepingCapacity: true)
            }
        }

        for token in tokens {
            if addedTokens.contains(token) {
                flush()
                subTexts.append(token)
            } else {
                ByteLevelAlphabet.decode(token, into: &bytes)
            }
        }
        flush()
        return subTexts
    }
}

final class ReplaceDecoder: Decoder {
    let pattern: StringReplacePattern?

    required init(config: Config) throws {
        pattern = try StringReplacePattern.from(config: config)
    }

    func decode(tokens: [String]) -> [String] {
        guard let pattern else { return tokens }
        return tokens.map { pattern.replace($0) }
    }
}

/// Merges runs of `<0xNN>` byte-fallback tokens back into text.
///
/// Follows `tokenizers`' `ByteFallback::decode_chain`: a trailing run of byte tokens is
/// flushed, and a run that is not valid UTF-8 becomes one U+FFFD per byte.
final class ByteFallbackDecoder: Decoder {
    required init(config: Config) {}

    func decode(tokens: [String]) -> [String] {
        var newTokens: [String] = []
        newTokens.reserveCapacity(tokens.count)
        var bytes: [UInt8] = []

        func flush() {
            guard !bytes.isEmpty else { return }
            if let string = String(bytes: bytes, encoding: .utf8) {
                newTokens.append(string)
            } else {
                for _ in bytes { newTokens.append("\u{FFFD}") }
            }
            bytes.removeAll(keepingCapacity: true)
        }

        for token in tokens {
            if let byte = Self.parseByte(token) {
                bytes.append(byte)
            } else {
                flush()
                newTokens.append(token)
            }
        }
        flush()
        return newTokens
    }

    /// Parses `<0xAB>` style byte-fallback tokens.
    static func parseByte(_ token: String) -> UInt8? {
        let utf8 = token.utf8
        guard utf8.count == 6 else { return nil }
        var it = utf8.makeIterator()
        guard it.next() == UInt8(ascii: "<"), it.next() == UInt8(ascii: "0"), it.next() == UInt8(ascii: "x") else {
            return nil
        }
        guard let h = it.next(), let l = it.next(), it.next() == UInt8(ascii: ">") else { return nil }
        guard let hi = hexValue(h), let lo = hexValue(l) else { return nil }
        return hi << 4 | lo
    }

    private static func hexValue(_ c: UInt8) -> UInt8? {
        switch c {
        case UInt8(ascii: "0")...UInt8(ascii: "9"): return c - UInt8(ascii: "0")
        case UInt8(ascii: "a")...UInt8(ascii: "f"): return c - UInt8(ascii: "a") + 10
        case UInt8(ascii: "A")...UInt8(ascii: "F"): return c - UInt8(ascii: "A") + 10
        default: return nil
        }
    }
}

final class FuseDecoder: Decoder {
    required init(config: Config) {}

    func decode(tokens: [String]) -> [String] {
        [tokens.joined()]
    }
}

final class StripDecoder: Decoder {
    let content: String
    let start: Int
    let stop: Int

    required init(config: Config) throws {
        content = try require(config.content.string(), "Strip decoder", field: "content")
        start = try require(config.start.integer(), "Strip decoder", field: "start")
        stop = try require(config.stop.integer(), "Strip decoder", field: "stop")
    }

    func decode(tokens: [String]) -> [String] {
        let scalar = content.unicodeScalars.first ?? " "
        return tokens.map { token in
            var scalars = Substring(token).unicodeScalars
            var trimmed = 0
            while trimmed < start, scalars.first == scalar {
                scalars.removeFirst()
                trimmed += 1
            }
            trimmed = 0
            while trimmed < stop, scalars.last == scalar {
                scalars.removeLast()
                trimmed += 1
            }
            return String(scalars)
        }
    }
}

final class MetaspaceDecoder: Decoder {
    let addPrefixSpace: Bool
    let replacement: String

    required init(config: Config) {
        let scheme: MetaspacePreTokenizer.PrependScheme
        if let schemeStr = config.prependScheme.string() {
            scheme = MetaspacePreTokenizer.PrependScheme(rawValue: schemeStr) ?? .always
        } else {
            scheme = config.addPrefixSpace.boolean(or: true) ? .always : .never
        }
        addPrefixSpace = scheme != .never
        replacement = config.replacement.string(or: "_")
    }

    func decode(tokens: [String]) -> [String] {
        var replaced = tokens.map { $0.replacingBytes(of: replacement, with: " ") }
        if addPrefixSpace, replaced.first?.hasBytePrefix(" ") ?? false {
            replaced[0] = String(replaced[0].droppingBytePrefix(" "))
        }
        return replaced
    }
}
