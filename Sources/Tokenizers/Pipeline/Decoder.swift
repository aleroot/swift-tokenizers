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
    case BPEDecoder
    case CTC
    case ByteLevel
    case Replace
    case ByteFallback
    case Fuse
    case Strip
    case Metaspace
    case Unknown = ""
}

struct DecoderFactory {
    static func fromConfig(config: Config?) throws -> (any Decoder)? {
        guard let config, let typeName = config.type.string() else { return nil }
        switch DecoderType(rawValue: typeName) {
        case .Sequence: return try DecoderSequence(config: config)
        case .ByteLevel: return ByteLevelDecoder(config: config)
        case .Replace: return try ReplaceDecoder(config: config)
        case .ByteFallback: return ByteFallbackDecoder(config: config)
        case .Fuse: return FuseDecoder(config: config)
        case .Strip: return try StripDecoder(config: config)
        case .Metaspace:
            try MetaspacePreTokenizer.validate(config)
            return MetaspaceDecoder(config: config)
        case .BPEDecoder: return BPEDecoder(config: config)
        case .CTC: return CTCDecoder(config: config)
        case .WordPiece: return try WordPieceDecoder(config: config)
        default: throw TokenizerError.unsupportedComponent("decoder `\(typeName)`")
        }
    }
}

final class WordPieceDecoder: Decoder {
    let prefix: String
    let cleanup: Bool

    required init(config: Config) throws {
        prefix = try require(config.prefix.string(), "WordPiece decoder", field: "prefix")
        cleanup = config.cleanup.boolean(or: false)
    }

    func decode(tokens: [String]) -> [String] {
        guard let first = tokens.first else { return [] }
        var result: [String] = []
        result.reserveCapacity(tokens.count)
        result.append(cleanup ? Self.cleanUpTokenization(first) : first)
        for token in tokens.dropFirst() {
            let piece = token.hasBytePrefix(prefix) ? String(token.droppingBytePrefix(prefix)) : " \(token)"
            result.append(cleanup ? Self.cleanUpTokenization(piece) : piece)
        }
        return result
    }

    static func cleanUpTokenization(_ token: String) -> String {
        // Keep the Rust replacement order and per-token semantics while avoiding eleven
        // Foundation searches and allocations for each ordinary vocabulary piece.
        TokenizationCleanup.cleanUp(token, wordPiece: true)
    }
}

final class DecoderSequence: Decoder {
    let decoders: [any Decoder]

    required init(config: Config) throws {
        let configs = try require(config.decoders.array(), "Sequence decoder", field: "decoders")
        decoders = try configs.compactMap { try DecoderFactory.fromConfig(config: $0) }
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
    required init(config: Config) {}

    func decode(tokens: [String]) -> [String] {
        var bytes: [UInt8] = []
        for token in tokens { ByteLevelAlphabet.decode(token, into: &bytes) }
        return [String(decoding: bytes, as: UTF8.self)]
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
            if String(bytes: bytes, encoding: .utf8) != nil {
                newTokens.append(String(decoding: bytes, as: UTF8.self))
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
        // HF removes every replacement character in the first token, including interior
        // markers. Literal spaces in that token are retained.
        tokens.enumerated().map { index, token in
            token.replacingBytes(of: replacement, with: index == 0 && addPrefixSpace ? "" : " ")
        }
    }
}

/// Character BPE end-of-word suffix decoding (HF tokenizers decoders/bpe.rs).
final class BPEDecoder: Decoder {
    let suffix: String
    required init(config: Config) { suffix = config.suffix.string(or: "</w>") }
    func decode(tokens: [String]) -> [String] {
        tokens.enumerated().map { index, token in
            token.replacingBytes(of: suffix, with: index == tokens.count - 1 ? "" : " ")
        }
    }
}

/// CTC collapse happens before pad removal, so pad-separated repeats survive.
final class CTCDecoder: Decoder {
    let pad: String
    let delimiter: String
    let cleanup: Bool
    required init(config: Config) {
        pad = config.padToken.string(or: "<pad>")
        delimiter = config.wordDelimiterToken.string(or: "|")
        cleanup = config.cleanup.boolean(or: true)
    }
    func decode(tokens: [String]) -> [String] {
        var previous: String?
        var result: [String] = []
        for token in tokens {
            defer { previous = token }
            if let previous, previous.utf8.elementsEqual(token.utf8) { continue }
            var piece = token.replacingBytes(of: pad, with: "")
            if cleanup {
                piece = WordPieceDecoder.cleanUpTokenization(piece).replacingBytes(of: delimiter, with: " ")
            }
            if !piece.isEmpty { result.append(piece) }
        }
        return result
    }
}
