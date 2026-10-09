import Foundation

/// Models that keep their vocabulary in a ``Vocabulary`` expose it for byte decoding.
protocol VocabularyOwningModel {
    var decodeVocabulary: Vocabulary { get }
}

extension BPETokenizer: VocabularyOwningModel {
    var decodeVocabulary: Vocabulary { vocab }
}

extension UnigramTokenizer: VocabularyOwningModel {
    var decodeVocabulary: Vocabulary { vocabulary }
}

/// The decoder shapes whose whole-sequence output can be produced from per-id byte slices.
/// Anything else keeps the reference string path.
enum PieceDecodeShape: Sendable {
    /// `Sequence` of literal `Replace` rules, an optional `ByteFallback`, an optional `Fuse`
    /// and an optional leading-`Strip` (Llama, Gemma, Mistral, …).
    case spm(replacements: [StringReplacePattern], byteFallback: Bool, strip: (scalar: UInt8, count: Int)?)
    /// A standalone `Metaspace` decoder (XLM-RoBERTa, T5): the replacement becomes a space on
    /// every token, and is removed from the first one when the prefix space is enabled.
    case metaspace(pattern: [UInt8], removeLeading: Bool)
    /// A standalone `WordPiece` decoder (BERT): continuation prefixes are stripped, other
    /// tokens are preceded by a space, and each piece may be cleaned up.
    case wordPiece(prefix: [UInt8], cleanup: Bool)

    /// `nil` when `decoder` is not one of the supported shapes.
    static func classify(_ decoder: (any Decoder)?) -> PieceDecodeShape? {
        switch decoder {
        case let sequence as DecoderSequence:
            var index = 0
            var replacements: [StringReplacePattern] = []
            while index < sequence.decoders.count, let replace = sequence.decoders[index] as? ReplaceDecoder {
                guard let pattern = replace.pattern, pattern.isLiteral else { return nil }
                replacements.append(pattern)
                index += 1
            }
            guard index < sequence.decoders.count else {
                // Only replacements: the join fuses their outputs the same way.
                return replacements.isEmpty ? nil : .spm(replacements: replacements, byteFallback: false, strip: nil)
            }
            var byteFallback = false
            if sequence.decoders[index] is ByteFallbackDecoder {
                byteFallback = true
                index += 1
            }
            // Strip after Fuse trims one string. Without Fuse it trims each piece, which a
            // single concatenation does not model, so that shape keeps the reference path.
            var fused = false
            if index < sequence.decoders.count, sequence.decoders[index] is FuseDecoder {
                fused = true
                index += 1
            }
            var strip: (scalar: UInt8, count: Int)?
            if index < sequence.decoders.count, let stripDecoder = sequence.decoders[index] as? StripDecoder {
                guard fused, stripDecoder.stop == 0, stripDecoder.start >= 0,
                    stripDecoder.content.utf8.count == 1, let scalar = stripDecoder.content.utf8.first
                else { return nil }
                strip = (scalar, stripDecoder.start)
                index += 1
            }
            guard index == sequence.decoders.count, !replacements.isEmpty || byteFallback else { return nil }
            return .spm(replacements: replacements, byteFallback: byteFallback, strip: strip)
        case let decoder as MetaspaceDecoder:
            guard !decoder.replacement.isEmpty else { return nil }
            return .metaspace(pattern: Array(decoder.replacement.utf8), removeLeading: decoder.addPrefixSpace)
        case let decoder as WordPieceDecoder:
            return .wordPiece(prefix: Array(decoder.prefix.utf8), cleanup: decoder.cleanup)
        default:
            return nil
        }
    }
}

/// Decodes only the requested ids from immutable vocabulary bytes. Initialization retains
/// the decoder plan without walking or copying the vocabulary; all scratch is call-local.
final class PieceDecoder: Sendable {
    private let vocabulary: Vocabulary
    private let normalizedSpellings: [Int: String]
    private let hasOverrides: Bool
    private let shape: PieceDecodeShape

    init(vocabulary: Vocabulary, normalizedSpellings: [Int: String], shape: PieceDecodeShape) {
        self.vocabulary = vocabulary
        self.normalizedSpellings = normalizedSpellings
        hasOverrides = !normalizedSpellings.isEmpty
        self.shape = shape
    }

    @inline(__always)
    private func withToken<R>(_ id: Int, _ body: (UnsafeBufferPointer<UInt8>) -> R) -> R? {
        if hasOverrides, var text = normalizedSpellings[id] {
            return text.withUTF8(body)
        }
        guard vocabulary.contains(id: id) else { return nil }
        return vocabulary.withBytes(of: id, body)
    }

    func decode(_ tokens: [Int], skipping skipped: Set<Int>, cleanup: Bool = false) -> String {
        guard !tokens.isEmpty else { return "" }
        switch shape {
        case .wordPiece, .metaspace:
            return decodeLocal(tokens, skipping: skipped, cleanup: cleanup)
        case let .spm(replacements, fallback, strip):
            return decodeSPM(
                tokens, skipping: skipped, replacements: replacements, fallback: fallback,
                strip: strip, cleanup: cleanup)
        }
    }

    private func decodeLocal(_ tokens: [Int], skipping skipped: Set<Int>, cleanup: Bool) -> String {
        let skip = !skipped.isEmpty
        var capacity = 0
        for id in tokens {
            if skip, skipped.contains(id) { continue }
            // One extra byte covers the WordPiece separator, including empty tokens.
            _ = withToken(id) { capacity += $0.count + 1 }
        }
        guard capacity > 0 else { return "" }
        return String(unsafeUninitializedCapacity: capacity) { output in
            let base = output.baseAddress!
            var written = 0
            var first = true
            for id in tokens {
                if skip, skipped.contains(id) { continue }
                _ = withToken(id) { bytes in
                    written += writeLocal(bytes, lead: first, into: base + written)
                    first = false
                }
            }
            if cleanup, written > 0 {
                written = TokenizationCleanup.compact(base, count: written, wordPiece: false)
            }
            return written
        }
    }

    /// Writes a complete local piece. The destination has room for `bytes.count + 1` bytes.
    private func writeLocal(
        _ bytes: UnsafeBufferPointer<UInt8>, lead: Bool,
        into output: UnsafeMutablePointer<UInt8>
    ) -> Int {
        switch shape {
        case let .wordPiece(prefix, cleanup):
            var start = 0
            var written = 0
            if !lead {
                if bytes.starts(with: prefix) {
                    start = prefix.count
                } else {
                    output.initialize(to: UInt8(ascii: " "))
                    written = 1
                }
            }
            let length = bytes.count - start
            if length > 0 {
                (output + written).initialize(from: bytes.baseAddress! + start, count: length)
                written += length
            }
            return cleanup ? TokenizationCleanup.compact(output, count: written, wordPiece: true) : written
        case let .metaspace(pattern, removeLeading):
            var written = 0
            var read = 0
            let m = pattern.count
            pattern.withUnsafeBufferPointer { pattern in
                while read < bytes.count {
                    if m > 0, read + m <= bytes.count,
                        memcmp(bytes.baseAddress! + read, pattern.baseAddress!, m) == 0
                    {
                        if !lead || !removeLeading {
                            (output + written).initialize(to: UInt8(ascii: " "))
                            written += 1
                        }
                        read += m
                    } else {
                        (output + written).initialize(to: bytes[read])
                        written += 1
                        read += 1
                    }
                }
            }
            return written
        case .spm:
            preconditionFailure("Local decoding requires WordPiece or Metaspace")
        }
    }

    enum StreamPiece: Sendable {
        case absent
        case fallback(UInt8)
        case text(String)
    }

    func streamPiece(_ id: Int, lead: Bool) -> StreamPiece {
        switch shape {
        case .wordPiece, .metaspace:
            return withToken(id) { .text(localString($0, lead: lead)) } ?? .absent
        case .spm:
            return withDecoded(id, lead: lead) { bytes, fallback in
                if let fallback { return .fallback(fallback) }
                return .text(String(decoding: bytes, as: UTF8.self))
            } ?? .absent
        }
    }

    private func localString(_ bytes: UnsafeBufferPointer<UInt8>, lead: Bool) -> String {
        String(unsafeUninitializedCapacity: bytes.count + 1) { output in
            writeLocal(bytes, lead: lead, into: output.baseAddress!)
        }
    }

    func withDecoded<R>(
        _ id: Int, lead: Bool,
        _ body: (UnsafeBufferPointer<UInt8>, _ fallback: UInt8?) -> R
    ) -> R? {
        withToken(id) { bytes in
            switch shape {
            case .wordPiece, .metaspace:
                var text = localString(bytes, lead: lead)
                return text.withUTF8 { body($0, nil) }
            case let .spm(replacements, fallback, _):
                var token: [UInt8] = []
                var scratch: [UInt8] = []
                return withReplaced(bytes, replacements: replacements, token: &token, scratch: &scratch) { piece in
                    if fallback, let byte = ByteFallbackDecoder.parseByte(piece) {
                        return body(UnsafeBufferPointer(start: nil, count: 0), byte)
                    }
                    return body(piece, nil)
                }
            }
        }
    }

    /// Only materializes a token buffer when a replacement actually matches. Unchanged
    /// pieces are borrowed directly from the vocabulary, and buffers are reused across ids.
    private func withReplaced<R>(
        _ bytes: UnsafeBufferPointer<UInt8>, replacements: [StringReplacePattern],
        token: inout [UInt8], scratch: inout [UInt8], _ body: (UnsafeBufferPointer<UInt8>) -> R
    ) -> R {
        var changed = false
        for pattern in replacements {
            if changed {
                if token.withUnsafeBufferPointer({ pattern.matches(in: $0) }) {
                    scratch.removeAll(keepingCapacity: true)
                    token.withUnsafeBufferPointer { pattern.replace($0, into: &scratch) }
                    swap(&token, &scratch)
                }
            } else if pattern.matches(in: bytes) {
                token.removeAll(keepingCapacity: true)
                pattern.replace(bytes, into: &token)
                changed = true
            }
        }
        return changed ? token.withUnsafeBufferPointer(body) : body(bytes)
    }

    private func decodeSPM(
        _ tokens: [Int], skipping skipped: Set<Int>, replacements: [StringReplacePattern],
        fallback: Bool, strip: (scalar: UInt8, count: Int)?, cleanup: Bool
    ) -> String {
        var output: [UInt8] = []
        output.reserveCapacity(tokens.count)
        var token: [UInt8] = []
        var scratch: [UInt8] = []
        var run: [UInt8] = []
        let skip = !skipped.isEmpty
        func flushRun() {
            guard !run.isEmpty else { return }
            let valid = run.withUnsafeBufferPointer { Self.isUTF8($0.baseAddress!, count: $0.count) }
            if valid {
                output.append(contentsOf: run)
            } else {
                for _ in run { output.append(contentsOf: [0xEF, 0xBF, 0xBD]) }
            }
            run.removeAll(keepingCapacity: true)
        }
        for id in tokens {
            if skip, skipped.contains(id) { continue }
            _ = withToken(id) { bytes in
                withReplaced(bytes, replacements: replacements, token: &token, scratch: &scratch) { piece in
                    if fallback, let byte = ByteFallbackDecoder.parseByte(piece) {
                        run.append(byte)
                    } else {
                        flushRun()
                        output.append(contentsOf: piece)
                    }
                }
            }
        }
        flushRun()
        var written = output.count
        if written > 0 {
            written = output.withUnsafeMutableBufferPointer { buffer in
                let base = buffer.baseAddress!
                var removed = 0
                if let strip {
                    while removed < buffer.count, removed < strip.count, base[removed] == strip.scalar { removed += 1 }
                }
                if removed > 0 { memmove(base, base + removed, buffer.count - removed) }
                let length = buffer.count - removed
                return cleanup ? TokenizationCleanup.compact(base, count: length, wordPiece: false) : length
            }
        }
        return String(decoding: output.prefix(written), as: UTF8.self)
    }

    /// Strict UTF-8, matching `String(bytes:encoding:) != nil` for a byte-fallback run.
    private static func isUTF8(_ base: UnsafePointer<UInt8>, count: Int) -> Bool {
        var i = 0
        while i < count {
            let b = base[i]
            if b < 0x80 {
                i += 1
                continue
            }
            let remaining = count - i
            if b < 0xC2 || b > 0xF4 { return false }
            if b < 0xE0 {
                guard remaining >= 2, base[i + 1] & 0xC0 == 0x80 else { return false }
                i += 2
            } else if b < 0xF0 {
                guard remaining >= 3, base[i + 1] & 0xC0 == 0x80, base[i + 2] & 0xC0 == 0x80 else { return false }
                if b == 0xE0, base[i + 1] < 0xA0 { return false }
                if b == 0xED, base[i + 1] >= 0xA0 { return false }
                i += 3
            } else {
                guard remaining >= 4,
                    base[i + 1] & 0xC0 == 0x80, base[i + 2] & 0xC0 == 0x80, base[i + 3] & 0xC0 == 0x80
                else { return false }
                if b == 0xF0, base[i + 1] < 0x90 { return false }
                if b == 0xF4, base[i + 1] >= 0x90 { return false }
                i += 4
            }
        }
        return true
    }
}
