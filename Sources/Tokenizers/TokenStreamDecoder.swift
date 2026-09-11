import Foundation

/// An independently owned decoding session. Concatenating every `append` result and
/// `finish()` equals decoding the complete token sequence, including malformed UTF-8.
/// Copying a session copies its state; create one session per generated response.
public struct TokenStreamDecoder: Sendable {
    /// False for pipelines whose global transformations require the complete sequence.
    /// Such pipelines buffer IDs and emit their result once, from `finish()`.
    public let isIncremental: Bool
    private let tokenizer: any Tokenizer
    private let skipSpecialTokens: Bool
    private let skipped: Set<Int>
    private let mode: Mode
    private let cleanupEnabled: Bool
    private var cleanup: [StreamingReplacement]
    private var pendingUTF8: [UInt8] = []
    private var pendingFallback: [UInt8] = []
    private var bufferedIDs: [Int] = []
    private var first = true
    private var stripRemaining: Int
    private var finished = false

    private enum Mode: Sendable {
        case byteLevel(ByteLevelDecodeTable?)
        case wordPiece(WordPieceDecoder)
        case metaspace(MetaspaceDecoder)
        case joined(separator: String)
        case fallback(replacements: [ReplaceDecoder], strip: UInt8?, count: Int)
        case buffered
    }

    init(tokenizer: any Tokenizer, skipSpecialTokens: Bool) {
        self.init(
            tokenizer: tokenizer, skipSpecialTokens: skipSpecialTokens, skipped: [],
            mode: .buffered, cleanup: false)
    }

    init(
        tokenizer: PreTrainedTokenizer, skipSpecialTokens: Bool, decoder: (any Decoder)?,
        table: ByteLevelDecodeTable?, cleanup: Bool, skipped: Set<Int>
    ) {
        let mode: Mode
        switch decoder {
        case is ByteLevelDecoder: mode = .byteLevel(table)
        case let decoder as WordPieceDecoder: mode = .wordPiece(decoder)
        case let decoder as MetaspaceDecoder: mode = .metaspace(decoder)
        case is FuseDecoder: mode = .joined(separator: "")
        case is ByteFallbackDecoder: mode = .fallback(replacements: [], strip: nil, count: 0)
        case let sequence as DecoderSequence:
            mode = Self.sequenceMode(sequence.decoders)
        case nil: mode = .joined(separator: " ")
        default: mode = .buffered
        }
        self.init(
            tokenizer: tokenizer, skipSpecialTokens: skipSpecialTokens,
            skipped: skipSpecialTokens ? skipped : [], mode: mode, cleanup: cleanup)
    }

    private init(tokenizer: any Tokenizer, skipSpecialTokens: Bool, skipped: Set<Int>, mode: Mode, cleanup: Bool) {
        self.tokenizer = tokenizer
        self.skipSpecialTokens = skipSpecialTokens
        self.skipped = skipped
        self.mode = mode
        cleanupEnabled = cleanup
        self.cleanup = cleanup ? TokenizationCleanup.replacements.map(StreamingReplacement.init) : []
        if case .buffered = mode { isIncremental = false } else { isIncremental = true }
        if case let .fallback(_, _, count) = mode { stripRemaining = count } else { stripRemaining = 0 }
    }

    private static func sequenceMode(_ decoders: [any Decoder]) -> Mode {
        if decoders.isEmpty { return .joined(separator: "") }
        if decoders.count == 1 {
            if decoders[0] is ByteLevelDecoder { return .byteLevel(nil) }
            if let decoder = decoders[0] as? WordPieceDecoder { return .wordPiece(decoder) }
            if let decoder = decoders[0] as? MetaspaceDecoder { return .metaspace(decoder) }
        }
        // SentencePiece/Llama: token-local replacements, byte fallback, then optional
        // fusion and leading-space removal. Never move a replacement across Fuse.
        var index = 0
        var replacements: [ReplaceDecoder] = []
        while index < decoders.count, let replacement = decoders[index] as? ReplaceDecoder {
            replacements.append(replacement)
            index += 1
        }
        guard index < decoders.count, decoders[index] is ByteFallbackDecoder else { return .buffered }
        index += 1
        if index == decoders.count { return .fallback(replacements: replacements, strip: nil, count: 0) }
        guard decoders[index] is FuseDecoder else { return .buffered }
        index += 1
        if index == decoders.count { return .fallback(replacements: replacements, strip: nil, count: 0) }
        guard index + 1 == decoders.count, let strip = decoders[index] as? StripDecoder,
            strip.stop == 0, strip.start >= 0,
            let scalar = strip.content.unicodeScalars.first, scalar.isASCII
        else { return .buffered }
        return .fallback(replacements: replacements, strip: UInt8(scalar.value), count: strip.start)
    }

    /// Consumes one token and emits only text that cannot be changed by later tokens.
    /// Incomplete UTF-8 is held until completed or finished. Byte-fallback runs are held
    /// until their end: one invalid byte makes the *entire run* invalid in that decoder.
    public mutating func append(token: Int) throws -> String {
        guard !finished else {
            throw TokenizerError.invalidConfiguration("Reset the decoder before appending after finish()")
        }
        if case .buffered = mode { bufferedIDs.append(token); return "" }
        guard !skipped.contains(token) else { return "" }
        var bytes: [UInt8] = []
        if case let .byteLevel(table?) = mode {
            if !cleanupEnabled { return table.withBytes(for: token) { emitUTF8($0, final: false) } }
            table.appendBytes(for: token, into: &bytes)
        } else {
            guard var text = tokenizer.convertIdToToken(token) else { return "" }
            switch mode {
            case .byteLevel: ByteLevelAlphabet.decode(text, into: &bytes)
            case let .wordPiece(decoder):
                if !first {
                    text =
                        text.hasBytePrefix(decoder.prefix)
                        ? String(text.droppingBytePrefix(decoder.prefix)) : " " + text
                }
                bytes = Array((decoder.cleanup ? WordPieceDecoder.cleanUpTokenization(text) : text).utf8)
            case let .metaspace(decoder):
                bytes = Array(
                    text.replacingBytes(of: decoder.replacement, with: first && decoder.addPrefixSpace ? "" : " ").utf8)
            case let .joined(separator): bytes = Array((first ? text : separator + text).utf8)
            case let .fallback(replacements, _, _):
                for replacement in replacements { text = replacement.pattern?.replace(text) ?? text }
                if let byte = ByteFallbackDecoder.parseByte(text) {
                    pendingFallback.append(byte)
                    return ""
                }
                bytes = flushFallback()
                bytes.append(contentsOf: text.utf8)
            case .buffered: break
            }
            first = false
        }
        return emit(bytes, final: false)
    }

    /// Drains pending text. Idempotent; call `reset()` before reusing the session.
    public mutating func finish() -> String {
        guard !finished else { return "" }
        finished = true
        if case .buffered = mode {
            let result = tokenizer.decode(tokens: bufferedIDs, skipSpecialTokens: skipSpecialTokens)
            bufferedIDs.removeAll(keepingCapacity: true)
            return result
        }
        return emit(flushFallback(), final: true)
    }

    /// Starts another independent sequence with the same tokenizer and options.
    public mutating func reset() {
        pendingUTF8.removeAll(keepingCapacity: true)
        pendingFallback.removeAll(keepingCapacity: true)
        bufferedIDs.removeAll(keepingCapacity: true)
        cleanup = cleanupEnabled ? TokenizationCleanup.replacements.map(StreamingReplacement.init) : []
        first = true
        finished = false
        if case let .fallback(_, _, count) = mode { stripRemaining = count }
    }

    private mutating func flushFallback() -> [UInt8] {
        guard !pendingFallback.isEmpty else { return [] }
        defer { pendingFallback.removeAll(keepingCapacity: true) }
        if String(bytes: pendingFallback, encoding: .utf8) != nil { return pendingFallback }
        return Array(String(repeating: "\u{FFFD}", count: pendingFallback.count).utf8)
    }

    private mutating func emit(_ input: [UInt8], final: Bool) -> String {
        var bytes = input
        if stripRemaining > 0, case let .fallback(_, strip?, _) = mode {
            var removed = 0
            while removed < bytes.count, stripRemaining > 0 {
                guard bytes[removed] == strip else { stripRemaining = 0; break }
                removed += 1
                stripRemaining -= 1
            }
            if removed > 0 { bytes.removeFirst(removed) }
        }
        for index in cleanup.indices { bytes = cleanup[index].append(bytes, final: final) }
        return bytes.withUnsafeBufferPointer { emitUTF8($0, final: final) }
    }

    private mutating func emitUTF8(_ bytes: UnsafeBufferPointer<UInt8>, final: Bool) -> String {
        if pendingUTF8.isEmpty {
            let count = final ? bytes.count : Self.stableUTF8Count(bytes)
            if count < bytes.count { pendingUTF8.append(contentsOf: bytes[count...]) }
            return String(decoding: bytes.prefix(count), as: UTF8.self)
        }
        pendingUTF8.append(contentsOf: bytes)
        let count = final ? pendingUTF8.count : pendingUTF8.withUnsafeBufferPointer(Self.stableUTF8Count)
        guard count > 0 else { return "" }
        let result = String(decoding: pendingUTF8.prefix(count), as: UTF8.self)
        pendingUTF8.removeFirst(count)
        return result
    }

    /// Retain at most three bytes, and only when they can still become a valid scalar.
    /// Invalid prefixes are repaired immediately, just as String(decoding:as:) repairs them.
    private static func stableUTF8Count(_ bytes: UnsafeBufferPointer<UInt8>) -> Int {
        guard !bytes.isEmpty else { return 0 }
        var start = bytes.count - 1
        while start > 0, bytes[start] & 0xC0 == 0x80, bytes.count - start < 4 { start -= 1 }
        let lead = bytes[start]
        let width: Int
        switch lead {
        case 0xC2...0xDF: width = 2
        case 0xE0...0xEF: width = 3
        case 0xF0...0xF4: width = 4
        default: return bytes.count
        }
        let count = bytes.count - start
        guard count < width else { return bytes.count }
        if count > 1 {
            let second = bytes[start + 1]
            guard second & 0xC0 == 0x80,
                !(lead == 0xE0 && second < 0xA0), !(lead == 0xED && second >= 0xA0),
                !(lead == 0xF0 && second < 0x90), !(lead == 0xF4 && second >= 0x90)
            else { return bytes.count }
        }
        return start
    }
}

public extension Tokenizer {
    /// Creates an independent stream. Custom conformers default to correct buffered decoding.
    func makeStreamDecoder() -> TokenStreamDecoder { makeStreamDecoder(skipSpecialTokens: false) }

    func makeStreamDecoder(skipSpecialTokens: Bool) -> TokenStreamDecoder {
        TokenStreamDecoder(tokenizer: self, skipSpecialTokens: skipSpecialTokens)
    }
}

/// One literal replacement pass with bounded lookahead. Composing the passes preserves
/// the exact order of ordinary decoding cleanup, even when patterns cross token boundaries.
private struct StreamingReplacement: Sendable {
    let pattern: [UInt8]
    let replacement: [UInt8]
    private var pending: [UInt8] = []

    init(_ rule: (pattern: [UInt8], replacement: [UInt8])) {
        pattern = rule.pattern
        replacement = rule.replacement
    }

    mutating func append(_ bytes: [UInt8], final: Bool) -> [UInt8] {
        pending.append(contentsOf: bytes)
        var result: [UInt8] = []
        var index = 0
        while index < pending.count {
            let available = min(pattern.count, pending.count - index)
            let matches = pending[index..<index + available].elementsEqual(pattern.prefix(available))
            if matches {
                if available == pattern.count {
                    result.append(contentsOf: replacement); index += pattern.count; continue
                }
                if !final { break }
            }
            result.append(pending[index])
            index += 1
        }
        pending.removeFirst(index)
        return result
    }
}
