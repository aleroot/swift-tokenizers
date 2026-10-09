// Precomputed decoded bytes for every vocabulary id of the SentencePiece-shaped decoders,
// so decoding is a concatenation of byte slices instead of per-token string work. Mirrors
// ``ByteLevelDecodeTable``: the table is read-only after explicit decode preparation.

import Foundation

/// Models that keep their vocabulary in a ``Vocabulary`` expose it for the decode tables.
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
            return .metaspace(pattern: Array(decoder.replacement.utf8), removeLeading: decoder.addPrefixSpace)
        case let decoder as WordPieceDecoder:
            return .wordPiece(prefix: Array(decoder.prefix.utf8), cleanup: decoder.cleanup)
        default:
            return nil
        }
    }
}

/// Byte slices for one decoder shape, computed once per tokenizer.
///
/// Absent ids and byte-fallback ids are packed into the high bits of `offsets`, so a decode
/// touches that array and the byte storage, not a second flags table. `leadStorage` holds the
/// first-token spelling where it differs (`Metaspace` prefix removal, `WordPiece` rules).
final class PieceDecodeTable: Sendable {
    private let storage: [UInt8]
    private let offsets: [UInt32]
    /// Parallel to `offsets` when the first emitted token has its own spelling. Nil otherwise.
    private let leadStorage: [UInt8]?
    private let leadOffsets: [UInt32]?
    private let strip: (scalar: UInt8, count: Int)?
    private let hasFallback: Bool
    private let count: Int

    /// Bit 31 marks an id the reference path skips. Bit 30 marks a byte-fallback id, whose
    /// single stored byte is the parsed value. The low 30 bits are the start offset.
    private static let absentBit: UInt32 = 1 << 31
    private static let fallbackBit: UInt32 = 1 << 30
    private static let offsetMask: UInt32 = (1 << 30) - 1

    private init(
        storage: [UInt8], offsets: [UInt32], leadStorage: [UInt8]?, leadOffsets: [UInt32]?,
        strip: (scalar: UInt8, count: Int)?, hasFallback: Bool, count: Int
    ) {
        self.storage = storage
        self.offsets = offsets
        self.leadStorage = leadStorage
        self.leadOffsets = leadOffsets
        self.strip = strip
        self.hasFallback = hasFallback
        self.count = count
    }

    /// Builds the table from every id the reference path would resolve: the model vocabulary
    /// plus any normalized added-token spellings.
    convenience init(
        vocabulary: Vocabulary, normalizedSpellings: [Int: String], shape: PieceDecodeShape
    ) {
        let count = max(vocabulary.count, (normalizedSpellings.keys.max() ?? -1) + 1)
        var storage: [UInt8] = []
        storage.reserveCapacity(vocabulary.withStorage { bytes, _ in bytes.count })
        var offsets = [UInt32](repeating: 0, count: count + 1)
        var leadStorage: [UInt8] = []
        let buildLead: Bool
        let strip: (scalar: UInt8, count: Int)?
        let hasFallback: Bool
        switch shape {
        case let .spm(_, fallback, stripValue):
            buildLead = false
            strip = stripValue
            hasFallback = fallback
        case let .metaspace(_, removeLeading):
            buildLead = removeLeading
            strip = nil
            hasFallback = false
        case .wordPiece:
            buildLead = true
            strip = nil
            hasFallback = false
        }
        var leadOffsets = buildLead ? [UInt32](repeating: 0, count: count + 1) : []
        if buildLead { leadStorage.reserveCapacity(storage.capacity) }

        var patternBytes: [UInt8] = []
        if case let .metaspace(pattern, _) = shape { patternBytes = pattern }

        // One reused token buffer keeps the per-id replacement work allocation-light.
        var tokenBytes: [UInt8] = []
        var piece: [UInt8] = []
        for id in 0..<count {
            let start = UInt32(storage.count)
            let leadStart = UInt32(leadStorage.count)
            // `withUTF8` is mutating: it may make a bridged string contiguous.
            // Empty spellings still occupy the first-token slot; WordPiece may emit a
            // separator for them in later positions.
            guard var token = normalizedSpellings[id] ?? vocabulary.token(id) else {
                offsets[id] = start | Self.absentBit
                if buildLead { leadOffsets[id] = leadStart | Self.absentBit }
                continue
            }
            tokenBytes.removeAll(keepingCapacity: true)
            token.withUTF8 { tokenBytes.append(contentsOf: $0) }

            var fallback = false
            switch shape {
            case let .spm(replacements, byteFallback, _):
                for pattern in replacements {
                    piece.removeAll(keepingCapacity: true)
                    tokenBytes.withUnsafeBufferPointer { pattern.replace($0, into: &piece) }
                    swap(&tokenBytes, &piece)
                }
                // `ByteFallback` parses the replaced text, and only exact six-byte tokens.
                if byteFallback, tokenBytes.count == 6,
                    let byte = ByteFallbackDecoder.parseByte(String(decoding: tokenBytes, as: UTF8.self))
                {
                    storage.append(byte)
                    fallback = true
                } else {
                    storage.append(contentsOf: tokenBytes)
                }
            case .metaspace:
                let pattern = patternBytes
                let m = pattern.count
                tokenBytes.withUnsafeBufferPointer { buffer in
                    guard let base = buffer.baseAddress else { return }
                    let n = buffer.count
                    var i = 0
                    while i < n {
                        if m > 0, i + m <= n,
                            pattern.withUnsafeBufferPointer({ memcmp(base + i, $0.baseAddress!, m) == 0 })
                        {
                            storage.append(UInt8(ascii: " "))
                            i += m
                        } else {
                            storage.append(base[i])
                            if buildLead { leadStorage.append(base[i]) }
                            i += 1
                        }
                    }
                }
            case let .wordPiece(prefix, cleanup):
                // The first token keeps its prefix and gains no leading space. Later tokens
                // drop the continuation prefix or take a leading space. Cleanup, when enabled,
                // runs on each form, matching ``WordPieceDecoder``.
                let lead = cleanup ? TokenizationCleanup.cleanUp(tokenBytes, wordPiece: true) : Array(tokenBytes)
                leadStorage.append(contentsOf: lead)
                if tokenBytes.starts(with: prefix) {
                    tokenBytes.removeFirst(prefix.count)
                } else {
                    tokenBytes.insert(UInt8(ascii: " "), at: 0)
                }
                let stored = cleanup ? TokenizationCleanup.cleanUp(tokenBytes, wordPiece: true) : tokenBytes
                storage.append(contentsOf: stored)
            }
            offsets[id] = start | (fallback ? Self.fallbackBit : 0)
            if buildLead { leadOffsets[id] = leadStart }
        }
        offsets[count] = UInt32(storage.count)
        if buildLead { leadOffsets[count] = UInt32(leadStorage.count) }
        precondition(
            storage.count < (1 << 30) && leadStorage.count < (1 << 30),
            "decode table exceeds the packed-offset range")
        self.init(
            storage: storage.trimmed(), offsets: offsets,
            leadStorage: buildLead ? leadStorage.trimmed() : nil,
            leadOffsets: buildLead ? leadOffsets : nil,
            strip: strip, hasFallback: hasFallback, count: count)
    }

    /// Concatenates the decoded bytes of `tokens`.
    func decode(_ tokens: [Int], skipping skipped: Set<Int>, cleanup: Bool = false) -> String {
        if hasFallback {
            return concatFallback(tokens, skipping: skipped, cleanup: cleanup)
        }
        if leadOffsets != nil {
            return concatLead(tokens, skipping: skipped, cleanup: cleanup)
        }
        return concatPlain(tokens, skipping: skipped, cleanup: cleanup)
    }

    /// One streaming token. Pieces are complete UTF-8, so the string is built directly from the
    /// stored bytes. Absent ids and out-of-range ids are `.absent`.
    enum StreamPiece: Sendable {
        case absent
        case fallback(UInt8)
        case text(String)
    }

    func streamPiece(_ id: Int, lead: Bool) -> StreamPiece {
        guard id >= 0, id < count else { return .absent }
        let word = offsets[id]
        if word & Self.absentBit != 0 { return .absent }
        if word & Self.fallbackBit != 0 {
            return .fallback(storage[Int(word & Self.offsetMask)])
        }
        if lead, let leadOffsets, let leadStorage {
            let start = Int(leadOffsets[id] & Self.offsetMask)
            let end = Int(leadOffsets[id + 1] & Self.offsetMask)
            return .text(Self.spelling(leadStorage, start, end))
        }
        let start = Int(word & Self.offsetMask)
        let end = Int(offsets[id + 1] & Self.offsetMask)
        return .text(Self.spelling(storage, start, end))
    }

    private static func spelling(_ bytes: [UInt8], _ start: Int, _ end: Int) -> String {
        bytes.withUnsafeBufferPointer { buffer in
            String(decoding: UnsafeBufferPointer(rebasing: buffer[start..<end]), as: UTF8.self)
        }
    }

    /// Present-id bytes for streaming. `fallback` is set for a byte-fallback id, and `bytes`
    /// is then empty. Returns nil when `id` is out of range or the reference path would skip it.
    /// `lead` selects the first-token spelling.
    func withDecoded<R>(
        _ id: Int, lead: Bool, _ body: (UnsafeBufferPointer<UInt8>, _ fallback: UInt8?) -> R
    ) -> R? {
        guard id >= 0, id < count else { return nil }
        let word = offsets[id]
        if word & Self.absentBit != 0 { return nil }
        if word & Self.fallbackBit != 0 {
            return body(UnsafeBufferPointer(start: nil, count: 0), storage[Int(word & Self.offsetMask)])
        }
        if lead, let leadOffsets, let leadStorage {
            let start = Int(leadOffsets[id] & Self.offsetMask)
            let end = Int(leadOffsets[id + 1] & Self.offsetMask)
            return leadStorage.withUnsafeBufferPointer { buffer in
                body(UnsafeBufferPointer(rebasing: buffer[start..<end]), nil)
            }
        }
        let start = Int(word & Self.offsetMask)
        let end = Int(offsets[id + 1] & Self.offsetMask)
        return storage.withUnsafeBufferPointer { buffer in
            body(UnsafeBufferPointer(rebasing: buffer[start..<end]), nil)
        }
    }

    private func concatPlain(_ tokens: [Int], skipping skipped: Set<Int>, cleanup: Bool) -> String {
        guard !tokens.isEmpty else { return "" }
        let skip = !skipped.isEmpty
        return tokens.withUnsafeBufferPointer { ids in
            offsets.withUnsafeBufferPointer { offsetBuffer in
                storage.withUnsafeBufferPointer { storageBuffer in
                    let ob = offsetBuffer.baseAddress!
                    var capacity = 0
                    for index in 0..<ids.count {
                        let id = ids[index]
                        if id < 0 || id >= count { continue }
                        let word = ob[id]
                        if word & Self.absentBit != 0 { continue }
                        if skip, skipped.contains(id) { continue }
                        capacity += Int((ob[id + 1] & Self.offsetMask) &- (word & Self.offsetMask))
                    }
                    guard capacity > 0 else { return "" }
                    let sb = storageBuffer.baseAddress!
                    return String(unsafeUninitializedCapacity: capacity) { output in
                        let base = output.baseAddress!
                        var written = 0
                        for index in 0..<ids.count {
                            let id = ids[index]
                            if id < 0 || id >= count { continue }
                            let word = ob[id]
                            if word & Self.absentBit != 0 { continue }
                            if skip, skipped.contains(id) { continue }
                            let start = Int(word & Self.offsetMask)
                            let length = Int((ob[id + 1] & Self.offsetMask) &- UInt32(start))
                            if length > 0 {
                                (base + written).initialize(from: sb + start, count: length)
                                written += length
                            }
                        }
                        return Self.applyTail(base, written: written, strip: strip, cleanup: cleanup)
                    }
                }
            }
        }
    }

    private func concatLead(_ tokens: [Int], skipping skipped: Set<Int>, cleanup: Bool) -> String {
        guard !tokens.isEmpty, let leadOffsets, let leadStorage else { return "" }
        let skip = !skipped.isEmpty
        return tokens.withUnsafeBufferPointer { ids in
            offsets.withUnsafeBufferPointer { offsetBuffer in
                leadOffsets.withUnsafeBufferPointer { leadOffsetBuffer in
                    storage.withUnsafeBufferPointer { storageBuffer in
                        leadStorage.withUnsafeBufferPointer { leadBuffer in
                            let ob = offsetBuffer.baseAddress!
                            let lb = leadOffsetBuffer.baseAddress!
                            let sb = storageBuffer.baseAddress
                            let leadBase = leadBuffer.baseAddress
                            var capacity = 0
                            var leading = true
                            for index in 0..<ids.count {
                                let id = ids[index]
                                if id < 0 || id >= count || ob[id] & Self.absentBit != 0 { continue }
                                if skip, skipped.contains(id) { continue }
                                if leading {
                                    leading = false
                                    let start = lb[id] & Self.offsetMask
                                    capacity += Int((lb[id + 1] & Self.offsetMask) &- start)
                                } else {
                                    let start = ob[id] & Self.offsetMask
                                    capacity += Int((ob[id + 1] & Self.offsetMask) &- start)
                                }
                            }
                            guard capacity > 0 else { return "" }
                            return String(unsafeUninitializedCapacity: capacity) { output in
                                let base = output.baseAddress!
                                var written = 0
                                var leading = true
                                for index in 0..<ids.count {
                                    let id = ids[index]
                                    if id < 0 || id >= count || ob[id] & Self.absentBit != 0 { continue }
                                    if skip, skipped.contains(id) { continue }
                                    let source: UnsafePointer<UInt8>?
                                    let start: Int
                                    let length: Int
                                    if leading {
                                        leading = false
                                        start = Int(lb[id] & Self.offsetMask)
                                        length = Int((lb[id + 1] & Self.offsetMask) &- UInt32(start))
                                        source = leadBase
                                    } else {
                                        start = Int(ob[id] & Self.offsetMask)
                                        length = Int((ob[id + 1] & Self.offsetMask) &- UInt32(start))
                                        source = sb
                                    }
                                    if length > 0, let source {
                                        (base + written).initialize(from: source + start, count: length)
                                        written += length
                                    }
                                }
                                return Self.applyTail(base, written: written, strip: nil, cleanup: cleanup)
                            }
                        }
                    }
                }
            }
        }
    }

    private func concatFallback(_ tokens: [Int], skipping skipped: Set<Int>, cleanup: Bool) -> String {
        guard !tokens.isEmpty else { return "" }
        let skip = !skipped.isEmpty
        return tokens.withUnsafeBufferPointer { ids in
            offsets.withUnsafeBufferPointer { offsetBuffer in
                storage.withUnsafeBufferPointer { storageBuffer in
                    let ob = offsetBuffer.baseAddress!
                    var capacity = 0
                    for index in 0..<ids.count {
                        let id = ids[index]
                        if id < 0 || id >= count { continue }
                        let word = ob[id]
                        if word & Self.absentBit != 0 { continue }
                        if skip, skipped.contains(id) { continue }
                        if word & Self.fallbackBit != 0 {
                            capacity += 3
                        } else {
                            let start = word & Self.offsetMask
                            capacity += Int((ob[id + 1] & Self.offsetMask) &- start)
                        }
                    }
                    guard capacity > 0 else { return "" }
                    let sb = storageBuffer.baseAddress!
                    return String(unsafeUninitializedCapacity: capacity) { output in
                        let base = output.baseAddress!
                        var written = 0
                        // Only allocated when the sequence actually contains a fallback byte.
                        var run: [UInt8] = []
                        func flushRun() {
                            guard !run.isEmpty else { return }
                            let valid = run.withUnsafeBufferPointer { buffer in
                                guard let start = buffer.baseAddress else { return true }
                                return Self.isUTF8(start, count: buffer.count)
                            }
                            if valid {
                                run.withUnsafeBufferPointer { buffer in
                                    guard let start = buffer.baseAddress, buffer.count > 0 else { return }
                                    (base + written).initialize(from: start, count: buffer.count)
                                    written += buffer.count
                                }
                            } else {
                                for _ in run {
                                    (base + written).initialize(to: 0xEF)
                                    (base + written + 1).initialize(to: 0xBF)
                                    (base + written + 2).initialize(to: 0xBD)
                                    written += 3
                                }
                            }
                            run.removeAll(keepingCapacity: true)
                        }
                        for index in 0..<ids.count {
                            let id = ids[index]
                            if id < 0 || id >= count { continue }
                            let word = ob[id]
                            if word & Self.absentBit != 0 { continue }
                            if skip, skipped.contains(id) { continue }
                            if word & Self.fallbackBit != 0 {
                                run.append(sb[Int(word & Self.offsetMask)])
                                continue
                            }
                            flushRun()
                            let start = Int(word & Self.offsetMask)
                            let length = Int((ob[id + 1] & Self.offsetMask) &- UInt32(start))
                            if length > 0 {
                                (base + written).initialize(from: sb + start, count: length)
                                written += length
                            }
                        }
                        flushRun()
                        return Self.applyTail(base, written: written, strip: strip, cleanup: cleanup)
                    }
                }
            }
        }
    }

    /// Strips a leading scalar and compacts cleanup rules. Both shorten the buffer.
    private static func applyTail(
        _ base: UnsafeMutablePointer<UInt8>, written: Int, strip: (scalar: UInt8, count: Int)?, cleanup: Bool
    ) -> Int {
        var written = written
        if let strip {
            var remaining = strip.count
            var removed = 0
            while removed < written, remaining > 0, base[removed] == strip.scalar {
                removed += 1
                remaining -= 1
            }
            if removed > 0 {
                memmove(base, base + removed, written - removed)
                written -= removed
            }
        }
        if cleanup, written > 0 {
            written = TokenizationCleanup.compact(base, count: written, wordPiece: false)
        }
        return written
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
