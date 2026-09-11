import Foundation

/// Per-call state. Reuses the model's merge/lattice algorithms and, where token lengths
/// are exact, its existing ID cache. Nothing is retained by an IDs-only tokenizer call.
final class OffsetModelEncoder {
    let model: any TokenizingModel
    private var symbols: [BPETokenizer.Symbol] = []
    private var mergeScratch = BPETokenizer.MergeScratch()
    private var lattice: UnigramTokenizer.Lattice?
    private var cachedEncoder: PieceEncoder?
    private var ids: [Int] = []
    private let cachedByteLengths: Bool

    init(model: any TokenizingModel) {
        self.model = model
        if let bpe = model as? BPETokenizer {
            cachedByteLengths = !bpe.hasAffixes && bpe.symbolCount == bpe.vocab.count
                && bpe.byteSymbolIds.allSatisfy { $0 >= 0 }
        } else { cachedByteLengths = false }
    }

    deinit { cachedEncoder?.finish() }

    private func encodeIDs(_ bytes: UnsafeBufferPointer<UInt8>, using model: any FastTokenizingModel) {
        if cachedEncoder == nil { cachedEncoder = model.makeEncoder(); cachedEncoder?.begin() }
        ids.removeAll(keepingCapacity: true)
        cachedEncoder?.encode(bytes: bytes, byteLevel: false, into: &ids)
    }

    func encode(_ text: AlignedText, byteLevel: Bool, into output: inout [AlignedToken]) throws {
        guard !text.bytes.isEmpty else { return }
        if let bpe = model as? BPETokenizer {
            text.bytes.withUnsafeBufferPointer { bytes in
                if byteLevel, cachedByteLengths {
                    if cachedEncoder == nil { cachedEncoder = bpe.makeEncoder(); cachedEncoder?.begin() }
                    ids.removeAll(keepingCapacity: true)
                    cachedEncoder?.encode(bytes: bytes, byteLevel: true, into: &ids)
                    var start = 0
                    for id in ids {
                        let length = bpe.vocab.withBytes(of: id) { $0.reduce(0) { $0 + ($1 & 0xC0 == 0x80 ? 0 : 1) } }
                        let end = start + length
                        output.append(AlignedToken(id: id, offset: text.sourceRange(start..<end)))
                        start = end
                    }
                    return
                }
                if bpe.ignoreMerges {
                    let spelling = byteLevel ? ByteLevelAlphabet.encode(bytes) : String(decoding: bytes, as: UTF8.self)
                    if let id = bpe.modelVocabulary.id(of: spelling) {
                        output.append(AlignedToken(id: id, offset: text.sourceRange(0..<bytes.count)))
                        return
                    }
                }
                if byteLevel { bpe.byteLevelSymbols(bytes, into: &symbols) }
                else { bpe.scalarSymbols(bytes, into: &symbols) }
                if bpe.hasAffixes || symbols.contains(where: { $0.id < 0 }) {
                    bpe.referenceSymbols(bytes, byteLevel: byteLevel, into: &symbols)
                }
                bpe.merge(&symbols, scratch: &mergeScratch)
                for symbol in symbols {
                    let range = Int(symbol.start)..<Int(symbol.end)
                    let origin = text.sourceRange(range)
                    if bpe.isVocabularyId(symbol.id) {
                        output.append(AlignedToken(id: Int(symbol.id), offset: origin))
                    } else if bpe.byteFallback {
                        let part = UnsafeBufferPointer(rebasing: bytes[range])
                        let fallback = byteLevel ? Array(ByteLevelAlphabet.encode(part).utf8) : Array(part)
                        for byte in fallback {
                            let id = bpe.hexaTokenIds[Int(byte)]
                            if id >= 0 { output.append(AlignedToken(id: Int(id), offset: origin)) }
                            else if let unknown = bpe.modelUnknownTokenId { output.append(AlignedToken(id: unknown, offset: origin)) }
                        }
                    } else if let unknown = bpe.modelUnknownTokenId {
                        output.append(AlignedToken(id: unknown, offset: origin))
                    }
                }
            }
        } else if let unigram = model as? UnigramTokenizer {
            let text = byteLevel ? text.byteLevel() : text
            if lattice == nil { lattice = UnigramTokenizer.Lattice() }
            text.bytes.withUnsafeBufferPointer { bytes in
                encodeIDs(bytes, using: unigram)
                if !ids.contains(where: { $0 == unigram.unknownTokenId }),
                   ids.reduce(0, { $0 + unigram.vocabulary.byteCount(of: $1) }) == bytes.count {
                    var start = 0
                    for id in ids {
                        let end = start + unigram.vocabulary.byteCount(of: id)
                        output.append(AlignedToken(id: id, offset: text.sourceRange(start..<end)))
                        start = end
                    }
                    return
                }
                for piece in unigram.segment(bytes, lattice: lattice!) {
                    let range = Int(piece.start)..<Int(piece.end)
                    if piece.isUnknown, unigram.byteFallback,
                       bytes[range].allSatisfy({ unigram.byteFallbackIds[Int($0)] >= 0 }) {
                        for i in range {
                            output.append(AlignedToken(id: unigram.byteFallbackIds[Int(bytes[i])], offset: text.sourceRange(range)))
                        }
                    } else {
                        output.append(AlignedToken(id: Int(piece.tokenId), offset: text.sourceRange(range),
                            spelling: Int(piece.tokenId) == unigram.unknownTokenId ? String(decoding: bytes[range], as: UTF8.self) : nil))
                    }
                }
            }
        } else if let wordLevel = model as? WordLevelTokenizer {
            // One token per chunk, covering all of it.
            let text = byteLevel ? text.byteLevel() : text
            text.bytes.withUnsafeBufferPointer { bytes in
                guard let id = wordLevel.id(of: bytes) ?? wordLevel.unknownTokenId else { return }
                output.append(AlignedToken(id: id, offset: text.sourceRange(0..<bytes.count)))
            }
        } else if let bert = model as? BertTokenizer {
            let text = byteLevel ? text.byteLevel() : text
            if bert.serializedWordPiece {
                text.bytes.withUnsafeBufferPointer { encodeIDs($0, using: bert) }
                bert.appendOffsets(text, ids: ids, into: &output)
            } else { try bert.encode(text, into: &output) }
        } else {
            throw TokenizerError.unsupportedComponent("offsets for model \(type(of: model))")
        }
    }
}

extension EncodePipeline {
    /// Original positions are required by Metaspace's `first` rule even for IDs-only encoding.
    func runAligned(
        _ source: AlignedText, onToken: (Int, AlignedText) -> Void,
        onPiece: (AlignedText, Bool) throws -> Void
    ) throws {
        // Split callbacks cannot throw; collect only the inexpensive section descriptors.
        func sections(_ text: AlignedText, _ splitter: AddedTokenSplitter?) -> [(Range<Int>, Int?)] {
            guard let splitter else { return text.bytes.isEmpty ? [] : [(0..<text.bytes.count, nil)] }
            var result: [(Range<Int>, Int?)] = []
            text.bytes.withUnsafeBufferPointer { bytes in
                splitter.scan(bytes: bytes, onText: { result.append(($0, nil)) }, onToken: { result.append(($1, $0)) })
            }
            return result
        }
        for (range, token) in sections(source, splitter) {
            if let token { onToken(token, source.slice(range)); continue }
            var part = source.slice(range)
            part = try part.normalized(by: normalizer)
            for (subrange, token) in sections(part, normalizedSplitter) {
                if let token { onToken(token, part.slice(subrange)); continue }
                let flags: PreTokenizerFlags = part.sourceRange(subrange).lowerBound == 0 ? .firstSection : []
                let piece = part.slice(subrange)
                try preTokenizer.runAligned(piece, flags: flags) { piece, byteLevel in
                    try onPiece(piece, byteLevel)
                }
            }
        }
    }

    func encode(_ text: String, model: any TokenizingModel) throws -> [AlignedToken] {
        let source = AlignedText(text)
        let encoder = OffsetModelEncoder(model: model)
        var output: [AlignedToken] = []
        output.reserveCapacity(source.bytes.count / 3 + 4)
        var sectionStart = 0
        func fuse() {
            guard let unknown = fuseUnknownId, sectionStart < output.count else { return }
            var write = sectionStart
            for read in sectionStart..<output.count {
                let token = output[read]
                if write > sectionStart, token.id == unknown, output[write - 1].id == unknown,
                   let previous = output[write - 1].offset, let current = token.offset {
                    output[write - 1].offset = previous.lowerBound..<max(previous.upperBound, current.upperBound)
                } else { output[write] = token; write += 1 }
            }
            output.removeSubrange(write...)
        }
        try runAligned(
            source,
            onToken: { id, piece in
                fuse()
                output.append(
                    AlignedToken(id: id, offset: piece.sourceRange(0..<piece.bytes.count), spelling: piece.text))
                sectionStart = output.count
            },
            onPiece: { piece, byteLevel in
                try encoder.encode(piece, byteLevel: byteLevel, into: &output)
            })
        fuse()
        // Byte tokens can end inside a UTF-8 scalar. Expose the complete original scalar
        // for each of them, matching HF's character offsets and producing safe UI ranges.
        source.bytes.withUnsafeBufferPointer { bytes in
            for i in output.indices {
                guard let range = output[i].offset, !range.isEmpty else { continue }
                var start = range.lowerBound
                var end = range.upperBound
                while start > 0, bytes[start] & 0xC0 == 0x80 { start -= 1 }
                while end < bytes.count, bytes[end] & 0xC0 == 0x80 { end += 1 }
                output[i].offset = start..<end
            }
        }
        return output
    }
}

extension PreTokenizationRunner {
    func runAligned(_ text: AlignedText, flags: PreTokenizerFlags,
                    _ body: (AlignedText, Bool) throws -> Void) throws {
        // No stages: the text is its own single piece, empty or not, exactly as for a tokenizer
        // that declares no pre-tokenizer.
        guard !stages.isEmpty else { return try body(text, false) }
        guard rewritesText else {
            // Only a trailing `byteLevel` marker can appear here: a marker followed by more
            // stages is compiled into a rewrite, which is what `rewritesText` reports.
            let byteLevel = stages.last?.shape == .byteLevel
            var current = [0..<text.bytes.count]
            var next: [Range<Int>] = []
            try text.bytes.withUnsafeBufferPointer { bytes in
                for stage in stages where stage.shape == .split {
                    next.removeAll(keepingCapacity: true)
                    for range in current {
                        let mark = next.count
                        stage.split(UnsafeBufferPointer(rebasing: bytes[range]),
                            flags: range.lowerBound == 0 ? flags : [], into: &next)
                        if range.lowerBound != 0 {
                            for i in mark..<next.count {
                                next[i] = next[i].lowerBound + range.lowerBound..<next[i].upperBound + range.lowerBound
                            }
                        }
                    }
                    swap(&current, &next)
                }
                for range in current where !range.isEmpty { try body(text.slice(range), byteLevel) }
            }
            return
        }
        for (piece, byteLevel) in try alignedPieces(text, flags: flags) { try body(piece, byteLevel) }
    }

    func alignedPieces(_ text: AlignedText, flags: PreTokenizerFlags) throws -> [(AlignedText, Bool)] {
        var pieces: [(AlignedText, PreTokenizerFlags)] = [(text, flags)]
        var byteLevel = false
        for stage in stages {
            switch stage.shape {
            case .byteLevel: byteLevel = true
            case .split:
                pieces = pieces.flatMap { text, flags in
                    var ranges: [Range<Int>] = []
                    text.bytes.withUnsafeBufferPointer { stage.split($0, flags: flags, into: &ranges) }
                    return ranges.filter { !$0.isEmpty }.map {
                        (text.slice($0), text.sourceRange($0).lowerBound == 0 ? PreTokenizerFlags.firstSection : [])
                    }
                }
            case .rewrite:
                guard let rewriter = stage.source as? any ByteRewriter else {
                    throw TokenizerError.unsupportedComponent("offsets for pre-tokenizer \(type(of: stage))")
                }
                var next: [(AlignedText, PreTokenizerFlags)] = []
                for (text, flags) in pieces {
                    var bytes: [UInt8] = []
                    var ranges: [Range<Int>] = []
                    text.bytes.withUnsafeBufferPointer { rewriter.rewrite($0, flags: flags, into: &bytes, pieces: &ranges) }
                    var origins: [Range<Int>] = []
                    origins.reserveCapacity(bytes.count)
                    switch rewriter {
                    case is ByteLevelMaterializer:
                        text.bytes.withUnsafeBufferPointer { input in
                            var i = 0
                            while i < input.count {
                                let width = UTF8Cursor.width(input[i])
                                let origin = text.sourceRange(i..<i + width)
                                for byte in input[i..<i + width] {
                                    let scalar = Unicode.Scalar(ByteLevelAlphabet.byteToScalar[Int(byte)])!
                                    origins.append(contentsOf: repeatElement(origin, count: scalar.utf8.count))
                                }
                                i += width
                            }
                        }
                    default:
                        let replacement: [UInt8]
                        let prefix: Bool
                        if let metaspace = rewriter as? MetaspacePreTokenizer {
                            replacement = Array(metaspace.stringReplacement.utf8)
                            let marker = Array(metaspace.replacement.utf8)
                            let needsPrefix = metaspace.prependScheme == .always
                                || metaspace.prependScheme == .first && flags.contains(.firstSection)
                            prefix = !text.bytes.isEmpty && needsPrefix && !text.bytes.starts(with: marker)
                                && !(text.bytes.first == 0x20 && replacement.starts(with: marker))
                        } else if rewriter is ByteLevelPreTokenizer.PrefixSpaceRewriter {
                            replacement = [0x20]
                            prefix = !text.bytes.isEmpty && text.bytes.first != 0x20
                        } else { throw TokenizerError.unsupportedComponent("offsets for pre-tokenizer \(type(of: rewriter))") }
                        if prefix, let first = text.bytes.first {
                            let origin = text.sourceRange(0..<UTF8Cursor.width(first))
                            origins.append(contentsOf: repeatElement(origin, count: replacement.count))
                        }
                        for (i, byte) in text.bytes.enumerated() {
                            let count = byte == 0x20 ? replacement.count : 1
                            origins.append(contentsOf: repeatElement(text.sourceRange(i..<i + 1), count: count))
                        }
                    }
                    guard bytes.count == origins.count else {
                        throw TokenizerError.invalidConfiguration("Pre-tokenizer produced an invalid alignment")
                    }
                    let rewritten = AlignedText(bytes: bytes[...], origins: origins[...], sourceStart: text.sourceStart)
                    next.append(contentsOf: ranges.filter { !$0.isEmpty }.map {
                            (
                                rewritten.slice($0),
                                rewritten.sourceRange($0).lowerBound == 0 ? PreTokenizerFlags.firstSection : []
                            )
                        })
                }
                pieces = next
            }
        }
        return pieces.map { ($0.0, byteLevel) }
    }
}
