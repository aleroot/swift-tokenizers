// The encode pipeline over raw UTF-8: added-token splitting → normalization →
// pre-tokenization → model, with no intermediate strings.
//
// `EncodePipeline` owns the immutable stage configuration; `EncodeScratch` holds the buffers
// one encode call needs (pooled per tokenizer so steady-state encoding allocates only the
// result array). `PreTokenizationRunner` executes the pre-tokenization stages over lists of
// byte ranges and is shared with the public `[String]` pre-tokenizer API.

import Foundation

// MARK: - Pre-tokenization runner

/// Executes ``PreTokenizationStage``s over a chunk of UTF-8 text.
struct PreTokenizationRunner: Sendable {
    /// The stages, with a non-final `byteLevel` marker compiled into a rewrite that
    /// materialises the alphabet form (the following stages then see plain text, as in
    /// `tokenizers`).
    let stages: [PreTokenizationStage]

    init(stages: [PreTokenizationStage]) {
        var compiled: [PreTokenizationStage] = []
        for (index, stage) in stages.enumerated() {
            if case .byteLevel = stage, index != stages.count - 1 {
                compiled.append(.rewrite(ByteLevelMaterializer()))
            } else {
                compiled.append(stage)
            }
        }
        self.stages = compiled
    }

    /// Calls `body(piece, byteLevel)` for every piece of `bytes`, in order.
    /// - Parameter options: `.firstSection` applies to the first piece only, as in
    ///   `tokenizers` (Metaspace's `prepend_scheme: first` checks the original offset).
    func run(
        _ bytes: UnsafeBufferPointer<UInt8>, options: PreTokenizerOptions, scratch: ScratchBuffers,
        _ body: (UnsafeBufferPointer<UInt8>, Bool) -> Void
    ) {
        var current = scratch.takeRanges()
        var next = scratch.takeRanges()
        // After a rewrite stage the ranges index `text` instead of the input.
        var text: [UInt8] = []
        var rewritten = false
        defer {
            scratch.recycle(current)
            scratch.recycle(next)
            if rewritten { scratch.recycle(text) }
        }
        current.append(0..<bytes.count)
        var byteLevel = false
        let rest = options.subtracting([.firstSection])

        for stage in stages {
            switch stage {
            case .byteLevel:
                byteLevel = true
                continue
            case let .split(splitter):
                next.removeAll(keepingCapacity: true)
                if rewritten {
                    text.withUnsafeBufferPointer { Self.split(splitter, $0, current, options, rest, into: &next) }
                } else {
                    Self.split(splitter, bytes, current, options, rest, into: &next)
                }
            case let .rewrite(rewriter):
                next.removeAll(keepingCapacity: true)
                var output = scratch.take()
                if rewritten {
                    text.withUnsafeBufferPointer {
                        Self.rewrite(rewriter, $0, current, options, rest, into: &output, pieces: &next)
                    }
                    scratch.recycle(text)
                } else {
                    Self.rewrite(rewriter, bytes, current, options, rest, into: &output, pieces: &next)
                }
                text = output
                rewritten = true
            }
            swap(&current, &next)
        }

        if rewritten {
            text.withUnsafeBufferPointer { text in
                for range in current where !range.isEmpty {
                    body(UnsafeBufferPointer(rebasing: text[range]), byteLevel)
                }
            }
        } else {
            for range in current where !range.isEmpty {
                body(UnsafeBufferPointer(rebasing: bytes[range]), byteLevel)
            }
        }
    }

    /// Splits every piece of `text`, appending absolute ranges to `output`.
    @inline(__always)
    private static func split(
        _ splitter: any ByteSplitter, _ text: UnsafeBufferPointer<UInt8>, _ pieces: [Range<Int>],
        _ first: PreTokenizerOptions, _ rest: PreTokenizerOptions, into output: inout [Range<Int>]
    ) {
        for (index, range) in pieces.enumerated() {
            let mark = output.count
            splitter.split(UnsafeBufferPointer(rebasing: text[range]), options: index == 0 ? first : rest, into: &output)
            Self.offset(&output, from: mark, by: range.lowerBound)
        }
    }

    /// Rewrites every piece of `text` into `rewritten`, appending absolute ranges to `output`.
    @inline(__always)
    private static func rewrite(
        _ rewriter: any ByteRewriter, _ text: UnsafeBufferPointer<UInt8>, _ pieces: [Range<Int>],
        _ first: PreTokenizerOptions, _ rest: PreTokenizerOptions, into rewritten: inout [UInt8],
        pieces output: inout [Range<Int>]
    ) {
        for (index, range) in pieces.enumerated() {
            let base = rewritten.count
            let mark = output.count
            rewriter.rewrite(
                UnsafeBufferPointer(rebasing: text[range]), options: index == 0 ? first : rest, into: &rewritten,
                pieces: &output)
            Self.offset(&output, from: mark, by: base)
        }
    }

    @inline(__always)
    private static func offset(_ ranges: inout [Range<Int>], from start: Int, by delta: Int) {
        guard delta != 0 else { return }
        for k in start..<ranges.count {
            ranges[k] = ranges[k].lowerBound + delta..<ranges[k].upperBound + delta
        }
    }

    /// A `ByteLevel` stage that is followed by further stages: maps the text through the
    /// byte-level alphabet so the later stages operate on the mapped characters.
    struct ByteLevelMaterializer: ByteRewriter {
        init() {}
        init(config: Config) { self.init() }

        func rewrite(
            _ bytes: UnsafeBufferPointer<UInt8>, options: PreTokenizerOptions, into output: inout [UInt8],
            pieces: inout [Range<Int>]
        ) {
            let base = output.count
            ByteLevelAlphabet.appendEncoded(bytes, to: &output)
            pieces.append(0..<(output.count - base))
        }
    }
}

/// A small pool of reusable buffers handed to pipeline stages for intermediate text and
/// piece lists.
final class ScratchBuffers {
    private var freeBytes: [[UInt8]] = []
    private var freeRanges: [[Range<Int>]] = []

    /// An empty byte buffer (with whatever capacity a previous user left in it).
    @inline(__always)
    func take() -> [UInt8] {
        guard var buffer = freeBytes.popLast() else { return [] }
        buffer.removeAll(keepingCapacity: true)
        return buffer
    }

    /// Returns a buffer obtained from ``take()`` for reuse.
    @inline(__always)
    func recycle(_ buffer: [UInt8]) {
        freeBytes.append(buffer)
    }

    /// An empty range list.
    @inline(__always)
    func takeRanges() -> [Range<Int>] {
        guard var ranges = freeRanges.popLast() else { return [] }
        ranges.removeAll(keepingCapacity: true)
        return ranges
    }

    @inline(__always)
    func recycle(_ ranges: [Range<Int>]) {
        freeRanges.append(ranges)
    }
}

// MARK: - Pipeline

/// The stages between raw text and token ids (everything except post-processing).
struct EncodePipeline: Sendable {
    /// Splits around added tokens matched on the raw text.
    let splitter: AddedTokenSplitter?
    let normalizer: (any ByteNormalizer)?
    /// Splits around added tokens declared `normalized: true`, matched after normalization.
    let normalizedSplitter: AddedTokenSplitter?
    let preTokenizer: PreTokenizationRunner?
    /// Collapse runs of the unknown id inside each section (`fuse_unk`, WordPiece models).
    let fuseUnknownId: Int?

    /// Calls `onToken(id)` for every added token and `onPiece(bytes, byteLevel)` for every
    /// pre-tokenized piece of `text`, in input order.
    func run(
        _ bytes: UnsafeBufferPointer<UInt8>, scratch: EncodeScratch,
        onToken: (Int) -> Void, onPiece: (UnsafeBufferPointer<UInt8>, Bool) -> Void
    ) {
        scratch.sections.removeAll(keepingCapacity: true)
        if let splitter {
            splitter.split(bytes: bytes, into: &scratch.sections)
        } else if !bytes.isEmpty {
            scratch.sections.append(.text(0..<bytes.count))
        }

        for section in scratch.sections {
            switch section {
            case let .token(id):
                onToken(id)
            case let .text(range):
                // `firstSection`: the text starts the input (`tokenizers` checks original offset 0).
                let options: PreTokenizerOptions = range.lowerBound == 0 ? [.firstSection] : []
                let raw = UnsafeBufferPointer(rebasing: bytes[range])
                if let normalizer, !normalizer.isIdentity(on: raw) {
                    scratch.normalized.removeAll(keepingCapacity: true)
                    normalizer.normalize(raw, into: &scratch.normalized, scratch: scratch.buffers)
                    scratch.normalized.withUnsafeBufferPointer { normalized in
                        preTokenizeNormalized(normalized, options: options, scratch: scratch, onToken, onPiece)
                    }
                } else {
                    preTokenizeNormalized(raw, options: options, scratch: scratch, onToken, onPiece)
                }
            }
        }
    }

    /// Splits normalized text around `normalized: true` added tokens, then pre-tokenizes.
    @inline(__always)
    private func preTokenizeNormalized(
        _ normalized: UnsafeBufferPointer<UInt8>, options: PreTokenizerOptions, scratch: EncodeScratch,
        _ onToken: (Int) -> Void, _ onPiece: (UnsafeBufferPointer<UInt8>, Bool) -> Void
    ) {
        guard !normalized.isEmpty else { return }
        guard let normalizedSplitter else {
            preTokenize(normalized, options: options, scratch: scratch, onPiece)
            return
        }
        scratch.subsections.removeAll(keepingCapacity: true)
        normalizedSplitter.split(bytes: normalized, into: &scratch.subsections)
        for subsection in scratch.subsections {
            switch subsection {
            case let .token(id): onToken(id)
            case let .text(subrange):
                preTokenize(
                    UnsafeBufferPointer(rebasing: normalized[subrange]),
                    options: subrange.lowerBound == 0 ? options : [], scratch: scratch, onPiece)
            }
        }
    }

    @inline(__always)
    private func preTokenize(
        _ bytes: UnsafeBufferPointer<UInt8>, options: PreTokenizerOptions, scratch: EncodeScratch,
        _ onPiece: (UnsafeBufferPointer<UInt8>, Bool) -> Void
    ) {
        if let preTokenizer {
            preTokenizer.run(bytes, options: options, scratch: scratch.buffers, onPiece)
        } else {
            onPiece(bytes, false)
        }
    }

    /// Encodes `text` to ids (without post-processing) using `encoder`.
    func encode(_ text: String, encoder: PieceEncoder, scratch: EncodeScratch) -> [Int] {
        var copy = text
        return copy.withUTF8 { bytes -> [Int] in
            var ids: [Int] = []
            ids.reserveCapacity(bytes.count / 3 + 4)
            var sectionStart = 0
            run(
                bytes, scratch: scratch,
                onToken: { id in
                    if let fuseUnknownId { Self.fuseUnknown(&ids, from: sectionStart, unknownId: fuseUnknownId) }
                    ids.append(id)
                    sectionStart = ids.count
                },
                onPiece: { piece, byteLevel in
                    encoder.encode(bytes: piece, byteLevel: byteLevel, into: &ids)
                })
            if let fuseUnknownId { Self.fuseUnknown(&ids, from: sectionStart, unknownId: fuseUnknownId) }
            return ids
        }
    }

    /// Collapses runs of the unknown id inside `ids[from...]` into a single id.
    static func fuseUnknown(_ ids: inout [Int], from start: Int, unknownId: Int) {
        guard start < ids.count else { return }
        var write = start
        var previousIsUnknown = false
        for read in start..<ids.count {
            let id = ids[read]
            let isUnknown = id == unknownId
            if isUnknown, previousIsUnknown { continue }
            ids[write] = id
            write += 1
            previousIsUnknown = isUnknown
        }
        ids.removeSubrange(write...)
    }
}

/// Buffers for one encode call. Obtained from ``EncodeScratchPool``.
final class EncodeScratch {
    var sections: [AddedTokenSplitter.ByteSection] = []
    var subsections: [AddedTokenSplitter.ByteSection] = []
    var normalized: [UInt8] = []
    let buffers = ScratchBuffers()
}

/// A lock-protected free list of ``EncodeScratch`` objects so concurrent callers never share
/// buffers, while a single caller reuses the same allocation across calls.
final class EncodeScratchPool: @unchecked Sendable {
    private let free = Locked<[EncodeScratch]>([])

    @inline(__always)
    func take() -> EncodeScratch {
        free.withLock { $0.popLast() } ?? EncodeScratch()
    }

    @inline(__always)
    func recycle(_ scratch: EncodeScratch) {
        free.withLock { pool in
            if pool.count < 4 { pool.append(scratch) }
        }
    }
}
