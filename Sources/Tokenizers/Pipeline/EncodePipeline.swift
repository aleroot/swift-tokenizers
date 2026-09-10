// The encode pipeline over raw UTF-8: added-token splitting → normalization → pre-tokenization
// → model, with no intermediate strings. `EncodeScratch` holds the per-call buffers, pooled per
// tokenizer; `PreTokenizationRunner` executes the pre-tokenization stages over byte ranges.

import Foundation

// MARK: - Pre-tokenization runner

/// Executes ``PreTokenizationStage``s over a chunk of UTF-8 text.
struct PreTokenizationRunner: Sendable {
    /// The stages, with a non-final `byteLevel` marker compiled into a rewrite that
    /// materialises the alphabet form (the following stages then see plain text, as in
    /// `tokenizers`).
    let stages: [PreTokenizationStage]
    let needsOriginalStart: Bool
    let rewritesBeforeFirst: Bool

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
        needsOriginalStart = compiled.contains {
            if case let .rewrite(rewriter) = $0, let metaspace = rewriter as? MetaspacePreTokenizer {
                return metaspace.prependScheme == .first
            }
            return false
        }
        var rewritten = false
        var needsMapping = false
        for stage in compiled {
            if case let .rewrite(rewriter) = stage {
                if let metaspace = rewriter as? MetaspacePreTokenizer, metaspace.prependScheme == .first {
                    needsMapping = needsMapping || rewritten
                }
                rewritten = true
            }
        }
        rewritesBeforeFirst = needsMapping
    }

    /// Calls `body(piece, byteLevel)` for every piece of `bytes`, in order.
    /// - Parameter options: `.firstSection` is passed on only to the piece that still starts at
    ///   offset 0 of the text, as in `tokenizers` (Metaspace's `prepend_scheme: first` checks
    ///   the original offset, so a leading-whitespace split loses it).
    func run(
        _ bytes: UnsafeBufferPointer<UInt8>, options: PreTokenizerOptions, scratch: ScratchBuffers,
        _ body: (UnsafeBufferPointer<UInt8>, Bool) -> Void
    ) {
        if rewritesBeforeFirst {
            let source = AlignedText(
                bytes: Array(bytes)[...], origins: nil,
                sourceStart: options.contains(.firstSection) ? 0 : 1)
            if let pieces = try? alignedPieces(source, options: options) {
                for (piece, byteLevel) in pieces { piece.bytes.withUnsafeBufferPointer { body($0, byteLevel) } }
                return
            }
        }
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
        // Avoid allocating a `Set` per call for the common `[.firstSection]` / `[]` inputs.
        let rest: PreTokenizerOptions =
            options.count == 1 && options.contains(.firstSection) || options.isEmpty
            ? [] : options.subtracting([.firstSection])
        var first = options

        for stage in stages {
            switch stage {
            case .byteLevel:
                byteLevel = true
                continue
            case let .split(splitter):
                next.removeAll(keepingCapacity: true)
                if rewritten {
                    text.withUnsafeBufferPointer { Self.split(splitter, $0, current, first, rest, into: &next) }
                } else {
                    Self.split(splitter, bytes, current, first, rest, into: &next)
                }
            case let .rewrite(rewriter):
                // Rewriters compact surviving pieces into a new buffer. Once a split has
                // removed the original start, its new byte offset zero is not the start
                // of the input. Ordinary split-only pipelines need no extra origin check.
                if current.first?.lowerBound != 0 { first = rest }
                next.removeAll(keepingCapacity: true)
                var output = scratch.take()
                if rewritten {
                    text.withUnsafeBufferPointer {
                        Self.rewrite(rewriter, $0, current, first, rest, into: &output, pieces: &next)
                    }
                    scratch.recycle(text)
                } else {
                    Self.rewrite(rewriter, bytes, current, first, rest, into: &output, pieces: &next)
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
        for range in pieces {
            let mark = output.count
            splitter.split(
                UnsafeBufferPointer(rebasing: text[range]), options: range.lowerBound == 0 ? first : rest, into: &output
            )
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
        for range in pieces {
            let base = rewritten.count
            let mark = output.count
            rewriter.rewrite(
                UnsafeBufferPointer(rebasing: text[range]), options: range.lowerBound == 0 ? first : rest,
                into: &rewritten, pieces: &output)
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
/// piece lists. Owned by one encode call at a time, so exclusivity is not enforced dynamically.
final class ScratchBuffers {
    @exclusivity(unchecked) private var freeBytes: [[UInt8]] = []
    @exclusivity(unchecked) private var freeRanges: [[Range<Int>]] = []

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
        if preTokenizer?.needsOriginalStart == true, let normalizer, !normalizer.isIdentity(on: bytes) {
            // Only `first` needs provenance in the IDs-only path. Prepare before
            // calling consumers so a failed optional trace cannot emit partial IDs.
            var sections: [(Int?, AlignedText, Bool)] = []
            if (try? runAligned(
                AlignedText(String(decoding: bytes, as: UTF8.self)),
                onToken: {
                    sections.append(($0, $1, false))
                }, onPiece: { sections.append((nil, $0, $1)) })) != nil
            {
                for (id, piece, byteLevel) in sections {
                    if let id { onToken(id) } else { piece.bytes.withUnsafeBufferPointer { onPiece($0, byteLevel) } }
                }
                return
            }
        }
        scratch.reusable = bytes.count <= EncodeScratch.maximumReusableInputBytes
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

/// Buffers for one encode call, plus the model's ``PieceEncoder`` so its working state
/// (lattice, merge buffers, WordPiece scratch) survives across calls. Obtained from
/// ``EncodeScratchPool``; used by one caller at a time.
final class EncodeScratch {
    /// Reuse normal prompt/batch buffers without retaining document-sized outliers indefinitely.
    static let maximumReusableInputBytes = 1 << 20
    var reusable = true
    @exclusivity(unchecked) var sections: [AddedTokenSplitter.ByteSection] = []
    @exclusivity(unchecked) var subsections: [AddedTokenSplitter.ByteSection] = []
    @exclusivity(unchecked) var normalized: [UInt8] = []
    let buffers = ScratchBuffers()
    private var encoder: PieceEncoder?
    private var encoderModel: ObjectIdentifier?

    /// The pooled encoder for `model`, created on first use.
    ///
    /// - Parameter identity: the model's identity, which the tokenizer resolves once so the
    ///   check costs no reference counting. It is what makes reuse safe: a scratch outlives
    ///   the tokenizer it was lent to, so it can be handed to a different tokenizer whose
    ///   model needs a different encoder. Comparing identities is sound because the encoder
    ///   holds its model, so an equal identity is necessarily the same live object.
    @inline(__always)
    func encoder(for model: any FastTokenizingModel, identity: ObjectIdentifier?) -> PieceEncoder {
        if let encoder, let identity, encoderModel == identity { return encoder }
        let encoder = model.makeEncoder()
        self.encoder = encoder
        encoderModel = identity
        return encoder
    }
}

/// Lends each encoding thread its own ``EncodeScratch``, so concurrent callers never share
/// buffers and a repeat caller reuses the same allocations. Ownership is genuinely
/// thread-local: the hot path is one thread-specific load and no atomic. A pool carries no
/// state of its own; it is the identity under which its tokenizer's scratches are filed.
///
/// Striping a free list by the thread handle is not an option on Darwin, where handles are
/// evenly spaced stack addresses: hashing them collapses many threads onto a few stripes, and
/// a starved stripe reallocates the scratch (and with it the model's ``PieceEncoder``, its
/// lattice or merge buffers and its pretoken cache) on nearly every call.
final class EncodeScratchPool: @unchecked Sendable {
    /// Per-thread storage for every pool in the process, so the library holds one
    /// thread-specific key however many tokenizers are alive.
    private static let store = ThreadScratchStore()

    @inline(__always)
    func take() -> EncodeScratch {
        Self.store.take(owner: ObjectIdentifier(self))
    }

    @inline(__always)
    func recycle(_ scratch: EncodeScratch) {
        guard scratch.reusable else { return }
        Self.store.put(scratch, owner: ObjectIdentifier(self))
    }
}

/// The thread-specific side of ``EncodeScratchPool``: one cache per thread, holding the
/// scratch most recently lent to each of a few pools. Entries are keyed by pool because a
/// scratch caches a ``PieceEncoder`` built for one model and a thread may encode with several
/// tokenizers. A cache is released when its thread exits, so buffers never outlive their user
/// and the number of live scratch objects stays bounded by the number of encoding threads.
private final class ThreadScratchStore: @unchecked Sendable {
    /// How many tokenizers one thread can alternate between before evicting a scratch. Kept
    /// small because an entry holds a model's encoder, so a stale entry keeps that model alive
    /// until the thread encodes with enough other tokenizers to evict it, or exits.
    private static let entriesPerThread = 2

    /// A thread's scratches. Only ever touched by that thread, so it needs no synchronisation.
    private final class Cache {
        struct Entry {
            let owner: ObjectIdentifier
            let scratch: EncodeScratch
        }

        var entries = [Entry?](repeating: nil, count: ThreadScratchStore.entriesPerThread)
    }

    private let key: pthread_key_t

    init() {
        var key = pthread_key_t()
        // Darwin declares the destructor's argument non-optional, other platforms optional.
        #if canImport(Darwin)
            pthread_key_create(&key) { Unmanaged<Cache>.fromOpaque($0).release() }
        #else
            pthread_key_create(&key) { pointer in
                guard let pointer else { return }
                Unmanaged<Cache>.fromOpaque(pointer).release()
            }
        #endif
        self.key = key
    }

    /// Hands over the calling thread's scratch for `owner`, taking it out of the cache so that
    /// a nested encode on the same thread gets its own buffers rather than sharing these.
    @inline(__always)
    func take(owner: ObjectIdentifier) -> EncodeScratch {
        guard let cache = cache(creating: false) else { return EncodeScratch() }
        for index in 0..<Self.entriesPerThread {
            guard let entry = cache.entries[index], entry.owner == owner else { continue }
            cache.entries[index] = nil
            return entry.scratch
        }
        return EncodeScratch()
    }

    /// Returns `scratch` to the calling thread's cache, replacing this pool's entry if it has
    /// one, filling a free slot if there is one, and otherwise evicting the last entry.
    @inline(__always)
    func put(_ scratch: EncodeScratch, owner: ObjectIdentifier) {
        guard let cache = cache(creating: true) else { return }
        var slot = Self.entriesPerThread - 1
        for index in 0..<Self.entriesPerThread {
            guard let entry = cache.entries[index] else {
                slot = index
                break
            }
            if entry.owner == owner {
                slot = index
                break
            }
        }
        cache.entries[slot] = Cache.Entry(owner: owner, scratch: scratch)
    }

    /// The calling thread's cache, created on demand when `creating`.
    @inline(__always)
    private func cache(creating: Bool) -> Cache? {
        if let pointer = pthread_getspecific(key) {
            return Unmanaged<Cache>.fromOpaque(pointer).takeUnretainedValue()
        }
        guard creating else { return nil }
        let cache = Cache()
        pthread_setspecific(key, Unmanaged.passRetained(cache).toOpaque())
        return cache
    }
}
