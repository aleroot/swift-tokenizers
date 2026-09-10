// The encode pipeline over raw UTF-8: added-token splitting → normalization → pre-tokenization
// → model, with no intermediate strings. `EncodeScratch` holds the per-call buffers, pooled per
// tokenizer; `PreTokenizationRunner` executes the pre-tokenization stages over byte ranges.

import Foundation

// MARK: - Pre-tokenization runner

/// Executes compiled ``PreTokenizationStage``s over a chunk of UTF-8 text.
///
/// The runner owns its stages and lends them out by unmanaged reference, so running one is a
/// single virtual call and no reference counting. A tokenizer is shared by every thread that
/// encodes with it, and so are its stages: reading them out of an array per piece costs a
/// contended atomic pair per stage, which is what stops the pipeline scaling past a few threads.
final class PreTokenizationRunner: @unchecked Sendable {
    /// The stages, with a non-final `byteLevel` marker compiled into a rewrite that
    /// materialises the alphabet form (the following stages then see plain text, as in
    /// `tokenizers`).
    let stages: [PreTokenizationStage]
    /// ``stages`` as unmanaged references, for the hot loop to borrow. The array above is what
    /// keeps them alive.
    private let borrowed: UnsafeMutableBufferPointer<Unmanaged<PreTokenizationStage>>
    let needsOriginalStart: Bool
    let rewritesBeforeFirst: Bool
    /// Whether any stage replaces the text later stages see. The offset pipeline branches on it.
    let rewritesText: Bool

    init(stages: [PreTokenizationStage]) {
        var compiled: [PreTokenizationStage] = []
        for (index, stage) in stages.enumerated() {
            if stage.shape == .byteLevel, index != stages.count - 1 {
                compiled.append(RewriteStage(ByteLevelMaterializer()))
            } else {
                compiled.append(stage)
            }
        }
        self.stages = compiled
        let table = UnsafeMutableBufferPointer<Unmanaged<PreTokenizationStage>>.allocate(capacity: compiled.count)
        for (index, stage) in compiled.enumerated() { table[index] = .passUnretained(stage) }
        borrowed = table

        // Metaspace's `prepend_scheme: first` tests the offset in the *original* text, so the
        // stages have to be run through the alignment pipeline whenever anything rewrote the
        // text before that Metaspace.
        needsOriginalStart = compiled.contains { Self.prependsToFirstSectionOnly($0) }
        var rewritten = false
        var needsMapping = false
        for stage in compiled where stage.shape == .rewrite {
            if Self.prependsToFirstSectionOnly(stage) { needsMapping = needsMapping || rewritten }
            rewritten = true
        }
        rewritesText = rewritten
        rewritesBeforeFirst = needsMapping
    }

    deinit { borrowed.deallocate() }

    /// Whether `stage` is a Metaspace that prepends its replacement only to the first section.
    private static func prependsToFirstSectionOnly(_ stage: PreTokenizationStage) -> Bool {
        (stage.source as? MetaspacePreTokenizer)?.prependScheme == .first
    }

    /// Calls `body(piece, byteLevel)` for every piece of `bytes`, in order.
    /// - Parameter flags: `.firstSection` is passed on only to the piece that still starts at
    ///   offset 0 of the text, as in `tokenizers` (Metaspace's `prepend_scheme: first` checks
    ///   the original offset, so a leading-whitespace split loses it).
    func run(
        _ bytes: UnsafeBufferPointer<UInt8>, flags: PreTokenizerFlags, scratch: ScratchBuffers,
        _ body: (UnsafeBufferPointer<UInt8>, Bool) -> Void
    ) {
        // No stages: the text is its own single piece, as it is for a tokenizer that declares no
        // pre-tokenizer at all.
        guard !borrowed.isEmpty else { return body(bytes, false) }
        if rewritesBeforeFirst {
            let source = AlignedText(
                bytes: Array(bytes)[...], origins: nil,
                sourceStart: flags.contains(.firstSection) ? 0 : 1)
            if let pieces = try? alignedPieces(source, flags: flags) {
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
        var first = flags
        let rest = flags.subtracting(.firstSection)

        for index in 0..<borrowed.count {
            borrowed[index]._withUnsafeGuaranteedRef { stage in
                switch stage.shape {
                case .byteLevel:
                    byteLevel = true
                case .split:
                    next.removeAll(keepingCapacity: true)
                    if rewritten {
                        text.withUnsafeBufferPointer { Self.split(stage, $0, current, first, rest, into: &next) }
                    } else {
                        Self.split(stage, bytes, current, first, rest, into: &next)
                    }
                    swap(&current, &next)
                case .rewrite:
                    // Rewriters compact surviving pieces into a new buffer. Once a split has
                    // removed the original start, its new byte offset zero is not the start
                    // of the input. Ordinary split-only pipelines need no extra origin check.
                    if current.first?.lowerBound != 0 { first = rest }
                    next.removeAll(keepingCapacity: true)
                    var output = scratch.take()
                    if rewritten {
                        text.withUnsafeBufferPointer {
                            Self.rewrite(stage, $0, current, first, rest, into: &output, pieces: &next)
                        }
                        scratch.recycle(text)
                    } else {
                        Self.rewrite(stage, bytes, current, first, rest, into: &output, pieces: &next)
                    }
                    text = output
                    rewritten = true
                    swap(&current, &next)
                }
            }
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
        _ stage: PreTokenizationStage, _ text: UnsafeBufferPointer<UInt8>, _ pieces: [Range<Int>],
        _ first: PreTokenizerFlags, _ rest: PreTokenizerFlags, into output: inout [Range<Int>]
    ) {
        for range in pieces {
            let mark = output.count
            stage.split(
                UnsafeBufferPointer(rebasing: text[range]), flags: range.lowerBound == 0 ? first : rest,
                into: &output)
            offset(&output, from: mark, by: range.lowerBound)
        }
    }

    /// Rewrites every piece of `text` into `rewritten`, appending absolute ranges to `output`.
    @inline(__always)
    private static func rewrite(
        _ stage: PreTokenizationStage, _ text: UnsafeBufferPointer<UInt8>, _ pieces: [Range<Int>],
        _ first: PreTokenizerFlags, _ rest: PreTokenizerFlags, into rewritten: inout [UInt8],
        pieces output: inout [Range<Int>]
    ) {
        for range in pieces {
            let base = rewritten.count
            let mark = output.count
            stage.rewrite(
                UnsafeBufferPointer(rebasing: text[range]), flags: range.lowerBound == 0 ? first : rest,
                into: &rewritten, pieces: &output)
            offset(&output, from: mark, by: base)
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
            _ bytes: UnsafeBufferPointer<UInt8>, flags: PreTokenizerFlags, into output: inout [UInt8],
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
    /// Never `nil`: a tokenizer with no pre-tokenizer gets a runner with no stages, which hands
    /// back the text it is given as one piece. Reading an optional reference per section would
    /// cost a pair of contended reference-count updates on a shared tokenizer.
    let preTokenizer: PreTokenizationRunner
    /// Collapse runs of the unknown id inside each section (`fuse_unk`, WordPiece models).
    let fuseUnknownId: Int?

    /// Calls `onToken(id)` for every added token and `onPiece(bytes, byteLevel)` for every
    /// pre-tokenized piece of `text`, in input order.
    func run(
        _ bytes: UnsafeBufferPointer<UInt8>, scratch: EncodeScratch,
        onToken: (Int) -> Void, onPiece: (UnsafeBufferPointer<UInt8>, Bool) -> Void
    ) {
        if preTokenizer.needsOriginalStart, let normalizer, !normalizer.isIdentity(on: bytes) {
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
                let flags: PreTokenizerFlags = range.lowerBound == 0 ? .firstSection : []
                let raw = UnsafeBufferPointer(rebasing: bytes[range])
                if let normalizer, !normalizer.isIdentity(on: raw) {
                    scratch.normalized.removeAll(keepingCapacity: true)
                    normalizer.normalize(raw, into: &scratch.normalized, scratch: scratch.buffers)
                    scratch.normalized.withUnsafeBufferPointer { normalized in
                        preTokenizeNormalized(normalized, flags: flags, scratch: scratch, onToken, onPiece)
                    }
                } else {
                    preTokenizeNormalized(raw, flags: flags, scratch: scratch, onToken, onPiece)
                }
            }
        }
    }

    /// Splits normalized text around `normalized: true` added tokens, then pre-tokenizes.
    @inline(__always)
    private func preTokenizeNormalized(
        _ normalized: UnsafeBufferPointer<UInt8>, flags: PreTokenizerFlags, scratch: EncodeScratch,
        _ onToken: (Int) -> Void, _ onPiece: (UnsafeBufferPointer<UInt8>, Bool) -> Void
    ) {
        guard !normalized.isEmpty else { return }
        guard let normalizedSplitter else {
            preTokenizer.run(normalized, flags: flags, scratch: scratch.buffers, onPiece)
            return
        }
        scratch.subsections.removeAll(keepingCapacity: true)
        normalizedSplitter.split(bytes: normalized, into: &scratch.subsections)
        for subsection in scratch.subsections {
            switch subsection {
            case let .token(id): onToken(id)
            case let .text(subrange):
                preTokenizer.run(
                    UnsafeBufferPointer(rebasing: normalized[subrange]),
                    flags: subrange.lowerBound == 0 ? flags : [], scratch: scratch.buffers, onPiece)
            }
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

    /// The pooled encoder for the model `identity` names, created by `make` on first use.
    ///
    /// - Parameters:
    ///   - identity: the model's identity, which the tokenizer resolves once so the check costs
    ///     no reference counting. It is what makes reuse safe: a scratch outlives the tokenizer
    ///     it was lent to, so it can be handed to a different tokenizer whose model needs a
    ///     different encoder. Comparing identities is sound because the encoder holds its model,
    ///     so an equal identity is necessarily the same live object.
    ///   - make: builds the encoder, or answers `nil` when the model has no byte-level path. It
    ///     runs at most once per model per scratch, which is where the caller can afford to read
    ///     the model itself: the cached path copies no existential.
    @inline(__always)
    func encoder(identity: ObjectIdentifier, make: () -> PieceEncoder?) -> PieceEncoder? {
        if let encoder, encoderModel == identity { return encoder }
        guard let made = make() else { return nil }
        encoder = made
        encoderModel = identity
        return made
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
