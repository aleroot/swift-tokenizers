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
/// `@unchecked Sendable`: the private borrowed table is initialized once and never mutated;
/// `stages` strongly owns every entry through the runner's lifetime. Stages share no scratch.
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
    /// Never `nil`: a tokenizer that declares none gets ``IdentityNormalizer``, which answers
    /// `isIdentity` at once. Reading an optional existential per section costs a pair of
    /// contended reference-count updates on a shared tokenizer.
    let normalizer: any ByteNormalizer
    private let normalizerNeedsSourceOffset: Bool
    /// Splits around added tokens declared `normalized: true`, matched after normalization.
    let normalizedSplitter: AddedTokenSplitter?
    /// Never `nil`: a tokenizer with no pre-tokenizer gets a runner with no stages, which hands
    /// back the text it is given as one piece. Reading an optional reference per section would
    /// cost a pair of contended reference-count updates on a shared tokenizer.
    let preTokenizer: PreTokenizationRunner
    /// Collapse runs of the unknown id inside each section (`fuse_unk`, WordPiece models).
    let fuseUnknownId: Int?

    init(
        splitter: AddedTokenSplitter?, normalizer: any ByteNormalizer,
        normalizedSplitter: AddedTokenSplitter?, preTokenizer: PreTokenizationRunner, fuseUnknownId: Int?
    ) {
        self.splitter = splitter
        self.normalizer = normalizer
        normalizerNeedsSourceOffset = normalizer.needsSourceOffset
        self.normalizedSplitter = normalizedSplitter
        self.preTokenizer = preTokenizer
        self.fuseUnknownId = fuseUnknownId
    }

    /// Calls `onToken(id)` for every added token and `onPiece(bytes, byteLevel)` for every
    /// pre-tokenized piece of `text`, in input order.
    func run(
        _ bytes: UnsafeBufferPointer<UInt8>, scratch: EncodeScratch,
        onToken: (Int) -> Void, onPiece: (UnsafeBufferPointer<UInt8>, Bool) -> Void
    ) {
        if preTokenizer.needsOriginalStart, !normalizer.isIdentity(on: bytes) {
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
                let unchanged =
                    normalizerNeedsSourceOffset
                    ? normalizer.isIdentity(on: raw, at: range.lowerBound) : normalizer.isIdentity(on: raw)
                if !unchanged {
                    scratch.normalized.removeAll(keepingCapacity: true)
                    if normalizerNeedsSourceOffset {
                        normalizer.normalize(
                            raw, at: range.lowerBound, into: &scratch.normalized, scratch: scratch.buffers)
                    } else {
                        normalizer.normalize(raw, into: &scratch.normalized, scratch: scratch.buffers)
                    }
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
    private var offsetEncoder: OffsetModelEncoder?
    private var offsetEncoderModel: ObjectIdentifier?

    /// The pooled encoder for the model `identity` names, created by `make` on first use.
    ///
    /// - Parameters:
    ///   - identity: the model's identity, which the tokenizer resolves once so the check costs
    ///     no model reference counting. Cached encoders are specific to a model; the identity
    ///     check prevents reuse with another model. Comparing identities is sound because
    ///     the encoder holds its model,
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

    /// The pooled source-alignment encoder for the model `identity` names, under the same reuse
    /// rules as ``encoder(identity:make:)``. Offsets are opt-in, so a tokenizer that only ever
    /// returns IDs never builds one.
    @inline(__always)
    func offsetEncoder(
        identity: ObjectIdentifier, make: () throws -> OffsetModelEncoder
    ) rethrows
        -> OffsetModelEncoder
    {
        if let offsetEncoder, offsetEncoderModel == identity { return offsetEncoder }
        let made = try make()
        offsetEncoder = made
        offsetEncoderModel = identity
        return made
    }
}

/// Owns the reusable buffers for this tokenizer. TLS only keeps weak handles, so
/// releasing the tokenizer releases its models and scratch on every encoding thread.
final class EncodeScratchPool: Sendable {
    private static let store = ThreadScratchStore()
    private let slots = Locked<[ObjectIdentifier: Slot]>([:])

    /// The registry only retains slots; it never reads their scratch. Scratch is accessed
    /// exclusively by the thread that registered the slot, with active calls retaining it.
    fileprivate final class Slot: @unchecked Sendable {
        var scratch: EncodeScratch?
    }

    fileprivate func register() -> Slot {
        let slot = Slot()
        slots.withLock { $0[ObjectIdentifier(slot)] = slot }
        return slot
    }

    fileprivate func unregister(_ identity: ObjectIdentifier) {
        // Release buffers outside the registry lock.
        let removed = slots.withLock { $0.removeValue(forKey: identity) }
        withExtendedLifetime(removed) {}
    }

    @inline(__always)
    func take() -> EncodeScratch {
        guard let slot = Self.store.slot(for: self) else { return EncodeScratch() }
        let scratch = slot.scratch
        slot.scratch = nil  // Nested encodes must use separate buffers.
        return scratch ?? EncodeScratch()
    }

    @inline(__always)
    func recycle(_ scratch: EncodeScratch) {
        guard scratch.reusable else { return }
        Self.store.slot(for: self)?.scratch = scratch
    }
}

/// A bounded weak index into tokenizer-owned storage. Registration and eviction take
/// the owning pool's lock; the warm encode path only accesses its thread's slot.
private final class ThreadScratchStore: Sendable {
    private static let entriesPerThread = 2

    private final class Cache {
        final class Entry {
            let owner: ObjectIdentifier
            let identity: ObjectIdentifier
            weak var pool: EncodeScratchPool?
            weak var slot: EncodeScratchPool.Slot?

            init(pool: EncodeScratchPool, slot: EncodeScratchPool.Slot) {
                owner = ObjectIdentifier(pool)
                identity = ObjectIdentifier(slot)
                self.pool = pool
                self.slot = slot
            }

            deinit { pool?.unregister(identity) }
        }

        var entries = [Entry?](repeating: nil, count: ThreadScratchStore.entriesPerThread)
        var nextEviction = 0
    }

    private let key: pthread_key_t

    init() {
        var key = pthread_key_t()
        #if canImport(Darwin)
            let result = pthread_key_create(&key) { Unmanaged<Cache>.fromOpaque($0).release() }
        #else
            let result = pthread_key_create(&key) { pointer in
                guard let pointer else { return }
                Unmanaged<Cache>.fromOpaque(pointer).release()
            }
        #endif
        precondition(result == 0, "Unable to allocate tokenizer scratch TLS key")
        self.key = key
    }

    @inline(__always)
    func slot(for pool: EncodeScratchPool) -> EncodeScratchPool.Slot? {
        guard let cache = cache(creating: true) else { return nil }
        let owner = ObjectIdentifier(pool)
        for entry in cache.entries {
            if let entry, entry.owner == owner, let slot = entry.slot { return slot }
        }
        return register(pool, in: cache)
    }

    @inline(never)
    private func register(_ pool: EncodeScratchPool, in cache: Cache) -> EncodeScratchPool.Slot {
        let slot = pool.register()
        let index = cache.entries.firstIndex { $0?.slot == nil } ?? cache.nextEviction
        cache.nextEviction = (index + 1) % Self.entriesPerThread
        cache.entries[index] = Cache.Entry(pool: pool, slot: slot)
        return slot
    }

    /// The calling thread's cache, created on demand when `creating`.
    @inline(__always)
    private func cache(creating: Bool) -> Cache? {
        if let pointer = pthread_getspecific(key) {
            return Unmanaged<Cache>.fromOpaque(pointer).takeUnretainedValue()
        }
        guard creating else { return nil }
        return makeCache()
    }

    /// Keep allocation and failure handling out of the inlined, warm scratch-recycling path.
    @inline(never)
    private func makeCache() -> Cache? {
        let cache = Cache()
        let retained = Unmanaged.passRetained(cache)
        guard pthread_setspecific(key, retained.toOpaque()) == 0 else {
            retained.release()
            return nil
        }
        return cache
    }
}
