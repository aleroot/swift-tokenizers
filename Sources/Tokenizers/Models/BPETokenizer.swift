// Byte-Pair Encoding model. Symbols are token ids; merges resolve through ``MergeTable`` keyed on
// `(leftId, rightId)`. Merge products outside the vocabulary get synthetic ids `>= vocab.count`
// and take the byte-fallback path at output. Short words use a linear lowest-rank loop, long
// words a heap with lazy deletion; both pick the lowest rank first, leftmost on ties.

import Foundation

/// A pair of token strings used in BPE merge tables (kept for API compatibility).
struct BytePair: Hashable, Sendable {
    let a: String
    let b: String

    init(_ a: String, _ b: String) {
        self.a = a
        self.b = b
    }

    init(tuple: [String]) {
        a = tuple[0]
        b = tuple[1]
    }
}

/// `@unchecked Sendable`: the symbol tables are filled during initialization and read-only
/// afterwards. They are raw buffers owned by the instance so the per-symbol loops never touch a
/// reference count shared between concurrent encodes.
final class BPETokenizer: PreTrainedTokenizerModel, FastTokenizingModel, @unchecked Sendable {
    let vocab: Vocabulary
    let modelVocabulary: ModelVocabulary
    let merges: MergeTable

    /// Ids of the single-alphabet-character tokens for every byte value (`-1` if missing).
    let byteSymbolIds: UnsafeBufferPointer<Int32>
    /// Ids of single-scalar tokens for scalars below `scalarTableLimit` (`-1` if missing).
    let scalarSymbolIds: UnsafeBufferPointer<Int32>
    static let scalarTableLimit: UInt32 = 0x800
    /// Ids of `<0xNN>` byte-fallback tokens (`-1` if missing).
    let hexaTokenIds: UnsafeBufferPointer<Int32>

    deinit {
        byteSymbolIds.deallocate()
        scalarSymbolIds.deallocate()
        hexaTokenIds.deallocate()
    }

    private static func owned(_ values: [Int32]) -> UnsafeBufferPointer<Int32> {
        let buffer = UnsafeMutableBufferPointer<Int32>.allocate(capacity: values.count)
        _ = buffer.initialize(from: values)
        return UnsafeBufferPointer(buffer)
    }

    /// Number of symbol ids in use (vocabulary size plus synthetic merge products).
    let symbolCount: Int

    /// `true` when a long SentencePiece "word" (e.g. a whole section of Llama-2 / Mistral /
    /// Phi-3 / Gemma input) can be encoded chunk by chunk at `▁` word starts — exactly, and
    /// with cache hits. Every BPE output token is a merge product, so a merge can only span a
    /// `▁` boundary if its product string contains `▁` in a non-initial position (outside a
    /// run of `▁`). Such products are rare (gemma-4 has exactly one, `>▁</`); they are kept in
    /// ``metaspaceChunkGuards`` and a boundary is left uncut whenever one of them occurs
    /// across it. Interior-`▁` vocabulary entries that no merge produces are unreachable and
    /// irrelevant.
    let metaspaceChunking: Bool
    /// Merge products with an interior `▁`, as UTF-8, with the offset of that `▁`.
    let metaspaceChunkGuards: [(bytes: [UInt8], offset: Int)]
    /// Above this many guards, chunking is disabled rather than checked per boundary.
    static let maxMetaspaceChunkGuards = 16

    /// The total number of tokens in the vocabulary.
    var vocabCount: Int { vocab.populatedCount }

    let bosToken: String?
    let bosTokenId: Int?
    let eosToken: String?
    let eosTokenId: Int?
    let unknownToken: String?
    let unknownTokenId: Int?
    let fuseUnknownTokens: Bool
    let byteFallback: Bool
    let continuingPrefix: String
    let endSuffix: String
    let ignoreMerges: Bool
    var hasAffixes: Bool { !continuingPrefix.isEmpty || !endSuffix.isEmpty }

    /// Pretoken → ids memo, shared across calls (see ``PretokenCache``).
    let cache = PretokenCache()

    // MARK: - Construction

    static func mergesFromConfig(_ config: Config?) -> [[String]]? {
        guard let config, let merges = config.array() else { return nil }
        var result: [[String]] = []
        result.reserveCapacity(merges.count)
        for element in merges {
            if let pair = element.array() {
                // tokenizers >= 0.20: each merge is a two-element list.
                guard pair.count == 2, let a = pair[0].string(), let b = pair[1].string() else { continue }
                result.append([a, b])
            } else if let s = element.string() {
                // Legacy "a b" strings: split on the first space only (a piece may itself
                // contain spaces) and skip lines without one.
                if let idx = s.unicodeScalars.firstIndex(of: " ") {
                    let a = String(s.unicodeScalars[..<idx])
                    let b = String(s.unicodeScalars[s.unicodeScalars.index(after: idx)...])
                    result.append([a, b])
                }
            }
        }
        return result
    }

    /// Interns merge pairs into the integer table. Pieces are looked up by UTF-8 bytes so
    /// merge products never allocate a concatenated `String`.
    private struct MergeIntern {
        let vocab: Vocabulary
        let prefixBytes: [UInt8]
        var synthetic: [BinaryDistinctString: Int32] = [:]
        var nextSynthetic: Int32
        var table: MergeTableBuilder
        var concat: [UInt8] = []
        var chunkingSafe: Bool
        var guards: [(bytes: [UInt8], offset: Int)] = []

        init(
            vocab: Vocabulary, continuingPrefix: String, endSuffix: String, ignoreMerges: Bool,
            expectedCount: Int
        ) {
            self.vocab = vocab
            prefixBytes = continuingPrefix.isEmpty ? [] : Array(continuingPrefix.utf8)
            nextSynthetic = Int32(vocab.count)
            table = MergeTableBuilder(expectedCount: expectedCount)
            concat.reserveCapacity(64)
            chunkingSafe =
                vocab.id(of: sentencePieceUnderline) != nil && continuingPrefix.isEmpty && endSuffix.isEmpty
                && !ignoreMerges
        }

        mutating func intern(_ bytes: UnsafeBufferPointer<UInt8>) -> Int32 {
            let existing = vocab.id(of: bytes)
            if existing >= 0 { return existing }
            let key = BinaryDistinctString(String(decoding: bytes, as: UTF8.self))
            if let id = synthetic[key] { return id }
            let id = nextSynthetic
            synthetic[key] = id
            nextSynthetic += 1
            return id
        }

        mutating func add(
            left leftBytes: UnsafeBufferPointer<UInt8>, right rightBytes: UnsafeBufferPointer<UInt8>, rank: Int
        ) {
            let left = intern(leftBytes)
            let right = intern(rightBytes)
            var skip = 0
            if !prefixBytes.isEmpty, rightBytes.count >= prefixBytes.count {
                var matches = true
                for i in 0..<prefixBytes.count where rightBytes[i] != prefixBytes[i] {
                    matches = false
                    break
                }
                if matches { skip = prefixBytes.count }
            }
            concat.removeAll(keepingCapacity: true)
            concat.append(contentsOf: leftBytes)
            if skip < rightBytes.count {
                concat.append(contentsOf: UnsafeBufferPointer(rebasing: rightBytes[skip...]))
            }
            let existing = concat.withUnsafeBufferPointer { vocab.id(of: $0) }
            let merged: Int32
            if existing >= 0 {
                merged = existing
            } else {
                let key = BinaryDistinctString(String(decoding: concat, as: UTF8.self))
                if let id = synthetic[key] {
                    merged = id
                } else {
                    merged = nextSynthetic
                    synthetic[key] = merged
                    nextSynthetic += 1
                }
            }
            table.insert(left: left, right: right, rank: UInt32(rank), merged: merged)
            if chunkingSafe {
                for offset in BPETokenizer.interiorMetaspaceOffsets(concat) {
                    if !guards.contains(where: { $0.offset == offset && $0.bytes == concat }) {
                        guards.append((concat, offset))
                    }
                }
                if guards.count > BPETokenizer.maxMetaspaceChunkGuards {
                    chunkingSafe = false
                    guards.removeAll(keepingCapacity: false)
                }
            }
        }

        func finish() -> (table: MergeTable, symbolCount: Int, chunking: Bool, guards: [(bytes: [UInt8], offset: Int)])
        {
            (table.build(), Int(nextSynthetic), chunkingSafe, chunkingSafe ? guards : [])
        }
    }

    private static func buildMergeTable(
        mergeList: [Config], vocab: Vocabulary, continuingPrefix: String, endSuffix: String,
        ignoreMerges: Bool
    ) -> (table: MergeTable, symbolCount: Int, chunking: Bool, guards: [(bytes: [UInt8], offset: Int)]) {
        var intern = MergeIntern(
            vocab: vocab, continuingPrefix: continuingPrefix, endSuffix: endSuffix, ignoreMerges: ignoreMerges,
            expectedCount: mergeList.count)
        for (rank, element) in mergeList.enumerated() {
            let a: String
            let b: String
            if let pair = element.array() {
                guard pair.count == 2, let left = pair[0].string(), let right = pair[1].string() else { continue }
                a = left
                b = right
            } else if let s = element.string(), let idx = s.unicodeScalars.firstIndex(of: " ") {
                a = String(s.unicodeScalars[..<idx])
                b = String(s.unicodeScalars[s.unicodeScalars.index(after: idx)...])
            } else {
                continue
            }
            var aCopy = a
            var bCopy = b
            aCopy.withUTF8 { leftBytes in
                bCopy.withUTF8 { rightBytes in
                    intern.add(left: leftBytes, right: rightBytes, rank: rank)
                }
            }
        }
        return intern.finish()
    }

    private static func buildMergeTable(
        pairs: PackedStringPairs, vocab: Vocabulary, continuingPrefix: String, endSuffix: String,
        ignoreMerges: Bool
    ) -> (table: MergeTable, symbolCount: Int, chunking: Bool, guards: [(bytes: [UInt8], offset: Int)]) {
        var intern = MergeIntern(
            vocab: vocab, continuingPrefix: continuingPrefix, endSuffix: endSuffix, ignoreMerges: ignoreMerges,
            expectedCount: pairs.count)
        for rank in 0..<pairs.count {
            pairs.withPair(at: rank) { left, right in
                intern.add(left: left, right: right, rank: rank)
            }
        }
        return intern.finish()
    }

    required init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws {
        if !tokenizerData.model.dropout.isNull() {
            guard let dropout = tokenizerData.model.dropout.double(), dropout >= 0, dropout <= 1 else {
                throw TokenizerError.invalidConfiguration("BPE dropout must be between zero and one")
            }
            guard dropout == 0 else {
                throw TokenizerError.unsupportedComponent("BPE dropout (stochastic tokenization)")
            }
        }
        let vocab = try Vocabulary(
            vocab: tokenizerData.model.vocab, addedTokens: addedTokens, addedTokenConfig: tokenizerData.addedTokens)
        self.vocab = vocab
        modelVocabulary = try ModelVocabulary(vocab, config: tokenizerData.model.vocab)
        continuingPrefix = tokenizerData.model.continuingSubwordPrefix.string(or: "")
        endSuffix = tokenizerData.model.endOfWordSuffix.string(or: "")
        ignoreMerges = tokenizerData.model.ignoreMerges.boolean(or: false)

        let built: (table: MergeTable, symbolCount: Int, chunking: Bool, guards: [(bytes: [UInt8], offset: Int)])
        if let packed = tokenizerData.model.merges.asPackedStringPairs() {
            built = Self.buildMergeTable(
                pairs: packed, vocab: vocab, continuingPrefix: continuingPrefix, endSuffix: endSuffix,
                ignoreMerges: ignoreMerges)
        } else if let mergeList = tokenizerData.model.merges.array() {
            built = Self.buildMergeTable(
                mergeList: mergeList, vocab: vocab, continuingPrefix: continuingPrefix,
                endSuffix: endSuffix, ignoreMerges: ignoreMerges)
        } else {
            throw TokenizerError.invalidConfiguration("BPE model is missing `merges`")
        }
        merges = built.table
        symbolCount = built.symbolCount
        metaspaceChunking = built.chunking
        metaspaceChunkGuards = built.guards

        var byteIds = [Int32](repeating: -1, count: 256)
        for b in 0..<256 {
            byteIds[b] = modelVocabulary.id(ofScalar: Unicode.Scalar(ByteLevelAlphabet.byteToScalar[b])!)
        }
        byteSymbolIds = Self.owned(byteIds)

        var scalarIds = [Int32](repeating: -1, count: Int(Self.scalarTableLimit))
        for v in 0..<Self.scalarTableLimit {
            guard let scalar = Unicode.Scalar(v) else { continue }
            scalarIds[Int(v)] = modelVocabulary.id(ofScalar: scalar)
        }
        scalarSymbolIds = Self.owned(scalarIds)

        var hexa = [Int32](repeating: -1, count: 256)
        for b in 0..<256 {
            hexa[b] = Int32(modelVocabulary.id(of: Self.hexaTokenStrings[b]) ?? -1)
        }
        hexaTokenIds = Self.owned(hexa)

        if let unk = TokenizerModel.unknownToken(from: tokenizerConfig) ?? tokenizerData.model.unkToken.string() {
            unknownToken = unk
            unknownTokenId = vocab.id(of: unk)
        } else {
            unknownToken = nil
            unknownTokenId = nil
        }
        eosToken = addedTokenAsString(tokenizerConfig.eosToken)
        eosTokenId = eosToken.flatMap { vocab.id(of: $0) }
        bosToken = addedTokenAsString(tokenizerConfig.bosToken)
        bosTokenId = bosToken.flatMap { vocab.id(of: $0) }
        fuseUnknownTokens = tokenizerConfig.fuseUnk.boolean(or: tokenizerData.model.fuseUnk.boolean(or: false))
        byteFallback = tokenizerData.model.byteFallback.boolean(or: false)
    }

    /// `<0x00>` … `<0xFF>`
    static let hexaTokenStrings: [String] = (0..<256).map { String(format: "<0x%02X>", $0) }

    /// `true` if `token` has a `▁` at a non-initial position that is not preceded by `▁`.
    static func hasInteriorMetaspace(_ token: String) -> Bool {
        var copy = token
        return copy.withUTF8(hasInteriorMetaspace)
    }

    /// Byte offsets of every `▁` in `bytes` at a non-initial position not preceded by `▁`.
    static func interiorMetaspaceOffsets(_ bytes: [UInt8]) -> [Int] {
        var result: [Int] = []
        var i = 1
        let n = bytes.count
        while i + 2 < n {
            if bytes[i] == 0xE2, bytes[i + 1] == 0x96, bytes[i + 2] == 0x81 {
                let precededByMetaspace = i >= 3 && bytes[i - 3] == 0xE2 && bytes[i - 2] == 0x96 && bytes[i - 1] == 0x81
                if !precededByMetaspace { result.append(i) }
                i += 3
            } else {
                i += 1
            }
        }
        return result
    }

    /// Byte-level variant: `▁` is `E2 96 81` in UTF-8.
    static func hasInteriorMetaspace(_ bytes: UnsafeBufferPointer<UInt8>) -> Bool {
        var i = 1
        let n = bytes.count
        while i + 2 < n {
            if bytes[i] == 0xE2, bytes[i + 1] == 0x96, bytes[i + 2] == 0x81 {
                let precededByMetaspace = i >= 3 && bytes[i - 3] == 0xE2 && bytes[i - 2] == 0x96 && bytes[i - 1] == 0x81
                if !precededByMetaspace { return true }
                i += 3
            } else {
                i += 1
            }
        }
        return false
    }

    // MARK: - Vocabulary access

    func convertTokenToId(_ token: String) -> Int? {
        vocab.id(of: token) ?? unknownTokenId
    }

    func convertIdToToken(_ id: Int) -> String? {
        vocab.token(id)
    }

    @inline(__always)
    func isVocabularyId(_ id: Int32) -> Bool {
        id >= 0 && Int(id) < vocab.count && vocab.contains(id: Int(id))
    }

    // MARK: - Symbol construction

    /// A BPE symbol: a token id plus the UTF-8 byte range it covers in the word.
    struct Symbol {
        var id: Int32
        var start: Int32
        var end: Int32
    }

    /// Fills `symbols` with the initial symbols of a byte-level word (one per byte).
    @inline(__always)
    func byteLevelSymbols(_ bytes: UnsafeBufferPointer<UInt8>, into symbols: inout [Symbol]) {
        symbols.removeAll(keepingCapacity: true)
        symbols.reserveCapacity(bytes.count)
        for (i, b) in bytes.enumerated() {
            symbols.append(Symbol(id: byteSymbolIds[Int(b)], start: Int32(i), end: Int32(i + 1)))
        }
    }

    /// Fills `symbols` with the initial symbols of a plain word (one per Unicode scalar).
    @inline(__always)
    func scalarSymbols(_ word: Substring, into symbols: inout [Symbol]) {
        var copy = word
        copy.withUTF8 { scalarSymbols($0, into: &symbols) }
    }

    /// Fills `symbols` with one symbol per Unicode scalar of the UTF-8 `bytes`.
    @inline(__always)
    func scalarSymbols(_ bytes: UnsafeBufferPointer<UInt8>, into symbols: inout [Symbol]) {
        symbols.removeAll(keepingCapacity: true)
        var i = 0
        let n = bytes.count
        while i < n {
            let (value, width) = UTF8Cursor.decode(bytes, at: i)
            let id: Int32
            if value < Self.scalarTableLimit {
                id = scalarSymbolIds[Int(value)]
            } else if let scalar = Unicode.Scalar(value) {
                id = modelVocabulary.id(ofScalar: scalar)
            } else {
                id = -1
            }
            symbols.append(Symbol(id: id, start: Int32(i), end: Int32(i + width)))
            i += width
        }
    }

    /// Used only for configured affixes or missing initial symbols. Resolve fallback and
    /// pending unknowns before merges, matching tokenizers' BPE::merge_word.
    func referenceSymbols(_ bytes: UnsafeBufferPointer<UInt8>, byteLevel: Bool, into symbols: inout [Symbol]) {
        symbols.removeAll(keepingCapacity: true)
        var pendingUnknown: Symbol?
        var i = 0
        while i < bytes.count {
            let value: UInt32
            let width: Int
            if byteLevel {
                value = ByteLevelAlphabet.byteToScalar[Int(bytes[i])]; width = 1
            } else {
                (value, width) = UTF8Cursor.decode(bytes, at: i)
            }
            let end = i + width
            var spelling = String(Unicode.Scalar(value)!)
            if i > 0 { spelling = continuingPrefix + spelling }
            if end == bytes.count { spelling += endSuffix }
            if let id = modelVocabulary.id(of: spelling) {
                if let pendingUnknown { symbols.append(pendingUnknown) }
                pendingUnknown = nil
                symbols.append(Symbol(id: Int32(id), start: Int32(i), end: Int32(end)))
            } else if byteFallback, spelling.utf8.allSatisfy({ hexaTokenIds[Int($0)] >= 0 }) {
                for byte in spelling.utf8 {
                    symbols.append(Symbol(id: hexaTokenIds[Int(byte)], start: Int32(i), end: Int32(end)))
                }
            } else if let unknownTokenId {
                if let previous = pendingUnknown, fuseUnknownTokens {
                    pendingUnknown = Symbol(id: previous.id, start: previous.start, end: Int32(end))
                } else {
                    if let pendingUnknown { symbols.append(pendingUnknown) }
                    pendingUnknown = Symbol(id: Int32(unknownTokenId), start: Int32(i), end: Int32(end))
                }
            }
            i = end
        }
        if let pendingUnknown { symbols.append(pendingUnknown) }
    }

    // MARK: - Merging

    /// Runs BPE merges over `symbols` in place, compacting the array to the surviving pieces.
    func merge(_ symbols: inout [Symbol], scratch: inout MergeScratch) {
        let n = symbols.count
        guard n >= 2 else { return }
        if n <= 96 {
            mergeLinear(&symbols, scratch: &scratch)
        } else {
            mergeHeap(&symbols, scratch: &scratch)
        }
    }

    /// Linear loop: repeatedly find the lowest-rank adjacent pair (leftmost on ties).
    func mergeLinear(_ symbols: inout [Symbol], scratch: inout MergeScratch) {
        var ranks = scratch.ranks
        defer { scratch.ranks = ranks }
        ranks.removeAll(keepingCapacity: true)

        var count = symbols.count
        // ranks[i] is the rank of merging symbols[i] with symbols[i + 1].
        for i in 0..<(count - 1) {
            ranks.append(merges.rank(left: symbols[i].id, right: symbols[i + 1].id))
        }

        while count > 1 {
            var best = UInt32.max
            var bestIndex = -1
            for i in 0..<(count - 1) where ranks[i] < best {
                best = ranks[i]
                bestIndex = i
            }
            guard bestIndex >= 0 else { break }

            let i = bestIndex
            let merged = merges.lookup(left: symbols[i].id, right: symbols[i + 1].id)!.merged
            symbols[i].id = merged
            symbols[i].end = symbols[i + 1].end
            symbols.remove(at: i + 1)
            ranks.remove(at: i)
            count -= 1

            if i > 0 {
                ranks[i - 1] = merges.rank(left: symbols[i - 1].id, right: symbols[i].id)
            }
            if i < count - 1 {
                ranks[i] = merges.rank(left: symbols[i].id, right: symbols[i + 1].id)
            }
        }
    }

    fileprivate struct Candidate: Comparable {
        let rank: UInt32
        let left: Int32

        static func < (lhs: Candidate, rhs: Candidate) -> Bool {
            if lhs.rank != rhs.rank { return lhs.rank < rhs.rank }
            return lhs.left < rhs.left
        }
    }

    /// Priority-queue variant for long words (linked list + min-heap with lazy deletion).
    func mergeHeap(_ symbols: inout [Symbol], scratch: inout MergeScratch) {
        let n = symbols.count
        guard n >= 2 else { return }

        // Move the buffers out while merging: copying them out would leave a second array
        // owner in scratch and trigger copy-on-write allocations on every long word.
        var next: [Int32] = []
        var prev: [Int32] = []
        var alive: [Bool] = []
        var heap = MinHeap<Candidate>()
        swap(&next, &scratch.next)
        swap(&prev, &scratch.prev)
        swap(&alive, &scratch.alive)
        swap(&heap, &scratch.heap)
        defer {
            swap(&next, &scratch.next)
            swap(&prev, &scratch.prev)
            swap(&alive, &scratch.alive)
            swap(&heap, &scratch.heap)
        }
        if next.count < n {
            let additional = n - next.count
            next.append(contentsOf: repeatElement(-1, count: additional))
            prev.append(contentsOf: repeatElement(-1, count: additional))
            alive.append(contentsOf: repeatElement(true, count: additional))
        }
        // Keep the initialized storage at its high-water mark and reset only the active
        // prefix. Every link stays inside that prefix or terminates at -1.
        next.withUnsafeMutableBufferPointer { next in
            prev.withUnsafeMutableBufferPointer { prev in
                alive.withUnsafeMutableBufferPointer { alive in
                    for i in 0..<n {
                        prev[i] = Int32(i - 1)
                        next[i] = i == n - 1 ? -1 : Int32(i + 1)
                        alive[i] = true
                    }
                }
            }
        }

        // The previous merge drained the heap, including stale candidates.
        heap.reserveCapacity(n)

        func enqueue(_ left: Int) {
            let right = next[left]
            guard right >= 0 else { return }
            let rank = merges.rank(left: symbols[left].id, right: symbols[Int(right)].id)
            if rank != UInt32.max {
                heap.push(Candidate(rank: rank, left: Int32(left)))
            }
        }

        for i in 0..<(n - 1) { enqueue(i) }

        while let top = heap.pop() {
            let i = Int(top.left)
            guard alive[i] else { continue }
            let j = Int(next[i])
            guard j >= 0, alive[j] else { continue }
            guard let entry = merges.lookup(left: symbols[i].id, right: symbols[j].id), entry.rank == top.rank else {
                continue
            }
            symbols[i].id = entry.merged
            symbols[i].end = symbols[j].end
            let k = next[j]
            next[i] = k
            if k >= 0 { prev[Int(k)] = Int32(i) }
            alive[j] = false

            if prev[i] >= 0 { enqueue(Int(prev[i])) }
            enqueue(i)
        }

        var write = 0
        var cursor = 0
        while cursor >= 0 {
            symbols[write] = symbols[cursor]
            write += 1
            cursor = Int(next[cursor])
        }
        symbols.removeSubrange(write...)
    }

    // MARK: - Encoding

    func makeEncoder() -> PieceEncoder {
        Encoder(model: self)
    }

    /// Reusable per-call working state.
    struct MergeScratch {
        var ranks: [UInt32] = []
        // Allocated only by the long-word path and exclusively owned by this encoder.
        fileprivate var next: [Int32] = []
        fileprivate var prev: [Int32] = []
        fileprivate var alive: [Bool] = []
        fileprivate var heap = MinHeap<Candidate>()
    }

    /// Stateful encoder holding scratch buffers, pooled across `encode` calls. Each call tries
    /// to take ownership of the shared pretoken cache; if another thread holds it, encoding
    /// memoises into a private table rather than blocking.
    ///
    /// The buffers are only ever touched by the single caller holding the encoder, so dynamic
    /// exclusivity enforcement on them is pure overhead.
    final class Encoder: PieceEncoder {
        let model: BPETokenizer
        @exclusivity(unchecked) var symbols: [Symbol] = []
        @exclusivity(unchecked) var scratch = MergeScratch()
        @exclusivity(unchecked) var fallbackBytes: [UInt8] = []
        @exclusivity(unchecked) private var lease = PretokenCacheLease()

        init(model: BPETokenizer) {
            self.model = model
            symbols.reserveCapacity(64)
        }

        override func begin() { lease.begin(shared: model.cache) }

        override func finish() { lease.finish(shared: model.cache) }

        override func encode(piece: Substring, byteLevel: Bool, into ids: inout [Int]) {
            var word = piece
            word.withUTF8 { bytes in
                encode(bytes: bytes, byteLevel: byteLevel, into: &ids)
            }
        }

        override func encode(bytes: UnsafeBufferPointer<UInt8>, byteLevel: Bool, into ids: inout [Int]) {
            guard !bytes.isEmpty else { return }
            if !byteLevel, model.metaspaceChunking, bytes.count > 3 {
                encodeChunked(bytes, into: &ids)
            } else {
                encodeWord(bytes, byteLevel: byteLevel, into: &ids)
            }
        }

        /// Splits a SentencePiece word at every `▁` that starts a new word (not part of a
        /// `▁` run, and not inside a guarded merge product) and encodes the chunks
        /// independently. See ``BPETokenizer/metaspaceChunking``.
        private func encodeChunked(_ bytes: UnsafeBufferPointer<UInt8>, into ids: inout [Int]) {
            let n = bytes.count
            let guards = model.metaspaceChunkGuards
            var start = 0
            var p = 1
            while p + 2 < n {
                if bytes[p] == 0xE2, bytes[p + 1] == 0x96, bytes[p + 2] == 0x81 {
                    let precededByMetaspace =
                        p >= 3 && bytes[p - 3] == 0xE2 && bytes[p - 2] == 0x96 && bytes[p - 1] == 0x81
                    if !precededByMetaspace, !Self.isGuarded(bytes, at: p, guards: guards) {
                        encodeWord(UnsafeBufferPointer(rebasing: bytes[start..<p]), byteLevel: false, into: &ids)
                        start = p
                    }
                    p += 3
                } else {
                    p += 1
                }
            }
            encodeWord(UnsafeBufferPointer(rebasing: bytes[start..<n]), byteLevel: false, into: &ids)
        }

        /// `true` if a guarded merge product occurs in `bytes` with its interior `▁` at `p`.
        @inline(__always)
        private static func isGuarded(
            _ bytes: UnsafeBufferPointer<UInt8>, at p: Int, guards: [(bytes: [UInt8], offset: Int)]
        ) -> Bool {
            for g in guards {
                let lo = p - g.offset
                let hi = lo + g.bytes.count
                guard lo >= 0, hi <= bytes.count else { continue }
                var match = true
                for k in 0..<g.bytes.count where bytes[lo + k] != g.bytes[k] {
                    match = false
                    break
                }
                if match { return true }
            }
            return false
        }

        private func encodeWord(_ bytes: UnsafeBufferPointer<UInt8>, byteLevel: Bool, into ids: inout [Int]) {
            let cache = lease.cache
            if let cache, cache.lookup(bytes, byteLevel: byteLevel, into: &ids) {
                return
            }
            let start = ids.count
            if model.ignoreMerges {
                // `ignore_merges` (Llama 3): a word that is itself a token skips the merges.
                let id: Int32
                if byteLevel {
                    fallbackBytes.removeAll(keepingCapacity: true)
                    ByteLevelAlphabet.appendEncoded(bytes, to: &fallbackBytes)
                    id = fallbackBytes.withUnsafeBufferPointer { model.modelVocabulary.id(of: $0) }
                } else {
                    id = model.modelVocabulary.id(of: bytes)
                }
                if id >= 0 {
                    ids.append(Int(id))
                    if let cache { cache.insert(bytes, byteLevel: byteLevel, ids: ids[start...]) }
                    return
                }
            }

            if byteLevel {
                model.byteLevelSymbols(bytes, into: &symbols)
            } else {
                model.scalarSymbols(bytes, into: &symbols)
            }
            if model.hasAffixes || symbols.contains(where: { $0.id < 0 }) {
                model.referenceSymbols(bytes, byteLevel: byteLevel, into: &symbols)
            }
            model.merge(&symbols, scratch: &scratch)

            for symbol in symbols {
                if model.isVocabularyId(symbol.id) {
                    ids.append(Int(symbol.id))
                } else {
                    appendFallback(bytes: bytes, symbol: symbol, byteLevel: byteLevel, into: &ids)
                }
            }

            if let cache {
                cache.insert(bytes, byteLevel: byteLevel, ids: ids[start...])
            }
        }

        /// Byte-fallback for a piece that is not a vocabulary token.
        private func appendFallback(
            bytes: UnsafeBufferPointer<UInt8>, symbol: Symbol, byteLevel: Bool, into ids: inout [Int]
        ) {
            if !model.byteFallback {
                if let unknown = model.unknownTokenId { ids.append(unknown) }
                return
            }
            let slice = UnsafeBufferPointer(rebasing: bytes[Int(symbol.start)..<Int(symbol.end)])
            fallbackBytes.removeAll(keepingCapacity: true)
            if byteLevel {
                // The reference implementation hex-encodes the *alphabet* string of the piece.
                ByteLevelAlphabet.appendEncoded(slice, to: &fallbackBytes)
            } else {
                fallbackBytes.append(contentsOf: slice)
            }
            for b in fallbackBytes {
                let id = model.hexaTokenIds[Int(b)]
                if id >= 0 {
                    ids.append(Int(id))
                } else if let unk = model.unknownTokenId {
                    ids.append(unk)
                }
            }
        }
    }

    // MARK: - String API (compatibility)

    /// Tokenizes an already pre-tokenized (alphabet-encoded) chunk into token strings.
    func tokenize(text: String) -> [String] {
        let encoder = makeEncoder()
        encoder.begin()
        defer { encoder.finish() }
        var ids: [Int] = []
        encoder.encode(piece: Substring(text), byteLevel: false, into: &ids)
        return ids.compactMap { vocab.token($0) }
    }

    /// Applies BPE merges to `token` and returns the resulting pieces (no fallback handling).
    func bpe(token: String) -> [String] {
        var symbols: [Symbol] = []
        var scratch = MergeScratch()
        scalarSymbols(Substring(token), into: &symbols)
        guard !symbols.isEmpty else { return [] }
        merge(&symbols, scratch: &scratch)
        var copy = token
        return copy.withUTF8 { bytes in
            symbols.map { symbol in
                String(
                    decoding: UnsafeBufferPointer(rebasing: bytes[Int(symbol.start)..<Int(symbol.end)]), as: UTF8.self)
            }
        }
    }

    /// Splits with the GPT-2 regex and maps every byte to the byte-level alphabet.
    func byteEncode(text: String) -> [String] {
        var pieces: [Substring] = []
        KnownSplitPattern.gpt2.split(Substring(text), into: &pieces)
        return pieces.map { ByteLevelAlphabet.encode($0.utf8) }
    }

    /// Splits with the GPT-2 regex and returns `<0xNN>` tokens for every byte.
    func hexaEncode(text: String) -> [String] {
        var result: [String] = []
        for b in text.utf8 {
            result.append(Self.hexaTokenStrings[Int(b)])
        }
        return result
    }
}

// MARK: - Min-heap

struct MinHeap<Element: Comparable> {
    private var storage: [Element] = []

    var isEmpty: Bool { storage.isEmpty }

    mutating func reserveCapacity(_ n: Int) {
        storage.reserveCapacity(n)
    }

    mutating func push(_ element: Element) {
        storage.append(element)
        var i = storage.count - 1
        while i > 0 {
            let parent = (i - 1) / 2
            if storage[parent] <= storage[i] { break }
            storage.swapAt(parent, i)
            i = parent
        }
    }

    mutating func pop() -> Element? {
        guard !storage.isEmpty else { return nil }
        let top = storage[0]
        let last = storage.removeLast()
        if storage.isEmpty { return top }
        storage[0] = last
        var i = 0
        let n = storage.count
        while true {
            let l = 2 * i + 1
            let r = 2 * i + 2
            var smallest = i
            if l < n, storage[l] < storage[smallest] { smallest = l }
            if r < n, storage[r] < storage[smallest] { smallest = r }
            if smallest == i { break }
            storage.swapAt(i, smallest)
            i = smallest
        }
        return top
    }
}
