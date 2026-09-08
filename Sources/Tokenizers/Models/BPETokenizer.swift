// Byte-Pair Encoding model.
//
// Design notes
// ------------
// * Symbols are token ids, not strings. A pretoken is turned into its initial symbol ids in
//   one pass (per UTF-8 byte for byte-level vocabularies, per Unicode scalar otherwise) and
//   merges are resolved through ``MergeTable`` keyed on `(leftId, rightId)`.
// * Merge products that are not vocabulary entries receive synthetic ids `>= vocab.count`
//   so they can still participate in later merges exactly like the reference algorithm;
//   at output time such pieces go through the byte-fallback (`<0xNN>`) path.
// * Short words use a linear "find min rank" loop (like tiktoken); long words switch to a
//   heap with lazy deletion (like huggingface/tokenizers). Both produce identical output:
//   lowest rank first, leftmost on ties.

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

final class BPETokenizer: PreTrainedTokenizerModel, FastTokenizingModel, Sendable {
    let vocab: Vocabulary
    let merges: MergeTable

    /// Ids of the single-alphabet-character tokens for every byte value (`-1` if missing).
    let byteSymbolIds: [Int32]
    /// Ids of single-scalar tokens for scalars below `scalarTableLimit` (`-1` if missing).
    let scalarSymbolIds: [Int32]
    static let scalarTableLimit: UInt32 = 0x800
    /// Ids of `<0xNN>` byte-fallback tokens (`-1` if missing).
    let hexaTokenIds: [Int32]

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
                // Legacy "a b" strings.
                let parts = s.unicodeScalars.split(separator: " ", omittingEmptySubsequences: false).map { String($0) }
                result.append(parts)
            }
        }
        return result
    }

    required init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws {
        guard let mergeList = Self.mergesFromConfig(tokenizerData.model.merges) else {
            throw TokenizerError.invalidConfiguration("BPE model is missing `merges`")
        }
        guard let vocabDict = tokenizerData.model.vocab.dictionary() else {
            throw TokenizerError.missingVocab
        }

        let vocab = try Vocabulary(vocab: vocabDict, addedTokens: addedTokens)
        self.vocab = vocab

        // Merge table. Strings that are not vocabulary entries get synthetic ids.
        var synthetic: [BinaryDistinctString: Int32] = [:]
        var nextSynthetic = Int32(vocab.count)
        var table = MergeTable(expectedCount: mergeList.count)

        func intern(_ s: String) -> Int32 {
            if let id = vocab.id(of: s) { return Int32(id) }
            let key = BinaryDistinctString(s)
            if let id = synthetic[key] { return id }
            let id = nextSynthetic
            synthetic[key] = id
            nextSynthetic += 1
            return id
        }

        for (rank, merge) in mergeList.enumerated() where merge.count >= 2 {
            let a = merge[0]
            let b = merge[1]
            let left = intern(a)
            let right = intern(b)
            let merged = intern(a + b)
            table.insert(left: left, right: right, rank: UInt32(rank), merged: merged)
        }
        merges = table
        symbolCount = Int(nextSynthetic)

        var chunkingSafe = vocab.id(of: sentencePieceUnderline) != nil
        var guards: [(bytes: [UInt8], offset: Int)] = []
        if chunkingSafe {
            var seen: Set<[UInt8]> = []
            for merge in mergeList where merge.count >= 2 {
                let product = Array((merge[0] + merge[1]).utf8)
                guard seen.insert(product).inserted else { continue }
                for offset in Self.interiorMetaspaceOffsets(product) {
                    guards.append((product, offset))
                }
                if guards.count > Self.maxMetaspaceChunkGuards {
                    chunkingSafe = false
                    break
                }
            }
        }
        metaspaceChunking = chunkingSafe
        metaspaceChunkGuards = chunkingSafe ? guards : []

        // Symbol tables.
        var byteIds = [Int32](repeating: -1, count: 256)
        for b in 0..<256 {
            byteIds[b] = vocab.id(ofScalar: Unicode.Scalar(ByteLevelAlphabet.byteToScalar[b])!)
        }
        byteSymbolIds = byteIds

        var scalarIds = [Int32](repeating: -1, count: Int(Self.scalarTableLimit))
        for v in 0..<Self.scalarTableLimit {
            guard let scalar = Unicode.Scalar(v) else { continue }
            scalarIds[Int(v)] = vocab.id(ofScalar: scalar)
        }
        scalarSymbolIds = scalarIds

        var hexa = [Int32](repeating: -1, count: 256)
        for b in 0..<256 {
            hexa[b] = Int32(vocab.id(of: Self.hexaTokenStrings[b]) ?? -1)
        }
        hexaTokenIds = hexa

        // Special tokens.
        if let unk = TokenizerModel.unknownToken(from: tokenizerConfig) {
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
        fuseUnknownTokens = tokenizerConfig.fuseUnk.boolean(or: false)
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
                id = vocab.id(ofScalar: scalar)
            } else {
                id = -1
            }
            symbols.append(Symbol(id: id, start: Int32(i), end: Int32(i + width)))
            i += width
        }
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

    /// tiktoken-style loop: repeatedly find the lowest-rank adjacent pair (leftmost on ties).
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

    private struct Candidate: Comparable {
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
        var next = [Int32](repeating: -1, count: n)
        var prev = [Int32](repeating: -1, count: n)
        var alive = [Bool](repeating: true, count: n)
        for i in 0..<n {
            prev[i] = Int32(i - 1)
            next[i] = i == n - 1 ? -1 : Int32(i + 1)
        }

        var heap = MinHeap<Candidate>()
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
    }

    /// Stateful encoder holding scratch buffers, created once per `encode` call. It tries to
    /// take ownership of the shared pretoken cache; if another thread holds it, encoding
    /// proceeds without memoisation rather than blocking.
    final class Encoder: PieceEncoder {
        let model: BPETokenizer
        var symbols: [Symbol] = []
        var scratch = MergeScratch()
        var fallbackBytes: [UInt8] = []
        /// Whether this encoder owns the shared cache for its lifetime (immutable so hot-path
        /// reads need no exclusivity enforcement).
        let usesCache: Bool
        private var finished = false

        init(model: BPETokenizer) {
            self.model = model
            usesCache = model.cache.lock.tryLock()
            symbols.reserveCapacity(64)
        }

        override func finish() {
            guard usesCache, !finished else { return }
            finished = true
            model.cache.lock.unlock()
        }

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
            if usesCache, model.cache.lookup(bytes, byteLevel: byteLevel, into: &ids) {
                return
            }
            let start = ids.count

            if byteLevel {
                model.byteLevelSymbols(bytes, into: &symbols)
            } else {
                model.scalarSymbols(bytes, into: &symbols)
            }
            model.merge(&symbols, scratch: &scratch)

            for symbol in symbols {
                if model.isVocabularyId(symbol.id), Int(symbol.id) != model.unknownTokenId {
                    ids.append(Int(symbol.id))
                } else {
                    appendFallback(bytes: bytes, symbol: symbol, byteLevel: byteLevel, into: &ids)
                }
            }

            if usesCache {
                model.cache.insert(bytes, byteLevel: byteLevel, ids: ids[start...])
            }
        }

        /// Byte-fallback for a piece that is not a vocabulary token.
        private func appendFallback(
            bytes: UnsafeBufferPointer<UInt8>, symbol: Symbol, byteLevel: Bool, into ids: inout [Int]
        ) {
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
        var symbols: [Symbol] = []
        var scratch = MergeScratch()
        scalarSymbols(Substring(text), into: &symbols)
        merge(&symbols, scratch: &scratch)

        var tokens: [String] = []
        tokens.reserveCapacity(symbols.count)
        var copy = text
        copy.withUTF8 { bytes in
            for symbol in symbols {
                if isVocabularyId(symbol.id), Int(symbol.id) != unknownTokenId {
                    tokens.append(vocab.token(Int(symbol.id))!)
                } else {
                    for b in bytes[Int(symbol.start)..<Int(symbol.end)] {
                        tokens.append(Self.hexaTokenStrings[Int(b)])
                    }
                }
            }
        }
        return tokens
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
