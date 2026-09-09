// SentencePiece Unigram model (T5, XLM-RoBERTa, …). Segmentation is a Viterbi search for
// the highest-scoring path through the lattice of vocabulary matches. The lattice is
// computed with a forward dynamic program over UTF-8 byte offsets: `best[i]` is the best
// score reaching byte `i`, together with the token and start offset that achieved it.
// Candidate tokens at each position are enumerated with a double-array trie walk over the
// raw bytes, so no scalar decoding or string allocation happens on the hot path.
// Ties resolve to the first candidate encountered (earlier start, then shorter token),
// matching the reference implementation's node ordering.

import Foundation

final class UnigramTokenizer: PreTrainedTokenizerModel, FastTokenizingModel, Sendable {
    /// Score of every model piece, indexed by id.
    let scores: [Double]
    /// Model pieces plus added tokens (`id ↔ token`).
    let vocabulary: Vocabulary
    /// Model pieces keyed by UTF-8 bytes, value = id.
    let trie: DoubleArrayTrie

    let unknownTokenId: Int?
    let unknownTokenScore: Double
    let unknownToken: String?
    let minScore: Double

    let bosToken: String?
    let bosTokenId: Int?
    let eosToken: String?
    let eosTokenId: Int?
    let fuseUnknownTokens: Bool = true
    let byteFallback: Bool
    let byteFallbackIds: [Int]

    /// Memoises piece → ids; see ``PretokenCache``.
    let cache = PretokenCache()

    required init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws {
        let packed: PackedScoredTokens
        if let table = tokenizerData.model.vocab.asPackedScoredTokens() {
            packed = table
        } else if let configVocab = tokenizerData.model.vocab.array() {
            packed = try Self.pack(configVocab)
        } else {
            throw TokenizerError.missingVocab
        }
        scores = packed.scores

        var minScore: Double = 999
        for score in packed.scores where score < minScore { minScore = score }
        self.minScore = minScore

        guard let unknownTokenId = tokenizerData.model["unkId"].integer(), unknownTokenId >= 0,
            unknownTokenId < packed.count
        else {
            throw TokenizerError.malformedVocab
        }
        self.unknownTokenId = unknownTokenId
        unknownToken = packed.token(at: unknownTokenId)
        unknownTokenScore = minScore - 10

        let vocabulary = try Vocabulary(scored: packed, addedTokens: addedTokens)
        self.vocabulary = vocabulary
        trie = packed.utf8.withUnsafeBufferPointer { utf8 in
            packed.offsets.withUnsafeBufferPointer { offsets in
                DoubleArrayTrie(utf8: utf8, offsets: offsets, count: packed.count)
            }
        }

        byteFallback = tokenizerData.model.byteFallback.boolean(or: false)
        byteFallbackIds = BPETokenizer.hexaTokenStrings.map { vocabulary.id(of: $0) ?? -1 }

        bosToken = addedTokenAsString(tokenizerConfig.bosToken)
        bosTokenId = bosToken.flatMap { vocabulary.id(of: $0) }
        eosToken = addedTokenAsString(tokenizerConfig.eosToken)
        eosTokenId = eosToken.flatMap { vocabulary.id(of: $0) }
    }

    /// Packs a generic `[[token, score], …]` Config vocabulary.
    private static func pack(_ rows: [Config]) throws -> PackedScoredTokens {
        var utf8: [UInt8] = []
        var offsets: [UInt32] = [0]
        var scores: [Double] = []
        offsets.reserveCapacity(rows.count + 1)
        scores.reserveCapacity(rows.count)
        for row in rows {
            let tuple = row.array(or: [])
            guard tuple.count == 2, var token = tuple.first?.string(), let score = tuple.last?.double(),
                score.isFinite
            else {
                throw TokenizerError.malformedVocab
            }
            token.withUTF8 { utf8.append(contentsOf: $0) }
            offsets.append(UInt32(utf8.count))
            scores.append(score)
        }
        return PackedScoredTokens(utf8: utf8, offsets: offsets, scores: scores)
    }

    func convertTokenToId(_ token: String) -> Int? {
        vocabulary.id(of: token) ?? unknownTokenId
    }

    func convertIdToToken(_ id: Int) -> String? {
        vocabulary.token(id)
    }

    // MARK: - Viterbi

    /// A segment of the input: byte range and the id of the token covering it.
    struct Piece {
        var start: Int32
        var end: Int32
        var tokenId: Int32
    }

    /// Reusable dynamic-programming state, sized to the longest piece seen so far.
    final class Lattice {
        struct Node {
            var score: Double
            var start: Int32
            var token: Int32
        }

        private(set) var nodes: UnsafeMutablePointer<Node>
        private var capacity: Int
        var pieces: [Piece] = []

        init() {
            capacity = 256
            nodes = .allocate(capacity: capacity)
        }

        deinit { nodes.deallocate() }

        /// Marks offsets `0...count` unreachable (offset 0 reachable with score 0).
        @inline(__always)
        func reset(count: Int) {
            if count >= capacity {
                nodes.deallocate()
                capacity = max(count + 1, capacity * 2)
                nodes = .allocate(capacity: capacity)
            }
            for i in 0...count { nodes[i].score = -.infinity }
            nodes[0] = Node(score: 0, start: 0, token: -1)
            pieces.removeAll(keepingCapacity: true)
        }
    }

    /// Best path through the lattice of `bytes` (well-formed UTF-8). The returned pieces are in
    /// order and consecutive unknown pieces are already fused.
    func segment(_ bytes: UnsafeBufferPointer<UInt8>, lattice: Lattice) -> [Piece] {
        let n = bytes.count
        lattice.reset(count: n)
        guard n > 0 else { return lattice.pieces }

        let nodes = lattice.nodes
        let unkScore = unknownTokenScore
        let unkId = Int32(unknownTokenId ?? 0)

        scores.withUnsafeBufferPointer { scores in
            var begin = 0
            while begin < n {
                let width = UTF8Cursor.width(bytes[begin])
                let base = nodes[begin].score
                // Every scalar boundary is reachable (a token or the unknown fallback ends here).
                var hasSingle = false
                trie.forEachPrefix(of: bytes, from: begin) { length, id in
                    let end = begin + length
                    let score = base + scores[Int(id)]
                    if score > nodes[end].score {
                        nodes[end] = Lattice.Node(score: score, start: Int32(begin), token: id)
                    }
                    if length == width { hasSingle = true }
                }
                if !hasSingle {
                    let end = begin + width
                    let score = base + unkScore
                    if score > nodes[end].score {
                        nodes[end] = Lattice.Node(score: score, start: Int32(begin), token: unkId)
                    }
                }
                begin += width
            }
        }

        // Backtrack, fusing runs of the unknown token. The array is swapped out of the
        // lattice so appends are not routed through class-property exclusivity checks.
        var pieces: [Piece] = []
        swap(&pieces, &lattice.pieces)
        var end = n
        while end > 0 {
            let node = nodes[end]
            let start = Int(node.start)
            if node.token == unkId, let last = pieces.last, last.tokenId == unkId {
                pieces[pieces.count - 1].start = Int32(start)
            } else {
                pieces.append(Piece(start: Int32(start), end: Int32(end), tokenId: node.token))
            }
            end = start
        }
        pieces.reverse()
        lattice.pieces = pieces
        return pieces
    }

    // MARK: - Encoding

    func makeEncoder() -> PieceEncoder { Encoder(model: self) }

    final class Encoder: PieceEncoder {
        let model: UnigramTokenizer
        let lattice = Lattice()
        private var alphabetScratch: [UInt8] = []
        /// Whether this encoder owns the shared cache for its lifetime.
        let usesCache: Bool
        private var finished = false

        init(model: UnigramTokenizer) {
            self.model = model
            usesCache = model.cache.lock.tryLock()
        }

        override func finish() {
            guard usesCache, !finished else { return }
            finished = true
            model.cache.lock.unlock()
        }

        override func encode(piece: Substring, byteLevel: Bool, into ids: inout [Int]) {
            var copy = piece
            copy.withUTF8 { encode(bytes: $0, byteLevel: byteLevel, into: &ids) }
        }

        override func encode(bytes: UnsafeBufferPointer<UInt8>, byteLevel: Bool, into ids: inout [Int]) {
            guard !bytes.isEmpty else { return }
            if byteLevel {
                // Rare: Unigram behind a ByteLevel pre-tokenizer. Map through the alphabet first.
                alphabetScratch.removeAll(keepingCapacity: true)
                ByteLevelAlphabet.appendEncoded(bytes, to: &alphabetScratch)
                alphabetScratch.withUnsafeBufferPointer { encodeSegments($0, into: &ids) }
                return
            }
            if usesCache, model.cache.lookup(bytes, byteLevel: false, into: &ids) { return }
            let start = ids.count
            encodeSegments(bytes, into: &ids)
            if usesCache { model.cache.insert(bytes, byteLevel: false, ids: ids[start...]) }
        }

        private func encodeSegments(_ bytes: UnsafeBufferPointer<UInt8>, into ids: inout [Int]) {
            let unk = Int32(model.unknownTokenId ?? -1)
            for piece in model.segment(bytes, lattice: lattice) {
                if piece.tokenId == unk, model.byteFallback,
                    model.appendByteFallback(bytes[Int(piece.start)..<Int(piece.end)], into: &ids)
                {
                    continue
                }
                ids.append(Int(piece.tokenId))
            }
        }
    }

    /// Appends the `<0xXX>` ids for an unknown span when every byte has one; returns `false`
    /// (appending nothing) otherwise.
    fileprivate func appendByteFallback(_ span: Slice<UnsafeBufferPointer<UInt8>>, into ids: inout [Int]) -> Bool {
        for byte in span where byteFallbackIds[Int(byte)] < 0 { return false }
        for byte in span { ids.append(byteFallbackIds[Int(byte)]) }
        return true
    }

    /// Tokenizes text into token strings (unknown spans are returned verbatim).
    func tokenize(text: String) -> [String] {
        var copy = text
        let lattice = Lattice()
        return copy.withUTF8 { bytes -> [String] in
            var tokens: [String] = []
            let unk = Int32(unknownTokenId ?? -1)
            for piece in segment(bytes, lattice: lattice) {
                let span = bytes[Int(piece.start)..<Int(piece.end)]
                if piece.tokenId == unk, byteFallback, span.allSatisfy({ byteFallbackIds[Int($0)] >= 0 }) {
                    for byte in span { tokens.append(BPETokenizer.hexaTokenStrings[Int(byte)]) }
                } else {
                    tokens.append(String(decoding: UnsafeBufferPointer(rebasing: span), as: UTF8.self))
                }
            }
            return tokens
        }
    }
}
