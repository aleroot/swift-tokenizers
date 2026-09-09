// SentencePiece Unigram model (T5, XLM-RoBERTa, …). Segmentation is a Viterbi search for
// the highest-scoring path through the lattice of vocabulary matches. The lattice is
// computed with a forward dynamic program over flat arrays: `best[i]` is the best score
// reaching scalar offset `i`, together with the token and start offset that achieved it.
// Ties resolve to the first candidate encountered (earlier start, then shorter token),
// matching the reference implementation's node ordering.

import Foundation

final class UnigramTokenizer: PreTrainedTokenizerModel, FastTokenizingModel, Sendable {
    struct SentencePieceToken {
        var token: String
        var score: Double
    }

    /// The complete vocabulary with scores, indexed by id.
    let vocab: [SentencePieceToken]
    let scores: [Double]
    let vocabulary: Vocabulary
    let trie: ScalarTrie

    let unknownPiece: SentencePieceToken
    var unknownTokenScore: Double { unknownPiece.score }
    let unknownTokenId: Int?
    var unknownToken: String? { unknownPiece.token }
    let minScore: Double

    let bosToken: String?
    let bosTokenId: Int?
    let eosToken: String?
    let eosTokenId: Int?
    let fuseUnknownTokens: Bool = true
    let byteFallback: Bool
    let byteFallbackIds: [Int]

    required init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws {
        let vocab: [SentencePieceToken]
        if let packed = tokenizerData.model.vocab.asPackedScoredTokens() {
            var pieces: [SentencePieceToken] = []
            pieces.reserveCapacity(packed.count)
            for i in 0..<packed.count {
                pieces.append(SentencePieceToken(token: packed.token(at: i), score: packed.scores[i]))
            }
            vocab = pieces
        } else if let configVocab = tokenizerData.model.vocab.array() {
            var pieces: [SentencePieceToken] = []
            pieces.reserveCapacity(configVocab.count)
            for piece in configVocab {
                let tuple = piece.array(or: [])
                guard tuple.count == 2, let token = tuple.first?.string(), let scoreValue = tuple.last else {
                    throw TokenizerError.malformedVocab
                }
                let score: Double
                if let d = scoreValue.double(), d.isFinite {
                    score = d
                } else {
                    throw TokenizerError.malformedVocab
                }
                pieces.append(SentencePieceToken(token: token, score: score))
            }
            vocab = pieces
        } else {
            throw TokenizerError.missingVocab
        }
        self.vocab = vocab
        scores = vocab.map(\.score)

        var minScore: Double = 999
        for token in vocab where token.score < minScore { minScore = token.score }
        self.minScore = minScore

        guard let unknownTokenId = tokenizerData.model["unkId"].integer(), unknownTokenId >= 0,
            unknownTokenId < vocab.count
        else {
            throw TokenizerError.malformedVocab
        }
        self.unknownTokenId = unknownTokenId
        unknownPiece = SentencePieceToken(token: vocab[unknownTokenId].token, score: minScore - 10)

        var entries: [(String, Int)] = []
        entries.reserveCapacity(vocab.count + addedTokens.count)
        for (id, piece) in vocab.enumerated() { entries.append((piece.token, id)) }
        for (token, id) in addedTokens { entries.append((token, id)) }
        let vocabulary = try Vocabulary(entries: entries)
        self.vocabulary = vocabulary
        byteFallback = tokenizerData.model.byteFallback.boolean(or: false)
        byteFallbackIds = BPETokenizer.hexaTokenStrings.map { vocabulary.id(of: $0) ?? -1 }

        bosToken = addedTokenAsString(tokenizerConfig.bosToken)
        bosTokenId = bosToken.flatMap { vocabulary.id(of: $0) }
        eosToken = addedTokenAsString(tokenizerConfig.eosToken)
        eosTokenId = eosToken.flatMap { vocabulary.id(of: $0) }

        var trie = ScalarTrie()
        for (id, piece) in vocab.enumerated() {
            trie.insert(piece.token, id: Int32(id))
        }
        self.trie = trie
    }

    func convertTokenToId(_ token: String) -> Int? {
        vocabulary.id(of: token) ?? unknownTokenId
    }

    func convertIdToToken(_ id: Int) -> String? {
        vocabulary.token(id)
    }

    // MARK: - Viterbi

    /// Best path through the lattice. Returns `(start, end, tokenId)` triples in scalar offsets.
    func segment(_ scalars: UnsafeBufferPointer<Unicode.Scalar>, scratch: inout Scratch) -> ArraySlice<Piece> {
        let n = scalars.count
        scratch.reset(count: n)
        guard n > 0 else { return scratch.output[...] }

        let unkScore = unknownTokenScore
        let unkId = Int32(unknownTokenId ?? 0)

        for begin in 0..<n {
            let base = scratch.bestScore[begin]
            guard scratch.reachable[begin] else { continue }
            var hasSingle = false
            trie.forEachPrefix(of: scalars, from: begin) { length, id in
                let end = begin + length
                let score = base + scores[Int(id)]
                if !scratch.reachable[end] || score > scratch.bestScore[end] {
                    scratch.reachable[end] = true
                    scratch.bestScore[end] = score
                    scratch.bestStart[end] = Int32(begin)
                    scratch.bestToken[end] = id
                }
                if length == 1 { hasSingle = true }
            }
            if !hasSingle {
                let end = begin + 1
                let score = base + unkScore
                if !scratch.reachable[end] || score > scratch.bestScore[end] {
                    scratch.reachable[end] = true
                    scratch.bestScore[end] = score
                    scratch.bestStart[end] = Int32(begin)
                    scratch.bestToken[end] = unkId
                }
            }
        }

        // Backtrack.
        var end = n
        while end > 0 {
            let start = Int(scratch.bestStart[end])
            scratch.output.append(Piece(start: Int32(start), end: Int32(end), tokenId: scratch.bestToken[end]))
            end = start
        }
        scratch.output.reverse()
        var write = 0
        for piece in scratch.output {
            if write > 0, piece.tokenId == unkId, scratch.output[write - 1].tokenId == unkId {
                scratch.output[write - 1].end = piece.end
            } else {
                scratch.output[write] = piece
                write += 1
            }
        }
        scratch.output.removeSubrange(write...)
        return scratch.output[...]
    }

    struct Piece {
        var start: Int32
        var end: Int32
        var tokenId: Int32
    }

    struct Scratch {
        var bestScore: [Double] = []
        var bestStart: [Int32] = []
        var bestToken: [Int32] = []
        var reachable: [Bool] = []
        var output: [Piece] = []
        var scalars: [Unicode.Scalar] = []

        mutating func reset(count n: Int) {
            bestScore.removeAll(keepingCapacity: true)
            bestScore.append(contentsOf: repeatElement(0, count: n + 1))
            bestStart.removeAll(keepingCapacity: true)
            bestStart.append(contentsOf: repeatElement(0, count: n + 1))
            bestToken.removeAll(keepingCapacity: true)
            bestToken.append(contentsOf: repeatElement(0, count: n + 1))
            reachable.removeAll(keepingCapacity: true)
            reachable.append(contentsOf: repeatElement(false, count: n + 1))
            reachable[0] = true
            output.removeAll(keepingCapacity: true)
        }
    }

    // MARK: - Encoding

    func makeEncoder() -> PieceEncoder { Encoder(model: self) }

    final class Encoder: PieceEncoder {
        let model: UnigramTokenizer
        var scratch = Scratch()

        init(model: UnigramTokenizer) { self.model = model }

        override func encode(piece: Substring, byteLevel: Bool, into ids: inout [Int]) {
            var text = byteLevel ? Substring(ByteLevelAlphabet.encode(piece.utf8)) : piece
            scratch.scalars.removeAll(keepingCapacity: true)
            text.withUTF8 { bytes in
                var i = 0
                while i < bytes.count {
                    let (value, width) = UTF8Cursor.decode(bytes, at: i)
                    scratch.scalars.append(Unicode.Scalar(value) ?? "\u{FFFD}")
                    i += width
                }
            }
            let unk = model.unknownTokenId
            scratch.scalars.withUnsafeBufferPointer { scalars in
                for segment in model.segment(scalars, scratch: &scratch) {
                    let id = Int(segment.tokenId)
                    if id == unk, model.byteFallback {
                        let surface = String(String.UnicodeScalarView(scalars[Int(segment.start)..<Int(segment.end)]))
                        if surface.utf8.allSatisfy({ model.byteFallbackIds[Int($0)] >= 0 }) {
                            ids.append(contentsOf: surface.utf8.map { model.byteFallbackIds[Int($0)] })
                            continue
                        }
                    }
                    ids.append(id)
                }
            }
        }
    }

    /// Tokenizes text into token strings (unknown spans are returned verbatim).
    func tokenize(text: String) -> [String] {
        var scratch = Scratch()
        scratch.scalars.append(contentsOf: text.unicodeScalars)
        return scratch.scalars.withUnsafeBufferPointer { scalars -> [String] in
            var tokens: [String] = []
            for piece in segment(scalars, scratch: &scratch) {
                var view = String.UnicodeScalarView()
                view.append(contentsOf: scalars[Int(piece.start)..<Int(piece.end)])
                let surface = String(view)
                if Int(piece.tokenId) == unknownTokenId, byteFallback,
                    surface.utf8.allSatisfy({ byteFallbackIds[Int($0)] >= 0 })
                {
                    tokens.append(contentsOf: surface.utf8.map { BPETokenizer.hexaTokenStrings[Int($0)] })
                } else {
                    tokens.append(surface)
                }
            }
            return tokens
        }
    }
}
