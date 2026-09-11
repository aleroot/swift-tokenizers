// WordLevel: the whole pre-tokenized chunk is one token, or the unknown token.
// Ported from `tokenizers::models::wordlevel`.

import Foundation

public final class WordLevelTokenizer: PreTrainedTokenizerModel, Sendable {
    /// Vocabulary keyed by exact byte sequence (no Unicode canonical folding).
    private let vocabulary: Vocabulary
    private let modelVocabulary: ModelVocabulary

    public let bosToken: String?
    public let bosTokenId: Int?
    public let eosToken: String?
    public let eosTokenId: Int?
    public let unknownToken: String?
    public let unknownTokenId: Int?
    /// Every chunk yields exactly one token, so consecutive unknowns are already distinct
    /// pre-tokens and upstream never fuses them.
    public let fuseUnknownTokens = false

    public required init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws {
        let vocabulary = try Vocabulary(
            vocab: tokenizerData.model.vocab, addedTokens: addedTokens, addedTokenConfig: tokenizerData.addedTokens)
        self.vocabulary = vocabulary
        let modelVocabulary = try ModelVocabulary(vocabulary, config: tokenizerData.model.vocab)
        self.modelVocabulary = modelVocabulary
        unknownToken = tokenizerData.model.unkToken.string() ?? TokenizerModel.unknownToken(from: tokenizerConfig)
        unknownTokenId = unknownToken.flatMap { modelVocabulary.id(of: $0) }
        // The public encode API is nonthrowing. Reject here instead of silently losing
        // unknown chunks; HF reports MissingUnkToken when it encounters one at encode time.
        guard unknownTokenId != nil else {
            throw TokenizerError.invalidConfiguration("WordLevel requires an unknown token in the model vocabulary")
        }
        bosToken = addedTokenAsString(tokenizerConfig.bosToken)
        bosTokenId = bosToken.flatMap { vocabulary.id(of: $0) }
        eosToken = addedTokenAsString(tokenizerConfig.eosToken)
        eosTokenId = eosToken.flatMap { vocabulary.id(of: $0) }
    }

    public func tokenize(text: String) -> [String] {
        if modelVocabulary.id(of: text) != nil { return [text] }
        return [unknownToken!]  // Validated at construction.
    }

    public func convertTokenToId(_ token: String) -> Int? {
        vocabulary.id(of: token) ?? unknownTokenId
    }

    public func convertIdToToken(_ id: Int) -> String? {
        vocabulary.token(id)
    }
}

extension WordLevelTokenizer: FastTokenizingModel {
    func makeEncoder() -> PieceEncoder { Encoder(model: self) }

    /// One hash lookup per chunk; the byte-level rewrite buffer is the only state worth reusing.
    final class Encoder: PieceEncoder {
        private let model: WordLevelTokenizer
        @exclusivity(unchecked) private var alphabetScratch: [UInt8] = []

        init(model: WordLevelTokenizer) { self.model = model }

        override func encode(piece: Substring, byteLevel: Bool, into ids: inout [Int]) {
            var copy = piece
            copy.withUTF8 { encode(bytes: $0, byteLevel: byteLevel, into: &ids) }
        }

        override func encode(bytes: UnsafeBufferPointer<UInt8>, byteLevel: Bool, into ids: inout [Int]) {
            guard !bytes.isEmpty else { return }
            if byteLevel {
                alphabetScratch.removeAll(keepingCapacity: true)
                ByteLevelAlphabet.appendEncoded(bytes, to: &alphabetScratch)
                alphabetScratch.withUnsafeBufferPointer { encode(bytes: $0, byteLevel: false, into: &ids) }
                return
            }
            if let id = model.id(of: bytes) ?? model.unknownTokenId { ids.append(id) }
        }
    }

    /// Id of a chunk given as raw UTF-8, without materialising a `String`.
    @inline(__always)
    func id(of bytes: UnsafeBufferPointer<UInt8>) -> Int? {
        let id = modelVocabulary.id(of: bytes)
        return id < 0 ? nil : Int(id)
    }
}
