// BERT-style WordPiece tokenization: basic (whitespace + punctuation) splitting followed by
// greedy longest-match-first subword segmentation.

import Foundation

public final class BertTokenizer: Sendable {
    private let basicTokenizer: BasicTokenizer
    private let wordpieceTokenizer: WordpieceTokenizer
    private let tokenizeChineseChars: Bool
    private let serializedWordPiece: Bool

    /// Vocabulary keyed by exact scalar sequence (no Unicode canonical folding).
    private let vocab: [BinaryDistinctString: Int]
    private let ids_to_tokens: [Int: String]

    public let bosToken: String?
    public let bosTokenId: Int?
    public let eosToken: String?
    public let eosTokenId: Int?
    public let fuseUnknownTokens: Bool

    public init(
        vocab: [String: Int],
        merges: [String]?,
        tokenizeChineseChars: Bool = true,
        bosToken: String? = nil,
        eosToken: String? = nil,
        fuseUnknownTokens: Bool = false,
        doLowerCase: Bool = true
    ) {
        var distinct = [BinaryDistinctString: Int](minimumCapacity: vocab.count)
        var reverse = [Int: String](minimumCapacity: vocab.count)
        for (token, id) in vocab {
            distinct[BinaryDistinctString(token)] = id
            reverse[id] = token
        }
        serializedWordPiece = false
        self.vocab = distinct
        ids_to_tokens = reverse
        basicTokenizer = BasicTokenizer(doLowerCase: doLowerCase)
        wordpieceTokenizer = WordpieceTokenizer(vocab: distinct)
        self.tokenizeChineseChars = tokenizeChineseChars
        self.bosToken = bosToken
        bosTokenId = bosToken.flatMap { distinct[BinaryDistinctString($0)] }
        self.eosToken = eosToken
        eosTokenId = eosToken.flatMap { distinct[BinaryDistinctString($0)] }
        self.fuseUnknownTokens = fuseUnknownTokens
    }

    public required convenience init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws
    {
        guard let vocab = tokenizerData.model.vocab.dictionary() else {
            throw TokenizerError.missingVocab
        }

        let tokenizeChineseChars = tokenizerConfig.handleChineseChars.boolean(or: true)
        let eosToken = tokenizerConfig.eosToken.string()
        let bosToken = tokenizerConfig.bosToken.string()
        let fuseUnknown = tokenizerConfig.fuseUnk.boolean(or: false)
        let doLowerCase = tokenizerConfig.doLowerCase.boolean(or: true)

        var vocabulary = [BinaryDistinctString: Int](minimumCapacity: vocab.count)
        for (key, value) in vocab {
            if let id = value.integer() { vocabulary[key] = id }
        }
        if let pairs = tokenizerData.addedTokens.array() {
            for element in pairs {
                guard let id = element["id"].integer(), let key = element["content"].string() else { continue }
                vocabulary[BinaryDistinctString(key)] = id
            }
        }
        for (token, id) in addedTokens { vocabulary[BinaryDistinctString(token)] = id }

        let maximum = tokenizerData.model.maxInputCharsPerWord.integer(or: 100)
        guard maximum >= 0 else { throw TokenizerError.invalidConfiguration("Negative WordPiece word limit") }
        self.init(
            distinctVocab: vocabulary, tokenizeChineseChars: tokenizeChineseChars, bosToken: bosToken,
            eosToken: eosToken,
            fuseUnknownTokens: fuseUnknown, doLowerCase: doLowerCase,
            serializedWordPiece: tokenizerData.model.type.string() == "WordPiece",
            unkToken: tokenizerData.model.unkToken.string(or: "[UNK]"),
            prefix: tokenizerData.model.continuingSubwordPrefix.string(or: "##"), maximum: maximum
        )
    }

    init(
        distinctVocab vocab: [BinaryDistinctString: Int],
        tokenizeChineseChars: Bool,
        bosToken: String?,
        eosToken: String?,
        fuseUnknownTokens: Bool,
        doLowerCase: Bool,
        serializedWordPiece: Bool = false, unkToken: String = "[UNK]", prefix: String = "##", maximum: Int = 100
    ) {
        self.serializedWordPiece = serializedWordPiece
        self.vocab = vocab
        var reverse = [Int: String](minimumCapacity: vocab.count)
        for (token, id) in vocab { reverse[id] = token.string }
        ids_to_tokens = reverse
        basicTokenizer = BasicTokenizer(doLowerCase: doLowerCase)
        wordpieceTokenizer = WordpieceTokenizer(vocab: vocab, unkToken: unkToken, prefix: prefix, maximum: maximum)
        self.tokenizeChineseChars = tokenizeChineseChars
        self.bosToken = bosToken
        bosTokenId = bosToken.flatMap { vocab[BinaryDistinctString($0)] }
        self.eosToken = eosToken
        eosTokenId = eosToken.flatMap { vocab[BinaryDistinctString($0)] }
        self.fuseUnknownTokens = fuseUnknownTokens
    }

    public func tokenize(text: String) -> [String] {
        if serializedWordPiece { return wordpieceTokenizer.tokenize(word: text) }
        let text = tokenizeChineseCharsIfNeed(text)
        var tokens: [String] = []
        for token in basicTokenizer.tokenize(text: text) {
            tokens.append(contentsOf: wordpieceTokenizer.tokenize(word: token))
        }
        return tokens
    }

    /// Tokenizes and maps to ids in one step (tokens absent from the vocabulary are dropped,
    /// as in the original BERT implementation).
    func tokenizeToIds(text: String) -> [Int] {
        tokenize(text: text).compactMap { vocab[BinaryDistinctString($0)] }
    }

    func unTokenize(tokens: [Int]) -> [String] {
        tokens.compactMap { ids_to_tokens[$0] }
    }

    func convertWordpieceToBasicTokenList(_ wordpieceTokenList: [String]) -> String {
        var tokenList: [String] = []
        var individualToken = ""
        for token in wordpieceTokenList {
            if token.hasBytePrefix(wordpieceTokenizer.prefix) {
                individualToken += String(token.droppingBytePrefix(wordpieceTokenizer.prefix))
            } else {
                if !individualToken.isEmpty {
                    tokenList.append(individualToken)
                }
                individualToken = token
            }
        }
        tokenList.append(individualToken)
        return tokenList.joined(separator: " ")
    }

    private func tokenizeChineseCharsIfNeed(_ text: String) -> String {
        guard tokenizeChineseChars else { return text }
        var hasCJK = false
        for scalar in text.unicodeScalars where scalar.value >= 0x3400 && scalar.isCJKUnifiedIdeograph {
            hasCJK = true
            break
        }
        guard hasCJK else { return text }
        var output = ""
        for c in text {
            if let scalar = c.unicodeScalars.first, scalar.isCJKUnifiedIdeograph {
                output.append(" ")
                output.append(c)
                output.append(" ")
            } else {
                output.append(c)
            }
        }
        return output
    }
}

extension BertTokenizer: PreTrainedTokenizerModel {
    public var unknownToken: String? { wordpieceTokenizer.unkToken }
    public var unknownTokenId: Int? { vocab[BinaryDistinctString(unknownToken!)] }

    func encode(text: String) -> [Int] { tokenizeToIds(text: text) }

    func decode(tokens: [Int]) -> String {
        convertWordpieceToBasicTokenList(unTokenize(tokens: tokens))
    }

    public func convertTokenToId(_ token: String) -> Int? {
        vocab[BinaryDistinctString(token)] ?? unknownTokenId
    }

    public func convertIdToToken(_ id: Int) -> String? {
        ids_to_tokens[id]
    }
}

extension BertTokenizer: FastTokenizingModel {
    func makeEncoder() -> PieceEncoder { Encoder(model: self) }

    final class Encoder: PieceEncoder {
        let model: BertTokenizer
        init(model: BertTokenizer) { self.model = model }

        override func encode(piece: Substring, byteLevel: Bool, into ids: inout [Int]) {
            let text = byteLevel ? ByteLevelAlphabet.encode(piece.utf8) : String(piece)
            if model.serializedWordPiece {
                if let pieces = model.wordpieceTokenizer.tokenizeToIds(word: text) {
                    ids.append(contentsOf: pieces)
                } else if let unknown = model.unknownTokenId {
                    ids.append(unknown)
                }
                return
            }
            for token in model.tokenize(text: text) {
                if let id = model.convertTokenToId(token) {
                    ids.append(id)
                }
            }
        }
    }
}

final class BasicTokenizer: Sendable {
    let doLowerCase: Bool

    init(doLowerCase: Bool = true) {
        self.doLowerCase = doLowerCase
    }

    let neverSplit: Set<String> = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
    private static let neverSplitBytes: [[UInt8]] = ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"].map {
        Array($0.utf8)
    }

    /// Byte-level implementation for pure-ASCII input: accent stripping is a no-op, whitespace
    /// is space/tab (`NSCharacterSet.whitespaces`), and punctuation is the ASCII range set.
    /// Returns `nil` when the input is not ASCII.
    private static func tokenizeASCII(_ bytes: UnsafeBufferPointer<UInt8>, lowercase: Bool) -> [String]? {
        for b in bytes where b >= 0x80 { return nil }
        var tokens: [String] = []
        var i = 0
        let n = bytes.count

        @inline(__always) func isPunctuation(_ b: UInt8) -> Bool {
            (b >= 33 && b <= 47) || (b >= 58 && b <= 64) || (b >= 91 && b <= 96) || (b >= 123 && b <= 126)
        }

        while i < n {
            // Split on space / tab (components(separatedBy: .whitespaces) semantics).
            if bytes[i] == 0x20 || bytes[i] == 0x09 {
                i += 1
                continue
            }
            var j = i
            while j < n, bytes[j] != 0x20, bytes[j] != 0x09 { j += 1 }
            let word = UnsafeBufferPointer(rebasing: bytes[i..<j])
            i = j

            if neverSplitBytes.contains(where: {
                $0.count == word.count && memcmp($0, word.baseAddress!, word.count) == 0
            }) {
                tokens.append(String(decoding: word, as: UTF8.self))
                continue
            }
            var current: [UInt8] = []
            for var b in word {
                if lowercase, b >= 0x41, b <= 0x5A { b |= 0x20 }
                if isPunctuation(b) {
                    if !current.isEmpty {
                        tokens.append(String(decoding: current, as: UTF8.self))
                        current.removeAll(keepingCapacity: true)
                    }
                    tokens.append(String(UnicodeScalar(b)))
                } else {
                    current.append(b)
                }
            }
            if !current.isEmpty {
                tokens.append(String(decoding: current, as: UTF8.self))
            }
        }
        return tokens
    }

    func maybeStripAccents(_ text: String) -> String {
        guard doLowerCase else { return text }
        return BertNormalizer.stripAccents(text)
    }

    func maybeLowercase(_ text: String) -> String {
        guard doLowerCase else { return text }
        return text.lowercased()
    }

    func tokenize(text: String) -> [String] {
        var copy = text
        if let fast = copy.withUTF8({ Self.tokenizeASCII($0, lowercase: doLowerCase) }) {
            return fast
        }
        let splitTokens = maybeStripAccents(text).components(separatedBy: NSCharacterSet.whitespaces)
        var tokens: [String] = []
        for token in splitTokens {
            if neverSplit.contains(token) {
                tokens.append(token)
                continue
            }
            var currentTok = ""
            for c in maybeLowercase(token) {
                if !c.isExtendedPunctuation {
                    currentTok.append(c)
                } else if !currentTok.isEmpty {
                    tokens.append(currentTok)
                    tokens.append(String(c))
                    currentTok = ""
                } else {
                    tokens.append(String(c))
                }
            }
            if !currentTok.isEmpty {
                tokens.append(currentTok)
            }
        }
        return tokens
    }
}

private extension Character {
    /// https://github.com/huggingface/transformers/blob/8c1b5d37827a6691fef4b2d926f2d04fb6f5a9e3/src/transformers/tokenization_utils.py#L367
    var isExtendedPunctuation: Bool {
        if isPunctuation { return true }
        if let value = unicodeScalars.first?.value {
            switch value {
            case 33...47, 58...64, 91...96, 123...126: return true
            default: return false
            }
        }
        return false
    }
}

final class WordpieceTokenizer: Sendable {
    let unkToken: String
    let prefix: String
    private let maxInputCharsPerWord: Int
    private let vocab: [BinaryDistinctString: Int]
    /// Packed copy of the vocabulary so candidate substrings are looked up as byte slices
    /// without materializing a `String` per attempt.
    private let packed: Vocabulary?
    private let continuationPrefix: [UInt8]

    init(vocab: [BinaryDistinctString: Int], unkToken: String = "[UNK]", prefix: String = "##", maximum: Int = 100) {
        self.unkToken = unkToken
        self.prefix = prefix
        maxInputCharsPerWord = maximum
        continuationPrefix = Array(prefix.utf8)
        self.vocab = vocab
        packed = try? Vocabulary(entries: vocab.map { ($0.key.string, $0.value) })
    }

    /// Greedy longest-match-first segmentation of a single word into WordPiece subwords.
    func tokenize(word: String) -> [String] {
        guard let packed else { return tokenizeSlow(Array(word.unicodeScalars)) }
        guard let ids = tokenizeToIds(word: word) else { return [unkToken] }
        return ids.compactMap { packed.token($0) }
    }

    /// The encode pipeline needs IDs directly; don't allocate token strings and hash them again.
    func tokenizeToIds(word: String) -> [Int]? {
        guard let packed else {
            return tokenizeSlow(Array(word.unicodeScalars)).compactMap { vocab[BinaryDistinctString($0)] }
        }
        let count = word.unicodeScalars.count
        guard count <= maxInputCharsPerWord else { return nil }
        if word.isEmpty { return [] }
        var copy = word
        return copy.withUTF8 { bytes -> [Int]? in
            let whole = packed.id(of: bytes)
            if whole >= 0 { return [Int(whole)] }
            var boundaries: [Int] = [0]
            boundaries.reserveCapacity(count + 1)
            var offset = 0
            while offset < bytes.count {
                offset += UTF8Cursor.decode(bytes, at: offset).1
                boundaries.append(offset)
            }
            var ids: [Int] = []
            var scratch = continuationPrefix
            var start = 0
            while start < count {
                var end = count
                var found: Int32 = -1
                while start < end {
                    let slice = UnsafeBufferPointer(rebasing: bytes[boundaries[start]..<boundaries[end]])
                    if start == 0 {
                        found = packed.id(of: slice)
                    } else {
                        scratch.removeSubrange(continuationPrefix.count...)
                        scratch.append(contentsOf: slice)
                        found = scratch.withUnsafeBufferPointer { packed.id(of: $0) }
                    }
                    if found >= 0 { break }
                    end -= 1
                }
                guard found >= 0 else { return nil }
                ids.append(Int(found))
                start = end
            }
            return ids
        }
    }

    private func tokenizeSlow(_ scalars: [Unicode.Scalar]) -> [String] {
        var subTokens: [String] = []
        var start = 0
        while start < scalars.count {
            var end = scalars.count
            var current: String?
            while start < end {
                var substr = String(String.UnicodeScalarView(scalars[start..<end]))
                if start > 0 { substr = prefix + substr }
                if vocab[BinaryDistinctString(substr)] != nil {
                    current = substr
                    break
                }
                end -= 1
            }
            guard let current else { return [unkToken] }
            subTokens.append(current)
            start = end
        }
        return subTokens
    }
}
