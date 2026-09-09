// BERT-style WordPiece tokenization: greedy longest-match-first subword segmentation of each
// pre-tokenized word. With a serialized `tokenizer.json` (`model.type == "WordPiece"`) the
// normalizer and pre-tokenizer pipeline does the basic tokenization; the legacy
// `vocab.txt` path keeps a built-in basic tokenizer for compatibility.
//
// The hot path (`Encoder.encode(bytes:)`) works on the UTF-8 bytes of a word: one hash probe
// for the whole word, then longest-first candidates bounded by the longest vocabulary entry,
// with the `##` continuation prefix assembled in a reusable scratch buffer.

import Foundation

public final class BertTokenizer: Sendable {
    private let basicTokenizer: BasicTokenizer
    private let wordpieceTokenizer: WordpieceTokenizer
    private let tokenizeChineseChars: Bool
    private let serializedWordPiece: Bool

    /// Vocabulary keyed by exact byte sequence (no Unicode canonical folding).
    private let vocabulary: Vocabulary

    /// Memoises word → ids. Words are short and few, so a compact cache keeps the footprint
    /// well under a megabyte while repeated words (most of natural language) skip the search.
    let cache = PretokenCache(.compact)

    public let bosToken: String?
    public let bosTokenId: Int?
    public let eosToken: String?
    public let eosTokenId: Int?
    public let fuseUnknownTokens: Bool

    public convenience init(
        vocab: [String: Int],
        merges: [String]?,
        tokenizeChineseChars: Bool = true,
        bosToken: String? = nil,
        eosToken: String? = nil,
        fuseUnknownTokens: Bool = false,
        doLowerCase: Bool = true
    ) {
        // A `[String: Int]` vocabulary is well-formed by construction.
        let vocabulary = try! Vocabulary(vocab: vocab)
        self.init(
            vocabulary: vocabulary, tokenizeChineseChars: tokenizeChineseChars, bosToken: bosToken,
            eosToken: eosToken, fuseUnknownTokens: fuseUnknownTokens, doLowerCase: doLowerCase)
    }

    public required convenience init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws
    {
        var extra = addedTokens
        if let pairs = tokenizerData.addedTokens.array() {
            for element in pairs {
                guard let id = element["id"].integer(), let key = element["content"].string() else { continue }
                if extra[key] == nil { extra[key] = id }
            }
        }
        let vocabulary = try Vocabulary(vocab: tokenizerData.model.vocab, addedTokens: extra)

        let maximum = tokenizerData.model.maxInputCharsPerWord.integer(or: 100)
        guard maximum >= 0 else { throw TokenizerError.invalidConfiguration("Negative WordPiece word limit") }
        self.init(
            vocabulary: vocabulary,
            tokenizeChineseChars: tokenizerConfig.handleChineseChars.boolean(or: true),
            bosToken: tokenizerConfig.bosToken.string(),
            eosToken: tokenizerConfig.eosToken.string(),
            fuseUnknownTokens: tokenizerConfig.fuseUnk.boolean(or: false),
            doLowerCase: tokenizerConfig.doLowerCase.boolean(or: true),
            // A serialized pipeline (normalizer / pre-tokenizer in `tokenizer.json`) already
            // basic-tokenizes; pre-2020 exports omit `model.type` but are WordPiece models too.
            serializedWordPiece: tokenizerData.model.type.string() == "WordPiece"
                || !tokenizerData.preTokenizer.isNull() || !tokenizerData.normalizer.isNull(),
            unkToken: tokenizerData.model.unkToken.string(or: "[UNK]"),
            prefix: tokenizerData.model.continuingSubwordPrefix.string(or: "##"), maximum: maximum
        )
    }

    init(
        vocabulary: Vocabulary,
        tokenizeChineseChars: Bool,
        bosToken: String?,
        eosToken: String?,
        fuseUnknownTokens: Bool,
        doLowerCase: Bool,
        serializedWordPiece: Bool = false, unkToken: String = "[UNK]", prefix: String = "##", maximum: Int = 100
    ) {
        self.serializedWordPiece = serializedWordPiece
        self.vocabulary = vocabulary
        basicTokenizer = BasicTokenizer(doLowerCase: doLowerCase)
        wordpieceTokenizer = WordpieceTokenizer(
            vocabulary: vocabulary, unkToken: unkToken, prefix: prefix, maximum: maximum)
        self.tokenizeChineseChars = tokenizeChineseChars
        self.bosToken = bosToken
        bosTokenId = bosToken.flatMap { vocabulary.id(of: $0) }
        self.eosToken = eosToken
        eosTokenId = eosToken.flatMap { vocabulary.id(of: $0) }
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
        tokenize(text: text).compactMap { vocabulary.id(of: $0) }
    }

    func unTokenize(tokens: [Int]) -> [String] {
        tokens.compactMap { vocabulary.token($0) }
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
    public var unknownTokenId: Int? { wordpieceTokenizer.unkId }

    func encode(text: String) -> [Int] { tokenizeToIds(text: text) }

    func decode(tokens: [Int]) -> String {
        convertWordpieceToBasicTokenList(unTokenize(tokens: tokens))
    }

    public func convertTokenToId(_ token: String) -> Int? {
        vocabulary.id(of: token) ?? unknownTokenId
    }

    public func convertIdToToken(_ id: Int) -> String? {
        vocabulary.token(id)
    }
}

extension BertTokenizer: FastTokenizingModel {
    func makeEncoder() -> PieceEncoder { Encoder(model: self) }

    final class Encoder: PieceEncoder {
        let model: BertTokenizer
        @exclusivity(unchecked) private var scratch = WordpieceTokenizer.Scratch()
        @exclusivity(unchecked) private var alphabetScratch: [UInt8] = []
        /// Whether this encoder owns the shared cache for the current call.
        @exclusivity(unchecked) private(set) var usesCache = false

        init(model: BertTokenizer) { self.model = model }

        override func begin() {
            usesCache = model.cache.lock.tryLock()
        }

        override func finish() {
            guard usesCache else { return }
            usesCache = false
            model.cache.lock.unlock()
        }

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
            let mark = ids.count
            if usesCache, model.cache.lookup(bytes, byteLevel: false, into: &ids) { return }
            if model.serializedWordPiece {
                if !model.wordpieceTokenizer.encode(bytes, into: &ids, scratch: &scratch),
                    let unknown = model.unknownTokenId
                {
                    ids.append(unknown)
                }
            } else {
                // Legacy configuration: run the built-in basic tokenizer on the piece.
                for token in model.tokenize(text: String(decoding: bytes, as: UTF8.self)) {
                    if let id = model.convertTokenToId(token) {
                        ids.append(id)
                    }
                }
            }
            if usesCache { model.cache.insert(bytes, byteLevel: false, ids: ids[mark...]) }
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
        guard ASCII.isASCII(bytes) else { return nil }
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
    let unkId: Int?
    let prefix: String
    private let maxInputCharsPerWord: Int
    private let vocabulary: Vocabulary
    private let continuationPrefix: [UInt8]
    /// Byte length of the longest vocabulary entry: no candidate longer than this can match.
    private let maxTokenBytes: Int
    /// BMP scalars that begin some word-initial token / some `##` continuation token. A word
    /// whose current position starts with any other scalar cannot be segmented, so scripts
    /// absent from the vocabulary are rejected without probing.
    private let initialStarts: ScalarBitmap
    private let continuationStarts: ScalarBitmap

    /// Per-encoder reusable buffers.
    struct Scratch {
        /// `##` + candidate bytes.
        var continuation: [UInt8] = []
    }

    init(vocabulary: Vocabulary, unkToken: String = "[UNK]", prefix: String = "##", maximum: Int = 100) {
        self.unkToken = unkToken
        self.prefix = prefix
        maxInputCharsPerWord = maximum
        continuationPrefix = Array(prefix.utf8)
        self.vocabulary = vocabulary
        unkId = vocabulary.id(of: unkToken)
        var longest = 0
        var initialStarts = ScalarBitmap()
        var continuationStarts = ScalarBitmap()
        let prefixBytes = continuationPrefix
        for id in 0..<vocabulary.count where vocabulary.contains(id: id) {
            longest = max(longest, vocabulary.byteCount(of: id))
            vocabulary.withBytes(of: id) { token in
                guard !token.isEmpty else { return }
                if !prefixBytes.isEmpty, token.count > prefixBytes.count, token.starts(with: prefixBytes) {
                    continuationStarts.insert(UTF8Cursor.decode(token, at: prefixBytes.count).value)
                } else {
                    initialStarts.insert(UTF8Cursor.decode(token, at: 0).value)
                }
            }
        }
        maxTokenBytes = longest
        self.initialStarts = initialStarts
        // With an empty prefix every token can continue a word.
        self.continuationStarts = prefixBytes.isEmpty ? initialStarts : continuationStarts
    }

    /// Greedy longest-match-first segmentation of a single word into WordPiece subwords.
    func tokenize(word: String) -> [String] {
        var copy = word
        var scratch = Scratch()
        var ids: [Int] = []
        let encoded = copy.withUTF8 { encode($0, into: &ids, scratch: &scratch) }
        guard encoded else { return [unkToken] }
        return ids.compactMap { vocabulary.token($0) }
    }

    /// The encode pipeline needs IDs directly; don't allocate token strings and hash them again.
    func tokenizeToIds(word: String) -> [Int]? {
        var copy = word
        var scratch = Scratch()
        var ids: [Int] = []
        return copy.withUTF8 { encode($0, into: &ids, scratch: &scratch) } ? ids : nil
    }

    /// The scalar boundary preceding `end` (at least `floor`).
    @inline(__always)
    private static func previousScalarBoundary(_ word: UnsafeBufferPointer<UInt8>, before end: Int, floor: Int) -> Int {
        var end = end - 1
        while end > floor, word[end] & 0xC0 == 0x80 { end -= 1 }
        return end
    }

    /// Appends the subword ids of `word` (well-formed UTF-8) to `ids`. Returns `false`, leaving
    /// `ids` untouched, when the word is too long or a suffix has no match (the caller emits
    /// the unknown token).
    func encode(_ word: UnsafeBufferPointer<UInt8>, into ids: inout [Int], scratch: inout Scratch) -> Bool {
        let n = word.count
        guard n > 0 else { return true }
        if n > maxInputCharsPerWord {
            // Only then can the scalar count exceed the limit.
            var scalarCount = 0
            for byte in word where byte & 0xC0 != 0x80 { scalarCount += 1 }
            guard scalarCount <= maxInputCharsPerWord else { return false }
        }

        let whole = vocabulary.id(of: word)
        if whole >= 0 {
            ids.append(Int(whole))
            return true
        }

        // Ids go straight into `ids` and are rolled back if the word turns out not to be
        // segmentable: no intermediate buffer and no copy on success.
        let mark = ids.count
        let prefixCount = continuationPrefix.count
        var start = 0
        while start < n {
            // No token starts with this scalar: the word cannot be segmented.
            let starts = start == 0 ? initialStarts : continuationStarts
            guard starts.mayContain(UTF8Cursor.decode(word, at: start).value) else {
                ids.removeSubrange(mark...)
                return false
            }
            // Longest candidate first, never longer than the longest vocabulary entry.
            let limit = start == 0 ? maxTokenBytes : maxTokenBytes - prefixCount
            var end = min(n, start + max(limit, 1))
            while end > start, end < n, word[end] & 0xC0 == 0x80 { end -= 1 }  // scalar boundary
            var found: Int32 = -1
            if start == 0 {
                while end > start {
                    found = vocabulary.id(of: UnsafeBufferPointer(rebasing: word[start..<end]))
                    if found >= 0 { break }
                    end = Self.previousScalarBoundary(word, before: end, floor: start)
                }
            } else {
                // `##` + the longest candidate, assembled once; shorter candidates are prefixes of it.
                scratch.continuation.removeAll(keepingCapacity: true)
                scratch.continuation.append(contentsOf: continuationPrefix)
                scratch.continuation.append(contentsOf: UnsafeBufferPointer(rebasing: word[start..<end]))
                scratch.continuation.withUnsafeBufferPointer { candidate in
                    while end > start {
                        let length = prefixCount + (end - start)
                        found = vocabulary.id(of: UnsafeBufferPointer(rebasing: candidate[0..<length]))
                        if found >= 0 { break }
                        end = Self.previousScalarBoundary(word, before: end, floor: start)
                    }
                }
            }
            guard found >= 0 else {
                ids.removeSubrange(mark...)
                return false
            }
            ids.append(Int(found))
            start = end
        }
        return true
    }
}

/// A set of Basic Multilingual Plane scalars (8 KiB bitmap). Scalars outside the BMP are
/// reported as possibly present.
struct ScalarBitmap: Sendable {
    private var words = [UInt64](repeating: 0, count: 0x10000 / 64)

    mutating func insert(_ value: UInt32) {
        guard value < 0x10000 else { return }
        words[Int(value >> 6)] |= 1 << UInt64(value & 63)
    }

    @inline(__always)
    func mayContain(_ value: UInt32) -> Bool {
        guard value < 0x10000 else { return true }
        return words[Int(value >> 6)] & (1 << UInt64(value & 63)) != 0
    }
}
