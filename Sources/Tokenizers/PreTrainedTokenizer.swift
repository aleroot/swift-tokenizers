// The complete tokenizer: added-token splitting → normalization → pre-tokenization →
// model → post-processing, plus decoding and chat templates.

import Foundation
import Jinja

let specialTokenAttributes: [String] = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
]

/// A tokenizer assembled from Hugging Face `tokenizer.json` / `tokenizer_config.json` files.
///
/// All state is immutable after initialization except the compiled chat-template cache,
/// which is guarded by a lock; instances are therefore safe to share across threads and
/// tasks (`@unchecked` only because a non-final class cannot be checked by the compiler).
public class PreTrainedTokenizer: @unchecked Sendable, Tokenizer {
    class var decodesNormalizedAddedTokens: Bool { true }

    let model: any TokenizingModel

    public var bosToken: String? { model.bosToken }
    public var bosTokenId: Int? { model.bosTokenId }
    public var eosToken: String? { model.eosToken }
    public var eosTokenId: Int? { model.eosTokenId }
    public var unknownToken: String? { model.unknownToken }
    public var unknownTokenId: Int? { model.unknownTokenId }
    public var fuseUnknownTokens: Bool { model.fuseUnknownTokens }

    /// Every added token string (special or not).
    let addedTokens: Set<String>
    /// Added tokens flagged `special`, with their ids.
    let specialTokens: [String: Int]
    let specialTokenIds: Set<Int>

    private let preTokenizer: (any StagedPreTokenizer)?
    private let normalizer: (any ByteNormalizer)?
    private let postProcessor: (any PostProcessor)?
    private let decoder: (any Decoder)?
    private let normalizedAddedTokenSpellings: [Int: String]
    private let tokenizerConfig: Config
    private let cleanUpTokenizationSpaces: Bool

    /// Raw text → pieces, over UTF-8 bytes.
    private let pipeline: EncodePipeline
    private let scratchPool = EncodeScratchPool()
    private let fastModel: (any FastTokenizingModel)?
    private let fastPostProcessor: (any FastPostProcessor)?
    /// Built on first decode: embedding and reranking apps never pay for it.
    private let byteLevelDecodeTable: Lazy<ByteLevelDecodeTable>?

    /// Compiled Jinja templates keyed by their source.
    private let compiledChatTemplates = Locked<[String: Template]>([:])

    // MARK: - Initialization

    public required init(tokenizerConfig: Config, tokenizerData: Config, strict: Bool = true) throws {
        var addedTokens: [String: Int] = [:]
        var specialTokens: [String: Int] = [:]
        var splitterTokens: [AddedTokenSplitter.Token] = []

        var normalizedTokens: [AddedTokenSplitter.Token] = []
        var normalizedSpellings: [Int: String] = [:]
        let normalizer = try NormalizerFactory.fromConfig(config: tokenizerData["normalizer"])
        for addedToken in tokenizerData["addedTokens"].array(or: []) {
            guard let id = addedToken["id"].integer() else { continue }  // malformed: token with no id
            guard let content = addedToken.content.string() else { continue }  // malformed: token with no content
            addedTokens[content] = id
            if addedToken["special"].boolean(or: false) {
                specialTokens[content] = id
            }
            let normalized = addedToken["normalized"].boolean(or: false) && normalizer != nil
            let match = normalized ? normalizer!.normalize(text: content) : content
            if Self.decodesNormalizedAddedTokens, normalized, !match.utf8.elementsEqual(content.utf8) {
                normalizedSpellings[id] = match
            }
            let token = AddedTokenSplitter.Token(
                content: match,
                id: id,
                lstrip: addedToken["lstrip"].boolean(or: false),
                rstrip: addedToken["rstrip"].boolean(or: false),
                scalarCount: match.unicodeScalars.count,
                singleWord: addedToken["single_word"].boolean(or: false)
            )
            if normalized { normalizedTokens.append(token) } else { splitterTokens.append(token) }
        }
        // Longest content first, so a shorter token never shadows a longer one it prefixes.
        splitterTokens.sort { $0.scalarCount > $1.scalarCount }

        normalizedAddedTokenSpellings = normalizedSpellings
        self.specialTokens = specialTokens
        specialTokenIds = Set(specialTokens.values)
        self.addedTokens = Set(addedTokens.keys)

        let preTokenizer = try PreTokenizerFactory.fromConfig(config: tokenizerData["preTokenizer"])
        self.preTokenizer = preTokenizer
        self.normalizer = normalizer
        postProcessor = try PostProcessorFactory.fromConfig(config: tokenizerData["postProcessor"])
        decoder = try DecoderFactory.fromConfig(
            config: tokenizerData["decoder"], addedTokens: self.addedTokens.union(normalizedSpellings.values))
        // `transformers` >= 4.45 defaults `clean_up_tokenization_spaces` to `False`.
        cleanUpTokenizationSpaces = tokenizerConfig.cleanUpTokenizationSpaces.boolean(or: false)
        self.tokenizerConfig = tokenizerConfig

        let model = try TokenizerModel.from(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens, strict: strict)
        self.model = model
        fastModel = model as? any FastTokenizingModel

        // `fuse_unk` is a property of the WordPiece-style models; BPE and Unigram fuse
        // (or byte-fall-back) unknowns themselves.
        let fusesUnknown = model.fuseUnknownTokens && !(model is BPETokenizer) && !(model is UnigramTokenizer)
        pipeline = EncodePipeline(
            splitter: AddedTokenSplitter(tokens: splitterTokens),
            normalizer: normalizer,
            normalizedSplitter: AddedTokenSplitter(tokens: normalizedTokens),
            preTokenizer: preTokenizer.map { PreTokenizationRunner(stages: $0.stages) },
            fuseUnknownId: fusesUnknown ? model.unknownTokenId : nil
        )

        if let sequence = postProcessor as? SequenceProcessing {
            fastPostProcessor = sequence.supportsFastPath ? sequence : nil
        } else {
            fastPostProcessor = postProcessor as? any FastPostProcessor
        }

        if decoder is ByteLevelDecoder, normalizedSpellings.isEmpty, let bpe = model as? BPETokenizer {
            let addedTokens = self.addedTokens
            byteLevelDecodeTable = Lazy { ByteLevelDecodeTable(vocabulary: bpe.vocab, addedTokens: addedTokens) }
        } else {
            byteLevelDecodeTable = nil
        }

        // Prepare shared Unicode classification data so a loaded tokenizer is ready
        // for its first request. Subsequent tokenizers reuse the tables.
        _ = ScalarClassifier.bmp
        _ = ScalarClassifier.bmpExtra
    }

    // MARK: - Pipeline stages (string API)

    func preTokenize(_ text: String, options: PreTokenizerOptions) -> [String] {
        guard let preTokenizer else { return [text] }
        return preTokenizer.preTokenize(text: text, options: options)
    }

    func normalize(_ text: String) -> String {
        guard let normalizer else { return text }
        return normalizer.normalize(text: text)
    }

    func postProcess(_ tokens: [String], addSpecialTokens: Bool = true) -> [String] {
        guard let postProcessor else { return tokens }
        return postProcessor.postProcess(tokens: tokens, tokensPair: nil, addSpecialTokens: addSpecialTokens)
    }

    func decodeTokens(_ tokens: [String]) -> [String] {
        guard let decoder else { return tokens }
        return decoder.decode(tokens: tokens)
    }

    /// Clean up a list of simple English tokenization artifacts like spaces before
    /// punctuation and abbreviated forms.
    func cleanUp(text: String) -> String {
        guard cleanUpTokenizationSpaces else { return text }
        return TokenizationCleanup.cleanUp(text)
    }

    // MARK: - Tokenization

    /// Runs `body` with a pooled scratch object.
    @inline(__always)
    private func withScratch<R>(_ body: (EncodeScratch) -> R) -> R {
        let scratch = scratchPool.take()
        defer { scratchPool.recycle(scratch) }
        return body(scratch)
    }

    public func tokenize(text: String) -> [String] {
        var copy = text
        var tokens: [String] = []
        let fuseUnknownId = pipeline.fuseUnknownId
        var sectionStart = 0
        /// Collapses runs of unknown tokens in the current section (`fuse_unk`).
        func fuse() {
            guard let fuseUnknownId, sectionStart < tokens.count else { return }
            var kept: [String] = []
            var previousIsUnknown = false
            for token in tokens[sectionStart...] {
                let isUnknown = model.convertTokenToId(token) == fuseUnknownId
                if !isUnknown || !previousIsUnknown { kept.append(token) }
                previousIsUnknown = isUnknown
            }
            tokens.removeSubrange(sectionStart...)
            tokens.append(contentsOf: kept)
        }
        withScratch { scratch in
            copy.withUTF8 { bytes in
                pipeline.run(
                    bytes, scratch: scratch,
                    onToken: { id in
                        fuse()
                        if let content = model.convertIdToToken(id) { tokens.append(content) }
                        sectionStart = tokens.count
                    },
                    onPiece: { piece, byteLevel in
                        let text = byteLevel ? ByteLevelAlphabet.encode(piece) : String(decoding: piece, as: UTF8.self)
                        tokens.append(contentsOf: model.tokenize(text: text))
                    })
            }
        }
        fuse()
        return tokens
    }

    public func encode(text: String, addSpecialTokens: Bool = true) -> [Int] {
        guard fastModel != nil else {
            return encodeViaStrings(text: text, addSpecialTokens: addSpecialTokens)
        }
        var ids = encodeWithoutPostProcessing(text: text)
        applyPostProcessor(to: &ids, addSpecialTokens: addSpecialTokens)
        return ids
    }

    /// Runs the pipeline up to (excluding) the post-processor.
    func encodeWithoutPostProcessing(text: String) -> [Int] {
        guard let fastModel else {
            return tokenize(text: text).compactMap { model.convertTokenToId($0) }
        }
        let encoder = fastModel.makeEncoder()
        defer { encoder.finish() }
        return withScratch { pipeline.encode(text, encoder: encoder, scratch: $0) }
    }

    /// Applies the configured post-processor to a sequence of ids.
    func applyPostProcessor(to ids: inout [Int], addSpecialTokens: Bool) {
        guard let postProcessor else { return }
        if let fastPostProcessor {
            fastPostProcessor.postProcess(ids: &ids, addSpecialTokens: addSpecialTokens) { [model] token in
                model.convertTokenToId(token)
            }
        } else {
            // Generic processors work on strings: round-trip through token strings.
            let tokens = ids.map { model.convertIdToToken($0) ?? "" }
            let processed = postProcessor.postProcess(
                tokens: tokens, tokensPair: nil, addSpecialTokens: addSpecialTokens)
            ids = processed.compactMap { model.convertTokenToId($0) }
        }
    }

    /// Reference pipeline over token strings (used for models without a fast path).
    private func encodeViaStrings(text: String, addSpecialTokens: Bool) -> [Int] {
        postProcess(tokenize(text: text), addSpecialTokens: addSpecialTokens).compactMap { model.convertTokenToId($0) }
    }

    public func encode(text: String) -> [Int] {
        encode(text: text, addSpecialTokens: true)
    }

    // MARK: - Decoding

    public func decode(tokens: [Int], skipSpecialTokens: Bool = false) -> String {
        if let byteLevelDecodeTable {
            return cleanUp(
                text: byteLevelDecodeTable.value.decode(tokens, skipping: skipSpecialTokens ? specialTokenIds : []))
        }

        var tokenStrings: [String] = []
        tokenStrings.reserveCapacity(tokens.count)
        for id in tokens {
            if skipSpecialTokens, specialTokenIds.contains(id) { continue }
            if let token = convertIdToToken(id) {
                tokenStrings.append(token)
            }
        }
        let decoded = decodeTokens(tokenStrings)
        return cleanUp(text: decoded.joined())
    }

    public func convertTokenToId(_ token: String) -> Int? {
        model.convertTokenToId(token)
    }

    public func convertIdToToken(_ id: Int) -> String? {
        normalizedAddedTokenSpellings[id] ?? model.convertIdToToken(id)
    }

    // MARK: - Chat templates

    public var hasChatTemplate: Bool {
        !tokenizerConfig.chatTemplate.isNull()
    }

    private func compiledTemplate(for source: String) throws -> Template {
        if let cached = compiledChatTemplates.withLock({ $0[source] }) {
            return cached
        }
        // Compile outside the lock; a concurrent duplicate compilation is harmless.
        let compiled = try Template(
            ChatTemplatePreprocessor.preprocess(source),
            with: .init(lstripBlocks: true, trimBlocks: true)
        )
        return compiledChatTemplates.withLock { cache in
            if let cached = cache[source] { return cached }
            cache[source] = compiled
            return compiled
        }
    }

    public func applyChatTemplate(messages: [Message]) throws -> [Int] {
        try applyChatTemplate(messages: messages, addGenerationPrompt: true)
    }

    public func applyChatTemplate(messages: [Message], tools: [ToolSpec]? = nil) throws -> [Int] {
        try applyChatTemplate(messages: messages, addGenerationPrompt: true, tools: tools)
    }

    public func applyChatTemplate(
        messages: [Message],
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) throws -> [Int] {
        try applyChatTemplate(
            messages: messages, addGenerationPrompt: true, tools: tools, additionalContext: additionalContext)
    }

    public func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
        try applyChatTemplate(messages: messages, chatTemplate: chatTemplate, addGenerationPrompt: true)
    }

    public func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
        try applyChatTemplate(messages: messages, chatTemplate: .literal(chatTemplate), addGenerationPrompt: true)
    }

    public func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument? = nil,
        addGenerationPrompt: Bool = false,
        truncation: Bool = false,
        maxLength: Int? = nil,
        tools: [ToolSpec]? = nil
    ) throws -> [Int] {
        try applyChatTemplate(
            messages: messages, chatTemplate: chatTemplate, addGenerationPrompt: addGenerationPrompt,
            truncation: truncation,
            maxLength: maxLength, tools: tools, additionalContext: nil
        )
    }

    /// Renders the chat template to a string without tokenizing it.
    public func renderChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument? = nil,
        addGenerationPrompt: Bool = false,
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) throws -> String {
        let template = try compiledTemplate(for: try selectChatTemplate(chatTemplate, tools: tools))

        var context: [String: Jinja.Value] = try [
            "messages": .array(messages.map { try ChatTemplateValue.make($0) }),
            "add_generation_prompt": .boolean(addGenerationPrompt),
        ]
        if let tools {
            context["tools"] = try .array(tools.map { try ChatTemplateValue.make($0) })
        }
        if let additionalContext {
            // Additional keys and values to be added to the context provided to the prompt
            // templating engine. For example, the app could set "tools_in_user_message" to
            // false for Llama 3.1 and 3.2 if a system message is provided.
            for (key, value) in additionalContext {
                context[key] = try ChatTemplateValue.make(value)
            }
        }

        for (key, value) in tokenizerConfig.dictionary(or: [:]) {
            guard specialTokenAttributes.contains(key.string), !value.isNull() else { continue }
            if let stringValue = value.string() {
                context[key.string] = .string(stringValue)
            } else if let dictionary = value.dictionary() {
                if let addedTokenString = addedTokenAsString(Config(dictionary)) {
                    context[key.string] = .string(addedTokenString)
                }
            } else if let array: [String] = value.get() {
                context[key.string] = .array(array.map { .string($0) })
            } else {
                context[key.string] = try Value(any: value)
            }
        }

        return try template.render(context)
    }

    public func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument? = nil,
        addGenerationPrompt: Bool = false,
        truncation: Bool = false,
        maxLength: Int? = nil,
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) throws -> [Int] {
        if let maxLength, maxLength < 0 {
            throw TokenizerError.invalidConfiguration("maxLength must be nonnegative")
        }
        if let limit = tokenizerConfig.modelMaxLength.integer(), limit < 0 {
            throw TokenizerError.invalidConfiguration("model_max_length must be nonnegative")
        }
        let rendered = try renderChatTemplate(
            messages: messages, chatTemplate: chatTemplate, addGenerationPrompt: addGenerationPrompt, tools: tools,
            additionalContext: additionalContext
        )
        var encodedTokens = encode(text: rendered, addSpecialTokens: false)
        var maxLength = maxLength ?? encodedTokens.count
        maxLength = min(maxLength, tokenizerConfig.modelMaxLength.integer() ?? maxLength)
        if encodedTokens.count > maxLength, truncation {
            encodedTokens = Array(encodedTokens.prefix(maxLength))
        }
        return encodedTokens
    }

    private func selectChatTemplate(_ chatTemplate: ChatTemplateArgument?, tools: [ToolSpec]?) throws -> String {
        if let chatTemplate, case let .literal(template) = chatTemplate {
            return template
        }

        let valueFromConfig: Config = tokenizerConfig.chatTemplate
        if let arrayValue = valueFromConfig.array() {
            // A list of named templates.
            var templateDict: [String: String] = [:]
            for item in arrayValue {
                guard let name = item["name"].string(), let template = item["template"].string() else { continue }
                templateDict[name] = template
            }
            if let chatTemplate, case let .name(name) = chatTemplate {
                guard let match = templateDict[name] else {
                    throw TokenizerError.chatTemplate(
                        "No chat template named \"\(name)\" was found in the tokenizer config")
                }
                return match
            }
            if let tools, !tools.isEmpty, let toolUse = templateDict["tool_use"] {
                return toolUse
            }
            if let defaultTemplate = templateDict["default"] {
                return defaultTemplate
            }
        } else if let stringValue = valueFromConfig.string() {
            return stringValue
        }
        throw TokenizerError.missingChatTemplate
    }
}

// MARK: - Byte-level decode table

/// Precomputed raw bytes for every vocabulary id of a byte-level tokenizer, so decoding is a
/// concatenation of byte slices followed by one UTF-8 validation pass.
final class ByteLevelDecodeTable: Sendable {
    private let storage: [UInt8]
    private let offsets: [UInt32]
    private let isAddedToken: [Bool]
    private let count: Int

    init(vocabulary: Vocabulary, addedTokens: Set<String>) {
        count = vocabulary.count
        var storage: [UInt8] = []
        var offsets = [UInt32](repeating: 0, count: count + 1)
        var isAdded = [Bool](repeating: false, count: count)
        for id in 0..<count {
            offsets[id] = UInt32(storage.count)
            guard vocabulary.contains(id: id) else { continue }
            vocabulary.withBytes(of: id) { bytes in
                // Added tokens are stored verbatim; everything else is alphabet-encoded.
                if !addedTokens.isEmpty, addedTokens.contains(String(decoding: bytes, as: UTF8.self)) {
                    isAdded[id] = true
                    storage.append(contentsOf: bytes)
                    return
                }
                var i = 0
                while i < bytes.count {
                    let (value, width) = UTF8Cursor.decode(bytes, at: i)
                    if value <= ByteLevelAlphabet.maxScalar, ByteLevelAlphabet.scalarToByte[Int(value)] >= 0 {
                        storage.append(UInt8(ByteLevelAlphabet.scalarToByte[Int(value)]))
                    } else {
                        storage.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[i..<(i + width)]))
                    }
                    i += width
                }
            }
        }
        offsets[count] = UInt32(storage.count)
        self.storage = storage
        self.offsets = offsets
        isAddedToken = isAdded
    }

    func decode(_ ids: [Int], skipping skipped: Set<Int>) -> String {
        var output = ""
        var bytes: [UInt8] = []
        bytes.reserveCapacity(ids.count * 4)

        func flush() {
            if !bytes.isEmpty {
                output.append(String(decoding: bytes, as: UTF8.self))
                bytes.removeAll(keepingCapacity: true)
            }
        }

        storage.withUnsafeBufferPointer { buffer in
            for id in ids {
                guard id >= 0, id < count else { continue }
                if !skipped.isEmpty, skipped.contains(id) { continue }
                let lo = Int(offsets[id])
                let hi = Int(offsets[id + 1])
                if isAddedToken[id] {
                    flush()
                    output.append(String(decoding: UnsafeBufferPointer(rebasing: buffer[lo..<hi]), as: UTF8.self))
                } else {
                    bytes.append(contentsOf: UnsafeBufferPointer(rebasing: buffer[lo..<hi]))
                }
            }
        }
        flush()
        return output
    }
}

// MARK: - Cleanup

enum TokenizationCleanup {
    private static let replacements: [(pattern: [UInt8], replacement: [UInt8])] = [
        (" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","),
        (" ' ", "'"), (" n't", "n't"), (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re"),
    ].map { (Array($0.0.utf8), Array($0.1.utf8)) }

    private static let wordPieceReplacements =
        Array(replacements.prefix(7))
        + [(pattern: Array(" do not".utf8), replacement: Array(" don't".utf8))]
        + Array(replacements.suffix(3))

    /// Sequentially applies the classic `clean_up_tokenization_spaces` replacements.
    static func cleanUp(_ text: String, wordPiece: Bool = false) -> String {
        // Quick reject: every pattern starts with a space followed by one of `.?!,'n`.
        var candidate = false
        var previousWasSpace = false
        for b in text.utf8 {
            if previousWasSpace {
                switch b {
                case UInt8(ascii: "."), UInt8(ascii: "?"), UInt8(ascii: "!"), UInt8(ascii: ","), UInt8(ascii: "'"),
                    UInt8(ascii: "n"):
                    candidate = true
                case UInt8(ascii: "d"):
                    candidate = wordPiece
                default:
                    break
                }
                if candidate { break }
            }
            previousWasSpace = b == UInt8(ascii: " ")
        }
        guard candidate else { return text }

        var bytes = Array(text.utf8)
        for (pattern, replacement) in wordPiece ? wordPieceReplacements : replacements {
            bytes = replace(bytes, pattern: pattern, with: replacement)
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private static func replace(_ bytes: [UInt8], pattern: [UInt8], with replacement: [UInt8]) -> [UInt8] {
        guard bytes.count >= pattern.count else { return bytes }
        var out: [UInt8] = []
        out.reserveCapacity(bytes.count)
        var i = 0
        let n = bytes.count
        let m = pattern.count
        let first = pattern[0]
        while i < n {
            if bytes[i] == first, i + m <= n, matches(bytes, at: i, pattern) {
                out.append(contentsOf: replacement)
                i += m
            } else {
                out.append(bytes[i])
                i += 1
            }
        }
        return out
    }

    @inline(__always)
    private static func matches(_ bytes: [UInt8], at i: Int, _ pattern: [UInt8]) -> Bool {
        for k in 0..<pattern.count where bytes[i + k] != pattern[k] { return false }
        return true
    }
}
