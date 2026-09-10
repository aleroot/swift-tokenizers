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
/// Stored properties are immutable and sendable. Shared caches synchronize internally;
/// encode scratch is lent exclusively to synchronous calls by thread-local storage.
/// `@unchecked` is required by the non-final class hierarchy. Subclasses must preserve these
/// invariants; inherited sendability does not check any state they add.
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
    /// The identity of the model's byte-level fast path, or `nil` when it has none and encoding
    /// goes through token strings. Only the identity is stored: the hot path has to know
    /// *whether* there is a fast path, and reading an optional existential to find out copies
    /// it, which is a pair of contended reference-count updates per encode on a shared model.
    private let fastModelIdentity: ObjectIdentifier?
    /// The id-level post-processing step, resolved once at load so that encoding reads a single
    /// non-optional reference and allocates no callback.
    private let idPostProcessor: any IdPostProcessor
    /// Built on first decode: embedding and reranking apps never pay for it.
    private let byteLevelDecodeTable: Lazy<ByteLevelDecodeTable>?

    /// Compiled Jinja templates keyed by their source, bounded by ``chatTemplateCacheLimit``.
    private let compiledChatTemplates = Locked<[String: Template]>([:])

    /// Number of templates currently cached. Exposed for tests.
    var compiledChatTemplateCount: Int { compiledChatTemplates.withLock(\.count) }

    // MARK: - Initialization

    public required init(tokenizerConfig: Config, tokenizerData: Config, strict: Bool = true) throws {
        var addedTokens: [String: Int] = [:]
        var specialTokens: [String: Int] = [:]
        var specialTokenIds: Set<Int> = []
        var splitterTokens: [AddedTokenSplitter.Token] = []

        var normalizedTokens: [AddedTokenSplitter.Token] = []
        var normalizedSpellings: [Int: String] = [:]
        let normalizer = try NormalizerFactory.fromConfig(config: tokenizerData["normalizer"])
        let flagOverrides = AddedTokenFlags.overrides(tokenizerConfig: tokenizerConfig)
        for serialized in tokenizerData["addedTokens"].array(or: []) {
            guard let id = serialized["id"].integer() else { continue }  // malformed: token with no id
            guard let content = serialized.content.string() else { continue }  // malformed: token with no content
            let addedToken = flagOverrides.apply(to: serialized, id: id, content: content)
            addedTokens[content] = id
            if addedToken["special"].boolean(or: false) {
                specialTokens[content] = id
                specialTokenIds.insert(id)
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
        self.specialTokenIds = specialTokenIds
        self.addedTokens = Set(addedTokens.keys)

        let preTokenizer = try PreTokenizerFactory.fromConfig(config: tokenizerData["preTokenizer"])
        self.preTokenizer = preTokenizer
        self.normalizer = normalizer
        postProcessor = try PostProcessorFactory.fromConfig(config: tokenizerData["postProcessor"])
        decoder = try DecoderFactory.fromConfig(config: tokenizerData["decoder"])
        // `transformers` >= 4.45 defaults `clean_up_tokenization_spaces` to `False`.
        cleanUpTokenizationSpaces = tokenizerConfig.cleanUpTokenizationSpaces.boolean(or: false)
        self.tokenizerConfig = tokenizerConfig

        let model = try TokenizerModel.from(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens, strict: strict)
        self.model = model
        fastModelIdentity = (model as? any FastTokenizingModel).map { ObjectIdentifier($0 as AnyObject) }

        // `fuse_unk` is a property of the WordPiece-style models; BPE and Unigram fuse
        // (or byte-fall-back) unknowns themselves.
        let fusesUnknown = model.fuseUnknownTokens && !(model is BPETokenizer) && !(model is UnigramTokenizer)
        pipeline = EncodePipeline(
            splitter: AddedTokenSplitter(tokens: splitterTokens),
            normalizer: normalizer ?? IdentityNormalizer(),
            normalizedSplitter: AddedTokenSplitter(tokens: normalizedTokens),
            preTokenizer: PreTokenizationRunner(stages: preTokenizer?.stages ?? []),
            fuseUnknownId: fusesUnknown ? model.unknownTokenId : nil
        )

        idPostProcessor = Self.makeIdPostProcessor(postProcessor, model: model)

        if decoder is ByteLevelDecoder, normalizedSpellings.isEmpty, let bpe = model as? BPETokenizer {
            byteLevelDecodeTable = Lazy { ByteLevelDecodeTable(vocabulary: bpe.vocab) }
        } else {
            byteLevelDecodeTable = nil
        }
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
        guard fastModelIdentity != nil else {
            return encodeViaStrings(text: text, addSpecialTokens: addSpecialTokens)
        }
        var ids = encodeWithoutPostProcessing(text: text)
        applyPostProcessor(to: &ids, addSpecialTokens: addSpecialTokens)
        return ids
    }

    /// Encodes with source alignment, without changing the IDs-only encode path.
    public func encode(text: String, addSpecialTokens: Bool = true, withOffsets: Bool) throws -> TokenEncoding {
        guard withOffsets else {
            return TokenEncoding(text: text, ids: encode(text: text, addSpecialTokens: addSpecialTokens))
        }
        var tokens = try pipeline.encode(text, model: model)
        if let postProcessor {
            tokens = try postProcessor.processOffsets(
                tokens, text: text, addSpecialTokens: addSpecialTokens,
                resolve: model.convertTokenToId, spelling: convertIdToToken)
        }
        return TokenEncoding(text: text, tokens: tokens)
    }

    /// Runs the pipeline up to (excluding) the post-processor.
    func encodeWithoutPostProcessing(text: String) -> [Int] {
        withEncoder { encoder, scratch in
            pipeline.encode(text, encoder: encoder, scratch: scratch)
        } ?? encodeViaTokenStrings(text: text)
    }

    /// Runs `body` with this tokenizer's pooled scratch and the model's pooled encoder, or
    /// returns `nil` when the model has no byte-level fast path.
    @inline(__always)
    private func withEncoder<R>(_ body: (PieceEncoder, EncodeScratch) -> R) -> R? {
        guard let identity = fastModelIdentity else { return nil }
        let scratch = scratchPool.take()
        defer { scratchPool.recycle(scratch) }
        // The model is cast only when the scratch has no encoder for it yet, so the cached path
        // copies no existential.
        let encoder = scratch.encoder(
            identity: identity, make: { (model as? any FastTokenizingModel)?.makeEncoder() })
        guard let encoder else { return nil }
        encoder.begin()
        defer { encoder.finish() }
        return body(encoder, scratch)
    }

    /// The reference path for a model with no byte-level encoder: token strings mapped through
    /// the vocabulary.
    private func encodeViaTokenStrings(text: String) -> [Int] {
        tokenize(text: text).compactMap { model.convertTokenToId($0) }
    }

    /// Applies the configured post-processor to a sequence of ids.
    func applyPostProcessor(to ids: inout [Int], addSpecialTokens: Bool) {
        idPostProcessor.postProcess(ids: &ids, addSpecialTokens: addSpecialTokens)
    }

    /// Resolves the id-level post-processing step: the processor's own, bound to this
    /// vocabulary, a string round-trip when it has none, or nothing when the tokenizer
    /// declares no post-processor at all.
    private static func makeIdPostProcessor(
        _ postProcessor: (any PostProcessor)?, model: any TokenizingModel
    ) -> any IdPostProcessor {
        guard let postProcessor else { return NoIdPostProcessing() }
        let sequenceIsFast = (postProcessor as? SequenceProcessing)?.supportsFastPath ?? true
        guard sequenceIsFast, let idProcessor = postProcessor as? any IdPostProcessor else {
            return StringIdPostProcessing(processor: postProcessor, model: model)
        }
        return idProcessor.bound { model.convertTokenToId($0) }
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
            guard !tokens.isEmpty else { return "" }
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
        guard let decoder else { return cleanUp(text: tokenStrings.joined(separator: " ")) }
        return cleanUp(text: decoder.decode(tokens: tokenStrings).joined())
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

    /// A tokenizer declares one template, or a handful when the config names several. Callers
    /// may also pass literal templates, so the cache is bounded: without a limit, a caller that
    /// renders a freshly built template per request would retain every one of them.
    static let chatTemplateCacheLimit = 16

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
            // Templates are interchangeable once compiled, so dropping the whole cache is as
            // good as evicting one entry and keeps the common single-template path allocation
            // free. Reaching the limit at all means the caller is not reusing templates.
            if cache.count >= Self.chatTemplateCacheLimit { cache.removeAll(keepingCapacity: true) }
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
            // Extra template context, e.g. `tools_in_user_message` for Llama 3.1 / 3.2.
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
    private let count: Int

    init(vocabulary: Vocabulary) {
        count = vocabulary.count
        var storage: [UInt8] = []
        var offsets = [UInt32](repeating: 0, count: count + 1)
        for id in 0..<count {
            offsets[id] = UInt32(storage.count)
            guard vocabulary.contains(id: id) else { continue }
            vocabulary.withBytes(of: id) { bytes in
                let start = storage.count
                var i = 0
                while i < bytes.count {
                    let (value, width) = UTF8Cursor.decode(bytes, at: i)
                    if value <= ByteLevelAlphabet.maxScalar, ByteLevelAlphabet.scalarToByte[Int(value)] >= 0 {
                        storage.append(UInt8(ByteLevelAlphabet.scalarToByte[Int(value)]))
                    } else {
                        // One scalar outside the alphabet makes the entire token literal.
                        storage.removeSubrange(start...)
                        storage.append(contentsOf: bytes)
                        return
                    }
                    i += width
                }
            }
        }
        offsets[count] = UInt32(storage.count)
        self.storage = storage.trimmed()
        self.offsets = offsets
    }

    func decode(_ ids: [Int], skipping skipped: Set<Int>) -> String {
        storage.withUnsafeBufferPointer { buffer in
            // Token inspection needs no concatenation buffer, including for partial UTF-8.
            if ids.count == 1 {
                let id = ids[0]
                guard id >= 0, id < count, !skipped.contains(id) else { return "" }
                return String(decoding: buffer[Int(offsets[id])..<Int(offsets[id + 1])], as: UTF8.self)
            }

            // Size the final string exactly instead of growing a temporary byte array and
            // copying it again. Validation still happens after concatenation: adjacent
            // tokens may supply separate bytes of the same Unicode scalar.
            var capacity = 0
            for id in ids {
                guard id >= 0, id < count else { continue }
                if !skipped.isEmpty, skipped.contains(id) { continue }
                capacity += Int(offsets[id + 1] - offsets[id])
            }
            guard capacity > 0 else { return "" }
            return String(unsafeUninitializedCapacity: capacity) { output in
                var written = 0
                for id in ids {
                    guard id >= 0, id < count else { continue }
                    if !skipped.isEmpty, skipped.contains(id) { continue }
                    let lo = Int(offsets[id])
                    let length = Int(offsets[id + 1]) - lo
                    if length > 0 {
                        (output.baseAddress! + written).initialize(from: buffer.baseAddress! + lo, count: length)
                        written += length
                    }
                }
                return written
            }
        }
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
