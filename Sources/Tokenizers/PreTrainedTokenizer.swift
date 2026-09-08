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

    private let splitter: AddedTokenSplitter?
    private let normalizedSplitter: AddedTokenSplitter?
    private let preTokenizer: (any PreTokenizer)?
    private let normalizer: (any Normalizer)?
    private let postProcessor: (any PostProcessor)?
    private let decoder: (any Decoder)?
    private let normalizedAddedTokenSpellings: [Int: String]
    private let tokenizerConfig: Config
    private let cleanUpTokenizationSpaces: Bool

    private let fastModel: (any FastTokenizingModel)?
    private let fastPostProcessor: (any FastPostProcessor)?
    private let byteLevelDecodeTable: ByteLevelDecodeTable?

    /// Present when the whole encode pipeline can run on raw UTF-8: no normalizer (or only
    /// NFC), a recognised byte-level split pattern, no `add_prefix_space`, and a BPE model.
    /// Covers GPT-2, Llama 3, Qwen, DeepSeek, Phi-4, Mistral-Tekken, o200k and Falcon-H1
    /// style tokenizers.
    private let byteFastPath: ByteLevelPipeline?
    /// Whether the fast path must NFC-normalize non-ASCII sections first.
    private let byteFastPathNFC: Bool

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
        splitter = AddedTokenSplitter(tokens: splitterTokens)

        preTokenizer = try PreTokenizerFactory.fromConfig(config: tokenizerData["preTokenizer"])
        self.normalizer = normalizer
        normalizedSplitter = AddedTokenSplitter(tokens: normalizedTokens)
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

        if let sequence = postProcessor as? SequenceProcessing {
            fastPostProcessor = sequence.supportsFastPath ? sequence : nil
        } else {
            fastPostProcessor = postProcessor as? any FastPostProcessor
        }

        if decoder is ByteLevelDecoder, normalizedSpellings.isEmpty, let bpe = model as? BPETokenizer {
            byteLevelDecodeTable = ByteLevelDecodeTable(vocabulary: bpe.vocab, addedTokens: self.addedTokens)
        } else {
            byteLevelDecodeTable = nil
        }

        if model is BPETokenizer, normalizedTokens.isEmpty, normalizer == nil || normalizer is NFCNormalizer {
            byteFastPath = Self.byteLevelPattern(of: preTokenizer)
            byteFastPathNFC = normalizer is NFCNormalizer
        } else {
            byteFastPath = nil
            byteFastPathNFC = false
        }

        // Prepare shared Unicode classification data so a loaded tokenizer is ready
        // for its first request. Subsequent tokenizers reuse the table.
        _ = ScalarClassifier.bmp
    }

    /// The pre-tokenization stages the byte-level fast path can run without materialising
    /// strings: an optional `Punctuation` pre-split, a known regex, `ByteLevel`, and an
    /// optional `[0-9]` isolation after it.
    struct ByteLevelPipeline {
        var punctuation: PunctuationPreTokenizer.Behavior?
        var pattern: KnownSplitPattern
        var isolateDigits: Bool
    }

    /// Recognises `ByteLevel(use_regex: true)`, `Sequence[Split(<known regex>), ByteLevel(use_regex: false)]`
    /// and the Falcon-H1 shape `Sequence[Punctuation, Split(<known>), ByteLevel, Split([0-9])]`.
    /// pre-tokenizers without `add_prefix_space`.
    private static func byteLevelPattern(of preTokenizer: (any PreTokenizer)?) -> ByteLevelPipeline? {
        if let byteLevel = preTokenizer as? ByteLevelPreTokenizer {
            return byteLevel.useRegex && !byteLevel.addPrefixSpace
                ? ByteLevelPipeline(punctuation: nil, pattern: .gpt2, isolateDigits: false) : nil
        }
        guard let sequence = preTokenizer as? PreTokenizerSequence else { return nil }
        var stages = sequence.preTokenizers[...]
        var pipeline = ByteLevelPipeline(punctuation: nil, pattern: .gpt2, isolateDigits: false)
        if let punctuation = stages.first as? PunctuationPreTokenizer {
            pipeline.punctuation = punctuation.behavior
            stages = stages.dropFirst()
        }
        guard let split = stages.first as? SplitPreTokenizer, let pattern = split.known else { return nil }
        pipeline.pattern = pattern
        stages = stages.dropFirst()
        guard let byteLevel = stages.first as? ByteLevelPreTokenizer, !byteLevel.useRegex, !byteLevel.addPrefixSpace
        else {
            return nil
        }
        stages = stages.dropFirst()
        if let digits = stages.first as? SplitPreTokenizer, digits.asciiDigitsIsolated {
            pipeline.isolateDigits = true
            stages = stages.dropFirst()
        }
        return stages.isEmpty ? pipeline : nil
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

    func fuseUnknown(_ tokens: [String]) -> [String] {
        guard fuseUnknownTokens, !(model is BPETokenizer), !(model is UnigramTokenizer) else { return tokens }
        var fused: [String] = []
        fused.reserveCapacity(tokens.count)
        var previousIsUnknown = false
        for token in tokens {
            let isUnknown = model.convertTokenToId(token) == model.unknownTokenId
            if !isUnknown || !previousIsUnknown {
                fused.append(token)
            }
            previousIsUnknown = isUnknown
        }
        return fused
    }

    /// Clean up a list of simple English tokenization artifacts like spaces before
    /// punctuation and abbreviated forms.
    func cleanUp(text: String) -> String {
        guard cleanUpTokenizationSpaces else { return text }
        return TokenizationCleanup.cleanUp(text)
    }

    // MARK: - Tokenization

    /// Sections of `text` around added tokens, in order.
    private func sections(of text: String) -> [AddedTokenSplitter.Section] {
        guard !text.isEmpty else { return [] }
        let rawSections = splitter?.split(text) ?? [.text(Substring(text))]
        guard normalizer != nil else { return rawSections }
        return rawSections.flatMap { section -> [AddedTokenSplitter.Section] in
            switch section {
            case .token: return [section]
            case .text(let chunk):
                let normalized = normalize(String(chunk))
                guard !normalized.isEmpty else { return [] }
                return normalizedSplitter?.split(normalized) ?? [.text(Substring(normalized))]
            }
        }
    }

    public func tokenize(text: String) -> [String] {
        var tokens: [String] = []
        for (index, section) in sections(of: text).enumerated() {
            switch section {
            case let .token(_, id):
                if let content = model.convertIdToToken(id) { tokens.append(content) }
            case let .text(chunk):
                let options: PreTokenizerOptions = index == 0 ? [.firstSection] : []
                var sectionTokens: [String] = []
                for piece in preTokenize(String(chunk), options: options) {
                    sectionTokens.append(contentsOf: fuseUnknown(model.tokenize(text: piece)))
                }
                tokens.append(contentsOf: sectionTokens)
            }
        }
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
        if let byteFastPath {
            return encodeByteLevel(text: text, pipeline: byteFastPath, model: fastModel)
        }

        var ids: [Int] = []
        ids.reserveCapacity(text.utf8.count / 3 + 4)

        let encoder = fastModel.makeEncoder()
        defer { encoder.finish() }

        var preTokens: [PreToken] = []
        let unknownId = model.unknownTokenId
        let fuse = model.fuseUnknownTokens && !(model is BPETokenizer) && !(model is UnigramTokenizer)

        for (index, section) in sections(of: text).enumerated() {
            switch section {
            case let .token(_, id):
                ids.append(id)
            case let .text(chunk):
                let sectionStart = ids.count
                let options: PreTokenizerOptions = index == 0 ? [.firstSection] : []
                let normalized = chunk  // sections(of:) already normalized non-added text once.
                preTokens.removeAll(keepingCapacity: true)
                if let preTokenizer {
                    preTokenizer.preTokenizeFast(normalized, options: options, into: &preTokens)
                } else {
                    preTokens.append(PreToken(text: normalized, byteLevel: false))
                }
                for preToken in preTokens {
                    encoder.encode(piece: preToken.text, byteLevel: preToken.byteLevel, into: &ids)
                }
                if fuse, let unknownId {
                    Self.fuseUnknown(&ids, from: sectionStart, unknownId: unknownId)
                }
            }
        }
        return ids
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

    /// Allocation-light pipeline over raw UTF-8 for byte-level BPE tokenizers.
    private func encodeByteLevel(
        text: String, pipeline: ByteLevelPipeline, model fastModel: any FastTokenizingModel
    ) -> [Int] {
        var text = text
        let encoder = fastModel.makeEncoder()
        defer { encoder.finish() }
        let unknownId = model.unknownTokenId
        let fuse = model.fuseUnknownTokens && !(model is BPETokenizer) && !(model is UnigramTokenizer)
        let splitter = self.splitter
        let nfc = byteFastPathNFC
        let pattern = pipeline.pattern
        let isolateDigits = pipeline.isolateDigits

        // The output array is created inside the closure and returned, rather than captured, so
        // appends are not routed through a heap box with dynamic exclusivity checks.
        return text.withUTF8 { bytes -> [Int] in
            var ids: [Int] = []
            ids.reserveCapacity(bytes.count / 3 + 4)

            var sections: [AddedTokenSplitter.ByteSection] = []
            if let splitter {
                splitter.split(bytes: bytes, into: &sections)
            } else {
                sections.append(.text(0..<bytes.count))
            }

            var ranges: [Range<Int>] = []
            var punctuationRanges: [Range<Int>] = []

            /// Encodes one piece, isolating ASCII digits first if the pipeline asks for it.
            @inline(__always)
            func encodePiece(_ piece: UnsafeBufferPointer<UInt8>, into ids: inout [Int]) {
                guard isolateDigits else {
                    encoder.encode(bytes: piece, byteLevel: true, into: &ids)
                    return
                }
                var start = 0
                for i in 0..<piece.count where piece[i] >= 0x30 && piece[i] <= 0x39 {
                    if i > start {
                        encoder.encode(
                            bytes: UnsafeBufferPointer(rebasing: piece[start..<i]), byteLevel: true, into: &ids)
                    }
                    encoder.encode(bytes: UnsafeBufferPointer(rebasing: piece[i..<i + 1]), byteLevel: true, into: &ids)
                    start = i + 1
                }
                if start < piece.count {
                    encoder.encode(
                        bytes: UnsafeBufferPointer(rebasing: piece[start..<piece.count]), byteLevel: true, into: &ids)
                }
            }

            /// Splits `chunk` with the known pattern and encodes every piece.
            func encodeChunk(_ chunk: UnsafeBufferPointer<UInt8>, into ids: inout [Int]) {
                ranges.removeAll(keepingCapacity: true)
                pattern.split(chunk, into: &ranges)
                for pieceRange in ranges {
                    encodePiece(UnsafeBufferPointer(rebasing: chunk[pieceRange]), into: &ids)
                }
            }

            func encodeSection(_ slice: UnsafeBufferPointer<UInt8>, into ids: inout [Int]) {
                let sectionStart = ids.count
                ranges.reserveCapacity(slice.count / 4 + 1)
                if let behavior = pipeline.punctuation {
                    punctuationRanges.removeAll(keepingCapacity: true)
                    PunctuationPreTokenizer.splitRanges(slice, behavior: behavior, into: &punctuationRanges)
                    for chunkRange in punctuationRanges {
                        encodeChunk(UnsafeBufferPointer(rebasing: slice[chunkRange]), into: &ids)
                    }
                } else {
                    encodeChunk(slice, into: &ids)
                }
                if fuse, let unknownId {
                    Self.fuseUnknown(&ids, from: sectionStart, unknownId: unknownId)
                }
            }

            for section in sections {
                switch section {
                case let .token(id):
                    ids.append(id)
                case let .text(range):
                    let slice = UnsafeBufferPointer(rebasing: bytes[range])
                    if nfc, !Self.isASCII(slice) {
                        // NFC is the identity on ASCII; only non-ASCII sections pay for normalization.
                        var normalized = String(decoding: slice, as: UTF8.self).precomposedStringWithCanonicalMapping
                        normalized.withUTF8 { encodeSection($0, into: &ids) }
                    } else {
                        encodeSection(slice, into: &ids)
                    }
                }
            }
            return ids
        }
    }

    /// `true` if every byte is < 0x80. Checks eight bytes per step.
    @inline(__always)
    private static func isASCII(_ bytes: UnsafeBufferPointer<UInt8>) -> Bool {
        guard let base = bytes.baseAddress else { return true }
        let raw = UnsafeRawPointer(base)
        var i = 0
        let n = bytes.count
        while i + 8 <= n {
            if raw.loadUnaligned(fromByteOffset: i, as: UInt64.self) & 0x8080_8080_8080_8080 != 0 { return false }
            i += 8
        }
        while i < n {
            if base[i] >= 0x80 { return false }
            i += 1
        }
        return true
    }

    /// Collapses runs of the unknown id inside `ids[from...]` into a single id.
    private static func fuseUnknown(_ ids: inout [Int], from start: Int, unknownId: Int) {
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
                text: byteLevelDecodeTable.decode(tokens, skipping: skipSpecialTokens ? specialTokenIds : []))
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

    private static let wordPieceReplacements = Array(replacements.prefix(7))
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
