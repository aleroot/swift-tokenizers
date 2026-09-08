// Public protocol surface. Kept source-compatible with swift-transformers' `Tokenizers`
// module so existing consumers (e.g. mlx-swift-lm's tokenizer bridge) compile unchanged.

import Foundation

/// A chat message: a dictionary such as `["role": "user", "content": "Hi"]`.
public typealias Message = [String: any Sendable]

/// A tool specification (JSON schema) passed to chat templates that support tool calling.
public typealias ToolSpec = [String: any Sendable]

/// Errors that can occur during tokenizer construction or use.
public enum TokenizerError: LocalizedError, Sendable {
    case missingConfig
    case missingTokenizerClassInConfig
    case unsupportedTokenizer(String)
    case missingVocab
    case malformedVocab
    case chatTemplate(String)
    case missingChatTemplate
    case tooLong(String)
    case mismatchedConfig(String)
    /// A pipeline component (normalizer, pre-tokenizer, post-processor, decoder) of a type
    /// this library does not implement.
    case unsupportedComponent(String)
    /// A pipeline component configuration is missing a required field or holds an invalid value.
    case invalidConfiguration(String)
    /// A required file (`tokenizer.json`, `tokenizer_config.json`, …) was not found.
    case missingFile(URL)

    public var errorDescription: String? {
        switch self {
        case .missingConfig:
            "Tokenizer configuration is missing."
        case .missingTokenizerClassInConfig:
            "The tokenizer class is not specified in the configuration."
        case let .unsupportedTokenizer(name):
            "The tokenizer type '\(name)' is not supported."
        case .missingVocab:
            "Vocabulary file is missing from the tokenizer configuration."
        case .malformedVocab:
            "The vocabulary file is malformed or corrupted."
        case let .chatTemplate(message):
            "Chat template error: \(message)"
        case .missingChatTemplate:
            "This tokenizer does not have a chat template, and no template was passed."
        case let .tooLong(message):
            "Input is too long: \(message)"
        case let .mismatchedConfig(message):
            "Tokenizer configuration mismatch: \(message)"
        case let .unsupportedComponent(name):
            "Unsupported tokenizer component: \(name)"
        case let .invalidConfiguration(message):
            "Invalid tokenizer configuration: \(message)"
        case let .missingFile(url):
            "Required tokenizer file not found: \(url.path)"
        }
    }
}

/// Unwraps a required configuration value, throwing ``TokenizerError/invalidConfiguration(_:)``
/// with a message naming the component and field when it is absent.
func require<T>(_ value: T?, _ component: String, field: String) throws -> T {
    guard let value else {
        throw TokenizerError.invalidConfiguration("\(component) is missing `\(field)`")
    }
    return value
}

// MARK: - Model protocols

/// The core tokenization model (BPE, Unigram, WordPiece…) that turns a pre-tokenized
/// chunk of text into tokens and maps tokens to ids.
public protocol TokenizingModel: Sendable {
    func tokenize(text: String) -> [String]
    func callAsFunction(_ text: String) -> [String]

    func convertTokenToId(_ token: String) -> Int?
    func convertTokensToIds(_ tokens: [String]) -> [Int?]
    func convertIdToToken(_ id: Int) -> String?
    func convertIdsToTokens(_ ids: [Int]) -> [String?]

    var bosToken: String? { get }
    var bosTokenId: Int? { get }
    var eosToken: String? { get }
    var eosTokenId: Int? { get }
    var unknownToken: String? { get }
    var unknownTokenId: Int? { get }
    var fuseUnknownTokens: Bool { get }
}

public extension TokenizingModel {
    func callAsFunction(_ text: String) -> [String] { tokenize(text: text) }
    func convertTokensToIds(_ tokens: [String]) -> [Int?] { tokens.map { convertTokenToId($0) } }
    func convertIdsToTokens(_ ids: [Int]) -> [String?] { ids.map { convertIdToToken($0) } }
}

/// A tokenizing model that can be built from `tokenizer_config.json` / `tokenizer.json`.
public protocol PreTrainedTokenizerModel: TokenizingModel {
    init(tokenizerConfig: Config, tokenizerData: Config, addedTokens: [String: Int]) throws
}

/// Internal fast path implemented by the bundled models: encodes pre-tokenized pieces
/// straight into token ids, avoiding intermediate `[String]` materialization.
protocol FastTokenizingModel: TokenizingModel {
    /// Creates an encoder holding per-call scratch state. Call `finish()` when done.
    func makeEncoder() -> PieceEncoder
}

/// Per-call encoding state. Subclassed by each model.
class PieceEncoder {
    /// Appends the ids for `piece` to `ids`.
    /// - Parameter byteLevel: `true` when the piece must be interpreted through the GPT-2
    ///   byte-level alphabet (each UTF-8 byte is one initial symbol).
    func encode(piece: Substring, byteLevel: Bool, into ids: inout [Int]) {}

    /// Appends the ids for a piece given as well-formed UTF-8 bytes.
    func encode(bytes: UnsafeBufferPointer<UInt8>, byteLevel: Bool, into ids: inout [Int]) {
        encode(piece: Substring(String(decoding: bytes, as: UTF8.self)), byteLevel: byteLevel, into: &ids)
    }

    /// Releases any shared resources (e.g. a cache lock). Must be called exactly once.
    func finish() {}
}

// MARK: - Tokenizer protocol

/// Selects which chat template to apply.
public enum ChatTemplateArgument: Sendable {
    /// A literal Jinja template.
    case literal(String)
    /// The name of a template from a tokenizer config that declares several templates.
    case name(String)
}

/// The complete tokenizer interface: encoding, decoding, special tokens and chat templates.
public protocol Tokenizer: Sendable {
    func tokenize(text: String) -> [String]

    func encode(text: String) -> [Int]
    func encode(text: String, addSpecialTokens: Bool) -> [Int]
    func callAsFunction(_ text: String, addSpecialTokens: Bool) -> [Int]

    func decode(tokens: [Int]) -> String
    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String

    func convertTokenToId(_ token: String) -> Int?
    func convertTokensToIds(_ tokens: [String]) -> [Int?]
    func convertIdToToken(_ id: Int) -> String?
    func convertIdsToTokens(_ ids: [Int]) -> [String?]

    var bosToken: String? { get }
    var bosTokenId: Int? { get }
    var eosToken: String? { get }
    var eosTokenId: Int? { get }
    var unknownToken: String? { get }
    var unknownTokenId: Int? { get }

    var hasChatTemplate: Bool { get }

    func applyChatTemplate(messages: [Message]) throws -> [Int]
    func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int]
    func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int]
    func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int]
    func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int]

    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?
    ) throws -> [Int]

    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int]
}

public extension Tokenizer {
    var hasChatTemplate: Bool { false }

    func callAsFunction(_ text: String, addSpecialTokens: Bool = true) -> [Int] {
        encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokens: [Int]) -> String {
        decode(tokens: tokens, skipSpecialTokens: false)
    }

    func convertTokensToIds(_ tokens: [String]) -> [Int?] {
        tokens.map { convertTokenToId($0) }
    }

    func convertIdsToTokens(_ ids: [Int]) -> [String?] {
        ids.map { convertIdToToken($0) }
    }

    func applyChatTemplate(
        messages: [Message],
        chatTemplate: ChatTemplateArgument?,
        addGenerationPrompt: Bool,
        truncation: Bool,
        maxLength: Int?,
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        if additionalContext == nil {
            return try applyChatTemplate(
                messages: messages, chatTemplate: chatTemplate, addGenerationPrompt: addGenerationPrompt,
                truncation: truncation, maxLength: maxLength, tools: tools
            )
        }
        throw TokenizerError.chatTemplate("Not implemented")
    }
}

// MARK: - Shared helpers

/// Reads an added-token entry that is either a plain string or a serialized `AddedToken`
/// object (`{"content": "...", "lstrip": ..., ...}`).
func addedTokenAsString(_ addedToken: Config?) -> String? {
    guard let addedToken else { return nil }
    if let s = addedToken.string() { return s }
    return addedToken.content.string()
}

let sentencePieceUnderline = "▁"
