// Factory for building tokenizers from configuration objects or a local model folder.
// Downloading is intentionally out of scope: fetch the files with whatever client your app
// already uses, then point `AutoTokenizer.from(modelFolder:)` at the directory.

import Foundation

/// Creates the appropriate tokenizer for a configuration.
public enum AutoTokenizer {}

enum PreTrainedTokenizerClasses {
    /// Class overrides for custom behaviour (not to be confused with the model classes in
    /// `TokenizerModel`).
    static let tokenizerClasses: [String: PreTrainedTokenizer.Type] = [
        "LlamaTokenizer": LlamaPreTrainedTokenizer.self,
        "CodeLlamaTokenizer": LlamaPreTrainedTokenizer.self,
        "GemmaTokenizer": LlamaPreTrainedTokenizer.self,
    ]
}

extension AutoTokenizer {
    /// Determines the appropriate tokenizer class for the given configuration.
    static func tokenizerClass(for tokenizerConfig: Config) -> PreTrainedTokenizer.Type {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass.string() else {
            return PreTrainedTokenizer.self
        }
        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")
        return PreTrainedTokenizerClasses.tokenizerClasses[tokenizerName] ?? PreTrainedTokenizer.self
    }
}

public extension AutoTokenizer {
    /// Creates a tokenizer from configuration objects.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The contents of `tokenizer_config.json`.
    ///   - tokenizerData: The contents of `tokenizer.json`.
    ///   - strict: When `true`, unknown `tokenizer_class` values throw instead of falling back to BPE.
    static func from(tokenizerConfig: Config, tokenizerData: Config, strict: Bool = true) throws -> any Tokenizer {
        let tokenizerClass = tokenizerClass(for: tokenizerConfig)
        return try tokenizerClass.init(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, strict: strict)
    }

    /// Loads a tokenizer from a local model folder containing `tokenizer.json` and,
    /// optionally, `tokenizer_config.json`, `config.json`, `chat_template.json` /
    /// `chat_template.jinja`.
    static func from(modelFolder: URL, strict: Bool = true) async throws -> any Tokenizer {
        try load(from: modelFolder, strict: strict)
    }

    /// Synchronous variant of ``from(modelFolder:strict:)``.
    static func load(from modelFolder: URL, strict: Bool = true) throws -> any Tokenizer {
        let configuration = try LocalModelConfiguration(modelFolder: modelFolder)
        guard let tokenizerConfig = configuration.tokenizerConfig else {
            throw TokenizerError.missingFile(modelFolder.appendingPathComponent("tokenizer_config.json"))
        }
        return try from(tokenizerConfig: tokenizerConfig, tokenizerData: configuration.tokenizerData, strict: strict)
    }
}

// MARK: - Llama

/// Mirrors `LlamaTokenizerFast.update_post_processor`: Llama-family tokenizers rebuild their
/// post-processor from `add_bos_token` (default `true`) and `add_eos_token` (default `false`),
/// regardless of what `tokenizer.json` declares.
/// https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/tokenization_llama_fast.py
func llamaPostProcessorConfig(tokenizerConfig: Config) throws -> Config {
    let addBosToken = tokenizerConfig.addBosToken.boolean(or: true)
    let bosToken = addedTokenAsString(tokenizerConfig.bosToken)
    if addBosToken, bosToken == nil {
        throw TokenizerError.mismatchedConfig("add_bos_token is True but bos_token is nil")
    }

    let addEosToken = tokenizerConfig.addEosToken.boolean(or: false)
    let eosToken = addedTokenAsString(tokenizerConfig.eosToken)
    if addEosToken, eosToken == nil {
        throw TokenizerError.mismatchedConfig("add_eos_token is True but eos_token is nil")
    }

    func special(_ token: String, typeId: Int) -> Config {
        Config(["SpecialToken": Config(["id": Config(token), "type_id": Config(typeId)])])
    }

    func sequence(_ id: String, typeId: Int) -> Config {
        Config(["Sequence": Config(["id": Config(id), "type_id": Config(typeId)])])
    }

    /// `[bos?] <sequence> [eos?]` with the given type id.
    func segment(_ id: String, typeId: Int) -> [Config] {
        var items: [Config] = []
        if addBosToken, let bosToken { items.append(special(bosToken, typeId: typeId)) }
        items.append(sequence(id, typeId: typeId))
        if addEosToken, let eosToken { items.append(special(eosToken, typeId: typeId)) }
        return items
    }

    let single = segment("A", typeId: 0)
    let pair = single + segment("B", typeId: 1)

    return Config([
        "type": Config(PostProcessorType.TemplateProcessing.rawValue),
        "single": Config(single),
        "pair": Config(pair),
    ])
}

/// Llama-family tokenizer (`LlamaTokenizer`, `CodeLlamaTokenizer`, `GemmaTokenizer`).
///
/// `transformers` loads `tokenizer.json` verbatim for these models — the `legacy` flag only
/// affects conversion from a slow SentencePiece model — so no Metaspace pre-tokenizer is
/// injected. The only adjustment is the post-processor rebuild performed by
/// `LlamaTokenizerFast.__init__`.
final class LlamaPreTrainedTokenizer: PreTrainedTokenizer, @unchecked Sendable {
    let isLegacy: Bool

    required init(tokenizerConfig: Config, tokenizerData: Config, strict: Bool = true) throws {
        isLegacy = tokenizerConfig.legacy.boolean(or: true)
        var configDictionary = tokenizerData.dictionary(or: [:])
        configDictionary["post_processor"] = try llamaPostProcessorConfig(tokenizerConfig: tokenizerConfig)
        try super.init(tokenizerConfig: tokenizerConfig, tokenizerData: Config(configDictionary), strict: strict)
    }
}

// MARK: - Local configuration

/// Reads tokenizer configuration files from a local model directory and applies the same
/// resolution rules as the Hub-backed loader: chat templates from `chat_template.jinja` /
/// `chat_template.json` are merged into the tokenizer config, and a bundled fallback
/// `tokenizer_config.json` is used for model types that historically shipped without one.
public struct LocalModelConfiguration: Sendable {
    public let modelConfig: Config?
    public let tokenizerData: Config
    private let rawTokenizerConfig: Config?

    public init(modelFolder: URL) throws {
        let fm = FileManager.default

        let tokenizerDataURL = modelFolder.appendingPathComponent("tokenizer.json")
        guard fm.fileExists(atPath: tokenizerDataURL.path) else {
            throw TokenizerError.missingFile(tokenizerDataURL)
        }
        tokenizerData = try Config(jsonFile: tokenizerDataURL)

        let modelConfigURL = modelFolder.appendingPathComponent("config.json")
        modelConfig = fm.fileExists(atPath: modelConfigURL.path) ? try Config(jsonFile: modelConfigURL) : nil

        var tokenizerConfig: Config?
        let tokenizerConfigURL = modelFolder.appendingPathComponent("tokenizer_config.json")
        if fm.fileExists(atPath: tokenizerConfigURL.path) {
            tokenizerConfig = try Config(jsonFile: tokenizerConfigURL)
        }

        // Prefer a .jinja template over a .json one.
        var chatTemplate: String?
        let jinjaURL = modelFolder.appendingPathComponent("chat_template.jinja")
        let jsonURL = modelFolder.appendingPathComponent("chat_template.json")
        if fm.fileExists(atPath: jinjaURL.path) {
            chatTemplate = try String(contentsOf: jinjaURL, encoding: .utf8)
        } else if fm.fileExists(atPath: jsonURL.path) {
            chatTemplate = try Config(jsonFile: jsonURL).chatTemplate.string()
        }

        if let chatTemplate {
            if var dict = tokenizerConfig?.dictionary() {
                dict["chat_template"] = Config(chatTemplate)
                tokenizerConfig = Config(dict)
            } else {
                tokenizerConfig = Config(["chat_template": Config(chatTemplate)])
            }
        }
        rawTokenizerConfig = tokenizerConfig
    }

    /// Model type from `config.json` (`model_type`).
    public var modelType: String? {
        modelConfig?.modelType.string()
    }

    /// The tokenizer configuration with class inference and fallbacks applied.
    public var tokenizerConfig: Config? {
        if let hubConfig = rawTokenizerConfig {
            if hubConfig.tokenizerClass.string() != nil { return hubConfig }
            guard let modelType else { return hubConfig }

            if let fallback = Self.fallbackTokenizerConfig(for: modelType) {
                // Values present in the repository's config take precedence over the fallback.
                let merged = fallback.dictionary(or: [:]).merging(hubConfig.dictionary(or: [:])) { _, fromRepo in
                    fromRepo
                }
                return Config(merged)
            }

            var configuration = hubConfig.dictionary(or: [:])
            configuration["tokenizer_class"] = Config("\(modelType.capitalized)Tokenizer")
            return Config(configuration)
        }

        guard let modelType else { return nil }
        return Self.fallbackTokenizerConfig(for: modelType)
    }

    /// Bundled fallback configuration for a `model_type`, if any (`gpt2`, `t5`).
    public static func fallbackTokenizerConfig(for modelType: String) -> Config? {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_-"))
        guard !modelType.isEmpty, modelType.unicodeScalars.allSatisfy(allowed.contains) else { return nil }
        guard let url = Bundle.module.url(forResource: "\(modelType)_tokenizer_config", withExtension: "json") else {
            return nil
        }
        return try? Config(jsonFile: url)
    }
}
