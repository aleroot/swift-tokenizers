// Resolves the tokenization model for a configuration: by Hugging Face `tokenizer_class`
// name only when the authoritative `model.type` of `tokenizer.json` is absent.

import Foundation

enum TokenizerModel {
    /// `tokenizer_class` names (without a `Fast` suffix) with a known model implementation.
    static let knownTokenizers: [String: any PreTrainedTokenizerModel.Type] = [
        "BertTokenizer": BertTokenizer.self,
        "CodeGenTokenizer": BPETokenizer.self,
        "CodeLlamaTokenizer": BPETokenizer.self,
        "CohereTokenizer": BPETokenizer.self,
        "DistilbertTokenizer": BertTokenizer.self,
        "DistilBertTokenizer": BertTokenizer.self,
        "FalconTokenizer": BPETokenizer.self,
        "GemmaTokenizer": BPETokenizer.self,
        "GPT2Tokenizer": BPETokenizer.self,
        "LlamaTokenizer": BPETokenizer.self,
        "RobertaTokenizer": BPETokenizer.self,
        "T5Tokenizer": UnigramTokenizer.self,
        "TokenizersBackend": BPETokenizer.self,
        "PreTrainedTokenizer": BPETokenizer.self,
        "Qwen2Tokenizer": BPETokenizer.self,
        "WhisperTokenizer": BPETokenizer.self,
        "XLMRobertaTokenizer": UnigramTokenizer.self,
        "Xlm-RobertaTokenizer": UnigramTokenizer.self,
    ]

    /// `tokenizer.json` `model.type` values, as written by the `tokenizers` library.
    static let modelTypes: [String: any PreTrainedTokenizerModel.Type] = [
        "BPE": BPETokenizer.self,
        "Unigram": UnigramTokenizer.self,
        "WordPiece": BertTokenizer.self,
    ]

    static func unknownToken(from tokenizerConfig: Config) -> String? {
        tokenizerConfig.unkToken.content.string() ?? tokenizerConfig.unkToken.string()
    }

    /// Instantiates the model for `tokenizerConfig` / `tokenizerData`.
    ///
    /// Serialized `model.type` is authoritative, including for generic Python tokenizer
    /// classes. The class registry supports older files without a serialized model type.
    static func from(
        tokenizerConfig: Config,
        tokenizerData: Config,
        addedTokens: [String: Int],
        strict: Bool = true
    ) throws -> any TokenizingModel {
        let tokenizerName = tokenizerConfig.tokenizerClass.string()?.replacingOccurrences(of: "Fast", with: "")
        let tokenizerClass: any PreTrainedTokenizerModel.Type
        if let modelType = tokenizerData.model.type.string() {
            guard let byModelType = modelTypes[modelType] else {
                throw TokenizerError.unsupportedComponent("model `\(modelType)`")
            }
            tokenizerClass = byModelType
        } else if let tokenizerName, let registered = knownTokenizers[tokenizerName] {
            tokenizerClass = registered
        } else if strict {
            if let tokenizerName { throw TokenizerError.unsupportedTokenizer(tokenizerName) }
            throw TokenizerError.missingTokenizerClassInConfig
        } else {
            tokenizerClass = BPETokenizer.self
        }
        return try tokenizerClass.init(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens)
    }
}
