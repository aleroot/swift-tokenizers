// Resolves the tokenization model for a configuration: by Hugging Face `tokenizer_class`
// name first, then by the authoritative `model.type` of `tokenizer.json`.

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
    /// The `tokenizer_class` registry is consulted first (it carries model-family behaviour
    /// such as BERT's basic tokenizer). Unregistered classes — new Hub classes appear
    /// regularly — resolve through `tokenizer.json`'s own `model.type`, which is what the
    /// `tokenizers` library itself uses. Only when neither is known does `strict` decide
    /// between throwing and assuming BPE.
    static func from(
        tokenizerConfig: Config,
        tokenizerData: Config,
        addedTokens: [String: Int],
        strict: Bool = true
    ) throws -> any TokenizingModel {
        guard let tokenizerClassName = tokenizerConfig.tokenizerClass.string() else {
            throw TokenizerError.missingTokenizerClassInConfig
        }
        let tokenizerName = tokenizerClassName.replacingOccurrences(of: "Fast", with: "")

        let tokenizerClass: any PreTrainedTokenizerModel.Type
        if let registered = knownTokenizers[tokenizerName] {
            tokenizerClass = registered
        } else if let modelType = tokenizerData.model.type.string(), let byModelType = modelTypes[modelType] {
            tokenizerClass = byModelType
        } else if strict {
            throw TokenizerError.unsupportedTokenizer(tokenizerName)
        } else {
            tokenizerClass = BPETokenizer.self
        }
        return try tokenizerClass.init(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData, addedTokens: addedTokens)
    }
}
