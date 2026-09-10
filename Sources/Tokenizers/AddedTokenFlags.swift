// `transformers` adjusts the added-token flags serialized in `tokenizer.json` while
// constructing a fast tokenizer: `tokenizer_config.json`'s `added_tokens_decoder` re-adds any
// token whose flags differ, and several tokenizer classes rebuild their mask token as
// `AddedToken(mask_token, lstrip=True)` when the configuration leaves `mask_token` unset.

import Foundation

struct AddedTokenFlags {
    /// Flag sets from `added_tokens_decoder`, keyed by id.
    private let byId: [Int: Config]
    /// Mask token content that gets `lstrip` from the tokenizer class, if any.
    private let maskWithLeadingWhitespace: String?

    static func overrides(tokenizerConfig: Config) -> AddedTokenFlags {
        var byId: [Int: Config] = [:]
        if let decoder = tokenizerConfig["added_tokens_decoder"].dictionary() {
            for (key, value) in decoder {
                guard let id = Int(key.string), value.content.string() != nil else { continue }
                byId[id] = value
            }
        }
        return AddedTokenFlags(byId: byId, maskWithLeadingWhitespace: Self.classMaskToken(tokenizerConfig))
    }

    /// Returns `serialized` with the flags `transformers` would end up with for this token.
    func apply(to serialized: Config, id: Int, content: String) -> Config {
        var result = serialized
        if let override = byId[id], override.content.string() == content {
            var merged = serialized.dictionary(or: [:])
            for field in ["lstrip", "rstrip", "normalized", "single_word", "special"] {
                let key = BinaryDistinctString(field)
                if let value = override.dictionary()?[key] { merged[key] = value }
            }
            result = Config(merged)
        }
        if content == maskWithLeadingWhitespace {
            var merged = result.dictionary(or: [:])
            merged[BinaryDistinctString("lstrip")] = Config(true)
            merged[BinaryDistinctString("rstrip")] = Config(false)
            merged[BinaryDistinctString("normalized")] = Config(false)
            merged[BinaryDistinctString("single_word")] = Config(false)
            merged[BinaryDistinctString("special")] = Config(true)
            result = Config(merged)
        }
        return result
    }

    /// Fast tokenizer classes whose `__init__` wraps a string `mask_token` in
    /// `AddedToken(lstrip=True, rstrip=False)`, with the default content each one uses.
    /// The wrapping only happens when `mask_token` is absent from `tokenizer_config.json`:
    /// a configured value is replaced by the serialized added token before `__init__` runs.
    private static let maskTokenDefaults: [String: String] = [
        "albert": "[MASK]", "bart": "<mask>", "barthez": "<mask>", "bigbird": "[MASK]",
        "blenderbot": "<mask>", "camembert": "<mask>", "cpm": "<mask>", "deberta": "[MASK]",
        "fnet": "[MASK]", "layoutxlm": "<mask>", "led": "<mask>", "longformer": "<mask>",
        "markuplm": "<mask>", "mbart": "<mask>", "mbart50": "<mask>", "mpnet": "<mask>",
        "mvp": "<mask>", "nllb": "<mask>", "rembert": "[MASK]", "roberta": "<mask>",
        "xlmroberta": "<mask>", "xlnet": "<mask>",
    ]

    private static func classMaskToken(_ tokenizerConfig: Config) -> String? {
        guard tokenizerConfig["mask_token"].isNull(), let name = tokenizerConfig.tokenizerClass.string() else {
            return nil
        }
        var key = name.lowercased()
        for suffix in ["fast", "tokenizer"] where key.hasSuffix(suffix) { key.removeLast(suffix.count) }
        key.removeAll { $0 == "-" || $0 == "_" }
        return maskTokenDefaults[key]
    }
}
