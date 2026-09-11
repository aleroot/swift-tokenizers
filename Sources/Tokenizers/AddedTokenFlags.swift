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

    /// The flags `transformers` ends up with for one serialized added token.
    struct Resolved {
        var lstrip: Bool
        var rstrip: Bool
        var normalized: Bool
        var singleWord: Bool
        var special: Bool
    }

    /// Resolves the flags of `serialized` (an `added_tokens` entry) without materializing a
    /// merged configuration per token.
    func resolve(_ serialized: Config, id: Int, content: String) -> Resolved {
        let override: Config? = byId[id].flatMap { $0.content.string() == content ? $0 : nil }
        func flag(_ name: BinaryDistinctString) -> Bool {
            if let override, !override[name].isNull() { return override[name].boolean(or: false) }
            return serialized[name].boolean(or: false)
        }
        if content == maskWithLeadingWhitespace {
            return Resolved(lstrip: true, rstrip: false, normalized: false, singleWord: false, special: true)
        }
        return Resolved(
            lstrip: flag("lstrip"), rstrip: flag("rstrip"), normalized: flag("normalized"),
            singleWord: flag("single_word"), special: flag("special"))
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
