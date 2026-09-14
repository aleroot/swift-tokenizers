// Tokenizer-class defaults, independent of model generation/stop configuration.
// See transformers' RobertaTokenizer, XLMRobertaTokenizer and GPT2Tokenizer constructors.

enum SpecialTokenDefaults {
    static func applying(to config: Config) -> Config {
        var name = config.tokenizerClass.string() ?? ""
        if name.hasSuffix("Fast") { name.removeLast(4) }
        let defaults: [BinaryDistinctString: Config]
        switch name {
        case "RobertaTokenizer", "XLMRobertaTokenizer", "Xlm-RobertaTokenizer":
            defaults = ["bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"]
        case "GPT2Tokenizer":
            defaults = ["bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>"]
        default:
            return config
        }
        var values = config.dictionary(or: [:])
        for (key, value) in defaults where values[key] == nil { values[key] = value }
        // Explicit null is an override. Defaults never create vocabulary entries or IDs.
        return Config(values)
    }
}
