import Foundation

/// Conversion profiles for the conventional Transformers split-file formats. Vocabulary
/// files do not identify a normalization pipeline, so unknown classes fail explicitly.
/// The serialized tokenizer.json path always takes precedence over this converter.
enum SplitTokenizerLoader {
    private enum Profile: String {
        case bert = "BertTokenizer", mpnet = "MPNetTokenizer", gpt2 = "GPT2Tokenizer"
        case qwen = "Qwen2Tokenizer", roberta = "RobertaTokenizer", clip = "CLIPTokenizer"

        var wordPiece: Bool { self == .bert || self == .mpnet }
    }

    static func load(folder: URL, config: Config?, modelType: String?) throws -> (data: Config, config: Config) {
        let inferred = [
            "bert": "BertTokenizer", "distilbert": "BertTokenizer", "mpnet": "MPNetTokenizer",
            "gpt2": "GPT2Tokenizer", "qwen2": "Qwen2Tokenizer", "qwen3": "Qwen2Tokenizer",
            "roberta": "RobertaTokenizer", "clip": "CLIPTokenizer",
        ]
        var name = config?.tokenizerClass.string() ?? modelType.flatMap { inferred[$0] }
        if name?.hasSuffix("Fast") == true { name?.removeLast(4) }
        if name == "DistilBertTokenizer" || name == "DistilbertTokenizer" { name = "BertTokenizer" }
        guard let name, let profile = Profile(rawValue: name) else {
            throw TokenizerError.unsupportedComponent(
                "split-file tokenizer profile \(name ?? "(missing tokenizer_class)"); provide tokenizer.json")
        }
        var configuration = config?.dictionary() ?? [:]
        configuration["tokenizer_class"] = Config(name)
        let specialMap = folder.appendingPathComponent("special_tokens_map.json")
        if FileManager.default.fileExists(atPath: specialMap.path) {
            for (key, value) in try Config(jsonFile: specialMap).dictionary(or: [:]) where configuration[key] == nil {
                configuration[key] = value
            }
        }
        func defaultToken(_ key: BinaryDistinctString, _ value: String, lstrip: Bool = false) {
            guard configuration[key] == nil else { return }
            configuration[key] =
                lstrip
                ? ["content": Config(value), "lstrip": true, "normalized": false, "special": true]
                : Config(value)
        }
        switch profile {
        case .bert, .mpnet:
            defaultToken("unk_token", "[UNK]")
            defaultToken("cls_token", profile == .bert ? "[CLS]" : "<s>")
            defaultToken("sep_token", profile == .bert ? "[SEP]" : "</s>")
            defaultToken("pad_token", profile == .bert ? "[PAD]" : "<pad>")
            defaultToken("mask_token", profile == .bert ? "[MASK]" : "<mask>", lstrip: profile == .mpnet)
        case .gpt2, .qwen:
            for key: BinaryDistinctString in ["unk_token", "eos_token"] { defaultToken(key, "<|endoftext|>") }
            if profile == .gpt2 {
                defaultToken("bos_token", "<|endoftext|>")
            } else {
                defaultToken("pad_token", "<|endoftext|>")
            }
        case .roberta:
            for (key, value): (BinaryDistinctString, String) in [
                ("bos_token", "<s>"), ("cls_token", "<s>"), ("eos_token", "</s>"),
                ("sep_token", "</s>"), ("unk_token", "<unk>"), ("pad_token", "<pad>"),
            ] { defaultToken(key, value) }
            defaultToken("mask_token", "<mask>", lstrip: true)
        case .clip:
            defaultToken("bos_token", "<|startoftext|>")
            for key: BinaryDistinctString in ["unk_token", "eos_token", "pad_token"] {
                defaultToken(key, "<|endoftext|>")
            }
        }
        // Slow BERT/MPNet construction supplies this default to the fast converter.
        if profile.wordPiece, configuration["clean_up_tokenization_spaces"] == nil {
            configuration["clean_up_tokenization_spaces"] = true
        }
        let cfg = Config(configuration)
        guard cfg.neverSplit.array(or: []).isEmpty else {
            throw TokenizerError.unsupportedComponent("split-file never_split; export tokenizer.json with added tokens")
        }
        let vocab: Config
        if profile.wordPiece {
            var entries: [BinaryDistinctString: Config] = [:]
            for (id, token) in try lines(folder.appendingPathComponent("vocab.txt")).enumerated() {
                guard entries[BinaryDistinctString(token)] == nil else { throw TokenizerError.malformedVocab }
                entries[BinaryDistinctString(token)] = Config(id)
            }
            vocab = Config(entries)
        } else {
            // Reuse the packed JSON scanner instead of allocating a Config node per token.
            var json = Data(#"{"model":{"vocab":"#.utf8)
            json.append(try read(folder.appendingPathComponent("vocab.json")))
            json.append(contentsOf: "}}".utf8)
            vocab = try Config(tokenizerJSON: json).model.vocab
        }
        var data: [BinaryDistinctString: Config] = [:]
        let added = try addedTokens(folder: folder, config: cfg, vocab: vocab)
        data["added_tokens"] = Config(added)
        func token(_ key: BinaryDistinctString) throws -> String {
            try require(addedTokenAsString(cfg[key]), "split-file profile", field: key.string)
        }
        func pair(_ key: BinaryDistinctString) throws -> Config {
            let content = try token(key)
            let id =
                vocab[BinaryDistinctString(content)].integer()
                ?? added.first { $0.content.string()?.utf8.elementsEqual(content.utf8) == true }?.id.integer()
            return [Config(content), Config(try require(id, "split-file profile", field: content))]
        }
        let byteLevel: Config = ["type": "ByteLevel"]
        let prefixSpace = cfg.addPrefixSpace.boolean(or: false)
        if profile.wordPiece {
            let basic = cfg.doBasicTokenize.boolean(or: true)
            data["model"] = [
                "type": "WordPiece", "vocab": vocab, "unk_token": Config(try token("unk_token")),
                "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
            ]
            data["normalizer"] = [
                "type": "BertNormalizer", "clean_text": true,
                "handle_chinese_chars": Config(basic && cfg.tokenizeChineseChars.boolean(or: true)),
                "strip_accents": basic ? cfg.stripAccents : false,
                "lowercase": Config(basic && cfg.doLowerCase.boolean(or: true)),
            ]
            data["pre_tokenizer"] = ["type": "BertPreTokenizer"]
            data["decoder"] = ["type": "WordPiece", "prefix": "##", "cleanup": true]
            data["post_processor"] = [
                "type": "BertProcessing", "cls": try pair("cls_token"), "sep": try pair("sep_token"),
            ]
        } else {
            let merges = try lines(folder.appendingPathComponent("merges.txt"))
            data["model"] = [
                "type": "BPE", "vocab": vocab, "merges": Config(merges.map { Config($0) }),
                "fuse_unk": false, "byte_fallback": false,
                "unk_token": profile == .clip ? Config(try token("unk_token")) : Config(),
                "end_of_word_suffix": profile == .clip ? "</w>" : Config(),
            ]
            data["pre_tokenizer"] = ["type": "ByteLevel", "add_prefix_space": Config(prefixSpace), "use_regex": true]
            data["decoder"] = byteLevel
            data["post_processor"] = ["type": "ByteLevel", "trim_offsets": false]
            if profile == .qwen {
                data["normalizer"] = ["type": "NFC"]
                data["pre_tokenizer"] = [
                    "type": "Sequence",
                    "pretokenizers": [
                        [
                            "type": "Split",
                            "pattern": [
                                "Regex": Config(
                                    #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
                                )
                            ], "behavior": "Isolated", "invert": false,
                        ],
                        ["type": "ByteLevel", "add_prefix_space": Config(prefixSpace), "use_regex": false],
                    ],
                ]
            } else if profile == .clip {
                data["normalizer"] = [
                    "type": "Sequence",
                    "normalizers": [
                        ["type": "NFC"], ["type": "Replace", "pattern": ["Regex": #"\s+"#], "content": " "],
                        ["type": "Lowercase"],
                    ],
                ]
                data["pre_tokenizer"] = [
                    "type": "Sequence",
                    "pretokenizers": [
                        [
                            "type": "Split",
                            "pattern": ["Regex": #"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#],
                            "behavior": "Removed", "invert": true,
                        ],
                        ["type": "ByteLevel", "add_prefix_space": false, "use_regex": true],
                    ],
                ]
                // CLIPTokenizerFast wraps backend.decode with suffix replacement and strip.
                data["decoder"] = [
                    "type": "Sequence",
                    "decoders": [
                        byteLevel,
                        ["type": "Replace", "pattern": ["String": "</w>"], "content": " "],
                        [
                            "type": "Replace", "pattern": ["Regex": #"^[\s\x{1C}-\x{1F}]+|[\s\x{1C}-\x{1F}]+$"#],
                            "content": "",
                        ],
                    ],
                ]
            }
            if profile == .roberta || profile == .clip {
                data["post_processor"] = [
                    "type": "RobertaProcessing",
                    "cls": try pair(profile == .clip ? "bos_token" : "cls_token"),
                    "sep": try pair(profile == .clip ? "eos_token" : "sep_token"),
                    "add_prefix_space": Config(profile == .roberta && prefixSpace),
                    "trim_offsets": Config(profile == .roberta),
                ]
            } else if profile == .gpt2 && cfg.addBosToken.boolean(or: false) {
                let bos = try token("bos_token")
                let single: Config = [
                    ["SpecialToken": ["id": Config(bos), "type_id": 0]],
                    ["Sequence": ["id": "A", "type_id": 0]],
                ]
                data["post_processor"] = ["type": "TemplateProcessing", "single": single, "pair": single]
            }
        }
        return (Config(data), cfg)
    }

    private static func read(_ url: URL) throws -> Data {
        guard FileManager.default.fileExists(atPath: url.path) else { throw TokenizerError.missingFile(url) }
        return try Data(contentsOf: url)
    }

    private static func lines(_ url: URL) throws -> [String] {
        let bytes = try read(url)
        guard let text = String(data: bytes, encoding: .utf8) else { throw TokenizerError.malformedVocab }
        var lines = text.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }
        return lines.map { $0.hasSuffix("\r") ? String($0.dropLast()) : $0 }
    }

    private static func addedTokens(folder: URL, config: Config, vocab: Config) throws -> [Config] {
        var byID: [Int: Config] = [:]
        var byContent: [BinaryDistinctString: Int] = [:]
        func insert(_ content: String, id: Int, flags: Config) throws {
            let key = BinaryDistinctString(content)
            if let originalID = vocab[key].integer(), originalID != id {
                throw TokenizerError.invalidConfiguration("Conflicting split-file added token \(content)")
            }
            guard id >= 0, id < Int(Int32.max), byContent[key] == nil || byContent[key] == id,
                byID[id] == nil || byID[id]?.content.string()?.utf8.elementsEqual(content.utf8) == true
            else {
                throw TokenizerError.invalidConfiguration("Conflicting split-file added token \(content)")
            }
            var entry = flags.dictionary(or: [:])
            entry["content"] = Config(content)
            entry["id"] = Config(id)
            if entry["normalized"] == nil { entry["normalized"] = Config(!flags.special.boolean(or: false)) }
            byID[id] = Config(entry)
            byContent[key] = id
        }
        let file = folder.appendingPathComponent("added_tokens.json")
        if FileManager.default.fileExists(atPath: file.path) {
            for (content, id) in try Config(jsonFile: file).dictionary(or: [:]) {
                guard case let .integer(number) = id.value else {
                    throw TokenizerError.invalidConfiguration("added_tokens.json requires integer IDs")
                }
                try insert(content.string, id: number, flags: [:])
            }
        }
        for (id, entry) in config.addedTokensDecoder.dictionary(or: [:]) {
            try insert(
                try require(entry.content.string(), "added_tokens_decoder", field: "content"),
                id: try require(Int(id.string), "added_tokens_decoder", field: "id"), flags: entry)
        }
        if let packed = vocab.asPackedStringMap() {
            for index in 0..<packed.count {
                let id = Int(packed.ids[index])
                if let entry = byID[id] {
                    let matches = packed.withBytes(at: index) { $0.elementsEqual(entry.content.string(or: "").utf8) }
                    guard matches else {
                        throw TokenizerError.invalidConfiguration("Added token ID \(id) replaces a vocabulary token")
                    }
                }
            }
        } else {
            for (content, id) in vocab.dictionary(or: [:]) {
                guard case let .integer(number) = id.value else { throw TokenizerError.malformedVocab }
                if let entry = byID[number], entry.content.string()?.utf8.elementsEqual(content.string.utf8) != true {
                    throw TokenizerError.invalidConfiguration("Added token ID \(number) replaces a vocabulary token")
                }
            }
        }
        // Register special tokens in vocabulary as atomic tokens; append absent defaults in
        // deterministic attribute order, after explicitly assigned added-token IDs.
        let maximum =
            vocab.asPackedStringMap().map { $0.ids.withUnsafeBufferPointer { $0.max().map(Int.init) } ?? -1 }
            ?? vocab.dictionary(or: [:]).values.compactMap { $0.integer() }.max() ?? -1
        let lastID = max(maximum, byID.keys.max() ?? -1)
        guard lastID < Vocabulary.maximumIdSpace else { throw TokenizerError.malformedVocab }
        var next = lastID + 1
        for attribute in specialTokenAttributes {
            let value = config[BinaryDistinctString(attribute)]
            for special in value.array() ?? (value.isNull() ? [] : [value]) {
                guard let content = addedTokenAsString(special) else { continue }
                let key = BinaryDistinctString(content)
                let id: Int
                if let addedID = byContent[key] {
                    id = addedID
                } else if let modelID = vocab[key].integer() {
                    id = modelID
                } else {
                    id = next
                }
                if id == next { next += 1 }
                var flags = byID[id]?.dictionary() ?? special.dictionary(or: [:])
                flags["special"] = true
                let explicit: Config = config.addedTokensDecoder[BinaryDistinctString(String(id))]["normalized"]
                flags["normalized"] =
                    !explicit.isNull() ? explicit : (special.normalized.isNull() ? false : special.normalized)
                try insert(content, id: id, flags: Config(flags))
            }
        }
        return byID.sorted { $0.key < $1.key }.map(\.value)
    }
}
