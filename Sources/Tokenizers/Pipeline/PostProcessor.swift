import Foundation

/// Adds special tokens ([CLS]/[SEP], <s>/</s>, BOS/EOS…) around tokenized sequences.
public protocol PostProcessor: Sendable {
    func postProcess(tokens: [String], tokensPair: [String]?, addSpecialTokens: Bool) -> [String]
    func callAsFunction(tokens: [String], tokensPair: [String]?, addSpecialTokens: Bool) -> [String]
    /// Creates the post-processor from its `tokenizer.json` entry.
    /// - Throws: ``TokenizerError/invalidConfiguration(_:)`` when required fields are missing.
    init(config: Config) throws
}

public extension PostProcessor {
    func callAsFunction(tokens: [String], tokensPair: [String]? = nil, addSpecialTokens: Bool = true) -> [String] {
        postProcess(tokens: tokens, tokensPair: tokensPair, addSpecialTokens: addSpecialTokens)
    }
}

/// Internal id-level fast path. `resolve` maps a special token string to its id.
protocol FastPostProcessor: PostProcessor {
    func postProcess(ids: inout [Int], addSpecialTokens: Bool, resolve: (String) -> Int?)
}

enum PostProcessorType: String {
    case TemplateProcessing
    case ByteLevel
    case RobertaProcessing
    case BertProcessing
    case Sequence
}

struct PostProcessorFactory {
    static func fromConfig(config: Config?) throws -> (any PostProcessor)? {
        guard let config, let typeName = config.type.string() else { return nil }
        switch PostProcessorType(rawValue: typeName) {
        case .TemplateProcessing: return try TemplateProcessing(config: config)
        case .ByteLevel: return ByteLevelPostProcessor(config: config)
        case .RobertaProcessing: return try RobertaProcessing(config: config)
        case .BertProcessing: return try BertProcessing(config: config)
        case .Sequence: return try SequenceProcessing(config: config)
        default: throw TokenizerError.unsupportedComponent("post-processor `\(typeName)`")
        }
    }
}

final class TemplateProcessing: PostProcessor, FastPostProcessor {
    /// One `SpecialToken` entry of the processor's own `special_tokens` map. A piece can
    /// expand to several tokens, and its ids are authoritative: `tokenizers` inserts them
    /// without consulting the vocabulary.
    struct SpecialToken: Sendable {
        let ids: [Int]
        let tokens: [String]

        init?(config: Config) {
            guard let tokens = config.tokens.array()?.compactMap({ $0.string() }), !tokens.isEmpty,
                let ids = config.ids.array()?.compactMap({ $0.integer() }), ids.count == tokens.count
            else { return nil }
            self.ids = ids
            self.tokens = tokens
        }
    }

    enum Item: Sendable {
        /// The template's identifier, and its `special_tokens` entry when the file has one.
        case special(String, SpecialToken?)
        case sequenceA
        case sequenceB
        case ignored
    }

    let single: [Config]
    let pair: [Config]
    let singleItems: [Item]
    let pairItems: [Item]
    /// Ids to splice around the sequence, resolved once for the overwhelmingly common
    /// `[specials…] $A [specials…]` template whose pieces all declare their own ids.
    private let staticAffixes: (prefix: [Int], suffix: [Int])?

    required init(config: Config) throws {
        single = try require(config.single.array(), "TemplateProcessing", field: "single")
        pair = try require(config.pair.array(), "TemplateProcessing", field: "pair")
        var specials: [String: SpecialToken] = [:]
        for (key, value) in config.specialTokens.dictionary(or: [:]) {
            specials[key.string] = SpecialToken(config: value)
        }
        singleItems = single.map { Self.item($0, specials: specials) }
        pairItems = pair.map { Self.item($0, specials: specials) }
        guard
            !singleItems.contains(where: {
                if case .sequenceB = $0 { return true }; return false
            })
        else {
            throw TokenizerError.invalidConfiguration("TemplateProcessing single template references sequence B")
        }
        staticAffixes = Self.staticAffixes(of: singleItems)
    }

    /// Splits `items` into the ids before and after the single `$A`, or `nil` when the shape
    /// or the declared ids make that impossible and the general path has to run.
    private static func staticAffixes(of items: [Item]) -> (prefix: [Int], suffix: [Int])? {
        var prefix: [Int] = []
        var suffix: [Int] = []
        var seenSequence = false
        for item in items {
            switch item {
            case .sequenceA:
                if seenSequence { return nil }
                seenSequence = true
            case let .special(_, special):
                guard let special else { return nil }  // needs a vocabulary lookup
                if seenSequence {
                    suffix.append(contentsOf: special.ids)
                } else {
                    prefix.append(contentsOf: special.ids)
                }
            case .sequenceB, .ignored:
                return nil
            }
        }
        return seenSequence ? (prefix, suffix) : nil
    }

    private static func item(_ config: Config, specials: [String: SpecialToken]) -> Item {
        if let id = config.SpecialToken.id.string() { return .special(id, specials[id] ?? nil) }
        switch config.Sequence.id.string() {
        case "A": return .sequenceA
        case "B": return .sequenceB
        default: return .ignored
        }
    }

    func postProcess(tokens: [String], tokensPair: [String]? = nil, addSpecialTokens: Bool = true) -> [String] {
        let items = tokensPair == nil ? singleItems : pairItems
        var out: [String] = []
        out.reserveCapacity(tokens.count + (tokensPair?.count ?? 0) + items.count)
        for item in items {
            switch item {
            case let .special(id, special):
                if addSpecialTokens { out.append(contentsOf: special?.tokens ?? [id]) }
            case .sequenceA:
                out.append(contentsOf: tokens)
            case .sequenceB:
                out.append(contentsOf: tokensPair!)
            case .ignored:
                break
            }
        }
        return out
    }

    func postProcess(ids: inout [Int], addSpecialTokens: Bool, resolve: (String) -> Int?) {
        // Pre-resolved affixes: two splices, no lookups and no temporary arrays.
        if let staticAffixes {
            guard addSpecialTokens else { return }
            if !staticAffixes.prefix.isEmpty { ids.insert(contentsOf: staticAffixes.prefix, at: 0) }
            if !staticAffixes.suffix.isEmpty { ids.append(contentsOf: staticAffixes.suffix) }
            return
        }
        let input = ids
        ids.removeAll(keepingCapacity: true)
        for item in singleItems {
            switch item {
            case .sequenceA: ids.append(contentsOf: input)
            case let .special(token, special):
                if addSpecialTokens { Self.appendIds(token, special, resolve, to: &ids) }
            case .sequenceB, .ignored: break
            }
        }
    }

    /// Appends the ids a special piece contributes: its declared ids, or a vocabulary lookup
    /// for hand-written configurations that omit the `special_tokens` map.
    @inline(__always)
    static func appendIds(
        _ token: String, _ special: SpecialToken?, _ resolve: (String) -> Int?, to output: inout [Int]
    ) {
        if let special {
            output.append(contentsOf: special.ids)
        } else if let id = resolve(token) {
            output.append(id)
        }
    }
}

final class ByteLevelPostProcessor: PostProcessor, FastPostProcessor {
    let trimOffsets: Bool
    let addPrefixSpace: Bool

    required init(config: Config) {
        trimOffsets = config.trimOffsets.boolean(or: true)
        addPrefixSpace = config.addPrefixSpace.boolean(or: true)
    }

    func postProcess(tokens: [String], tokensPair: [String]? = nil, addSpecialTokens: Bool = true) -> [String] {
        tokens
    }

    func postProcess(ids: inout [Int], addSpecialTokens: Bool, resolve: (String) -> Int?) {}
}

final class RobertaProcessing: PostProcessor, FastPostProcessor {
    let sep: (UInt, String)
    let cls: (UInt, String)
    /// Trim all remaining space, or leave one space character if `addPrefixSpace` is `true`.
    let trimOffset: Bool
    /// Keep one space character on each side. Depends on `trimOffsets` being `true`.
    let addPrefixSpace: Bool

    required init(config: Config) throws {
        sep = try require(config.sep.token(), "RobertaProcessing", field: "sep")
        cls = try require(config.cls.token(), "RobertaProcessing", field: "cls")
        trimOffset = config.trimOffsets.boolean() ?? config.trimOffset.boolean(or: true)
        addPrefixSpace = config.addPrefixSpace.boolean(or: true)
    }

    func postProcess(tokens: [String], tokensPair: [String]?, addSpecialTokens: Bool = true) -> [String] {
        // Like `tokenizers`, the processor is a no-op when special tokens are not requested.
        // `trim_offsets` only moves offset boundaries (see `PostProcessor.processOffsets`);
        // upstream never rewrites token content, so neither does this.
        guard addSpecialTokens else { return tokens + (tokensPair ?? []) }
        var outTokens = [cls.1] + tokens + [sep.1]
        if let tokensPair, !tokensPair.isEmpty {
            // RoBERTa pairs carry a second `sep`:
            // https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/roberta/hub_interface.py#L58-L65
            outTokens += [sep.1] + tokensPair + [sep.1]
        }
        return outTokens
    }

    func postProcess(ids: inout [Int], addSpecialTokens: Bool, resolve: (String) -> Int?) {
        guard addSpecialTokens else { return }
        // The ids declared next to each token are authoritative, as in `tokenizers`.
        ids.insert(Int(cls.0), at: 0)
        ids.append(Int(sep.0))
    }
}

final class BertProcessing: PostProcessor, FastPostProcessor {
    let sep: (UInt, String)
    let cls: (UInt, String)

    required init(config: Config) throws {
        sep = try require(config.sep.token(), "BertProcessing", field: "sep")
        cls = try require(config.cls.token(), "BertProcessing", field: "cls")
    }

    func postProcess(tokens: [String], tokensPair: [String]?, addSpecialTokens: Bool = true) -> [String] {
        guard addSpecialTokens else { return tokens + (tokensPair ?? []) }
        var outTokens = [cls.1] + tokens + [sep.1]
        if let tokensPair, !tokensPair.isEmpty {
            outTokens += tokensPair + [sep.1]
        }
        return outTokens
    }

    func postProcess(ids: inout [Int], addSpecialTokens: Bool, resolve: (String) -> Int?) {
        guard addSpecialTokens else { return }
        ids.insert(Int(cls.0), at: 0)
        ids.append(Int(sep.0))
    }
}

final class SequenceProcessing: PostProcessor, FastPostProcessor {
    let processors: [any PostProcessor]

    required init(config: Config) throws {
        let configs = try require(config.processors.array(), "Sequence post-processor", field: "processors")
        processors = try configs.compactMap { try PostProcessorFactory.fromConfig(config: $0) }
    }

    func postProcess(tokens: [String], tokensPair: [String]?, addSpecialTokens: Bool = true) -> [String] {
        var currentTokens = tokens
        var currentTokensPair = tokensPair
        for processor in processors {
            currentTokens = processor.postProcess(
                tokens: currentTokens, tokensPair: currentTokensPair, addSpecialTokens: addSpecialTokens)
            currentTokensPair = nil
        }
        return currentTokens
    }

    func postProcess(ids: inout [Int], addSpecialTokens: Bool, resolve: (String) -> Int?) {
        for processor in processors {
            guard let fast = processor as? any FastPostProcessor else { continue }
            fast.postProcess(ids: &ids, addSpecialTokens: addSpecialTokens, resolve: resolve)
        }
    }

    /// `true` if every nested processor supports the id-level fast path.
    var supportsFastPath: Bool {
        processors.allSatisfy { $0 is any FastPostProcessor }
    }
}
