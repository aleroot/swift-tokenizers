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
    enum Item: Sendable {
        case special(String)
        case sequenceA
        case sequenceB
        case ignored
    }

    let single: [Config]
    let pair: [Config]
    let singleItems: [Item]
    let pairItems: [Item]

    required init(config: Config) throws {
        single = try require(config.single.array(), "TemplateProcessing", field: "single")
        pair = try require(config.pair.array(), "TemplateProcessing", field: "pair")
        singleItems = single.map(Self.item)
        pairItems = pair.map(Self.item)
    }

    private static func item(_ config: Config) -> Item {
        if let id = config.SpecialToken.id.string() { return .special(id) }
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
            case let .special(id):
                if addSpecialTokens { out.append(id) }
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
        // Fast path for the overwhelmingly common `[specials…] A [specials…]` shape.
        var prefix: [Int] = []
        var suffix: [Int] = []
        var seenA = false
        for item in singleItems {
            switch item {
            case let .special(token):
                guard addSpecialTokens, let id = resolve(token) else { continue }
                if seenA { suffix.append(id) } else { prefix.append(id) }
            case .sequenceA:
                seenA = true
            case .sequenceB, .ignored:
                break
            }
        }
        if !prefix.isEmpty {
            ids.insert(contentsOf: prefix, at: 0)
        }
        if !suffix.isEmpty {
            ids.append(contentsOf: suffix)
        }
    }
}

final class ByteLevelPostProcessor: PostProcessor, FastPostProcessor {
    required init(config: Config) {}

    func postProcess(tokens: [String], tokensPair: [String]? = nil, addSpecialTokens: Bool = true) -> [String] {
        tokens
    }

    func postProcess(ids: inout [Int], addSpecialTokens: Bool, resolve: (String) -> Int?) {}
}

final class RobertaProcessing: PostProcessor, FastPostProcessor {
    private let sep: (UInt, String)
    private let cls: (UInt, String)
    /// Trim all remaining space, or leave one space character if `addPrefixSpace` is `true`.
    private let trimOffset: Bool
    /// Keep one space character on each side. Depends on `trimOffsets` being `true`.
    private let addPrefixSpace: Bool

    required init(config: Config) throws {
        sep = try require(config.sep.token(), "RobertaProcessing", field: "sep")
        cls = try require(config.cls.token(), "RobertaProcessing", field: "cls")
        trimOffset = config.trimOffset.boolean(or: true)
        addPrefixSpace = config.addPrefixSpace.boolean(or: true)
    }

    func postProcess(tokens: [String], tokensPair: [String]?, addSpecialTokens: Bool = true) -> [String] {
        // Like `tokenizers`, the processor is a no-op when special tokens are not requested.
        guard addSpecialTokens else { return tokens + (tokensPair ?? []) }
        var outTokens = tokens
        var tokensPair = tokensPair
        if trimOffset {
            if addPrefixSpace {
                outTokens = outTokens.map { Self.trimExtraSpaces($0) }
                tokensPair = tokensPair?.map { Self.trimExtraSpaces($0) }
            } else {
                outTokens = outTokens.map { $0.trimmingCharacters(in: .whitespaces) }
                tokensPair = tokensPair?.map { $0.trimmingCharacters(in: .whitespaces) }
            }
        }

        outTokens = [cls.1] + outTokens + [sep.1]
        if let tokensPair, !tokensPair.isEmpty {
            // Yes, it adds another `sep`.
            // https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/roberta/hub_interface.py#L58-L65
            outTokens += [sep.1] + tokensPair + [sep.1]
        }
        return outTokens
    }

    func postProcess(ids: inout [Int], addSpecialTokens: Bool, resolve: (String) -> Int?) {
        // Offset trimming only affects tokens containing literal whitespace, which cannot occur
        // in byte-level vocabularies.
        guard addSpecialTokens else { return }
        if let c = resolve(cls.1) { ids.insert(c, at: 0) }
        if let s = resolve(sep.1) { ids.append(s) }
    }

    /// Some tokens need one space around them.
    /// https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/pre_tokenizers/byte_level.rs#L203-L235
    private static func trimExtraSpaces(_ token: String) -> String {
        let prefixOffset = findPrefixIndex(token)
        let suffixOffset = findSuffixIndex(token)
        let prefixIndex = token.index(token.startIndex, offsetBy: prefixOffset)
        let suffixIndex = token.index(token.startIndex, offsetBy: token.count - suffixOffset)
        return String(token[prefixIndex..<suffixIndex])
    }

    private static func findPrefixIndex(_ text: String) -> Int {
        guard let first = text.first, first.isWhitespace else { return 0 }
        return text.prefix(while: { $0.isWhitespace }).count - 1
    }

    private static func findSuffixIndex(_ text: String) -> Int {
        guard let last = text.last, last.isWhitespace else { return 0 }
        return text.reversed().prefix(while: { $0.isWhitespace }).count - 1
    }
}

final class BertProcessing: PostProcessor, FastPostProcessor {
    private let sep: (UInt, String)
    private let cls: (UInt, String)

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
        if let c = resolve(cls.1) { ids.insert(c, at: 0) }
        if let s = resolve(sep.1) { ids.append(s) }
    }
}

final class SequenceProcessing: PostProcessor, FastPostProcessor {
    private let processors: [any PostProcessor]

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
