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

/// The id-level half of post-processing: splicing special-token ids into a sequence.
///
/// Deliberately not a refinement of ``PostProcessor``, so that a tokenizer can hold one
/// non-optional reference to whatever performs the step. Reading an optional existential
/// property per encode call costs a pair of reference-count updates on an object every encoding
/// thread shares, which is what stops a shared tokenizer scaling past a few threads.
protocol IdPostProcessor: Sendable {
    func postProcess(ids: inout [Int], addSpecialTokens: Bool)

    /// This processor with everything it needs from a vocabulary already resolved. The default
    /// is `self`, which is right for every processor whose special pieces declare their own ids;
    /// binding happens once, when the tokenizer is loaded, so encoding never borrows the model.
    func bound(resolving resolve: (String) -> Int?) -> any IdPostProcessor
}

extension IdPostProcessor {
    func bound(resolving resolve: (String) -> Int?) -> any IdPostProcessor { self }
}

/// Leaves ids untouched, for a tokenizer that declares no post-processor. An empty struct, so
/// the existential holding it is copied without touching a reference count.
struct NoIdPostProcessing: IdPostProcessor {
    func postProcess(ids: inout [Int], addSpecialTokens: Bool) {}
}

/// Round-trips through token strings, for a processor that has no id-level path. A class so the
/// two existentials it holds stay inside the ``IdPostProcessor`` existential instead of being
/// boxed on the heap.
final class StringIdPostProcessing: IdPostProcessor {
    private let processor: any PostProcessor
    private let model: any TokenizingModel

    init(processor: any PostProcessor, model: any TokenizingModel) {
        self.processor = processor
        self.model = model
    }

    func postProcess(ids: inout [Int], addSpecialTokens: Bool) {
        let tokens = ids.map { model.convertIdToToken($0) ?? "" }
        let processed = processor.postProcess(
            tokens: tokens, tokensPair: nil, addSpecialTokens: addSpecialTokens)
        ids = processed.compactMap { model.convertTokenToId($0) }
    }
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

final class TemplateProcessing: PostProcessor, IdPostProcessor {
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

    /// A template piece in id space: literal ids to splice, or the sequence itself.
    private enum IdItem: Equatable {
        case ids([Int])
        case sequence
    }

    let single: [Config]
    let pair: [Config]
    let singleItems: [Item]
    let pairItems: [Item]
    /// `singleItems` compiled to ids.
    private let idItems: [IdItem]
    /// The ids around the sequence, when the compiled template contains exactly one sequence
    /// piece: the overwhelmingly common `[specials] $A [specials]` shape then costs two splices,
    /// with no loop and no temporary array.
    private let affixes: (prefix: [Int], suffix: [Int])?
    /// Whether a special piece has no declared ids, so binding to a vocabulary can change what
    /// the template compiles to.
    private let needsVocabulary: Bool

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
        needsVocabulary = singleItems.contains {
            if case let .special(_, special) = $0 { return special == nil }; return false
        }
        // Unbound: a special piece with no declared ids resolves to nothing, which is what a
        // failed vocabulary lookup contributes.
        idItems = Self.compile(singleItems) { _ in nil }
        affixes = Self.affixes(of: idItems)
    }

    private init(
        single: [Config], pair: [Config], singleItems: [Item], pairItems: [Item],
        resolving resolve: (String) -> Int?
    ) {
        self.single = single
        self.pair = pair
        self.singleItems = singleItems
        self.pairItems = pairItems
        needsVocabulary = false
        idItems = Self.compile(singleItems, resolving: resolve)
        affixes = Self.affixes(of: idItems)
    }

    func bound(resolving resolve: (String) -> Int?) -> any IdPostProcessor {
        guard needsVocabulary else { return self }
        return TemplateProcessing(
            single: single, pair: pair, singleItems: singleItems, pairItems: pairItems, resolving: resolve)
    }

    /// Resolves a template to ids. Sequence B and the pieces a template ignores contribute
    /// nothing at either level, so they drop out here.
    private static func compile(_ items: [Item], resolving resolve: (String) -> Int?) -> [IdItem] {
        items.compactMap { item in
            switch item {
            case .sequenceA: return .sequence
            case let .special(token, special):
                let ids = Self.ids(of: token, special, resolving: resolve)
                return ids.isEmpty ? nil : .ids(ids)
            case .sequenceB, .ignored: return nil
            }
        }
    }

    /// The ids a special piece contributes: its declared ids, or a vocabulary lookup for
    /// hand-written configurations that omit the `special_tokens` map.
    static func ids(of token: String, _ special: SpecialToken?, resolving resolve: (String) -> Int?) -> [Int] {
        special?.ids ?? resolve(token).map { [$0] } ?? []
    }

    /// Splits `items` into the ids before and after the sequence, or `nil` when the template
    /// names the sequence zero or several times and the general path has to run.
    private static func affixes(of items: [IdItem]) -> (prefix: [Int], suffix: [Int])? {
        guard
            let sequence = items.firstIndex(of: .sequence),
            !items[items.index(after: sequence)...].contains(.sequence)
        else { return nil }
        var prefix: [Int] = []
        var suffix: [Int] = []
        for (index, item) in items.enumerated() {
            guard case let .ids(ids) = item else { continue }
            if index < sequence {
                prefix.append(contentsOf: ids)
            } else {
                suffix.append(contentsOf: ids)
            }
        }
        return (prefix, suffix)
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

    func postProcess(ids: inout [Int], addSpecialTokens: Bool) {
        // Pre-resolved affixes: two splices, no lookups and no temporary arrays.
        if let affixes {
            guard addSpecialTokens else { return }
            if !affixes.prefix.isEmpty { ids.insert(contentsOf: affixes.prefix, at: 0) }
            if !affixes.suffix.isEmpty { ids.append(contentsOf: affixes.suffix) }
            return
        }
        let input = ids
        ids.removeAll(keepingCapacity: true)
        for item in idItems {
            switch item {
            case .sequence: ids.append(contentsOf: input)
            case let .ids(specials): if addSpecialTokens { ids.append(contentsOf: specials) }
            }
        }
    }
}

final class ByteLevelPostProcessor: PostProcessor, IdPostProcessor {
    let trimOffsets: Bool
    let addPrefixSpace: Bool

    required init(config: Config) {
        trimOffsets = config.trimOffsets.boolean(or: true)
        addPrefixSpace = config.addPrefixSpace.boolean(or: true)
    }

    func postProcess(tokens: [String], tokensPair: [String]? = nil, addSpecialTokens: Bool = true) -> [String] {
        tokens
    }

    func postProcess(ids: inout [Int], addSpecialTokens: Bool) {}
}

final class RobertaProcessing: PostProcessor, IdPostProcessor {
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

    func postProcess(ids: inout [Int], addSpecialTokens: Bool) {
        guard addSpecialTokens else { return }
        // The ids declared next to each token are authoritative, as in `tokenizers`.
        ids.insert(Int(cls.0), at: 0)
        ids.append(Int(sep.0))
    }
}

final class BertProcessing: PostProcessor, IdPostProcessor {
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

    func postProcess(ids: inout [Int], addSpecialTokens: Bool) {
        guard addSpecialTokens else { return }
        ids.insert(Int(cls.0), at: 0)
        ids.append(Int(sep.0))
    }
}

final class SequenceProcessing: PostProcessor, IdPostProcessor {
    let processors: [any PostProcessor]
    /// The nested processors that have an id-level path, in order. Resolved once, so encoding a
    /// sequence neither casts nor skips anything.
    private let idProcessors: [any IdPostProcessor]

    required init(config: Config) throws {
        let configs = try require(config.processors.array(), "Sequence post-processor", field: "processors")
        processors = try configs.compactMap { try PostProcessorFactory.fromConfig(config: $0) }
        idProcessors = processors.compactMap { $0 as? any IdPostProcessor }
    }

    private init(processors: [any PostProcessor], idProcessors: [any IdPostProcessor]) {
        self.processors = processors
        self.idProcessors = idProcessors
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

    func postProcess(ids: inout [Int], addSpecialTokens: Bool) {
        for processor in idProcessors {
            processor.postProcess(ids: &ids, addSpecialTokens: addSpecialTokens)
        }
    }

    func bound(resolving resolve: (String) -> Int?) -> any IdPostProcessor {
        SequenceProcessing(processors: processors, idProcessors: idProcessors.map { $0.bound(resolving: resolve) })
    }

    /// `true` if every nested processor supports the id-level path. A sequence that contains one
    /// which does not has to go through the string path, so that every piece of it still runs.
    var supportsFastPath: Bool {
        processors.allSatisfy { $0 is any IdPostProcessor }
    }
}
