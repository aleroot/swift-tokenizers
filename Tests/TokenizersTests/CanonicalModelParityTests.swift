import Foundation
import Testing

@testable import Tokenizers

/// Lower published legacy fixtures to the canonical dialect, independently of the
/// runtime, then compare with the checked-in Hugging Face ID/decode goldens.
@Suite("Canonical JSON model parity")
struct CanonicalModelParityTests {
    static let models = [
        "Qwen/Qwen3-0.6B", "google-bert/bert-base-uncased", "t5-base",
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    ]

    @Test("Published canonical files retain their byte model and flat template")
    func canonicalInput() throws {
        let data: Config = [
            "version": "2.0",
            "model": ["type": "BPE", "byte_level": true, "vocab": ["é": 0, "Ã": 1, "©": 2], "merges": []],
            "post_processor": [
                "type": "TemplateProcessing", "single": [["ids": [3]], ["seq": "A"]],
                "pair": [["seq": "A"], ["seq": "B", "type_id": 1]],
            ],
        ]
        let tokenizer = try AutoTokenizer.from(tokenizerConfig: [:], tokenizerData: Self.canonical(data))
        #expect(tokenizer.encode(text: "é") == [3, 1, 2])
    }

    @Test(arguments: models)
    func goldens(model: String) async throws {
        let configuration = try await HubFixtures.configuration(for: model)
        let data = try Self.canonical(configuration.tokenizerData)
        #expect(data.version.string() == "2.0")
        // Exercise the file parser's packed tables as well as the public construction API.
        let packed = try Config(tokenizerJSON: JSONEncoder().encode(data))
        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: #require(configuration.tokenizerConfig), tokenizerData: packed)
        let records = try DifferentialTests.loadHF(model)
        #expect(records.count > 250)
        for record in records {
            #expect(tokenizer.encode(text: record.text) == record.ids, Comment(rawValue: record.text.debugDescription))
            #expect(tokenizer.encode(text: record.text, addSpecialTokens: false) == record.idsNoSpecial)
            #expect(tokenizer.decode(tokens: record.ids).utf8.elementsEqual(record.decoded.utf8))
            #expect(
                tokenizer.decode(tokens: record.ids, skipSpecialTokens: true).utf8.elementsEqual(
                    record.decodedSkipSpecial.utf8))
            #expect(try tokenizer.encode(text: record.text, withOffsets: true).ids == record.ids)
        }
    }

    private static func canonical(_ data: Config) throws -> Config {
        if data.version.string() == "2.0" { return data }
        var root = try #require(data.dictionary())
        root["version"] = "2.0"
        var model = try #require(data.model.dictionary())
        if data.model.type.string() == "BPE" { model["byte_level"] = false }
        var normalizers: [Config] = data.normalizer.isNull() ? [] : [data.normalizer]

        func preTokenizer(_ node: Config) throws -> Config? {
            if node.isNull() { return nil }
            switch node.type.string() {
            case "ByteLevel":
                #expect(!node.addPrefixSpace.boolean(or: false))
                model["byte_level"] = true
                return node.useRegex.boolean(or: true)
                    ? [
                        "type": "Split", "pattern": ["Regex": Config(KnownSplitPattern.gpt2Source)],
                        "behavior": "Isolated", "invert": false,
                    ] : nil
            case "Metaspace":
                normalizers.append([
                    "type": "MetaspaceNormalizer", "replacement": node.replacement,
                    "prepend": node.prependScheme.isNull() ? "always" : node.prependScheme,
                    "drop_whitespace": false,
                ])
                return node.split.boolean(or: true)
                    ? [
                        "type": "Split", "pattern": ["String": node.replacement],
                        "behavior": "MergedWithNext", "invert": false,
                    ] : nil
            case "Sequence":
                let children = try #require(node.pretokenizers.array())
                if children.count == 2, children[0].type.string() == "WhitespaceSplit",
                    children[1].type.string() == "Metaspace"
                {
                    let metaspace = children[1]
                    #expect(metaspace.prependScheme.string(or: "always") == "always")
                    normalizers.append([
                        "type": "MetaspaceNormalizer", "replacement": metaspace.replacement,
                        "prepend": "always", "drop_whitespace": true,
                    ])
                    return [
                        "type": "Split", "pattern": ["String": metaspace.replacement],
                        "behavior": "MergedWithNext", "invert": false,
                    ]
                }
                let lowered = try children.compactMap(preTokenizer)
                return lowered.isEmpty ? nil : ["type": "Sequence", "pretokenizers": Config(lowered)]
            default: return node
            }
        }
        root["pre_tokenizer"] = try preTokenizer(data.preTokenizer)
        root["model"] = Config(model)
        root["normalizer"] = normalizers.isEmpty ? nil : ["type": "Sequence", "normalizers": Config(normalizers)]

        let processor: Config = data.postProcessor
        switch processor.type.string() {
        case "TemplateProcessing":
            func pieces(_ entries: Config) throws -> [Config] {
                let entries: [Config] = try #require(entries.array())
                return try entries.map { (item: Config) throws -> Config in
                    let sequence: Config = item["Sequence"]
                    if let name = sequence.id.string() {
                        return ["seq": Config(name), "type_id": Config(sequence.typeId.integer(or: 0))]
                    }
                    let special: Config = item["SpecialToken"]
                    let name = try #require(special.id.string())
                    let table: Config = processor["special_tokens"]
                    let entry: Config = table[BinaryDistinctString(name)]
                    let ids: [Config] = try #require(entry.ids.array())
                    return ["ids": Config(ids), "type_id": Config(special.typeId.integer(or: 0))]
                }
            }
            root["post_processor"] = [
                "type": "TemplateProcessing", "single": Config(try pieces(processor.single)),
                "pair": Config(try pieces(processor.pair)),
            ]
        case "BertProcessing":
            let cls = try #require(processor.cls.token()).0
            let sep = try #require(processor.sep.token()).0
            root["post_processor"] = [
                "type": "TemplateProcessing",
                "single": [["ids": [Config(Int(cls))]], ["seq": "A"], ["ids": [Config(Int(sep))]]],
                "pair": [
                    ["ids": [Config(Int(cls))]], ["seq": "A"], ["ids": [Config(Int(sep))]],
                    ["seq": "B", "type_id": 1], ["ids": [Config(Int(sep))], "type_id": 1],
                ],
            ]
        case "ByteLevel", nil: root["post_processor"] = nil
        default: throw TokenizerError.unsupportedComponent("test canonicalizer: \(processor.type)")
        }
        return Config(root)
    }
}
