// Sequence-pair encoding against Hugging Face ground truth. `Resources/differential/pairs__*.json`
// records what the Rust `tokenizers` core produces for `encode(text, pair)`, and must match
// exactly, segment ids included: a pair template decides those, so they are 0 and 1 for
// BERT-style templates and all 0 for RoBERTa-style ones.

import Foundation
import Testing

@testable import Tokenizers

@Suite("Sequence pairs vs Hugging Face")
struct PairEncodingTests {
    struct PairRecord: Decodable {
        let text: String
        let textPair: String
        let ids: [Int]
        let typeIds: [Int]
        let idsNoSpecial: [Int]
        let typeIdsNoSpecial: [Int]
    }

    static let models = [
        "google-bert/bert-base-uncased",
        "BAAI/bge-small-en-v1.5",
        "distilbert/distilbert-base-multilingual-cased",
        "FacebookAI/xlm-roberta-base",
        "intfloat/multilingual-e5-small",
        "FacebookAI/roberta-base",
        "Qwen/Qwen3-0.6B",
        "t5-base",
    ]

    static func load(_ model: String) throws -> [PairRecord] {
        let name = "pairs__" + model.replacingOccurrences(of: "/", with: "__")
        let url = try #require(Bundle.module.url(forResource: name, withExtension: "json"))
        return try JSONDecoder().decode([PairRecord].self, from: Data(contentsOf: url))
    }

    @Test(arguments: models)
    func matchesHuggingFace(model: String) async throws {
        let records = try Self.load(model)
        let configuration = try await HubFixtures.configuration(for: model)
        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: #require(configuration.tokenizerConfig), tokenizerData: configuration.tokenizerData)

        for record in records {
            let label = "(\(record.text.debugDescription), \(record.textPair.debugDescription))"
            let encoding = try tokenizer.encode(text: record.text, textPair: record.textPair)
            #expect(encoding.ids == record.ids, "ids \(label)")
            #expect(encoding.typeIds == record.typeIds, "type ids \(label)")
            #expect(encoding.textPair == record.textPair)

            let plain = try tokenizer.encode(
                text: record.text, textPair: record.textPair, addSpecialTokens: false)
            #expect(plain.ids == record.idsNoSpecial, "ids without specials \(label)")
            #expect(plain.typeIds == record.typeIdsNoSpecial, "type ids without specials \(label)")
        }
    }

    @Test("A single sequence carries no segment ids")
    func singleSequenceHasNoTypeIds() async throws {
        let configuration = try await HubFixtures.configuration(for: "google-bert/bert-base-uncased")
        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: #require(configuration.tokenizerConfig), tokenizerData: configuration.tokenizerData)
        let encoding = try tokenizer.encode(text: "one sequence", addSpecialTokens: true, withOffsets: false)
        #expect(encoding.typeIds == nil)
        #expect(encoding.textPair == nil)
    }

    @Test("The pair ids contain both sequences in order")
    func pairContainsBothSequences() async throws {
        let configuration = try await HubFixtures.configuration(for: "google-bert/bert-base-uncased")
        let tokenizer = try AutoTokenizer.from(
            tokenizerConfig: #require(configuration.tokenizerConfig), tokenizerData: configuration.tokenizerData)
        let first = tokenizer.encode(text: "query", addSpecialTokens: false)
        let second = tokenizer.encode(text: "passage", addSpecialTokens: false)
        let pair = try tokenizer.encode(text: "query", textPair: "passage")
        #expect(pair.ids.count == first.count + second.count + 3)  // [CLS] A [SEP] B [SEP]
        #expect(Array(pair.ids[1..<(1 + first.count)]) == first)
        #expect(pair.typeIds?.prefix(2 + first.count).allSatisfy { $0 == 0 } == true)
        #expect(pair.typeIds?.suffix(1 + second.count).allSatisfy { $0 == 1 } == true)
    }
}
