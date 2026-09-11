// Local harness template: copy into Tests/TokenizersTests only for an audit, then remove.
// Supply a tokenizers 0.23.2 fixture with MODEL_BOUNDARY_FIXTURE; MODEL_BOUNDARY_REPORT
// optionally saves the results. This is a strict raw-parity audit: canonicalTokens only
// explains known spelling differences, it does not turn a raw mismatch into a pass.
// The external fixtures/generator are not shipped here; this template alone is not a
// reproducible oracle. Permanent, self-contained boundary regressions live in ModelBoundaryTests.
import Foundation
import Testing

@testable import Tokenizers

@Suite(.serialized)
struct ModelBoundaryAuditTests {
    struct Fixture: Decodable {
        let reference: String
        let models: [Model]
    }
    struct Model: Decodable {
        let name: String
        let tokenizerJSON: String
        let tokenizerClass: String?
        let cases: [Case]
    }
    struct Case: Decodable {
        let text: String
        let ids: [Int]
        let tokens: [String]
        let canonicalTokens: [String]?
        let offsets: [[Int]?]
        let addSpecialTokens: Bool?
        let decoded: String
    }

    @Test func replay() throws {
        let path = try #require(ProcessInfo.processInfo.environment["MODEL_BOUNDARY_FIXTURE"])
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        #expect(fixture.reference == "0.23.2")
        #expect(!fixture.models.isEmpty)
        var failures: [String] = []
        var counts: [String: Int] = [:]
        var checked = 0
        var perModel: [String: [String: Int]] = [:]
        for model in fixture.models {
            let tokenizer = try PreTrainedTokenizer(
                tokenizerConfig: ["tokenizer_class": Config(model.tokenizerClass ?? "PreTrainedTokenizerFast")],
                tokenizerData:
                    Config(tokenizerJSON: Data(model.tokenizerJSON.utf8)))
            for test in model.cases {
                func check(_ surface: String, _ matches: Bool) {
                    counts[surface, default: 0] += matches ? 0 : 1
                    perModel[model.name, default: [:]][surface, default: 0] += matches ? 0 : 1
                    if !matches { failures.append("\(model.name) \(surface): \(test.text.debugDescription)") }
                }
                for surface in ["ids", "cachedIds"] {
                    check(
                        surface,
                        tokenizer.encode(text: test.text, addSpecialTokens: test.addSpecialTokens ?? false) == test.ids)
                }
                let actualTokens = tokenizer.tokenize(text: test.text).map { Array($0.utf8) }
                let rawTokensMatch = actualTokens == test.tokens.map { Array($0.utf8) }
                check("tokens", rawTokensMatch)
                if !rawTokensMatch, let canonical = test.canonicalTokens {
                    check("unexplainedTokens", actualTokens == canonical.map { Array($0.utf8) })
                }
                let encoding = try tokenizer.encode(
                    text: test.text, addSpecialTokens: test.addSpecialTokens ?? false, withOffsets: true)
                check("offsetIds", encoding.ids == test.ids)
                check("offsets", encoding.offsets == test.offsets.map { $0.map { $0[0]..<$0[1] } })
                check(
                    "decoded",
                    Array(tokenizer.decode(tokens: test.ids, skipSpecialTokens: false).utf8) == Array(test.decoded.utf8)
                )
                checked += 1
            }
        }
        print("MODEL BOUNDARY AUDIT: \(fixture.models.count) configurations, \(checked) inputs; mismatches \(counts)")
        if let report = ProcessInfo.processInfo.environment["MODEL_BOUNDARY_REPORT"] {
            try JSONSerialization.data(
                withJSONObject: [
                    "configurations": fixture.models.count,
                    "inputs": checked, "mismatches": counts, "models": perModel, "failures": failures,
                ],
                options: [.prettyPrinted, .sortedKeys]
            ).write(to: URL(fileURLWithPath: report))
        }
        if !failures.isEmpty {
            Issue.record("\(failures.count) mismatches:\n\(failures.prefix(40).joined(separator: "\n"))")
        }
    }
}
