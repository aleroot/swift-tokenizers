import Foundation
import Testing

@testable import Tokenizers

/// Bounded, deterministic fuzz smoke tests run in normal CI and under Address Sanitizer.
/// Failure messages include the seed and case so a failing input can be reproduced locally.
@Suite("Seeded JSON parser stress")
struct JSONFuzzTests {
    private func value(_ rng: inout SeededGenerator, depth: Int) -> Any {
        switch rng.next() % UInt64(depth == 0 ? 5 : 7) {
        case 0: return NSNull()
        case 1: return rng.next() % 2 == 0
        case 2: return Int(rng.next() % 2_000_001) - 1_000_000
        // Binary fractions avoid the deliberate serde_json/Foundation rounding difference.
        case 3: return Double(Int(rng.next() % 2001) - 1000) / 4
        case 4: return Fuzz.text(&rng) + "\u{0000}\"\\\n"
        case 5: return (0..<Int(rng.next() % 8)).map { _ in value(&rng, depth: depth - 1) }
        default:
            var result: [String: Any] = [:]
            for index in 0..<Int(rng.next() % 8) {
                result["\(index):" + Fuzz.text(&rng, maxAtoms: 4)] = value(&rng, depth: depth - 1)
            }
            return result
        }
    }

    @Test(arguments: [UInt64(42), 20260910, 0xDEADBEEF])
    func generatedJSON(seed: UInt64) throws {
        var rng = SeededGenerator(seed: seed)
        for iteration in 0..<500 {
            let object = value(&rng, depth: 4)
            let data = try JSONSerialization.data(withJSONObject: object, options: [.fragmentsAllowed, .sortedKeys])
            let reference = Config(any: try JSONSerialization.jsonObject(with: data, options: .fragmentsAllowed))
            #expect(try Config(jsonData: data) == reference, "seed \(seed), case \(iteration)")
            #expect(try Config(tokenizerJSON: data) == reference, "seed \(seed), case \(iteration)")
        }
    }

    @Test("Packed vocabulary and merge parsing agrees with generic construction")
    func packedTables() throws {
        var rng = SeededGenerator(seed: 20260911)
        for iteration in 0..<200 {
            // Distinct prefixes also keep canonically equivalent spellings byte-distinct
            // through Foundation's dictionary bridge.
            let words = (0..<Int(rng.next() % 80 + 1)).map { "\($0):" + Fuzz.text(&rng, maxAtoms: 5) }
            var vocab = Dictionary(uniqueKeysWithValues: words.enumerated().map { ($0.element, $0.offset * 3) })
            let pairs = zip(words, words.dropFirst()).map { [$0, $1] }
            // Valid BPE models require every merge product in the vocabulary too.
            for pair in pairs where vocab[pair[0] + pair[1]] == nil {
                vocab[pair[0] + pair[1]] = vocab.count * 3
            }
            let objects: [[String: Any]] = [
                ["model": ["type": "BPE", "vocab": vocab, "merges": pairs]],
                [
                    "model": [
                        "type": "Unigram", "unk_id": 0,
                        "vocab": words.enumerated().map { [$0.element, -Double($0.offset) / 4] as [Any] },
                    ]
                ],
            ]
            for object in objects {
                let data = try JSONSerialization.data(withJSONObject: object, options: .sortedKeys)
                let generic = try Config(jsonData: data)
                let packed = try Config(tokenizerJSON: data)
                let lhs = try PreTrainedTokenizer(tokenizerConfig: Config(), tokenizerData: generic)
                let rhs = try PreTrainedTokenizer(tokenizerConfig: Config(), tokenizerData: packed)
                for word in words {
                    #expect(lhs.convertTokenToId(word) == rhs.convertTokenToId(word), "case \(iteration)")
                }
                let text = words.joined(separator: " ")
                #expect(lhs.encode(text: text) == rhs.encode(text: text), "case \(iteration)")
            }
        }
    }

    @Test("Truncations and byte mutations never overrun either parser")
    func mutatedJSON() throws {
        var rng = SeededGenerator(seed: 20260912)
        let seeds = [
            #"{"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2,"é😀":3},"merges":[["a","b"]]}}"#,
            #"{"model":{"type":"Unigram","vocab":[["<unk>",0.0],["a",-1.25]],"unk_id":0}}"#,
            #"{"added_tokens":[{"id":3,"content":"[X]","single_word":true}],"normalizer":{"type":"Sequence","normalizers":[]}}"#,
            #"["\uD83D\uDE00","\u0000",{"a":[true,false,null,1,-2.5]}]"#,
        ]
        for seed in seeds {
            let bytes = Array(seed.utf8)
            for end in 0..<bytes.count {
                let truncated = Data(bytes.prefix(end))
                #expect(throws: JSONConfigError.self) { try Config(jsonData: truncated) }
                #expect(throws: JSONConfigError.self) { try Config(tokenizerJSON: truncated) }
            }
            var cases: [[UInt8]] = []
            for _ in 0..<500 {
                var mutated = bytes
                for _ in 0..<Int(rng.next() % 4 + 1) {
                    let index = Int(rng.next() % UInt64(mutated.count))
                    mutated[index] = UInt8(truncatingIfNeeded: rng.next())
                }
                cases.append(mutated)
            }
            for bytes in cases {
                let data = Data(bytes)
                // Mutations can remain valid JSON. Parsing, materializing nested values,
                // and hashing accepted trees stresses both generic and packed storage.
                for config in [try? Config(jsonData: data), try? Config(tokenizerJSON: data)] {
                    guard let config else { continue }
                    _ = config.hashValue
                }
            }
        }
    }
}
