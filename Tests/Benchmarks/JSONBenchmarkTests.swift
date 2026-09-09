import Foundation
import Testing

@testable import Tokenizers

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct JSONBenchmarkTests {
    @Test("JSON parsing and complete loads, measured sequentially")
    func models() async throws {
        for model in [
            "mlx-community/Qwen3-0.6B-Base-DQ5",
            "intfloat/multilingual-e5-small",
            "google-bert/bert-base-uncased",
        ] {
            let folder = try await BenchmarkFixtures.modelFolder(model)
            let data = try Data(contentsOf: folder.appendingPathComponent("tokenizer.json"))
            print("\n=== JSON: \(model), \(data.count) bytes ===")
            benchmarkMeasure(label: "Packed parse", iterations: 15) {
                _ = try! Config(tokenizerJSON: data)
            }
            benchmarkMeasure(label: "Complete load", iterations: 15) {
                _ = try! AutoTokenizer.load(from: folder)
            }
        }
    }

    @Test("JSON string workloads")
    func strings() throws {
        for (name, text) in [
            ("short ASCII", "token"),
            ("short Unicode", "日本語é"),
            ("long ASCII", String(repeating: "abcdefgh", count: 128)),
            ("long Unicode", String(repeating: "日本語é😀", count: 64)),
            ("escaped", String(repeating: "abc\"def\\ghi\n", count: 64)),
        ] {
            let row = try JSONEncoder().encode(text)
            var data = Data("{\"model\":{\"vocab\":[".utf8)
            for i in 0..<2048 {
                if i > 0 { data.append(0x2C) }
                data.append(contentsOf: "[".utf8)
                data.append(row)
                data.append(contentsOf: ",-1.5]".utf8)
            }
            data.append(contentsOf: "]}}".utf8)
            print("\n=== JSON strings: \(name), \(data.count) bytes ===")
            benchmarkMeasure(label: "Packed parse", iterations: 15) {
                _ = try! Config(tokenizerJSON: data)
            }
            benchmarkMeasure(label: "Generic parse", iterations: 15) {
                _ = try! Config(jsonData: data)
            }
        }
    }
}
