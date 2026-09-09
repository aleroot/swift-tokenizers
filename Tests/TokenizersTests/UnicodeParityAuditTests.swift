import Foundation
import Testing

@testable import Tokenizers

/// Opt-in exhaustive compatibility audit, with a freshly generated Python/Rust oracle.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["UNICODE_PARITY_FIXTURE"] != nil))
struct UnicodeParityAuditTests {
    struct Fixture: Decodable {
        let reference: String
        let scalarCount: Int
        let normalizers: [String: Reference]
    }
    struct Reference: Decodable {
        let configuration: Config
        let mappings: [String: String]
        let sequences: [Sequence]
    }
    struct Sequence: Decodable {
        let text: String
        let expected: String
    }

    @Test("Exhaustive scalar and seeded sequence parity with Python tokenizers")
    func audit() throws {
        let path = try #require(ProcessInfo.processInfo.environment["UNICODE_PARITY_FIXTURE"])
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        #expect(fixture.reference == "0.23.2")
        #expect(fixture.scalarCount == 0x110000 - 0x800)
        var reports: [String: [String: Any]] = [:]
        var total = 0
        for name in fixture.normalizers.keys.sorted() {
            let reference = fixture.normalizers[name]!
            let normalizer = try #require(try NormalizerFactory.fromConfig(config: reference.configuration))
            var scalarMismatches: [UInt32] = []
            var sequenceMismatches: [Int] = []
            let mappings = Dictionary(uniqueKeysWithValues: reference.mappings.map { (UInt32($0.key)!, $0.value) })
            for value in UInt32(0)..<0x110000 {
                guard let scalar = Unicode.Scalar(value) else { continue }
                let text = String(scalar)
                let expected = mappings[value] ?? text
                if !normalizer.normalize(text: text).utf8.elementsEqual(expected.utf8) {
                    scalarMismatches.append(value)
                }
            }
            for (index, sequence) in reference.sequences.enumerated() {
                if !normalizer.normalize(text: sequence.text).utf8.elementsEqual(sequence.expected.utf8) {
                    sequenceMismatches.append(index)
                }
            }
            reports[name] = ["scalars": scalarMismatches, "sequences": sequenceMismatches]
            total += scalarMismatches.count + sequenceMismatches.count
            print(
                "UNICODE PARITY \(name): \(scalarMismatches.count) scalar, \(sequenceMismatches.count) sequence mismatches; first scalars \(scalarMismatches.prefix(10).map { String($0, radix: 16) })"
            )
        }
        if let output = ProcessInfo.processInfo.environment["UNICODE_PARITY_REPORT"] {
            try JSONSerialization.data(withJSONObject: reports, options: [.sortedKeys, .prettyPrinted])
                .write(to: URL(fileURLWithPath: output))
        }
        #expect(total == 0, "See per-normalizer counts and UNICODE_PARITY_REPORT for exact mismatches")
    }
}
