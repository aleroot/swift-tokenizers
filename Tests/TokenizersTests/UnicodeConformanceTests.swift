import Foundation
import Testing

@testable import Tokenizers

/// The official UCD corpus is supplied externally so its version is explicit.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["UNICODE_CONFORMANCE_FILE"] != nil))
struct UnicodeConformanceTests {
    @Test("Unicode normalization conformance corpus, byte for byte")
    func corpus() throws {
        let path = try #require(ProcessInfo.processInfo.environment["UNICODE_CONFORMANCE_FILE"])
        let corpus = try String(contentsOfFile: path, encoding: .utf8)
        let forms = ["NFC", "NFD", "NFKC", "NFKD"]
        let normalizers: [any ByteNormalizer] = [
            NFCNormalizer(config: [:]), NFDNormalizer(config: [:]), NFKCNormalizer(config: [:]),
            NFKDNormalizer(config: [:]),
        ]
        let foundation: [(String) -> String] = [
            { $0.precomposedStringWithCanonicalMapping }, { $0.decomposedStringWithCanonicalMapping },
            { $0.precomposedStringWithCompatibilityMapping }, { $0.decomposedStringWithCompatibilityMapping },
        ]
        var counts = [Int](repeating: 0, count: 8)
        var examples = [[String]](repeating: [], count: 8)
        var checks = 0
        var partOne = false
        var listedScalars = Set<UInt32>()
        func hex(_ text: String) -> String {
            text.unicodeScalars.map { String($0.value, radix: 16) }.joined(separator: " ")
        }
        for (lineNumber, line) in corpus.split(separator: "\n").enumerated() {
            let data = line.split(separator: "#", omittingEmptySubsequences: false)[0]
            if data.hasPrefix("@") { partOne = data.hasPrefix("@Part1") }
            guard !data.isEmpty, !data.hasPrefix("@") else { continue }
            let columns = data.split(separator: ";").prefix(5)
            guard columns.count == 5 else { continue }
            let strings = try columns.map { column in
                try String(
                    String.UnicodeScalarView(
                        column.split(whereSeparator: \.isWhitespace).map {
                            let value = try #require(UInt32($0, radix: 16))
                            return try #require(Unicode.Scalar(value))
                        }))
            }
            if partOne { listedScalars.formUnion(strings[0].unicodeScalars.map(\.value)) }
            for form in 0..<4 {
                for input in 0..<5 {
                    let expectedIndex =
                        form == 0 ? (input < 3 ? 1 : 3) : form == 1 ? (input < 3 ? 2 : 4) : form == 2 ? 3 : 4
                    let expected = strings[expectedIndex]
                    for backend in 0..<2 {
                        let actual =
                            backend == 0
                            ? normalizers[form].normalize(text: strings[input]) : foundation[form](strings[input])
                        let index = backend * 4 + form
                        if !actual.utf8.elementsEqual(expected.utf8) {
                            counts[index] += 1
                            if examples[index].count < 5 {
                                examples[index].append(
                                    "line \(lineNumber + 1), input \(hex(strings[input])), expected \(hex(expected)), actual \(hex(actual))"
                                )
                            }
                        }
                    }
                    checks += 1
                }
            }
        }
        let corpusChecks = checks
        // UAX #15 additionally requires identity for every assigned scalar absent from
        // Part 1. UnicodeData supplies that version's repertoire, including First/Last ranges.
        if let ucdPath = ProcessInfo.processInfo.environment["UNICODE_CONFORMANCE_UCD"] {
            let ucd = try String(contentsOfFile: ucdPath, encoding: .utf8)
            var rangeStart: UInt32?
            for line in ucd.split(separator: "\n") {
                let fields = line.split(separator: ";", omittingEmptySubsequences: false)
                let value = try #require(UInt32(fields[0], radix: 16))
                if fields[1].hasSuffix(", First>") {
                    rangeStart = value
                    continue
                }
                let start = fields[1].hasSuffix(", Last>") ? try #require(rangeStart) : value
                rangeStart = nil
                for codepoint in start...value {
                    guard !listedScalars.contains(codepoint), let scalar = Unicode.Scalar(codepoint) else { continue }
                    let text = String(scalar)
                    for form in 0..<4 {
                        for backend in 0..<2 {
                            let actual = backend == 0 ? normalizers[form].normalize(text: text) : foundation[form](text)
                            let index = backend * 4 + form
                            if !actual.utf8.elementsEqual(text.utf8) {
                                counts[index] += 1
                                if examples[index].count < 5 {
                                    examples[index].append("unlisted U+\(String(codepoint, radix: 16))")
                                }
                            }
                        }
                        checks += 1
                    }
                }
            }
        }
        for index in counts.indices {
            print(
                "CONFORMANCE \(index < 4 ? "Tokenizers" : "Foundation") \(forms[index % 4]): \(counts[index]) failures; \(examples[index])"
            )
        }
        print(
            "CONFORMANCE \(path): \(corpusChecks) corpus + \(checks - corpusChecks) unlisted scalar checks per backend")
        #expect(corpusChecks > 300_000)
        #expect(counts.prefix(4).allSatisfy { $0 == 0 })
    }
}
