// Verifies BMP caches and supplementary lookups against this runtime.

import Testing

@testable import Tokenizers

@Suite("Unicode classification caches")
struct UnicodeTablesTests {
    @Test("Category shortcut preserves the runtime Unicode word property")
    func wordProperty() {
        let mismatch = (UInt32(0)..<0x110000).first { value in
            guard let scalar = Unicode.Scalar(value) else { return false }
            let properties = scalar.properties
            let categoryIsWord: Bool
            switch properties.generalCategory {
            case .nonspacingMark, .spacingMark, .enclosingMark, .decimalNumber, .connectorPunctuation:
                categoryIsWord = true
            default: categoryIsWord = false
            }
            let expected = categoryIsWord || properties.isAlphabetic || value == 0x200C || value == 0x200D
            return (ScalarClassifier.extraFlags(value: value) & ScalarExtraFlags.word != 0) != expected
        }
        #expect(mismatch == nil, "word property differs at U+\(mismatch.map { String($0, radix: 16) } ?? "-")")
    }

    @Test("Effective tables match every BMP scalar of this runtime")
    func tablesMatchProperties() {
        let flags = (UInt32(0)..<0x10000).map {
            Unicode.Scalar($0).map(ScalarClassifier.flagsSlow) ?? ScalarFlags.other
        }
        let extraFlags = (UInt32(0)..<0x10000).map {
            Unicode.Scalar($0).map(ScalarClassifier.extraFlagsSlow) ?? ScalarExtraFlags.control
        }
        // Report one code point, never expand 65,536-element arrays into a CI failure log.
        let flagsMismatch = flags.indices.first { ScalarClassifier.flags(value: UInt32($0)) != flags[$0] }
        let extraMismatch = extraFlags.indices.first {
            ScalarClassifier.extraFlags(value: UInt32($0)) != extraFlags[$0]
        }
        #expect(flagsMismatch == nil, "classification differs at \(flagsMismatch.map { String($0, radix: 16) } ?? "-")")
        #expect(extraMismatch == nil, "extra flags differ at \(extraMismatch.map { String($0, radix: 16) } ?? "-")")
    }
}
