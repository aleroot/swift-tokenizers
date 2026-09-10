// Verifies the checked-in normalization tables against Foundation's normalization and the
// scalar properties of the running toolchain, regenerates them on request, and checks the
// quick-check / decomposition / lowercase helpers against Foundation on random text.

import Foundation
import Testing

@testable import Tokenizers

@Suite("Unicode normalization tables")
struct UnicodeNormalizationTests {
    static let generatedFile = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Sources/Tokenizers/Core/UnicodeNormalization.generated.swift")

    /// Everything derived from the toolchain's Unicode data.
    struct Derived {
        var bmp = [UInt16](repeating: 0, count: 0x10000)
        /// `scalar << 16 | offset`, sorted, plus a sentinel closing the last mapping.
        var decompositionIndex: [UInt32] = []
        var decompositions: [UInt32] = []
        /// `scalar << 16 | lowercase`, sorted.
        var lowercaseIndex: [UInt32] = []
        /// Sorted disjoint `[start, end)` pairs of non-trivial supplementary scalars.
        var supplementaryRanges: [UInt32] = []
    }

    static func scalars(_ text: String) -> [UInt32] { text.unicodeScalars.map(\.value) }
    static func string(_ value: UInt32) -> String { String(Unicode.Scalar(value)!) }

    static func derive() -> Derived {
        var derived = Derived()
        var primaryComposites: [(UInt32, [UInt32])] = []
        typealias P = UnicodeNormalization.Property
        for value in UInt32(0)..<0x10000 {
            guard let scalar = Unicode.Scalar(value) else { continue }
            let text = string(value)
            let nfd = scalars(text.decomposedStringWithCanonicalMapping)
            let nfc = scalars(text.precomposedStringWithCanonicalMapping)
            let nfkd = scalars(text.decomposedStringWithCompatibilityMapping)
            var properties = UInt16(scalar.properties.canonicalCombiningClass.rawValue)
            if nfc != [value] { properties |= P.nfcUnstable }
            if nfd != [value] {
                properties |= P.canonicalDecomposition
                let isHangulSyllable =
                    value >= UnicodeNormalization.hangulBase
                    && value < UnicodeNormalization.hangulBase + UnicodeNormalization.hangulCount
                if !isHangulSyllable {
                    derived.decompositionIndex.append(value << 16 | UInt32(derived.decompositions.count))
                    derived.decompositions.append(contentsOf: nfd)
                }
                if nfc == [value] { primaryComposites.append((value, nfd)) }
            }
            if nfkd != nfd { properties |= P.compatibilityDecomposition }
            let lowercase = scalars(scalar.properties.lowercaseMapping)
            if lowercase != [value] {
                if lowercase.count == 1, lowercase[0] < 0x10000 {
                    properties |= P.lowercaseMapped
                    derived.lowercaseIndex.append(value << 16 | lowercase[0])
                } else {
                    properties |= P.lowercaseComplex
                }
            }
            if scalar.properties.generalCategory == .nonspacingMark { properties |= P.nonspacingMark }
            derived.bmp[Int(value)] = properties
        }
        derived.decompositionIndex.append(0xFFFF << 16 | UInt32(derived.decompositions.count))

        // NFC `Maybe`: the second scalar of every canonical composition pair. A primary
        // composite `c` (NFD ≠ c, NFC(c) == c) decomposes in one step to `(a, b)` where `b` is
        // the last scalar of its full decomposition and `a` recomposes from the rest; Hangul
        // syllables compose algorithmically from V and T jamo.
        for (_, nfd) in primaryComposites where !(nfd.count == 1) {
            derived.bmp[Int(nfd.last!)] |= P.nfcUnstable
        }
        for value in UInt32(0x1161)...0x1175 { derived.bmp[Int(value)] |= P.nfcUnstable }
        for value in UInt32(0x11A8)...0x11C2 { derived.bmp[Int(value)] |= P.nfcUnstable }

        var runStart: UInt32?
        for value in UInt32(0x10000)...0x10FFFF {
            var nontrivial = false
            if let scalar = Unicode.Scalar(value) {
                let text = string(value)
                let category = scalar.properties.generalCategory
                nontrivial =
                    scalar.properties.canonicalCombiningClass.rawValue != 0
                    || scalars(text.decomposedStringWithCompatibilityMapping) != [value]
                    || scalars(text.precomposedStringWithCanonicalMapping) != [value]
                    || scalars(scalar.properties.lowercaseMapping) != [value]
                    || category == .nonspacingMark || category == .spacingMark || category == .enclosingMark
            }
            if nontrivial {
                if runStart == nil { runStart = value }
            } else if let start = runStart {
                derived.supplementaryRanges.append(contentsOf: [start, value])
                runStart = nil
            }
        }
        if let start = runStart { derived.supplementaryRanges.append(contentsOf: [start, 0x110000]) }
        return derived
    }

    @Test("Stable tables and conservative fallbacks cover this runtime's Unicode data")
    func tablesMatchToolchain() throws {
        let derived = Self.derive()
        let runs = Self.encodeBMPRuns(derived.bmp)

        if ProcessInfo.processInfo.environment["REGENERATE_UNICODE_TABLES"] == "1" {
            try Self.verifyMaybeSetByComposition(derived.bmp)
            var source =
                "// Generated by `REGENERATE_UNICODE_TABLES=1 swift test --filter UnicodeNormalizationTests`. Do not edit.\n"
            source += "// See UnicodeNormalization.swift for the layout of each table.\n\n"
            source += "extension UnicodeNormalization {\n"
            source += Self.array("runs", runs, comment: "\(runs.count) runs over the BMP: start << 16 | properties")
            source += Self.array(
                "decompositionIndex", derived.decompositionIndex,
                comment: "\(derived.decompositionIndex.count - 1) canonical decompositions: scalar << 16 | offset")
            source += Self.array("decompositions", derived.decompositions, comment: "decomposed scalars, back to back")
            source += Self.array(
                "lowercaseIndex", derived.lowercaseIndex,
                comment: "\(derived.lowercaseIndex.count) simple lowercase mappings: scalar << 16 | lowercase")
            source += "}\n"
            try source.write(to: Self.generatedFile, atomically: true, encoding: .utf8)
        }

        let mismatch = derived.bmp.indices.first {
            let actual = UnicodeNormalization.bmp[$0]
            return actual != UnicodeNormalization.Property.requiresRuntime && actual != derived.bmp[$0]
        }
        #expect(mismatch == nil, "stable BMP properties differ at U+\(mismatch.map { String($0, radix: 16) } ?? "-")")
        var missing: UInt32?
        for index in stride(from: 0, to: derived.supplementaryRanges.count, by: 2) {
            for value in derived.supplementaryRanges[index]..<derived.supplementaryRanges[index + 1] {
                if !UnicodeNormalization.isSupplementaryNontrivial(value) { missing = value; break }
            }
            if missing != nil { break }
        }
        #expect(missing == nil, "supplementary fallback missing U+\(missing.map { String($0, radix: 16) } ?? "-")")
    }

    /// Brute-force check of the derived `Maybe` set: no scalar outside `nfcUnstable` composes
    /// with any first element of a canonical pair.
    static func verifyMaybeSetByComposition(_ bmp: [UInt16]) throws {
        var firstElements = Set<UInt32>()
        for value in UInt32(0)..<0x10000 where Unicode.Scalar(value) != nil {
            let text = string(value)
            let nfd = scalars(text.decomposedStringWithCanonicalMapping)
            guard nfd.count > 1, scalars(text.precomposedStringWithCanonicalMapping) == [value] else { continue }
            var prefix = ""
            for scalar in nfd.dropLast() { prefix.unicodeScalars.append(Unicode.Scalar(scalar)!) }
            firstElements.formUnion(scalars(prefix.precomposedStringWithCanonicalMapping))
        }
        for second in UInt32(0)..<0x10000 where Unicode.Scalar(second) != nil {
            guard bmp[Int(second)] & UnicodeNormalization.Property.nfcUnstable == 0 else { continue }
            // Hangul composition is algorithmic (V 1161…1175, T 11A8…11C2, marked explicitly);
            // Foundation also composes L + U+1176, which Unicode (and HF's Rust) do not.
            if (0x1100...0x11FF).contains(second) { continue }
            for first in firstElements {
                var pair = ""
                pair.unicodeScalars.append(Unicode.Scalar(first)!)
                pair.unicodeScalars.append(Unicode.Scalar(second)!)
                if scalars(pair.precomposedStringWithCanonicalMapping).count == 1 {
                    throw TokenizerError.invalidConfiguration(
                        "U+\(String(second, radix: 16)) composes with U+\(String(first, radix: 16)) but is not marked NFC-unstable"
                    )
                }
            }
        }
    }

    private static func encodeBMPRuns(_ table: [UInt16]) -> [UInt32] {
        table.indices.compactMap { index in
            guard index == 0 || table[index] != table[index - 1] else { return nil }
            return UInt32(index) << 16 | UInt32(table[index])
        }
    }

    static func array(_ name: String, _ values: [UInt32], comment: String) -> String {
        var source = "    /// \(comment).\n    static let \(name): [UInt32] = [\n"
        for chunk in stride(from: 0, to: values.count, by: 8) {
            let line = values[chunk..<min(chunk + 8, values.count)].map { String(format: "0x%08X", $0) }
            source += "        " + line.joined(separator: ", ") + ",\n"
        }
        return source + "    ]\n"
    }

    // MARK: - Behaviour against Foundation

    @Test("Paged lowercase mapping covers every BMP value, including identity pages")
    func pagedLowercase() {
        var expected = Array(UInt32(0)..<0x10000)
        for entry in UnicodeNormalization.lowercaseIndex {
            expected[Int(entry >> 16)] = entry & 0xFFFF
        }
        let mismatch = expected.indices.first { UnicodeNormalization.lowercase(UInt32($0)) != expected[$0] }
        #expect(mismatch == nil, "lowercase differs at U+\(mismatch.map { String($0, radix: 16) } ?? "-")")
    }

    @Test("Supplementary composition letters cannot bypass normalization")
    func supplementaryComposition() {
        // Kirat Rai vowel signs are letters (Lm), with combining class zero.
        // A marks-only fallback set incorrectly certifies their pairs as NFC.
        for value: UInt32 in [0x16D63, 0x16D67] {
            #expect(UnicodeNormalization.properties(of: value) & UnicodeNormalization.Property.requiresRuntime != 0)
        }
        for var text in ["\u{16D67}\u{16D67}", "\u{16D63}\u{16D67}"] {
            #expect(
                !text.withUTF8 { UnicodeNormalization.isNormalized($0, mask: UnicodeNormalization.Property.notNFC) })
            // Kirat Rai was added after the reference normalizer's Unicode 9 data.
            #expect(NFCNormalizer(config: [:]).normalize(text: text).utf8.elementsEqual(text.utf8))
        }
    }

    static let sampleTexts = [
        "plain ascii", "café résumé naïve", "e\u{301}le\u{300}ve", "Ǖ ǖ ḉ ự", "İstanbul ǅ ﬁ Straße",
        "한글 \u{1100}\u{1161}\u{11A8} ᄀ", "ẛ̣ ΐ", "Ａ１", "สวัสดี ครับ", "नमस्ते दुनिया", "👩‍💻 🇮🇹", "a\u{0323}\u{0307}",
        "a\u{0307}\u{0323}", "ｶﾞ", "Ω≠Ω", "\u{1D15E} \u{1E000}", "\u{0344}", "x\u{0F73}", "\u{0CCB}", "ﬃ",
    ]

    @Test("Quick checks agree with Foundation on normalized and unnormalized text")
    func quickChecks() {
        typealias P = UnicodeNormalization.Property
        let forms: [(UInt16, (String) -> String)] = [
            (P.notNFC, { $0.precomposedStringWithCanonicalMapping }),
            (P.notNFD, { $0.decomposedStringWithCanonicalMapping }),
            (P.notNFKC, { $0.precomposedStringWithCompatibilityMapping }),
            (P.notNFKD, { $0.decomposedStringWithCompatibilityMapping }),
        ]
        for text in Self.sampleTexts {
            for (mask, normalize) in forms {
                var copy = text
                let claimed = copy.withUTF8 { UnicodeNormalization.isNormalized($0, mask: mask) }
                // A `true` must be exact; a `false` may be a `Maybe`.
                if claimed {
                    #expect(
                        normalize(text).utf8.elementsEqual(text.utf8), "\(text.debugDescription) claimed normalized")
                }
            }
        }
        // Everyday NFC text (precomposed Latin, Thai, Devanagari, Hangul syllables, emoji) is
        // recognised exactly; only sequences ending in a composable mark are `Maybe`.
        for text in [
            "plain ascii", "café résumé naïve", "한글 각", "สวัสดี ครับ", "नमस्ते दुनिया", "👩‍💻 🇮🇹", "日本語", "“quotes” — dash",
        ] {
            var copy = text
            #expect(copy.withUTF8 { UnicodeNormalization.isNormalized($0, mask: P.notNFC) }, "\(text)")
        }
        var maybe = "ạ\u{0307}"
        #expect(!maybe.withUTF8 { UnicodeNormalization.isNormalized($0, mask: P.notNFC) })
    }

    @Test("Table decomposition and lowercase agree with Foundation and the stdlib")
    func decompositionAndLowercase() {
        typealias P = UnicodeNormalization.Property
        for value in UInt32(0)..<0x10000 {
            guard let scalar = Unicode.Scalar(value) else { continue }
            let properties = UnicodeNormalization.properties(of: value)
            if properties & P.canonicalDecomposition != 0 {
                var decomposed: [UInt32] = []
                UnicodeNormalization.decompose(value) { decomposed.append($0) }
                #expect(decomposed == Self.scalars(Self.string(value).decomposedStringWithCanonicalMapping))
            }
            if properties & P.lowercaseMapped != 0 {
                #expect([UnicodeNormalization.lowercase(value)] == Self.scalars(scalar.properties.lowercaseMapping))
            }
        }
        #expect(UnicodeNormalization.properties(of: 0x1F600) == 0)
        #expect(UnicodeNormalization.properties(of: 0x1D165) & P.requiresRuntime != 0)
    }

    @Test("Version-dependent scalars take exact runtime normalization and case paths")
    func runtimeDependentScalars() throws {
        let lowercase = LowercaseNormalizer(config: [:])
        let bert = BertNormalizer(config: ["lowercase": true, "strip_accents": true])
        let forms: [(any Normalizer, (String) -> String)] = [
            (NFCNormalizer(config: [:]), { $0.precomposedStringWithCanonicalMapping }),
            (NFDNormalizer(config: [:]), { $0.decomposedStringWithCanonicalMapping }),
            (NFKCNormalizer(config: [:]), { $0.precomposedStringWithCompatibilityMapping }),
            (NFKDNormalizer(config: [:]), { $0.decomposedStringWithCompatibilityMapping }),
        ]
        for index in stride(from: 0, to: UnicodeCompatibility.bmpRanges.count, by: 2) {
            for value in UnicodeCompatibility.bmpRanges[index]..<UnicodeCompatibility.bmpRanges[index + 1] {
                #expect(UnicodeNormalization.properties(of: value) == UnicodeNormalization.Property.requiresRuntime)
                // U+0130 forces the complex case path; the result of the preceding scalar
                // must be the same whether the fast path succeeds or rolls back.
                for text in [Self.string(value), "A" + Self.string(value) + "\u{301}", Self.string(value) + "İ"] {
                    #expect(lowercase.normalize(text: text).utf8.elementsEqual(text.lowercased().utf8))
                    let expected = BertNormalizer.stripAccents(text).lowercased()
                    #expect(bert.normalize(text: text).utf8.elementsEqual(expected.utf8))
                    for (normalizer, reference) in forms
                    where text.unicodeScalars.allSatisfy({
                        ($0.properties.age?.major ?? Int.max) <= 9
                    }) {
                        #expect(normalizer.normalize(text: text).utf8.elementsEqual(reference(text).utf8))
                    }
                }
            }
        }
    }
    @Test("Tokenizer Unicode versions preserve newer scalars and historical categories")
    func referenceUnicodeVersions() throws {
        let bert = BertNormalizer(config: [:])
        #expect(bert.normalize(text: "\u{2028}\u{2029}") == "  ")
        #expect(bert.normalize(text: "a\u{2B820}b\u{2B920}c") == "a\u{2B820}b \u{2B920} c")
        #expect(bert.normalize(text: "\u{890}é\u{1E08F}") == "\u{890}e\u{1E08F}")
        #expect(bert.normalize(text: "\u{1734}\u{1885}\u{A9BD}\u{1171E}") == "\u{1885}\u{A9BD}")
        #expect(StripAccentsNormalizer(config: [:]).normalize(text: "\u{1CF2}\u{1CF3}\u{111C9}") == "\u{111C9}")
        for form: any Normalizer in [
            NFDNormalizer(config: [:]), NFCNormalizer(config: [:]),
            NFKDNormalizer(config: [:]), NFKCNormalizer(config: [:]),
        ] {
            // New combining marks are inert starters; older marks still reorder after them.
            #expect(form.normalize(text: "e\u{1ABF}\u{301}\u{323}").utf8.elementsEqual("e\u{1ABF}\u{323}\u{301}".utf8))
            #expect(form.normalize(text: "㋿\u{16D63}\u{16D67}").utf8.elementsEqual("㋿\u{16D63}\u{16D67}".utf8))
        }
        let tokenizer = try PreTrainedTokenizer(
            tokenizerConfig: [:],
            tokenizerData: [
                "normalizer": ["type": "NFC"],
                "model": ["type": "BPE", "vocab": ["e": 0, "\u{1ABF}": 1, "\u{301}": 2, "\u{323}": 3], "merges": []],
            ])
        let result = try tokenizer.encode(text: "e\u{1ABF}\u{301}\u{323}", withOffsets: true)
        #expect(result.ids == [0, 1, 3, 2])
        // HF attaches reordered normalization edits to the original scalar positions.
        #expect(result.offsets == [0..<1, 1..<4, 4..<6, 6..<8])
    }

}
