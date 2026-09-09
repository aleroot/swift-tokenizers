import Foundation
import Testing

@testable import Tokenizers

@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["RUN_BENCHMARKS"] == "1"))
struct UnicodeBenchmarkTests {
    @Test(
        "Classification first use (run this test alone)",
        .enabled(if: ProcessInfo.processInfo.environment["UNICODE_COLD_BENCHMARK"] == "1"))
    func classificationFirstUse() {
        // Validate after timing so Swift Testing's own first-use cost is not attributed
        // to the cache. Run in a fresh process, without other benchmark tests selected.
        let start = DispatchTime.now().uptimeNanoseconds
        let flags = ScalarClassifier.flags(value: 0x391)
        let primaryEnd = DispatchTime.now().uptimeNanoseconds
        let extra = ScalarClassifier.extraFlags(value: 0x391)
        let extraEnd = DispatchTime.now().uptimeNanoseconds
        print(String(format: "  first classification: %.3f ms", Double(primaryEnd - start) / 1_000_000))
        print(String(format: "  first extra flags: %.3f ms", Double(extraEnd - primaryEnd) / 1_000_000))
        #expect(flags == ScalarFlags.letter | ScalarFlags.upper)
        #expect(extra == ScalarExtraFlags.word)
    }

    @Test("Classification cache construction cost")
    func classificationConstruction() {
        let expected = ScalarClassifier.bmp
        let expectedExtra = ScalarClassifier.bmpExtra
        benchmarkMeasure(label: "runtime classification", iterations: 15) {
            let flags = ScalarClassifier.makeBMPFlags()
            let extra = ScalarClassifier.makeBMPExtraFlags()
            #expect(flags == expected)
            #expect(extra == expectedExtra)
        }
    }

    @Test("BERT normalization tables versus one whole-text runtime fallback")
    func bertTables() {
        let normalizer = BertNormalizer(config: [:])
        for (name, sample) in [
            ("ASCII", "THE QUICK BROWN FOX jumps over THE LAZY DOG. "),
            ("Latin", "ÉLÈVE À ZÜRICH — CAFÉ, RÉSUMÉ, NAÏVE, ÜBER. "),
            ("mixed", "ΑΘΉΝΑ МОСКВА ZÜRICH 東京 한글 नमस्ते "),
        ] {
            var text = String(repeating: sample, count: 1000)
            let expected = Array(normalizer.normalize(text: text).utf8)
            benchmarkMeasure(label: name + " BERT tables", iterations: 15) {
                #expect(normalizer.normalize(text: text).utf8.elementsEqual(expected))
            }
            benchmarkMeasure(label: name + " BERT runtime", iterations: 15) {
                // Use the actual fallback once for the whole input. Calling it separately
                // for every non-ASCII run would overstate the cost of removing the tables.
                var output: [UInt8] = []
                let scratch = ScratchBuffers()
                text.withUTF8 { normalizer.normalizeWithFoundation($0, into: &output, scratch: scratch) }
                #expect(output == expected)
            }
        }
    }

    @Test("Unicode lookup alternatives, with observable checksums")
    func lookups() {
        var dense = Array(UInt16.min...UInt16.max)
        for entry in UnicodeNormalization.lowercaseIndex {
            dense[Int(entry >> 16)] = UInt16(truncatingIfNeeded: entry)
        }
        let mapped = UnicodeNormalization.lowercaseIndex.map { $0 >> 16 }
        let values = Array(repeating: mapped, count: 100).flatMap { $0 }
        let expected = values.reduce(UInt64(0)) { $0 + UInt64(dense[Int($1)]) }
        func time(_ name: String, _ lookup: (UInt32) -> UInt32) {
            benchmarkMeasure(label: name, iterations: 15) {
                let sum = values.reduce(UInt64(0)) { $0 + UInt64(lookup($1)) }
                #expect(sum == expected)
            }
        }
        time("lowercase binary search") { value in
            var lo = 0
            var hi = UnicodeNormalization.lowercaseIndex.count - 1
            let key = value << 16
            while lo < hi {
                let mid = (lo + hi) >> 1
                if UnicodeNormalization.lowercaseIndex[mid] & 0xFFFF_0000 < key { lo = mid + 1 } else { hi = mid }
            }
            return UnicodeNormalization.lowercaseIndex[lo] & 0xFFFF
        }
        time("lowercase production", UnicodeNormalization.lowercase)
        time("lowercase dense", { UInt32(dense[Int($0)]) })

        let text = String(repeating: "HÉLLO ΑΒΓДЕЖ café résumé 你好 한글 ", count: 1000)
        let scalars = text.unicodeScalars.map(\.value)
        let flags = scalars.reduce(UInt64(0)) { $0 + UInt64(ScalarClassifier.flags(value: $1)) }
        benchmarkMeasure(label: "classification table", iterations: 15) {
            #expect(scalars.reduce(UInt64(0)) { $0 + UInt64(ScalarClassifier.flags(value: $1)) } == flags)
        }
        benchmarkMeasure(label: "classification runtime", iterations: 15) {
            #expect(scalars.reduce(UInt64(0)) { $0 + UInt64(ScalarClassifier.flagsSlow(Unicode.Scalar($1)!)) } == flags)
        }
    }

    @Test("Normalization on ASCII and multilingual text")
    func normalizers() {
        let normalizer = LowercaseNormalizer(config: [:])
        for (name, sample) in [
            ("ASCII", "THE QUICK BROWN FOX jumps over THE LAZY DOG. "),
            ("Latin", "ÉLÈVE À ZÜRICH — CAFÉ, RÉSUMÉ, NAÏVE, ÜBER. "),
            ("mixed", "ΑΘΉΝΑ МОСКВА ZÜRICH 東京 한글 नमस्ते "),
        ] {
            let text = String(repeating: sample, count: 1000)
            let expected = Array(text.lowercased().utf8)
            benchmarkMeasure(label: name + " tokenizer lower", iterations: 15) {
                #expect(normalizer.normalize(text: text).utf8.elementsEqual(expected))
            }
            benchmarkMeasure(label: name + " stdlib lower", iterations: 15) {
                #expect(text.lowercased().utf8.elementsEqual(expected))
            }
        }
    }

    @Test("Unicode form fast paths and fallback cost")
    func forms() {
        for (name, sample) in [
            ("ASCII", "The quick brown fox jumps over the lazy dog. "),
            ("composed", "café résumé naïve 한글 東京 "),
            ("decomposed", "cafe\u{301} re\u{301}sume\u{301} \u{1100}\u{1161}\u{11A8} "),
        ] {
            let text = String(repeating: sample, count: 1000)
            for (form, normalizer, legacy) in [
                (
                    "NFC", NFCNormalizer(config: [:]) as any Normalizer,
                    { (s: String) in s.precomposedStringWithCanonicalMapping }
                ),
                (
                    "NFKC", NFKCNormalizer(config: [:]) as any Normalizer,
                    { (s: String) in s.precomposedStringWithCompatibilityMapping }
                ),
                (
                    "NFD", NFDNormalizer(config: [:]) as any Normalizer,
                    { (s: String) in s.decomposedStringWithCanonicalMapping }
                ),
            ] {
                let expected = Array(text.applyingTransform(StringTransform(form), reverse: false)!.utf8)
                benchmarkMeasure(label: name + " " + form + " tokenizer", iterations: 15) {
                    #expect(normalizer.normalize(text: text).utf8.elementsEqual(expected))
                }
                benchmarkMeasure(label: name + " " + form + " legacy", iterations: 15) {
                    #expect(legacy(text).utf8.elementsEqual(expected))
                }
            }
        }
    }
}
