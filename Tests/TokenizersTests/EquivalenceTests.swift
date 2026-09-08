// Randomised equivalence tests for the hand-optimised components against their reference
// implementations (NSRegularExpression, the upstream regex-based splitter, JSONSerialization,
// and the heap-based BPE merge loop).

import Foundation
import Testing

@testable import Tokenizers

/// Deterministic xorshift generator so failures are reproducible.
struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed }
    mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}

enum Fuzz {
    /// Building blocks covering every class the scanners distinguish. `\u{0B}` and `\u{85}` are
    /// excluded on purpose: ICU's `\s` (`[\t\n\f\r\p{Z}]`) does not include them while Unicode
    /// White_Space (used by Rust `tokenizers` and by our scanner) does.
    static let atoms: [String] = [
        " ", "  ", "\t", "\n", "\r", "\r\n", "\u{0C}", "\u{A0}", "\u{2003}", "\u{3000}", "\u{2028}",
        "a", "Z", "hello", "World", "über", "naïve", "Zürich", "日本", "語", "한", "ж", "Ω", "ﬁ",
        // Case classes exercised by the o200k alternation: Lu/Lt/Ll/Lm/Lo mixes.
        "camelCase", "HTTPServer", "iOS", "ABC", "ǅ", "ǅx", "Xǅ", "ʰ", "aʰB", "AʰB", "ʰA", "日A", "A日b", "Éé", "ÉÉ", "ſ",
        "e\u{0301}A", "A\u{0301}", "A\u{0301}B", "!\u{0301}", "!!\u{0301}!", " \u{0301}", "://", "/\n/", "-\n/-",
        "don't", "DON'T", "it's", "he'sx", "I'LL", "we're", "'A",
        "0", "42", "١٢٣", "३", "Ⅻ", "½", "²",
        ".", ",", "!?", "...", "-", "—", "\"", "(", ")", "#", "@", "$", "%", "/", "\\", "<", ">", "|", "~", "`",
        "'", "'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'T", "'RE", "'Ve", "'M", "'LL", "'D", "'x", "'ſ",
        "'sure",
        "\u{0301}", "\u{0308}", "\u{200D}", "😀", "🚀", "👨‍👩‍👧", "🇮🇹", "1️⃣",
        "<|endoftext|>", "<s>", "</s>", "[MASK]",
    ]

    static func text(_ rng: inout SeededGenerator, maxAtoms: Int = 24) -> String {
        let count = Int(rng.next() % UInt64(maxAtoms)) + 1
        var s = ""
        for _ in 0..<count {
            s += atoms[Int(rng.next() % UInt64(atoms.count))]
        }
        return s
    }
}

@Suite("Scanner equivalence with NSRegularExpression")
struct ScannerEquivalenceTests {
    static let patterns: [(KnownSplitPattern, String)] = [
        (.gpt2, KnownSplitPattern.gpt2Source),
        (.llama3, KnownSplitPattern.llama3Source),
        (.qwen2, KnownSplitPattern.qwen2Source),
        (.qwen3, KnownSplitPattern.qwen3Source),
        (.o200k, KnownSplitPattern.o200kSource),
        (.falcon, KnownSplitPattern.falconSource),
    ]

    @Test(arguments: [0, 1, 2, 3, 4, 5])
    func randomTexts(patternIndex: Int) throws {
        let (pattern, source) = Self.patterns[patternIndex]
        let regex = try NSRegularExpression(pattern: source)
        var rng = SeededGenerator(seed: UInt64(1000 + patternIndex))
        var failures: [String] = []
        for _ in 0..<6000 {
            let text = Fuzz.text(&rng)
            var pieces: [Substring] = []
            pattern.split(Substring(text), into: &pieces)
            let ours = pieces.map(String.init)
            let reference = splitMatches(in: text, with: regex)
            if ours != reference {
                failures.append("\(text.debugDescription): scanner \(ours) regex \(reference)")
                if failures.count > 5 { break }
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }

    @Test("Hand-picked edge cases")
    func edgeCases() throws {
        let cases = [
            "", " ", "  ", "   ", "\n", " \n", "\n ", "  \n  ", "a\n\nb", "a \n b", "\t\t\ta",
            "'s'S'ſ", "it's", "IT'S", "don't", "'twas", "'", "''", "' s",
            "123", "1234", "12 345", " 1", " 12345", "١٢٣٤",
            "hello   world", "hello\u{A0}world", "x\u{2028}y",
            "...\n", "!!!\r\n\r\n", " ?\n", "a-b", "a — b",
            "e\u{0301}", "\u{0301}abc", "😀😀", " 😀", "😀 a",
            // o200k case splitting and mark placement
            "camelCase", "HTTPServer", "HTTPServerX", "ABC", "ABCd", "ǅ", "ǅǅ", "ǅa", "aǅ", "ʰABC", "ʰAʰB", "AʰB",
            "ABʰ", "日本ABC", "ABC日本",
            "A\u{0301}", "A\u{0301}B", "AB\u{0301}", "\u{0301}AB", "!\u{0301}", "!!\u{0301}", "!!\u{0301}!!",
            " \u{0301}ab", "\u{0301}", " \u{0301}",
            "don't", "DON'T", "he'sx", "I'LL", "'s", "a's's", "ABC's", "ABc'RE",
            "://", "-\n/-", "/\n\n/", " /", "a/b", "1234", "12345678", "١٢٣٤",
        ]
        for (pattern, source) in Self.patterns {
            let regex = try NSRegularExpression(pattern: source)
            for text in cases {
                var pieces: [Substring] = []
                pattern.split(Substring(text), into: &pieces)
                #expect(
                    pieces.map(String.init) == splitMatches(in: text, with: regex),
                    "\(pattern) \(text.debugDescription)")
            }
        }
    }
}

@Suite("Added-token splitter equivalence with regex splitting")
struct AddedTokenSplitterEquivalenceTests {
    struct Spec {
        let content: String
        let lstrip: Bool
        let rstrip: Bool
    }

    /// The upstream approach: one alternation regex with optional `\s*` around each token.
    static func referenceSplit(_ text: String, tokens: [Spec]) -> [String] {
        let sorted = tokens.sorted { $0.content.count > $1.content.count }
        let pattern = sorted.map {
            let token = NSRegularExpression.escapedPattern(for: $0.content)
            return "\($0.lstrip ? #"\s*"# : "")(\(token))\($0.rstrip ? #"\s*"# : "")"
        }.joined(separator: "|")
        let regex = try! NSRegularExpression(pattern: pattern)
        return text.split(by: regex)
    }

    static func flatten(_ sections: [AddedTokenSplitter.Section]) -> [String] {
        sections.map { section in
            switch section {
            case let .text(t): String(t)
            case let .token(t, _): String(t)
            }
        }
    }

    @Test func randomTexts() {
        let specs = [
            Spec(content: "<|endoftext|>", lstrip: false, rstrip: false),
            Spec(content: "<s>", lstrip: false, rstrip: false),
            Spec(content: "</s>", lstrip: false, rstrip: true),
            Spec(content: "[MASK]", lstrip: true, rstrip: true),
            Spec(content: "<|im_start|>", lstrip: false, rstrip: false),
            Spec(content: "<|im", lstrip: false, rstrip: false),
            Spec(content: "'s", lstrip: true, rstrip: false),
        ]
        let tokens = specs.enumerated().map { index, spec in
            AddedTokenSplitter.Token(
                content: spec.content, id: index, lstrip: spec.lstrip, rstrip: spec.rstrip,
                scalarCount: spec.content.unicodeScalars.count)
        }.sorted { $0.scalarCount > $1.scalarCount }
        let splitter = AddedTokenSplitter(tokens: tokens)!

        var rng = SeededGenerator(seed: 77)
        var failures: [String] = []
        for _ in 0..<3000 {
            var text = Fuzz.text(&rng, maxAtoms: 16)
            if rng.next() % 3 == 0 { text += " <|im_start|>user" }
            let ours = Self.flatten(splitter.split(text))
            let reference = Self.referenceSplit(text, tokens: specs)
            if ours != reference {
                failures.append("\(text.debugDescription): ours \(ours) reference \(reference)")
                if failures.count > 5 { break }
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }

    @Test func tokenIdsAndWhitespaceAbsorption() {
        let tokens = [
            AddedTokenSplitter.Token(content: "<|end|>", id: 7, lstrip: false, rstrip: true, scalarCount: 7),
            AddedTokenSplitter.Token(content: "<pad>", id: 3, lstrip: true, rstrip: false, scalarCount: 5),
        ]
        let splitter = AddedTokenSplitter(tokens: tokens)!
        let sections = splitter.split("a<|end|>   b  <pad>c")
        var summary: [String] = []
        for section in sections {
            switch section {
            case let .text(t): summary.append("T(\(t))")
            case let .token(t, id): summary.append("K(\(t),\(id))")
            }
        }
        #expect(summary == ["T(a)", "K(<|end|>,7)", "T(b)", "K(<pad>,3)", "T(c)"])
    }
}

@Suite("JSON parser equivalence with JSONSerialization")
struct JSONParserEquivalenceTests {
    @Test func literals() throws {
        let json = """
            {"s": "a\\"b\\\\c\\/d\\n\\t\\u00e9\\ud83d\\ude00", "i": -42, "big": 18446744073709551616, "f": 1.5e3,
             "neg": -0.25, "t": true, "f2": false, "n": null, "arr": [1, [2, 3], {"k": "v"}], "e": {}, "ea": []}
            """
        let config = try Config(jsonString: json)
        #expect(config.s.string() == "a\"b\\c/d\n\té😀")
        #expect(config.i.integer() == -42)
        #expect(config.big.double() == 18446744073709551616.0)
        #expect(config.f.double() == 1500)
        #expect(config.neg.double() == -0.25)
        #expect(config.t.boolean() == true)
        #expect(config.f2.boolean() == false)
        #expect(config.n.isNull())
        #expect(config.arr[1][0].integer() == 2)
        #expect(config.arr[2].k.string() == "v")
        #expect(config.e.dictionary()?.isEmpty == true)
        #expect(config.ea.array()?.isEmpty == true)
    }

    @Test func errors() {
        for bad in ["{", "[1,", "{\"a\" 1}", "tru", "\"unterminated", "{\"a\": 1} x", "[1 2]", "\"\\x\"", "-", "01a"] {
            #expect(throws: JSONConfigError.self, "\(bad)") { try Config(jsonString: bad) }
        }
    }

    @Test func binaryDistinctKeys() throws {
        let config = try Config(
            jsonString: "{\"vocab\": {\"\u{00E0}\": 1, \"a\u{0300}\": 2, \";\": 3, \"\u{037E}\": 4}}")
        let vocab = config.vocab.dictionary(or: [:])
        #expect(vocab.count == 4)
        #expect(vocab[BinaryDistinctString("\u{037E}")]?.integer() == 4)
        #expect(vocab[BinaryDistinctString(";")]?.integer() == 3)
    }

    @Test func matchesFoundationOnFixtures() throws {
        for name in ["tokenizer", "tokenizer_config", "gemma_encoded", "tokenizer_tests"] {
            let url = try #require(Bundle.module.url(forResource: name, withExtension: "json"))
            let data = try Data(contentsOf: url)
            let ours = try Config(jsonData: data)
            let object = try JSONSerialization.jsonObject(with: data)
            let reference = Config(any: object)
            #expect(ours == reference, "\(name)")
        }
    }
}

@Suite("BPE merge strategies agree")
struct BPEMergeEquivalenceTests {
    @Test func linearMatchesHeap() async throws {
        let tokenizer = try await HubFixtures.preTrainedTokenizer(for: "Qwen/Qwen3-0.6B")
        let model = try #require(tokenizer.model as? BPETokenizer)
        var rng = SeededGenerator(seed: 4242)
        var scratch = BPETokenizer.MergeScratch()
        for _ in 0..<500 {
            // Long words exercise both strategies with the same input.
            let text = Fuzz.text(&rng, maxAtoms: 60).replacingOccurrences(of: " ", with: "")
            var copy = Substring(text)
            copy.withUTF8 { bytes in
                var linear: [BPETokenizer.Symbol] = []
                model.byteLevelSymbols(bytes, into: &linear)
                var heap = linear
                model.mergeLinear(&linear, scratch: &scratch)
                model.mergeHeap(&heap, scratch: &scratch)
                #expect(linear.map(\.id) == heap.map(\.id), "\(text.debugDescription)")
                #expect(linear.map(\.end) == heap.map(\.end))
            }
        }
    }
}
