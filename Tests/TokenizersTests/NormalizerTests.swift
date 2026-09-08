import Foundation
import Testing

@testable import Tokenizers

@Suite("Normalizer Tests")
struct NormalizerTests {
    @Test("Strip preserves combining marks attached to boundary whitespace")
    func stripCombiningMarks() {
        // tokenizers 0.23.2 normalizers.Strip: whitespace is scalar-based, not grapheme-based.
        for (input, both, left, right) in [
            (" \u{3099} ", "\u{3099}", "\u{3099} ", " \u{3099}"),
            (" \u{301} ", "\u{301}", "\u{301} ", " \u{301}"),
            ("\u{a0}\u{301}\u{a0}", "\u{301}", "\u{301}\u{a0}", "\u{a0}\u{301}"),
            ("a \u{3099} ", "a \u{3099}", "a \u{3099} ", "a \u{3099}"),
        ] {
            for (stripLeft, stripRight, expected) in [(true, true, both), (true, false, left), (false, true, right)] {
                let normalizer = StripNormalizer(config: ["strip_left": Config(stripLeft), "strip_right": Config(stripRight)])
                #expect(normalizer.normalize(text: input).utf8.elementsEqual(expected.utf8))
            }
        }
    }

    @Test("Lowercase normalizer functionality")
    func lowercaseNormalizer() throws {
        let testCases: [(String, String)] = [
            ("Café", "café"),
            ("François", "françois"),
            ("Ωmega", "ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "häagen-dazs"),
            ("你好!", "你好!"),
            ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"),
            ("\u{00C5}", "\u{00E5}"),
        ]

        for (arg, expect) in testCases {
            let config = Config([String: Config]())
            let normalizer = LowercaseNormalizer(config: config)
            #expect(normalizer.normalize(text: arg) == expect)
        }

        let config = Config(["type": NormalizerType.Lowercase.rawValue])
        #expect(try NormalizerFactory.fromConfig(config: config) as? LowercaseNormalizer != nil)
    }

    @Test("NFD normalizer functionality")
    func nfdNormalizer() throws {
        let testCases: [(String, String)] = [
            ("caf\u{65}\u{301}", "cafe\u{301}"),
            ("François", "François"),
            ("Ωmega", "Ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "Häagen-Dazs"),
            ("你好!", "你好!"),
            ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"),
            ("\u{00C5}", "\u{0041}\u{030A}"),
        ]

        for (arg, expect) in testCases {
            let config = Config([String: Config]())
            let normalizer = NFDNormalizer(config: config)
            #expect(normalizer.normalize(text: arg) == expect)
        }

        let config = Config(["type": NormalizerType.NFD.rawValue])
        #expect(try NormalizerFactory.fromConfig(config: config) as? NFDNormalizer != nil)
    }

    @Test("NFC normalizer functionality")
    func nfcNormalizer() throws {
        let testCases: [(String, String)] = [
            ("café", "café"),
            ("François", "François"),
            ("Ωmega", "Ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "Häagen-Dazs"),
            ("你好!", "你好!"),
            ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"),
            ("\u{00C5}", "\u{00C5}"),
        ]

        for (arg, expect) in testCases {
            let config = Config([String: Config]())
            let normalizer = NFCNormalizer(config: config)
            #expect(normalizer.normalize(text: arg) == expect)
        }

        let config = Config(["type": NormalizerType.NFC.rawValue])
        #expect(try NormalizerFactory.fromConfig(config: config) as? NFCNormalizer != nil)
    }

    @Test("NFKD normalizer functionality")
    func nfkdNormalizer() throws {
        let testCases: [(String, String)] = [
            ("café", "cafe\u{301}"),
            ("François", "François"),
            ("Ωmega", "Ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "Häagen-Dazs"),
            ("你好!", "你好!"),
            ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "ABC⓵⓶⓷{,},i9,i9,アパート,1⁄4"),
            ("\u{00C5}", "Å"),
        ]

        for (arg, expect) in testCases {
            let config = Config([String: Config]())
            let normalizer = NFKDNormalizer(config: config)
            #expect(normalizer.normalize(text: arg) == expect)
        }

        let config = Config(["type": NormalizerType.NFKD.rawValue])
        #expect(try NormalizerFactory.fromConfig(config: config) as? NFKDNormalizer != nil)
    }

    @Test("NFKC normalizer functionality")
    func nfkcNormalizer() throws {
        let testCases: [(String, String)] = [
            ("café", "café"),
            ("François", "François"),
            ("Ωmega", "Ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "Häagen-Dazs"),
            ("你好!", "你好!"),
            ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "ABC⓵⓶⓷{,},i9,i9,アパート,1⁄4"),
            ("\u{00C5}", "\u{00C5}"),
        ]

        for (arg, expect) in testCases {
            let config = Config([String: Config]())
            let normalizer = NFKCNormalizer(config: config)
            #expect(normalizer.normalize(text: arg) == expect)
        }

        let config = Config(["type": NormalizerType.NFKC.rawValue])
        #expect(try NormalizerFactory.fromConfig(config: config) as? NFKCNormalizer != nil)
    }

    @Test("Strip accents functionality")
    func stripAccents() throws {
        let testCases = [
            ("département", "departement")
        ]

        // TODO: test combinations with/without lowercase
        let config = Config(["stripAccents": true])
        let normalizer = BertNormalizer(config: config)
        for (arg, expect) in testCases {
            #expect(normalizer.normalize(text: arg) == expect)
        }
    }

    @Test("Bert normalizer functionality")
    func bertNormalizer() throws {
        let testCases: [(String, String)] = [
            ("Café", "café"),
            ("François", "françois"),
            ("Ωmega", "ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen\tDazs", "häagen dazs"),
            ("你好!", " 你  好 !"),
            ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"),
            ("\u{00C5}", "\u{00E5}"),
        ]

        for (arg, expect) in testCases {
            let config = Config(["stripAccents": false])
            let normalizer = BertNormalizer(config: config)
            #expect(normalizer.normalize(text: arg) == expect)
        }

        let config = Config(["type": NormalizerType.Bert.rawValue])
        #expect(try NormalizerFactory.fromConfig(config: config) as? BertNormalizer != nil)
    }

    @Test("Bert normalizer defaults functionality")
    func bertNormalizerDefaults() throws {
        // Python verification: t._tokenizer.normalizer.normalize_str("Café")
        let testCases: [(String, String)] = [
            ("Café", "cafe"),
            ("François", "francois"),
            ("Ωmega", "ωmega"),
            ("über", "uber"),
            ("háček", "hacek"),
            ("Häagen\tDazs", "haagen dazs"),
            ("你好!", " 你  好 !"),
            ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"),
            ("Å", "a"),
        ]

        for (arg, expect) in testCases {
            let config = Config([String: Config]())
            let normalizer = BertNormalizer(config: config)
            #expect(normalizer.normalize(text: arg) == expect)
        }

        let config = Config(["type": NormalizerType.Bert.rawValue])
        #expect(try NormalizerFactory.fromConfig(config: config) as? BertNormalizer != nil)
    }

    @Test("Precompiled normalization requires the model's map")
    func precompiledNormalizer() throws {
        #expect(throws: TokenizerError.self) {
            try NormalizerFactory.fromConfig(config: ["type": "Precompiled"])
        }
        // Real map behavior is covered byte-for-byte by UpstreamComponentTests.
    }

    @Test("Strip accents normalizer functionality")
    func stripAccentsNormalizer() throws {
        let testCases: [(String, String)] = [
            ("café", "café"),
            ("François", "François"),
            ("Ωmega", "Ωmega"),
            ("über", "über"),
            ("háček", "háček"),
            ("Häagen-Dazs", "Häagen-Dazs"),
            ("你好!", "你好!"),
            ("𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼", "𝔄𝔅ℭ⓵⓶⓷︷,︸,i⁹,i₉,㌀,¼"),
            ("\u{00C5}", "\u{00C5}"),
        ]

        for (arg, expect) in testCases {
            let config = Config([String: Config]())
            let normalizer = StripAccentsNormalizer(config: config)
            #expect(normalizer.normalize(text: arg).utf8.elementsEqual(expect.utf8))
        }

        let config = Config(["type": NormalizerType.StripAccents.rawValue])
        #expect(try NormalizerFactory.fromConfig(config: config) as? StripAccentsNormalizer != nil)
    }

    @Test("Strip normalizer functionality")
    func stripNormalizer() throws {
        let testCases: [(String, String, Bool, Bool)] = [
            ("  hello  ", "hello", true, true),
            ("  hello  ", "hello  ", true, false),
            ("  hello  ", "  hello", false, true),
            ("  hello  ", "  hello  ", false, false),
            ("\t\nHello\t\n", "Hello", true, true),
            ("   ", "", true, true),
            ("", "", true, true),
        ]

        for (input, expected, leftStrip, rightStrip) in testCases {
            let config = Config([
                "type": NormalizerType.Strip.rawValue,
                "stripLeft": leftStrip,
                "stripRight": rightStrip,
            ])
            let normalizer = StripNormalizer(config: config)
            #expect(normalizer.normalize(text: input) == expected)
        }

        let config = Config(["type": NormalizerType.Strip.rawValue])
        #expect(try NormalizerFactory.fromConfig(config: config) as? StripNormalizer != nil)
    }
}
