// The DeepSeek pre-tokenizer is a sequence of three `Split` regexes. They must be recognised
// verbatim from the model's own `tokenizer.json`, or encoding silently falls back to
// `NSRegularExpression` and runs an order of magnitude slower.

import Foundation
import Testing

@testable import Tokenizers

@Suite("DeepSeek pre-tokenizer")
struct DeepSeekScannerTests {
    static func components() throws -> Config {
        let url = try #require(Bundle.module.url(forResource: "deepseek_v41_components", withExtension: "json"))
        return try Config(jsonData: Data(contentsOf: url))
    }

    @Test("Every split of the shipped sequence is on the scanner fast path")
    func fastPath() throws {
        let splits = try #require(Self.components().pre_tokenizer.pretokenizers.array())
        let expected: [KnownSplitPattern] = [.deepseekNumbers, .deepseekCJK, .deepseek]
        var known: [KnownSplitPattern] = []
        for config in splits where config.type.string() == "Split" {
            let split = try SplitPreTokenizer(config: config)
            known.append(try #require(split.known, "\(config.pattern.Regex.string() ?? "?") fell back to regex"))
        }
        #expect(known == expected)
    }

    @Test("The sequence pre-tokenizes DeepSeek's own prompt syntax")
    func sequence() throws {
        let pre = try #require(try PreTokenizerFactory.fromConfig(config: Self.components().pre_tokenizer))
        // Byte-level output, so the pieces are the byte-alphabet spelling of the UTF-8 bytes.
        // U+FF5C is `\p{S}`, which glues it to the surrounding ASCII punctuation.
        #expect(pre.preTokenize(text: "<｜User｜>hi") == ["<ï½ľ", "User", "ï½ľ>", "hi"])
        #expect(pre.preTokenize(text: "1234567") == ["123", "456", "7"])
        // The Han run is isolated from the Latin one by the second split.
        #expect(pre.preTokenize(text: "日本語abc") == ["æĹ¥æľ¬èªŀ", "abc"])
        // `[!"#$…~][A-Za-z]+`: punctuation glued to the ASCII word that follows it.
        #expect(pre.preTokenize(text: "get_json_schema(x)") == ["get", "_json", "_schema", "(x", ")"])
    }

    /// The chunked byte scans must agree with the scalar definition at every alignment.
    @Test("Scanners agree with the reference regex at every chunk alignment")
    func alignment() throws {
        let base = "漢字 kana ひらがな カタカナ mixed 1234 ＿ 東京tokyo, 日本．テスト\t和\n字"
        var texts: [String] = []
        for shift in 0..<20 {
            texts.append(String(repeating: "a", count: shift) + base)
            texts.append(String(repeating: "漢", count: shift) + base)
            texts.append(String(repeating: "1", count: shift) + base)
            texts.append(String(repeating: " ", count: shift) + base)
        }
        for (pattern, source) in ScannerEquivalenceTests.isolatingPatterns {
            let regex = try NSRegularExpression(pattern: source)
            for text in texts {
                var pieces: [Substring] = []
                pattern.split(Substring(text), into: &pieces)
                let ns = text as NSString
                var reference: [String] = []
                var cursor = 0
                regex.enumerateMatches(in: text, range: NSRange(location: 0, length: ns.length)) { match, _, _ in
                    guard let match else { return }
                    if match.range.location > cursor {
                        reference.append(
                            ns.substring(with: NSRange(location: cursor, length: match.range.location - cursor)))
                    }
                    reference.append(ns.substring(with: match.range))
                    cursor = match.range.location + match.range.length
                }
                if cursor < ns.length { reference.append(ns.substring(from: cursor)) }
                #expect(pieces.map(String.init) == reference, "\(pattern) \(text.debugDescription)")
            }
        }
    }
}
