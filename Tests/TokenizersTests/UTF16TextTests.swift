import Testing

@testable import Tokenizers

@Suite("UTF-16 transcoding of the response buffer")
struct UTF16TextTests {
    static let samples = [
        "", "a", "Hello, world", "\u{7F}\u{80}", "é", "\u{7FF}\u{800}", "€", "\u{FFFF}", "\u{10000}", "\u{10FFFF}",
        "😀", "👩‍👩‍👧", "🇮🇹", "e\u{301}", "中文テキスト", "ไทย", "<|\"|>", "\u{FEFF}", "\u{E000}\u{F8FF}",
        "mixed ascii, 2-byte é, 3-byte 中, 4-byte 𝄞 and again ascii",
    ]

    @Test("Matches the standard library", arguments: samples)
    func matchesStandardLibrary(_ text: String) {
        var units: [UInt16] = [0x41]
        UTF16Text.append(text, to: &units)
        #expect(units == [0x41] + Array(text.utf16))
        #expect(UTF16Text.string(units[1...]) == text)
        #expect(UTF16Text.string(units[...]) == "A" + text)
    }

    @Test("Every scalar round-trips")
    func everyScalar() {
        let scalars = (0...0x10FFFF).lazy.compactMap(Unicode.Scalar.init)
        var text = ""
        text.unicodeScalars.append(contentsOf: scalars)
        var units: [UInt16] = []
        UTF16Text.append(text, to: &units)
        #expect(units == Array(text.utf16))
        #expect(UTF16Text.string(units[...]) == text)
    }

    @Test("Unpaired surrogates decode to U+FFFD like the standard library")
    func unpairedSurrogates() {
        let cases: [[UInt16]] = [
            [0xD800], [0xDC00], [0x41, 0xD83D], [0xD83D, 0x41], [0xDE00, 0xD83D],
            [0xD83D, 0xD83D, 0xDE00], [0xDBFF, 0xDFFF, 0xDFFF], [0xDFFF, 0xDBFF],
        ]
        for units in cases {
            #expect(UTF16Text.string(units[...]) == String(decoding: units, as: UTF16.self), "\(units)")
        }
    }

    @Test("A slice that splits a surrogate pair decodes each half as U+FFFD")
    func splitPair() {
        let units = Array("a😀b".utf16)
        #expect(UTF16Text.string(units[..<2]) == "a\u{FFFD}")
        #expect(UTF16Text.string(units[2...]) == "\u{FFFD}b")
    }
}
