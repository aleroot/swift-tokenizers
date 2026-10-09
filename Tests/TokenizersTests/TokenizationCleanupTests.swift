import Testing

@testable import Tokenizers

@Suite("Fused tokenization cleanup parity")
struct TokenizationCleanupTests {
    @Test("In-place cleanup preserves the ordered replacement contract")
    func differential() {
        let fragments = [
            "", " ", " .", " ?", " !", " ,", " ' ", " n't", " 'm", " 's", " 've", " 're",
            " do not", " do no", " '", " n'", "é", "e\u{301}", "😀", "\u{0}",
        ]
        var cases: [[UInt8]] = []
        for a in fragments {
            for b in fragments {
                for c in fragments { cases.append(Array((a + b + c).utf8)) }
            }
        }
        var rng: UInt64 = 0x243F_6A88_85A3_08D3
        let alphabet = Array(" .?!, 'nmSsverdot".utf8)
        func next() -> UInt64 {
            rng ^= rng << 13
            rng ^= rng >> 7
            rng ^= rng << 17
            return rng
        }
        for _ in 0..<20_000 {
            let length = Int(next() % 128)
            cases.append((0..<length).map { _ in alphabet[Int(next() % UInt64(alphabet.count))] })
        }
        for bytes in cases {
            for wordPiece in [false, true] {
                let expected = sequential(bytes, wordPiece: wordPiece)
                let actual = TokenizationCleanup.cleanUp(bytes, wordPiece: wordPiece)
                #expect(actual == expected, "wordPiece=\(wordPiece), input=\(String(decoding: bytes, as: UTF8.self))")
                #expect(
                    Array(
                        TokenizationCleanup.cleanUp(String(decoding: bytes, as: UTF8.self), wordPiece: wordPiece).utf8)
                        == expected)
                if actual != expected { return }
            }
        }
    }

    private func sequential(_ bytes: [UInt8], wordPiece: Bool) -> [UInt8] {
        var rules = TokenizationCleanup.replacements
        if wordPiece { rules.insert((Array(" do not".utf8), Array(" don't".utf8)), at: 7) }
        var current = bytes
        for (pattern, replacement) in rules {
            var output: [UInt8] = []
            var cursor = 0
            while cursor < current.count {
                if cursor + pattern.count <= current.count,
                    current[cursor..<cursor + pattern.count].elementsEqual(pattern)
                {
                    output.append(contentsOf: replacement)
                    cursor += pattern.count
                } else {
                    output.append(current[cursor])
                    cursor += 1
                }
            }
            current = output
        }
        return current
    }
}
