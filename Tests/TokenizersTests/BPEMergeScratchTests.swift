import Testing

@testable import Tokenizers

@Suite("BPE scratch reuse")
struct BPEMergeScratchTests {
    @Test("Reused heap matches linear merges across sizes, ties, and stale candidates")
    func reuse() throws {
        // Overlapping products repeatedly invalidate queued candidates. Repeated letters
        // also put many equal-rank candidates in the heap, which must merge leftmost first.
        let model = try BPETokenizer(
            tokenizerConfig: Config(),
            tokenizerData: [
                "model": [
                    "type": "BPE",
                    "vocab": ["a": 0, "b": 1, "aa": 2, "ab": 3, "ba": 4, "bb": 5, "aaa": 6, "aaaa": 7],
                    "merges": [["a", "a"], ["a", "b"], ["b", "a"], ["b", "b"], ["aa", "a"], ["aa", "aa"]],
                ]
            ], addedTokens: [:])
        var scratch = BPETokenizer.MergeScratch()
        var rng = SeededGenerator(seed: 0xBFE2026)
        let sizes = [0, 1, 2, 95, 96, 97, 128, 511, 2048, 97, 96, 1, 0, 4096, 129, 2]
        for pass in 0..<4 {
            for size in sizes {
                let input = (0..<size).map { i in
                    BPETokenizer.Symbol(id: pass == 0 ? 0 : Int32(rng.next() % 2), start: Int32(i), end: Int32(i + 1))
                }
                var expected = input
                var referenceScratch = BPETokenizer.MergeScratch()
                if size >= 2 { model.mergeLinear(&expected, scratch: &referenceScratch) }
                var actual = input
                model.mergeHeap(&actual, scratch: &scratch)
                #expect(actual.map(\.id) == expected.map(\.id), "pass \(pass), size \(size)")
                #expect(actual.map(\.start) == expected.map(\.start))
                #expect(actual.map(\.end) == expected.map(\.end))
                // Exercise dispatch on both sides of the 96-symbol cutoff using the same
                // scratch after the direct heap call has drained its stale entries.
                actual = input
                model.merge(&actual, scratch: &scratch)
                #expect(actual.map(\.id) == expected.map(\.id))
                #expect(actual.map(\.end) == expected.map(\.end))
            }
        }
    }
}
