import Foundation
import Testing

@testable import Tokenizers

/// The Rust `tokenizers` crate reads `tokenizer.json` with `serde_json`'s default float
/// algorithm, which is not correctly rounded. Unigram scores must reproduce its bit patterns
/// exactly, otherwise Viterbi ties resolve differently from the reference (T5 `"-------"`).
@Suite("Reference float parsing")
struct ReferenceFloatParserTests {
    /// Bit patterns observed by round-tripping the literals through `tokenizers` 0.22.
    static let reference: [(literal: String, bits: UInt64)] = [
        ("-12.130167007446289", 0xC028_42A5_3FFF_FFFF),  // correctly rounded: 0xC02842A540000000
        ("-2.0122928619384766", 0xC000_192D_0000_0001),  // correctly rounded: 0xC000192D00000000
        ("-5.129043102264404", 0xC014_8423_E000_0000),
        ("-9.931743621826172", 0xC023_DD0D_8000_0000),
        ("0.0", 0x0000_0000_0000_0000),
        ("-0.0", 0x8000_0000_0000_0000),
        ("1.5", 0x3FF8_0000_0000_0000),
        ("0.1", 0x3FB9_9999_9999_999A),
        ("123456789.123456789", 0x419D_6F34_547E_6B75),
        ("18446744073709551615.5", 0x43F0_0000_0000_0000),
        ("184467440737095516150.25", 0x4424_0000_0000_0000),
        ("1844674407370955161.9", 0x43B9_9999_9999_999A),
        ("18446744073709551616.5", 0x43F0_0000_0000_0000),
        ("12345678901234567890123456789", 0x45C3_F20D_9923_5F65),
        ("1e22", 0x4480_F0CF_064D_D592),
        ("1e23", 0x44B5_2D02_C7E1_4AF6),
        ("3.14159e-3", 0x3F69_BC64_49D6_8DC1),
        ("1.7976931348623157e308", 0x7FEF_FFFF_FFFF_FFFF),
        ("5e-324", 0x0000_0000_0000_0001),
        ("1e-400", 0x0000_0000_0000_0000),
        ("-0.000123456789012345678901234567890", 0xBF20_2E85_BE18_0B74),
    ]

    @Test("Floats match the reference bit for bit")
    func referenceBits() throws {
        for (literal, bits) in Self.reference {
            let parsed: Double? = try Config(jsonString: "[\(literal)]")[0].double()
            let pattern: UInt64? = parsed?.bitPattern
            let description = parsed.map { "\($0)" } ?? "nil"
            #expect(pattern == bits, "\(literal) parsed as \(description)")
        }
    }

    @Test("Out-of-range literals are rejected like the reference")
    func outOfRange() {
        for literal in ["2e308", "1e99999999999"] {
            #expect(throws: JSONConfigError.self) { try Config(jsonString: "[\(literal)]") }
        }
    }

    @Test("Unigram scores flow through the reference algorithm in packed and generic tables")
    func unigramScores() throws {
        let json = """
            {"version":"1.0","model":{"type":"Unigram","unk_id":0,
             "vocab":[["<unk>",0.0],["a",-12.130167007446289],["b",-2.0122928619384766]]}}
            """
        let packed = try Config(tokenizerJSON: Data(json.utf8))
        let generic = try Config(jsonString: json)
        for config in [packed, generic] {
            let model = try UnigramTokenizer(tokenizerConfig: Config(), tokenizerData: config, addedTokens: [:])
            #expect(model.scores[1].bitPattern == 0xC028_42A5_3FFF_FFFF)
            #expect(model.scores[2].bitPattern == 0xC000_192D_0000_0001)
        }
    }
}
