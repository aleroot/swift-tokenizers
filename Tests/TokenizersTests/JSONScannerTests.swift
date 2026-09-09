import Foundation
import Testing

@testable import Tokenizers

@Suite("JSON byte scanning")
struct JSONScannerTests {
    @Test("Indentation skips only JSON whitespace, at every word boundary")
    func whitespace() throws {
        for length in 0..<80 {
            let spaces = String(repeating: " ", count: length)
            for separator in [spaces, "\n" + spaces, "\t" + spaces + "\r\n"] {
                let json = "[" + separator + "\"é\"," + separator + "42" + separator + "]" + spaces
                let result = try Config(jsonString: json)
                #expect(result[0].string() == "é")
                #expect(result[1].integer() == 42)
            }
            for invalid in ["\u{0000}", "\u{000B}", "\u{000C}", "\u{00A0}"] {
                #expect(throws: JSONConfigError.self) {
                    try Config(jsonString: "[" + spaces + invalid + "0]")
                }
            }
        }
    }

    @Test("String boundaries and ASCII status match an independent scalar scan")
    func boundaries() {
        func check(_ buffer: [UInt8], from start: Int) {
            buffer.withUnsafeBufferPointer { bytes in
                let end = bytes[start...].firstIndex { $0 == 0x22 || $0 == 0x5C || $0 < 0x20 } ?? bytes.count
                let ascii = bytes[start..<end].allSatisfy { $0 < 0x80 }
                let actual = ByteKernels.jsonStringEnd(bytes, from: start)
                #expect(actual.end == end)
                #expect(actual.ascii == ascii)
            }
        }
        for buffer in ByteKernelsTests.buffers() {
            for start in 0...buffer.count { check(buffer, from: start) }
        }
        // Every byte in every position, including bytes after a terminator. Adjacent
        // values exercise carries/borrows in word-at-a-time implementations.
        for offset in 0..<16 {
            for lane in 0..<48 {
                for value in UInt16(0)...255 {
                    var buffer = [UInt8](repeating: 0x61, count: offset + 64)
                    buffer[offset + lane] = UInt8(value)
                    buffer[offset + lane + 1] = UInt8(truncatingIfNeeded: value + 1)
                    buffer[offset + lane + 2] = 0x22
                    buffer[offset + lane + 3] = 0xFF
                    check(buffer, from: offset)
                }
            }
        }
    }

    @Test("Packed and generic JSON preserve exact string bytes at block boundaries")
    func strings() throws {
        for padding in 0..<48 {
            for value in [
                "", "é", "日本語", "😀", "\u{FEFF}", "\u{007F}", "a\u{0300}",
                "\"\\/\n\r\t\u{0000}\u{001F}", String(repeating: "abcé😀\"\\", count: 32),
            ] {
                let text = String(repeating: "a", count: padding) + value
                let encoded = try JSONEncoder().encode(text)
                var json = Data("{\"model\":{\"vocab\":{".utf8)
                json.append(encoded)
                json.append(contentsOf: ":7}}}".utf8)
                let generic = try Config(jsonData: json)
                let packed = try Config(tokenizerJSON: json)
                let key = try #require(generic.model.vocab.dictionary()?.keys.first)
                #expect(key.description.utf8.elementsEqual(text.utf8))
                let table = try #require(packed.model.vocab.asPackedStringMap())
                #expect(table.utf8.elementsEqual(text.utf8))
                #expect(table.ids == [7])
            }
        }
    }

    @Test("Malformed UTF-8 and unescaped controls are rejected in both parse modes")
    func malformedStrings() {
        let invalid: [[UInt8]] =
            [
                [0x80], [0xBF], [0xC0, 0x80], [0xC1, 0xBF], [0xC2], [0xC2, 0x20],
                [0xE0, 0x9F, 0xBF], [0xED, 0xA0, 0x80], [0xE1, 0x80],
                [0xF0, 0x8F, 0xBF, 0xBF], [0xF4, 0x90, 0x80, 0x80], [0xF5, 0x80, 0x80, 0x80],
                [0xF0, 0x90, 0x80], [0xFE], [0xFF],
            ] + (UInt8(0)..<0x20).map { [$0] }
        for padding in 0..<48 {
            for sequence in invalid {
                for escapedPrefix in [false, true] {
                    var data = Data("{\"model\":{\"vocab\":{\"".utf8)
                    data.append(contentsOf: repeatElement(UInt8(0x61), count: padding))
                    if escapedPrefix { data.append(contentsOf: "\\n".utf8) }
                    data.append(contentsOf: sequence)
                    data.append(contentsOf: "\":0}}}".utf8)
                    #expect(throws: JSONConfigError.self) { try Config(jsonData: data) }
                    #expect(throws: JSONConfigError.self) { try Config(tokenizerJSON: data) }
                }
            }
        }
    }
}
