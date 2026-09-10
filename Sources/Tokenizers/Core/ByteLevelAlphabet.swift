// The GPT-2 "bytes to unicode" mapping. Byte-level BPE tokenizers represent every one
// of the 256 byte values by a printable Unicode scalar so vocabularies stay valid text:
// printable Latin-1 bytes map to themselves; the remaining 68 bytes are mapped, in
// order, onto U+0100...

import Foundation

/// Compact, table-driven view of the byte-level alphabet used on hot paths.
enum ByteLevelAlphabet {
    /// `scalar(for: byte)` — the Unicode scalar value representing each byte.
    static let byteToScalar: [UInt32] = {
        var table = [UInt32](repeating: 0, count: 256)
        var n: UInt32 = 0
        for b in 0..<256 {
            let isPrintable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)
            if isPrintable {
                table[b] = UInt32(b)
            } else {
                table[b] = 256 + n
                n += 1
            }
        }
        return table
    }()

    /// Highest scalar value used by the alphabet (U+0143).
    static let maxScalar: UInt32 = 0x143

    /// Reverse lookup: scalar value → byte, or `-1` when the scalar is not part of the alphabet.
    /// Indexed directly by scalar value; the alphabet is bounded by ``maxScalar``.
    static let scalarToByte: [Int16] = {
        var table = [Int16](repeating: -1, count: Int(maxScalar) + 1)
        for (b, s) in byteToScalar.enumerated() {
            table[Int(s)] = Int16(b)
        }
        return table
    }()

    /// Each alphabet scalar as UTF-8 (1 or 2 bytes), packed for fast appends.
    static let byteToUTF8: [(UInt8, UInt8, UInt8)] = byteToScalar.map { scalar in
        if scalar < 0x80 {
            return (UInt8(scalar), 0, 1)
        }
        return (UInt8(0xC0 | (scalar >> 6)), UInt8(0x80 | (scalar & 0x3F)), 2)
    }

    /// Byte value for an alphabet scalar, or `nil`.
    @inline(__always)
    static func byte(for scalar: Unicode.Scalar) -> UInt8? {
        let v = scalar.value
        guard v <= maxScalar else { return nil }
        let b = scalarToByte[Int(v)]
        return b < 0 ? nil : UInt8(b)
    }

    /// Encodes raw bytes into their alphabet string representation (e.g. `" hi"` → `"Ġhi"`).
    static func encode(_ bytes: some Sequence<UInt8>) -> String {
        var scalars = String.UnicodeScalarView()
        for b in bytes {
            scalars.append(Unicode.Scalar(byteToScalar[Int(b)])!)
        }
        return String(scalars)
    }

    /// Appends the alphabet encoding of `bytes` as UTF-8 into `output`.
    @inline(__always)
    static func appendEncoded(_ bytes: UnsafeBufferPointer<UInt8>, to output: inout [UInt8]) {
        for b in bytes {
            let (b0, b1, n) = byteToUTF8[Int(b)]
            output.append(b0)
            if n == 2 { output.append(b1) }
        }
    }

    /// Decodes an alphabet token. If any scalar is outside the alphabet, the entire token
    /// passes through as UTF-8, matching the reference decoder's per-token fallback.
    static func decode(_ string: String, into output: inout [UInt8]) {
        let start = output.count
        for scalar in string.unicodeScalars {
            if let b = byte(for: scalar) {
                output.append(b)
            } else {
                output.removeSubrange(start...)
                output.append(contentsOf: string.utf8)
                return
            }
        }
    }

    /// Returns `true` if every scalar of `string` belongs to the alphabet.
    static func isFullyDecodable(_ string: String) -> Bool {
        for scalar in string.unicodeScalars where byte(for: scalar) == nil {
            return false
        }
        return true
    }
}

// MARK: - Legacy dictionary views (kept for source compatibility)

/// Byte → alphabet character mapping as a dictionary.
let byteEncoder: [UTF8.CodeUnit: String] = {
    var dict = [UTF8.CodeUnit: String](minimumCapacity: 256)
    for b in 0..<256 {
        dict[UInt8(b)] = String(Unicode.Scalar(ByteLevelAlphabet.byteToScalar[b])!)
    }
    return dict
}()

/// Alphabet character → byte mapping as a dictionary.
let byteDecoder: [String: UTF8.CodeUnit] = {
    var dict = [String: UTF8.CodeUnit](minimumCapacity: 256)
    for (b, s) in byteEncoder { dict[s] = b }
    return dict
}()

/// Dense byte → alphabet character table.
let byteEncoderTable: [String] = (0..<256).map { String(Unicode.Scalar(ByteLevelAlphabet.byteToScalar[$0])!) }
