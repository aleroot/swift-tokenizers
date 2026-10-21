/// Direct UTF-8 ⇄ UTF-16 transcoding for the parser's buffer. The standard library's generic
/// `String(decoding:as: UTF16.self)` and `Array.append(contentsOf: string.utf16)` go through
/// generic per-unit iteration; these loops run over contiguous storage with an ASCII fast path.
enum UTF16Text {
    /// Appends the UTF-16 code units of `text`, which never outnumber its UTF-8 bytes.
    static func append(_ text: String, to units: inout [UInt16]) {
        var text = text
        text.withUTF8 { bytes in
            let start = units.count
            units.append(contentsOf: repeatElement(0, count: bytes.count))
            let end = units.withUnsafeMutableBufferPointer { output in
                var written = start
                var index = 0
                while index < bytes.count {
                    let lead = bytes[index]
                    if lead < 0x80 {
                        output[written] = UInt16(lead)
                        written += 1
                        index += 1
                        continue
                    }
                    // Valid UTF-8, since it comes from a `String`.
                    let length = lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4
                    var scalar = UInt32(lead) & (0x7F >> length)
                    for offset in 1..<length { scalar = scalar << 6 | UInt32(bytes[index + offset] & 0x3F) }
                    if scalar < 0x10000 {
                        output[written] = UInt16(scalar)
                        written += 1
                    } else {
                        scalar -= 0x10000
                        output[written] = 0xD800 | UInt16(scalar >> 10)
                        output[written + 1] = 0xDC00 | UInt16(scalar & 0x3FF)
                        written += 2
                    }
                    index += length
                }
                return written
            }
            units.removeLast(units.count - end)
        }
    }

    /// The text of `units`; an unpaired surrogate becomes U+FFFD.
    static func string(_ units: UnsafeBufferPointer<UInt16>) -> String {
        String(unsafeUninitializedCapacity: units.count * 3) { output in
            var written = 0
            var index = 0
            while index < units.count {
                var scalar = UInt32(units[index])
                index += 1
                if scalar < 0x80 {
                    output[written] = UInt8(scalar)
                    written += 1
                    continue
                }
                if UTF16.isLeadSurrogate(UInt16(scalar)) {
                    if index < units.count, UTF16.isTrailSurrogate(units[index]) {
                        scalar = 0x10000 + ((scalar - 0xD800) << 10) + (UInt32(units[index]) - 0xDC00)
                        index += 1
                    } else {
                        scalar = 0xFFFD
                    }
                } else if UTF16.isTrailSurrogate(UInt16(scalar)) {
                    scalar = 0xFFFD
                }
                if scalar < 0x800 {
                    output[written] = UInt8(0xC0 | scalar >> 6)
                    output[written + 1] = UInt8(0x80 | scalar & 0x3F)
                    written += 2
                } else if scalar < 0x10000 {
                    output[written] = UInt8(0xE0 | scalar >> 12)
                    output[written + 1] = UInt8(0x80 | scalar >> 6 & 0x3F)
                    output[written + 2] = UInt8(0x80 | scalar & 0x3F)
                    written += 3
                } else {
                    output[written] = UInt8(0xF0 | scalar >> 18)
                    output[written + 1] = UInt8(0x80 | scalar >> 12 & 0x3F)
                    output[written + 2] = UInt8(0x80 | scalar >> 6 & 0x3F)
                    output[written + 3] = UInt8(0x80 | scalar & 0x3F)
                    written += 4
                }
            }
            return written
        }
    }

    static func string(_ units: ArraySlice<UInt16>) -> String {
        units.withUnsafeBufferPointer(string)
    }
}
