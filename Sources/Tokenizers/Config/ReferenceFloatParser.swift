// Floating-point parsing that reproduces `serde_json`'s default (non-`float_roundtrip`)
// algorithm, which is what the Rust `tokenizers` crate uses to read `tokenizer.json`.
//
// That algorithm accumulates the mantissa digits into a `u64`, converts once to `f64`, then
// applies a single power-of-ten multiply or divide. Each step rounds, so the result can differ
// from the correctly rounded value by an ulp. Unigram scores parsed this way decide Viterbi
// tie-breaks; the reference and this library must therefore agree bit for bit on every score.

enum ReferenceFloatParser {
    /// `1e0 … 1e308`, correctly rounded, as the reference's `POW10` table.
    private static let pow10: [Double] = (0...308).map { Double("1e\($0)")! }

    /// Parses a JSON number already validated against the JSON grammar. Returns `nil` when
    /// the reference would reject the literal (out of range).
    static func parse(_ bytes: UnsafeBufferPointer<UInt8>) -> Double? {
        var i = 0
        let n = bytes.count
        var positive = true
        if i < n, bytes[i] == UInt8(ascii: "-") {
            positive = false
            i += 1
        }

        var significand: UInt64 = 0
        var exponent: Int32 = 0

        // Integer part. Once the mantissa would overflow, further digits only scale the exponent.
        var mantissaSaturated = false
        while i < n, isDigit(bytes[i]) {
            let digit = UInt64(bytes[i] &- 0x30)
            if !mantissaSaturated {
                let (scaled, o1) = significand.multipliedReportingOverflow(by: 10)
                let (sum, o2) = scaled.addingReportingOverflow(digit)
                if o1 || o2 {
                    mantissaSaturated = true
                    exponent += 1
                } else {
                    significand = sum
                }
            } else {
                exponent += 1
            }
            i += 1
        }

        // Fraction: from the first digit that would overflow the mantissa, the rest are ignored.
        if i < n, bytes[i] == UInt8(ascii: ".") {
            i += 1
            mantissaSaturated = false
            while i < n, isDigit(bytes[i]) {
                if !mantissaSaturated {
                    let digit = UInt64(bytes[i] &- 0x30)
                    let (scaled, o1) = significand.multipliedReportingOverflow(by: 10)
                    let (sum, o2) = scaled.addingReportingOverflow(digit)
                    if o1 || o2 {
                        mantissaSaturated = true
                    } else {
                        significand = sum
                        exponent -= 1
                    }
                }
                i += 1
            }
        }

        // Exponent.
        if i < n, bytes[i] == UInt8(ascii: "e") || bytes[i] == UInt8(ascii: "E") {
            i += 1
            var positiveExponent = true
            if i < n, bytes[i] == UInt8(ascii: "+") {
                i += 1
            } else if i < n, bytes[i] == UInt8(ascii: "-") {
                positiveExponent = false
                i += 1
            }
            var exp: Int32 = 0
            while i < n, isDigit(bytes[i]) {
                let digit = Int32(bytes[i] &- 0x30)
                let (scaled, o1) = exp.multipliedReportingOverflow(by: 10)
                let (sum, o2) = scaled.addingReportingOverflow(digit)
                if o1 || o2 {
                    // Reference: error for a non-zero mantissa with a huge positive exponent,
                    // signed zero otherwise.
                    if significand != 0, positiveExponent { return nil }
                    return positive ? 0.0 : -0.0
                }
                exp = sum
                i += 1
            }
            exponent = positiveExponent ? exponent.addingSaturating(exp) : exponent.subtractingSaturating(exp)
        }

        return fromParts(positive: positive, significand: significand, exponent: exponent)
    }

    private static func fromParts(positive: Bool, significand: UInt64, exponent: Int32) -> Double? {
        var f = Double(significand)
        var exponent = exponent
        while true {
            let magnitude = Int(exponent.magnitude)
            if magnitude < pow10.count {
                let pow = pow10[magnitude]
                if exponent >= 0 {
                    f *= pow
                    if f.isInfinite { return nil }
                } else {
                    f /= pow
                }
                break
            }
            if f == 0 { break }
            if exponent >= 0 { return nil }
            f /= 1e308
            exponent += 308
        }
        return positive ? f : -f
    }

    @inline(__always)
    private static func isDigit(_ byte: UInt8) -> Bool {
        byte >= 0x30 && byte <= 0x39
    }
}

extension Int32 {
    fileprivate func addingSaturating(_ other: Int32) -> Int32 {
        let (sum, overflow) = addingReportingOverflow(other)
        return overflow ? (other > 0 ? .max : .min) : sum
    }

    fileprivate func subtractingSaturating(_ other: Int32) -> Int32 {
        let (difference, overflow) = subtractingReportingOverflow(other)
        return overflow ? (other > 0 ? .min : .max) : difference
    }
}
