import Foundation

/// Unicode rules of tokenizers 0.23.2: unicode-normalization-alignments 0.1.12
/// uses Unicode 9; unicode_categories 0.1.1 uses Unicode 8. Case mapping and regex
/// properties remain separate. Character age avoids shipping another Unicode database.
@usableFromInline
enum TokenizerUnicode {
    @usableFromInline
    @inline(__always)
    static func assigned(_ scalar: Unicode.Scalar, through version: Int) -> Bool {
        // This contiguous Latin/combining-mark repertoire predates Unicode 8.
        scalar.value < 0x378 || (scalar.properties.age.map { $0.major <= version } ?? false)
    }

    static func combiningClass(_ scalar: Unicode.Scalar) -> UInt8 {
        assigned(scalar, through: 9) ? scalar.properties.canonicalCombiningClass.rawValue : 0
    }

    /// Normalization mappings and combining classes of existing characters are stable.
    /// Later characters act as inert starters, separating independently normalized runs.
    static func normalize(_ text: String, transform: StringTransform) -> String {
        let scalars = text.unicodeScalars
        var start = scalars.startIndex
        var output = ""
        for index in scalars.indices where normalizationBarrier(scalars[index]) {
            if start < index {
                output += String(scalars[start..<index]).applyingTransform(transform, reverse: false)!
            }
            output.unicodeScalars.append(scalars[index])
            start = scalars.index(after: index)
        }
        if start == scalars.startIndex { return text.applyingTransform(transform, reverse: false)! }
        if start < scalars.endIndex {
            output += String(scalars[start...]).applyingTransform(transform, reverse: false)!
        }
        return output
    }

    @inline(__always)
    private static func normalizationBarrier(_ scalar: Unicode.Scalar) -> Bool {
        typealias Property = UnicodeNormalization.Property
        // An inert scalar already behaves as a starter in ICU. Reuse the existing
        // quick-check tables; only potentially changing scalars need an age lookup.
        let mask = Property.notNFKC | Property.notNFKD | Property.combiningClassMask
        guard UnicodeNormalization.properties(of: scalar.value) & mask != 0 else { return false }
        return !assigned(scalar, through: 9)
    }

    static func isNonspacingMark(_ value: UInt32) -> Bool {
        // Category changes to characters already assigned in Unicode 8.
        switch value {
        case 0x1734, 0x1171E: return true
        case 0x1885, 0x1886, 0xA9BD, 0x111C9: return false
        default: break
        }
        guard ScalarClassifier.extraFlags(value: value) & ScalarExtraFlags.nonspacingMark != 0,
            let scalar = Unicode.Scalar(value)
        else { return false }
        return assigned(scalar, through: 8)
    }

    static func isMark(_ value: UInt32) -> Bool {
        // Changes to Mark membership after Unicode 9.
        switch value {
        case 0x1CF2, 0x1CF3: return true
        case 0x111C9: return false
        default: break
        }
        guard ScalarClassifier.flags(value: value) & ScalarFlags.mark != 0,
            let scalar = Unicode.Scalar(value)
        else { return false }
        return assigned(scalar, through: 9)
    }

    static func isControl(_ value: UInt32) -> Bool {
        guard ScalarClassifier.extraFlags(value: value) & ScalarExtraFlags.control != 0,
            let scalar = Unicode.Scalar(value)
        else { return false }
        return assigned(scalar, through: 8)
    }

    static func isPunctuation(_ value: UInt32) -> Bool {
        // Former Po characters, now So and Mn respectively.
        if value == 0x166D || value == 0x111C9 { return true }
        guard ScalarClassifier.flags(value: value) & ScalarFlags.punctuation != 0,
            let scalar = Unicode.Scalar(value)
        else { return false }
        return assigned(scalar, through: 8)
    }
}
