// Unicode normalization data for the byte-level normalizers: combining classes, NF* quick-check
// status, canonical decompositions and simple lowercase mappings of the BMP, plus the sparse
// supplementary-plane ranges that are not trivially normalized. Scalars the tables do not cover
// fall back to Foundation.
//
// `UnicodeNormalization.generated.swift` is produced and verified by `UnicodeNormalizationTests`
// (`REGENERATE_UNICODE_TABLES=1`).

enum UnicodeNormalization {
    /// Per-scalar properties: the canonical combining class in the low byte, flags above.
    enum Property {
        static let combiningClassMask: UInt16 = 0xFF
        /// NFC quick-check `No` or `Maybe`: the scalar may compose with or decompose in NFC.
        static let nfcUnstable: UInt16 = 1 << 8
        /// Has a canonical decomposition (NFD quick-check `No`); includes Hangul syllables.
        static let canonicalDecomposition: UInt16 = 1 << 9
        /// Has a compatibility decomposition beyond the canonical one (NFKD quick-check `No`).
        static let compatibilityDecomposition: UInt16 = 1 << 10
        /// The full lowercase mapping is a single BMP scalar other than itself (see ``lowercase(_:)``).
        static let lowercaseMapped: UInt16 = 1 << 11
        /// The full lowercase mapping is not a single scalar (e.g. U+0130 → `i̇`).
        static let lowercaseComplex: UInt16 = 1 << 12
        /// General category `Mn`.
        static let nonspacingMark: UInt16 = 1 << 13
        /// A supplementary-plane scalar with a non-zero combining class, a decomposition, a case
        /// mapping or a mark category: not covered by the BMP tables.
        static let supplementary: UInt16 = 1 << 14

        /// Flags that make a scalar's NFC status depend on context or differ from identity.
        static let notNFC: UInt16 = nfcUnstable | supplementary
        static let notNFD: UInt16 = canonicalDecomposition | supplementary
        static let notNFKC: UInt16 = nfcUnstable | compatibilityDecomposition | supplementary
        static let notNFKD: UInt16 = canonicalDecomposition | compatibilityDecomposition | supplementary
    }

    /// The expanded BMP table (128 KiB), built from the run-length data at first use.
    static let bmp: [UInt16] = expand(runs)

    /// Properties of `value` (any scalar).
    @inline(__always)
    static func properties(of value: UInt32) -> UInt16 {
        if value < 0x10000 { return bmp[Int(value)] }
        return isSupplementaryNontrivial(value) ? Property.supplementary : 0
    }

    /// Whether the supplementary-plane scalar `value` lies in one of the non-trivial ranges.
    static func isSupplementaryNontrivial(_ value: UInt32) -> Bool {
        // `supplementaryRanges` holds sorted, disjoint `[start, end)` pairs.
        var lo = 0
        var hi = supplementaryRanges.count / 2
        while lo < hi {
            let mid = (lo + hi) >> 1
            if supplementaryRanges[2 * mid + 1] <= value {
                lo = mid + 1
            } else {
                hi = mid
            }
        }
        return lo < supplementaryRanges.count / 2 && supplementaryRanges[2 * lo] <= value
    }

    /// Calls `body` with each scalar of the full canonical decomposition of the BMP scalar
    /// `value` (which must have ``Property/canonicalDecomposition``), in canonical order.
    @inline(__always)
    static func decompose(_ value: UInt32, _ body: (UInt32) -> Void) {
        if value >= hangulBase, value < hangulBase + hangulCount {
            let index = value - hangulBase
            body(0x1100 + index / 588)
            body(0x1161 + (index % 588) / 28)
            let trailing = index % 28
            if trailing != 0 { body(0x11A7 + trailing) }
            return
        }
        // `decompositionIndex` holds `scalar << 16 | offset` sorted by scalar; the mapping of an
        // entry runs up to the next entry's offset (a sentinel closes the last one).
        var lo = 0
        var hi = decompositionIndex.count - 1
        let key = value << 16
        while lo < hi {
            let mid = (lo + hi) >> 1
            if decompositionIndex[mid] & 0xFFFF_0000 < key { lo = mid + 1 } else { hi = mid }
        }
        let entry = decompositionIndex[lo]
        precondition(entry & 0xFFFF_0000 == key, "scalar has no canonical decomposition")
        let start = Int(entry & 0xFFFF)
        let end = Int(decompositionIndex[lo + 1] & 0xFFFF)
        for i in start..<end { body(decompositions[i]) }
    }

    /// The simple lowercase mapping of the BMP scalar `value` (which must have
    /// ``Property/lowercaseMapped``).
    @inline(__always)
    static func lowercase(_ value: UInt32) -> UInt32 {
        // `lowercaseIndex` holds `scalar << 16 | lowercase` sorted by scalar.
        var lo = 0
        var hi = lowercaseIndex.count - 1
        let key = value << 16
        while lo < hi {
            let mid = (lo + hi) >> 1
            if lowercaseIndex[mid] & 0xFFFF_0000 < key { lo = mid + 1 } else { hi = mid }
        }
        return lowercaseIndex[lo] & 0xFFFF
    }

    static let hangulBase: UInt32 = 0xAC00
    static let hangulCount: UInt32 = 11172

    /// Expands `runs` (`start << 16 | properties`, ascending) into a 65 536-entry table.
    static func expand(_ runs: [UInt32]) -> [UInt16] {
        var table = [UInt16](repeating: 0, count: 0x10000)
        table.withUnsafeMutableBufferPointer { table in
            for (index, run) in runs.enumerated() {
                let start = Int(run >> 16)
                let end = index + 1 < runs.count ? Int(runs[index + 1] >> 16) : 0x10000
                let value = UInt16(truncatingIfNeeded: run)
                for v in start..<end { table[v] = value }
            }
        }
        return table
    }

    /// Run-length encodes a 65 536-entry property table.
    static func encode(_ table: [UInt16]) -> [UInt32] {
        var runs: [UInt32] = []
        for v in 0..<0x10000 {
            if let last = runs.last, UInt16(truncatingIfNeeded: last) == table[v] { continue }
            runs.append(UInt32(v) << 16 | UInt32(table[v]))
        }
        return runs
    }
}

// MARK: - Quick checks

extension UnicodeNormalization {
    /// Whether the UTF-8 `bytes` are already in the normalization form whose "not normalized"
    /// property mask is `mask` (one of ``Property/notNFC`` …): the Unicode quick-check
    /// algorithm with every `Maybe` treated as a failure, so a `true` result is exact.
    @inline(__always)
    static func isNormalized(_ bytes: UnsafeBufferPointer<UInt8>, mask: UInt16) -> Bool {
        let n = bytes.count
        var i = 0
        var lastClass: UInt16 = 0
        while i < n {
            if bytes[i] < 0x80 {
                // ASCII is stable under every form and resets the combining sequence.
                i = ByteKernels.firstNonASCII(bytes, from: i + 1)
                lastClass = 0
                continue
            }
            let (value, width) = UTF8Cursor.decode(bytes, at: i)
            let properties = properties(of: value)
            if properties & mask != 0 { return false }
            let combiningClass = properties & Property.combiningClassMask
            if combiningClass != 0, lastClass > combiningClass { return false }
            lastClass = combiningClass
            i += width
        }
        return true
    }
}
