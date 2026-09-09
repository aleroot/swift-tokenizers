// Run-length encoded BMP classification (``ScalarFlags`` and ``ScalarExtraFlags``), expanded
// into 64 KiB lookup tables at first use.
//
// `UnicodeTables.generated.swift` is produced and verified by `UnicodeTablesTests`
// (`REGENERATE_UNICODE_TABLES=1`).

enum UnicodeTables {
    /// One run of scalars sharing the same flags: `start << 16 | flags << 8 | extraFlags`.
    /// Runs are contiguous and ascending; a run ends where the next one starts.
    struct Run {
        let packed: UInt32
        var start: Int { Int(packed >> 16) }
        var flags: UInt8 { UInt8(truncatingIfNeeded: packed >> 8) }
        var extraFlags: UInt8 { UInt8(truncatingIfNeeded: packed) }
    }

    /// Expands the runs into a 65 536-entry table of the selected flag byte.
    static func expand(_ field: KeyPath<Run, UInt8>) -> [UInt8] {
        var table = [UInt8](repeating: 0, count: 0x10000)
        table.withUnsafeMutableBufferPointer { table in
            for (index, packed) in runs.enumerated() {
                let run = Run(packed: packed)
                let end = index + 1 < runs.count ? Run(packed: runs[index + 1]).start : 0x10000
                let value = run[keyPath: field]
                for v in run.start..<end { table[v] = value }
            }
        }
        return table
    }

    /// Run-length encodes two parallel 65 536-entry tables.
    static func encode(flags: [UInt8], extraFlags: [UInt8]) -> [UInt32] {
        var runs: [UInt32] = []
        for v in 0..<0x10000 {
            let packed = UInt32(v) << 16 | UInt32(flags[v]) << 8 | UInt32(extraFlags[v])
            if let last = runs.last, last & 0xFFFF == packed & 0xFFFF { continue }
            runs.append(packed)
        }
        return runs
    }
}
