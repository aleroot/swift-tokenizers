// Fast non-cryptographic hashing of short byte strings, used by the packed vocabulary
// index and the pretoken cache. Reads 8 bytes at a time and finishes with a
// splitmix-style avalanche so low bits are well distributed for power-of-two tables.

@usableFromInline
enum ByteHash {
    @inlinable
    static func hash(_ bytes: UnsafeBufferPointer<UInt8>) -> UInt64 {
        var h: UInt64 = 0x9E37_79B9_7F4A_7C15 ^ UInt64(bytes.count)
        var i = 0
        let n = bytes.count
        guard let base = bytes.baseAddress else { return mix(h) }
        let raw = UnsafeRawPointer(base)
        while i + 8 <= n {
            let word = raw.loadUnaligned(fromByteOffset: i, as: UInt64.self)
            h = (h ^ word) &* 0xBF58_476D_1CE4_E5B9
            h ^= h >> 29
            i += 8
        }
        let remaining = n - i
        if remaining > 0 {
            // Assemble the tail from at most three aligned-width loads instead of a byte loop.
            var tail: UInt64 = 0
            var shift: UInt64 = 0
            if remaining & 4 != 0 {
                tail |= UInt64(raw.loadUnaligned(fromByteOffset: i, as: UInt32.self))
                i += 4
                shift = 32
            }
            if remaining & 2 != 0 {
                tail |= UInt64(raw.loadUnaligned(fromByteOffset: i, as: UInt16.self)) << shift
                i += 2
                shift += 16
            }
            if remaining & 1 != 0 {
                tail |= UInt64(base[i]) << shift
            }
            h = (h ^ tail) &* 0x94D0_49BB_1331_11EB
            h ^= h >> 31
        }
        return mix(h)
    }

    @inlinable
    static func mix(_ value: UInt64) -> UInt64 {
        var z = value
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Hash of a 64-bit key (used for pair tables).
    @inlinable
    static func hash(key: UInt64) -> UInt64 {
        mix(key &* 0x9E37_79B9_7F4A_7C15)
    }
}
