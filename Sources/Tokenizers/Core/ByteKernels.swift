// Vectorised scans over UTF-8 buffers. Swift's `SIMD16<UInt8>` lowers to a single NEON
// register on Apple silicon (and to SSE/AVX on x86), so these kernels process 16 bytes per
// step instead of taking a branch per byte.
//
// Only the lane-wise arithmetic and bitwise operators of `SIMD16<UInt8>` compile to vector
// instructions in the current standard library; the pointwise comparisons (`.<`, `.==`) and
// the horizontal reductions (`any`, `max()`, `wrappedSum()`) expand to a scalar loop over
// the lanes. The kernels therefore build lane masks (`0x80` in a matching lane, `0` elsewhere)
// from additions and XORs — exact for ASCII lanes, which is all the classifications need —
// and reduce a vector by viewing it as two 64-bit words.
//
// Every kernel is exact for arbitrary bytes: the vector loop handles full chunks, the last
// partial chunk is either re-read as an overlapping vector (for idempotent operations) or
// finished with a scalar loop. Callers pass `UnsafeBufferPointer`s so no copy is made.

import Foundation

enum ByteKernels {
    typealias Vector = SIMD16<UInt8>
    static let width = 16

    @inline(__always)
    static func load(_ base: UnsafePointer<UInt8>, _ offset: Int) -> Vector {
        UnsafeRawPointer(base).loadUnaligned(fromByteOffset: offset, as: Vector.self)
    }

    @inline(__always)
    static func store(_ vector: Vector, _ base: UnsafeMutablePointer<UInt8>, _ offset: Int) {
        UnsafeMutableRawPointer(base).storeBytes(of: vector, toByteOffset: offset, as: Vector.self)
    }

    // MARK: Lane masks

    /// Lanes with the high bit set: `0x80` where `v` is not ASCII, `0` elsewhere.
    @inline(__always)
    static func nonASCII(_ v: Vector) -> Vector { v & 0x80 }

    /// `0x80` in ASCII lanes where `lower <= v < upper` (`lower`, `upper` ≤ 0x80), `0` elsewhere.
    ///
    /// For an ASCII byte `v + (0x80 - bound)` cannot wrap, so its high bit is set exactly when
    /// `v >= bound`; XOR-ing the two tests keeps the lanes between the bounds.
    @inline(__always)
    static func range(_ v: Vector, _ lower: UInt8, _ upper: UInt8) -> Vector {
        ((v &+ (0x80 &- lower)) ^ (v &+ (0x80 &- upper))) & ~v & 0x80
    }

    /// `0x80` in lanes equal to `byte` (any value), `0` elsewhere.
    @inline(__always)
    static func equals(_ v: Vector, _ byte: UInt8) -> Vector {
        let difference = v ^ byte
        return ~((difference &+ 0x7F) | difference) & 0x80
    }

    /// Lanes holding a C0 control byte or DEL (`0x00...0x1F`, `0x7F`).
    @inline(__always)
    static func controls(_ v: Vector) -> Vector {
        range(v, 0x00, 0x20) | equals(v, 0x7F)
    }

    /// Lanes holding ASCII uppercase letters.
    @inline(__always)
    static func uppercase(_ v: Vector) -> Vector {
        range(v, 0x41, 0x5B)
    }

    /// Lanes holding ASCII whitespace as `char::is_whitespace` sees it: space and `\t\n\v\f\r`.
    @inline(__always)
    static func whitespace(_ v: Vector) -> Vector {
        equals(v, 0x20) | range(v, 0x09, 0x0E)
    }

    /// Lanes holding ASCII punctuation (`!`…`/`, `:`…`@`, `[`…`` ` ``, `{`…`~`), i.e. every
    /// printable ASCII byte that is not a letter, digit or space.
    @inline(__always)
    static func punctuation(_ v: Vector) -> Vector {
        range(v, 0x21, 0x30) | range(v, 0x3A, 0x41) | range(v, 0x5B, 0x61) | range(v, 0x7B, 0x7F)
    }

    /// Lanes holding ASCII word characters (regex `\w`: letters, digits, `_`).
    @inline(__always)
    static func word(_ v: Vector) -> Vector {
        range(v, 0x30, 0x3A) | range(v, 0x41, 0x5B) | range(v, 0x61, 0x7B) | equals(v, 0x5F)
    }

    /// The 16 bytes at `offset` when they are all ASCII, or `nil` (fewer than 16 bytes remain or
    /// one of them is not ASCII): the guard for chunked ASCII fast paths.
    @inline(__always)
    static func asciiChunk(_ bytes: UnsafeBufferPointer<UInt8>, at offset: Int) -> Vector? {
        guard offset + width <= bytes.count else { return nil }
        let v = load(bytes.baseAddress!, offset)
        return anyLane(nonASCII(v)) ? nil : v
    }

    // MARK: Horizontal operations

    /// `true` if any lane of `mask` is set.
    @inline(__always)
    static func anyLane(_ mask: Vector) -> Bool {
        let words = unsafeBitCast(mask, to: SIMD2<UInt64>.self)
        return (words.x | words.y) != 0
    }

    /// `true` if every lane of `mask` is set.
    @inline(__always)
    static func allLanes(_ mask: Vector) -> Bool {
        let words = unsafeBitCast(mask, to: SIMD2<UInt64>.self)
        return (words.x & words.y) == 0x8080_8080_8080_8080
    }

    /// The bit for each lane of `mask` (`bit i` ↔ `lane i`).
    @inline(__always)
    static func bits(_ mask: Vector) -> UInt16 {
        let words = unsafeBitCast(mask, to: SIMD2<UInt64>.self)
        return UInt16(laneBits(words.x)) | UInt16(laneBits(words.y)) << 8
    }

    /// Gathers the high bit of each byte of `word` into the low byte of the result.
    @inline(__always)
    static func laneBits(_ word: UInt64) -> UInt64 {
        (((word >> 7) & 0x0101_0101_0101_0101) &* 0x0102_0408_1020_4080) >> 56
    }

    // MARK: ASCII scans

    /// Index of the first byte ≥ 0x80 at or after `start`, or `bytes.count`.
    @inline(__always)
    static func firstNonASCII(_ bytes: UnsafeBufferPointer<UInt8>, from start: Int = 0) -> Int {
        let n = bytes.count
        guard let base = bytes.baseAddress else { return n }
        var i = start
        while i + width <= n {
            if anyLane(nonASCII(load(base, i))) { break }
            i += width
        }
        while i < n, base[i] < 0x80 { i += 1 }
        return i
    }

    /// `true` if every byte is < 0x80.
    @inline(__always)
    static func isASCII(_ bytes: UnsafeBufferPointer<UInt8>) -> Bool {
        firstNonASCII(bytes) == bytes.count
    }

    /// Index of the first byte < 0x80 at or after `start`, or `bytes.count`.
    @inline(__always)
    static func firstASCII(_ bytes: UnsafeBufferPointer<UInt8>, from start: Int) -> Int {
        let n = bytes.count
        guard let base = bytes.baseAddress else { return n }
        var i = start
        while i + width <= n {
            if !allLanes(nonASCII(load(base, i))) { break }
            i += width
        }
        while i < n, base[i] >= 0x80 { i += 1 }
        return i
    }

    /// Index of the first occurrence of `byte` at or after `start`, or `bytes.count`.
    @inline(__always)
    static func firstIndex(of byte: UInt8, in bytes: UnsafeBufferPointer<UInt8>, from start: Int) -> Int {
        let n = bytes.count
        guard start < n, let base = bytes.baseAddress else { return n }
        guard let hit = memchr(base + start, Int32(byte), n - start) else { return n }
        return UnsafePointer<UInt8>(hit.assumingMemoryBound(to: UInt8.self)) - base
    }

    /// Index of the first byte equal to `first` or `second` at or after `start`, or `bytes.count`.
    @inline(__always)
    static func firstIndex(
        of first: UInt8, or second: UInt8, in bytes: UnsafeBufferPointer<UInt8>, from start: Int
    )
        -> Int
    {
        let n = bytes.count
        guard let base = bytes.baseAddress else { return n }
        var i = start
        while i + width <= n {
            let v = load(base, i)
            let lanes = equals(v, first) | equals(v, second)
            if anyLane(lanes) { return i + bits(lanes).trailingZeroBitCount }
            i += width
        }
        while i < n, base[i] != first, base[i] != second { i += 1 }
        return i
    }

    /// `true` if `bytes` contain two adjacent occurrences of `byte`.
    @inline(__always)
    static func containsRepeat(of byte: UInt8, in bytes: UnsafeBufferPointer<UInt8>) -> Bool {
        let n = bytes.count
        guard let base = bytes.baseAddress else { return false }
        var i = 0
        var previous: UInt16 = 0  // bit 0: the byte before lane 0 is `byte`
        while i + width <= n {
            let lanes = bits(equals(load(base, i), byte))
            if lanes & (lanes << 1 | previous) != 0 { return true }
            previous = lanes >> 15
            i += width
        }
        while i < n {
            if base[i] == byte, i > 0, base[i - 1] == byte { return true }
            i += 1
        }
        return false
    }

    /// `true` if any byte of `bytes` is a C0 control or DEL.
    @inline(__always)
    static func containsControl(_ bytes: UnsafeBufferPointer<UInt8>) -> Bool {
        contains(bytes, controls) { $0 < 0x20 || $0 == 0x7F }
    }

    /// `true` if any byte of `bytes` is an ASCII uppercase letter.
    @inline(__always)
    static func containsUppercase(_ bytes: UnsafeBufferPointer<UInt8>) -> Bool {
        contains(bytes, uppercase) { $0 >= 0x41 && $0 <= 0x5A }
    }

    /// `true` if any byte satisfies `predicate` (`lanes` is its vector form).
    @inline(__always)
    static func contains(
        _ bytes: UnsafeBufferPointer<UInt8>, _ lanes: (Vector) -> Vector, _ predicate: (UInt8) -> Bool
    ) -> Bool {
        let n = bytes.count
        guard let base = bytes.baseAddress else { return false }
        var i = 0
        while i + width <= n {
            if anyLane(lanes(load(base, i))) { return true }
            i += width
        }
        if i < n {
            if n >= width { return anyLane(lanes(load(base, n - width))) }
            while i < n {
                if predicate(base[i]) { return true }
                i += 1
            }
        }
        return false
    }

    // MARK: Transformations

    /// Lowercases ASCII letters in place (other bytes, including non-ASCII, are untouched).
    @inline(__always)
    static func lowercaseASCII(_ bytes: UnsafeMutableBufferPointer<UInt8>) {
        let n = bytes.count
        guard let base = bytes.baseAddress else { return }
        var i = 0
        while i + width <= n {
            store(lowercased(load(base, i)), base, i)
            i += width
        }
        if i < n {
            if n >= width {
                // Idempotent, so re-reading the last full vector is safe.
                store(lowercased(load(base, n - width)), base, n - width)
            } else {
                while i < n {
                    if base[i] &- 0x41 < 26 { base[i] |= 0x20 }
                    i += 1
                }
            }
        }
    }

    /// `v` with ASCII uppercase letters lowercased (`0x80` mask bit → `0x20` case bit).
    @inline(__always)
    static func lowercased(_ v: Vector) -> Vector {
        v | (uppercase(v) &>> 2)
    }
}

extension Array where Element == UInt8 {
    /// Appends the bytes that `body` writes to the pointer it receives (at most `maximum` of
    /// them); `body` returns how many it wrote. Lets a kernel emit output with plain stores
    /// instead of one `append` per fragment.
    @inline(__always)
    mutating func appendUninitialized(maximum: Int, _ body: (UnsafeMutablePointer<UInt8>) -> Int) {
        let start = count
        append(contentsOf: repeatElement(0, count: maximum))
        let written = withUnsafeMutableBufferPointer { body($0.baseAddress! + start) }
        removeLast(maximum - written)
    }
}
