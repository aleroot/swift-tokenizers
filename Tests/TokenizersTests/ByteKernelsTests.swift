// Checks the SIMD byte kernels against scalar reference implementations on random buffers of
// every length around the vector width, including non-ASCII bytes in every lane position.

import Testing

@testable import Tokenizers

@Suite("Byte kernels")
struct ByteKernelsTests {
    /// Deterministic pseudo-random bytes biased towards ASCII with a sprinkling of the
    /// interesting classes (controls, uppercase, punctuation, whitespace, non-ASCII).
    static func buffers() -> [[UInt8]] {
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        func next() -> UInt8 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let roll = UInt8(truncatingIfNeeded: state >> 56)
            switch roll % 8 {
            case 0: return UInt8(truncatingIfNeeded: state >> 40) | 0x80
            case 1: return UInt8(truncatingIfNeeded: state >> 40) & 0x1F
            case 2: return 0x41 + UInt8(truncatingIfNeeded: state >> 40) % 26
            case 3:
                return [0x20, 0x09, 0x0A, 0x0D, 0x7F, 0x21, 0x2F, 0x3A, 0x40, 0x5B, 0x60, 0x7B, 0x7E][
                    Int(state >> 40) % 13]
            default: return 0x61 + UInt8(truncatingIfNeeded: state >> 40) % 26
            }
        }
        var result: [[UInt8]] = [[]]
        for length in 1...70 {
            for _ in 0..<8 { result.append((0..<length).map { _ in next() }) }
        }
        // All-ASCII and all-non-ASCII buffers of every length.
        for length in 1...40 {
            result.append([UInt8](repeating: 0x61, count: length))
            result.append([UInt8](repeating: 0xC3, count: length))
        }
        return result
    }

    @Test("ASCII boundary scans")
    func asciiScans() {
        for buffer in Self.buffers() {
            buffer.withUnsafeBufferPointer { bytes in
                for start in stride(from: 0, through: bytes.count, by: max(1, bytes.count / 5)) {
                    let expectedNonASCII = bytes[start...].firstIndex { $0 >= 0x80 } ?? bytes.count
                    #expect(ByteKernels.firstNonASCII(bytes, from: start) == expectedNonASCII)
                    let expectedASCII = bytes[start...].firstIndex { $0 < 0x80 } ?? bytes.count
                    #expect(ByteKernels.firstASCII(bytes, from: start) == expectedASCII)
                    let expectedSpace = bytes[start...].firstIndex(of: 0x20) ?? bytes.count
                    #expect(ByteKernels.firstIndex(of: 0x20, in: bytes, from: start) == expectedSpace)
                }
                #expect(ByteKernels.isASCII(bytes) == bytes.allSatisfy { $0 < 0x80 })
            }
        }
    }

    @Test("Containment predicates")
    func containment() {
        for buffer in Self.buffers() {
            buffer.withUnsafeBufferPointer { bytes in
                #expect(ByteKernels.containsControl(bytes) == bytes.contains { $0 < 0x20 || $0 == 0x7F })
                #expect(ByteKernels.containsUppercase(bytes) == bytes.contains { $0 >= 0x41 && $0 <= 0x5A })
            }
        }
    }

    @Test("Lane masks match the scalar classification for every byte value")
    func laneMasks() {
        for value in 0...255 {
            let byte = UInt8(value)
            let v = ByteKernels.Vector(repeating: byte)
            let ascii = value < 0x80
            #expect(
                ByteKernels.bits(ByteKernels.controls(v)) == (ascii && (value < 0x20 || value == 0x7F) ? 0xFFFF : 0),
                "controls \(value)")
            #expect(
                ByteKernels.bits(ByteKernels.uppercase(v)) == (value >= 0x41 && value <= 0x5A ? 0xFFFF : 0),
                "uppercase \(value)")
            #expect(
                ByteKernels.bits(ByteKernels.whitespace(v))
                    == (value == 0x20 || (value >= 0x09 && value <= 0x0D) ? 0xFFFF : 0),
                "whitespace \(value)")
            let punctuation =
                (value >= 33 && value <= 47) || (value >= 58 && value <= 64) || (value >= 91 && value <= 96)
                || (value >= 123 && value <= 126)
            #expect(ByteKernels.bits(ByteKernels.punctuation(v)) == (punctuation ? 0xFFFF : 0), "punctuation \(value)")
            #expect(ByteKernels.bits(ByteKernels.nonASCII(v)) == (ascii ? 0 : 0xFFFF), "nonASCII \(value)")
        }
        // Bit order: lane i ↔ bit i.
        var lanes = ByteKernels.Vector(repeating: 0x61)
        lanes[0] = 0x41
        lanes[5] = 0x5A
        lanes[15] = 0x42
        #expect(ByteKernels.bits(ByteKernels.uppercase(lanes)) == 1 | 1 << 5 | 1 << 15)
        #expect(ByteKernels.anyLane(ByteKernels.uppercase(lanes)))
        #expect(!ByteKernels.allLanes(ByteKernels.uppercase(lanes)))
        #expect(ByteKernels.allLanes(ByteKernels.nonASCII(ByteKernels.Vector(repeating: 0xE2))))
    }

    @Test("In-place ASCII lowercasing")
    func lowercasing() {
        for buffer in Self.buffers() {
            var copy = buffer
            copy.withUnsafeMutableBufferPointer { ByteKernels.lowercaseASCII($0) }
            #expect(copy == buffer.map { $0 >= 0x41 && $0 <= 0x5A ? $0 | 0x20 : $0 })
        }
    }
}
