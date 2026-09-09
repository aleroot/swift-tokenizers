import Foundation

/// Identity text needs no alignment table. Slices share their backing storage; only
/// rewriting stages allocate a map, and only during an offsets encode.
struct AlignedText {
    let bytes: ArraySlice<UInt8>
    let origins: ArraySlice<Range<Int>>?
    let sourceStart: Int

    init(_ text: String) {
        bytes = Array(text.utf8)[...]
        origins = nil
        sourceStart = 0
    }

    init(bytes: ArraySlice<UInt8>, origins: ArraySlice<Range<Int>>?, sourceStart: Int) {
        self.bytes = bytes
        self.origins = origins
        self.sourceStart = sourceStart
    }

    init(_ units: [Unit]) {
        var bytes: [UInt8] = []
        var origins: [Range<Int>] = []
        for unit in units {
            bytes.append(contentsOf: unit.scalar.utf8)
            origins.append(contentsOf: repeatElement(unit.origin, count: unit.scalar.utf8.count))
        }
        self.init(bytes: bytes[...], origins: origins[...], sourceStart: units.first?.origin.lowerBound ?? 0)
    }

    var text: String { String(decoding: bytes, as: UTF8.self) }

    func slice(_ range: Range<Int>) -> Self {
        Self(bytes: bytes[bytes.startIndex + range.lowerBound..<bytes.startIndex + range.upperBound],
             origins: origins.map { $0[$0.startIndex + range.lowerBound..<$0.startIndex + range.upperBound] },
             sourceStart: sourceStart + range.lowerBound)
    }

    func sourceRange(_ range: Range<Int>) -> Range<Int> {
        guard let origins else { return sourceStart + range.lowerBound..<sourceStart + range.upperBound }
        guard !range.isEmpty else {
            let start = range.lowerBound < origins.count
                ? origins[origins.startIndex + range.lowerBound].lowerBound : origins.last?.upperBound ?? sourceStart
            return start..<start
        }
        let start = origins[origins.startIndex + range.lowerBound].lowerBound
        return start..<max(start, origins[origins.startIndex + range.upperBound - 1].upperBound)
    }

    struct Unit {
        var scalar: Unicode.Scalar
        let origin: Range<Int>
    }

    var units: [Unit] {
        bytes.withUnsafeBufferPointer { bytes in
            var result: [Unit] = []
            var i = 0
            while i < bytes.count {
                let (value, width) = UTF8Cursor.decode(bytes, at: i)
                result.append(Unit(scalar: Unicode.Scalar(value)!, origin: sourceRange(i..<i + width)))
                i += width
            }
            return result
        }
    }

    func prepending(_ prefix: String) -> Self {
        guard !bytes.isEmpty, !prefix.isEmpty else { return self }
        var units = units
        let origin = units[0].origin
        units.insert(contentsOf: prefix.unicodeScalars.map { Unit(scalar: $0, origin: origin) }, at: 0)
        return Self(units)
    }

    func byteLevel() -> Self {
        var output: [UInt8] = []
        var alignments: [Range<Int>] = []
        bytes.withUnsafeBufferPointer { bytes in
            var i = 0
            while i < bytes.count {
                let width = UTF8Cursor.width(bytes[i])
                let origin = sourceRange(i..<i + width)
                for byte in bytes[i..<i + width] {
                    let scalar = Unicode.Scalar(ByteLevelAlphabet.byteToScalar[Int(byte)])!
                    output.append(contentsOf: scalar.utf8)
                    alignments.append(contentsOf: repeatElement(origin, count: scalar.utf8.count))
                }
                i += width
            }
        }
        return Self(bytes: output[...], origins: alignments[...], sourceStart: sourceStart)
    }

    /// Apply the reference normalizer's scalar edit stream. A positive change inserts
    /// after the previous input scalar; a negative change consumes following scalars.
    func transformed(_ edits: [(Unicode.Scalar, Int)]) throws -> Self {
        let input = units
        var cursor = 0
        var output: [Unit] = []
        for (scalar, change) in edits {
            let index = change > 0 ? cursor - 1 : cursor
            guard index < input.count else {
                throw TokenizerError.invalidConfiguration("Normalizer produced an invalid alignment")
            }
            let origin = index < 0 ? sourceStart..<sourceStart : input[index].origin
            output.append(Unit(scalar: scalar, origin: origin))
            if change <= 0 { cursor += 1 - change }
        }
        return Self(output)
    }
}
