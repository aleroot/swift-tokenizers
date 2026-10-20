import Foundation

/// JSON 2.0 separates SentencePiece's text rewrite from its delimiter split. Works
/// directly on UTF-8; the same edit stream supplies offsets without diffing strings.
/// `@unchecked Sendable`: the marker buffer is immutable and owned by this instance.
final class MetaspaceNormalizer: ByteNormalizer, @unchecked Sendable {
    enum Prepend: String {
        case always, first, never
    }

    let replacement: String
    let prepend: Prepend
    let dropWhitespace: Bool
    let needsSourceOffset: Bool
    private let marker: UnsafeBufferPointer<UInt8>
    private let markerIsSpace: Bool

    init(config: Config) throws {
        guard let replacement = config.replacement.string(), replacement.unicodeScalars.count == 1,
            let scheme = config.prepend.string(), let prepend = Prepend(rawValue: scheme),
            case let .boolean(dropWhitespace) = config.dropWhitespace.value
        else {
            throw TokenizerError.invalidConfiguration(
                "MetaspaceNormalizer requires one replacement scalar, prepend (always/first/never), and drop_whitespace"
            )
        }
        self.replacement = replacement
        self.prepend = prepend
        self.dropWhitespace = dropWhitespace
        needsSourceOffset = prepend == .first
        markerIsSpace = replacement == " "
        let storage = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: replacement.utf8.count)
        _ = storage.initialize(from: replacement.utf8)
        marker = UnsafeBufferPointer(storage)
    }

    deinit { marker.deallocate() }

    @inline(__always)
    private func startsWithMarker(_ bytes: UnsafeBufferPointer<UInt8>, at start: Int, end: Int) -> Bool {
        guard end - start >= marker.count, bytes[start] == marker[0] else { return false }
        return marker.count == 1 || memcmp(bytes.baseAddress! + start, marker.baseAddress!, marker.count) == 0
    }

    @inline(__always)
    private static func isWhitespace(_ bytes: UnsafeBufferPointer<UInt8>, at index: Int) -> Bool {
        let byte = bytes[index]
        if byte < 0x80 { return byte == 0x20 || byte >= 0x09 && byte <= 0x0D }
        let (scalar, _) = UTF8Cursor.decode(bytes, at: index)
        return ScalarClassifier.flags(value: scalar) & ScalarFlags.whitespace != 0
    }

    /// Skip ASCII word runs sixteen bytes at a time; Unicode keeps the scalar rule.
    @inline(__always)
    private static func firstWhitespace(in bytes: UnsafeBufferPointer<UInt8>, from start: Int) -> Int {
        var index = start
        while index < bytes.count {
            if bytes[index] < 0x80, index + ByteKernels.width <= bytes.count {
                let chunk = ByteKernels.load(bytes.baseAddress!, index)
                let stops = ByteKernels.bits(ByteKernels.whitespace(chunk) | ByteKernels.nonASCII(chunk))
                if stops == 0 {
                    index += ByteKernels.width
                    continue
                }
                index += stops.trailingZeroBitCount
            }
            if isWhitespace(bytes, at: index) { return index }
            index += UTF8Cursor.width(bytes[index])
        }
        return index
    }

    @inline(__always)
    private func prepends(at sourceOffset: Int) -> Bool {
        prepend == .always || prepend == .first && sourceOffset == 0
    }

    func isIdentity(on bytes: UnsafeBufferPointer<UInt8>) -> Bool { isIdentity(on: bytes, at: 0) }

    func isIdentity(on bytes: UnsafeBufferPointer<UInt8>, at sourceOffset: Int) -> Bool {
        guard !bytes.isEmpty else { return true }
        if prepends(at: sourceOffset), dropWhitespace || bytes[0] != 0x20,
            !startsWithMarker(bytes, at: 0, end: bytes.count)
        {
            return false
        }
        if dropWhitespace {
            if Self.firstWhitespace(in: bytes, from: 0) < bytes.count { return false }
        } else if !markerIsSpace, ByteKernels.firstIndex(of: 0x20, in: bytes, from: 0) < bytes.count {
            return false
        }
        return true
    }

    func normalize(_ bytes: UnsafeBufferPointer<UInt8>, into output: inout [UInt8], scratch: ScratchBuffers) {
        normalize(bytes, at: 0, into: &output, scratch: scratch)
    }

    func normalize(
        _ bytes: UnsafeBufferPointer<UInt8>, at sourceOffset: Int,
        into output: inout [UInt8], scratch: ScratchBuffers
    ) {
        guard !bytes.isEmpty else { return }
        // A source byte can expand to at most one marker, plus an optional prefix.
        // Reserve once and bulk-copy the edits; per-fragment Array appends otherwise
        // dominate the cached model's work on SentencePiece input.
        let marker = marker
        output.appendUninitialized(maximum: bytes.count * marker.count + marker.count) { destination in
            var written = 0
            rewrite(
                bytes, at: sourceOffset,
                onText: { range in
                    (destination + written).update(from: bytes.baseAddress! + range.lowerBound, count: range.count)
                    written += range.count
                },
                onMarker: { _ in
                    let start = destination + written
                    // The usual UTF-8 marker is three bytes; avoid a tiny memcpy per word.
                    switch marker.count {
                    case 1: start[0] = marker[0]
                    case 3:
                        start[0] = marker[0]
                        start[1] = marker[1]
                        start[2] = marker[2]
                    default: start.update(from: marker.baseAddress!, count: marker.count)
                    }
                    written += marker.count
                })
            return written
        }
    }

    /// A marker carries the range of the scalar it replaces, or the first scalar of
    /// a word when prepended. Removed whitespace produces no edit.
    func rewrite(
        _ bytes: UnsafeBufferPointer<UInt8>, at sourceOffset: Int,
        onText: (Range<Int>) -> Void, onMarker: (Range<Int>) -> Void
    ) {
        guard !bytes.isEmpty else { return }
        if dropWhitespace {
            let ascii = ByteKernels.isASCII(bytes)
            var firstWord = sourceOffset == 0 && !Self.isWhitespace(bytes, at: 0)
            var index = 0
            while index < bytes.count {
                if Self.isWhitespace(bytes, at: index) {
                    index += UTF8Cursor.width(bytes[index])
                    continue
                }
                let start = index
                if ascii {
                    index = Self.firstWhitespace(in: bytes, from: start)
                } else {
                    repeat {
                        index += UTF8Cursor.width(bytes[index])
                    } while index < bytes.count && !Self.isWhitespace(bytes, at: index)
                }
                if (prepend == .always || prepend == .first && firstWord),
                    !startsWithMarker(bytes, at: start, end: index)
                {
                    onMarker(start..<start + UTF8Cursor.width(bytes[start]))
                }
                firstWord = false
                onText(start..<index)
            }
        } else {
            if prepends(at: sourceOffset), bytes[0] != 0x20,
                !startsWithMarker(bytes, at: 0, end: bytes.count)
            {
                onMarker(0..<UTF8Cursor.width(bytes[0]))
            }
            var start = 0
            while start < bytes.count {
                let space = ByteKernels.firstIndex(of: 0x20, in: bytes, from: start)
                if space > start { onText(start..<space) }
                if space == bytes.count { return }
                onMarker(space..<space + 1)
                start = space + 1
            }
        }
    }
}
