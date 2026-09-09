import Foundation

/// An opt-in encoding aligned to the original input, before normalization.
/// The offset table is nil when not requested. Its entries are UTF-8 byte ranges.
/// A nil entry marks a special token inserted by the post-processor; a special token actually present in the input has a source range.
/// Several tokens can cover the same scalar (for example, an emoji's byte tokens).
public struct TokenEncoding: Sendable {
    public let ids: [Int]
    public let offsets: [Range<Int>?]?
    public let text: String

    /// Creates an encoding for a custom tokenizer. Ranges must use Unicode-scalar
    /// boundaries in `text`; offsets and IDs must have the same count.
    public init(text: String, ids: [Int], offsets: [Range<Int>?]?) throws {
        var copy = text
        let valid = copy.withUTF8 { bytes in
            guard let offsets else { return true }
            return ids.count == offsets.count && offsets.allSatisfy { offset in
                guard let offset else { return true }
                guard offset.lowerBound >= 0, offset.upperBound <= bytes.count else { return false }
                return (offset.lowerBound == bytes.count || bytes[offset.lowerBound] & 0xC0 != 0x80)
                    && (offset.upperBound == bytes.count || bytes[offset.upperBound] & 0xC0 != 0x80)
            }
        }
        guard valid else { throw TokenizerError.invalidConfiguration("Invalid source offsets") }
        self.text = text
        self.ids = ids
        self.offsets = offsets
    }

    init(text: String, ids: [Int]) {
        self.text = text
        self.ids = ids
        offsets = nil
    }

    init(text: String, tokens: [AlignedToken]) {
        self.text = text
        ids = tokens.map(\.id)
        // Post-processors trim byte offsets. Like HF's character conversion, a
        // trimmed boundary inside a scalar rounds down to that scalar's start.
        var copy = text
        offsets = copy.withUTF8 { bytes in
            tokens.map { token in
                token.offset.map { range in
                    var start = range.lowerBound
                    var end = range.upperBound
                    while start > 0, start < bytes.count, bytes[start] & 0xC0 == 0x80 { start -= 1 }
                    while end > 0, end < bytes.count, bytes[end] & 0xC0 == 0x80 { end -= 1 }
                    return start..<end
                }
            }
        }
    }

    /// Converts all source offsets to Foundation ranges in one pass over the text.
    /// Expansion uses Swift Character boundaries to paint complete grapheme clusters.
    /// Expansion can cause adjacent tokens' ranges to overlap.
    public func utf16Ranges(expandingToGraphemeClusters: Bool = false) -> [NSRange?]? {
        guard let offsets else { return nil }
        var copy = text
        let ranges: [NSRange?] = copy.withUTF8 { bytes in
            if ASCII.isASCII(bytes) {
                return offsets.map { $0.map { NSRange(location: $0.lowerBound, length: $0.count) } }
            }
            var positions = [Int](repeating: 0, count: bytes.count + 1)
            var i = 0
            var utf16 = 0
            while i < bytes.count {
                let (value, width) = UTF8Cursor.decode(bytes, at: i)
                for j in i..<i + width { positions[j] = utf16 }
                i += width
                utf16 += value > 0xFFFF ? 2 : 1
            }
            positions[i] = utf16
            return offsets.map { offset in
                offset.map {
                    NSRange(location: positions[$0.lowerBound], length: positions[$0.upperBound] - positions[$0.lowerBound])
                }
            }
        }
        guard expandingToGraphemeClusters else { return ranges }
        // Index only multi-scalar graphemes. Repeated NSString range expansion can
        // repeatedly traverse the same Unicode text for every byte token.
        var clusters: [Range<Int>] = []
        var position = 0
        for character in text {
            var length = 0
            var count = 0
            for scalar in character.unicodeScalars {
                length += scalar.value > 0xFFFF ? 2 : 1
                count += 1
            }
            if count > 1 { clusters.append(position..<position + length) }
            position += length
        }
        guard !clusters.isEmpty else { return ranges }
        func containing(_ position: Int) -> Range<Int>? {
            var low = 0
            var high = clusters.count
            while low < high {
                let mid = (low + high) / 2
                if clusters[mid].lowerBound < position { low = mid + 1 } else { high = mid }
            }
            guard low > 0, position < clusters[low - 1].upperBound else { return nil }
            return clusters[low - 1]
        }
        return ranges.map { range in
            range.map {
                guard $0.length > 0 else { return $0 }
                let start = containing($0.location)?.lowerBound ?? $0.location
                let end = containing(NSMaxRange($0))?.upperBound ?? NSMaxRange($0)
                return NSRange(location: start, length: end - start)
            }
        }
    }
}

public extension Tokenizer {
    /// The existing overload returns IDs. This overload additionally provides an
    /// encoding result, with an optional offset table controlled by `withOffsets`.
    /// Third-party conformers retain source compatibility and may override this method.
    func encode(text: String, addSpecialTokens: Bool, withOffsets: Bool) throws -> TokenEncoding {
        guard !withOffsets else {
            throw TokenizerError.unsupportedComponent("offset encoding for this tokenizer")
        }
        return TokenEncoding(text: text, ids: encode(text: text, addSpecialTokens: addSpecialTokens))
    }

    func encode(text: String, withOffsets: Bool) throws -> TokenEncoding {
        try encode(text: text, addSpecialTokens: true, withOffsets: withOffsets)
    }
}

struct AlignedToken {
    let id: Int
    var offset: Range<Int>?
    var spelling: String? = nil
}
