// A string wrapper whose equality and hashing are defined over the exact UTF-8
// scalar sequence rather than Unicode canonical equivalence.
//
// Swift's `String` treats "à" (U+0061 U+0300) and "à" (U+00E0) as equal. Tokenizer
// vocabularies frequently contain both spellings as *different* tokens, so a
// vocabulary keyed by plain `String` would silently collapse entries. This type
// preserves the distinction while keeping the compact native `String` storage
// (small strings stay inline, no extra allocation per key).

import Foundation

public struct BinaryDistinctString: Hashable, Comparable, Sendable, ExpressibleByStringLiteral, CustomStringConvertible
{
    /// The wrapped Swift string.
    public let string: String

    /// Creates a binary-distinct string from a Swift `String`.
    @inlinable
    public init(_ string: String) {
        self.string = string
    }

    /// Creates a binary-distinct string from an `NSString`.
    public init(_ nsString: NSString) {
        self.string = String(nsString)
    }

    /// Creates a binary-distinct string from a `Substring`.
    @inlinable
    public init(_ substring: Substring) {
        self.string = String(substring)
    }

    public init(stringLiteral value: String) {
        self.string = value
    }

    /// Bridged `NSString` view, provided for source compatibility.
    public var nsString: NSString { string as NSString }

    /// Number of `Character`s in the string.
    public var count: Int { string.count }

    public var description: String { string }

    // MARK: Equality / Hashing over raw UTF-8

    @inlinable
    public static func == (lhs: BinaryDistinctString, rhs: BinaryDistinctString) -> Bool {
        let l = lhs.string.utf8
        let r = rhs.string.utf8
        guard l.count == r.count else { return false }
        return l.elementsEqual(r)
    }

    @inlinable
    public func hash(into hasher: inout Hasher) {
        var copy = string
        let handled: Bool = copy.withUTF8 { buffer -> Bool in
            hasher.combine(bytes: UnsafeRawBufferPointer(buffer))
            return true
        }
        _ = handled
    }

    public static func < (lhs: BinaryDistinctString, rhs: BinaryDistinctString) -> Bool {
        lhs.string.utf8.lexicographicallyPrecedes(rhs.string.utf8)
    }

    public static func + (lhs: BinaryDistinctString, rhs: BinaryDistinctString) -> BinaryDistinctString {
        BinaryDistinctString(lhs.string + rhs.string)
    }

    // MARK: Convenience

    public func hasPrefix(_ prefix: BinaryDistinctString) -> Bool {
        string.utf8.starts(with: prefix.string.utf8)
    }

    public func hasSuffix(_ suffix: BinaryDistinctString) -> Bool {
        let s = string.utf8
        let x = suffix.string.utf8
        guard x.count <= s.count else { return false }
        return s.suffix(x.count).elementsEqual(x)
    }

    public func lowercased() -> BinaryDistinctString {
        BinaryDistinctString(string.lowercased())
    }

    public func replacingOccurrences(
        of target: BinaryDistinctString, with replacement: BinaryDistinctString
    ) -> BinaryDistinctString {
        BinaryDistinctString(string.replacingOccurrences(of: target.string, with: replacement.string))
    }
}

extension BinaryDistinctString: Codable {
    public init(from decoder: any Swift.Decoder) throws {
        self.string = try decoder.singleValueContainer().decode(String.self)
    }

    public func encode(to encoder: any Swift.Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(string)
    }
}
