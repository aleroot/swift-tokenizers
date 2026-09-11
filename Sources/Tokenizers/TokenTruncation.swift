/// Which end of the input sequence to discard before inserting special tokens.
public enum TruncationSide: Sendable {
    case left
    case right
}

/// An explicit truncation request; the limit includes inserted special tokens.
public struct TokenTruncation: Sendable {
    public var maxLength: Int
    public var side: TruncationSide

    public init(maxLength: Int, side: TruncationSide = .right) {
        self.maxLength = maxLength
        self.side = side
    }
}

public extension Tokenizer {
    /// Encodes within a total token budget, reserving space for the post-processor.
    /// Unlike taking a prefix of encoded IDs, this preserves inserted BOS/EOS/CLS/SEP.
    /// Throws if the budget cannot hold the required special tokens.
    func encode(
        text: String, addSpecialTokens: Bool = true, maxLength: Int,
        truncationSide: TruncationSide = .right, withOffsets: Bool = false
    ) throws -> TokenEncoding {
        try encode(
            text: text, addSpecialTokens: addSpecialTokens,
            truncation: TokenTruncation(maxLength: maxLength, side: truncationSide), withOffsets: withOffsets)
    }

    /// Primitive for custom conformers. The convenience overload forwards all arguments
    /// explicitly so calls through `any Tokenizer` dispatch to the conformer's implementation.
    func encode(
        text: String, addSpecialTokens: Bool, truncation: TokenTruncation, withOffsets: Bool
    ) throws -> TokenEncoding {
        throw TokenizerError.unsupportedComponent("special-token-aware truncation for this tokenizer")
    }
}

extension Array {
    mutating func truncate(to count: Int, side: TruncationSide) {
        guard self.count > count else { return }
        switch side {
        case .left: removeFirst(self.count - count)
        case .right: removeLast(self.count - count)
        }
    }
}
