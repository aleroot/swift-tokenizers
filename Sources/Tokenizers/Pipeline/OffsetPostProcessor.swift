import Foundation

extension PostProcessor {
    /// HF's character-offset encoding trims in scalar coordinates. Convert around
    /// processing so a whitespace scalar occupying several UTF-8 bytes trims completely.
    func processOffsets(
        _ tokens: [AlignedToken], text: String, addSpecialTokens: Bool,
        resolve: (String) -> Int?, spelling: (Int) -> String?
    ) throws -> [AlignedToken] {
        var copy = text
        guard trimsOffsets, !copy.withUTF8({ ASCII.isASCII($0) }) else {
            return try processOffsets(tokens, addSpecialTokens: addSpecialTokens, resolve: resolve, spelling: spelling)
        }
        var boundaries = [0]
        for scalar in text.unicodeScalars { boundaries.append(boundaries.last! + scalar.utf8.count) }
        func scalarIndex(_ byte: Int) -> Int {
            var low = 0
            var high = boundaries.count
            while low < high {
                let mid = (low + high) / 2
                if boundaries[mid] < byte { low = mid + 1 } else { high = mid }
            }
            return low
        }
        let scalars = tokens.map { token in
            var token = token
            token.offset = token.offset.map { scalarIndex($0.lowerBound)..<scalarIndex($0.upperBound) }
            return token
        }
        return try processOffsets(scalars, addSpecialTokens: addSpecialTokens, resolve: resolve, spelling: spelling).map
        { token in
            var token = token
            token.offset = token.offset.map { boundaries[$0.lowerBound]..<boundaries[$0.upperBound] }
            return token
        }
    }

    private var trimsOffsets: Bool {
        switch self {
        case let processor as ByteLevelPostProcessor: return processor.trimOffsets
        case let processor as RobertaProcessing: return processor.trimOffset
        case let processor as SequenceProcessing: return processor.processors.contains { $0.trimsOffsets }
        default: return false
        }
    }

    func processOffsets(_ tokens: [AlignedToken], addSpecialTokens: Bool,
                        resolve: (String) -> Int?, spelling: (Int) -> String?) throws -> [AlignedToken] {
        switch self {
        case let template as TemplateProcessing:
            var result: [AlignedToken] = []
            for item in template.singleItems {
                switch item {
                case .sequenceA: result.append(contentsOf: tokens)
                case let .special(token):
                    if addSpecialTokens, let id = resolve(token) { result.append(AlignedToken(id: id, offset: nil)) }
                case .sequenceB, .ignored: break
                }
            }
            return result
        case let byteLevel as ByteLevelPostProcessor:
            return byteLevel.trimOffsets ? Self.trim(tokens, prefixSpace: byteLevel.addPrefixSpace, spelling: spelling) : tokens
        case let roberta as RobertaProcessing:
            var result = roberta.trimOffset ? Self.trim(tokens, prefixSpace: roberta.addPrefixSpace, spelling: spelling) : tokens
            if addSpecialTokens {
                if let id = resolve(roberta.cls.1) { result.insert(AlignedToken(id: id, offset: nil), at: 0) }
                if let id = resolve(roberta.sep.1) { result.append(AlignedToken(id: id, offset: nil)) }
            }
            return result
        case let bert as BertProcessing:
            guard addSpecialTokens else { return tokens }
            var result = tokens
            if let id = resolve(bert.cls.1) { result.insert(AlignedToken(id: id, offset: nil), at: 0) }
            if let id = resolve(bert.sep.1) { result.append(AlignedToken(id: id, offset: nil)) }
            return result
        case let sequence as SequenceProcessing:
            return try sequence.processors.reduce(tokens) {
                try $1.processOffsets($0, addSpecialTokens: addSpecialTokens, resolve: resolve, spelling: spelling)
            }
        default: throw TokenizerError.unsupportedComponent("offsets for post-processor \(type(of: self))")
        }
    }

    private static func trim(_ tokens: [AlignedToken], prefixSpace: Bool, spelling: (Int) -> String?) -> [AlignedToken] {
        tokens.enumerated().map { index, token in
            guard let range = token.offset, let text = token.spelling ?? spelling(token.id) else { return token }
            func whitespace(_ scalar: Unicode.Scalar) -> Bool { scalar == "Ġ" || scalar.properties.isWhitespace }
            var leading = text.unicodeScalars.prefix(while: whitespace).count
            let trailing = text.unicodeScalars.reversed().prefix(while: whitespace).count
            if prefixSpace, leading == 1, index == 0 || range.lowerBound == 0 { leading = 0 }
            let start = min(range.lowerBound + leading, range.upperBound)
            let end = max(start, range.upperBound - trailing)
            var result = token
            result.offset = start..<end
            return result
        }
    }
}
