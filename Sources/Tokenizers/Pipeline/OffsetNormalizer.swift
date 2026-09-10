import Foundation

extension AlignedText {
    func normalized(by normalizer: any ByteNormalizer) throws -> Self {
        if bytes.withUnsafeBufferPointer({ normalizer.isIdentity(on: $0) }) { return self }
        if (normalizer is BertNormalizer || normalizer is LowercaseNormalizer),
           bytes.withUnsafeBufferPointer({ ASCII.isASCII($0) }) {
            var output: [UInt8] = []
            bytes.withUnsafeBufferPointer { normalizer.normalize($0, into: &output, scratch: ScratchBuffers()) }
            // ASCII case/whitespace substitutions preserve positions. Cleaning a
            // control character changes the count and uses the traced path below.
            if output.count == bytes.count {
                return Self(bytes: output[...], origins: origins, sourceStart: sourceStart)
            }
        }
        switch normalizer {
        case let sequence as NormalizerSequence:
            return try sequence.normalizers.reduce(self) { try $0.normalized(by: $1) }
        case let prepend as PrependNormalizer:
            return prepending(prepend.prepend)
        case let replace as ReplaceNormalizer:
            guard let pattern = replace.pattern else { return self }
            return try replacing(pattern)
        case is LowercaseNormalizer:
            return lowercased()
        case is NFDNormalizer: return try unicodeForm(compatible: false, compose: false)
        case is NFKDNormalizer: return try unicodeForm(compatible: true, compose: false)
        case is NFCNormalizer: return try unicodeForm(compatible: false, compose: true)
        case is NFKCNormalizer: return try unicodeForm(compatible: true, compose: true)
        case let bert as BertNormalizer:
            var output: [Unit] = []
            for unit in units {
                let value = unit.scalar.value
                if bert.shouldCleanText {
                    if value == 0xFFFD || BertNormalizer.isControl(value) { continue }
                    if BertNormalizer.isWhitespace(value) {
                        output.append(Unit(scalar: " ", origin: unit.origin))
                        continue
                    }
                }
                let chinese = bert.shouldHandleChineseChars && BertNormalizer.isCJKUnifiedIdeograph(value)
                if chinese { output.append(Unit(scalar: " ", origin: unit.origin)) }
                output.append(unit)
                if chinese { output.append(Unit(scalar: " ", origin: unit.origin)) }
            }
            var result = Self(output)
            if bert.shouldStripAccents {
                result = try result.unicodeForm(compatible: false, compose: false)
                result = Self(result.units.filter {
                        !TokenizerUnicode.isNonspacingMark($0.scalar.value)
                    })
            }
            return bert.shouldLowercase ? result.lowercased() : result
        case is StripAccentsNormalizer:
            return Self(units.filter { !TokenizerUnicode.isMark($0.scalar.value) })
        case let strip as StripNormalizer:
            var output = units[...]
            func whitespace(_ unit: Unit) -> Bool {
                ScalarClassifier.flags(value: unit.scalar.value) & ScalarFlags.whitespace != 0
            }
            if strip.leftStrip { output = output.drop(while: whitespace) }
            if strip.rightStrip {
                while let last = output.last, whitespace(last) { output = output.dropLast() }
            }
            return Self(Array(output))
        case let precompiled as PrecompiledNormalizer:
            var edits: [(Unicode.Scalar, Int)] = []
            func replace(_ old: String, _ new: String) {
                let count = new.unicodeScalars.count
                let diff = count - old.unicodeScalars.count
                edits.append(contentsOf: new.unicodeScalars.map { ($0, 0) })
                if diff > 0 {
                    for i in edits.count - diff..<edits.count { edits[i].1 = 1 }
                } else if diff < 0, !edits.isEmpty { edits[edits.count - 1].1 += diff }
            }
            for grapheme in text {
                let part = String(grapheme)
                if part.utf8.count < 6, let mapped = precompiled.offsetReplacement(part) {
                    replace(part, mapped)
                } else {
                    for scalar in part.unicodeScalars {
                        let part = String(scalar)
                        if let mapped = precompiled.offsetReplacement(part) { replace(part, mapped) }
                        else { edits.append((scalar, 0)) }
                    }
                }
            }
            return try transformed(edits)
        default:
            throw TokenizerError.unsupportedComponent("offsets for normalizer \(type(of: normalizer))")
        }
    }

    private func lowercased() -> Self {
        var output: [Unit] = []
        var mappings: [Unicode.Scalar: [Unicode.Scalar]] = [:]
        for unit in units {
            let value = unit.scalar.value
            if value < 0x80 {
                output.append(Unit(scalar: Unicode.Scalar(value >= 65 && value <= 90 ? value + 32 : value)!, origin: unit.origin))
            } else {
                let mapped: [Unicode.Scalar]
                if let cached = mappings[unit.scalar] { mapped = cached }
                else {
                    mapped = Array(String(unit.scalar).lowercased().unicodeScalars)
                    mappings[unit.scalar] = mapped
                }
                output.append(contentsOf: mapped.map { Unit(scalar: $0, origin: unit.origin) })
            }
        }
        return Self(output)
    }

    private func replacing(_ pattern: StringReplacePattern) throws -> Self {
        let regex: NSRegularExpression
        let replacement: String
        switch pattern {
        case let .regexp(value, content): regex = value; replacement = content
        case let .string(value, content):
            regex = try NSRegularExpression(pattern: value.isEmpty ? "(?:)" : NSRegularExpression.escapedPattern(for: String(decoding: value, as: UTF8.self)))
            replacement = String(decoding: content, as: UTF8.self)
        case let .run(byte, minimum, content):
            regex = try NSRegularExpression(pattern: NSRegularExpression.escapedPattern(for: String(Unicode.Scalar(byte))) + "{\(minimum),}")
            replacement = String(decoding: content, as: UTF8.self)
        }
        let source = text
        var output: [UInt8] = []
        var origins: [Range<Int>] = []
        var end = 0
        // Foundation regex offsets are UTF-16, including scalar boundaries within graphemes.
        let utf16ToByte = Self.utf16ToByte(source)
        func appendOriginal(_ range: Range<Int>) {
            let part = slice(range)
            for unit in part.units {
                output.append(contentsOf: unit.scalar.utf8)
                origins.append(contentsOf: repeatElement(unit.origin, count: unit.scalar.utf8.count))
            }
        }
        for match in regex.matches(in: source, range: NSRange(location: 0, length: source.utf16.count)) {
            let start = utf16ToByte[match.range.location]
            let next = utf16ToByte[NSMaxRange(match.range)]
            appendOriginal(end..<start)
            var previous = next - 1
            while previous > 0, bytes[bytes.startIndex + previous] & 0xC0 == 0x80 { previous -= 1 }
            let origin = next == 0 ? sourceStart..<sourceStart : sourceRange(previous..<next)
            output.append(contentsOf: replacement.utf8)
            origins.append(contentsOf: repeatElement(origin, count: replacement.utf8.count))
            end = next
        }
        appendOriginal(end..<bytes.count)
        return Self(bytes: output[...], origins: origins[...], sourceStart: sourceStart)
    }

    private static func utf16ToByte(_ text: String) -> [Int] {
        var output = [0]
        var byte = 0
        for scalar in text.unicodeScalars {
            if scalar.value > 0xFFFF { output.append(byte) }
            byte += scalar.utf8.count
            output.append(byte)
        }
        return output
    }

    /// Carry decomposition/recomposition edits, rather than diffing normalized strings.
    /// This follows HF's alignment semantics while using the same Unicode forms as encode.
    private func unicodeForm(compatible: Bool, compose: Bool) throws -> Self {
        var edits: [(Unicode.Scalar, Int)] = []
        var decompositions: [Unicode.Scalar: [Unicode.Scalar]] = [:]
        for scalar in text.unicodeScalars {
            let properties = UnicodeNormalization.properties(of: scalar.value)
            if properties & (compatible ? UnicodeNormalization.Property.notNFKD : UnicodeNormalization.Property.notNFD) == 0 {
                edits.append((scalar, 0)); continue
            }
            let decomposition: [Unicode.Scalar]
            if let cached = decompositions[scalar] { decomposition = cached }
            else {
                let value = compatible ? NFKDNormalizer.apply(String(scalar)) : NFDNormalizer.apply(String(scalar))
                decomposition = Array(value.unicodeScalars)
                decompositions[scalar] = decomposition
            }
            for (i, value) in decomposition.enumerated() { edits.append((value, i == 0 ? 0 : 1)) }
        }
        // Stable canonical ordering within each non-starter run.
        var start = 0
        for i in 0...edits.count {
            if i == edits.count || TokenizerUnicode.combiningClass(edits[i].0) == 0 {
                edits[start..<i].sort { TokenizerUnicode.combiningClass($0.0) < TokenizerUnicode.combiningClass($1.0) }
                start = i + 1
            }
        }
        if compose {
            var output: [(Unicode.Scalar, Int)] = []
            var compositions: [UInt64: UInt32] = [:]
            var starter: Int?
            var lastClass: UInt8 = 0
            for edit in edits {
                let cls = TokenizerUnicode.combiningClass(edit.0)
                if let index = starter, lastClass == 0 || lastClass < cls,
                   UnicodeNormalization.properties(of: edit.0.value) & UnicodeNormalization.Property.notNFC != 0 {
                    let key = UInt64(output[index].0.value) << 32 | UInt64(edit.0.value)
                    let value: UInt32
                    if let cached = compositions[key] { value = cached }
                    else {
                        let pair = NFCNormalizer.apply(String(output[index].0) + String(edit.0))
                        value = pair.unicodeScalars.count == 1 ? pair.unicodeScalars.first!.value : UInt32.max
                        compositions[key] = value
                    }
                    if let scalar = Unicode.Scalar(value) {
                        output[index] = (scalar, output[index].1 + edit.1 - 1)
                        continue
                    }
                }
                if cls == 0 { starter = output.count }
                output.append(edit)
                lastClass = cls
            }
            edits = output
        }
        return try transformed(edits)
    }
}
