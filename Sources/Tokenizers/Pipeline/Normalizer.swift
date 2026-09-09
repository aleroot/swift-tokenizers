// Text normalization applied before pre-tokenization. Every built-in normalizer works on raw
// UTF-8 (`normalize(_:into:scratch:)`); the public `normalize(text:)` is derived from it.

import Foundation

/// Text normalization applied before pre-tokenization (lowercasing, Unicode normalization,
/// accent stripping, …).
public protocol Normalizer: Sendable {
    func normalize(text: String) -> String
    func callAsFunction(text: String) -> String
    /// Creates the normalizer from its `tokenizer.json` entry.
    /// - Throws: ``TokenizerError/invalidConfiguration(_:)`` when required fields are missing.
    init(config: Config) throws
}

public extension Normalizer {
    func callAsFunction(text: String) -> String { normalize(text: text) }
}

/// Byte-level contract implemented by every built-in normalizer.
protocol ByteNormalizer: Normalizer {
    /// Appends the normalized form of `bytes` (well-formed UTF-8) to `output`.
    /// - Parameter scratch: buffers for intermediate results; never aliases `output`.
    func normalize(_ bytes: UnsafeBufferPointer<UInt8>, into output: inout [UInt8], scratch: ScratchBuffers)

    /// `true` when normalizing `bytes` is guaranteed to return them unchanged, letting callers
    /// skip the copy. Must be cheap (a linear scan at most); `false` is always a safe answer.
    func isIdentity(on bytes: UnsafeBufferPointer<UInt8>) -> Bool
}

extension ByteNormalizer {
    func isIdentity(on bytes: UnsafeBufferPointer<UInt8>) -> Bool { false }

    func normalize(text: String) -> String {
        var copy = text
        var output: [UInt8] = []
        let scratch = ScratchBuffers()
        copy.withUTF8 { normalize($0, into: &output, scratch: scratch) }
        return String(decoding: output, as: UTF8.self)
    }
}

enum NormalizerType: String {
    case Sequence
    case Prepend
    case Replace
    case Lowercase
    case NFD
    case NFC
    case NFKD
    case NFKC
    case Bert
    case BertNormalizer
    case Precompiled
    case StripAccents
    case Strip
    case Unknown = ""
}

struct NormalizerFactory {
    static func fromConfig(config: Config?) throws -> (any ByteNormalizer)? {
        guard let config, let typeName = config.type.string() else { return nil }
        switch NormalizerType(rawValue: typeName) {
        case .Sequence: return try NormalizerSequence(config: config)
        case .Prepend: return PrependNormalizer(config: config)
        case .Replace: return try ReplaceNormalizer(config: config)
        case .Lowercase: return LowercaseNormalizer(config: config)
        case .NFD: return NFDNormalizer(config: config)
        case .NFC: return NFCNormalizer(config: config)
        case .NFKD: return NFKDNormalizer(config: config)
        case .NFKC: return NFKCNormalizer(config: config)
        case .Bert, .BertNormalizer: return BertNormalizer(config: config)
        case .Precompiled: return try PrecompiledNormalizer(config: config)
        case .StripAccents: return StripAccentsNormalizer(config: config)
        case .Strip: return StripNormalizer(config: config)
        default: throw TokenizerError.unsupportedComponent("normalizer `\(typeName)`")
        }
    }
}

final class NormalizerSequence: ByteNormalizer {
    let normalizers: [any ByteNormalizer]

    required init(config: Config) throws {
        let configs = try require(config.normalizers.array(), "Sequence normalizer", field: "normalizers")
        normalizers = try configs.compactMap { try NormalizerFactory.fromConfig(config: $0) }
    }

    func isIdentity(on bytes: UnsafeBufferPointer<UInt8>) -> Bool {
        normalizers.allSatisfy { $0.isIdentity(on: bytes) }
    }

    func normalize(_ bytes: UnsafeBufferPointer<UInt8>, into output: inout [UInt8], scratch: ScratchBuffers) {
        // Stages that would not change the text are skipped; the rest ping-pong between two
        // scratch buffers.
        var index = 0
        while index < normalizers.count, normalizers[index].isIdentity(on: bytes) { index += 1 }
        guard index < normalizers.count else {
            output.append(contentsOf: bytes)
            return
        }
        var current = scratch.take()
        var next = scratch.take()
        defer {
            scratch.recycle(current)
            scratch.recycle(next)
        }
        normalizers[index].normalize(bytes, into: &current, scratch: scratch)
        index += 1
        while index < normalizers.count {
            let stage = normalizers[index]
            index += 1
            if current.withUnsafeBufferPointer({ stage.isIdentity(on: $0) }) { continue }
            next.removeAll(keepingCapacity: true)
            current.withUnsafeBufferPointer { stage.normalize($0, into: &next, scratch: scratch) }
            swap(&current, &next)
        }
        output.append(contentsOf: current)
    }
}

final class PrependNormalizer: ByteNormalizer {
    let prepend: String
    private let prependBytes: [UInt8]

    required init(config: Config) {
        prepend = config.prepend.string(or: "")
        prependBytes = Array(prepend.utf8)
    }

    func isIdentity(on bytes: UnsafeBufferPointer<UInt8>) -> Bool { bytes.isEmpty || prependBytes.isEmpty }

    func normalize(_ bytes: UnsafeBufferPointer<UInt8>, into output: inout [UInt8], scratch: ScratchBuffers) {
        // `NormalizedString::prepend` is a no-op on empty input.
        if !bytes.isEmpty { output.append(contentsOf: prependBytes) }
        output.append(contentsOf: bytes)
    }
}

final class ReplaceNormalizer: ByteNormalizer {
    let pattern: StringReplacePattern?

    required init(config: Config) throws {
        pattern = try StringReplacePattern.from(config: config)
    }

    func isIdentity(on bytes: UnsafeBufferPointer<UInt8>) -> Bool { pattern?.matches(in: bytes) == false }

    func normalize(_ bytes: UnsafeBufferPointer<UInt8>, into output: inout [UInt8], scratch: ScratchBuffers) {
        guard let pattern else {
            output.append(contentsOf: bytes)
            return
        }
        pattern.replace(bytes, into: &output)
    }
}

/// Per-scalar full lowercase mapping (`char::to_lowercase` in `tokenizers`, `String.lowercased()`
/// in Swift: no final-sigma context).
final class LowercaseNormalizer: ByteNormalizer {
    required init(config: Config) {}

    func isIdentity(on bytes: UnsafeBufferPointer<UInt8>) -> Bool {
        ByteKernels.isASCII(bytes) && !ByteKernels.containsUppercase(bytes)
    }

    func normalize(_ bytes: UnsafeBufferPointer<UInt8>, into output: inout [UInt8], scratch: ScratchBuffers) {
        let mark = output.count
        if Self.appendLowercased(bytes, to: &output) { return }
        output.removeSubrange(mark...)
        ASCII.append(String(decoding: bytes, as: UTF8.self).lowercased(), to: &output)
    }

    /// Appends the lowercase form of `bytes` using the SIMD ASCII kernel and the BMP case
    /// table. Returns `false` (with `output` in an unspecified state past its previous count)
    /// when a scalar's mapping is not a stable single BMP scalar.
    static func appendLowercased(_ bytes: UnsafeBufferPointer<UInt8>, to output: inout [UInt8]) -> Bool {
        typealias Property = UnicodeNormalization.Property
        let n = bytes.count
        output.reserveCapacity(output.count + n)
        var i = 0
        while i < n {
            let j = ByteKernels.firstNonASCII(bytes, from: i)
            if j > i {
                let base = output.count
                output.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[i..<j]))
                output.withUnsafeMutableBufferPointer {
                    ByteKernels.lowercaseASCII(UnsafeMutableBufferPointer(rebasing: $0[base...]))
                }
                i = j
                if i >= n { break }
            }
            let (value, width) = UTF8Cursor.decode(bytes, at: i)
            let properties = UnicodeNormalization.properties(of: value)
            if properties & (Property.lowercaseComplex | Property.requiresRuntime) != 0 { return false }
            if properties & Property.lowercaseMapped != 0 {
                UTF8Cursor.encode(UnicodeNormalization.lowercase(value), into: &output)
            } else {
                output.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[i..<i + width]))
            }
            i += width
        }
        return true
    }
}

/// A Unicode normalization form. Text that the quick check proves to be in the form already
/// (all everyday text) is copied; Foundation normalizes the rest.
protocol UnicodeFormNormalizer: ByteNormalizer {
    static var transform: StringTransform { get }
    /// ``UnicodeNormalization/Property`` flags of scalars that are not (certainly) in the form.
    static var unstable: UInt16 { get }
}

extension UnicodeFormNormalizer {
    static func apply(_ text: String) -> String {
        // NSString's precomposed/decomposed properties use a different implementation
        // with known conformance failures (including Hangul composition). These standard
        // ICU transforms are available through Foundation's public API on every target.
        text.applyingTransform(transform, reverse: false)!
    }

    func isIdentity(on bytes: UnsafeBufferPointer<UInt8>) -> Bool {
        UnicodeNormalization.isNormalized(bytes, mask: Self.unstable)
    }

    func normalize(_ bytes: UnsafeBufferPointer<UInt8>, into output: inout [UInt8], scratch: ScratchBuffers) {
        if isIdentity(on: bytes) {
            output.append(contentsOf: bytes)
        } else {
            ASCII.append(Self.apply(String(decoding: bytes, as: UTF8.self)), to: &output)
        }
    }
}

final class NFDNormalizer: UnicodeFormNormalizer {
    required init(config: Config) {}
    static let unstable = UnicodeNormalization.Property.notNFD
    static let transform = StringTransform("NFD")
}

final class NFCNormalizer: UnicodeFormNormalizer {
    required init(config: Config) {}
    static let unstable = UnicodeNormalization.Property.notNFC
    static let transform = StringTransform("NFC")
}

final class NFKDNormalizer: UnicodeFormNormalizer {
    required init(config: Config) {}
    static let unstable = UnicodeNormalization.Property.notNFKD
    static let transform = StringTransform("NFKD")
}

final class NFKCNormalizer: UnicodeFormNormalizer {
    required init(config: Config) {}
    static let unstable = UnicodeNormalization.Property.notNFKC
    static let transform = StringTransform("NFKC")
}

final class BertNormalizer: ByteNormalizer {
    let shouldCleanText: Bool
    let shouldHandleChineseChars: Bool
    let shouldStripAccents: Bool
    let shouldLowercase: Bool

    required init(config: Config) {
        shouldCleanText = config.cleanText.boolean(or: true)
        shouldHandleChineseChars = config.handleChineseChars.boolean(or: true)
        shouldLowercase = config.lowercase.boolean(or: true)
        shouldStripAccents = config.stripAccents.boolean(or: shouldLowercase)
    }

    func isIdentity(on bytes: UnsafeBufferPointer<UInt8>) -> Bool {
        guard ByteKernels.isASCII(bytes) else { return false }
        if shouldCleanText, ByteKernels.containsControl(bytes) { return false }
        if shouldLowercase, ByteKernels.containsUppercase(bytes) { return false }
        return true
    }

    func normalize(_ bytes: UnsafeBufferPointer<UInt8>, into output: inout [UInt8], scratch: ScratchBuffers) {
        // Every step (cleaning, CJK padding, NFD + mark removal, lowercasing) is context-free
        // across an ASCII / non-ASCII boundary, so ASCII runs take the SIMD kernels and
        // non-ASCII runs are mapped scalar by scalar through the BMP tables; only scalars the
        // tables do not cover send their run through Foundation.
        output.reserveCapacity(output.count + bytes.count)
        let n = bytes.count
        var i = 0
        while i < n {
            let j = ByteKernels.firstNonASCII(bytes, from: i)
            if j > i {
                normalizeASCII(UnsafeBufferPointer(rebasing: bytes[i..<j]), into: &output)
                i = j
                if i >= n { break }
            }
            let k = ByteKernels.firstASCII(bytes, from: i)
            let run = UnsafeBufferPointer(rebasing: bytes[i..<k])
            let mark = output.count
            if !appendNormalizedBMP(run, to: &output) {
                output.removeSubrange(mark...)
                normalizeWithFoundation(run, into: &output, scratch: scratch)
            }
            i = k
        }
    }

    /// ASCII has no CJK ideographs or combining marks, so only cleaning and lowercasing apply.
    @inline(__always)
    private func normalizeASCII(_ bytes: UnsafeBufferPointer<UInt8>, into output: inout [UInt8]) {
        if shouldCleanText, ByteKernels.containsControl(bytes) {
            for var byte in bytes {
                switch byte {
                case 0x09, 0x0A, 0x0D: byte = 0x20
                case 0x00...0x1F, 0x7F: continue
                default: break
                }
                if shouldLowercase, byte >= 0x41, byte <= 0x5A { byte |= 0x20 }
                output.append(byte)
            }
            return
        }
        let base = output.count
        output.append(contentsOf: bytes)
        if shouldLowercase {
            output.withUnsafeMutableBufferPointer {
                ByteKernels.lowercaseASCII(UnsafeMutableBufferPointer(rebasing: $0[base...]))
            }
        }
    }

    /// Table-driven form of ``normalizeWithFoundation(_:into:scratch:)`` for a non-ASCII run:
    /// canonical decomposition, `Mn` removal and simple lowercase mapping per scalar. Returns
    /// `false` (with `output` in an unspecified state past its previous count) when a scalar
    /// is outside the stable tables: a version-dependent or supplementary scalar, a non-`Mn` scalar
    /// with a combining class (canonical reordering could interleave it with marks) or a
    /// multi-scalar lowercase mapping.
    private func appendNormalizedBMP(_ bytes: UnsafeBufferPointer<UInt8>, to output: inout [UInt8]) -> Bool {
        typealias Property = UnicodeNormalization.Property
        var i = 0
        while i < bytes.count {
            let (value, width) = UTF8Cursor.decode(bytes, at: i)
            i += width
            let properties = UnicodeNormalization.properties(of: value)
            if properties & Property.requiresRuntime != 0 { return false }
            if shouldCleanText {
                if value == 0xFFFD || Self.isControl(value) { continue }
                if Self.isWhitespace(value) {
                    output.append(0x20)
                    continue
                }
            }
            let chinese = shouldHandleChineseChars && Self.isCJKUnifiedIdeograph(value)
            if chinese { output.append(0x20) }
            if shouldStripAccents, properties & Property.canonicalDecomposition != 0 {
                var accepted = true
                UnicodeNormalization.decompose(value) { part in
                    if accepted {
                        accepted = appendStrippedLowercased(
                            part, UnicodeNormalization.properties(of: part), to: &output)
                    }
                }
                if !accepted { return false }
            } else if !appendStrippedLowercased(value, properties, to: &output) {
                return false
            }
            if chinese { output.append(0x20) }
        }
        return true
    }

    /// Drops `value` if it is a nonspacing mark (when stripping accents), otherwise appends its
    /// (lowercased) UTF-8. Returns `false` when the tables cannot decide.
    @inline(__always)
    private func appendStrippedLowercased(_ value: UInt32, _ properties: UInt16, to output: inout [UInt8]) -> Bool {
        typealias Property = UnicodeNormalization.Property
        if properties & Property.requiresRuntime != 0 { return false }
        if shouldStripAccents {
            if properties & Property.nonspacingMark != 0 { return true }
            if properties & Property.combiningClassMask != 0 { return false }
        }
        var mapped = value
        if shouldLowercase {
            if properties & Property.lowercaseComplex != 0 { return false }
            if properties & Property.lowercaseMapped != 0 { mapped = UnicodeNormalization.lowercase(value) }
        }
        UTF8Cursor.encode(mapped, into: &output)
        return true
    }

    /// Reference implementation for runs the tables do not cover: Foundation NFD, mark
    /// removal and `String.lowercased()`.
    func normalizeWithFoundation(
        _ bytes: UnsafeBufferPointer<UInt8>, into output: inout [UInt8], scratch: ScratchBuffers
    ) {
        var run = scratch.take()
        defer { scratch.recycle(run) }
        if shouldCleanText || shouldHandleChineseChars {
            var i = 0
            while i < bytes.count {
                let (value, width) = UTF8Cursor.decode(bytes, at: i)
                defer { i += width }
                if shouldCleanText {
                    if value == 0xFFFD || Self.isControl(value) { continue }
                    if Self.isWhitespace(value) {
                        run.append(0x20)
                        continue
                    }
                }
                let chinese = shouldHandleChineseChars && Self.isCJKUnifiedIdeograph(value)
                if chinese { run.append(0x20) }
                run.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[i..<i + width]))
                if chinese { run.append(0x20) }
            }
        } else {
            run.append(contentsOf: bytes)
        }

        var text = String(decoding: run, as: UTF8.self)
        if shouldStripAccents { text = Self.stripAccents(text) }
        if shouldLowercase { text = text.lowercased() }
        ASCII.append(text, to: &output)
    }

    /// `\t` `\n` `\r` or `Zs`.
    @inline(__always)
    static func isWhitespace(_ value: UInt32) -> Bool {
        if value < 0x80 { return value == 0x20 || value == 0x09 || value == 0x0A || value == 0x0D }
        return ScalarClassifier.extraFlags(value: value) & ScalarExtraFlags.spaceSeparator != 0
    }

    /// Cc/Cf/Cs/Co except `\t` `\n` `\r` (unassigned code points are left untouched, as in `tokenizers`).
    @inline(__always)
    static func isControl(_ value: UInt32) -> Bool {
        if value < 0x80 { return (value < 0x20 && value != 0x09 && value != 0x0A && value != 0x0D) || value == 0x7F }
        return ScalarClassifier.extraFlags(value: value) & ScalarExtraFlags.control != 0
    }

    /// https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    @inline(__always)
    static func isCJKUnifiedIdeograph(_ value: UInt32) -> Bool {
        (value >= 0x4E00 && value <= 0x9FFF)
            || (value >= 0x3400 && value <= 0x4DBF)
            || (value >= 0x20000 && value <= 0x2A6DF)
            || (value >= 0x2A700 && value <= 0x2B73F)
            || (value >= 0x2B740 && value <= 0x2B81F)
            || (value >= 0x2B820 && value <= 0x2CEAF)
            || (value >= 0xF900 && value <= 0xFAFF)
            || (value >= 0x2F800 && value <= 0x2FA1F)
    }

    /// NFD-decompose then drop every nonspacing mark (general category Mn), matching HF's
    /// `_run_strip_accents`. The output stays decomposed, as in the reference.
    static func stripAccents(_ bytes: UnsafeBufferPointer<UInt8>, into output: inout [UInt8]) {
        if ASCII.isASCII(bytes) {
            output.append(contentsOf: bytes)
            return
        }
        var decomposed = NFDNormalizer.apply(String(decoding: bytes, as: UTF8.self))
        decomposed.withUTF8 { decomposed in
            output.reserveCapacity(output.count + decomposed.count)
            var i = 0
            while i < decomposed.count {
                let b0 = decomposed[i]
                if b0 < 0x80 {
                    output.append(b0)
                    i += 1
                    continue
                }
                let (value, width) = UTF8Cursor.decode(decomposed, at: i)
                if ScalarClassifier.extraFlags(value: value) & ScalarExtraFlags.nonspacingMark == 0 {
                    output.append(contentsOf: UnsafeBufferPointer(rebasing: decomposed[i..<i + width]))
                }
                i += width
            }
        }
    }

    /// String form of ``stripAccents(_:into:)`` (used by the legacy BERT basic tokenizer).
    static func stripAccents(_ text: String) -> String {
        var copy = text
        var output: [UInt8] = []
        copy.withUTF8 { stripAccents($0, into: &output) }
        return String(decoding: output, as: UTF8.self)
    }
}

extension Unicode.Scalar {
    var isCJKUnifiedIdeograph: Bool { BertNormalizer.isCJKUnifiedIdeograph(value) }
}

/// SentencePiece's serialized Darts map, following spm_precompiled 0.1.3.
/// The map is model data: substituting Foundation NFKC changes token IDs.
final class PrecompiledNormalizer: ByteNormalizer {
    private let trie: [UInt32]
    /// NUL-terminated replacement strings, back to back.
    private let replacements: [UInt8]
    /// Per ASCII byte: offset of its replacement in `replacements`, or `-1` when it maps to itself.
    private let asciiReplacementOffsets: [Int32]
    private let crlfReplacementOffset: Int32
    /// `true` when no ASCII byte (or CRLF) maps to anything: ASCII chunks are then copied verbatim.
    private let asciiIsIdentity: Bool
    /// `true` when only C0 controls / DEL map to something (SentencePiece's `nmt_nfkc` maps):
    /// an ASCII run without control bytes is then copied verbatim after one SIMD scan.
    private let asciiReplacesOnlyControls: Bool

    required init(config: Config) throws {
        guard let encoded = config.precompiledCharsmap.string(), let data = Data(base64Encoded: encoded),
            data.count >= 8
        else {
            throw TokenizerError.invalidConfiguration("Precompiled normalizer requires a valid base64 charsmap")
        }
        let bytes = Array(data)
        func uint32(at i: Int) -> UInt32 {
            UInt32(bytes[i]) | UInt32(bytes[i + 1]) << 8 | UInt32(bytes[i + 2]) << 16 | UInt32(bytes[i + 3]) << 24
        }
        let size = Int(uint32(at: 0))
        guard size >= 4, size % 4 == 0, size <= bytes.count - 4 else {
            throw TokenizerError.invalidConfiguration("Invalid Precompiled trie size")
        }
        trie = stride(from: 4, to: size + 4, by: 4).map { uint32(at: $0) }
        replacements = Array(bytes[(size + 4)...])
        guard replacements.last == 0, String(bytes: replacements, encoding: .utf8) != nil else {
            throw TokenizerError.invalidConfiguration("Invalid Precompiled replacement table")
        }
        guard Self.offset(trie[0]) < trie.count else {
            throw TokenizerError.invalidConfiguration("Invalid Precompiled root offset")
        }
        let trie = self.trie
        let replacements = self.replacements
        asciiReplacementOffsets = (0..<128).map { value in
            Self.replacementOffset(for: [UInt8(value)], trie: trie, replacements: replacements) ?? -1
        }
        crlfReplacementOffset =
            Self.replacementOffset(for: [UInt8(13), 10], trie: trie, replacements: replacements) ?? -1
        asciiIsIdentity = asciiReplacementOffsets.allSatisfy { $0 < 0 } && crlfReplacementOffset < 0
        asciiReplacesOnlyControls = asciiReplacementOffsets.enumerated().allSatisfy { byte, offset in
            offset < 0 || byte < 0x20 || byte == 0x7F
        }
    }

    func isIdentity(on bytes: UnsafeBufferPointer<UInt8>) -> Bool {
        guard ByteKernels.isASCII(bytes) else { return false }
        return asciiIsIdentity(on: bytes)
    }

    /// Whether the ASCII-only `bytes` map to themselves.
    @inline(__always)
    private func asciiIsIdentity(on bytes: UnsafeBufferPointer<UInt8>) -> Bool {
        if asciiIsIdentity { return true }
        if asciiReplacesOnlyControls { return !ByteKernels.containsControl(bytes) }
        for byte in bytes where asciiReplacementOffsets[Int(byte)] >= 0 { return false }
        if crlfReplacementOffset >= 0, bytes.count > 1 {
            for i in 0..<(bytes.count - 1) where bytes[i] == 13 && bytes[i + 1] == 10 { return false }
        }
        return true
    }

    /// Appends the NUL-terminated replacement starting at `offset`.
    @inline(__always)
    private func appendReplacement(at offset: Int32, to output: inout [UInt8]) {
        var i = Int(offset)
        while replacements[i] != 0 {
            output.append(replacements[i])
            i += 1
        }
    }

    @inline(__always) private static func offset(_ unit: UInt32) -> Int {
        Int(unit >> 10) << Int((unit & (1 << 9)) >> 6)
    }

    /// Offset in `replacements` of the first prefix match, as the reference does (not the
    /// longest match).
    private static func replacementOffset<C: Collection>(
        for bytes: C, trie: [UInt32], replacements: [UInt8]
    ) -> Int32? where C.Element == UInt8 {
        var node = Self.offset(trie[0])
        for byte in bytes {
            if byte == 0 { break }
            node ^= Int(byte)
            guard node < trie.count else { return nil }
            let unit = trie[node]
            guard unit & 0x800000ff == UInt32(byte) else { return nil }
            node ^= Self.offset(unit)
            if unit & 0x100 != 0 {
                guard node < trie.count else { return nil }
                let start = Int(trie[node] & 0x7fffffff)
                guard start < replacements.count else { return nil }
                return Int32(start)
            }
        }
        return nil
    }

    private func replacementOffset<C: Collection>(for bytes: C) -> Int32? where C.Element == UInt8 {
        Self.replacementOffset(for: bytes, trie: trie, replacements: replacements)
    }

    func normalize(_ bytes: UnsafeBufferPointer<UInt8>, into output: inout [UInt8], scratch: ScratchBuffers) {
        output.reserveCapacity(output.count + bytes.count)
        let n = bytes.count
        var i = 0
        while i < n {
            // ASCII runs go through the per-byte table without constructing graphemes.
            if bytes[i] < 0x80 {
                var j = ByteKernels.firstNonASCII(bytes, from: i + 1)
                // Keep a trailing ASCII scalar for the non-ASCII run if it may start a grapheme
                // that a following mark extends (e.g. `e` + U+0301 maps as one unit).
                if j < n, j > i, Self.extendsGrapheme(bytes, at: j) { j -= 1 }
                appendASCII(UnsafeBufferPointer(rebasing: bytes[i..<j]), to: &output)
                i = j
                if i >= n { break }
            }
            // A non-ASCII run (plus at most one leading ASCII scalar): the reference maps whole
            // graphemes first (when shorter than 6 bytes), then scalars.
            let j = ByteKernels.firstASCII(bytes, from: i + 1)
            let run = UnsafeBufferPointer(rebasing: bytes[i..<j])
            let mark = output.count
            if !appendClusters(run, to: &output) {
                output.removeSubrange(mark...)
                for grapheme in String(decoding: run, as: UTF8.self) {
                    appendGrapheme(grapheme, to: &output)
                }
            }
            i = j
        }
    }

    /// Whether the scalar at `i` continues the grapheme cluster started by the scalar before it.
    @inline(__always)
    private static func extendsGrapheme(_ bytes: UnsafeBufferPointer<UInt8>, at i: Int) -> Bool {
        let (value, _) = UTF8Cursor.decode(bytes, at: i)
        return ScalarClassifier.extraFlags(value: value) & ScalarExtraFlags.graphemeExtend != 0
    }

    /// Maps a run of scalars cluster by cluster without materialising a `String`.
    ///
    /// Only clusters shorter than 6 bytes are looked up whole (a prefix match then replaces
    /// the entire cluster, as in the reference), so cluster boundaries need to be exact only
    /// for short clusters: a base scalar plus the grapheme-extending scalars (`Extend`,
    /// `SpacingMark`, ZWJ) that follow it. Every other UAX #29 rule joins scalars into
    /// clusters of 6 bytes or more (Hangul jamo, regional indicators, emoji sequences, Indic
    /// conjuncts) — except `Control` (a boundary on both sides) and `Prepend`, which make the
    /// function return `false` (with `output` in an unspecified state past its previous
    /// count) so the caller can use the reference grapheme iterator.
    private func appendClusters(_ bytes: UnsafeBufferPointer<UInt8>, to output: inout [UInt8]) -> Bool {
        let n = bytes.count
        var i = 0
        while i < n {
            let (value, width) = UTF8Cursor.decode(bytes, at: i)
            if ScalarClassifier.extraFlags(value: value) & ScalarExtraFlags.graphemeControl != 0 { return false }
            var end = i + width
            while end < n, Self.extendsGrapheme(bytes, at: end) { end += UTF8Cursor.width(bytes[end]) }
            if end - i < 6 {
                if let offset = replacementOffset(for: bytes[i..<end]) {
                    appendReplacement(at: offset, to: &output)
                    i = end
                    continue
                }
                if end == i + width {
                    output.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[i..<end]))
                    i = end
                    continue
                }
            }
            while i < end {
                let scalarEnd = i + UTF8Cursor.width(bytes[i])
                if let offset = replacementOffset(for: bytes[i..<scalarEnd]) {
                    appendReplacement(at: offset, to: &output)
                } else {
                    output.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[i..<scalarEnd]))
                }
                i = scalarEnd
            }
        }
        return true
    }

    @inline(__always)
    private func appendASCII(_ bytes: UnsafeBufferPointer<UInt8>, to output: inout [UInt8]) {
        if asciiIsIdentity(on: bytes) {
            output.append(contentsOf: bytes)
            return
        }
        var i = 0
        while i < bytes.count {
            let byte = bytes[i]
            if byte == 13, crlfReplacementOffset >= 0, i + 1 < bytes.count, bytes[i + 1] == 10 {
                appendReplacement(at: crlfReplacementOffset, to: &output)
                i += 2
                continue
            }
            let offset = asciiReplacementOffsets[Int(byte)]
            if offset < 0 { output.append(byte) } else { appendReplacement(at: offset, to: &output) }
            i += 1
        }
    }

    private func appendGrapheme(_ grapheme: Character, to output: inout [UInt8]) {
        let utf8 = grapheme.utf8
        if utf8.count == 1 {
            let byte = utf8.first!
            let offset = asciiReplacementOffsets[Int(byte)]
            if offset < 0 { output.append(byte) } else { appendReplacement(at: offset, to: &output) }
        } else if utf8.count < 6, let offset = replacementOffset(for: utf8) {
            appendReplacement(at: offset, to: &output)
        } else {
            for scalar in grapheme.unicodeScalars {
                if let offset = replacementOffset(for: scalar.utf8) {
                    appendReplacement(at: offset, to: &output)
                } else {
                    output.append(contentsOf: scalar.utf8)
                }
            }
        }
    }
}

final class StripAccentsNormalizer: ByteNormalizer {
    required init(config: Config) {}

    func isIdentity(on bytes: UnsafeBufferPointer<UInt8>) -> Bool { ASCII.isASCII(bytes) }

    func normalize(_ bytes: UnsafeBufferPointer<UInt8>, into output: inout [UInt8], scratch: ScratchBuffers) {
        // Standalone StripAccents removes marks (Mn/Mc/Me) without decomposing precomposed letters.
        output.reserveCapacity(output.count + bytes.count)
        var i = 0
        while i < bytes.count {
            let b0 = bytes[i]
            if b0 < 0x80 {
                output.append(b0)
                i += 1
                continue
            }
            let (value, width) = UTF8Cursor.decode(bytes, at: i)
            if ScalarClassifier.flags(value: value) & ScalarFlags.mark == 0 {
                output.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[i..<i + width]))
            }
            i += width
        }
    }
}

final class StripNormalizer: ByteNormalizer {
    let leftStrip: Bool
    let rightStrip: Bool

    required init(config: Config) {
        leftStrip = config.stripLeft.boolean(or: true)
        rightStrip = config.stripRight.boolean(or: true)
    }

    func isIdentity(on bytes: UnsafeBufferPointer<UInt8>) -> Bool {
        trimmedRange(of: bytes) == 0..<bytes.count
    }

    func normalize(_ bytes: UnsafeBufferPointer<UInt8>, into output: inout [UInt8], scratch: ScratchBuffers) {
        output.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[trimmedRange(of: bytes)]))
    }

    /// The range that survives trimming. Rust trims Unicode scalar values
    /// (`char::is_whitespace`), never whole graphemes.
    private func trimmedRange(of bytes: UnsafeBufferPointer<UInt8>) -> Range<Int> {
        var start = 0
        var end = bytes.count
        if leftStrip {
            while start < end {
                let (value, width) = UTF8Cursor.decode(bytes, at: start)
                guard ScalarClassifier.flags(value: value) & ScalarFlags.whitespace != 0 else { break }
                start += width
            }
        }
        if rightStrip {
            while end > start {
                var scalarStart = end - 1
                while scalarStart > start, bytes[scalarStart] & 0xC0 == 0x80 { scalarStart -= 1 }
                let (value, _) = UTF8Cursor.decode(bytes, at: scalarStart)
                guard ScalarClassifier.flags(value: value) & ScalarFlags.whitespace != 0 else { break }
                end = scalarStart
            }
        }
        return start..<end
    }
}

// MARK: - Replace patterns

enum StringReplacePattern: Sendable {
    case regexp(regexp: NSRegularExpression, replacement: String)
    case string(pattern: [UInt8], replacement: [UInt8])
    /// `X{n,}` for a single ASCII character `X`: runs of at least `minimum` bytes are replaced.
    case run(byte: UInt8, minimum: Int, replacement: [UInt8])

    func replace(_ text: String) -> String {
        var copy = text
        var output: [UInt8] = []
        copy.withUTF8 { replace($0, into: &output) }
        return String(decoding: output, as: UTF8.self)
    }

    /// Appends `bytes` with every match replaced to `output`.
    func replace(_ bytes: UnsafeBufferPointer<UInt8>, into output: inout [UInt8]) {
        switch self {
        case let .regexp(regexp, replacement):
            let text = String(decoding: bytes, as: UTF8.self)
            let range = NSRange(text.startIndex..., in: text)
            ASCII.append(
                regexp.stringByReplacingMatches(in: text, options: [], range: range, withTemplate: replacement),
                to: &output)
        case let .string(pattern, replacement):
            Self.replaceLiteral(bytes, pattern: pattern, with: replacement, into: &output)
        case let .run(byte, minimum, replacement):
            var i = 0
            var cursor = 0
            let n = bytes.count
            while i < n {
                guard bytes[i] == byte else {
                    i += 1
                    continue
                }
                var j = i + 1
                while j < n, bytes[j] == byte { j += 1 }
                if j - i >= minimum {
                    output.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[cursor..<i]))
                    output.append(contentsOf: replacement)
                    cursor = j
                }
                i = j
            }
            output.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[cursor..<n]))
        }
    }

    /// Whether `bytes` contain at least one match (`false` means replacement is the identity).
    func matches(in bytes: UnsafeBufferPointer<UInt8>) -> Bool {
        switch self {
        case .regexp:
            return true  // unknown without running the engine; treat as a potential match
        case let .string(pattern, _):
            guard let first = pattern.first, pattern.count <= bytes.count, let base = bytes.baseAddress else {
                return false
            }
            var i = 0
            let n = bytes.count
            let m = pattern.count
            while i + m <= n {
                guard let hit = memchr(base + i, Int32(first), n - m + 1 - i) else { return false }
                let p = UnsafePointer<UInt8>(hit.assumingMemoryBound(to: UInt8.self)) - base
                if m == 1 || memcmp(base + p, pattern, m) == 0 { return true }
                i = p + 1
            }
            return false
        case let .run(byte, minimum, _):
            let n = bytes.count
            if minimum <= 1 { return ByteKernels.firstIndex(of: byte, in: bytes, from: 0) < n }
            // A run of `minimum` needs two adjacent occurrences; that check is one SIMD pass.
            guard ByteKernels.containsRepeat(of: byte, in: bytes) else { return false }
            if minimum == 2 { return true }
            var i = 0
            while i < n {
                guard bytes[i] == byte else {
                    i += 1
                    continue
                }
                var j = i + 1
                while j < n, bytes[j] == byte { j += 1 }
                if j - i >= minimum { return true }
                i = j
            }
            return false
        }
    }

    /// Non-overlapping left-to-right literal replacement (`str::replace` semantics).
    static func replaceLiteral(
        _ bytes: UnsafeBufferPointer<UInt8>, pattern: [UInt8], with replacement: [UInt8], into output: inout [UInt8]
    ) {
        let n = bytes.count
        let m = pattern.count
        guard m > 0, m <= n, let base = bytes.baseAddress else {
            output.append(contentsOf: bytes)
            return
        }
        let first = Int32(pattern[0])
        var cursor = 0
        var i = 0
        while i + m <= n {
            guard let hit = memchr(base + i, first, n - m + 1 - i) else { break }
            let p = UnsafePointer<UInt8>(hit.assumingMemoryBound(to: UInt8.self)) - base
            if m == 1 || memcmp(base + p, pattern, m) == 0 {
                output.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[cursor..<p]))
                output.append(contentsOf: replacement)
                cursor = p + m
                i = cursor
            } else {
                i = p + 1
            }
        }
        output.append(contentsOf: UnsafeBufferPointer(rebasing: bytes[cursor..<n]))
    }

    static func from(config: Config) throws -> StringReplacePattern? {
        guard let replacement = config.content.string() else { return nil }
        if let pattern = config.pattern.String.string() {
            return .string(pattern: Array(pattern.utf8), replacement: Array(replacement.utf8))
        }
        if let pattern = config.pattern.Regex.string() {
            // Many SentencePiece configs express a literal via a regex with a single-scalar
            // pattern such as " ". Treat trivial patterns as literals to skip the regex engine.
            if Self.isLiteral(pattern), !replacement.contains("$"), !replacement.contains("\\") {
                return .string(pattern: Array(pattern.utf8), replacement: Array(replacement.utf8))
            }
            // ` {2,}` (T5, XLM-R, …): collapse runs of a character.
            if let run = Self.runPattern(pattern), !replacement.contains("$"), !replacement.contains("\\") {
                return .run(byte: run.byte, minimum: run.minimum, replacement: Array(replacement.utf8))
            }
            return .regexp(regexp: try compileRegex(pattern, component: "Replace normalizer"), replacement: replacement)
        }
        return nil
    }

    private static let metacharacters: Set<Character> = [
        "\\", "^", "$", ".", "|", "?", "*", "+", "(", ")", "[", "]", "{", "}",
    ]

    private static func isLiteral(_ pattern: String) -> Bool {
        !pattern.isEmpty && !pattern.contains(where: { metacharacters.contains($0) })
    }

    /// Recognises `X{n,}` where `X` is a single non-metacharacter ASCII byte.
    private static func runPattern(_ pattern: String) -> (byte: UInt8, minimum: Int)? {
        let bytes = Array(pattern.utf8)
        guard bytes.count >= 5, bytes[0] < 0x80, !metacharacters.contains(Character(UnicodeScalar(bytes[0]))),
            bytes[1] == UInt8(ascii: "{"), bytes[bytes.count - 2] == UInt8(ascii: ","),
            bytes[bytes.count - 1] == UInt8(ascii: "}")
        else { return nil }
        let digits = bytes[2..<(bytes.count - 2)]
        guard !digits.isEmpty, digits.allSatisfy({ $0 >= 0x30 && $0 <= 0x39 }),
            let minimum = Int(String(decoding: digits, as: UTF8.self)), minimum >= 1
        else { return nil }
        return (bytes[0], minimum)
    }
}
