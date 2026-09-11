import Foundation

/// The model's view of the shared vocabulary excludes tokens added to the pipeline.
/// Dense model IDs need only a bound check; sparse vocabularies use one bit per ID.
/// Token bytes and the hash table remain shared with the public vocabulary.
struct ModelVocabulary: Sendable {
    let vocabulary: Vocabulary
    private let count: Int32
    private let present: [UInt64]

    init(_ vocabulary: Vocabulary, config: Config) throws {
        self.vocabulary = vocabulary
        var maxId: Int32 = -1
        var bits: [UInt64] = []
        func mark(_ ids: UnsafeBufferPointer<Int32>) {
            for id in ids where id > maxId { maxId = id }
            bits = [UInt64](repeating: 0, count: (Int(maxId) + 64) / 64)
            for id in ids { bits[Int(id) >> 6] |= 1 << UInt64(id & 63) }
        }
        if let packed = config.asPackedStringMap() {
            packed.ids.withUnsafeBufferPointer(mark)
        } else if let values = config.dictionary() {
            let ids = try values.values.map {
                guard let id = $0.integer(), id >= 0, id < vocabulary.count else {
                    throw TokenizerError.malformedVocab
                }
                return Int32(id)
            }
            ids.withUnsafeBufferPointer(mark)
        } else {
            throw TokenizerError.missingVocab
        }
        count = maxId + 1
        let populated = bits.reduce(0) { $0 + $1.nonzeroBitCount }
        present = populated == Int(count) ? [] : bits
    }

    /// Legacy direct model construction has no separate added vocabulary.
    init(_ vocabulary: Vocabulary) {
        self.vocabulary = vocabulary
        count = Int32(vocabulary.count)
        present = []
    }

    @inline(__always)
    func contains(_ id: Int32) -> Bool {
        id >= 0 && id < count && (present.isEmpty || present[Int(id) >> 6] & (1 << UInt64(id & 63)) != 0)
    }

    @inline(__always)
    func id(of bytes: UnsafeBufferPointer<UInt8>) -> Int32 {
        let id = vocabulary.id(of: bytes)
        return contains(id) ? id : -1
    }

    func id(of token: String) -> Int? {
        var copy = token
        let id = copy.withUTF8 { self.id(of: $0) }
        return id >= 0 ? Int(id) : nil
    }

    func id(ofScalar scalar: Unicode.Scalar) -> Int32 {
        let id = vocabulary.id(ofScalar: scalar)
        return contains(id) ? id : -1
    }
}
