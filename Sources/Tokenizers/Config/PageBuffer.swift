// Table storage whose pages come straight from the VM system and go back to it when the
// storage is released. The large tables parsed out of `tokenizer.json` and the scratch the
// tokenizer models build from them live here: they are consumed while the tokenizer is built
// and then dropped, and memory freed through `munmap` leaves the process footprint
// immediately, whereas blocks returned to `malloc` can stay resident in the allocator's caches.

import Foundation

/// A fixed block of anonymous pages holding trivial elements.
///
/// Pages are zero-filled and cost nothing until they are first touched, so a block may be
/// sized to a generous upper bound. `@unchecked Sendable`: the block is a plain allocation
/// and carries no mutable state of its own.
final class PageBlock<Element>: @unchecked Sendable {
    let base: UnsafeMutablePointer<Element>
    /// Elements the block can hold, at least the requested count.
    let count: Int
    /// Zero when the mapping failed and the allocator supplied the block instead.
    private let mappedBytes: Int

    init(count requested: Int) {
        let elements = max(requested, 1)
        let stride = MemoryLayout<Element>.stride
        let page = Int(getpagesize())
        let bytes = (elements * stride + page - 1) / page * page
        if let raw = mmap(nil, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0), raw != MAP_FAILED {
            base = raw.bindMemory(to: Element.self, capacity: bytes / stride)
            count = bytes / stride
            mappedBytes = bytes
            return
        }
        // Address space exhausted: fall back to the allocator, zeroed to match the mapping.
        base = .allocate(capacity: elements)
        count = elements
        mappedBytes = 0
        memset(base, 0, elements * stride)
    }

    deinit {
        if mappedBytes == 0 {
            base.deallocate()
        } else {
            munmap(UnsafeMutableRawPointer(base), mappedBytes)
        }
    }
}

/// A contiguous, append-only buffer of trivial elements backed by anonymous pages.
///
/// Untouched pages cost nothing, so callers reserve an upper bound up front (the tables never
/// outgrow the JSON they are parsed from) and appends never copy. Growth is still handled for
/// safety.
///
/// `@unchecked Sendable`: a buffer is filled by the parser that creates it and is read-only
/// once it is stored in a `Config`.
final class PageBuffer<Element>: @unchecked Sendable {
    private(set) var count = 0
    private var storage: PageBlock<Element>
    /// Mirrors of the block's fields so appending touches no other object.
    private var base: UnsafeMutablePointer<Element>
    private var capacity: Int

    /// Creates a buffer able to hold `capacity` elements without reallocating.
    init(capacity: Int) {
        storage = PageBlock(count: capacity)
        base = storage.base
        self.capacity = storage.count
    }

    /// Copies an array into a page-backed buffer.
    convenience init(_ elements: [Element]) {
        self.init(capacity: elements.count)
        elements.withUnsafeBufferPointer { source in
            guard let address = source.baseAddress, !source.isEmpty else { return }
            base.initialize(from: address, count: source.count)
        }
        count = elements.count
    }

    @inline(__always)
    subscript(index: Int) -> Element {
        precondition(index >= 0 && index < count, "index out of range")
        return base[index]
    }

    @inline(__always)
    func append(_ element: Element) {
        if count == capacity { grow(toHold: count + 1) }
        base[count] = element
        count += 1
    }

    func append(contentsOf source: UnsafeBufferPointer<Element>) {
        guard let address = source.baseAddress, !source.isEmpty else { return }
        if count + source.count > capacity { grow(toHold: count + source.count) }
        (base + count).update(from: address, count: source.count)
        count += source.count
    }

    /// Reserves room for `n` more elements and returns the write position, leaving `count`
    /// unchanged until ``commit(_:)``.
    @inline(__always)
    func reserveTail(_ n: Int) -> UnsafeMutablePointer<Element> {
        if count + n > capacity { grow(toHold: count + n) }
        return base + count
    }

    @inline(__always)
    func commit(_ n: Int) { count += n }

    /// Runs `body` with a zero-filled block of `count` elements whose pages leave the process
    /// footprint as soon as the call returns. Use for load-time tables that are large enough
    /// that the allocator would hold on to them afterwards.
    static func withScratch<R>(count: Int, _ body: (UnsafeMutablePointer<Element>) throws -> R) rethrows -> R {
        let buffer = PageBuffer<Element>(capacity: max(count, 1))
        defer { withExtendedLifetime(buffer) {} }
        return try body(buffer.reserveTail(count))
    }

    /// Drops the last `n` elements.
    func removeLast(_ n: Int) {
        precondition(n >= 0 && n <= count)
        count -= n
    }

    func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Element>) throws -> R) rethrows -> R {
        try body(UnsafeBufferPointer(start: base, count: count))
    }

    func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Element>) throws -> R) rethrows -> R {
        try body(UnsafeMutableBufferPointer(start: base, count: count))
    }

    /// The elements as an array (a copy).
    var elements: [Element] {
        withUnsafeBufferPointer { Array($0) }
    }

    // MARK: - Pages

    private func grow(toHold needed: Int) {
        var target = max(capacity * 2, 16)
        while target < needed { target *= 2 }
        let grown = PageBlock<Element>(count: target)
        if count > 0 { grown.base.update(from: base, count: count) }
        storage = grown
        base = grown.base
        capacity = grown.count
    }
}

extension PageBuffer where Element: Equatable {
    static func == (lhs: PageBuffer, rhs: PageBuffer) -> Bool {
        if lhs === rhs { return true }
        guard lhs.count == rhs.count else { return false }
        return lhs.withUnsafeBufferPointer { a in rhs.withUnsafeBufferPointer { b in a.elementsEqual(b) } }
    }
}

extension PageBuffer where Element: Hashable {
    func hash(into hasher: inout Hasher) {
        hasher.combine(count)
        withUnsafeBufferPointer { for element in $0 { hasher.combine(element) } }
    }
}

extension PageBuffer where Element == UInt8 {
    /// Appends a token spelling. Vocabulary and merge entries are a handful of bytes long,
    /// where a call into `memmove` costs several times the copy itself; overlapping
    /// fixed-width loads cover every short length without a loop.
    @inline(__always)
    func append(bytes source: UnsafeBufferPointer<UInt8>) {
        let n = source.count
        guard n > 0, let read = source.baseAddress else { return }
        let write = reserveTail(n)
        switch n {
        case 16...:
            write.update(from: read, count: n)
        case 8...:
            let head = UnsafeRawPointer(read).loadUnaligned(as: UInt64.self)
            let tail = UnsafeRawPointer(read + n - 8).loadUnaligned(as: UInt64.self)
            UnsafeMutableRawPointer(write).storeBytes(of: head, as: UInt64.self)
            UnsafeMutableRawPointer(write + n - 8).storeBytes(of: tail, as: UInt64.self)
        case 4...:
            let head = UnsafeRawPointer(read).loadUnaligned(as: UInt32.self)
            let tail = UnsafeRawPointer(read + n - 4).loadUnaligned(as: UInt32.self)
            UnsafeMutableRawPointer(write).storeBytes(of: head, as: UInt32.self)
            UnsafeMutableRawPointer(write + n - 4).storeBytes(of: tail, as: UInt32.self)
        case 2...:
            let head = UnsafeRawPointer(read).loadUnaligned(as: UInt16.self)
            let tail = UnsafeRawPointer(read + n - 2).loadUnaligned(as: UInt16.self)
            UnsafeMutableRawPointer(write).storeBytes(of: head, as: UInt16.self)
            UnsafeMutableRawPointer(write + n - 2).storeBytes(of: tail, as: UInt16.self)
        default:
            write.pointee = read.pointee
        }
        commit(n)
    }
}
