// Growable table storage whose pages come straight from the VM system and go back to it when
// the buffer is released. The large tables parsed out of `tokenizer.json` live here: they are
// consumed while the tokenizer is built and then dropped, and memory freed through `munmap`
// leaves the process footprint immediately, whereas blocks returned to `malloc` can stay
// resident in the allocator's caches.

import Foundation

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
    private var base: UnsafeMutablePointer<Element>
    private var capacity: Int
    private var mappedBytes: Int

    /// Creates a buffer able to hold `capacity` elements without reallocating.
    init(capacity: Int) {
        let (pointer, bytes, elements) = Self.map(elements: max(capacity, 1))
        base = pointer
        mappedBytes = bytes
        self.capacity = elements
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

    deinit {
        Self.unmap(base, bytes: mappedBytes)
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
        let (pointer, bytes, elements) = Self.map(elements: target)
        if count > 0 { pointer.update(from: base, count: count) }
        Self.unmap(base, bytes: mappedBytes)
        base = pointer
        mappedBytes = bytes
        capacity = elements
    }

    private static var pageSize: Int { Int(getpagesize()) }

    /// Maps enough anonymous pages for `elements`, returning the usable element capacity.
    private static func map(elements: Int) -> (UnsafeMutablePointer<Element>, Int, Int) {
        let stride = MemoryLayout<Element>.stride
        let page = pageSize
        let bytes = ((elements * stride + page - 1) / page) * page
        let raw = mmap(nil, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0)
        if let raw, raw != MAP_FAILED {
            return (raw.bindMemory(to: Element.self, capacity: bytes / stride), bytes, bytes / stride)
        }
        // Address space exhausted: fall back to the allocator; `bytes == 0` marks the block.
        let pointer = UnsafeMutablePointer<Element>.allocate(capacity: elements)
        return (pointer, 0, elements)
    }

    private static func unmap(_ pointer: UnsafeMutablePointer<Element>, bytes: Int) {
        if bytes == 0 {
            pointer.deallocate()
        } else {
            munmap(UnsafeMutableRawPointer(pointer), bytes)
        }
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
