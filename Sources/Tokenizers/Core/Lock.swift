import Foundation

#if canImport(os)
    import os
#endif

/// Minimal unfair mutex. On Darwin this wraps `os_unfair_lock`; elsewhere it falls back
/// to `NSLock`. Provides `tryLock` so hot paths can opportunistically use shared caches
/// without ever blocking a concurrent caller.
final class UnfairLock: @unchecked Sendable {
    #if canImport(os)
        private let pointer: os_unfair_lock_t

        init() {
            pointer = .allocate(capacity: 1)
            pointer.initialize(to: os_unfair_lock())
        }

        deinit {
            pointer.deinitialize(count: 1)
            pointer.deallocate()
        }

        @inline(__always) func lock() { os_unfair_lock_lock(pointer) }
        @inline(__always) func unlock() { os_unfair_lock_unlock(pointer) }
        @inline(__always) func tryLock() -> Bool { os_unfair_lock_trylock(pointer) }
    #else
        private let inner = NSLock()
        @inline(__always) func lock() { inner.lock() }
        @inline(__always) func unlock() { inner.unlock() }
        @inline(__always) func tryLock() -> Bool { inner.try() }
    #endif

    @inline(__always)
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}

/// A value guarded by an ``UnfairLock``. Access is only possible through ``withLock(_:)``,
/// which is what makes the wrapper safe to share across isolation domains.
final class Locked<Value>: @unchecked Sendable {
    private var value: Value
    private let lock = UnfairLock()

    init(_ value: Value) {
        self.value = value
    }

    @inline(__always)
    func withLock<T>(_ body: (inout Value) throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body(&value)
    }
}
