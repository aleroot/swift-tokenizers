import Foundation
import Testing

@testable import Tokenizers

@Suite("Tokenizer lifetime")
struct TokenizerLifetimeTests {
    private final class WeakReference: @unchecked Sendable {
        weak var value: AnyObject?
        init(_ value: AnyObject?) { self.value = value }
    }

    private func tokenizer(_ type: String = "BPE") throws -> PreTrainedTokenizer {
        let model: Config
        switch type {
        case "Unigram": model = ["type": "Unigram", "vocab": [["<unk>", 0.0], ["a", -1.0]], "unk_id": 0]
        case "WordPiece": model = ["type": "WordPiece", "vocab": ["[UNK]": 0, "a": 1], "unk_token": "[UNK]"]
        case "WordLevel": model = ["type": "WordLevel", "vocab": ["<unk>": 0, "a": 1], "unk_token": "<unk>"]
        default: model = ["type": "BPE", "vocab": ["a": 0], "merges": []]
        }
        return try PreTrainedTokenizer(tokenizerConfig: [:], tokenizerData: ["model": model])
    }

    @Test(arguments: ["BPE", "Unigram", "WordPiece", "WordLevel"], [false, true])
    func releasesUsedModel(type: String, offsets: Bool) throws {
        weak var model: AnyObject?
        weak var wrapper: PreTrainedTokenizer?
        try autoreleasepool {
            let tokenizer = try tokenizer(type)
            model = tokenizer.model as AnyObject
            wrapper = tokenizer
            if offsets {
                _ = try tokenizer.encode(text: "a", withOffsets: true)
            } else {
                _ = tokenizer.encode(text: "a")
            }
        }
        #expect(wrapper == nil)
        #expect(model == nil)
    }

    @Test("Releasing on another thread reclaims idle worker scratch before thread exit")
    func releaseAcrossThreads() throws {
        let holder = Locked<PreTrainedTokenizer?>(try tokenizer())
        let model = holder.withLock { WeakReference($0!.model as AnyObject) }
        let ready = DispatchSemaphore(value: 0)
        let exit = DispatchSemaphore(value: 0)
        let finished = DispatchSemaphore(value: 0)
        for offsets in [false, true] {
            Thread.detachNewThread {
                autoreleasepool {
                    let tokenizer = holder.withLock { $0! }
                    if offsets {
                        _ = try? tokenizer.encode(text: "a", withOffsets: true)
                    } else {
                        _ = tokenizer.encode(text: "a")
                    }
                }
                ready.signal()
                exit.wait()
                finished.signal()
            }
        }
        ready.wait()
        ready.wait()
        holder.withLock { $0 = nil }
        #expect(model.value == nil)
        exit.signal()
        exit.signal()
        finished.wait()
        finished.wait()
    }

    @Test("Repeated model switches do not retain released models")
    func switchingModels() throws {
        let survivor = try tokenizer()
        for _ in 0..<32 {
            weak var model: AnyObject?
            try autoreleasepool {
                let temporary = try tokenizer()
                model = temporary.model as AnyObject
                _ = temporary.encode(text: "a")
                #expect(survivor.encode(text: "aa") == [0, 0])
            }
            #expect(model == nil)
        }
    }

    private final class WorkerContext: @unchecked Sendable {
        let pool = EncodeScratchPool()
        let scratch = WeakReference(nil)
    }

    @Test("Thread exit releases its scratch while the tokenizer pool stays alive")
    func threadExit() throws {
        let context = WorkerContext()
        let retained = Unmanaged.passRetained(context)
        #if canImport(Darwin)
            var thread: pthread_t?
        #else
            var thread = pthread_t()
        #endif
        let result = pthread_create(
            &thread, nil,
            { pointer in
                let raw: UnsafeMutableRawPointer? = pointer
                guard let raw else { return nil }
                let context = Unmanaged<WorkerContext>.fromOpaque(raw).takeRetainedValue()
                let scratch = context.pool.take()
                context.scratch.value = scratch
                context.pool.recycle(scratch)
                return nil
            }, retained.toOpaque())
        if result != 0 { retained.release() }
        try #require(result == 0)
        #if canImport(Darwin)
            let joined = pthread_join(try #require(thread), nil)
        #else
            let joined = pthread_join(thread, nil)
        #endif
        #expect(joined == 0)
        #expect(context.scratch.value == nil)
        withExtendedLifetime(context.pool) {}
    }

    @Test("Nested scratch borrowing remains exclusive")
    func nestedBorrowing() {
        let pool = EncodeScratchPool()
        let outer = pool.take()
        let inner = pool.take()
        #expect(outer !== inner)
        pool.recycle(inner)
        pool.recycle(outer)
        #expect(pool.take() === outer)
    }
}
