import Dispatch
import Foundation
import Testing

@testable import Tokenizers

@Suite("Sendable ownership and synchronization")
struct SendableTests {
    @Test("Locked serializes mutations and unlocks after a throwing body")
    func lockedValue() {
        enum Failure: Error { case expected }
        let count = Locked(0)
        do {
            try count.withLock { value in
                value += 1
                throw Failure.expected
            }
        } catch {}
        DispatchQueue.concurrentPerform(iterations: 8) { _ in
            for _ in 0..<1_000 { count.withLock { $0 += 1 } }
        }
        #expect(count.withLock { $0 } == 8_001)
    }

    @Test("Concurrent lazy initialization publishes one winning value")
    func lazyPublication() {
        final class Value: Sendable {}
        let calls = Locked(0)
        let lazy = Lazy {
            calls.withLock { $0 += 1 }
            return Value()
        }
        let identities = Locked<Set<ObjectIdentifier>>([])
        DispatchQueue.concurrentPerform(iterations: 16) { _ in
            for _ in 0..<100 {
                let value = lazy.value
                identities.withLock { _ = $0.insert(ObjectIdentifier(value)) }
            }
        }
        #expect(identities.withLock { $0 } == [ObjectIdentifier(lazy.value)])
        let factoryCalls = calls.withLock { $0 }
        #expect(factoryCalls >= 1 && factoryCalls <= 16)
        _ = lazy.value
        #expect(calls.withLock { $0 } == factoryCalls)
    }

    @Test("Nested scratch leases remain exclusive and reused encoders follow model identity")
    func nestedScratchAndModelIdentity() throws {
        let pool = EncodeScratchPool()
        let outer = pool.take()
        let inner = pool.take()
        #expect(outer !== inner)
        outer.normalized = [1, 2, 3]
        inner.normalized = [4, 5]
        pool.recycle(inner)
        let reused = pool.take()
        #expect(reused === inner)
        #expect(outer.normalized == [1, 2, 3])

        func model(id: Int) throws -> WordLevelTokenizer {
            try WordLevelTokenizer(
                tokenizerConfig: [:],
                tokenizerData: ["model": ["type": "WordLevel", "unk_token": "<unk>", "vocab": ["<unk>": 0, "a": Config(id)]]],
                addedTokens: [:])
        }
        let first = try model(id: 1)
        let second = try model(id: 2)
        for model in [first, second, first] {
            let encoder = try #require(reused.encoder(identity: ObjectIdentifier(model), make: model.makeEncoder))
            encoder.begin()
            var ids: [Int] = []
            encoder.encode(piece: "a", byteLevel: false, into: &ids)
            encoder.finish()
            #expect(ids == [model.convertTokenToId("a")!])
        }
        pool.recycle(reused)
        pool.recycle(outer)
    }

    @Test("One shared pool lends distinct scratch to simultaneously active threads")
    func threadLocalScratch() {
        let pool = EncodeScratchPool()
        let active = Locked<Set<ObjectIdentifier>>([])
        let ready = DispatchGroup()
        let finished = DispatchGroup()
        let release = DispatchSemaphore(value: 0)
        for _ in 0..<8 {
            ready.enter()
            finished.enter()
            Thread.detachNewThread {
                let scratch = pool.take()
                let identity = ObjectIdentifier(scratch)
                #expect(active.withLock { $0.insert(identity).inserted })
                ready.leave()
                release.wait()
                active.withLock { _ = $0.remove(identity) }
                pool.recycle(scratch)
                let reused = pool.take()
                #expect(reused === scratch)
                pool.recycle(reused)
                finished.leave()
            }
        }
        ready.wait()
        #expect(active.withLock { $0.count } == 8)
        for _ in 0..<8 { release.signal() }
        finished.wait()
        #expect(active.withLock { $0.isEmpty })
    }

    @Test(
        "Shared tokenizers agree across encode, offsets, cold decode and template eviction",
        arguments: [
            "Qwen/Qwen3-0.6B", "coreml-projects/Llama-2-7b-chat-coreml",
            "FacebookAI/xlm-roberta-base", "google-bert/bert-base-uncased", "google-t5/t5-small",
        ])
    func concurrentInference(repo: String) async throws {
        let folder = try await HubFixtures.modelFolder(for: repo)
        let reference = try #require(AutoTokenizer.load(from: folder) as? PreTrainedTokenizer)
        let tokenizer = try #require(AutoTokenizer.load(from: folder) as? PreTrainedTokenizer)
        let texts = ["Hello, world! 123456789", "café a\u{300} 中文 ภาษาไทย 😀", "a a a b b a", ""]
        let ids = texts.map { reference.encode(text: $0) }
        let decoded = ids.map { reference.decode(tokens: $0) }
        let offsets = try texts.map { try reference.encode(text: $0, withOffsets: true) }
        try await withThrowingTaskGroup(of: Void.self) { group in
            for worker in 0..<16 {
                group.addTask {
                    for iteration in 0..<40 {
                        let index = (worker + iteration) % texts.count
                        // No prior decode on this instance: exercise Lazy's publication too.
                        #expect(tokenizer.decode(tokens: ids[index]) == decoded[index])
                        #expect(tokenizer.encode(text: texts[index]) == ids[index])
                        #expect(tokenizer.tokenize(text: texts[index]) == reference.tokenize(text: texts[index]))
                        let encoded = try tokenizer.encode(text: texts[index], withOffsets: true)
                        #expect(encoded.ids == offsets[index].ids)
                        #expect(encoded.offsets == offsets[index].offsets)
                        let suffix = iteration % 20
                        let template = "{{ messages[0]['content'] }}:\(suffix)"
                        let rendered = try tokenizer.renderChatTemplate(
                            messages: [["role": "user", "content": "worker-\(worker)"]],
                            chatTemplate: .literal(template))
                        #expect(rendered == "worker-\(worker):\(suffix)")
                    }
                }
            }
            try await group.waitForAll()
        }
        #expect(tokenizer.compiledChatTemplateCount <= 16)
    }
}
