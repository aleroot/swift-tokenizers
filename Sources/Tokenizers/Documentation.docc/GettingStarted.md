# Getting started

Load a tokenizer once, then encode and decode from anywhere in your app.

## Overview

The whole path from a package dependency to token ids, plus where to put the one step that is
not free.

### Add the package

```swift
dependencies: [
    .package(url: "https://github.com/aleroot/swift-tokenizers", from: "1.0.0"),
],
targets: [
    .target(name: "MyApp", dependencies: [
        .product(name: "Tokenizers", package: "swift-tokenizers"),
    ]),
]
```

Swift 6, macOS 13, iOS 16, tvOS 16, watchOS 9, visionOS 1.

### Load a tokenizer

Point the loader at the folder that holds the model files. It reads `tokenizer.json` and,
when present, `tokenizer_config.json`, `config.json` and a chat template file.

```swift
let tokenizer = try AutoTokenizer.load(from: modelFolder)              // synchronous
let tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder) // async
```

Both build the same object. The async form exists so that code written against
swift-transformers keeps compiling, not because loading suspends.

Downloading is out of scope. Fetch the files with whatever client the app already uses, put
them in a directory, and pass that directory.

### Encode and decode

```swift
let ids = tokenizer.encode(text: "Hello world")                    // adds special tokens
let bare = tokenizer.encode(text: "Hello world", addSpecialTokens: false)

let text = tokenizer.decode(tokens: ids)
let clean = tokenizer.decode(tokens: ids, skipSpecialTokens: true)
```

`addSpecialTokens` controls the markers the model expects around a sequence, such as a
beginning-of-sequence token or BERT's `[CLS]` and `[SEP]`. Leave it on when you feed a model,
turn it off when you are counting or splitting text.

To see what the model sees, ask for the pieces instead of the ids:

```swift
tokenizer.tokenize(text: "internationalization")
// ["international", "ization"]
```

### Look tokens up

```swift
tokenizer.convertTokenToId("Hello")   // 9707
tokenizer.convertIdToToken(9707)      // "Hello"

tokenizer.bosToken, tokenizer.eosTokenId, tokenizer.unknownToken
```

``Tokenizer/convertIdToToken(_:)`` returns `nil` for ids the vocabulary does not define, and
it is a constant-time lookup, so walking an entire 150,000 entry vocabulary is a few
milliseconds. Guided generation and grammar constrained decoding rely on that.

### Stay inside a context window

Cutting a prefix off an encoded array also cuts the special tokens the model needs. Ask the
tokenizer to fit the text instead:

```swift
let encoding = try tokenizer.encode(text: document, maxLength: 512, truncationSide: .right)
encoding.ids.count   // at most 512, with BOS/EOS still in place
```

See ``TokenTruncation``.

### Where to do the loading

Encoding is cheap, a chat turn costs microseconds. Loading is the part a user can feel: the
library reads a `tokenizer.json` that is often 10 to 30 MB and turns it into lookup tables.
On a Mac that is tens of milliseconds. On a phone the same work runs several times slower
when it lands on an efficiency core.

So: load once, off the main thread, at a normal quality of service. A load dispatched at
`.background` or `.utility` can take several times longer than the same load at `.userInitiated`,
because the scheduler keeps it on the slow cores.

```swift
actor Models {
    private var tokenizer: (any Tokenizer)?

    func tokenizer(at folder: URL) throws -> any Tokenizer {
        if let tokenizer { return tokenizer }
        let loaded = try AutoTokenizer.load(from: folder)
        tokenizer = loaded
        return loaded
    }
}
```

A loaded tokenizer is `Sendable` and holds immutable tables, so one instance can encode from
as many tasks as you like, in parallel, without locking. Do not create one per request.

### With mlx-swift-lm

mlx-swift-lm does not ship a tokenizer. It expects a module named `Tokenizers` and bridges to
it through its `#huggingFaceTokenizerLoader()` and `#adaptHuggingFaceTokenizer` macros. This
package provides exactly that module, so the macros expand unchanged:

```swift
let loader = #huggingFaceTokenizerLoader()
let tokenizer = try await loader.load(from: modelFolder)
```

A test in this repository replicates the generated bridge code verbatim, so a change that
would break mlx-swift-lm fails to compile here first.

### When something is missing

Every failure is a ``TokenizerError`` with a readable description: a missing file, an
unsupported `tokenizer_class`, a pipeline component this library does not implement, or a
configuration field that is absent or invalid. Print it, do not swallow it. The message names
the component.
