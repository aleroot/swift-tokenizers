<p align="center">
  <h1 align="center">swift-tokenizers</h1>
  <p align="center">High-performance tokenizers in pure Swift.</p>
</p>

<p align="center">
  <a href="https://github.com/aleroot/swift-tokenizers/actions/workflows/ci.yml"><img src="https://github.com/aleroot/swift-tokenizers/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/Swift-6.0-F05138?logo=swift&logoColor=white" alt="Swift 6">
  <img src="https://img.shields.io/badge/platforms-macOS%20%7C%20iOS%20%7C%20tvOS%20%7C%20watchOS%20%7C%20visionOS-blue" alt="Platforms">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
</p>

swift-tokenizers is a native Swift 6 implementation of the tokenizers used by today's language
models: byte-level and SentencePiece **BPE**, **Unigram** and **WordPiece**, engineered for
speed. It encodes at 200+ MB/s on a single Apple silicon core, well beyond the Rust
implementations, with sub-microsecond latency on short inputs. It loads the `tokenizer.json` files published on the
Hugging Face Hub and reproduces the output of Hugging Face `tokenizers` token for token. It ships
as a drop-in replacement for the `Tokenizers` product of
[swift-transformers](https://github.com/huggingface/swift-transformers), plugs seamlessly into
[mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm), and is the tokenizer behind
[Lampo](https://apps.apple.com/app/lampo/id6760648195).

```swift
import Tokenizers

let tokenizer = try AutoTokenizer.load(from: modelFolder)

let ids = tokenizer.encode(text: "Hello world")                     // [9707, 1879]
let text = tokenizer.decode(tokens: ids)                            // "Hello world"
let prompt = try tokenizer.applyChatTemplate(messages: [
    ["role": "user", "content": "What is the capital of France?"]
])
```

## Highlights

- **Bit-exact with Hugging Face.** Every token, offset and special-token rule matches
  `transformers` / Rust `tokenizers`, verified on 24 tokenizer families × 300 adversarial inputs,
  on real embedding, reranker and LLM model folders, and through the complete swift-transformers
  test-suite.
- **Extremely fast.** 200+ MB/s single-threaded on Apple silicon: 30-40x the Rust `tokenizers`
  core, 4-6x tiktoken and two orders of magnitude faster than swift-transformers, with
  sub-microsecond latency on short queries. A 150k-token vocabulary loads in about 25 ms.
- **Complete.** Added and special tokens, every normalizer, pre-tokenizer, post-processor and
  decoder that modern `tokenizer.json` files use, chat templates with tools, and O(1) vocabulary
  reflection for guided generation.
- **Drop-in.** Same `Tokenizer`, `PreTrainedTokenizer`, `AutoTokenizer`, `Config` and
  `TokenizerError` API as swift-transformers; existing consumers compile unchanged.
- **Lean.** A single dependency, [swift-jinja](https://github.com/huggingface/swift-jinja), for
  chat templates. No Hub client, no downloader. Every tokenizer is `Sendable`; malformed input is a
  `TokenizerError`, never a trap.

## Installation

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

Requires Swift 6. Supports macOS 13, iOS 16, tvOS 16, watchOS 9 and visionOS 1.

## Usage

### Loading

Point the library at a folder containing `tokenizer.json`, plus `tokenizer_config.json`,
`config.json`, `chat_template.json` or `chat_template.jinja` when present, as downloaded with
whichever Hub client your app already uses:

```swift
let tokenizer = try AutoTokenizer.load(from: modelFolder)              // synchronous
let tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder) // async
```

Unknown `tokenizer_class` names resolve through the `model.type` of `tokenizer.json`, so new
Hub classes load without a library update.

### Encoding and decoding

```swift
let ids = tokenizer.encode(text: "Hello world", addSpecialTokens: true)
let tokens = tokenizer.tokenize(text: "Hello world")
let text = tokenizer.decode(tokens: ids, skipSpecialTokens: true)

tokenizer.bosToken, tokenizer.eosTokenId, tokenizer.unknownToken
tokenizer.convertTokenToId("Hello"), tokenizer.convertIdToToken(9707)
```

### Chat templates and tools

```swift
let messages: [Message] = [
    ["role": "system", "content": "You are a helpful assistant."],
    ["role": "user", "content": "What is the weather in Paris?"],
]
let tool: ToolSpec = [
    "type": "function",
    "function": [
        "name": "get_weather",
        "description": "Get the current weather in a city",
        "parameters": [
            "type": "object",
            "properties": ["city": ["type": "string"] as [String: any Sendable]],
            "required": ["city"],
        ] as [String: any Sendable],
    ] as [String: any Sendable],
]

let ids = try tokenizer.applyChatTemplate(messages: messages, tools: [tool])
let prompt = try (tokenizer as! PreTrainedTokenizer)
    .renderChatTemplate(messages: messages, addGenerationPrompt: true, tools: [tool])
```

Tool definitions render in the canonical key order of `transformers.utils.get_json_schema`,
byte-identical to Python for standard schemas. To pin the order of your own keys, use a
`KeyValuePairs` literal or a pre-built `Jinja.Value` anywhere inside a message or tool.

## Supported components

| | |
|---|---|
| **Models** | byte-level and SentencePiece BPE (GPT-2, Llama 2 / 3, Qwen, Mistral, Gemma, DeepSeek, Phi, Falcon, Whisper, Cohere, RoBERTa, o200k …), Unigram (T5, XLM-RoBERTa, multilingual-e5, bge-m3 …), WordPiece (BERT, DistilBERT, MiniLM, bge, nomic …) |
| **Normalizers** | Sequence, Prepend, Replace, Lowercase, NFC, NFD, NFKC, NFKD, BertNormalizer, Precompiled, StripAccents, Strip |
| **Pre-tokenizers** | Sequence, ByteLevel, Split, Metaspace, Whitespace, WhitespaceSplit, Punctuation, Digits, BertPreTokenizer |
| **Post-processors** | TemplateProcessing, ByteLevel, RobertaProcessing, BertProcessing, Sequence |
| **Decoders** | Sequence, ByteLevel, ByteFallback, Fuse, Strip, Replace, Metaspace, WordPiece |

## Performance

| | Encode (Qwen3 BPE, prose) | Load 10.9 MB `tokenizer.json` |
|---|---:|---:|
| **swift-tokenizers** | **212 MB/s** | **26 ms** |
| tiktoken 0.14 (Rust) | 35-53 MB/s | n/a |
| Hugging Face `tokenizers` 0.22 (Rust) | 6.7 MB/s | 99 ms |
| swift-transformers | 0.9 MB/s | 299 ms |

Apple M4 Pro, single thread, identical inputs and outputs. Every tokenizer family, embedding and
reranker models, end-to-end figures through mlx-swift-lm, cold-cache runs, memory, methodology
and how it is done: **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

## Migrating from swift-transformers

The `Tokenizers` module is source-compatible; swap the package dependency:

```swift
// before
.package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
.product(name: "Tokenizers", package: "swift-transformers"),

// after
.package(url: "https://github.com/aleroot/swift-tokenizers", from: "1.0.0"),
.product(name: "Tokenizers", package: "swift-tokenizers"),
```

- mlx-swift-lm's `#huggingFaceTokenizerLoader()` and `#adaptHuggingFaceTokenizer` macros work
  unchanged.
- `Config` now lives in `Tokenizers`; drop `import Hub` where that was all you needed.
- `AutoTokenizer.from(pretrained:)` becomes a download step followed by
  `AutoTokenizer.from(modelFolder:)`; `LanguageModelConfigurationFromHub(modelFolder:)` becomes
  `LocalModelConfiguration(modelFolder:)`.
- Pipeline component initializers are `init(config:) throws`.

## License

[Apache License 2.0](LICENSE).
