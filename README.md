# swift-tokenizers

A high-performance, modern Swift 6 implementation of Hugging Face tokenizers and a
**drop-in replacement for the `Tokenizers` product of
[swift-transformers](https://github.com/huggingface/swift-transformers)**, without the Hub
client, downloaders, or any other dependency your app does not need.

```swift
import Tokenizers

let tokenizer = try AutoTokenizer.load(from: modelFolder)   // tokenizer.json (+ tokenizer_config.json, …)
let ids = tokenizer.encode(text: "Hello world")
let text = tokenizer.decode(tokens: ids)
let prompt = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": "Hi"]])
```

* **Same API**: `Tokenizer`, `PreTrainedTokenizer`, `AutoTokenizer`, `Config`, `TokenizerError`,
  chat templates, special-token metadata, `convertIdToToken` vocabulary reflection — everything
  [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) and similar consumers use.
* **Bit-exact with Hugging Face**: verified against `transformers` / Rust `tokenizers` on 24
  tokenizer families × 300 adversarial inputs, plus the full swift-transformers test-suite.
* **Fast**: 4–8 ns/byte for byte-level BPE on Apple silicon — faster than tiktoken's Rust core
  and two orders of magnitude faster than swift-transformers.
* **One dependency**: [swift-jinja](https://github.com/huggingface/swift-jinja) for chat templates.

## Performance

Single-threaded, Apple M-series, release build, ~115 KB of English prose (the
swift-transformers benchmark corpus). Same machine, same inputs.

| Implementation | Qwen3 byte-level BPE encode | Load `tokenizer.json` (11 MB) |
|---|---:|---:|
| **swift-tokenizers** | **~240 MB/s** | **66 ms** |
| tiktoken 0.12 (Rust core, `cl100k_base`) | ~35 MB/s | — |
| Hugging Face `tokenizers` 0.22 (Rust, single thread) | ~6 MB/s | — |
| swift-transformers 1.x (yyjson) | ~1 MB/s | 316 ms |

| Family (swift-tokenizers) | Throughput | Load |
|---|---:|---:|
| Llama 3 / Qwen / GPT-2 / DeepSeek / Phi-4 (byte-level BPE) | 225–255 MB/s | 66 ms (Qwen3, 11 MB `tokenizer.json`) |
| Llama 2 / Mistral / Phi-3 / Gemma (SentencePiece BPE) | 50–70 MB/s | |
| BERT / WordPiece | ~9 MB/s | |
| T5 / XLM-R (Unigram) | ~8 MB/s | |
| Decode (byte-level) | ~430 MB/s | |

Reproduce with `RUN_BENCHMARKS=1 swift test -c release --filter Benchmarks`.

End-to-end through mlx-swift-lm's `#huggingFaceTokenizerLoader()` macro on real mlx-community
model folders (8.8 MB document, release build, same machine). Every result below is
byte-identical to Hugging Face `transformers`; the swift-transformers column also lists its
divergences from HF on the same 301-text corpus.

| Model folder | swift-tokenizers | swift-transformers 1.x |
|---|---:|---:|
| Qwen3-14B-4bit | 161 MB/s, load 92 ms | 0.9 MB/s, load 325 ms, 1 encode mismatch |
| DeepSeek-R1-Distill-Qwen-14B-4bit | 159 MB/s, load 112 ms | 0.9 MB/s, load 351 ms, 1 encode mismatch |
| Qwen3.5-9B-8bit | 160 MB/s, load 166 ms | 0.7 MB/s, load 647 ms |
| Muse-Glimmer-30B-4bit (o200k) | 168 MB/s, load 213 ms | 0.6 MB/s, load 1005 ms, 45 decode mismatches |
| Falcon-H1R-7B-8bit | 90 MB/s, load 92 ms | 0.5 MB/s, load 370 ms |
| gemma-4-12B-it-qat-OptiQ-4bit | 126 MB/s, load 252 ms | 1.2 MB/s, load 967 ms, no BOS, wrong chat template |

### Why it is fast

* **Integer BPE.** Symbols are token ids, not strings; merges are resolved through a flat
  open-addressing table keyed by `(leftId, rightId)`. A byte-level pretoken goes from raw UTF-8
  bytes to symbol ids through a 256-entry table — no `String` is ever built.
* **Regex-free pre-tokenization.** The GPT-2, Llama-3/cl100k, Qwen-2, Qwen-3.5, o200k
  (GPT-4o / gpt-oss / Muse) and Falcon-H1 split patterns are implemented as hand-written
  linear scanners over UTF-8 with a precomputed Unicode classification table (including the
  o200k case-aware `[Lu Lt Lm Lo M]*[Ll Lm Lo M]+` alternation). `Punctuation`, literal
  `Split` and `[0-9]` stages are byte scanners too. All are fuzz-tested against
  `NSRegularExpression` for exact equivalence; unknown patterns fall back to
  `NSRegularExpression`.
* **Zero-allocation hot path.** Added tokens are found with a byte trie (and `memchr` when they
  share a first byte), sections and pieces are byte ranges, and the pretoken → ids cache is an
  arena-backed table with a `tryLock` so concurrent encodes never block.
* **Packed vocabulary.** Tokens live in one UTF-8 buffer with an offsets table (O(1)
  `convertIdToToken`) and a byte-hash index that is *binary-distinct* — `"à"` (U+00E0) and
  `"a\u{300}"` are different tokens, as they are in every real vocabulary.
* **Fast configuration loading.** A purpose-built JSON parser produces `Config` trees directly
  (≈1.8× faster than `JSONSerialization` on a 11 MB `tokenizer.json`), with `Double` precision
  so Unigram scores round exactly like the Rust implementation.
* **Exact SentencePiece chunking.** Llama-2-style tokenizers treat a whole text section as one
  BPE word. Every BPE output token is a merge product, so a merge can only cross a `▁` word
  boundary if some product has an interior `▁`; those products (gemma-4 has exactly one,
  `>▁</`) are checked at each boundary and the section is otherwise encoded chunk by chunk
  with cache hits — a 10× speed-up with identical output.

## Compatibility with swift-transformers

The public surface of the `Tokenizers` module is source-compatible. The `Config` type (with
`BinaryDistinctString` keys) that used to live in `Hub` is now part of `Tokenizers`.

Not included, by design: `HubApi`, downloading, `LanguageModelConfigurationFromHub`, `Models`,
`Generation`. Load files with whatever client your app already uses and point
`AutoTokenizer.from(modelFolder:)` (async) or `AutoTokenizer.load(from:)` (sync) at the directory
containing `tokenizer.json`, and optionally `tokenizer_config.json`, `config.json`,
`chat_template.json` / `chat_template.jinja`. `LocalModelConfiguration` exposes the same
resolution rules (chat-template merging, fallback configs for `gpt2` / `t5`).

### Tool definitions render in the order models were trained on

Python `transformers` renders `tools | tojson` in dictionary *insertion* order. A Swift
`Dictionary` has none, so swift-transformers (and swift-jinja's own converter) sort keys
alphabetically and produce `{"function": {"description": …, "name": …}, "type": …}` — a shape
no model has seen. Tool specs are not arbitrary JSON, though: they follow the OpenAI
function-calling schema, and `transformers.utils.get_json_schema` — whose output the chat
templates were trained against — always emits `type, function` → `name, description,
parameters, return` → `type, properties, required` → `type, items, nullable, enum, description`.

swift-tokenizers lays out `Dictionary` inputs (messages, tools, `additionalContext`) in that
canonical order, which is byte-identical to the Python rendering for every standard tool
definition and OpenAI-format tool call. Only *user-named* keys have no canonical order (the
parameters under `properties`, the values under `arguments`); they fall back to alphabetical.
To control those too, nest an order-preserving literal anywhere — it still fits the
`[String: any Sendable]` contract, so it flows through mlx-swift-lm untouched:

```swift
let properties: KeyValuePairs<String, any Sendable> = [   // renders in exactly this order
    "city": ["type": "string", "description": "The city name"] as [String: any Sendable],
    "unit": ["type": "string", "enum": ["celsius", "fahrenheit"]] as [String: any Sendable],
]
let tool: [String: any Sendable] = [
    "type": "function",
    "function": ["name": "get_weather", "parameters": ["type": "object", "properties": properties] as [String: any Sendable]] as [String: any Sendable],
]
```

A pre-built `Jinja.Value` is accepted as well and rendered as is.

## Migrating

### mlx-swift-lm

mlx-swift-lm already talks to tokenizers through its `MLXLMCommon.Tokenizer` protocol and the
`#huggingFaceTokenizerLoader()` / `#adaptHuggingFaceTokenizer` macros, which only rely on the
`Tokenizers` module API. In your app's `Package.swift`:

```swift
// before
.package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
…
.product(name: "Tokenizers", package: "swift-transformers"),

// after
.package(url: "https://github.com/aleroot/swift-tokenizers", from: "1.0.0"),
…
.product(name: "Tokenizers", package: "swift-tokenizers"),
```

`import Tokenizers` and the macro expansions compile unchanged (see
`Tests/TokenizersTests/MLXBridgeCompatibilityTests.swift`, which builds the exact generated code).
Keep using your own downloader for model files; the tokenizer only needs the local folder.

### swift-transformers users

* Replace `import Hub` with `import Tokenizers` where you only needed `Config`.
* Replace `AutoTokenizer.from(pretrained:)` with a download step followed by
  `AutoTokenizer.from(modelFolder:)`.
* `LanguageModelConfigurationFromHub(modelFolder:)` → `LocalModelConfiguration(modelFolder:)`
  (synchronous).
* Pipeline components (`Normalizer`, `PreTokenizer`, `PostProcessor`, `Decoder`) declare
  `init(config:) throws`. Only components with required fields actually throw; the concrete
  initializers you were calling directly are unchanged unless they could fail.

## Error handling

A malformed or unsupported `tokenizer.json` is reported, never trapped on. Every failure
surfaces as a `TokenizerError` with a descriptive `errorDescription`:

| Error | When |
|---|---|
| `missingFile(URL)` | `tokenizer.json` (or a needed `tokenizer_config.json`) is not in the folder |
| `unsupportedTokenizer(String)` | `tokenizer_class` is unknown *and* `tokenizer.json` has no recognisable `model.type` (strict mode) |
| `unsupportedComponent(String)` | a normalizer / pre-tokenizer / post-processor / decoder type is not implemented |
| `invalidConfiguration(String)` | a component is missing a required field or has an invalid regex |
| `missingVocab`, `mismatchedConfig(String)` | the model section is unusable, or `add_bos_token` is set without a `bos_token` |
| `missingChatTemplate`, `chatTemplate(String)` | no template is configured, or rendering failed |

Unregistered `tokenizer_class` names (new Hub classes appear regularly) resolve through
`tokenizer.json`'s authoritative `model.type` (`BPE` / `Unigram` / `WordPiece`), which is what
the `tokenizers` library itself does — so they load correctly even in strict mode.

All tokenizers are `Sendable` and safe to share across tasks; encoding never blocks on the
internal cache (concurrent callers simply bypass it).

## Supported tokenizers

Models: byte-level and SentencePiece **BPE** (`GPT2`, `Llama`, `CodeLlama`, `Gemma`, `Qwen2`,
`Falcon`, `Whisper`, `Cohere`, `Roberta`, `TokenizersBackend`, …), **Unigram** (`T5`,
`XLMRoberta`), **WordPiece** (`Bert`, `DistilBert`).
Normalizers: Sequence, Prepend, Replace, Lowercase, NFC/NFD/NFKC/NFKD, Bert, Precompiled,
StripAccents, Strip. Pre-tokenizers: Sequence, ByteLevel, Split, Metaspace, Whitespace,
WhitespaceSplit, Punctuation, Digits, BertPreTokenizer. Post-processors: TemplateProcessing,
ByteLevel, RobertaProcessing, BertProcessing, Sequence. Decoders: Sequence, ByteLevel,
ByteFallback, Fuse, Strip, Replace, Metaspace, WordPiece.

## License

Apache License 2.0. See `NOTICE` for attributions (swift-transformers test-suite and fixtures,
Hugging Face reference implementations, swift-gigatoken).
