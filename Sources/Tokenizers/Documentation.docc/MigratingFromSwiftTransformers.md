# Migrating from swift-transformers

Keep familiar inference calls while moving local tokenizer loading to this package.

## Overview

The supported compatibility surface is local loading, ordinary encoding and decoding, chat
templates, special-token metadata and vocabulary lookup. This does not promise source or
behavioral identity for every public swift-transformers or Hub helper.

### The swap

```swift
// before
.package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
.product(name: "Tokenizers", package: "swift-transformers"),

// after
.package(url: "https://github.com/aleroot/swift-tokenizers", from: "1.0.0"),
.product(name: "Tokenizers", package: "swift-tokenizers"),
```

The module keeps the name `Tokenizers`. Common calls on ``Tokenizer``,
``PreTrainedTokenizer`` and ``AutoTokenizer`` retain their labels. mlx-swift-lm's
`#huggingFaceTokenizerLoader()` and `#adaptHuggingFaceTokenizer` macros use this interface;
see <doc:UsingWithMLX> for a complete local-loading example.

### What does change

| swift-transformers | here |
|---|---|
| `import Hub` for `Config` | `Config` construction and lookup live in `Tokenizers` |
| `Config.Data` and `ConfigError` | Internal storage is not public; use `Config` accessors and the errors documented by each operation |
| `AutoTokenizer.from(pretrained:)` | download with your own client, then ``AutoTokenizer/from(modelFolder:strict:)`` |
| `LanguageModelConfigurationFromHub(modelFolder:)` | ``LocalModelConfiguration`` |
| `init(config:)` on pipeline components | `init(config:) throws` |

Downloading is deliberately absent. A tokenizer library has no business owning an HTTP stack,
a cache directory or an authentication token, and most apps already have all three.

### What is not implemented

Inference only:

- No training, no vocabulary learning, no BPE dropout above zero.
- No padding. Pad in your own batching code, where you know the shape your model wants.

If a `tokenizer.json` asks for a component this library does not implement, loading throws
``TokenizerError/unsupportedComponent(_:)`` naming it. It does not quietly drop the component
and hand you a tokenizer that produces almost correct ids.

### Behaviour differences worth knowing

Both libraries aim at Hugging Face, and this one is checked against Hugging Face directly,
corpus by corpus, not against swift-transformers. Where the two disagree, the reasoning and
the evidence for each case are recorded in
[BENCHMARKS.md](https://github.com/aleroot/swift-tokenizers/blob/main/docs/BENCHMARKS.md).

Validated improvements include honoring exported BOS settings and preserving serialized
normalization pipelines. Baseline agreement is not a correctness oracle: both Swift libraries
also inherit Jinja differences when templates stringify dictionaries or null values. Standard
JSON serialization with `tojson` avoids those particular representation differences. Check the
actual templates and prompts your app uses, especially tools and structured message content.

For loading behavior, strictness and metadata precedence, see <doc:LoadingModelFolders>.

### What you gain besides speed

- Source offsets: <doc:TokenAlignment>
- Sequence pairs with segment ids: ``Tokenizer/encode(text:textPair:addSpecialTokens:)``
- Incremental decoding: <doc:StreamingGeneration>
- Assistant loss masks for fine tuning: <doc:ChatTemplates>
- Special token aware truncation: ``TokenTruncation``
