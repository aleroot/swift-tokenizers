# Migrating from swift-transformers

Swap one package line. Most code does not change at all.

## Overview

The module name, the types and the call sites stay the same. This is the short list of what
does not.

### The swap

```swift
// before
.package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
.product(name: "Tokenizers", package: "swift-transformers"),

// after
.package(url: "https://github.com/aleroot/swift-tokenizers", from: "1.0.0"),
.product(name: "Tokenizers", package: "swift-tokenizers"),
```

The module keeps the name `Tokenizers`, and ``Tokenizer``, ``PreTrainedTokenizer``,
``AutoTokenizer``, ``Config`` and ``TokenizerError`` keep their shapes. mlx-swift-lm's
`#huggingFaceTokenizerLoader()` and `#adaptHuggingFaceTokenizer` macros expand unchanged.

### What does change

| swift-transformers | here |
|---|---|
| `import Hub` for `Config` | `Config` lives in `Tokenizers` |
| `AutoTokenizer.from(pretrained:)` | download with your own client, then ``AutoTokenizer/from(modelFolder:strict:)`` |
| `LanguageModelConfigurationFromHub(modelFolder:)` | ``LocalModelConfiguration`` |
| `init(config:)` on pipeline components | `init(config:) throws` |

Downloading is deliberately absent. A tokenizer library has no business owning an HTTP stack,
a cache directory or an authentication token, and most apps already have all three.

### What is not implemented

Inference only, single sequences:

- No training, no vocabulary learning, no BPE dropout above zero.
- No padding and no sequence pairs. Pad in your own batching code, where you know the shape
  your model wants.

If a `tokenizer.json` asks for a component this library does not implement, loading throws
``TokenizerError/unsupportedComponent(_:)`` naming it. It does not quietly drop the component
and hand you a tokenizer that produces almost correct ids.

### Behaviour differences worth knowing

Both libraries aim at Hugging Face, and this one is checked against Hugging Face directly,
corpus by corpus, not against swift-transformers. Where the two disagree, the reasoning and
the evidence for each case are recorded in
[BENCHMARKS.md](https://github.com/aleroot/swift-tokenizers/blob/main/docs/BENCHMARKS.md).

The differences found in end to end testing were all cases where swift-transformers diverged
from Hugging Face: a missing BOS token on one model family, decode errors on unusual byte
sequences, and chat template keys rendered in a different order.

### What you gain besides speed

- Source offsets: <doc:TokenAlignment>
- Incremental decoding: <doc:StreamingGeneration>
- Assistant loss masks for fine tuning: <doc:ChatTemplates>
- Special token aware truncation: ``TokenTruncation``
