# Loading model folders

Which files the loader reads, in which order, and what to do when one is missing.

## Overview

The loader takes a plain directory. This is what it looks for inside, and what it does when
something is not there.

### The normal case

A folder downloaded from the Hugging Face Hub usually contains everything needed:

| File | Role |
|---|---|
| `tokenizer.json` | The complete tokenizer: vocabulary, merges, and the whole pipeline |
| `tokenizer_config.json` | Special tokens, the `tokenizer_class`, the chat template |
| `config.json` | Model metadata, used only to infer a missing `tokenizer_class` |
| `chat_template.jinja` or `chat_template.json` | A chat template kept in its own file |

``AutoTokenizer/load(from:strict:)`` reads them and returns a ready ``Tokenizer``. Nothing
else in the folder is touched, so weights can sit next to the tokenizer without being read.

If a chat template file is present it wins over the `chat_template` field inside
`tokenizer_config.json`, and `chat_template.jinja` wins over `chat_template.json`. This is the
same precedence the Python `AutoProcessor` applies.

### When `tokenizer.json` is absent

Older repositories publish the vocabulary in separate files. The loader converts those, but
only for pipelines it can identify:

| Files | Requires |
|---|---|
| `vocab.txt` | `BertTokenizer`, `DistilBertTokenizer` or `MPNetTokenizer` |
| `vocab.json` and `merges.txt` | `GPT2Tokenizer`, `Qwen2Tokenizer`, `RobertaTokenizer` or `CLIPTokenizer` |

The class comes from `tokenizer_class` in `tokenizer_config.json`, or is inferred from
`model_type` in `config.json`. `special_tokens_map.json` and `added_tokens.json` are folded in
when present.

A vocabulary file on its own does not say how text should be normalized or split, so an
unknown class throws ``TokenizerError/unsupportedComponent(_:)`` rather than guessing. If you
hit that, export a `tokenizer.json` from `transformers` and ship it.

### Strict loading

```swift
let tokenizer = try AutoTokenizer.load(from: folder, strict: true)   // the default
```

With `strict: true` an unrecognized `tokenizer_class` is an error. With `strict: false` the
loader falls back to the generic pipeline, which is useful for a new model family whose class
name this library has not seen but whose `tokenizer.json` is ordinary. Prefer the default and
find out early.

### Configuration you already hold

If your app has the bytes in memory, or stores them somewhere other than a directory, build
the ``Config`` values yourself:

```swift
let data = try Config(tokenizerJSON: tokenizerJSONBytes)
let config = try Config(jsonData: tokenizerConfigBytes)
let tokenizer = try AutoTokenizer.from(tokenizerConfig: config, tokenizerData: data)
```

Use ``Config/init(tokenizerJSON:)`` for `tokenizer.json` specifically. It packs the vocabulary
and merge list into compact tables instead of building a tree with hundreds of thousands of
nodes, and it keeps vocabulary keys that are byte distinct but Unicode equivalent apart, which
`JSONDecoder` would silently merge into one entry and corrupt the vocabulary.

``LocalModelConfiguration`` exposes the same folder reading logic if you want to inspect what
was found before building anything.

### Inspecting a folder

```swift
let configuration = try LocalModelConfiguration(modelFolder: folder)
configuration.modelType          // "qwen3"
configuration.tokenizerConfig    // with class inference and bundled fallbacks applied
```

Two model types that historically shipped without a `tokenizer_config.json`, `gpt2` and `t5`,
have a bundled fallback configuration. Values found in the folder always win over it.
