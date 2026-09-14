# ``Tokenizers``

Turn text into the token ids a language model expects, and turn them back into text.

## Overview

A language model does not read text, it reads integers. A tokenizer translates in both
directions, and it has to translate exactly the way the model was trained. Get it wrong by
one token and the model answers a question nobody asked.

swift-tokenizers reads the `tokenizer.json` files published on the Hugging Face Hub and
reproduces their behaviour in pure Swift: byte level and SentencePiece BPE, Unigram,
WordPiece and WordLevel, together with the normalizers, pre-tokenizers, post-processors and
decoders those files declare.

```swift
import Tokenizers

let tokenizer = try AutoTokenizer.load(from: modelFolder)

let ids = tokenizer.encode(text: "Hello world")   // [9707, 1879]
let text = tokenizer.decode(tokens: ids)          // "Hello world"
```

You bring the files, the library builds the tokenizer. There is no download client, and
nothing here talks to the network.

### Three things worth knowing early

**It is checked against the original.** Token ids are compared with Hugging Face's Rust
`tokenizers` and Python `transformers` on whole corpora, not on a handful of examples. Where
this library deliberately differs, the reason is written down in
[BENCHMARKS.md](https://github.com/aleroot/swift-tokenizers/blob/main/docs/BENCHMARKS.md).

**It is fast, and that matters on a phone.** Encoding is measured in hundreds of megabytes
per second, so a chat turn costs microseconds. The part you can feel is loading, which reads
and indexes a file that is often tens of megabytes. See <doc:GettingStarted> for where to put
that work.

**Every tokenizer is `Sendable`.** One instance serves the whole app, from any number of
tasks at once. Load it once, share it, never copy it per request.

## Topics

### Essentials

- <doc:GettingStarted>
- <doc:UsingWithMLX>
- <doc:LoadingModelFolders>
- ``AutoTokenizer``
- ``Tokenizer``

### Chat and generation

- <doc:ChatTemplates>
- <doc:StreamingGeneration>
- ``TokenStreamDecoder``
- ``ChatTemplateArgument``
- ``Message``
- ``ToolSpec``

### Working with the text

- <doc:TokenAlignment>
- ``TokenEncoding``
- ``TokenTruncation``
- ``TruncationSide``
- ``ChatTemplateEncoding``

### Moving from swift-transformers

- <doc:MigratingFromSwiftTransformers>

### Configuration

- ``Config``
- ``LocalModelConfiguration``
- ``BinaryDistinctString``
- ``TokenizerError``
- ``JSONConfigError``

### Building a tokenizer by hand

- ``PreTrainedTokenizer``
- ``TokenizingModel``
- ``PreTrainedTokenizerModel``
- ``BertTokenizer``
- ``WordLevelTokenizer``

### Pipeline components

- ``Normalizer``
- ``PreTokenizer``
- ``PreTokenizerOption``
- ``PreTokenizerOptions``
- ``PostProcessor``
- ``Decoder``
