# Using with mlx-swift-lm

Load models with your own downloader and use swift-tokenizers for MLX inference.

## Overview

Your app supplies a local model folder. swift-tokenizers reads the tokenizer files, while
mlx-swift-lm loads the weights, prepares model inputs and generates responses. The same
loader works with text models, vision-language models, embedders and rerankers.

### Add the packages

For an app using the current MLX macro APIs, add these dependencies and products to your
`Package.swift`. These examples require Swift 6.2 and MLX's deployment minimums: macOS 14
or iOS 17 (also tvOS 17 and visionOS 1). The MLX revision below is the one used for
the integration checks, including the newer reranker API; choose a release containing those
APIs when updating it.

```swift
// Package dependencies
.package(url: "https://github.com/aleroot/swift-tokenizers", from: "1.0.0"),
.package(
    url: "https://github.com/ml-explore/mlx-swift-lm",
    revision: "604fae710a4e3324346fc59e3845952350acd4b7",
    traits: []
),

// App target dependencies
.product(name: "Tokenizers", package: "swift-tokenizers"),
.product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
.product(name: "MLXLMCommon", package: "mlx-swift-lm"),
.product(name: "MLXLLM", package: "mlx-swift-lm"),
```

`traits: []` leaves MLX's optional Apple FoundationModels integration disabled. If your app
uses that integration, select its trait and follow its additional SDK requirements.

Choose one provider for the `Tokenizers` module throughout the dependency graph. Remove the
`Tokenizers` product from swift-transformers when replacing it; two modules with the same
name cannot be linked into the app. You can keep a separate downloader package.

### Load a local model and chat

```swift
import Foundation
import Tokenizers
import MLXHuggingFace
import MLXLMCommon
import MLXLLM

@MainActor
func answer(in modelFolder: URL) async throws -> String {
    let container = try await LLMModelFactory.shared.loadContainer(
        from: modelFolder,
        using: #huggingFaceTokenizerLoader()
    )
    let session = ChatSession(
        container,
        generateParameters: GenerateParameters(maxTokens: 256, temperature: 0.7)
    )
    return try await session.respond(to: "Explain Swift actors in two sentences.")
}
```

The macro finds `AutoTokenizer` through `import Tokenizers` and adapts it to MLX's tokenizer
protocol. No custom adapter or downloader is required for local loading.

Keep the container loaded across requests. Keep a `ChatSession` for each conversation to
preserve its history and KV cache; use each session from one task at a time. For streaming:

```swift
for try await chunk in session.streamResponse(to: "Give me a short example.") {
    print(chunk, terminator: "")
}
```

Release sessions, containers and any separately retained tokenizer when unloading a model.
The tokenizer's encoding scratch and model tables are reclaimed with its ownership, including
scratch used by still-running worker threads. MLX manages its own GPU allocator and caches.

### Use a tokenizer you already loaded

```swift
let native = try AutoTokenizer.load(from: modelFolder)
let mlxTokenizer = #adaptHuggingFaceTokenizer(native)
```

Use the adapted value where an API expects `MLXLMCommon.Tokenizer`. Use the native value for
additional swift-tokenizers features such as source offsets, sequence pairs and assistant
masks. Qualify `Tokenizers.Tokenizer` and `MLXLMCommon.Tokenizer` when both protocol names are
in scope.

### Other MLX model families

Add the relevant MLX product and import its module, then use the same loader:

```swift
// import MLXVLM
let vision = try await VLMModelFactory.shared.loadContainer(
    from: modelFolder, using: #huggingFaceTokenizerLoader())

// import MLXEmbedders
let embedder = try await EmbedderModelFactory.shared.loadContainer(
    from: modelFolder, using: #huggingFaceTokenizerLoader())

// import MLXRerankers
let reranker = try await RerankerModelFactory.shared.loadContainer(
    from: modelFolder, using: #huggingFaceTokenizerLoader())
```

Use MLX's input processors for images, videos and tools. They prepare model-specific message
content before calling the tokenizer. Do not manually prepend BOS or append EOS to a chat
prompt that the template has already formatted.

### Files and stop conditions

Keep each model's tokenizer files with its weights, including any standalone chat template
and legacy special-token map. See <doc:LoadingModelFolders> for supported layouts and metadata
precedence. Complete model folders also need MLX's model/processor configuration files.

MLX reads model generation settings, including multiple EOS IDs, separately from the
tokenizer's singular EOS metadata. Preserve `config.json` and `generation_config.json` and
let MLX combine those settings with the tokenizer. The tokenizer does not download or edit
these files.

### Migrating an existing MLX app

Swap the `Tokenizers` package product and keep the MLX loader/adaptation macros. For remote
loading, pass your existing implementation of `MLXLMCommon.Downloader` to the factory's
`from:using:configuration:` overload. Downloading and authentication stay with your app.

The supported surface is local inference, chat, token metadata and vocabulary lookup. See
<doc:MigratingFromSwiftTransformers> for API departures and known Jinja representation
limitations affecting some tool templates. Matching the exported tokenizer pipeline is the
correctness target; reproducing an upstream tokenizer bug is not a compatibility requirement.
