# Benchmarks

The main comparison tables were measured on one machine, in one sitting, against the same inputs,
including the other implementations. The [JSON parser follow-up](#json-parser-follow-up) separately
records a later before/after experiment on an M2. The [source-offset measurements](#source-offset-encoding)
also use an M2. All figures are measurements, not projections.

| | |
|---|---|
| Machine | Apple M4 Pro, 24 GB, macOS 26.6.2 |
| Toolchain | Apple Swift 6.3.3, `-c release`, Swift 6 language mode |
| Concurrency | single-threaded everywhere; `RAYON_NUM_THREADS=1` for the Rust cores |
| Reference builds | Hugging Face `tokenizers` 0.22.2, `transformers` 4.57.6 (CPython), `tiktoken` 0.14.0, swift-transformers @ `c21fdcd` |
| Correctness first | every swift-tokenizers result below was compared token-for-token with Hugging Face *before* it was timed — 0 mismatches (see [Correctness](#correctness-of-the-measured-outputs)) |
| Units | `MB/s` is MiB/s (1,048,576 bytes) except in the end-to-end table, which is the harness's decimal MB/s |

Run-to-run variation on this machine is ±3% for encode throughput and ±5% for load times; the
tables quote a single representative run.

**Contents**

- [Headline](#headline)
- [Encode throughput by input size](#encode-throughput-by-input-size)
- [Encode throughput by tokenizer family](#encode-throughput-by-tokenizer-family)
- [Cold cache and non-repetitive text](#cold-cache-and-non-repetitive-text)
- [Decode and vocabulary reflection](#decode-and-vocabulary-reflection)
- [Source-offset encoding](#source-offset-encoding)
- [Load time](#load-time)
- [Memory footprint](#memory-footprint)
- [Embedding models and rerankers](#embedding-models-and-rerankers)
- [End to end through mlx-swift-lm](#end-to-end-through-mlx-swift-lm)
- [Where the time goes](#where-the-time-goes)
- [Why it is fast](#why-it-is-fast)
- [Reproducing these numbers](#reproducing-these-numbers)
- [Methodology](#methodology)

## Headline

English prose, Qwen3 byte-level BPE (`mlx-community/Qwen3-0.6B-Base-DQ5`, 151,669 tokens,
10.9 MiB `tokenizer.json`), full pipeline: added tokens → normalization → pre-tokenization →
merges → post-processing.

| Implementation | Encode 11.5 KB | Encode 1.15 MB | Load `tokenizer.json` | Retained |
|---|---:|---:|---:|---:|
| **swift-tokenizers** | **212 MB/s** (4.5 ns/byte) | **214 MB/s** | **26 ms** | **12.7 MB** |
| `tiktoken` 0.14 (`cl100k_base`) | 34.5 MB/s | 34.4 MB/s | — | — |
| `tiktoken` 0.14 (`o200k_base`) | 52.2 MB/s | 52.8 MB/s | — | — |
| Hugging Face `tokenizers` 0.22.2 (Rust) | 6.7 MB/s | 5.4 MB/s | 99 ms | — |
| swift-transformers @ `c21fdcd` | 0.94 MB/s | 0.94 MB/s | 299 ms | — |

**32–40× Hugging Face's Rust core, 4–6× tiktoken, 225× swift-transformers** on the same text on
the same machine — while producing identical token ids. Where two runs of a reference disagreed,
the table quotes its faster run.

`tiktoken` is not an apples-to-apples pipeline (it runs its own regex pre-split plus merges over a
different vocabulary, with no normalizer, added tokens or post-processor); it is the fastest
widely used reference, so it is the interesting bar to clear.

## Encode throughput by input size

Same tokenizer, same corpus shape as `Tests/Benchmarks` (median of 3–200 iterations per case,
warm-up excluded). `ns/byte` is the swift-tokenizers figure.

| Input | swift-tokenizers | HF `tokenizers` | tiktoken `cl100k` | tiktoken `o200k` | swift-transformers |
|---|---:|---:|---:|---:|---:|
| short — 68 B | 0.001 ms · 125 MB/s | 0.011 ms · 5.8 MB/s | 0.002 ms · 28.8 MB/s | 0.002 ms · 38.9 MB/s | 0.076 ms · 0.9 MB/s |
| code — 563 B | 0.004 ms · 125 MB/s | 0.057 ms · 8.1 MB/s | 0.019 ms · 24.8 MB/s | 0.010 ms · 47.6 MB/s | 0.473 ms · 1.0 MB/s |
| medium — 1.1 KB | 0.005 ms · 202 MB/s | 0.166 ms · 6.6 MB/s | 0.033 ms · 33.0 MB/s | 0.022 ms · 50.8 MB/s | 1.196 ms · 0.9 MB/s |
| long — 11.5 KB | 0.052 ms · 212 MB/s | 1.639 ms · 6.7 MB/s | 0.318 ms · 34.5 MB/s | 0.210 ms · 52.2 MB/s | 11.712 ms · 0.9 MB/s |
| huge — 1.15 MB | 5.11 ms · 214 MB/s | 203.6 ms · 5.4 MB/s | 31.9 ms · 34.4 MB/s | 20.8 ms · 52.8 MB/s | 1171.0 ms · 0.9 MB/s |

Sub-microsecond calls matter as much as MB/s: a 68-byte query costs **1 µs**, against 11 µs for
Hugging Face and 76 µs for swift-transformers.

## Encode throughput by tokenizer family

The 11.5 KB prose case, one real `tokenizer.json` per family (all outputs verified against
Hugging Face).

| Tokenizer | Model | Vocab | Throughput | ns/byte |
|---|---|---:|---:|---:|
| `pcuenq/Llama-3.2-1B-Instruct-tokenizer` | byte-level BPE | 128,256 | 215 MB/s | 4.5 |
| `mlx-community/Qwen3-0.6B-Base-DQ5` | byte-level BPE | 151,669 | 212 MB/s | 4.5 |
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | SentencePiece BPE | 32,768 | 199 MB/s | 4.8 |
| `google-bert/bert-base-uncased` | WordPiece | 30,522 | 198 MB/s | 4.9 |
| `coreml-projects/Llama-2-7b-chat-coreml` | SentencePiece BPE | 32,000 | 131 MB/s | 7.3 |
| `intfloat/multilingual-e5-small` | Unigram (XLM-R) | 250,002 | 272–323 MB/s † | 3.1–3.5 |
| `t5-base` | Unigram | 32,128 | 95 MB/s | 10.1 |

† Measured by the embedding harness below (620 B passages / 2.4 KB passages) rather than the
in-repo suite; XLM-R Unigram is the fastest family we ship once its pieces are cached.

T5's Unigram lattice is the slowest path: 32k pieces with heavy `▁`-prefixed segmentation and no
byte-level alphabet. It is still 15× Hugging Face's Rust implementation of the same tokenizer.

## Cold cache and non-repetitive text

swift-tokenizers memoises pre-token → ids in a bounded arena, so corpora that repeat themselves
(the in-repo benchmark paragraph, a RAG passage batch, a chat log) are served partly from cache.
Two honest worst cases, both 4.0 MB encoded as 129 × 32 KB chunks, both verified to produce
*exactly* the same token counts as Hugging Face:

* **Repeated prose** — the benchmark paragraph repeated to 4 MB, encoded twice (cold = freshly
  loaded tokenizer, warm = second pass).
* **Non-repetitive prose** — synthetic text in which every sentence carries two unique tokens
  (`item1234`, `unique9738241`), so almost nothing can be cached. It is digit-heavy, which is also
  the worst case for WordPiece and Unigram segmentation.

| Family | Repeated, cold | Repeated, warm | Non-repetitive | HF `tokenizers` (non-repetitive) | tiktoken `cl100k` |
|---|---:|---:|---:|---:|---:|
| byte-level BPE (Qwen3) | 183 MB/s | 248 MB/s | 191 MB/s | 5.1 MB/s | 34.0 MB/s |
| Unigram XLM-R (e5-small) | 137 MB/s | 135 MB/s | 103 MB/s | 7.3 MB/s | — |
| WordPiece (BERT) | 187 MB/s | 189 MB/s | 70 MB/s | 5.9 MB/s | — |

The cache is worth up to **1.35×** on BPE prose and nothing at all when words are in-vocabulary
(one hash probe either way). Even with the cache cold and the text adversarial, the library stays
**12–37×** ahead of Hugging Face's Rust core.

Token counts on the non-repetitive corpus, ours vs Hugging Face: BPE 1,394,064 = 1,394,064;
Unigram 1,108,547 = 1,108,547; WordPiece 1,108,377 = 1,108,377.

## Decode and vocabulary reflection

| Operation | swift-tokenizers | swift-transformers |
|---|---:|---:|
| Decode 11.5 KB (2,540 tokens) | 0.025 ms · 439 MB/s | 1.374 ms · 8.0 MB/s |
| Streaming decode, 200 single-token steps | 0.23 ms total (1.2 µs/step) | — |
| `convertIdToToken` walk of the whole 151,669-entry vocabulary | 3.25 ms (21 ns/token) | — |

The vocabulary walk is the pattern MLX guided generation uses to discover the token space: it must
be O(1) per id and dense, so 150k lookups cost as much as one 11.5 KB encode.

## Source-offset encoding

Measured September 10, 2026 on Apple M2, 16 GB, macOS 26.6.2, Apple Swift 6.3.3,
release builds, on one thread.

Each input is a paragraph repeated 100 times. Each cell is microseconds per call: median of
six batches of five calls after discarding the first batch. These are warm-cache measurements.

| Tokenizer | English IDs | Multilingual IDs | Multilingual IDs + offsets |
|---|---:|---:|---:|
| GPT-2 | 58.4 | 104.3 | 603 |
| Qwen3 | 64.1 | 104.9 | 426 |
| BERT uncased | 38.6 | 187.4 | 1,800 |
| T5 small | 81.0 | 191.4 | 1,504 |
| Llama 7B | 92.4 | 51.0 | 1,208 |
| Mistral v0.3 | 61.1 | 44.8 | 713 |

Offset tracking costs more than IDs-only encoding. These measurements exclude UTF-16 range
conversion and grapheme expansion.

The English paragraph is “Maya checked the timetable, bought a ticket, and walked to platform
seven. Rain tapped against the glass roof. ” The multilingual paragraph is
“Café naïve — Ελληνικά 中文 العربية हिन्दी 한국어 👩🏽‍💻. ”, including its trailing space.

## Load time

Median of 5 loads of the same folder, warm filesystem cache.

| Tokenizer (`tokenizer.json`) | swift-tokenizers | HF `tokenizers` 0.22.2 | `transformers` 4.57.6 | swift-transformers |
|---|---:|---:|---:|---:|
| Qwen3-0.6B byte-level BPE (10.9 MiB) | 25.9 ms | 98.9 ms | 97.3 ms | 299.2 ms |
| Llama-3.2-1B byte-level BPE (16.4 MiB) | 34.6 ms | 128.7 ms | 134.8 ms | 475.6 ms |
| multilingual-e5-small Unigram, 250k pieces (16.3 MiB) | 49.9 ms | 217.5 ms | 326.7 ms | 273.8 ms |
| bert-base-uncased WordPiece, 30k (0.4 MiB) | 1.7 ms | 8.7 ms | 14.4 ms | 14.5 ms |

Stage breakdown for the two interesting cases:

| Stage | Qwen3 BPE (10.9 MiB) | XLM-R Unigram (16.3 MiB, 250,002 pieces) |
|---|---:|---:|
| Parse + pack `tokenizer.json` (`Config(tokenizerJSON:)`) | 22.7 ms | 22.7 ms |
| — for comparison: `Config(jsonData:)` | 33.6 ms | — |
| — for comparison: `JSONSerialization` alone | 54.7 ms | — |
| Build packed vocabulary | (in parse) | 6.5 ms |
| Build double-array trie | — | 20.0 ms (785,408 units, 9.2 MB) |
| Build `Precompiled` charsmap normalizer | — | 0.3 ms |
| Expand generated Unicode tables (historical implementation) | 0.05 ms | 0.05 ms |
| **Total `AutoTokenizer.load`** | **25.9 ms** | **49.9 ms** |

### JSON parser follow-up

Measured on an **Apple M2, macOS 26.6.2, Swift 6.3.3**, using release builds. The baseline is
the parser at `2ca1f36`; the optimized parser uses an eight-byte word scan for short strings,
16-byte SIMD scans for longer runs, and direct UTF-8 validation before creating Swift strings.
Ordinary strings and packed vocabulary entries now share one escape decoder.

Five baseline/optimized pairs ran in alternating order from preserved binaries, with no builds
running during measurement. Each figure is the median of five per-run medians, with 15 timed
iterations and three warm-ups per run. Model loads use warm filesystem and runtime caches.

| Tokenizer | Packed parse, before → after | Complete load, before → after |
|---|---:|---:|
| `mlx-community/Qwen3-0.6B-Base-DQ5` | 15.590 → 15.393 ms | 32.509 → 32.267 ms |
| `intfloat/multilingual-e5-small` | 35.535 → 34.573 ms | 72.869 → 71.757 ms |
| `google-bert/bert-base-uncased` | 0.975 → 0.979 ms | 2.203 → 2.191 ms |

The complete-load differences are only 0.5–1.5%, within ordinary timing variation. These
measurements do **not** establish a meaningful end-to-end loading speedup. String-heavy synthetic
inputs show where the change helps: each case contains 2,048 scored vocabulary entries.

| Packed JSON string workload | Before | After | Speedup |
|---|---:|---:|---:|
| Short ASCII, 5 bytes/string | 0.101 ms | 0.095 ms | 1.06× |
| Short Unicode, 11 bytes/string | 0.123 ms | 0.112 ms | 1.10× |
| Long ASCII, 1,024 bytes/string | 2.087 ms | 0.317 ms | 6.58× |
| Long Unicode, 960 bytes/string | 3.144 ms | 1.572 ms | 2.00× |
| Escaped strings, 768 decoded bytes/string | 3.928 ms | 3.685 ms | 1.07× |

Generic JSON parsing of the long ASCII and Unicode cases improved by 2.88× and 2.16× respectively.
The implementation remains pure Swift, uses bounded unaligned loads, and requires no input padding.
Run the same workloads with `RUN_BENCHMARKS=1 swift test -c release --filter JSONBenchmarkTests`.

### Lampo workload follow-up

Measured on **Apple M2, macOS 26.6.2, Swift 6.3.3**, in release mode against `91961b0`.
These replay Lampo's tokenizer calls for streaming, token inspection, and prefixed retrieval
queries/passages; they do not measure the whole app or GPU inference. Five pairs of preserved
binaries ran in alternating order without concurrent builds. Values are medians of per-run
medians: nine fresh tokenizer instances for first decode, 15 iterations for warm workloads,
and five iterations for repeated large documents. Loads use cached local files.

| Qwen3-0.6B workload | Before | After |
|---|---:|---:|
| First decode, 16 IDs | 14.292 ms | 3.837 ms |
| Inspect each paragraph token separately | 0.027 ms | 0.027 ms |
| Decode every growing paragraph prefix | 0.148 ms | 0.148 ms |

Decode-table construction now initializes added-token flags directly from IDs and avoids
allocating and hashing a String for each vocabulary entry. Empty decoding leaves the table
uninitialized. Vocabulary loading also preserves exact serialized added-token spellings and IDs
when Swift's canonical String equality would otherwise combine distinct entries.

| Tokenizer | 256 queries, before → after | 64 passages, before → after | 2.24 MiB document, before → after |
|---|---:|---:|---:|
| `mlx-community/Qwen3-0.6B-Base-DQ5` | 0.183 → 0.184 ms | 0.250 → 0.248 ms | 13.739 → 13.896 ms |
| `intfloat/multilingual-e5-small` | 0.215 → 0.212 ms | 0.196 → 0.192 ms | 14.814 → 14.804 ms |
| `google-bert/bert-base-uncased` | 0.208 → 0.204 ms | 0.167 → 0.157 ms | 12.170 → 12.274 ms |

Queries repeat the same short question; passages repeat a numbered 64-item batch. Both use warm
pretoken caches. Scratch buffers remain reusable for inputs up to 1 MiB and are released after
larger inputs. The measured repeated-document cost is about 1% for Qwen/BERT, with E5 unchanged.
Normal query/passage throughput remains similar; the small timing differences are not broad
speedup claims.

After encoding the 2,351,104-byte document and discarding its output, baseline live-heap growth
was **12.0 MiB Qwen, 20.2 MiB E5, and 14.5 MiB BERT**. The new version had no retained growth
(deltas of −3.8, −10.5, and −3.5 KiB as earlier small buffers were released). This uses
`malloc_zone_statistics` in fresh processes running only the encoding benchmark; it measures
live allocations, not peak memory or pages immediately returned to the OS. The input threshold
is a scratch-reuse policy, not a strict total-memory limit.

Lampo's Jina fallback loader, used when `tokenizer_config.json` is absent, now uses
`Config(tokenizerJSONFile:)`. Replaying its construction with the Qwen fixture already in memory
took **803.515 ms through JSONSerialization + generic Config versus 30.976 ms through packed
Config**, about **26× faster** (one run, nine timed iterations after three warm-ups per path).
This is a loader-path comparison on a Qwen-sized vocabulary, not a measured Jina model load.
The normal configured loader already uses packed parsing.

Run `RUN_BENCHMARKS=1 swift test -c release --filter LampoWorkloadBenchmarks` for timing.
For retained-heap measurements, run only `LampoWorkloadBenchmarks/encoding` in a fresh process
so unrelated Foundation work cannot affect the allocation deltas.

## Memory footprint

Live heap and `phys_footprint` retained after `AutoTokenizer.load`, measured with `task_info` in a
warm process (several tokenizers loaded in sequence, as in an app).

| Tokenizer | `tokenizer.json` | Live heap | `phys_footprint` |
|---|---:|---:|---:|
| Qwen3-Embedding-0.6B (byte-level BPE) | 10.9 MiB | +12.7 MB | +16.6 MB |
| Qwen3-Reranker-0.6B (byte-level BPE) | 10.9 MiB | +11.0 MB | +8.5 MB |
| Llama-3.2-1B-Instruct (byte-level BPE) | 16.4 MiB | +16.5 MB | +19.8 MB |
| multilingual-e5-small (Unigram, 250k) | 16.3 MiB | +19.1 MB | +55.8 MB † |
| bge-m3 / bge-reranker-v2-m3 (Unigram, 250k) | 16.3 MiB | +19.2 MB | +19.5 MB |
| jina-reranker-v2-base-multilingual (Unigram, 250k) | 16.3 MiB | +19.1 MB | +19.6 MB |
| bert-base-uncased (WordPiece, 30k) | 0.4 MiB | +1.1 MB | +0.6 MB |
| nomic-embed-text-v1.5 (WordPiece, 30k) | 0.7 MiB | +1.0 MB | +0.2 MB |

† The first Unigram loaded in a process keeps dirty pages from the trie builder's peak
allocations; every later one settles at ~19.5 MB.

The two large structures are the packed vocabulary (2,581,345 bytes of token UTF-8 plus a 4-byte
offset per piece and a byte-hash index) and the double-array trie (785,408 × 12-byte units). The
pre-token cache reserves a 2.5 MB arena that is filled lazily, and decode tables are built on
first use — a tokenizer that only ever encodes never pays for them.

## Embedding models and rerankers

Short queries and RAG passages through the tokenizers of popular on-device embedding and reranking
models. Corpus: 16 queries (≈35 B each), 16 passages (≈620 B), 4 long passages (≈2.4 KB),
13 multilingual texts (Thai, Devanagari, combining marks, ZWJ emoji, ≈216 B), plus a 512-passage
batch (318 KB). Per-text medians; the Hugging Face column is the same corpus through Rust
`tokenizers` 0.22.2 with `RAYON_NUM_THREADS=1`.

| Model | Type | Query | Passage | Long passage | Multilingual | 512-passage batch | HF passage | Speed-up |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bge-small-en-v1.5 | WordPiece | 0.5 µs | 2.0 µs · 300 MB/s | 6.5 µs · 349 MB/s | 1.7 µs · 120 MB/s | 1.05 ms · 288 MB/s | 108.7 µs · 5.4 MB/s | 54× |
| all-MiniLM-L6-v2 | WordPiece | 0.5 µs | 2.1 µs · 281 MB/s | 6.5 µs · 352 MB/s | 1.7 µs · 124 MB/s | 1.03 ms · 295 MB/s | 113.1 µs · 5.2 MB/s | 54× |
| nomic-embed-text-v1.5 | WordPiece | 0.4 µs | 1.9 µs · 306 MB/s | 6.5 µs · 348 MB/s | 1.6 µs · 126 MB/s | 1.04 ms · 293 MB/s | 108.5 µs · 5.5 MB/s | 57× |
| ms-marco-MiniLM-L-6-v2 | WordPiece | 0.4 µs | 2.1 µs · 281 MB/s | 6.5 µs · 350 MB/s | 1.7 µs · 124 MB/s | 1.03 ms · 294 MB/s | 110.3 µs · 5.4 MB/s | 53× |
| bert-base-uncased | WordPiece | 0.4 µs | 1.9 µs · 306 MB/s | 6.4 µs · 356 MB/s | 1.6 µs · 127 MB/s | 1.04 ms · 292 MB/s | 109.4 µs · 5.4 MB/s | 58× |
| multilingual-e5-small | Unigram (XLM-R) | 0.5 µs | 2.2 µs · 272 MB/s | 7.0 µs · 323 MB/s | 1.8 µs · 114 MB/s | 1.16 ms · 261 MB/s | 71.3 µs · 8.3 MB/s | 32× |
| bge-m3 | Unigram (XLM-R) | 0.5 µs | 2.4 µs · 246 MB/s | 7.4 µs · 305 MB/s | 1.9 µs · 111 MB/s | 1.17 ms · 258 MB/s | 71.2 µs · 8.3 MB/s | 30× |
| bge-reranker-v2-m3 | Unigram (XLM-R) | 0.5 µs | 2.2 µs · 271 MB/s | 7.2 µs · 314 MB/s | 1.9 µs · 108 MB/s | 1.19 ms · 256 MB/s | 70.8 µs · 8.4 MB/s | 32× |
| jina-reranker-v2-base-multilingual | Unigram (XLM-R) | 0.6 µs | 3.5 µs · 171 MB/s | 12.8 µs · 178 MB/s | 2.2 µs · 95 MB/s | 1.85 ms · 164 MB/s | 95.8 µs · 6.2 MB/s | 27× |
| Qwen3-Embedding-0.6B | byte-level BPE | 0.7 µs | 3.1 µs · 188 MB/s | 10.6 µs · 214 MB/s | 2.1 µs · 100 MB/s | 1.69 ms · 179 MB/s | 104.6 µs · 5.7 MB/s | 34× |
| Qwen3-Reranker-0.6B | byte-level BPE | 0.4 µs | 3.0 µs · 198 MB/s | 11.3 µs · 201 MB/s | 1.6 µs · 125 MB/s | 1.56 ms · 194 MB/s | 101.6 µs · 5.8 MB/s | 34× |
| Llama-3.2-1B-Instruct | byte-level BPE | 0.7 µs | 3.3 µs · 179 MB/s | 10.6 µs · 214 MB/s | 1.5 µs · 137 MB/s | 1.62 ms · 187 MB/s | 91.0 µs · 6.5 MB/s | 28× |

Load times for the same models: 1.6–1.8 ms (WordPiece, 0.4–0.7 MiB), 22.5–22.7 ms (Qwen3 BPE,
10.9 MiB), 34.6 ms (Llama-3.2 BPE, 16.4 MiB), 48.8–49.2 ms (XLM-R Unigram, 16.3 MiB).

A reranker scoring 100 candidates spends well under a millisecond tokenizing them; a
multilingual-e5-small index of 1,000 passages takes ~2.2 ms.

## End to end through mlx-swift-lm

Real mlx-community model folders, loaded through mlx-swift-lm's `#huggingFaceTokenizerLoader()`
macro and the `MLXLMCommon.Tokenizer` bridge — i.e. exactly the path an app takes. Same source
compiled twice: once against swift-tokenizers, once against swift-transformers. Document: 8.8 MB
→ 2.42 M tokens; corpus: 301 texts per model, every one compared with a Hugging Face golden.

| Model folder | swift-tokenizers | swift-transformers | swift-transformers vs Hugging Face |
|---|---:|---:|---|
| gemma-4-12B-it-qat-OptiQ-4bit | 190 MB/s · load 122 ms | 1.4 MB/s · load 1002 ms | no BOS on 301/301 texts, 54 decode errors, wrong chat template |
| Muse-Glimmer-30B-4bit (o200k) | 188 MB/s · load 111 ms | 0.5 MB/s · load 1002 ms | 45/301 decode errors |
| Qwen3.5-9B-8bit | 172 MB/s · load 88 ms | 0.8 MB/s · load 651 ms | — |
| Qwen3-14B-4bit | 170 MB/s · load 48 ms | 0.9 MB/s · load 353 ms | 1/301 encode errors (Thai combining marks) |
| DeepSeek-R1-Distill-Qwen-14B-4bit | 167 MB/s · load 65 ms | 0.8 MB/s · load 330 ms | 1/301 encode errors |
| Falcon-H1R-7B-8bit | 80 MB/s · load 57 ms | 0.6 MB/s · load 364 ms | — |

swift-tokenizers matched the Hugging Face goldens on all 301 texts of all six models, for both
`encode(addSpecialTokens:)` variants and both `decode(skipSpecialTokens:)` variants. It also
rendered tool-calling chat templates in Hugging Face's canonical key order, where
swift-transformers matched only after alphabetically sorting the tool keys (3 of 6 models).

Falcon-H1R is the slowest family we ship: its split pattern has the most alternations of the
hand-written scanners.

## Where the time goes

115 KB of prose through the Qwen3 pipeline, per stage:

| Stage | Time | Throughput |
|---|---:|---:|
| Pre-tokenizer scanner alone (`qwen2` pattern, 22,400 pieces) | 0.460 ms | 238 MB/s (4.0 ns/byte) |
| Full `encode(text:)` — added tokens, NFC proof, merges, post-processing | 0.526 ms (p50) | 213 MB/s |

The split *is* the hot path: everything after it (merge resolution through the flat
`(leftId, rightId)` table, cache probes, id appends) adds ~14%. That is why the scanners are
hand-written over UTF-8 with NEON lane masks instead of being expressed as regular expressions.

## Why it is fast

* **One byte-level pipeline.** Encoding never materialises intermediate strings: added-token
  splitting, every normalizer (`BertNormalizer`, `Precompiled` charsmaps, NFC/NFD, `Strip`,
  `Replace`, …) and every pre-tokenizer (`Metaspace`, `BertPreTokenizer`, `Whitespace`,
  `Punctuation`, `Digits`, `Split`, `ByteLevel`) operate on `UnsafeBufferPointer<UInt8>` with
  pooled scratch buffers. Normalizers report when they would leave a chunk unchanged so the
  common ASCII case is not even copied; mostly-ASCII text with a few accents or dashes only
  sends the non-ASCII runs through Foundation. The public `String` APIs are derived from the
  same byte implementations, so each rule exists once.
* **Double-array tries.** Unigram (SentencePiece) segmentation runs the Viterbi lattice on
  UTF-8 offsets and enumerates candidate pieces with a static double-array trie: one 12-byte
  unit per node, one load per input byte, no hashing. The trie is built at load time with an
  in-place radix partition fused with node placement (~20 ms for XLM-R's 250k pieces, 785,408
  units for 250,002 keys). The same structure matches added tokens.
* **Integer BPE.** Symbols are token ids, not strings; merges are resolved through a flat
  open-addressing table keyed by `(leftId, rightId)`. A byte-level pretoken goes from raw UTF-8
  bytes to symbol ids through a 256-entry table — no `String` is ever built.
* **Regex-free pre-tokenization.** The GPT-2, Llama-3/cl100k, Qwen-2, Qwen-3.5, o200k
  (GPT-4o / gpt-oss / Muse) and Falcon-H1 split patterns are implemented as hand-written
  linear scanners over UTF-8 with an immutable runtime-derived classification cache (including the
  o200k case-aware `[Lu Lt Lm Lo M]*[Ll Lm Lo M]+` alternation). `Punctuation`, literal
  `Split` and `[0-9]` stages are byte scanners too. All are fuzz-tested against
  `NSRegularExpression` for exact equivalence; unknown patterns fall back to
  `NSRegularExpression`.
* **SIMD byte kernels.** ASCII classification runs 16 bytes per NEON register:
  `SIMD16<UInt8>` lane masks for controls, case, whitespace, punctuation, digits and `\w`,
  first-non-ASCII / first-of-two-bytes scans, adjacent-repeat detection and in-place
  lowercasing. Swift's own `SIMD` comparisons and reductions lower to scalar loops, so the
  kernels build `0x80` lane masks from adds and XORs and reduce them as two 64-bit words.
  Pre-tokenizers classify a whole chunk at once and then visit only the lanes where the class
  changes; the added-token splitter jumps to token first bytes with `memchr` and backs up over
  the preceding whitespace run.
* **Normalization fast paths.** Generated BMP tables hold the canonical combining
  class, NFC / NFD / NFKC / NFKD quick-check status, canonical decompositions (Hangul computed
  algorithmically) and simple lowercase mappings, plus the sparse ranges of non-trivial
  supplementary scalars. The Unicode-form normalizers therefore *prove* text is already
  normalized (all everyday input) and copy it, `BertNormalizer` strips accents and lowercases
  BMP runs scalar by scalar, and SentencePiece `Precompiled` charsmaps copy control-free ASCII
  runs verbatim and map non-ASCII runs cluster by cluster. Only scalars outside the tables —
  supplementary-plane marks, version-dependent characters, multi-scalar lowercase mappings —
  fall back to Foundation's public ICU transforms and the Swift standard library. A small
  compatibility table derived from Unicode 14–17 makes normalization conservative for scalars
  whose properties changed. Classification uses the runtime directly when building its caches.
  Tests check every BMP scalar and every
  non-trivial supplementary scalar against the runtime without assuming that all OS versions
  ship identical Unicode data. Regenerate the compatibility ranges with
  `python3 scripts/unicode_compatibility.py --ucd-dir <ucd-dir>`; the script pins its input checksums.
* **Zero-allocation hot path.** Added tokens are found with a double-array trie (and `memchr`
  when they share a first byte), sections and pieces are byte ranges, model encoders and their
  working state (Viterbi lattice, merge buffers, WordPiece scratch) are pooled with the
  per-call scratch, and the pretoken → ids cache (WordPiece, BPE and Unigram) is an
  arena-backed table with a `tryLock` so concurrent encodes never block.
  Scratch used by inputs larger than 1 MiB is released so document-sized outliers do not
  permanently enlarge a live tokenizer's buffer pool.
* **WordPiece on bytes.** A word costs one hash probe when it is in the vocabulary; otherwise
  candidates are bounded by the longest vocabulary entry, the `##` continuation is assembled
  once per position, words starting with a scalar no token begins with are rejected
  immediately, and ids are appended straight into the result (rolled back if the word turns
  out not to be segmentable).
* **Packed vocabulary.** Tokens live in one UTF-8 buffer with an offsets table (O(1)
  `convertIdToToken`) and a byte-hash index that is *binary-distinct* — `"à"` (U+00E0) and
  `"a\u{300}"` are different tokens, as they are in every real vocabulary. Every table is
  allocated at its exact size, filled once and then read through raw pointers, so a probe is a
  hash, two loads and a `memcmp` with no reference counting, exclusivity checking or
  copy-on-write traffic; presence costs one bit per id.
* **Fast configuration loading.** A purpose-built JSON parser produces `Config` trees directly
  (2.4× faster than `JSONSerialization` on a 10.9 MiB `tokenizer.json`, and 1.5× faster than
  the generic `Config(jsonData:)` path) and packs vocabularies and merges into flat buffers,
  with `Double` precision so Unigram scores round exactly like the Rust implementation.
  Unicode classification tables ship as 2.6k run-length entries and the normalization tables as
  2.5k runs plus 1.5k decompositions; all expand in ~0.05 ms instead of querying scalar
  properties at launch.
* **Small footprint.** Merge tables index dense entries through 32-bit slots, caches store
  32-bit ids, and decode tables are built lazily — a Qwen3 tokenizer retains ~13 MB, a BERT
  tokenizer ~1 MB including its word cache.
* **Exact SentencePiece chunking.** Llama-2-style tokenizers treat a whole text section as one
  BPE word. Every BPE output token is a merge product, so a merge can only cross a `▁` word
  boundary if some product has an interior `▁`; those products (gemma-4 has exactly one,
  `>▁</`) are checked at each boundary and the section is otherwise encoded chunk by chunk
  with cache hits — a 10× speed-up with identical output.

## Reproducing these numbers

The in-repo suite downloads its fixtures to `~/Library/Caches/swift-tokenizers-tests` on first
run and prints the encode-by-size, encode-by-family, decode, vocabulary-walk, load and stage
tables:

```sh
RUN_BENCHMARKS=1 swift test -c release --filter Benchmarks
```

The reference columns time the same `tokenizer.json` on the same text, single-threaded:

```python
import os, statistics, time
os.environ["RAYON_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tokenizers import Tokenizer
import tiktoken

paragraph = "Byte-pair encoding (BPE) is a tokenization algorithm …\n\n"  # as in Tests/Benchmarks
texts = {"long": paragraph * 20, "huge": paragraph * 2000}
hf = Tokenizer.from_file("~/Library/Caches/swift-tokenizers-tests/mlx-community/Qwen3-0.6B-Base-DQ5/tokenizer.json")
tk = tiktoken.get_encoding("cl100k_base")

def mbps(fn, text, iterations):
    for _ in range(3): fn(text)
    samples = []
    for _ in range(iterations):
        t = time.perf_counter(); fn(text); samples.append(time.perf_counter() - t)
    return len(text.encode()) / 1048576 / statistics.median(samples)

for name, text in texts.items():
    print(name, f"HF {mbps(lambda s: hf.encode(s, add_special_tokens=False).ids, text, 10):.1f} MB/s",
          f"tiktoken {mbps(lambda s: tk.encode(s, disallowed_special=()), text, 10):.1f} MB/s")
```

The swift-transformers column comes from a SwiftPM executable with a path dependency on a
swift-transformers checkout that loads the same fixture folder with
`AutoTokenizer.from(modelFolder:)` and times `encode(text:addSpecialTokens: false)` on the same
strings with `clock_gettime_nsec_np(CLOCK_UPTIME_RAW)`.

The embedding, reranker and end-to-end tables come from local harnesses that need model folders
and an mlx-swift-lm checkout: each verifies every corpus text against Hugging Face goldens before
timing query / passage / batch encoding, load latency and `phys_footprint`; the end-to-end one is
compiled once against swift-tokenizers and once against swift-transformers and drives the real
`#huggingFaceTokenizerLoader()` macro over mlx-community folders.

## Methodology

* **Run-to-run variation.** ±3% for encode throughput, ±5% for load times. Hugging Face's load of
  the Qwen3 file measured 92.6 ms and 98.9 ms in two runs; the tables quote the slower figure, and
  the encode tables quote its faster run.
* **Timing.** `clock_gettime_nsec_np(CLOCK_UPTIME_RAW)` around each call (Swift) and
  `time.perf_counter()` (Python); 3 warm-up calls, then the median of 3–400 iterations depending
  on the case. The in-repo suite prints mean, standard deviation, p50 and p95; the tables quote
  the mean, which is within 3% of p50 for every case except `encode(text:)` on the 115 KB stage
  breakdown, where p50 is quoted.
* **One thread.** No parallel encoding anywhere. Hugging Face's Rust core is pinned with
  `RAYON_NUM_THREADS=1` and `TOKENIZERS_PARALLELISM=false`; its `encode_batch` (which would use
  rayon) is deliberately not used, so the comparison is per-call latency, which is what an app
  experiences.
* **Same bytes.** All implementations read the same `tokenizer.json` and encode the same strings.
  `tiktoken` cannot load a Hub tokenizer, so it runs its own `cl100k_base` / `o200k_base`
  vocabularies over the same text — a different tokenizer, not a different result.
* **Warm caches, cold process.** Fixtures and model folders are on a local SSD and read before
  timing. Unicode tables, trie and vocabulary construction are inside the load measurement, not
  amortised away.
* **Memory.** `task_info(TASK_VM_INFO).phys_footprint` and `malloc_zone_statistics.size_in_use`
  deltas around a full load, with the parsed configuration released first (as it is after
  `AutoTokenizer.load`). Per-component heap deltas printed by the in-repo memory suite are noisy —
  the allocator returns pages lazily — so this page quotes whole-tokenizer figures.
* **Correctness gates.** Timing runs only after outputs match: the differential suite (24
  tokenizers × 301 adversarial texts against Hugging Face goldens), the embedding harness
  (12 models × 68 texts, 0 mismatches) and the end-to-end harness (6 model folders × 301 texts,
  encode and decode, both special-token modes).

### Correctness of the measured outputs

| Check | Scope | Result |
|---|---|---|
| `DifferentialTests` | 24 tokenizer families × 301 adversarial texts, ids + decoded text vs Hugging Face `transformers` goldens | exact |
| Embedding harness | 12 embedding/reranker models × 68 texts (queries, passages, long passages, multilingual, code, rerank pairs, edge cases) | 816/816 exact |
| End-to-end harness | 6 mlx-community folders × 301 texts, `encode`/`decode` × special-token modes, chat templates with and without tools | exact |
| Cold-corpus cross-check | 4.0 MB of unseen synthetic prose, token counts vs Hugging Face, 3 families | identical counts |
