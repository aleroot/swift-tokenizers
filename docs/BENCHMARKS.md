# Benchmarks

The main comparison tables were measured on one machine, in one sitting, against the same inputs,
including the other implementations. The [JSON parser follow-up](#json-parser-follow-up) separately
records a later before/after experiment on an M2. The [source-offset measurements](#source-offset-encoding)
also use an M2. All figures are measurements, not projections.

The [headline](#headline), [SentencePiece](#sentencepiece) and [load time](#load-time) tables were
re-measured after the pre-tokenization and scratch-ownership work landed, with every engine run
one at a time on an otherwise idle machine. Short inputs are sensitive to CPU clock ramp: an
11.5 KB encode varies by about 15% depending on what ran before it, so the headline quotes the
1.15 MB document, which is stable to 1%.

| | |
|---|---|
| Machine | Apple M4 Pro, 24 GB, macOS 26.6.2 |
| Toolchain | Apple Swift 6.3.3, `-c release`, Swift 6 language mode |
| Concurrency | single-threaded everywhere except the "SentencePiece" and "Concurrent encoding" sections; `RAYON_NUM_THREADS=1` for the Rust cores |
| Reference builds | Hugging Face `tokenizers` 0.22.2, `transformers` 4.57.6 (CPython), `tiktoken` 0.14.0, `google/sentencepiece` @ master (C++, CMake Release), swift-transformers @ `c21fdcd` |
| Memory | `phys_footprint` delta across the load, after `malloc_zone_pressure_relief`, sampled identically in the Swift, C++ and Python harnesses |
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
- [SentencePiece](#sentencepiece)
- [Concurrent encoding](#concurrent-encoding)
- [Where the time goes](#where-the-time-goes)
- [Why it is fast](#why-it-is-fast)
- [Reproducing these numbers](#reproducing-these-numbers)
- [Methodology](#methodology)

## Headline

English prose, Qwen3 byte-level BPE (`mlx-community/Qwen3-0.6B-Base-DQ5`, 151,669 tokens,
10.9 MiB `tokenizer.json`), full pipeline: added tokens → normalization → pre-tokenization →
merges → post-processing.

| Implementation | Encode 1.15 MB | Load `tokenizer.json` | Retained |
|---|---:|---:|---:|
| **swift-tokenizers** | **230.9 MB/s** (4.1 ns/byte) | **23.0 ms** | **12.7 MB** |
| `tiktoken` 0.14 (`o200k_base`) | 55.0 MB/s | n/a | n/a |
| `tiktoken` 0.14 (`cl100k_base`) | 35.3 MB/s | n/a | n/a |
| Hugging Face `tokenizers` 0.22.2 (Rust) | 5.4 MB/s | 97.0 ms | n/a |
| swift-transformers @ `c21fdcd` | 0.92 MB/s | 297.0 ms | n/a |

**43× Hugging Face's Rust core, 4.2× tiktoken `o200k_base`, 251× swift-transformers** on the same
text on the same machine, while producing identical token ids. All five ran back to back in one
sitting; where two runs of a reference disagreed, the table quotes its faster run.

`tiktoken` is not an apples-to-apples pipeline (it runs its own regex pre-split plus merges over a
different vocabulary, with no normalizer, added tokens or post-processor); it is the fastest
widely used reference, so it is the interesting bar to clear.

## Encode throughput by input size

Same tokenizer, byte-identical inputs in every harness (68 B, 489 B, 1,150 B, 11,500 B,
1,150,000 B), median per case, all five engines run back to back in one sitting. Each engine gets
a full-clock warm-up first, because a 68-byte encode measured from a cold CPU governor reads about
20% slow. `ns/byte` is the swift-tokenizers figure.

| Input | swift-tokenizers | HF `tokenizers` | tiktoken `cl100k` | tiktoken `o200k` | swift-transformers |
|---|---:|---:|---:|---:|---:|
| short, 68 B | 0.0003 ms · 194 MB/s | 0.013 ms · 5.1 MB/s | 0.002 ms · 26.4 MB/s | 0.002 ms · 38.9 MB/s | 0.075 ms · 0.87 MB/s |
| code, 489 B | 0.0033 ms · 142 MB/s | 0.066 ms · 7.1 MB/s | 0.019 ms · 24.2 MB/s | 0.010 ms · 47.6 MB/s | 0.481 ms · 0.97 MB/s |
| medium, 1.1 KB | 0.0048 ms · 231 MB/s | 0.187 ms · 5.9 MB/s | 0.034 ms · 32.6 MB/s | 0.022 ms · 50.1 MB/s | 1.343 ms · 0.82 MB/s |
| long, 11.5 KB | 0.0463 ms · 237 MB/s | 1.827 ms · 6.0 MB/s | 0.326 ms · 33.6 MB/s | 0.210 ms · 52.2 MB/s | 13.551 ms · 0.81 MB/s |
| huge, 1.15 MB | 4.76 ms · 231 MB/s | 204.5 ms · 5.4 MB/s | 31.07 ms · 35.3 MB/s | 19.94 ms · 55.0 MB/s | 1190.4 ms · 0.92 MB/s |

Sub-microsecond calls matter as much as MB/s: a 68-byte query costs **0.33 µs**, against 13 µs for
Hugging Face and 75 µs for swift-transformers. The 11.5 KB case is the one to treat with caution:
it is short enough that CPU clock state moves it between 210 and 244 MB/s across runs.

## Encode throughput by tokenizer family

The 11.5 KB prose case, one real `tokenizer.json` per family (all outputs verified against
Hugging Face).

Median of three runs, one model loaded at a time, after a full-clock warm-up.

| Tokenizer | Model | Vocab | Throughput | ns/byte |
|---|---|---:|---:|---:|
| `pcuenq/Llama-3.2-1B-Instruct-tokenizer` | byte-level BPE | 128,256 | 249 MB/s | 3.8 |
| `mlx-community/Qwen3-0.6B-Base-DQ5` | byte-level BPE | 151,669 | 244 MB/s | 3.9 |
| `google-bert/bert-base-uncased` | WordPiece | 30,522 | 239 MB/s | 4.0 |
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | SentencePiece BPE | 32,768 | 228 MB/s | 4.2 |
| `coreml-projects/Llama-2-7b-chat-coreml` | SentencePiece BPE | 32,000 | 173 MB/s | 5.5 |
| `intfloat/multilingual-e5-small` | Unigram (XLM-R) | 250,002 | 130 MB/s | 7.3 |
| `t5-base` | Unigram | 32,128 | 113 MB/s | 8.4 |

The two Unigram models are the slowest paths here: a Viterbi lattice over `▁`-prefixed pieces with
no byte-level alphabet. On short cached passages they invert this ordering and become the fastest
family we ship (see [embedding models](#embedding-models-and-rerankers)), and against the same
tokenizer in other implementations they still lead by a wide margin
(see [SentencePiece](#sentencepiece)).

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
| Decode 11.5 KB (2,540 tokens) | 0.011 ms · 986 MB/s | 1.365 ms · 8.0 MB/s |
| Streaming decode, 200 single-token steps | 0.125 ms total (0.63 µs/step) | n/a |
| `convertIdToToken` walk of the whole 151,669-entry vocabulary | 3.06 ms (20 ns/token) | n/a |

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
| Qwen3-0.6B byte-level BPE (10.9 MiB) | 23.0 ms | 97.0 ms | 94.8 ms | 297.0 ms |
| Llama-3.2-1B byte-level BPE (16.4 MiB) | 35.4 ms | 129.8 ms | 131.4 ms | 469.0 ms |
| multilingual-e5-small Unigram, 250k pieces (16.3 MiB) | 41.6 ms | 211.9 ms | 321.2 ms | 270.2 ms |
| bert-base-uncased WordPiece, 30k (0.4 MiB) | 1.8 ms | 8.4 ms | 13.6 ms | 14.4 ms |

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
| **Total `AutoTokenizer.load`** | **23.0 ms** | **41.6 ms** |

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

### DeepSeek V3 / V4 split sequence

The DeepSeek families pre-tokenize with a `Sequence` of three `Split` regexes instead of one.
Before those three patterns had scanners they ran through `NSRegularExpression`, which also
serialised on ICU state across threads (`deepseek-ai/DeepSeek-V4.1-Flash`, 129,280-token vocab,
mixed prose corpus, M4 Pro):

| Threads | `NSRegularExpression` | Byte scanners |
|---:|---:|---:|
| 1 | 13.8 MB/s | 99.5 MB/s |
| 2 | n/a | 189 MB/s |
| 4 | n/a | 339 MB/s |
| 8 | 26.5 MB/s | 382 MB/s |

## SentencePiece

`google/sentencepiece` publishes [its own benchmark](https://github.com/google/sentencepiece/blob/master/doc/performance_benchmark.md):
FLORES-200 parallel sentences in English, Chinese, Japanese and Thai, 1,012 per language,
replicated 15 times to 11.29 MB / 60,720 lines, fed as one batch. This section reproduces that
corpus byte for byte (11,772,975 bytes of text) and runs it against the C++ engine itself rather
than its Python wrapper, so both sides measure tokenization instead of object conversion.

* **SentencePiece**: `libsentencepiece.a` built from master, `RunBatch` over its own `ThreadPool`,
  ids materialised into `std::vector<std::vector<int>>`, loading `t5-base/spiece.model` and
  `gemma-3-4b/tokenizer.model`.
* **swift-tokenizers**: the same corpus, one `[Int]` per line, the same dynamic load balancing
  across a fixed worker count, loading the Hugging Face `tokenizer.json` of the same two models.

Both take the best of five runs after a warm-up.

### Correctness

Checked before timing, on all 60,720 lines, by dumping ids from each engine and comparing the
files byte for byte:

| Model | vs SentencePiece C++ | vs Hugging Face `tokenizers` |
|---|---:|---:|
| T5 (Unigram, 32k) | **60720 / 60720 exact** | 60720 / 60720 exact |
| Gemma 3 (BPE, 262k) | **60720 / 60720 exact** | 60720 / 60720 exact |
| Qwen 3 (byte-level BPE, 152k) | n/a, no `.model` | 60720 / 60720 exact |

Identical ids from a `tokenizer.json` and from the original `.model` protobuf, including Thai and
Japanese without spaces, byte fallback and the SentencePiece charsmap normalizer. Token totals:
763,275 (T5), 1,989,660 (Gemma 3), 2,464,005 (Qwen 3).

swift-transformers is the one engine that does not reproduce these ids: on Qwen 3 it emits
2,011,230 tokens against the 2,464,005 that Hugging Face and this library both produce, so its
Qwen 3 throughput below is not measuring the same work.

### Throughput

Encoding throughput in MB/s, higher is better:

| Threads | 1 | 2 | 4 | 8 | 14 |
|---|---:|---:|---:|---:|---:|
| **T5 Unigram** SentencePiece C++ | 70.1 | 130.9 | 239.4 | 392.2 | **574.3** |
| **T5 Unigram** swift-tokenizers | **124.0** | **195.8** | **324.2** | **446.0** | 508.3 |
| **Gemma 3 BPE** SentencePiece C++ | 24.2 | 48.1 | 88.0 | 161.0 | 230.5 |
| **Gemma 3 BPE** swift-tokenizers | **43.2** | **72.2** | **122.2** | **196.6** | **241.1** |

Single-threaded, swift-tokenizers is **1.77× SentencePiece on T5 and 1.79× on Gemma 3**. It keeps
the lead at every thread count on Gemma 3, and on T5 through eight threads; at 14 threads the C++
engine passes it, because a shared Swift tokenizer still contends where a `SentencePieceProcessor`
does not. The machine has 10 performance and 4 efficiency cores.

Single-thread throughput on the same corpus for the engines that cannot read a `.model` file:
Hugging Face `tokenizers` 10.9 MB/s (T5), 12.4 MB/s (Gemma 3), 5.8 MB/s (Qwen 3); `tiktoken`
20.0 MB/s (`cl100k_base`) and 19.4 MB/s (`o200k_base`) over its own vocabulary.

### Memory

`phys_footprint` added by the load, after returning free pages to the OS, sampled the same way in
all three languages:

| Model | swift-tokenizers | SentencePiece C++ | HF `tokenizers` | swift-transformers |
|---|---:|---:|---:|---:|
| T5 (Unigram, 32k) | **7.9 MB** | 10.4 MB | 49.0 MB | 27.4 MB |
| Gemma 3 (BPE, 262k) | 65.3 MB | **56.4 MB** | 381.9 MB | 140.3 MB |
| Qwen 3 (byte-level BPE, 152k) | **21.2 MB** | n/a | 122.1 MB | 70.9 MB |

About 6× smaller than the Rust core on all three. SentencePiece is smaller on Gemma 3, where it
reads a 4.5 MiB protobuf while this library parses the equivalent 31.8 MiB `tokenizer.json`.

For scale, `sentencepiece`'s published table for the same corpus reports 27.41 MB/s (T5) and
7.44 MB/s (Gemma 3) single-threaded for itself, and 3.78 / 3.66 MB/s for Hugging Face Fast, on a
24-core machine through the Python wrapper.

### What limits the shared instance

Giving each worker its own tokenizer instance, same corpus, same harness (`--isolate`):

| Threads | 1 | 2 | 4 | 8 | 14 |
|---|---:|---:|---:|---:|---:|
| T5, one instance per worker | 121.4 | **230.5** | **417.5** | **587.4** | **823.6** |
| T5, one shared instance | **124.0** | 195.8 | 324.2 | 446.0 | 508.3 |
| T5, SentencePiece C++ | 70.1 | 130.9 | 239.4 | 392.2 | 574.3 |

The algorithms scale: with nothing shared they stay ahead of the C++ implementation at every
thread count. A shared instance now scales too, to 4.1× its own single-thread rate at 14 threads
where it used to fall back to 1.4×, but the gap to the isolated case is what reference counting
on the objects a shared tokenizer hands its threads still costs. `sample` at 14 threads puts `swift_retain` / `swift_release` at the top of
every stack, and the pattern is specific: loading a class or existential out of an `Array` on a
hot path is what collapses (measured in isolation, a three-element stage array falls from ~50 to
~1 million calls per second between 1 and 14 threads), while calling through a stored property,
or through `Unmanaged` with `_withUnsafeGuaranteedRef`, keeps scaling. Removing those array loads
from the pipeline is the next step; it needs the pre-tokenizers themselves to become the compiled
steps, because an adapter object in between costs more per call than the retain it saves.

### Load time

| Model | swift-tokenizers (`tokenizer.json`) | SentencePiece (`.model`) |
|---|---:|---:|
| T5 (32k) | **7.6 ms** (1.4 MB JSON) | 15.5 ms (0.8 MB protobuf) |
| Gemma 3 (262k) | 90 ms (33 MB JSON) | **27 ms** (4.7 MB protobuf) |

Warm file cache, median of three loads after discarding the first.

Gemma 3 is where the file format shows: 33 MB of JSON against 4.7 MB of protobuf. Reading the
`.model` protobuf directly would close that gap and is the one input format the library does not
yet accept.

## Concurrent encoding

Aggregate `encode(text:)` throughput of one shared tokenizer called from N threads
(`DispatchQueue.concurrentPerform`) over the mixed query / passage / multilingual corpus of the
embedding benchmark, native Swift strings, M4 Pro:

| Threads | Qwen3 (BPE) | Llama-2 (SentencePiece BPE) | XLM-R (Unigram) | BERT (WordPiece) |
|---:|---:|---:|---:|---:|
| 1 | 143 MB/s | 129 MB/s | 129 MB/s | 176 MB/s |
| 2 | 259 MB/s | 234 MB/s | 233 MB/s | 311 MB/s |
| 4 | 427 MB/s | 382 MB/s | 392 MB/s | 472 MB/s |
| 8 | 529 MB/s | 372 MB/s | 410 MB/s | 427 MB/s |

Before the shared reference counts were removed from the per-word paths, eight threads reached
143 MB/s on Qwen3 and 25 MB/s on XLM-R: each contended atomic on a shared object costs more than
encoding a short query.

Scratch buffers, and with them each model's encoder and its pretoken cache, are owned per thread
through a thread-specific key. An earlier version filed them in a free list striped by a hash of
the thread handle; Darwin hands out thread handles as evenly spaced stack addresses, so hashing
them collapsed 14 threads onto 4 stripes and a starved stripe rebuilt the encoder and its cache
on nearly every call. Moving to real thread-local ownership is worth 8-18% at eight threads and
nothing at all at one, which is the point: it removes contention rather than work.

The remaining gap to linear scaling is per-call ARC traffic on the tokenizer, pipeline and model
objects: see [SentencePiece](#sentencepiece), where private instances scale to 723 MB/s on the
same workload that caps at 300 MB/s when one instance is shared.

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
  (GPT-4o / gpt-oss / Muse), Falcon-H1 and DeepSeek (V2 / V3 / R1 / V4) split patterns are
  implemented as hand-written
  linear scanners over UTF-8 with an immutable runtime-derived classification cache (including the
  o200k case-aware `[Lu Lt Lm Lo M]*[Ll Lm Lo M]+` alternation). `Punctuation`, literal
  `Split` and `[0-9]` stages are byte scanners too. All are fuzz-tested against
  `NSRegularExpression` for exact equivalence; unknown patterns fall back to
  `NSRegularExpression`.
  DeepSeek is a three-stage `Sequence` (`\p{N}{1,3}`, a Han / kana class, then the main
  alternation) whose stages do not match every scalar, so those scanners emit the gaps between
  matches as well.
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
  arena-backed table with a `tryLock`; a caller that finds it taken memoises into a smaller
  table of its own, so concurrent encodes never block.
* **No shared reference counts on the hot path.** Tables read per byte or per word (vocabulary,
  merges, symbol ids, the Metaspace marker) are raw buffers owned by their model, and per-match
  appends borrow their source once. A retain on an object shared by every thread is an atomic
  on one cache line, and a handful per word was enough to make eight threads slower than one.
  Scratch free lists are striped by thread for the same reason.
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
  and reads floating-point scores with `serde_json`'s algorithm, so Unigram scores are
  bit-identical to the Rust implementation's and Viterbi ties resolve the same way.
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
| SentencePiece cross-check | T5 and Gemma 3, 4,048 FLORES-200 sentences in English, Chinese, Japanese and Thai, ids vs the C++ `.model` engine and vs Hugging Face | 4048/4048 exact, both models |
