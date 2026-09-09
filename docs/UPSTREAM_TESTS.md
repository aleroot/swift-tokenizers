# Unicode correctness and upstream coverage

Audit date: September 9, 2026. Local platform: Apple M2, macOS 26.6.2, Swift 6.3.3.
The Python reference is the Rust-backed `tokenizers==0.23.2` package in an isolated environment.

## Which implementation is correct?

The authority for NFC/NFD/NFKC/NFKD is [Unicode Standard Annex #15](https://www.unicode.org/reports/tr15/tr15-57.html),
with the matching version's `NormalizationTest.txt` and `UnicodeData.txt`. Neither library
is an independent specification. Unicode normalization stability applies to characters
assigned in the older version; newly assigned characters can legitimately change results.

Hugging Face's [v0.23.2 Cargo.lock](https://github.com/huggingface/tokenizers/blob/v0.23.2/bindings/python/Cargo.lock)
pins `unicode-normalization-alignments` 0.1.12. That crate's `src/tables.rs` declares
`UNICODE_VERSION = (9, 0, 0)`. The wheel passes all 374,440 listed invariants in the
[Unicode 9 corpus](https://www.unicode.org/Public/9.0.0/ucd/NormalizationTest.txt).
That does not establish conformance to later Unicode versions. For example, its NFKC
leaves U+32FF (SQUARE ERA NAME REIWA) unchanged; modern Unicode requires U+4EE4 U+548C (令和).
The package also uses separate dependencies for categories and lowercase, so it should not
be described as uniformly implementing Unicode 9 in every component.

Foundation's NSString normalization properties have algorithm defects as well as different
data coverage. On the tested OS:

- NFC incorrectly maps U+1100 U+1176 to U+AE4C. The archaic vowel is outside the modern
  Hangul composition range and the input must remain unchanged.
- NFC fails to compose the last two scalars of U+1100 U+AC00 U+11A8.
- NFKC of U+01C4 U+0323 returns D + Ž + dot below, instead of D + Ẓ + caron.
  These strings are canonically equivalent, so ordinary Swift String equality hides the defect.

The [open CoreFoundation source](https://github.com/swiftlang/swift-corelibs-foundation/blob/main/Sources/CoreFoundation/CFString.c)
contains inclusive Hangul bounds consistent with the first defect. This is source evidence,
not proof of the exact private implementation shipped in the tested OS.

The library now uses Foundation's public ICU normalization transforms for fallbacks.
ASCII, stable BMP normalization quick checks, and table-based accent stripping still avoid
those calls. The supplementary fallback set now includes canonical decomposition components:
some composition operands, such as Kirat Rai vowel signs, are letters with combining class
zero. Treating only marks/decomposable scalars as nontrivial falsely certified those pairs.

Results against [Unicode 17's corpus](https://www.unicode.org/Public/17.0.0/ucd/NormalizationTest.txt)
and the assigned-scalar identity invariant from UnicodeData.txt:

| Implementation | NFC failures | NFD failures | NFKC failures | NFKD failures |
|---|---:|---:|---:|---:|
| Python tokenizers 0.23.2 | 940 | 967 | 1,453 | 1,480 |
| Foundation normalization properties | 76 | 6 | 115 | 6 |
| Corrected Swift tokenizers | 0 | 0 | 0 | 0 |

Each implementation received 400,680 listed corpus checks plus 1,120,992 checks on assigned
scalars absent from Part 1: **1,521,672 checks per implementation**. Python's failures here
do not invalidate its Unicode 9 behavior. The corrected Swift results establish conformance
for this corpus and this OS, not every possible sequence on every supported Apple OS.
CI additionally runs the pinned, checksum-verified Unicode 9 corpus on both macOS runners;
permanent regression tests cover the Hangul bounds, starter handling, compatibility ordering,
and supplementary composition quick check.

The debug suite passed with the Unicode 9 corpus enabled, and the release suite passed with
the complete Unicode 17 audit enabled. Optional benchmarks and the known-failing Python parity
audit are separately gated. Focused Address Sanitizer and Thread Sanitizer runs passed 20 and
21 tests respectively. The final library cross-compiled for the iOS 16 simulator with warnings
treated as errors. The updated GitHub workflow still needs to run after these changes are pushed.

For model interoperability, reproducing the model's reference tokenizer is a separate
requirement from implementing newer Unicode. This library retains its runtime Unicode policy;
it does not silently emulate the old normalization dependency for every tokenizer.json.

## Do all Python tokenizer tests pass in Swift?

No. The upstream Python suite is not a drop-in test suite for this Swift API, and Unicode
parity is not complete. Current evidence is:

- All 814 stored component golden cases across 72 configurations pass. Regenerating them
  with tokenizers 0.23.2 produced identical configurations and cases.
- Both existing differential suites pass their 24 model cases. Their bundled metadata pins
  the reference versions; these are not every tokenizer on the Hugging Face Hub.
- Running upstream Python normalizer, decoder, and pre-tokenizer tests in Python produced
  71 passes. This validates that reference installation; it is not 71 additional Swift passes.
- An exhaustive sweep of 1,112,064 valid scalars for each of seven normalizers, plus 2,004
  adversarial/seeded strings per normalizer, exposes the following remaining exact-byte gaps:

| Normalizer | Scalar mismatches | Sequence mismatches |
|---|---:|---:|
| Lowercase | 0 | 0 |
| NFC | 0 | 218 |
| NFD | 21 | 218 |
| NFKC | 171 | 218 |
| NFKD | 192 | 218 |
| StripAccents | 450 | 878 |
| BertNormalizer | 787 | 878 |

That audit intentionally fails with 4,249 mismatches. It is not part of the default passing
test run and has no allowlist that would conceal failures. Fixing standards conformance can
increase some counts against an older reference, while removing actual implementation bugs.
The four Unicode forms' results should be interpreted with the independent conformance table
above. StripAccents/Bert also depend on category, cleaning, and casing semantics; their
remaining differences are not independently certified merely by passing normalization tests.

At the v0.23.2 source tag, the 19 Python test files contain 223 test function definitions
(before parametrization). Tests involving alignment/offset-bearing Encoding objects,
training, Python serialization/mutation, complete padding/truncation/pair APIs, WordLevel,
and unsupported components such as Nmt, UnicodeScripts, FixedLength and CharDelimiterSplit
do not have equivalent supported Swift APIs. Passing the supported golden cases must not be
presented as passing that entire suite.

## Reproduce

Use Python 3.10 or newer with `tokenizers==0.23.2`. Download a matching pair of official UCD
files; the examples below assume they are in `/tmp/ucd/17.0.0/`.

```sh
python scripts/unicode_conformance.py /tmp/ucd/17.0.0/NormalizationTest.txt \
  --ucd /tmp/ucd/17.0.0/UnicodeData.txt
UNICODE_CONFORMANCE_FILE=/tmp/ucd/17.0.0/NormalizationTest.txt \
UNICODE_CONFORMANCE_UCD=/tmp/ucd/17.0.0/UnicodeData.txt \
  swift test -c release --filter UnicodeConformanceTests

python scripts/unicode_parity_reference.py --output /tmp/unicode-oracle.json
UNICODE_PARITY_FIXTURE=/tmp/unicode-oracle.json \
UNICODE_PARITY_REPORT=/tmp/unicode-parity.json \
  swift test -c release --filter UnicodeParityAuditTests
```

The Python Unicode 17 conformance command and the Swift parity command are expected to
exit nonzero with the documented gaps. The Swift conformance command's result depends on
the OS Unicode data. The scripts record exact mismatches or representative corpus failures;
all comparisons use scalars/bytes rather than canonical-equivalence String equality.

Corpus SHA-256:

- Unicode 9: `2d48d848656b3cf889df59980ab13551988950c4ca8c5190a17c691a17f8b9bb`
- Unicode 16: `d811971453e7075e1ad56fb1b301eece5aa80757b81f6156e74a1bfb3ae5ceb1`
- Unicode 17: `5019ffd530751a741900c849c0e010332f142a3612234639bd200b82138a87db`
