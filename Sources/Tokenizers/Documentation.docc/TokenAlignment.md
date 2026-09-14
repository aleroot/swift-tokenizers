# Aligning tokens to text

Find out which part of the original string each token came from.

## Overview

Offsets connect each token back to the bytes it came from, which is what highlighting,
redaction and citation need.

### Ask for offsets

```swift
let encoding = try tokenizer.encode(text: sentence, withOffsets: true)
let bytes = Array(sentence.utf8)

for (id, range) in zip(encoding.ids, encoding.offsets ?? []) {
    guard let range else { continue }   // inserted special token, no source
    print(id, String(decoding: bytes[range], as: UTF8.self))
}
```

``TokenEncoding/offsets`` is `nil` when you did not ask for it. When you did, there is one
entry per token, and each entry is a range of **UTF-8 byte positions in the original text**,
before any normalization the pipeline applies. Lowercasing, accent stripping and NFKC all
happen after the offsets are recorded, so a range always points at text the user actually
typed.

Two entries behave specially:

- A `nil` range marks a token the post-processor inserted, such as BOS or `[CLS]`. A special
  token that really appears in the input does get a range.
- Several tokens can cover the same characters. One emoji encoded as four byte level tokens
  produces four ranges over the same scalar.

### Highlighting in a UI

`NSRange` works in UTF-16 units, so convert once for the whole encoding rather than per token:

```swift
guard let ranges = encoding.utf16Ranges(expandingToGraphemeClusters: true) else { return }

for (index, range) in ranges.enumerated() {
    guard let range else { continue }
    attributed.addAttribute(.backgroundColor, value: color(index), range: range)
}
```

`expandingToGraphemeClusters: true` grows each range to whole user perceived characters. It is
what you want for selection and highlighting, because half of a family emoji or a range that
ends before a combining accent renders as garbage. The cost is that neighbouring ranges can
then overlap, since they may share a cluster.

### What it costs

Offsets are tracked through pre-tokenization and post-processing, which is real work: expect a
few times the cost of encoding ids alone. Ask for them when you need alignment, not by habit.

### Custom tokenizers

Types that conform to ``Tokenizer`` themselves do not have to support offsets. The default
implementation returns ids with no offset table, and throws
``TokenizerError/unsupportedComponent(_:)`` if offsets are requested. To provide them, build a
``TokenEncoding`` with ``TokenEncoding/init(text:ids:offsets:)``, which validates that the
counts match and that every range lands on a scalar boundary.
