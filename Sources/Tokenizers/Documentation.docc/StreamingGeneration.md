# Streaming generation

Show text as the model produces it, without breaking characters in half.

## Overview

Showing a model's output as it arrives is not simply decoding repeatedly. Here is why, and
what to use instead.

### The problem

The obvious loop is to decode the whole sequence after every step and print the difference.
It works, it is quadratic, and it still gets emoji wrong.

A single token is not a single character. A byte level tokenizer can emit one emoji as four
separate tokens, each carrying one UTF-8 byte. Decode token by token and you print four
replacement characters instead of a face. SentencePiece byte fallback has the same problem
across a whole run of `<0xNN>` tokens, and cleanup rules such as removing the space before a
comma need to see the next token before deciding.

### The answer

``TokenStreamDecoder`` holds back exactly the bytes that a later token could still change,
and emits everything else immediately.

```swift
var stream = tokenizer.makeStreamDecoder(skipSpecialTokens: true)

for id in model.generate() {
    let piece = try stream.append(token: id)
    if !piece.isEmpty { await display(piece) }
}
await display(stream.finish())
```

Concatenating every `append` result plus `finish()` gives exactly what
``Tokenizer/decode(tokens:skipSpecialTokens:)`` would have returned for the whole sequence.
That holds for malformed output too: invalid bytes become replacement characters in the same
places.

Always call ``TokenStreamDecoder/finish()``. It flushes the bytes that were being held, and a
response that ends mid character loses its last glyph without it.

### Sessions

A decoder is a struct with value semantics. Copying it copies its state, so one session per
response, and no sharing between concurrent generations.

```swift
stream.reset()   // start another response with the same settings
```

Appending after `finish()` throws, so an accidentally reused session fails loudly instead of
producing text that silently continues the previous answer.

### Pipelines that cannot stream

A few decoder pipelines transform the sequence as a whole and genuinely cannot emit anything
early. For those, ``TokenStreamDecoder/isIncremental`` is `false`: the session buffers the ids
and returns the complete text from `finish()`. The API does not change, only the timing.

```swift
if stream.isIncremental {
    // safe to render partial text as it arrives
}
```

### Special tokens

`skipSpecialTokens: true` drops control tokens such as `<|im_end|>` from the visible text. It
does not stop generation. Keep watching the ids for ``Tokenizer/eosTokenId`` and end the loop
yourself.
