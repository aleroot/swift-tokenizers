# Normalizing transcripts

Put speech recognition output and reference transcripts in the canonical form word error
rates are computed on.

## Overview

Whisper's evaluations do not compare raw text. Both the hypothesis and the reference first go
through a text normalizer, so that "Mr. Smith paid $25." and "mister smith paid twenty five
dollars" count as the same words. The English speech recognition benchmarks, the Open ASR
Leaderboard among them, score with the same normalizers.

``EnglishTextNormalizer`` is the English one. It lowercases, removes bracketed annotations
such as `[music]`, drops hesitations, expands contractions and titles, writes numbers and
currencies in digits and applies British to American spellings. The spellings come from the
`normalizer.json` that Whisper model repositories ship:

```swift
let normalize = try EnglishTextNormalizer(contentsOf: modelFolder.appending(path: "normalizer.json"))

normalize("Mr. Smith won't pay twenty five dollars.")  // "mister smith will not pay $25"
normalize("The colour of the centre")                  // "the color of the center"
```

``BasicTextNormalizer`` is the language-independent one, for every other language. It keeps
letters of any script and turns marks, symbols and punctuation into spaces. It can also drop
diacritics, and separate every character for character error rates:

```swift
BasicTextNormalizer()("¿Qué tal? [risas]")                        // " qué tal "
BasicTextNormalizer(removeDiacritics: true)("¿Qué tal? [risas]")  // " que tal "
BasicTextNormalizer(splitLetters: true)("你好")                    // "你 好"
```

Both return exactly what `transformers`' `EnglishTextNormalizer` and `BasicTextNormalizer`
return, which `WhisperTokenizer.normalize` and `basic_normalize` apply. Scores computed on
either side are comparable. Characters assigned after Unicode 15, the version Python 3.12
reads, follow the newer Unicode data of the operating system.

Normalizers are values: create one and call it from any task.
