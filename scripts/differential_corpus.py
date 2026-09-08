#!/usr/bin/env python3
"""Deterministic corpus of tricky inputs used for differential testing against
swift-transformers. Writes Tests/TokenizersTests/Resources/differential_corpus.json."""
import json
import os
import random

random.seed(20240607)

base = [
    "",
    " ",
    "  ",
    "\n",
    "\n\n",
    " \n",
    "\n ",
    "\t",
    "Hello world",
    "Hello, world!",
    " Hello world",
    "Hello world ",
    "  leading and trailing  ",
    "Today she took a train to the West",
    "Who are you?",
    "I'm fine, you're great, we've won, they'll see, he'd know, it's ok, isn't it?",
    "I'M FINE, YOU'RE GREAT, WE'VE WON, THEY'LL SEE, HE'D KNOW, IT'S OK",
    "'s 't 're 've 'm 'll 'd 'S 'T 'RE 'VE 'M 'LL 'D 'x 'sure 'twas",
    "don't can't won't shouldn't",
    "123 4567 89012345 0 007",
    "The year 2024 had 366 days; 1,000,000 people; 3.14159; -42; 1e10",
    "café résumé naïve façade coöperate",
    "mąka département l'eure",
    "à à",  # composed vs decomposed
    "Ünïcödé Ⅻ ½ ² ∑ ∞ ≠ ← → ✓",
    "こんにちは、世界。トークナイザーのテストです。",
    "ザ で ば ぱ",
    "안녕하세요 세계",
    "你好，世界！这是一个分词器测试。",
    "Привет, мир! Это тест токенизатора.",
    "مرحبا بالعالم! هذا اختبار للمقطع.",
    "שלום עולם",
    "สวัสดีชาวโลก สวัส",
    "नमस्ते दुनिया",
    "🙂 😀😃😄 👨‍👩‍👧‍👦 🇮🇹 🏳️‍🌈 1️⃣",
    "emoji😀inside😀words",
    "tab\tseparated\tvalues",
    "line1\nline2\r\nline3\rline4",
    "multiple\n\n\nnewlines\n\n",
    "trailing newline\n",
    "\nleading newline",
    "spaces   between    words",
    "mixed \t whitespace \n\t mix",
    "non\u00a0breaking\u00a0space",
    "zero\u200bwidth\u200bspace",
    "ideographic\u3000space",
    "!!!???...,,,;;;:::",
    "(parentheses) [brackets] {braces} <angles>",
    "quotes \"double\" 'single' `back` “curly” ‘curly’",
    "hyphen-ated words and — em dashes – en dashes",
    "email@example.com https://example.com/path?query=1&b=2#frag",
    "C:\\Windows\\System32\\drivers\\etc\\hosts",
    "/usr/local/bin/python3 -m pip install --upgrade pip",
    "def f(x):\n    return x * 2\n\nprint(f(21))\n",
    "public final class GPT2BytePairEncoderConfiguration: Codable, Sendable {\n    public let vocabularyIdentifierToTokenStringMap: [Int: String]\n}",
    "SELECT * FROM users WHERE id = 42 AND name LIKE '%john%';",
    "{\"key\": [1, 2, 3], \"nested\": {\"a\": null, \"b\": true}}",
    "<div class=\"x\">&amp; &lt;tag&gt;</div>",
    "camelCaseIdentifier snake_case_identifier SCREAMING_SNAKE kebab-case",
    "supercalifragilisticexpialidocious internationalization",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "ab" * 70,
    "x" * 300,
    "1" * 40,
    "." * 50,
    " " * 30,
    "\n" * 12,
    "<s>",
    "</s>",
    "<unk>",
    "<pad>",
    "<|endoftext|>",
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>",
    "<bos>text<eos>",
    "<s>Who are you?</s>",
    "[CLS] sentence [SEP]",
    "[MASK] the [MASK]",
    "<extra_id_0> fill <extra_id_1>",
    "<0x41><0x42>",
    "<|user|>\nHi<|end|>\n<|assistant|>",
    " <|endoftext|> ",
    "text <|endoftext|>text",
    "<|endoftext|><|endoftext|>",
    "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    "▁already ▁metaspaced",
    "Ġbyte Ġlevel",
    "\ufeff# BOM at start",
    "\u0000 null byte",
    "control \u0001\u0002\u001f chars",
    "\u007f delete",
    "combining: e\u0301 a\u0300 o\u0308",
    "ligatures: ﬁ ﬂ ﬃ",
    "fullwidth: Ｈｅｌｌｏ　Ｗｏｒｌｄ ～",
    "math: 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝕳𝖊𝖑𝖑𝖔",
    "surrogates 😀 mixed with ascii 123 and 中文",
    "Ｔｅｓｔ １２３",
    "٣٤٥ Arabic digits ३४५ Devanagari digits",
    "Ⅳ Ⅸ ⅻ roman numerals",
    "½ ¼ ¾ fractions",
    "long text. " * 40,
]

words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "hello", "world",
         "Swift", "tokenizer", "BPE", "unicode", "Zürich", "naïve", "東京", "München", "123", "4.5",
         "don't", "it's", "we'll", "—", "…", "!", "?", ",", ".", ";", "(", ")", "\"", "'", "\n", "\t", "  ",
         "😀", "🚀", "é", "à", "ß", "ñ", "ø", "æ", "<s>", "</s>", "<|endoftext|>", "<0x0A>", "▁", "Ġ",
         "supercalifragilistic", "a", "I", "x2", "3D", "C++", "C#", "#include", "@user", "$100", "€50", "50%"]

random_texts = []
for _ in range(120):
    n = random.randint(1, 25)
    parts = []
    for _ in range(n):
        w = random.choice(words)
        sep = random.choice([" ", " ", " ", "", "\n", "  ", "\t"])
        parts.append(w + sep)
    random_texts.append("".join(parts))

# Random unicode scalar soup (avoid surrogates and U+000B / U+0085 whose \s classification
# differs between ICU and Unicode White_Space; avoid unassigned ranges that differ by Unicode version).
def rand_scalar():
    while True:
        r = random.random()
        if r < 0.5:
            c = random.randint(0x20, 0x7e)
        elif r < 0.7:
            c = random.randint(0xa0, 0x2ff)
        elif r < 0.8:
            c = random.randint(0x370, 0x52f)
        elif r < 0.9:
            c = random.randint(0x3040, 0x30ff)
        else:
            c = random.choice([0x1f600, 0x1f680, 0x4e2d, 0x6587, 0x2028, 0x2029, 0x3000, 0x2000, 0x200a, 0x1680, 0xa, 0xd, 0x9, 0x20, 0xc])
        if c in (0xb, 0x85, 0xad):
            continue
        return chr(c)

for _ in range(80):
    random_texts.append("".join(rand_scalar() for _ in range(random.randint(1, 60))))

corpus = base + random_texts
out = os.path.join(os.path.dirname(__file__), "..", "Tests", "TokenizersTests", "Resources", "differential_corpus.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=0)
print(f"wrote {len(corpus)} texts to {out}")
