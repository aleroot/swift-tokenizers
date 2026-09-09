#!/usr/bin/env python3
"""Produce an exhaustive scalar oracle from the Rust-backed Python package.

Run with tokenizers==0.23.2. Missing scalar mappings mean identity; sequence cases
exercise interactions that a scalar sweep alone cannot establish.
"""
import argparse
import json
import random
from pathlib import Path

import tokenizers
from tokenizers import normalizers as n

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert tokenizers.__version__ == "0.23.2", tokenizers.__version__
normalizers = {
    "NFC": n.NFC(), "NFD": n.NFD(), "NFKC": n.NFKC(), "NFKD": n.NFKD(),
    "Lowercase": n.Lowercase(), "StripAccents": n.StripAccents(),
    "BertNormalizer": n.BertNormalizer(),
}
rng = random.Random(20260909)
atoms = ["a", "É", "Σ", "İ", "\u0301", "\u0323", "\u0345", "\u0315", "\u034f", "\u093c",
         "\u1100", "\u1161", "\u1176", "\u11a8", "\u1ad0", "\ua7ce", "\u00a0", "\u200d", "\U0001e08f"]
texts = ["\u1100\u1176", "\u1100\u1161\u11a8", "A\u0301\u0323", "é\u0315\u0300"]
texts += ["".join(rng.choices(atoms, k=rng.randrange(1, 12))) for _ in range(2000)]
result = {"reference": tokenizers.__version__, "scalarCount": 0x110000 - 0x800, "normalizers": {}}
for name, normalizer in normalizers.items():
    mappings = {}
    for value in range(0x110000):
        if 0xD800 <= value < 0xE000:
            continue
        text = chr(value)
        normalized = normalizer.normalize_str(text)
        if normalized != text:
            mappings[str(value)] = normalized
    result["normalizers"][name] = {
        "configuration": json.loads(normalizer.__getstate__()), "mappings": mappings,
        "sequences": [{"text": text, "expected": normalizer.normalize_str(text)} for text in texts],
    }
    print(name, len(mappings), "non-identity mappings", flush=True)
args.output.write_text(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
