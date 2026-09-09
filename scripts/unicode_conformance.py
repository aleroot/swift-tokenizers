#!/usr/bin/env python3
"""Check Python tokenizers against the listed invariants in an official NormalizationTest.txt.

This deliberately uses the corpus as authority, not Python's unicodedata or Apple.
Supply a file from https://www.unicode.org/Public/<version>/ucd/NormalizationTest.txt.
Supply --ucd with that version's UnicodeData.txt to also check assigned unlisted scalars.
"""
import argparse
import hashlib
import json
from pathlib import Path

import tokenizers
from tokenizers import normalizers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--ucd", type=Path)
    args = parser.parse_args()
    assert tokenizers.__version__ == "0.23.2", tokenizers.__version__
    data = args.corpus.read_bytes()
    forms = {name: getattr(normalizers, name)() for name in ("NFC", "NFD", "NFKC", "NFKD")}
    failures = dict.fromkeys(forms, 0)
    examples = {name: [] for name in forms}
    checks = 0
    listed = set()
    part_one = False
    for line_number, line in enumerate(data.decode().splitlines(), 1):
        line = line.split("#")[0].strip()
        if line.startswith("@"):
            part_one = line.startswith("@Part1")
        if not line or line.startswith("@"):
            continue
        columns = ["".join(chr(int(value, 16)) for value in col.split()) for col in line.split(";")[:5]]
        if part_one:
            listed.update(map(ord, columns[0]))
        for name, normalizer in forms.items():
            for index, text in enumerate(columns):
                expected_index = {"NFC": 1 if index < 3 else 3, "NFD": 2 if index < 3 else 4, "NFKC": 3, "NFKD": 4}[name]
                expected = columns[expected_index]
                actual = normalizer.normalize_str(text)
                checks += 1
                if actual != expected:
                    failures[name] += 1
                    if len(examples[name]) < 5:
                        examples[name].append({"line": line_number, "input": text, "expected": expected, "actual": actual})
    corpus_checks = checks
    if args.ucd:
        start = None
        for line in args.ucd.read_text().splitlines():
            fields = line.split(";")
            value = int(fields[0], 16)
            if fields[1].endswith(", First>"):
                start = value
                continue
            lower = start if fields[1].endswith(", Last>") else value
            start = None
            for scalar in range(lower, value + 1):
                if scalar in listed or 0xD800 <= scalar <= 0xDFFF:
                    continue
                text = chr(scalar)
                for name, normalizer in forms.items():
                    checks += 1
                    if normalizer.normalize_str(text) != text:
                        failures[name] += 1
    print(json.dumps({"tokenizers": tokenizers.__version__, "corpus": data.decode().splitlines()[0],
                      "sha256": hashlib.sha256(data).hexdigest(), "corpusChecks": corpus_checks,
                      "unlistedScalarChecks": checks - corpus_checks,
                      "failures": failures, "examples": examples}, ensure_ascii=True, indent=2))
    raise SystemExit(int(any(failures.values())))


if __name__ == "__main__":
    main()
