#!/usr/bin/env python3
"""Refreshes the `decoded*` fields of the swift-transformers test fixtures using the current
Hugging Face `transformers` release, without touching token ids.

The original fixtures were produced when `clean_up_tokenization_spaces` defaulted to True;
transformers >= 4.45 defaults to False, which swift-tokenizers follows. Ids are cross-checked
against HF so the encode side of each fixture is verified at the same time.

    .reference/venv/bin/python scripts/refresh_fixture_decodes.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import AutoTokenizer  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "Tests", "TokenizersTests", "Resources")
CACHE = os.environ.get("SWIFT_TOKENIZERS_FIXTURES", os.path.expanduser("~/Library/Caches/swift-tokenizers-tests"))

DATASETS = {
    "coreml-projects/Llama-2-7b-chat-coreml": "llama_encoded.json",
    "distilbert/distilbert-base-multilingual-cased": "distilbert_cased_encoded.json",
    "distilgpt2": "gpt2_encoded_tokens.json",
    "openai/whisper-large-v2": "whisper_large_v2_encoded.json",
    "openai/whisper-tiny.en": "whisper_tiny_en_encoded.json",
    "pcuenq/Llama-3.2-1B-Instruct-tokenizer": "llama_3.2_encoded.json",
    "t5-base": "t5_base_encoded.json",
    "tiiuae/falcon-7b": "falcon_encoded.json",
}


def load(repo):
    return AutoTokenizer.from_pretrained(os.path.join(CACHE, repo), use_fast=True)


def main():
    changed = 0
    for repo, name in DATASETS.items():
        path = os.path.join(RES, name)
        data = json.load(open(path, encoding="utf-8"))
        tok = load(repo)
        ids = tok.encode(data["text"])
        if ids != data["token_ids"]:
            print(f"WARNING {name}: HF ids differ from fixture ids", file=sys.stderr)
        decoded = tok.decode(data["token_ids"])
        if decoded != data["decoded_text"]:
            data["decoded_text"] = decoded
            changed += 1
            json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"updated decoded_text in {name}")

    edge_path = os.path.join(RES, "tokenizer_tests.json")
    edge = json.load(open(edge_path, encoding="utf-8"))
    for repo, cases in edge.items():
        folder = os.path.join(CACHE, repo)
        if not os.path.exists(os.path.join(folder, "tokenizer.json")):
            print(f"skip {repo} (not cached)", file=sys.stderr)
            continue
        tok = load(repo)
        for case in cases:
            ids = case["encoded"]["input_ids"]
            hf_ids = tok.encode(case["input"])
            if hf_ids != ids:
                print(f"WARNING {repo}: HF ids differ for {case['input']!r}: {hf_ids} vs {ids}", file=sys.stderr)
            new_with = tok.decode(ids, skip_special_tokens=False)
            new_without = tok.decode(ids, skip_special_tokens=True)
            if new_with != case["decoded_with_special"] or new_without != case["decoded_without_special"]:
                case["decoded_with_special"] = new_with
                case["decoded_without_special"] = new_without
                changed += 1
    json.dump(edge, open(edge_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"done, {changed} records updated")


if __name__ == "__main__":
    main()
