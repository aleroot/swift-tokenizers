#!/usr/bin/env python3
"""Generates ground-truth encodings with Hugging Face `transformers` (Rust `tokenizers` core)
for the differential corpus. Requires the tokenizer files to be present in the test fixture
cache (run the Swift test-suite once) and a venv with transformers + tokenizers installed.

    .reference/venv/bin/python scripts/hf_golden.py

Writes Tests/TokenizersTests/Resources/differential/hf__<repo>.json with records:
    {text, ids, idsNoSpecial, decoded, decodedSkipSpecial}
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from tokenizers import Tokenizer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class RawTokenizer:
    """Fallback for repositories `transformers` cannot load (e.g. `TokenizersBackend`):
    uses the Rust `tokenizers` core directly on tokenizer.json."""

    def __init__(self, folder):
        self.tok = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))

    def encode(self, text, add_special_tokens=True):
        return self.tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.environ.get(
    "SWIFT_TOKENIZERS_FIXTURES",
    os.path.expanduser("~/Library/Caches/swift-tokenizers-tests"),
)
CORPUS = os.path.join(ROOT, "Tests", "TokenizersTests", "Resources", "differential_corpus.json")
OUT = os.path.join(ROOT, "Tests", "TokenizersTests", "Resources", "differential")

MODELS = [
    "coreml-projects/Llama-2-7b-chat-coreml",
    "distilbert/distilbert-base-multilingual-cased",
    "distilgpt2",
    "openai/whisper-large-v2",
    "openai/whisper-tiny.en",
    "pcuenq/Llama-3.2-1B-Instruct-tokenizer",
    "t5-base",
    "tiiuae/falcon-7b",
    "pcuenq/gemma-tokenizer",
    "microsoft/phi-4",
    "mlx-community/Phi-3-mini-4k-instruct-4bit-no-q-embed",
    "google-t5/t5-small",
    "huggyllama/llama-7b",
    "intfloat/multilingual-e5-small",
    "FacebookAI/xlm-roberta-base",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "google-bert/bert-base-uncased",
    "BAAI/bge-small-en-v1.5",
    "FacebookAI/roberta-base",
    "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
    "Qwen/Qwen3-0.6B",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "microsoft/Phi-3-mini-128k-instruct",
]


def main():
    corpus = json.load(open(CORPUS, encoding="utf-8"))
    os.makedirs(OUT, exist_ok=True)
    for model in MODELS:
        folder = os.path.join(CACHE, model)
        if not os.path.exists(os.path.join(folder, "tokenizer.json")):
            print(f"skip {model}: not cached", file=sys.stderr)
            continue
        try:
            tok = AutoTokenizer.from_pretrained(folder, use_fast=True)
        except Exception as e:  # noqa: BLE001
            print(f"{model}: transformers failed ({e}); using raw tokenizers", file=sys.stderr)
            tok = RawTokenizer(folder)
        records = []
        for text in corpus:
            ids = tok.encode(text, add_special_tokens=True)
            records.append(
                {
                    "text": text,
                    "ids": ids,
                    "idsNoSpecial": tok.encode(text, add_special_tokens=False),
                    "decoded": tok.decode(ids, skip_special_tokens=False),
                    "decodedSkipSpecial": tok.decode(ids, skip_special_tokens=True),
                }
            )
        name = "hf__" + model.replace("/", "__") + ".json"
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, sort_keys=True)
        print(f"wrote {name} ({len(records)} records)")


if __name__ == "__main__":
    main()
