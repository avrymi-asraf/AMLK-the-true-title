"""
Pipeline step 2 of 3: instruction formatting, probe variants, and dataset splitting.
Reads outputs/data/raw/combined.jsonl and writes Arrow splits to
outputs/data/processed/<variant>/{train,val,test}. Always normalizes pipe/bullet
references into prose (data/clean.py), drops multi-headline roundups, and builds
(prompt, completion) pairs with the hardened summarization prompt so SFTTrainer can
train with completion_only_loss=True. The --variant flag (whole|lead|body) builds the
inputs for the truncation/positional-shortcut probe.

Run: python -m data.preprocess --variant whole
Execution environment: local development machine.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import datasets as hf_datasets

from data.clean import is_roundup_digest, normalize_summary
from data.prompts import build_prompt, make_variant
from training.config import MAX_LENGTH, MODEL_ID, processed_profile_name

INPUT_PATH = Path("outputs/data/raw/combined.jsonl")
OUTPUT_ROOT = Path("outputs/data/processed")
VARIANTS = ("whole", "lead", "body")
ARTICLE_TOKEN_BUDGET = MAX_LENGTH - 256


def truncate_to_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Cut text to its first max_tokens tokens (keeps the lead — where news summaries live)."""
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def split_dataset(
    dataset: hf_datasets.Dataset,
    seed: int = 42,
) -> tuple[hf_datasets.Dataset, hf_datasets.Dataset, hf_datasets.Dataset]:
    """Split dataset 80% train / 10% val / 10% test. Returns (train, val, test)."""
    split = dataset.train_test_split(test_size=0.2, seed=seed)
    val_test = split["test"].train_test_split(test_size=0.5, seed=seed)
    return split["train"], val_test["train"], val_test["test"]


def main():
    parser = argparse.ArgumentParser(
        description="Format and split Hebrew summarization data (clean references, hardened prompt)")
    parser.add_argument("--variant", choices=VARIANTS, default="whole",
                        help="Article input for the truncation probe (whole|lead|body)")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / processed_profile_name(args.variant)
    if output_dir.exists():
        print(f"Output already exists at {output_dir}. Delete it to re-preprocess.")
        sys.exit(0)

    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run data/download.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {INPUT_PATH}...")
    with open(INPUT_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"Loaded {len(records)} records")

    # Drop multi-headline media roundups, then rewrite remaining digests into prose.
    before = len(records)
    records = [r for r in records if not is_roundup_digest(r["summary"])]
    print(f"Dropped {before - len(records)} roundup digests (3+ pipes)")
    for r in records:
        r["summary"] = normalize_summary(r["summary"])
    print(f"Normalized {len(records)} references")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN") or None)
    print(f"Building variant '{args.variant}' and truncating articles to {ARTICLE_TOKEN_BUDGET} tokens...")
    texts = [truncate_to_tokens(make_variant(r["text"], args.variant), tokenizer, ARTICLE_TOKEN_BUDGET)
             for r in records]
    dataset = hf_datasets.Dataset.from_dict({
        "text": texts,
        "summary": [r["summary"] for r in records],
        "source": [r["source"] for r in records],
        "prompt": [build_prompt(t) for t in texts],
        "completion": [r["summary"] for r in records],
    })

    print(f"Splitting 80/10/10 (variant={args.variant})...")
    train, val, test = split_dataset(dataset)
    print(f"  train: {len(train)}, val: {len(val)}, test: {len(test)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train.save_to_disk(str(output_dir / "train"))
    val.save_to_disk(str(output_dir / "val"))
    test.save_to_disk(str(output_dir / "test"))
    print(f"Saved splits to {output_dir}")


if __name__ == "__main__":
    main()
