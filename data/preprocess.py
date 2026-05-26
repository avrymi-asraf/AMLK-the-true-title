"""
Pipeline step 2 of 3: instruction formatting and dataset splitting.
Reads outputs/data/raw/combined.jsonl, formats each example as a Hebrew
summarization instruction pair (adding a 'formatted' column), splits 80/10/10,
and saves Arrow dataset splits to outputs/data/processed/.
Does NOT tokenize — tokenization is handled by SFTTrainer at training time.

Run: python data/preprocess.py
Execution environment: local development machine.
"""
import json
import sys
from pathlib import Path

import datasets as hf_datasets


INPUT_PATH = Path("outputs/data/raw/combined.jsonl")
OUTPUT_DIR = Path("outputs/data/processed")


def format_instruction(text: str, summary: str) -> str:
    """Format a text+summary pair as a causal LM instruction for Hebrew summarization."""
    return (
        f"Summarize the following Hebrew text:\n\n"
        f"{text}\n\n"
        f"Summary:\n{summary}"
    )


def split_dataset(
    dataset: hf_datasets.Dataset,
    seed: int = 42,
) -> tuple[hf_datasets.Dataset, hf_datasets.Dataset, hf_datasets.Dataset]:
    """Split dataset 80% train / 10% val / 10% test. Returns (train, val, test)."""
    split = dataset.train_test_split(test_size=0.2, seed=seed)
    val_test = split["test"].train_test_split(test_size=0.5, seed=seed)
    return split["train"], val_test["train"], val_test["test"]


def main():
    if OUTPUT_DIR.exists():
        print(f"Output already exists at {OUTPUT_DIR}. Delete it to re-preprocess.")
        sys.exit(0)

    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run data/download.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {INPUT_PATH}...")
    records = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records")

    dataset = hf_datasets.Dataset.from_dict({
        "text": [r["text"] for r in records],
        "summary": [r["summary"] for r in records],
        "source": [r["source"] for r in records],
        "formatted": [format_instruction(r["text"], r["summary"]) for r in records],
    })

    print("Splitting 80/10/10...")
    train, val, test = split_dataset(dataset)
    print(f"  train: {len(train)}, val: {len(val)}, test: {len(test)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train.save_to_disk(str(OUTPUT_DIR / "train"))
    val.save_to_disk(str(OUTPUT_DIR / "val"))
    test.save_to_disk(str(OUTPUT_DIR / "test"))
    print(f"Saved splits to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
