"""
Pipeline step 2 of 2: build the HuggingFace training dataset from HeSum records.

Reads curated (or E4-RAW) source JSON/JSONL, builds raw (prompt, completion) pairs
with the hardened summarization prompt, applies the positional probe --variant
(whole|lead|body), truncates each article to MAX_LENGTH-256 tokens so the summary
always survives, splits 80/10/10, and saves Arrow splits under
outputs/data/processed/<variant>/ (or --output).

For E4, --test-from <dir> substitutes another arm's test split (curated test on
the raw arm) and re-runs validate_train_dataset so train↔test text overlap is
caught after the swap. Chat-template wrap happens later at train/infer time.

Train contract (enforced by validate_train_dataset): each split has columns
text, summary, source, prompt, completion — what SFTTrainer + completion_only_loss
and the dual-arm prediction writers expect.

Run: python -m data.preprocess --variant whole [--force]
     python -m data.preprocess --input outputs/data/raw/raw_records.jsonl \\
         --test-from outputs/data/processed/e4cur --output outputs/data/processed/e4raw --force
Execution environment: local development machine (CPU; loads the base tokenizer only).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import datasets as hf_datasets

from data.download import (
    DEFAULT_INPUT as CURATED_JSON,
    OUTPUT_PATH as CURATED_JSONL,
    load_curated_json,
    normalize_curated_record,
)
from data.prompts import build_prompt, make_variant
from training.config import MAX_LENGTH, MODEL_ID, processed_profile_name

OUTPUT_ROOT = Path("outputs/data/processed")
VARIANTS = ("whole", "lead", "body")
ARTICLE_TOKEN_BUDGET = MAX_LENGTH - 256
TRAIN_COLUMNS = ("text", "summary", "source", "prompt", "completion")


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


def load_source_records(input_path: Path | None = None) -> list[dict]:
    """Load rows as {text, summary, source, hesum_id}.

    Preference order when input_path is None:
      1. outputs/data/curated/final_clean_hesum.json  (canonical curated product)
      2. outputs/data/curated/curated_records.jsonl   (normalized export from download)

    With --input, accepts curated JSON, curated JSONL, or E4-RAW JSONL
    (source=hesum-raw) from data.download_raw.
    """
    if input_path is not None:
        if input_path.suffix.lower() == ".jsonl":
            return _load_jsonl_records(input_path)
        return load_curated_json(input_path)

    if CURATED_JSON.exists():
        return load_curated_json(CURATED_JSON)
    if CURATED_JSONL.exists():
        return _load_jsonl_records(CURATED_JSONL)
    raise FileNotFoundError(
        f"No curated source found. Expected {CURATED_JSON} or {CURATED_JSONL}. "
        "Copy final_clean_hesum.json into outputs/data/curated/ "
        "(or run: python -m data.download)."
    )


def load_test_split(test_from: Path) -> hf_datasets.Dataset:
    """Load a saved test/ Arrow dir for E4 test-split swap."""
    test_dir = test_from / "test" if (test_from / "test").exists() else test_from
    if not test_dir.exists():
        raise FileNotFoundError(
            f"--test-from path not found: {test_from} (expected a processed dir "
            f"with a test/ subdir, or a test split directory itself)"
        )
    return hf_datasets.load_from_disk(str(test_dir))


def _load_jsonl_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Accept either already-normalized rows or raw curated shape.
            if "summary" in row and "text" in row:
                text = (row.get("text") or "").strip()
                summary = (row.get("summary") or "").strip()
                if not text or not summary:
                    continue
                records.append({
                    "text": text,
                    "summary": summary,
                    "source": row.get("source") or "hesum-curated",
                    "hesum_id": str(row.get("hesum_id") or "").strip(),
                })
            else:
                norm = normalize_curated_record(row)
                if norm is not None:
                    records.append(norm)
    return records


def build_train_dataset(
    records: list[dict],
    variant: str,
    tokenizer,
    article_token_budget: int = ARTICLE_TOKEN_BUDGET,
) -> hf_datasets.Dataset:
    """Build the train-format HF Dataset from normalized curated records."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if not records:
        raise ValueError("No records to build a training dataset from")

    texts = [
        truncate_to_tokens(make_variant(r["text"], variant), tokenizer, article_token_budget)
        for r in records
    ]
    summaries = [r["summary"] for r in records]
    return hf_datasets.Dataset.from_dict({
        "text": texts,
        "summary": summaries,
        "source": [r["source"] for r in records],
        "prompt": [build_prompt(t) for t in texts],
        "completion": list(summaries),
    })


def validate_train_dataset(
    train: hf_datasets.Dataset,
    val: hf_datasets.Dataset,
    test: hf_datasets.Dataset,
) -> None:
    """Raise ValueError if splits are not suitable for training/inference.

    Contract (must stay aligned with training/train.py + train_hf_job.py):
      - columns text, summary, source, prompt, completion (all non-empty strings)
      - completion == summary per row
      - prompt carries the article text (task instruction + body)
      - non-empty splits, no text overlap across train/val/test
    """
    for name, ds in (("train", train), ("val", val), ("test", test)):
        if len(ds) == 0:
            raise ValueError(f"{name} split is empty")
        missing = [c for c in TRAIN_COLUMNS if c not in ds.column_names]
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")

        # Spot-check every row for emptiness (dataset is ~6k — fine locally).
        for col in TRAIN_COLUMNS:
            values = ds[col]
            bad = [i for i, v in enumerate(values) if not isinstance(v, str) or not v.strip()]
            if bad:
                raise ValueError(
                    f"{name}.{col}: {len(bad)} empty/non-string rows "
                    f"(first index {bad[0]})"
                )

        for i, (summary, completion) in enumerate(zip(ds["summary"], ds["completion"])):
            if summary != completion:
                raise ValueError(
                    f"{name}[{i}]: completion must equal summary "
                    f"(got {completion[:40]!r} vs {summary[:40]!r})"
                )

        for i, (text, prompt) in enumerate(zip(ds["text"], ds["prompt"])):
            # Prompt must embed the article; truncation can shorten text but the
            # full (possibly truncated) article body must still appear.
            if text not in prompt:
                raise ValueError(
                    f"{name}[{i}]: prompt does not contain the article text"
                )

    train_set, val_set, test_set = set(train["text"]), set(val["text"]), set(test["text"])
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("train/val/test text sets overlap — split is leaky")


def save_splits(
    train: hf_datasets.Dataset,
    val: hf_datasets.Dataset,
    test: hf_datasets.Dataset,
    output_dir: Path,
) -> None:
    """Write train/val/test Arrow dirs under output_dir (overwrites if present)."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.save_to_disk(str(output_dir / "train"))
    val.save_to_disk(str(output_dir / "val"))
    test.save_to_disk(str(output_dir / "test"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build HuggingFace train/val/test splits from HeSum records "
            "(curated default; E4-RAW via --input + --test-from)"
        ),
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default="whole",
        help="Article input for the truncation probe (whole|lead|body)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Curated JSON/JSONL or E4-RAW JSONL (default: auto-detect curated)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output dir for Arrow splits "
            f"(default: {OUTPUT_ROOT}/<variant>). Use distinct dirs for E4 arms."
        ),
    )
    parser.add_argument(
        "--test-from",
        type=Path,
        default=None,
        help=(
            "Replace the built test split with test/ from this processed dir "
            "(E4: share curated test across arms). Re-validates after swap."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output dir",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Split seed (default 42)",
    )
    args = parser.parse_args(argv)

    output_dir = args.output or (OUTPUT_ROOT / processed_profile_name(args.variant))
    if output_dir.exists() and not args.force:
        print(f"Output already exists at {output_dir}. Pass --force to rebuild.")
        return 0

    try:
        records = load_source_records(args.input)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Loaded {len(records)} records")

    from transformers import AutoTokenizer

    print(f"Loading tokenizer {MODEL_ID} for article truncation...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, token=os.environ.get("HF_TOKEN") or None,
    )
    print(
        f"Building variant '{args.variant}' and truncating articles to "
        f"{ARTICLE_TOKEN_BUDGET} tokens..."
    )
    dataset = build_train_dataset(records, args.variant, tokenizer)

    print(f"Splitting 80/10/10 (variant={args.variant}, seed={args.seed})...")
    train, val, test = split_dataset(dataset, seed=args.seed)
    print(f"  train: {len(train)}, val: {len(val)}, test: {len(test)}")

    print("Validating train contract...")
    validate_train_dataset(train, val, test)
    print("  OK — columns, non-empty fields, completion==summary, no split leak")

    if args.test_from is not None:
        print(f"Swapping test split from {args.test_from}...")
        try:
            test = load_test_split(args.test_from)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        print(f"  test after swap: {len(test)}")
        # Mandatory: the first validate ran before the swap, so it cannot catch
        # raw-train ↔ curated-test overlap on the arm that needs that check.
        print("Re-validating train contract after test-split swap...")
        validate_train_dataset(train, val, test)
        print("  OK — post-swap: no train/val/test text overlap")

    save_splits(train, val, test, output_dir)
    print(f"Saved HuggingFace Arrow splits to {output_dir}")
    print(
        "Next: upload this dir to a Hub dataset repo, then "
        "python -m training.train --submit-hf --hf-user <you> "
        "--dataset-repo <repo> --skip-data-upload"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
