"""Materialize the leakage-safe uncleaned E4 training pool.

The paper's E4 compares two SFT runs that differ only in their training corpus. This module builds
the uncleaned-arm input for the shared preprocessing stage.

Reads the curation working copy of raw HeSum, excludes curated val+test hesum_ids
(so raw train never sees evaluation articles), drops exact-duplicate texts, samples
5,854 rows (so split_dataset's unconditional 80/10/10 yields 4,683/585/586), and
writes an ignored working JSONL as {text, summary, source=hesum-raw,
hesum_id}. The raw test split is later discarded and replaced with curated test
by swapping in the curated test split (paired evaluation on the same articles).

Run: python -m pipeline.stage_04_training_experiment.prepare_uncleaned_data [--force]
Execution environment: local development machine, CPU only.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import datasets as hf_datasets

from pipeline.common.paths import CURATION_WORK_DIR, DATA_CURATION_ARTIFACTS_DIR, TRAINING_WORK_DIR
from pipeline.stage_04_training_experiment.prepare_curated_data import (
    load_curated_json,
    write_records_jsonl,
)
from pipeline.stage_04_training_experiment.prepare_data import split_dataset

RAW_INPUT = CURATION_WORK_DIR / "raw_hesum.json"
CURATED_INPUT = DATA_CURATION_ARTIFACTS_DIR / "final_clean_hesum.json"
OUTPUT_PATH = TRAINING_WORK_DIR / "uncleaned" / "uncleaned_records.jsonl"
SOURCE_LABEL = "hesum-raw"

# Must equal curated size so split_dataset (80/10/10) yields the same train/val
# counts as the curated arm (4,683 / 585 / 586). Handing fewer silently shrinks
# all three splits and breaks the matched-size premise of E4.
TARGET_SAMPLE_SIZE = 5854
SAMPLE_SEED = 42
SPLIT_SEED = 42


def normalize_raw_record(record: dict) -> dict | None:
    """Map one raw HeSum row {id, text, headline} → train-facing fields.

    Returns None when text or headline is empty after strip.
    """
    text = (record.get("text") or "").strip()
    headline = (record.get("headline") or "").strip()
    if not text or not headline:
        return None
    hesum_id = str(record.get("id") or record.get("hesum_id") or "").strip()
    return {
        "text": text,
        "summary": headline,
        "source": SOURCE_LABEL,
        "hesum_id": hesum_id,
    }


def load_raw_hesum(path: Path = RAW_INPUT) -> list[dict]:
    """Load raw_hesum.json and normalize every usable row."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Expected raw HeSum at {RAW_INPUT}."
        )
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a JSON list of records, got {type(raw).__name__}")

    records: list[dict] = []
    skipped = 0
    for row in raw:
        if not isinstance(row, dict):
            skipped += 1
            continue
        norm = normalize_raw_record(row)
        if norm is None:
            skipped += 1
            continue
        records.append(norm)
    if skipped:
        print(f"  Skipped {skipped} empty/invalid raw rows")
    return records


def resolve_curated_json() -> Path:
    """Return the canonical curated artifact used to define held-out ids."""
    if CURATED_INPUT.exists():
        return CURATED_INPUT
    raise FileNotFoundError(
        f"No curated HeSum found at {CURATED_INPUT}; it is needed to compute held-out ids."
    )


def curated_held_out_ids(
    curated_path: Path | None = None,
    seed: int = SPLIT_SEED,
) -> set[str]:
    """Reproduce the curated 80/10/10 split on ids only; return val ∪ test ids.

    `pipeline.stage_04_training_experiment.prepare_data.build_train_dataset` drops `hesum_id`, so
    held-out ids cannot be read from saved Arrow splits. `train_test_split` partitions by index from the seed,
    so a Dataset with the same row order and seed yields the same partition
    regardless of which columns it carries.
    """
    path = curated_path if curated_path is not None else resolve_curated_json()
    # Use the same normalization and row order as `prepare_curated_data.load_curated_json`.
    recs = load_curated_json(path)
    if not recs:
        raise ValueError(f"No usable curated records in {path}")
    ds = hf_datasets.Dataset.from_dict({"hesum_id": [r["hesum_id"] for r in recs]})
    _train, val, test = split_dataset(ds, seed=seed)
    return set(val["hesum_id"]) | set(test["hesum_id"])


def exclude_held_out(records: list[dict], held_out: set[str]) -> list[dict]:
    """Drop rows whose hesum_id is in the curated val/test set."""
    return [r for r in records if r["hesum_id"] not in held_out]


def dedupe_by_text(records: list[dict]) -> list[dict]:
    """Keep the first occurrence of each exact text; drop later duplicates.

    Raw HeSum was never deduplicated (14 exact dups in the E4 pool). Leaving them
    lets two copies of one article land in different splits and fails
    validate_train_dataset's overlap check after sampling.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        out.append(r)
    return out


def sample_records(
    records: list[dict],
    n: int = TARGET_SAMPLE_SIZE,
    seed: int = SAMPLE_SEED,
) -> list[dict]:
    """Sample n records without replacement (fixed seed, sorted by id then sample)."""
    if n > len(records):
        raise ValueError(
            f"Need {n} records but pool has only {len(records)} after exclusion/dedup"
        )
    rng = random.Random(seed)
    # Stable order before sample so the RNG state is independent of input list order.
    ordered = sorted(records, key=lambda r: r["hesum_id"])
    return rng.sample(ordered, n)


def build_raw_training_pool(
    raw_path: Path = RAW_INPUT,
    curated_path: Path | None = None,
    sample_size: int = TARGET_SAMPLE_SIZE,
    sample_seed: int = SAMPLE_SEED,
    split_seed: int = SPLIT_SEED,
) -> tuple[list[dict], dict]:
    """Full E4-RAW pool pipeline: load → exclude held-out → dedupe → sample.

    Returns the sampled records and in-memory run statistics.
    """
    raw = load_raw_hesum(raw_path)
    held_out = curated_held_out_ids(curated_path, seed=split_seed)
    pool = exclude_held_out(raw, held_out)
    n_after_exclude = len(pool)
    pool = dedupe_by_text(pool)
    n_after_dedup = len(pool)
    sample = sample_records(pool, n=sample_size, seed=sample_seed)

    sample_ids = {r["hesum_id"] for r in sample}
    if sample_ids & held_out:
        raise RuntimeError("Internal error: sample intersects curated held-out ids")

    stats = {
        "raw_input": str(raw_path),
        "raw_usable": len(raw),
        "held_out_n": len(held_out),
        "pool_after_exclude": n_after_exclude,
        "pool_after_dedup": n_after_dedup,
        "sample_size": len(sample),
        "sample_seed": sample_seed,
        "split_seed": split_seed,
        "target_sample_size": sample_size,
        "source": SOURCE_LABEL,
    }
    return sample, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe raw HeSum records for E4-RAW "
            "(exclude curated val/test, dedupe, sample 5854)"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_INPUT,
        help=f"Path to raw_hesum.json (default: {RAW_INPUT})",
    )
    parser.add_argument(
        "--curated",
        type=Path,
        default=None,
        help="Curated final_clean_hesum.json for held-out ids (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Normalized JSONL path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=TARGET_SAMPLE_SIZE,
        help=f"Rows to sample so 80/10/10 matches curated (default: {TARGET_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=SAMPLE_SEED,
        help=f"Sample RNG seed (default: {SAMPLE_SEED})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists",
    )
    args = parser.parse_args(argv)

    if args.output.exists() and not args.force:
        print(f"Output already exists at {args.output}. Pass --force to rebuild.")
        return 0

    print(f"Building E4-RAW pool from {args.input}...")
    try:
        records, stats = build_raw_training_pool(
            raw_path=args.input,
            curated_path=args.curated,
            sample_size=args.sample_size,
            sample_seed=args.sample_seed,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(
        f"  raw usable={stats['raw_usable']}  "
        f"held_out={stats['held_out_n']}  "
        f"pool_after_exclude={stats['pool_after_exclude']}  "
        f"after_dedup={stats['pool_after_dedup']}  "
        f"sample={stats['sample_size']} (seed={stats['sample_seed']})"
    )
    write_records_jsonl(records, args.output)
    print(f"Saved to {args.output}")

    print(
        "Next: python -m pipeline.stage_04_training_experiment.prepare_data --input "
        f"{args.output} --arm uncleaned --force "
        f"--test-from {TRAINING_WORK_DIR / 'processed' / 'curated'} "
        f"--output {TRAINING_WORK_DIR / 'processed' / 'uncleaned'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
