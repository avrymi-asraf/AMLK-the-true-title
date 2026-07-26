"""Download and normalize the raw HeSum dataset.

This is stage 1 of the data-curation pipeline (see `CURATION_ROADMAP.md`): the entry point that
turns the external `biunlp/HeSum` Hub dataset into the stable, locally-owned `raw_hesum.json`
every later stage (tail trimming, the two deterministic filters, model curation) builds on.
Runs locally, CPU-only, network-bound.

Run:
    python -m data_curation.data_download.download_hesum

Output:
    data_curation/artifacts/raw_hesum.json
"""

from __future__ import annotations

import argparse

from data_curation.utils.json_io import save_json
from data_curation.utils.lzma_shim import ensure_lzma_importable
from data_curation.utils.paths import ARTIFACTS_DIR

DATASET_NAME = "biunlp/HeSum"
RAW_HESUM_PATH = ARTIFACTS_DIR / "raw_hesum.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for downloading the raw HeSum dataset."""
    parser = argparse.ArgumentParser(description="Download the raw HeSum dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the existing dataset if it already exists.",
    )
    return parser.parse_args()


def download_records() -> list[dict]:
    """Download HeSum splits and normalize them into records with string ids."""
    ensure_lzma_importable()
    import datasets

    print(f"Loading {DATASET_NAME} from HuggingFace Hub...")
    # token=False: HeSum is public, and some local setups carry an invalid cached HF token that
    # would otherwise turn every anonymous-eligible request into a 401.
    dataset = datasets.load_dataset(DATASET_NAME, token=False)
    merged_dataset = datasets.concatenate_datasets(dataset.values())

    records = []
    for idx, entry in enumerate(merged_dataset, start=1):
        record = {
            "id": str(idx),
            "text": entry["article"].strip(),
            "headline": entry["summary"].strip(),
        }
        records.append(record)

    return records


def main() -> None:
    """Download raw HeSum records unless the artifact already exists."""
    args = parse_args()

    if RAW_HESUM_PATH.exists() and not args.force:
        print(f"Raw HeSum dataset already exists: {RAW_HESUM_PATH}")
        print("Use --force to redownload and overwrite it.")
        return

    records = download_records()
    save_json(RAW_HESUM_PATH, records)

    print(f"Downloaded records: {len(records)}")
    print(f"Saved to: {RAW_HESUM_PATH}")


if __name__ == "__main__":
    main()
