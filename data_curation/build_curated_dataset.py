"""Rebuild the curated HeSum dataset from download plus supplied model results.

Run:
    python -m data_curation.build_curated_dataset

Required supplied files:
    data_curation/artifacts/source_filter_results.json
    data_curation/artifacts/headline_target_curation_results.json

Output:
    data_curation/artifacts/final_clean_hesum.json
"""

from __future__ import annotations

from collections import Counter
import hashlib

from data_curation.data_download.download_hesum import RAW_HESUM_PATH, download_records
from data_curation.final_dataset import build_final_dataset
from data_curation.model_curation.headline_target_curation.refine_records import (
    FINAL_RESULTS_PATH as HEADLINE_TARGET_CURATION_RESULTS_PATH,
)
from data_curation.model_curation.source_filter.filter_records import (
    FINAL_RESULTS_PATH as SOURCE_FILTER_RESULTS_PATH,
)
from data_curation.pre_model_cleanup import run_pre_model_cleanup
from data_curation.utils.json_io import load_json, save_json


REQUIRED_MODEL_RESULT_PATHS = [
    SOURCE_FILTER_RESULTS_PATH,
    HEADLINE_TARGET_CURATION_RESULTS_PATH,
]
EXPECTED_SOURCE_FILTER_INPUT_SHA256 = (
    "abb3184938e7fdecdd6cd0d5f402f227276c1f829a0feaf6d5c5b36cb3ce65c6"
)


def ensure_supplied_model_results_exist() -> None:
    """Raise an error if a required supplied model-result artifact is missing."""
    missing_paths = [path for path in REQUIRED_MODEL_RESULT_PATHS if not path.exists()]
    if missing_paths:
        formatted_paths = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Missing supplied model-result artifact files:\n"
            f"{formatted_paths}"
        )


def download_raw_hesum() -> None:
    """Download and save raw HeSum records when the raw artifact is absent."""
    if RAW_HESUM_PATH.exists():
        print(f"Raw HeSum dataset already exists: {RAW_HESUM_PATH}")
        return

    records = download_records()
    save_json(RAW_HESUM_PATH, records)
    print(f"Downloaded records: {len(records)}")
    print(f"Saved to: {RAW_HESUM_PATH}")


def validate_source_filter_results() -> None:
    """Validate that source-filter results cover exactly the rebuilt source input."""
    source_input_hash = hashlib.sha256(
        run_pre_model_cleanup.SOURCE_FILTER_INPUT_PATH.read_bytes()
    ).hexdigest()
    if source_input_hash != EXPECTED_SOURCE_FILTER_INPUT_SHA256:
        raise ValueError(
            "Rebuilt source-filter input does not match the model-curation input. "
            f"Expected sha256 {EXPECTED_SOURCE_FILTER_INPUT_SHA256}, got {source_input_hash}."
        )

    source_records = load_json(run_pre_model_cleanup.SOURCE_FILTER_INPUT_PATH)
    source_filter_results = load_json(SOURCE_FILTER_RESULTS_PATH)
    source_ids = {str(record["id"]) for record in source_records}
    result_ids = [str(row["id"]) for row in source_filter_results]
    duplicate_ids = build_final_dataset.find_duplicate_ids(result_ids)
    if duplicate_ids:
        raise ValueError(f"Duplicate source-filter result ids: {duplicate_ids[:5]}.")

    if set(result_ids) != source_ids:
        missing_ids = sorted(source_ids - set(result_ids))[:5]
        extra_ids = sorted(set(result_ids) - source_ids)[:5]
        raise ValueError(
            "Source-filter results do not match the rebuilt source input. "
            f"Missing ids: {missing_ids}; extra ids: {extra_ids}."
        )


def validate_headline_curation_results() -> None:
    """Validate that headline-curation results cover exactly the usable records."""
    source_filter_results = load_json(SOURCE_FILTER_RESULTS_PATH)
    headline_curation_results = load_json(HEADLINE_TARGET_CURATION_RESULTS_PATH)
    label_counts = Counter(row["filter_label"] for row in source_filter_results)
    usable_ids = {
        str(row["id"])
        for row in source_filter_results
        if row["filter_label"] == "usable"
    }
    result_ids = [str(row["id"]) for row in headline_curation_results]
    duplicate_ids = build_final_dataset.find_duplicate_ids(result_ids)
    if duplicate_ids:
        raise ValueError(f"Duplicate headline-curation result ids: {duplicate_ids[:5]}.")

    if set(result_ids) != usable_ids:
        missing_ids = sorted(usable_ids - set(result_ids))[:5]
        extra_ids = sorted(set(result_ids) - usable_ids)[:5]
        raise ValueError(
            "Headline-curation results do not match usable source records. "
            f"Missing ids: {missing_ids}; extra ids: {extra_ids}."
        )

    print(f"Usable source records: {label_counts['usable']}")


def main() -> None:
    """Rebuild deterministic artifacts and the final curated dataset."""
    ensure_supplied_model_results_exist()

    print("Step 1/3: downloading raw HeSum data")
    download_raw_hesum()

    print("Step 2/3: rebuilding deterministic pre-model cleanup artifacts")
    run_pre_model_cleanup.main()

    print("Step 3/3: validating supplied model results and building final dataset")
    validate_source_filter_results()
    validate_headline_curation_results()
    build_final_dataset.main()


if __name__ == "__main__":
    main()
