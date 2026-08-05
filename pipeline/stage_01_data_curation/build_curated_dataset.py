"""Build the final curated HeSum dataset from completed curation artifacts."""

from __future__ import annotations

from collections import Counter

from pipeline.common.json_io import load_json, save_json
from pipeline.common.paths import CURATION_WORK_DIR, DATA_CURATION_ARTIFACTS_DIR


SOURCE_FILTER_INPUT_PATH = CURATION_WORK_DIR / "source_filter_input.json"
SOURCE_FILTER_RESULTS_PATH = DATA_CURATION_ARTIFACTS_DIR / "source_filter_results.json"
HEADLINE_TARGET_CURATION_RESULTS_PATH = DATA_CURATION_ARTIFACTS_DIR / "headline_target_curation_results.json"
FINAL_CLEAN_HESUM_PATH = DATA_CURATION_ARTIFACTS_DIR / "final_clean_hesum.json"


def load_source_records() -> list[dict]:
    """Load the pre-model-cleaned records that entered model curation."""
    return load_json(SOURCE_FILTER_INPUT_PATH)


def load_usable_ids() -> set[str]:
    """Load the ids that passed source filtering as usable records."""
    source_filter_results = load_json(SOURCE_FILTER_RESULTS_PATH)
    result_ids = [str(row["id"]) for row in source_filter_results]
    duplicate_ids = find_duplicate_ids(result_ids)
    if duplicate_ids:
        raise ValueError(f"Duplicate source-filter results for ids: {duplicate_ids[:5]}.")

    label_counts = Counter(str(row["filter_label"]) for row in source_filter_results)
    if label_counts.get("usable", 0) == 0:
        raise ValueError("No usable records found in source-filter results.")

    return {
        str(row["id"])
        for row in source_filter_results
        if row["filter_label"] == "usable"
    }


def load_replacement_headlines() -> dict[str, str | None]:
    """Load the headline replacements keyed by record id."""
    replacements = {}
    for row in load_json(HEADLINE_TARGET_CURATION_RESULTS_PATH):
        record_id = str(row["id"])
        if record_id in replacements:
            raise ValueError(f"Duplicate headline curation result for id {record_id}.")

        replacement_headline = row["replacement_headline"]
        if replacement_headline is not None and not isinstance(replacement_headline, str):
            raise ValueError(f"Invalid replacement headline for id {record_id}.")

        replacements[record_id] = replacement_headline

    return replacements


def find_duplicate_ids(record_ids: list[str]) -> list[str]:
    """Return duplicate ids while preserving their first repeated order."""
    seen = set()
    duplicates = []
    duplicate_seen = set()
    for record_id in record_ids:
        if record_id in seen and record_id not in duplicate_seen:
            duplicates.append(record_id)
            duplicate_seen.add(record_id)
        seen.add(record_id)

    return duplicates


def build_final_record(record: dict, replacement_headline: str | None) -> dict:
    """Build one final dataset record with the public output schema."""
    headline = replacement_headline if replacement_headline is not None else record["headline"]
    return {
        "hesum_id": str(record["id"]),
        "text": record["text"],
        "headline": headline,
    }


def validate_final_records(records: list[dict]) -> None:
    """Validate that every final record has the exact expected schema and value types."""
    expected_keys = {"hesum_id", "text", "headline"}
    duplicate_ids = find_duplicate_ids([str(record["hesum_id"]) for record in records])
    if duplicate_ids:
        raise ValueError(f"Duplicate final record ids: {duplicate_ids[:5]}.")

    for index, record in enumerate(records):
        if set(record) != expected_keys:
            raise ValueError(f"Invalid final record keys at index {index}: {sorted(record)}.")
        if not all(isinstance(record[key], str) for key in expected_keys):
            raise ValueError(f"Invalid final record value type at index {index}.")


def build_final_dataset() -> list[dict]:
    """Create final records for usable sources with curated replacement headlines applied."""
    source_records = load_source_records()
    usable_ids = load_usable_ids()
    replacement_headlines = load_replacement_headlines()
    missing_replacement_ids = usable_ids - set(replacement_headlines)
    extra_replacement_ids = set(replacement_headlines) - usable_ids
    if missing_replacement_ids:
        sample = sorted(missing_replacement_ids)[:5]
        raise ValueError(f"Missing headline curation results for usable ids: {sample}.")
    if extra_replacement_ids:
        sample = sorted(extra_replacement_ids)[:5]
        raise ValueError(f"Headline curation results contain non-usable ids: {sample}.")

    final_records = [
        build_final_record(record, replacement_headlines[str(record["id"])])
        for record in source_records
        if str(record["id"]) in usable_ids
    ]
    validate_final_records(final_records)
    return final_records


def print_final_stats(final_records: list[dict], replacement_headlines: dict[str, str | None]) -> None:
    """Print final dataset counts after the artifact is saved."""
    replaced = sum(1 for headline in replacement_headlines.values() if headline is not None)
    kept = len(replacement_headlines) - replaced

    print("Final dataset is complete.")
    print(f"Final records: {len(final_records)}")
    print(f"Headlines kept unchanged: {kept}")
    print(f"Headlines replaced: {replaced}")
    print(f"Saved final artifact to: {FINAL_CLEAN_HESUM_PATH}")


def main() -> None:
    """Build and save the final curated HeSum dataset artifact."""
    final_records = build_final_dataset()
    save_json(FINAL_CLEAN_HESUM_PATH, final_records)
    print_final_stats(final_records, load_replacement_headlines())


if __name__ == "__main__":
    main()
