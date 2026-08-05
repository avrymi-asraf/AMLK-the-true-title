"""Summarize the two frozen model-curation outputs without paper-number constants."""

from __future__ import annotations

from collections import Counter

from pipeline.common.json_io import load_json, save_json
from pipeline.common.paths import DATA_CURATION_ARTIFACTS_DIR, SUMMARIES_DIR


SOURCE_FILTER_PATH = DATA_CURATION_ARTIFACTS_DIR / "source_filter_results.json"
HEADLINE_CURATION_PATH = DATA_CURATION_ARTIFACTS_DIR / "headline_target_curation_results.json"
OUTPUT_PATH = SUMMARIES_DIR / "data_curation.json"


def build_summary(source_rows: list[dict], headline_rows: list[dict]) -> dict:
    source_ids = [str(row["id"]) for row in source_rows]
    headline_ids = [str(row["id"]) for row in headline_rows]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source-filter artifact contains duplicate ids")
    if len(set(headline_ids)) != len(headline_ids):
        raise ValueError("headline-curation artifact contains duplicate ids")
    labels = Counter(row["filter_label"] for row in source_rows)
    usable_ids = {str(row["id"]) for row in source_rows if row["filter_label"] == "usable"}
    if set(headline_ids) != usable_ids:
        raise ValueError("headline-curation ids do not exactly cover usable source-filter ids")
    rewritten = sum(row["replacement_headline"] is not None for row in headline_rows)
    return {
        "model_curation_input": len(source_rows),
        "source_filter_labels": dict(sorted(labels.items())),
        "usable": len(usable_ids),
        "unusable": len(source_rows) - len(usable_ids),
        "headline_kept": len(headline_rows) - rewritten,
        "headline_rewritten": rewritten,
    }


def main() -> None:
    summary = build_summary(load_json(SOURCE_FILTER_PATH), load_json(HEADLINE_CURATION_PATH))
    save_json(OUTPUT_PATH, summary)
    print(f"Data-curation summary saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
