"""Generate the final paper's curation-strata table from the frozen row ledger."""

from __future__ import annotations

import csv

from pipeline.common.json_io import load_json
from pipeline.common.paths import DATA_CURATION_ARTIFACTS_DIR, TABLES_DIR


ROW_LABELS_PATH = DATA_CURATION_ARTIFACTS_DIR / "row_labels.json"
OUTPUT_PATH = TABLES_DIR / "curation_strata.csv"


def build_rows(rows: list[dict]) -> list[dict[str, object]]:
    predicates = [
        ("S0", "usable source; original headline kept", lambda row: row["source_label"] == "usable" and row["headline_action"] == "kept"),
        ("S2", "headline contains at least two pipe separators", lambda row: row["multi_pipe"]),
        ("S3", "multiple independent items", lambda row: row["source_label"] == "unusable_multiple_independent_items"),
        ("S4", "headline rewritten during curation", lambda row: row["headline_action"] == "rewritten"),
        ("S5", "other unusable-source labels", lambda row: row["source_label"] not in {None, "usable", "unusable_multiple_independent_items"}),
    ]
    return [
        {"stratum": name, "criterion": criterion, "n": sum(predicate(row) for row in rows)}
        for name, criterion, predicate in predicates
    ]


def main() -> None:
    table_rows = build_rows(load_json(ROW_LABELS_PATH))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stratum", "criterion", "n"))
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Strata table saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
