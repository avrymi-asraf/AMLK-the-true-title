"""Join frozen E1 rubric scores to row labels and compute paper statistics.

Run:
    python -m pipeline.stage_03_reference_experiments.e1_flagged_strata.summarize

Output:
    results/summaries/e1.json
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.common.json_io import load_json, save_json
from pipeline.common.paths import (
    DATA_CURATION_ARTIFACTS_DIR,
    REFERENCE_EXPERIMENT_ARTIFACTS_DIR,
    SUMMARIES_DIR,
)
from pipeline.common.statistics import bootstrap_cliffs_delta_ci, cliffs_delta, median

E1_SCORES_PATH = REFERENCE_EXPERIMENT_ARTIFACTS_DIR / "e1_rubric_scores.jsonl"
ROW_LABELS_PATH = DATA_CURATION_ARTIFACTS_DIR / "row_labels.json"
E1_SUMMARY_PATH = SUMMARIES_DIR / "e1.json"

DIMENSIONS = ["faithfulness", "single_focus", "informativeness", "cleanliness"]

# Stratum membership predicates over a joined (e1 score row, row_labels row) pair. S0 is the
# reference group for every Cliff's delta; S1 is skipped deliberately (Table 1, main.tex).
STRATUM_PREDICATES = {
    "S0": lambda r: r["reached_model_curation"] and r["source_label"] == "usable"
    and r["headline_action"] == "kept",
    "S2": lambda r: r["multi_pipe"],
    "S3": lambda r: r["source_label"] == "unusable_multiple_independent_items",
    "S4": lambda r: r["headline_action"] == "rewritten",
}


def load_jsonl(path: Path) -> list[dict]:
    """Load a UTF-8 JSON-lines file, one dict per non-blank line."""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_e1_scored_rows() -> list[dict]:
    """Load E1 judge output, dropping the ~30 rows where the judge call failed (empty `scores`)."""
    return [row for row in load_jsonl(E1_SCORES_PATH) if row["scores"]]


def join_e1_to_strata(e1_rows: list[dict], row_labels_by_id: dict[str, dict]) -> dict[str, dict[str, list[int]]]:
    """Group original-headline rubric scores by stratum and dimension.

    A row can satisfy more than one stratum predicate (the strata overlap by design — see
    `main.tex` Section: Defect taxonomy and analysis strata), so membership checks are
    independent, not `elif` branches.
    """
    groups: dict[str, dict[str, list[int]]] = {
        stratum: {dim: [] for dim in DIMENSIONS} for stratum in STRATUM_PREDICATES
    }
    for e1_row in e1_rows:
        row_label = row_labels_by_id.get(e1_row["hesum_id"])
        if row_label is None:
            continue
        for stratum, predicate in STRATUM_PREDICATES.items():
            if predicate(row_label):
                for dim in DIMENSIONS:
                    groups[stratum][dim].append(e1_row["scores"][dim]["score"])
    return groups


def build_e1_summary(groups: dict[str, dict[str, list[int]]]) -> dict:
    """Per-stratum n, medians, and Cliff's delta (+ bootstrap CI) vs.\\ S0 clean for every dimension."""
    summary: dict = {"strata": {}}
    for stratum, by_dim in groups.items():
        stratum_summary = {"n": len(by_dim[DIMENSIONS[0]]), "dimensions": {}}
        for dim in DIMENSIONS:
            values = by_dim[dim]
            entry: dict = {"median": median(values), "distribution": _level_counts(values)}
            if stratum != "S0":
                delta = cliffs_delta(values, groups["S0"][dim])
                ci_lo, ci_hi = bootstrap_cliffs_delta_ci(values, groups["S0"][dim])
                entry["cliffs_delta_vs_s0"] = delta
                entry["cliffs_delta_ci95"] = [ci_lo, ci_hi]
            stratum_summary["dimensions"][dim] = entry
        summary["strata"][stratum] = stratum_summary
    return summary


def _level_counts(values: list[int]) -> dict[str, int]:
    return {str(level): values.count(level) for level in range(1, 6)}


def main() -> None:
    e1_rows = load_e1_scored_rows()
    row_labels_by_id = {row["hesum_id"]: row for row in load_json(ROW_LABELS_PATH)}
    groups = join_e1_to_strata(e1_rows, row_labels_by_id)
    summary = build_e1_summary(groups)
    save_json(E1_SUMMARY_PATH, summary)

    for stratum, data in summary["strata"].items():
        print(f"{stratum}: n={data['n']}")
        for dim, entry in data["dimensions"].items():
            delta_str = f", delta={entry['cliffs_delta_vs_s0']:.3f}" if "cliffs_delta_vs_s0" in entry else ""
            print(f"  {dim}: median={entry['median']}{delta_str}")
    print(f"Saved to: {E1_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
