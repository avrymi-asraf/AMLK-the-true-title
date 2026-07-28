"""Build the frozen 150-row human-validation worklist for F9a (appendix judge validation).

Stratified across S0/S2/S3/S4 like `rubric_pilot.py`, excluding anchor rows. Each row carries
article text and headline(s) so annotators need only `human_validation_worklist.json` — no
regenerable pipeline artifacts. Rubric scores the original headline; pairwise (subset of
rewritten rows) compares original vs curated blindly. Local, CPU-only.

Run:
    python -m data_curation.analysis.build_human_validation_sample

Output:
    data_curation/artifacts/human_validation_worklist.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data_curation.analysis.rubric_anchors import ANCHOR_HESUM_IDS
from data_curation.analysis.rubric_pilot import STRATA, TAIL_BOILERPLATE_REMOVED_PATH, stratified_sample
from data_curation.analysis.row_labels import load_row_labels
from data_curation.utils.paths import ARTIFACTS_DIR

WORKLIST_PATH = ARTIFACTS_DIR / "human_validation_worklist.json"
FINAL_CLEAN_PATH = ARTIFACTS_DIR / "final_clean_hesum.json"


def row_strata(row: dict) -> list[str]:
    """Return all analysis stratum names this row belongs to."""
    return [name for name, predicate in STRATA.items() if predicate(row)]


def unique_rubric_ids(sampled: dict[str, list[dict]]) -> list[dict]:
    """Deduplicate stratified draw to one entry per hesum_id, keeping stratum membership."""
    by_id: dict[str, dict] = {}
    for stratum_name, stratum_rows in sampled.items():
        for row in stratum_rows:
            hesum_id = row["hesum_id"]
            if hesum_id not in by_id:
                by_id[hesum_id] = {"hesum_id": hesum_id, "strata": [], "row_label": row}
            if stratum_name not in by_id[hesum_id]["strata"]:
                by_id[hesum_id]["strata"].append(stratum_name)
    return list(by_id.values())


def select_pairwise_ids(
    rubric_entries: list[dict],
    pairwise_n: int,
    seed: int,
) -> set[str]:
    """Sample rewritten rows from the rubric pool for blind pairwise comparison."""
    rewritten = [
        entry["hesum_id"]
        for entry in rubric_entries
        if entry["row_label"].get("headline_action") == "rewritten"
    ]
    rng = random.Random(seed + 2)
    rng.shuffle(rewritten)
    return set(rewritten[:pairwise_n])


def load_tail_by_id(hesum_ids: set[str]) -> dict[str, dict]:
    with open(TAIL_BOILERPLATE_REMOVED_PATH, encoding="utf-8") as f:
        records = json.load(f)
    return {r["id"]: r for r in records if r["id"] in hesum_ids}


def load_curated_by_id(hesum_ids: set[str]) -> dict[str, dict]:
    with open(FINAL_CLEAN_PATH, encoding="utf-8") as f:
        records = json.load(f)
    return {r["hesum_id"]: r for r in records if r["hesum_id"] in hesum_ids}


def build_worklist_rows(
    rubric_entries: list[dict],
    pairwise_ids: set[str],
    tail_by_id: dict[str, dict],
    curated_by_id: dict[str, dict],
) -> list[dict]:
    """Assemble self-contained worklist rows with embedded text and task lists."""
    rows = []
    for entry in rubric_entries:
        hesum_id = entry["hesum_id"]
        tail = tail_by_id.get(hesum_id)
        if not tail or not tail.get("text") or not tail.get("headline"):
            continue

        tasks = ["rubric"]
        curated_headline = None
        if hesum_id in pairwise_ids:
            curated = curated_by_id.get(hesum_id)
            if curated and curated.get("headline"):
                tasks.append("pairwise")
                curated_headline = curated["headline"]

        if "pairwise" in tasks and curated_headline is None:
            tasks = ["rubric"]

        row = {
            "hesum_id": hesum_id,
            "tasks": tasks,
            "strata": entry["strata"],
            "text": tail["text"],
            "original_headline": tail["headline"],
        }
        if curated_headline is not None:
            row["curated_headline"] = curated_headline
        rows.append(row)

    rows.sort(key=lambda r: int(r["hesum_id"]) if r["hesum_id"].isdigit() else r["hesum_id"])
    return rows


def build_human_validation_worklist(
    *,
    n_per_stratum: int = 38,
    pairwise_n: int = 50,
    seed: int = 42,
    row_labels: list[dict] | None = None,
) -> dict:
    """Build the worklist dict (pure function for tests)."""
    labels = row_labels if row_labels is not None else load_row_labels()
    sampled = stratified_sample(labels, n_per_stratum, seed)
    rubric_entries = unique_rubric_ids(sampled)
    pairwise_ids = select_pairwise_ids(rubric_entries, pairwise_n, seed)

    ids = {entry["hesum_id"] for entry in rubric_entries}
    tail_by_id = load_tail_by_id(ids)
    curated_by_id = load_curated_by_id(ids)
    rows = build_worklist_rows(rubric_entries, pairwise_ids, tail_by_id, curated_by_id)

    return {
        "version": "v1",
        "seed": seed,
        "n_per_stratum": n_per_stratum,
        "pairwise_n": pairwise_n,
        "n_rows": len(rows),
        "n_pairwise": sum(1 for r in rows if "pairwise" in r["tasks"]),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build F9a human-validation worklist")
    parser.add_argument("--n-per-stratum", type=int, default=38,
                        help="Rows sampled per stratum before dedup (default 38 x 4 -> ~150 unique)")
    parser.add_argument("--pairwise-n", type=int, default=50,
                        help="Rewritten rows that also get a pairwise task")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(WORKLIST_PATH))
    args = parser.parse_args()

    worklist = build_human_validation_worklist(
        n_per_stratum=args.n_per_stratum,
        pairwise_n=args.pairwise_n,
        seed=args.seed,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(worklist, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {worklist['n_rows']} rows ({worklist['n_pairwise']} with pairwise) to {output}")


if __name__ == "__main__":
    main()
