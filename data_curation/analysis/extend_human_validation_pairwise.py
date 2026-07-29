"""Extend the F9a worklist with extra pairwise-only rows for one annotator.

Used when an annotator wants more blind A/B comparisons without re-sampling the
full rubric worklist. Pulls rewritten rows from outside the current worklist,
assigns them to a single annotator, and appends `pairwise` tasks only.

Run:
    python -m data_curation.analysis.extend_human_validation_pairwise --annotator amit --count 20

Output:
    Updates data_curation/artifacts/human_validation_worklist.json in place.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from data_curation.analysis.build_human_validation_sample import (
    WORKLIST_ANNOTATOR_IDS,
    WORKLIST_PATH,
    assignment_counts,
    load_curated_by_id,
    load_tail_by_id,
)
from data_curation.analysis.row_labels import load_row_labels
from evaluation.viewer.annotation_data import load_worklist

DEFAULT_EXTRA_COUNT = 20


def eligible_rewritten_pool(worklist: dict, row_labels: list[dict]) -> list[dict]:
    """Usable rewritten rows not already in the worklist."""
    in_worklist = {r["hesum_id"] for r in worklist.get("rows", [])}
    pool = []
    for row in row_labels:
        hesum_id = row["hesum_id"]
        if hesum_id in in_worklist:
            continue
        if row.get("headline_action") != "rewritten":
            continue
        if not row.get("reached_model_curation"):
            continue
        if row.get("source_label") != "usable":
            continue
        pool.append(row)
    return pool


def sample_extra_rows(pool: list[dict], count: int, seed: int) -> list[dict]:
    """Deterministic random sample of rewritten rows."""
    if count > len(pool):
        raise ValueError(f"Requested {count} extra rows but only {len(pool)} eligible")
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    return shuffled[:count]


def build_pairwise_extension_rows(
    sampled: list[dict],
    *,
    annotator_id: str,
    tail_by_id: dict[str, dict],
    curated_by_id: dict[str, dict],
) -> list[dict]:
    """Build pairwise-only worklist rows."""
    rows = []
    for row_label in sampled:
        hesum_id = row_label["hesum_id"]
        tail = tail_by_id.get(hesum_id)
        curated = curated_by_id.get(hesum_id)
        if not tail or not tail.get("text") or not tail.get("headline"):
            continue
        if not curated or not curated.get("headline"):
            continue
        if curated["headline"].strip() == tail["headline"].strip():
            continue
        from data_curation.analysis.build_human_validation_sample import row_strata

        rows.append({
            "hesum_id": hesum_id,
            "tasks": ["pairwise"],
            "strata": row_strata(row_label),
            "text": tail["text"],
            "original_headline": tail["headline"],
            "curated_headline": curated["headline"],
            "assigned_annotator": annotator_id,
            "extension": "pairwise_extra",
        })
    rows.sort(key=lambda r: int(r["hesum_id"]) if r["hesum_id"].isdigit() else r["hesum_id"])
    return rows


def extend_worklist_pairwise(
    worklist: dict,
    *,
    annotator_id: str,
    count: int,
    seed: int,
    row_labels: list[dict] | None = None,
) -> dict:
    """Return an updated worklist with extra pairwise rows for one annotator."""
    if annotator_id not in WORKLIST_ANNOTATOR_IDS:
        raise ValueError(f"Unknown annotator {annotator_id!r}; expected one of {WORKLIST_ANNOTATOR_IDS}")

    labels = row_labels if row_labels is not None else load_row_labels()
    pool = eligible_rewritten_pool(worklist, labels)
    sampled = sample_extra_rows(pool, count, seed)
    ids = {r["hesum_id"] for r in sampled}
    tail_by_id = load_tail_by_id(ids)
    curated_by_id = load_curated_by_id(ids)
    extra_rows = build_pairwise_extension_rows(
        sampled,
        annotator_id=annotator_id,
        tail_by_id=tail_by_id,
        curated_by_id=curated_by_id,
    )
    if len(extra_rows) < count:
        raise RuntimeError(
            f"Only built {len(extra_rows)}/{count} extension rows (missing tail/curated text)"
        )

    updated = dict(worklist)
    updated["rows"] = list(worklist["rows"]) + extra_rows
    updated["rows"].sort(
        key=lambda r: int(r["hesum_id"]) if r["hesum_id"].isdigit() else r["hesum_id"]
    )
    updated["n_rows"] = len(updated["rows"])
    updated["n_pairwise"] = sum(1 for r in updated["rows"] if "pairwise" in r["tasks"])
    updated["assignment"] = assignment_counts(updated["rows"])
    extensions = list(updated.get("pairwise_extensions", []))
    extensions.append({
        "annotator": annotator_id,
        "added": len(extra_rows),
        "seed": seed,
        "hesum_ids": [r["hesum_id"] for r in extra_rows],
    })
    updated["pairwise_extensions"] = extensions
    updated["version"] = worklist.get("version", "v1-split") + "+pairwise-ext"
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Add extra pairwise rows for one F9a annotator")
    parser.add_argument("--annotator", required=True, choices=WORKLIST_ANNOTATOR_IDS)
    parser.add_argument("--count", type=int, default=DEFAULT_EXTRA_COUNT,
                        help=f"Extra pairwise rows to add (default {DEFAULT_EXTRA_COUNT})")
    parser.add_argument("--seed", type=int, default=43,
                        help="RNG seed for sampling outside-worklist rows")
    parser.add_argument("--worklist", default=str(WORKLIST_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    worklist = load_worklist(args.worklist)
    updated = extend_worklist_pairwise(
        worklist,
        annotator_id=args.annotator,
        count=args.count,
        seed=args.seed,
    )
    added = updated["n_rows"] - worklist["n_rows"]
    print(
        f"{'Would add' if args.dry_run else 'Added'} {added} pairwise-only rows for {args.annotator} "
        f"(total pairwise now {updated['n_pairwise']})"
    )
    print(f"Assignment: {updated['assignment']}")

    if not args.dry_run:
        Path(args.worklist).write_text(
            json.dumps(updated, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
