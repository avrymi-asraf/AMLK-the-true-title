"""Evaluate the Reference Quality Rubric judge on the stratified pilot sample.

It samples across the four testable analysis strata (S0 clean, S2 multi_pipe_headline, S3
multiple_independent_items, S4 headline_rewritten), scores each row's ORIGINAL headline once, then
re-scores a subsample at nonzero temperature for test-retest reliability (quadratically weighted
kappa per dimension).
Reports per-dimension score distributions so a degenerate rubric (e.g. everything a 4 or 5) is caught
before corpus-scale scoring. Anchor rows
(`pipeline.stage_02_evaluation_instrument.rubric_anchors.ANCHOR_HESUM_IDS`) are excluded from
sampling so they cannot validate an instrument they were used to build. Local, API-bound
(GEMINI_API_KEY required), CPU-only — no GPU, no local model load.

Run:
    python -m pipeline.stage_02_evaluation_instrument.run_pilot [--n-per-stratum 75] [--retest-n 60] [--seed 42]

Output:
    artifacts/reference_experiments/rubric_pilot.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

from pipeline.common.paths import CURATION_WORK_DIR, REFERENCE_EXPERIMENT_ARTIFACTS_DIR
from pipeline.stage_01_data_curation.build_row_ledger import load_row_labels
from pipeline.stage_02_evaluation_instrument.rubric_anchors import ANCHOR_HESUM_IDS
from pipeline.stage_02_evaluation_instrument.rubric_judge import DIMENSIONS

PILOT_OUTPUT_PATH = REFERENCE_EXPERIMENT_ARTIFACTS_DIR / "rubric_pilot.json"
TAIL_BOILERPLATE_REMOVED_PATH = CURATION_WORK_DIR / "tail_boilerplate_removed.json"

# The four testable analysis strata from the design spec, section 3. S1 (over_token_budget) is
# deliberately absent (it's a covariate, not a stratum — see the spec's "Why there is no S1"); S5
# (other_unusable, 63 rows total) is prevalence-only and too small for a meaningful pilot slice.
STRATA = {
    "S0_clean": lambda r: (
        r["reached_model_curation"] and r["source_label"] == "usable" and r["headline_action"] == "kept"
    ),
    "S2_multi_pipe_headline": lambda r: r["multi_pipe"],
    "S3_multiple_independent_items": lambda r: r["source_label"] == "unusable_multiple_independent_items",
    "S4_headline_rewritten": lambda r: r["headline_action"] == "rewritten",
}


def stratified_sample(rows: list[dict], per_stratum: int, seed: int) -> dict[str, list[dict]]:
    """Sample up to `per_stratum` rows per analysis stratum, excluding anchor rows.

    Strata overlap by design (a row can be both a pipe digest and a rewrite target), so the same row
    can be drawn into more than one stratum's sample here — that mirrors how the real E1 regression
    treats strata as independent binary indicators rather than forcing a partition they don't have.
    """
    rng = random.Random(seed)
    eligible = [r for r in rows if r["hesum_id"] not in ANCHOR_HESUM_IDS]

    sampled = {}
    for stratum_name, predicate in STRATA.items():
        pool = [r for r in eligible if predicate(r)]
        rng.shuffle(pool)
        sampled[stratum_name] = pool[:per_stratum]
    return sampled


def load_text_by_id(hesum_ids: set[str]) -> dict[str, dict]:
    """Load {id: {text, headline}} for exactly the requested ids from the tail-trimmed records."""
    with open(TAIL_BOILERPLATE_REMOVED_PATH, encoding="utf-8") as f:
        records = json.load(f)
    return {r["id"]: r for r in records if r["id"] in hesum_ids}


def score_rows(rows_with_text: list[dict], model, temperature: float | None = None) -> list[dict]:
    """Score each row's original headline once, returning results tagged with id and stratum."""
    from pipeline.stage_02_evaluation_instrument.rubric_judge import score_headline

    scored = []
    for i, row in enumerate(rows_with_text):
        scores = score_headline(row["text"], row["headline"], model=model, temperature=temperature)
        scored.append({"hesum_id": row["hesum_id"], "stratum": row["stratum"], "scores": scores})
        if (i + 1) % 25 == 0:
            print(f"  scored {i + 1}/{len(rows_with_text)}")
    return scored


def score_distribution(scored: list[dict], dimension: str) -> dict[int, int]:
    """Count how many rows landed at each 1-5 level for one dimension, across all scored rows."""
    counts = Counter(row["scores"][dimension]["score"] for row in scored if dimension in row["scores"])
    return {level: counts.get(level, 0) for level in range(1, 6)}


def is_degenerate(distribution: dict[int, int], threshold: float = 0.85) -> bool:
    """Flag a dimension whose scores pile onto a single level — the rubric cannot discriminate."""
    total = sum(distribution.values())
    return total == 0 or max(distribution.values()) / total >= threshold


def weighted_kappa(scores_a: list[int], scores_b: list[int]) -> float:
    """Quadratically weighted Cohen's kappa between two scoring passes over the same rows."""
    from sklearn.metrics import cohen_kappa_score

    return round(cohen_kappa_score(scores_a, scores_b, weights="quadratic"), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pilot the Reference Quality Rubric judge")
    parser.add_argument("--n-per-stratum", type=int, default=75,
                         help="Rows sampled per stratum (default 75 x 4 strata, ~300 with overlap dedup)")
    parser.add_argument("--retest-n", type=int, default=60, help="Subsample size for the test-retest pass")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(PILOT_OUTPUT_PATH))
    args = parser.parse_args()

    import google.generativeai as genai

    from pipeline.stage_02_evaluation_instrument.gemini_client import GEMINI_MODEL

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(GEMINI_MODEL)

    print("Loading row labels and sampling strata...")
    rows = load_row_labels()
    sampled = stratified_sample(rows, args.n_per_stratum, args.seed)

    flat_rows = [
        {"hesum_id": r["hesum_id"], "stratum": stratum_name}
        for stratum_name, stratum_rows in sampled.items()
        for r in stratum_rows
    ]
    unique_ids = {r["hesum_id"] for r in flat_rows}
    print(f"Sampled {len(unique_ids)} unique rows across {len(sampled)} strata "
          f"({', '.join(f'{k}={len(v)}' for k, v in sampled.items())})")

    text_by_id = load_text_by_id(unique_ids)
    # One row per (id, stratum) pair for the per-stratum breakdown; a row in >1 stratum is scored
    # once per stratum tag it holds, matching the overlap-by-design semantics above.
    rows_with_text = [
        {**text_by_id[r["hesum_id"]], "hesum_id": r["hesum_id"], "stratum": r["stratum"]}
        for r in flat_rows
        if r["hesum_id"] in text_by_id
    ]

    print(f"Scoring {len(rows_with_text)} rows (pass 1)...")
    first_pass = score_rows(rows_with_text, model)

    rng = random.Random(args.seed + 1)
    retest_subset = rng.sample(rows_with_text, min(args.retest_n, len(rows_with_text)))
    print(f"Scoring {len(retest_subset)} rows again at temperature=0.7 (test-retest pass)...")
    second_pass = score_rows(retest_subset, model, temperature=0.7)

    first_by_id = {row["hesum_id"]: row["scores"] for row in first_pass}
    second_by_id = {row["hesum_id"]: row["scores"] for row in second_pass}
    retest_ids = [r["hesum_id"] for r in retest_subset]

    report = {
        "n_sampled": len(rows_with_text),
        "n_retest": len(retest_ids),
        "strata_sizes": {k: len(v) for k, v in sampled.items()},
        "score_distributions": {},
        "degenerate_dimensions": [],
        "test_retest_kappa": {},
        "first_pass": first_pass,
        "second_pass": second_pass,
    }

    for dimension in DIMENSIONS:
        dist = score_distribution(first_pass, dimension)
        report["score_distributions"][dimension] = dist
        if is_degenerate(dist):
            report["degenerate_dimensions"].append(dimension)

        paired = [
            (first_by_id[i][dimension]["score"], second_by_id[i][dimension]["score"])
            for i in retest_ids
            if dimension in first_by_id.get(i, {}) and dimension in second_by_id.get(i, {})
        ]
        if paired:
            a, b = zip(*paired)
            report["test_retest_kappa"][dimension] = weighted_kappa(list(a), list(b))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"\nSaved pilot report to {args.output}")
    print("\nScore distributions (1-5):")
    for dimension in DIMENSIONS:
        print(f"  {dimension}: {report['score_distributions'][dimension]}")
    print("\nTest-retest quadratic-weighted kappa:")
    for dimension, kappa in report["test_retest_kappa"].items():
        print(f"  {dimension}: {kappa}")
    if report["degenerate_dimensions"]:
        print(f"\nWARNING: degenerate score distribution in: {report['degenerate_dimensions']}")


if __name__ == "__main__":
    main()
