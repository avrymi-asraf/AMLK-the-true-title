"""Score or summarize the final E4 uncleaned-versus-curated comparison.

The CLI performs the expensive Gemini rubric and blind pairwise calls. The
``summarize_frozen`` function provides an API-free summary path for saved scoring artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from pipeline.common.json_io import save_json
from pipeline.common.paths import SUMMARIES_DIR, TRAINING_EXPERIMENT_ARTIFACTS_DIR
from pipeline.common.statistics import bootstrap_cliffs_delta_ci, cliffs_delta, wilson_ci
from pipeline.stage_02_evaluation_instrument.gemini_client import GEMINI_MODEL, strip_think
from pipeline.stage_02_evaluation_instrument.pairwise_judge import compare_headlines
from pipeline.stage_02_evaluation_instrument.rubric_judge import DIMENSIONS, score_headline


PREDICTIONS_UNCLEANED_PATH = TRAINING_EXPERIMENT_ARTIFACTS_DIR / "predictions_uncleaned.jsonl"
PREDICTIONS_CURATED_PATH = TRAINING_EXPERIMENT_ARTIFACTS_DIR / "predictions_curated.jsonl"
RUBRIC_SCORES_PATH = TRAINING_EXPERIMENT_ARTIFACTS_DIR / "rubric_scores.jsonl"
PAIRWISE_PATH = TRAINING_EXPERIMENT_ARTIFACTS_DIR / "pairwise_judgments.jsonl"
ARTIFACT_SUMMARY_PATH = TRAINING_EXPERIMENT_ARTIFACTS_DIR / "summary.json"
RESULT_SUMMARY_PATH = SUMMARIES_DIR / "e4.json"
RUBRIC_RESULT_SUMMARY_PATH = SUMMARIES_DIR / "e4_rubric.json"
PAIRWISE_RESULT_SUMMARY_PATH = SUMMARIES_DIR / "e4_pairwise.json"
DECISION_RULE = (
    "curation wins if the pairwise Wilson CI excludes 50% or the paired "
    "faithfulness Cliff's-delta CI excludes 0"
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def load_predictions(path: Path) -> list[dict]:
    rows = load_jsonl(path)
    for line_number, row in enumerate(rows, start=1):
        text = row.get("text")
        prediction = strip_think(row.get("prediction") or "")
        if not isinstance(text, str) or not text or not prediction:
            raise ValueError(f"{path}:{line_number} lacks non-empty text or prediction")
        row["prediction"] = prediction
    return rows


def index_by_text(rows: list[dict], label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for row in rows:
        text = row["text"]
        if text in indexed:
            raise ValueError(f"{label} predictions contain duplicate article text")
        indexed[text] = row
    return indexed


def pair_by_text(uncleaned_rows: list[dict], curated_rows: list[dict]) -> list[dict]:
    uncleaned = index_by_text(uncleaned_rows, "uncleaned")
    curated = index_by_text(curated_rows, "curated")
    if set(uncleaned) != set(curated):
        raise ValueError(
            "E4 prediction arms do not cover the same article texts "
            f"(uncleaned={len(uncleaned)}, curated={len(curated)}, shared={len(set(uncleaned) & set(curated))})"
        )
    return [
        {
            "text": text,
            "uncleaned_prediction": uncleaned[text]["prediction"],
            "curated_prediction": curated[text]["prediction"],
        }
        for text in sorted(uncleaned)
    ]


def _flat_scores(scores: dict) -> dict[str, int]:
    flat: dict[str, int] = {}
    for dimension in DIMENSIONS:
        entry = scores.get(dimension)
        score = entry.get("score") if isinstance(entry, dict) else None
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"judge returned an invalid {dimension} score")
        flat[dimension] = score
    return flat


def score_rubric(pairs: list[dict], model) -> list[dict]:
    rows: list[dict] = []
    for index, pair in enumerate(pairs, start=1):
        uncleaned_scores = _flat_scores(score_headline(pair["text"], pair["uncleaned_prediction"], model=model, temperature=0.0))
        curated_scores = _flat_scores(score_headline(pair["text"], pair["curated_prediction"], model=model, temperature=0.0))
        rows.append({
            "text_prefix": pair["text"][:80],
            "raw_scores": uncleaned_scores,
            "curated_scores": curated_scores,
        })
        if index % 25 == 0 or index == len(pairs):
            print(f"Rubric scored: {index}/{len(pairs)}")
    return rows


def rubric_paired_summary(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("E4 rubric artifact is empty")
    output: dict = {"n_pairs_attempted": len(rows), "by_dimension": {}}
    for dimension in DIMENSIONS:
        uncleaned_values: list[int] = []
        curated_values: list[int] = []
        for index, row in enumerate(rows, start=1):
            raw_scores = row.get("raw_scores")
            curated_scores = row.get("curated_scores")
            if not isinstance(raw_scores, dict) or not isinstance(curated_scores, dict):
                raise ValueError(f"E4 rubric row {index} lacks both score dictionaries")
            raw_score = raw_scores.get(dimension)
            curated_score = curated_scores.get(dimension)
            if any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5 for value in (raw_score, curated_score)):
                raise ValueError(f"E4 rubric row {index} has an invalid {dimension} pair")
            uncleaned_values.append(raw_score)
            curated_values.append(curated_score)
        delta = cliffs_delta(curated_values, uncleaned_values)
        low, high = bootstrap_cliffs_delta_ci(curated_values, uncleaned_values)
        entry = {
            "n_pairs": len(uncleaned_values),
            "raw_mean": round(sum(uncleaned_values) / len(uncleaned_values), 3),
            "curated_mean": round(sum(curated_values) / len(curated_values), 3),
            "mean_delta_curated_minus_raw": round(
                sum(curated - raw for raw, curated in zip(uncleaned_values, curated_values)) / len(uncleaned_values),
                3,
            ),
            "cliffs_delta_curated_vs_raw": round(delta, 4),
            "cliffs_ci": [round(low, 4), round(high, 4)],
            "ci_excludes_0": bool(high < 0 or low > 0),
        }
        output["by_dimension"][dimension] = entry
        output[f"raw_{dimension}_mean"] = entry["raw_mean"]
        output[f"curated_{dimension}_mean"] = entry["curated_mean"]
        output[f"{dimension}_mean_delta_curated_minus_raw"] = entry["mean_delta_curated_minus_raw"]
        output[f"{dimension}_cliffs_delta_curated_vs_raw"] = entry["cliffs_delta_curated_vs_raw"]
        output[f"{dimension}_cliffs_ci"] = entry["cliffs_ci"]
        output[f"{dimension}_ci_excludes_0"] = entry["ci_excludes_0"]
        output[f"n_{dimension}_pairs"] = entry["n_pairs"]
    return output


def arm_wins(winner: str, curated_is_a: bool) -> str:
    if winner == "tie":
        return "tie"
    return "curated" if (winner == "a") == curated_is_a else "raw"


def run_pairwise(pairs: list[dict], model, seed: int = 42) -> list[dict]:
    randomizer = random.Random(seed)
    rows: list[dict] = []
    for index, pair in enumerate(pairs, start=1):
        curated_is_a = randomizer.random() < 0.5
        headline_a, headline_b = (
            (pair["curated_prediction"], pair["uncleaned_prediction"])
            if curated_is_a
            else (pair["uncleaned_prediction"], pair["curated_prediction"])
        )
        verdict = compare_headlines(pair["text"], headline_a, headline_b, model=model)
        winner = verdict.get("winner")
        if winner not in {"a", "b", "tie"}:
            raise ValueError(f"pairwise judge returned an invalid winner at row {index}")
        rows.append({
            "curated_is_a": curated_is_a,
            "winner": winner,
            "outcome": arm_wins(winner, curated_is_a),
            "justification": verdict.get("justification", ""),
            "text_prefix": pair["text"][:80],
        })
        if index % 25 == 0 or index == len(pairs):
            print(f"Pairwise judged: {index}/{len(pairs)}")
    return rows


def pairwise_summary(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("E4 pairwise artifact is empty")
    outcomes = [row.get("outcome") for row in rows]
    if any(outcome not in {"curated", "raw", "tie"} for outcome in outcomes):
        raise ValueError("E4 pairwise artifact contains an invalid outcome")
    curated = outcomes.count("curated")
    raw = outcomes.count("raw")
    ties = outcomes.count("tie")
    decided = curated + raw
    low, high = wilson_ci(curated, decided) if decided else (0.0, 0.0)
    return {
        "n_judged": len(rows),
        "curated_wins": curated,
        "raw_wins": raw,
        "ties": ties,
        "failed": 0,
        "curated_win_rate_pct": round(100 * curated / decided, 2) if decided else 0.0,
        "wilson_ci_pct": [round(low, 2), round(high, 2)],
        "excludes_50": bool(decided and (high < 50 or low > 50)),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary(rubric_rows: list[dict], pairwise_rows: list[dict]) -> dict:
    pointwise = rubric_paired_summary(rubric_rows)
    pairwise = pairwise_summary(pairwise_rows)
    return {
        "instrument": "rubric_v1",
        "judge_model": GEMINI_MODEL,
        "judge_temperature": 0.0,
        "dimensions": list(DIMENSIONS),
        "n_pairs": len(rubric_rows),
        "decision_rule": DECISION_RULE,
        "pointwise": pointwise,
        "pairwise": pairwise,
        "curation_wins_by_decision_rule": bool(
            pairwise["excludes_50"] or pointwise["faithfulness_ci_excludes_0"]
        ),
    }


def summarize_frozen(
    rubric_path: Path = RUBRIC_SCORES_PATH,
    pairwise_path: Path = PAIRWISE_PATH,
    output_path: Path = RESULT_SUMMARY_PATH,
) -> dict:
    rubric_rows = load_jsonl(rubric_path)
    pairwise_rows = load_jsonl(pairwise_path)
    if len(rubric_rows) != len(pairwise_rows):
        raise ValueError(
            f"E4 rubric/pairwise coverage differs ({len(rubric_rows)} vs {len(pairwise_rows)})"
        )
    if len(rubric_rows) != 120:
        raise ValueError(f"expected 120 paired E4 rows, found {len(rubric_rows)}")
    summary = build_summary(rubric_rows, pairwise_rows)
    save_json(output_path, summary)
    return summary


def summarize_frozen_rubric(
    rubric_path: Path = RUBRIC_SCORES_PATH,
    output_path: Path = RUBRIC_RESULT_SUMMARY_PATH,
) -> dict:
    rows = load_jsonl(rubric_path)
    if len(rows) != 120:
        raise ValueError(f"expected 120 E4 rubric rows, found {len(rows)}")
    summary = {
        "instrument": "rubric_v1",
        "judge_model": GEMINI_MODEL,
        "judge_temperature": 0.0,
        "dimensions": list(DIMENSIONS),
        "n_pairs": len(rows),
        "pointwise": rubric_paired_summary(rows),
    }
    save_json(output_path, summary)
    return summary


def summarize_frozen_pairwise(
    pairwise_path: Path = PAIRWISE_PATH,
    output_path: Path = PAIRWISE_RESULT_SUMMARY_PATH,
) -> dict:
    rows = load_jsonl(pairwise_path)
    if len(rows) != 120:
        raise ValueError(f"expected 120 E4 pairwise rows, found {len(rows)}")
    summary = {
        "judge_model": GEMINI_MODEL,
        "judge_temperature": 0.0,
        "n_pairs": len(rows),
        "pairwise": pairwise_summary(rows),
    }
    save_json(output_path, summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uncleaned", type=Path, default=PREDICTIONS_UNCLEANED_PATH)
    parser.add_argument("--curated", type=Path, default=PREDICTIONS_CURATED_PATH)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is required for live E4 scoring")
    pairs = pair_by_text(load_predictions(args.uncleaned), load_predictions(args.curated))
    if args.limit and len(pairs) > args.limit:
        selected = sorted(random.Random(args.seed).sample(range(len(pairs)), args.limit))
        pairs = [pairs[index] for index in selected]
    if len(pairs) != 120:
        raise ValueError(f"the final E4 scoring set must contain 120 paired rows, found {len(pairs)}")

    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(GEMINI_MODEL)
    rubric_rows = score_rubric(pairs, model)
    pairwise_rows = run_pairwise(pairs, model, seed=args.seed)
    write_jsonl(RUBRIC_SCORES_PATH, rubric_rows)
    write_jsonl(PAIRWISE_PATH, pairwise_rows)
    summary = build_summary(rubric_rows, pairwise_rows)
    save_json(ARTIFACT_SUMMARY_PATH, summary)
    save_json(RESULT_SUMMARY_PATH, summary)
    print(f"E4 frozen rubric scores: {RUBRIC_SCORES_PATH}")
    print(f"E4 frozen pairwise judgments: {PAIRWISE_PATH}")
    print(f"E4 summary: {RESULT_SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
