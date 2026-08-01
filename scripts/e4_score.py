"""
E4 scoring driver: faithfulness/fluency deltas + blind pairwise win rate.

Pipeline role: after E4-RAW and E4-CUR adapters produce prediction JSONLs (same
shared curated test articles), this script runs the article-grounded Gemini judge
on both files, pairs rows by article text, runs pairwise_judge with randomized
sides, and prints Cliff's δ + Wilson CI via data_curation.analysis.stats — the
same statistics as E1–E3. ROUGE/BERTScore are available via evaluation.evaluate
but are style-confounded on curated references and are not the decision rule.

Run (from repo root, with GEMINI_API_KEY):
  python -m scripts.e4_score \\
      --raw outputs/results/predictions-e4-raw.jsonl \\
      --curated outputs/results/predictions-e4-curated.jsonl \\
      --limit 120 --output outputs/results/e4-score-summary.json

Execution environment: local machine, API-bound (Gemini), CPU only. No GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from data_curation.analysis.stats import (
    bootstrap_cliffs_delta_ci,
    cliffs_delta,
    wilson_ci,
)
from evaluation.evaluate import _judge_scores, sample_for_judge
from evaluation.gemini_client import GEMINI_MODEL, strip_think
from evaluation.pairwise_judge import compare_headlines


def load_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["prediction"] = strip_think(row.get("prediction") or "")
            rows.append(row)
    return rows


def index_by_text(rows: list[dict]) -> dict[str, dict]:
    """First row per article text (predictions are unique per test article)."""
    out: dict[str, dict] = {}
    for r in rows:
        text = r.get("text") or ""
        if text and text not in out:
            out[text] = r
    return out


def pair_by_text(raw_rows: list[dict], cur_rows: list[dict]) -> list[dict]:
    """Join arms on article text; drop unpaired rows."""
    raw_ix = index_by_text(raw_rows)
    cur_ix = index_by_text(cur_rows)
    shared = sorted(set(raw_ix) & set(cur_ix))
    return [
        {
            "text": t,
            "raw_prediction": raw_ix[t]["prediction"],
            "curated_prediction": cur_ix[t]["prediction"],
            "reference": cur_ix[t].get("reference") or cur_ix[t].get("summary") or "",
        }
        for t in shared
    ]


def arm_wins(winner: str, curated_is_a: bool) -> str:
    """Map judge winner + side randomization → curated | raw | tie."""
    if winner == "tie":
        return "tie"
    curated_won = (winner == "a") == curated_is_a
    return "curated" if curated_won else "raw"


def run_pairwise(
    pairs: list[dict],
    seed: int = 42,
    model=None,
) -> list[dict]:
    """Blind A/B for each pair; randomize which arm is Headline A."""
    rng = random.Random(seed)
    results: list[dict] = []
    for i, p in enumerate(pairs):
        curated_is_a = rng.random() < 0.5
        if curated_is_a:
            a, b = p["curated_prediction"], p["raw_prediction"]
        else:
            a, b = p["raw_prediction"], p["curated_prediction"]
        verdict = compare_headlines(p["text"], a, b, model=model)
        winner = verdict.get("winner", "")
        outcome = arm_wins(winner, curated_is_a) if winner else "failed"
        results.append({
            "curated_is_a": curated_is_a,
            "winner": winner,
            "outcome": outcome,
            "justification": verdict.get("justification", ""),
            "text_prefix": p["text"][:80],
        })
        if (i + 1) % 25 == 0:
            print(f"  pairwise {i + 1}/{len(pairs)}", file=sys.stderr)
    return results


def pairwise_summary(results: list[dict]) -> dict:
    """Win counts + Wilson CI for curated win rate (excluding ties/failures)."""
    outcomes = [r["outcome"] for r in results if r["outcome"] in ("curated", "raw", "tie")]
    n = len(outcomes)
    curated_n = sum(1 for o in outcomes if o == "curated")
    raw_n = sum(1 for o in outcomes if o == "raw")
    tie_n = sum(1 for o in outcomes if o == "tie")
    decided = curated_n + raw_n
    win_rate = (100.0 * curated_n / decided) if decided else 0.0
    lo, hi = wilson_ci(curated_n, decided) if decided else (0.0, 0.0)
    return {
        "n_judged": n,
        "curated_wins": curated_n,
        "raw_wins": raw_n,
        "ties": tie_n,
        "failed": sum(1 for r in results if r["outcome"] == "failed"),
        "curated_win_rate_pct": round(win_rate, 2),
        "wilson_ci_pct": [round(lo, 2), round(hi, 2)],
        "excludes_50": bool(decided and (hi < 50.0 or lo > 50.0)),
    }


def judge_arm_means(rows: list[dict], limit: int, seed: int) -> dict:
    """Gemini faithfulness/fluency means (temperature pinned in evaluate path)."""
    subset = sample_for_judge(rows, limit, seed) if limit else rows
    # Shape for _judge_scores: needs text + prediction.
    prepared = [{"text": r["text"], "prediction": r["prediction"]} for r in subset]
    return _judge_scores("gemini", GEMINI_MODEL, None, prepared)


def paired_score_deltas(
    raw_judge: dict,
    cur_judge: dict,
    raw_rows: list[dict],
    cur_rows: list[dict],
    limit: int,
    seed: int,
) -> dict:
    """Paired per-article faith/flu deltas when both arms scored the same subset.

    evaluate.sample_for_judge uses the same seed+limit over each list independently;
    for true pairing we re-sample indices from the joined set by text order.
    """
    pairs = pair_by_text(raw_rows, cur_rows)
    if limit and limit < len(pairs):
        indices = sorted(random.Random(seed).sample(range(len(pairs)), limit))
        pairs = [pairs[i] for i in indices]

    # Re-score is expensive; prefer per_example lists aligned by the same subset
    # only when both reports were produced from the same ordered join. Callers
    # typically pass fresh _judge_scores results — we compute deltas from those
    # per_example arrays when lengths match; otherwise report means only.
    raw_pe = raw_judge.get("per_example") or []
    cur_pe = cur_judge.get("per_example") or []
    out: dict = {
        "n_pairs_attempted": len(pairs),
        "raw_faithfulness_mean": raw_judge.get("faithfulness_mean"),
        "curated_faithfulness_mean": cur_judge.get("faithfulness_mean"),
        "raw_fluency_mean": raw_judge.get("fluency_mean"),
        "curated_fluency_mean": cur_judge.get("fluency_mean"),
    }
    if len(raw_pe) == len(cur_pe) and raw_pe:
        faith_r = [x["faithfulness"] for x in raw_pe if isinstance(x.get("faithfulness"), (int, float))]
        faith_c = [x["faithfulness"] for x in cur_pe if isinstance(x.get("faithfulness"), (int, float))]
        flu_r = [x["fluency"] for x in raw_pe if isinstance(x.get("fluency"), (int, float))]
        flu_c = [x["fluency"] for x in cur_pe if isinstance(x.get("fluency"), (int, float))]
        # Align by min length (same judge order only if both arms used the same sample).
        n = min(len(faith_r), len(faith_c))
        if n >= 2:
            fr, fc = faith_r[:n], faith_c[:n]
            # cliffs_delta(a, b): positive means a tends higher than b.
            # Here a=curated, b=raw so positive = curated better.
            delta = cliffs_delta([int(x) for x in fc], [int(x) for x in fr])
            lo, hi = bootstrap_cliffs_delta_ci(
                [int(x) for x in fc], [int(x) for x in fr],
            )
            out["faithfulness_cliffs_delta_curated_vs_raw"] = round(delta, 4)
            out["faithfulness_cliffs_ci"] = [round(lo, 4), round(hi, 4)]
            out["faithfulness_ci_excludes_0"] = bool(hi < 0 or lo > 0)
            out["n_faith_pairs"] = n
        n_f = min(len(flu_r), len(flu_c))
        if n_f >= 2:
            flr, flc = flu_r[:n_f], flu_c[:n_f]
            delta = cliffs_delta([int(x) for x in flc], [int(x) for x in flr])
            lo, hi = bootstrap_cliffs_delta_ci(
                [int(x) for x in flc], [int(x) for x in flr],
            )
            out["fluency_cliffs_delta_curated_vs_raw"] = round(delta, 4)
            out["fluency_cliffs_ci"] = [round(lo, 4), round(hi, 4)]
            out["fluency_ci_excludes_0"] = bool(hi < 0 or lo > 0)
            out["n_fluency_pairs"] = n_f
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="E4: score raw vs curated prediction files (judge + pairwise)",
    )
    parser.add_argument("--raw", type=Path, required=True, help="E4-RAW predictions JSONL")
    parser.add_argument("--curated", type=Path, required=True, help="E4-CUR predictions JSONL")
    parser.add_argument(
        "--limit", type=int, default=120,
        help="Judge/pairwise subset size (default 120; 0 = all paired rows)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Subset + side-randomization seed")
    parser.add_argument(
        "--skip-pointwise", action="store_true",
        help="Skip per-arm faithfulness/fluency judge (pairwise only)",
    )
    parser.add_argument(
        "--skip-pairwise", action="store_true",
        help="Skip blind pairwise A/B (pointwise only)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/results/e4-score-summary.json"),
        help="Summary JSON path",
    )
    parser.add_argument(
        "--pairwise-jsonl", type=Path, default=None,
        help="Optional path for per-row pairwise verdicts JSONL",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        return 1

    raw_rows = load_predictions(args.raw)
    cur_rows = load_predictions(args.curated)
    if not raw_rows or not cur_rows:
        print("ERROR: empty predictions file", file=sys.stderr)
        return 1

    pairs = pair_by_text(raw_rows, cur_rows)
    print(f"Paired by article text: {len(pairs)} "
          f"(raw={len(raw_rows)}, curated={len(cur_rows)})")
    if not pairs:
        print("ERROR: no shared article texts between arms", file=sys.stderr)
        return 1

    if args.limit and args.limit < len(pairs):
        indices = sorted(random.Random(args.seed).sample(range(len(pairs)), args.limit))
        pairs = [pairs[i] for i in indices]
        print(f"Subset: {len(pairs)} pairs (seed={args.seed})")

    summary: dict = {
        "raw_file": str(args.raw),
        "curated_file": str(args.curated),
        "n_pairs": len(pairs),
        "limit": args.limit,
        "seed": args.seed,
        "judge_model": GEMINI_MODEL,
        "judge_temperature": 0.0,
        "decision_rule": (
            "curation wins if pairwise Wilson CI excludes 50% "
            "OR paired judge cliffs CI excludes 0"
        ),
    }

    if not args.skip_pointwise:
        print(f"Pointwise judge on raw arm ({GEMINI_MODEL}, T=0)...")
        # Score only the paired subset's predictions for each arm.
        raw_subset = [
            {"text": p["text"], "prediction": p["raw_prediction"]} for p in pairs
        ]
        cur_subset = [
            {"text": p["text"], "prediction": p["curated_prediction"]} for p in pairs
        ]
        raw_judge = _judge_scores("gemini", GEMINI_MODEL, None, raw_subset)
        print(f"Pointwise judge on curated arm ({GEMINI_MODEL}, T=0)...")
        cur_judge = _judge_scores("gemini", GEMINI_MODEL, None, cur_subset)
        summary["pointwise"] = paired_score_deltas(
            raw_judge, cur_judge, raw_rows, cur_rows, 0, args.seed,
        )
        # Overwrite with aligned per_example from the same pair order.
        faith_r = [x["faithfulness"] for x in raw_judge["per_example"]
                   if isinstance(x.get("faithfulness"), (int, float))]
        faith_c = [x["faithfulness"] for x in cur_judge["per_example"]
                   if isinstance(x.get("faithfulness"), (int, float))]
        flu_r = [x["fluency"] for x in raw_judge["per_example"]
                 if isinstance(x.get("fluency"), (int, float))]
        flu_c = [x["fluency"] for x in cur_judge["per_example"]
                 if isinstance(x.get("fluency"), (int, float))]
        n = min(len(faith_r), len(faith_c))
        if n >= 2:
            fr = [int(x) for x in faith_r[:n]]
            fc = [int(x) for x in faith_c[:n]]
            delta = cliffs_delta(fc, fr)
            lo, hi = bootstrap_cliffs_delta_ci(fc, fr)
            summary["pointwise"]["faithfulness_cliffs_delta_curated_vs_raw"] = round(delta, 4)
            summary["pointwise"]["faithfulness_cliffs_ci"] = [round(lo, 4), round(hi, 4)]
            summary["pointwise"]["faithfulness_ci_excludes_0"] = bool(hi < 0 or lo > 0)
            summary["pointwise"]["n_faith_pairs"] = n
            # Paired mean delta for readability.
            summary["pointwise"]["faithfulness_mean_delta_curated_minus_raw"] = round(
                sum(fc[i] - fr[i] for i in range(n)) / n, 3,
            )
        n_f = min(len(flu_r), len(flu_c))
        if n_f >= 2:
            flr = [int(x) for x in flu_r[:n_f]]
            flc = [int(x) for x in flu_c[:n_f]]
            delta = cliffs_delta(flc, flr)
            lo, hi = bootstrap_cliffs_delta_ci(flc, flr)
            summary["pointwise"]["fluency_cliffs_delta_curated_vs_raw"] = round(delta, 4)
            summary["pointwise"]["fluency_cliffs_ci"] = [round(lo, 4), round(hi, 4)]
            summary["pointwise"]["fluency_ci_excludes_0"] = bool(hi < 0 or lo > 0)
            summary["pointwise"]["n_fluency_pairs"] = n_f
            summary["pointwise"]["fluency_mean_delta_curated_minus_raw"] = round(
                sum(flc[i] - flr[i] for i in range(n_f)) / n_f, 3,
            )
        summary["pointwise"]["raw_faithfulness_mean"] = raw_judge.get("faithfulness_mean")
        summary["pointwise"]["curated_faithfulness_mean"] = cur_judge.get("faithfulness_mean")
        summary["pointwise"]["raw_fluency_mean"] = raw_judge.get("fluency_mean")
        summary["pointwise"]["curated_fluency_mean"] = cur_judge.get("fluency_mean")

    if not args.skip_pairwise:
        print(f"Blind pairwise on {len(pairs)} pairs...")
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(GEMINI_MODEL)
        pairwise_rows = run_pairwise(pairs, seed=args.seed, model=model)
        summary["pairwise"] = pairwise_summary(pairwise_rows)
        out_jsonl = args.pairwise_jsonl or args.output.with_suffix(".pairwise.jsonl")
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for row in pairwise_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Pairwise rows: {out_jsonl}")

    # Decision rule convenience flag.
    pairwise_ok = summary.get("pairwise", {}).get("excludes_50", False)
    faith_ok = summary.get("pointwise", {}).get("faithfulness_ci_excludes_0", False)
    summary["curation_wins_by_decision_rule"] = bool(pairwise_ok or faith_ok)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Summary: {args.output}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
