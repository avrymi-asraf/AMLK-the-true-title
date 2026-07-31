"""
Measurement instrument for the training-improvement loop (docs/training-improvement-notebook.md).

Every improvement experiment is judged on ONE fixed test subset so arms are paired and
comparable: `subset_indices` picks the same rows of the test split for every run, `judge_file`
scores a predictions file's faithfulness/fluency per example (Gemini, temperature pinned), and
`paired_delta` reports mean difference vs a control arm with its standard error — the only
statistic allowed to decide whether a training change helped.

Execution environment: local machine (API + CPU only, no model load). GEMINI_API_KEY required.
Run: python -m evaluation.improve_eval judge --predictions f.jsonl --output f.judged.json
     python -m evaluation.improve_eval delta --arm a.judged.json --control b.judged.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluation.evaluate import JUDGE_PROMPT, _parse_json
from evaluation.gemini_client import (
    GEMINI_MODEL,
    GEMINI_TIMEOUT,
    JUDGE_GENERATION_CONFIG,
    call_with_retry,
    strip_think,
)

# The fixed evaluation subset. Small enough that one arm costs ~120 judge calls, large enough
# that the paired SE (see entry #0 of the notebook) can resolve a ~0.3 shift on a 1-5 scale.
SUBSET_SEED = 1234
SUBSET_N = 120


def subset_indices(total: int, n: int = SUBSET_N, seed: int = SUBSET_SEED) -> list[int]:
    """Deterministic sorted indices of the fixed eval subset within a split of `total` rows."""
    if n >= total:
        return list(range(total))
    return sorted(random.Random(seed).sample(range(total), n))


def load_predictions(path: str | Path) -> list[dict]:
    # Split on "\n" only: article text contains unicode line separators ( , ) that
    # str.splitlines() would treat as line breaks, cutting a JSON row in half.
    rows = [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]
    for r in rows:
        r["prediction"] = strip_think(r.get("prediction", ""))
    return rows


def take_subset(rows: list[dict]) -> list[dict]:
    """Restrict a full-test predictions file to the fixed subset (no-op if already subset-sized)."""
    if len(rows) <= SUBSET_N:
        return rows
    return [rows[i] for i in subset_indices(len(rows))]


def _judge_one(model, row: dict) -> dict:
    prompt = JUDGE_PROMPT.format(text=row["text"][:6000], prediction=row["prediction"])
    raw = call_with_retry(lambda: model.generate_content(
        prompt, generation_config=JUDGE_GENERATION_CONFIG,
        request_options={"timeout": GEMINI_TIMEOUT}).text)
    scores = _parse_json(raw)
    return {"faithfulness": scores.get("faithfulness"), "fluency": scores.get("fluency")}


def judge_file(rows: list[dict], workers: int = 4) -> dict:
    """Judge every row for faithfulness/fluency; returns per-example scores plus means.

    Rows keep their input order, so two judged files over the same subset are paired by index.
    """
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(GEMINI_MODEL)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        per_example = list(pool.map(lambda r: _judge_one(model, r), rows))
    return {
        "n": len(per_example),
        "judge_model": GEMINI_MODEL,
        "faithfulness_mean": _mean(per_example, "faithfulness"),
        "fluency_mean": _mean(per_example, "fluency"),
        "per_example": per_example,
    }


def _mean(per_example: list[dict], key: str) -> float | None:
    vals = [e[key] for e in per_example if isinstance(e[key], (int, float))]
    return round(sum(vals) / len(vals), 3) if vals else None


def paired_delta(arm: dict, control: dict, key: str) -> dict:
    """Mean paired difference (arm - control) on `key`, with SE over examples scored in both."""
    pairs = [(a[key], c[key]) for a, c in zip(arm["per_example"], control["per_example"])
             if isinstance(a[key], (int, float)) and isinstance(c[key], (int, float))]
    if not pairs:
        return {"metric": key, "n": 0}
    diffs = [a - c for a, c in pairs]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return {
        "metric": key,
        "n": n,
        "arm_mean": round(sum(a for a, _ in pairs) / n, 3),
        "control_mean": round(sum(c for _, c in pairs) / n, 3),
        "delta": round(mean, 3),
        "se": round(se, 3),
        "ci95": [round(mean - 1.96 * se, 3), round(mean + 1.96 * se, 3)],
    }


def main():
    parser = argparse.ArgumentParser(description="Judge/compare arms of the training-improvement loop")
    sub = parser.add_subparsers(dest="cmd", required=True)

    j = sub.add_parser("judge", help="judge a predictions file on the fixed subset")
    j.add_argument("--predictions", required=True)
    j.add_argument("--output", required=True)
    j.add_argument("--full", action="store_true", help="judge every row instead of the fixed subset")
    j.add_argument("--workers", type=int, default=4)

    d = sub.add_parser("delta", help="paired delta between a judged arm and a judged control")
    d.add_argument("--arm", required=True)
    d.add_argument("--control", required=True)

    a = sub.add_parser("arm", help="download an arm's Hub predictions, judge them, print the delta")
    a.add_argument("--repo", required=True, help="Hub model repo the training job pushed to")
    a.add_argument("--name", required=True, help="Short arm name (used for output filenames)")
    a.add_argument("--file", default="predictions-finetuned.jsonl")
    a.add_argument("--control", default="outputs/results/improve/base-judged-pass1.json")
    a.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()
    if args.cmd == "arm":
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(args.repo, args.file, repo_type="model")
        rows = take_subset(load_predictions(local))
        report = judge_file(rows, workers=args.workers)
        report["predictions"] = f"{args.repo}/{args.file}"
        out = Path("outputs/results/improve") / f"{args.name}-judged.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        control = json.loads(Path(args.control).read_text())
        print(f"n={report['n']}  faith={report['faithfulness_mean']}  flu={report['fluency_mean']}")
        for key in ("faithfulness", "fluency"):
            print(json.dumps(paired_delta(report, control, key), indent=2))
        return
    if args.cmd == "judge":
        rows = load_predictions(args.predictions)
        rows = rows if args.full else take_subset(rows)
        report = judge_file(rows, workers=args.workers)
        report["predictions"] = str(args.predictions)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({k: report[k] for k in ("n", "faithfulness_mean", "fluency_mean")}, indent=2))
    else:
        arm = json.loads(Path(args.arm).read_text())
        control = json.loads(Path(args.control).read_text())
        for key in ("faithfulness", "fluency"):
            print(json.dumps(paired_delta(arm, control, key), indent=2))


if __name__ == "__main__":
    main()
