"""
Evaluation pipeline add-on: breaks down an existing predictions file's ROUGE/BERTScore/failure
rates by a per-summary label. Works with either dimension produced over the corpus: the semantic
`topic_label` from evaluation/topic_clustering.py's topics.jsonl (a Databricks GPU run), or the
structural `style_label` from evaluation/style_labels.py's style_labels.jsonl (local, rule-based
— e.g. pipe-separated multi-headline digest vs a single-sentence summary). Both artifacts share
the same shape (one JSONL row per article, keyed by summary text), so the same join/group/score
logic works for either via --label-field. Joins on exact text match — every prediction's
`reference` field is copied verbatim from the original summary (see evaluation/infer.py,
training/train_hf_job.py), so it's a stable join key back to the whole-corpus label assignments
without touching data/preprocess.py or the Arrow splits. Reuses evaluate.py's
compute_rouge/compute_bertscore per group; a matching *.errors.json (from error_analysis.py) is
folded in for per-group failure-type rates if present.

Run: python -m evaluation.stratify_by_topic --predictions outputs/results/predictions-finetuned.jsonl \
    --labels outputs/data/raw/topics.jsonl --label-field topic_label \
    --errors outputs/results/finetuned-v3.errors.json \
    --output outputs/results/finetuned-by-topic.json
Execution environment: local machine, CPU only (same as evaluate.py) — no GPU/Databricks needed
for this step; the label file must already exist (topics.jsonl from the clustering notebook, or
style_labels.jsonl from `python -m evaluation.style_labels`).
"""
import argparse
import json
from pathlib import Path

from evaluation.error_analysis import FAILURE_TYPES
from evaluation.evaluate import compute_bertscore, compute_rouge
from evaluation.gemini_client import strip_think
from evaluation.topic_clustering import NOISE_LABEL

DEFAULT_MIN_COUNT = 10


def load_label_rows(path: str) -> dict[str, dict]:
    """Map summary text -> full label row, from a topics.jsonl or style_labels.jsonl file."""
    rows_by_summary = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows_by_summary[row["summary"]] = row
    return rows_by_summary


def join_predictions_with_labels(predictions: list[dict], rows_by_summary: dict,
                                  label_field: str = "topic_label") -> tuple[list[dict], int]:
    """Attach label_field to each prediction by matching reference == summary.

    Returns (joined, n_unmatched) so a broken join key is surfaced instead of silently dropped.
    """
    joined, unmatched = [], 0
    for p in predictions:
        row = rows_by_summary.get(p["reference"])
        if row is None or label_field not in row:
            unmatched += 1
            continue
        joined.append({**p, label_field: row[label_field]})
    return joined, unmatched


def group_by_label(joined: list[dict], label_field: str = "topic_label",
                    min_count: int = DEFAULT_MIN_COUNT,
                    never_skip: frozenset = frozenset()) -> dict[str, dict]:
    """Group joined predictions by label_field.

    Groups with fewer than min_count matched examples are marked skipped instead of given a
    noisy score, except labels in `never_skip` (e.g. NOISE_LABEL for topics) which are always
    reported in full regardless of size — they're informative by themselves, never merged away.
    """
    by_label: dict[str, list[dict]] = {}
    for row in joined:
        by_label.setdefault(row[label_field], []).append(row)

    result = {}
    for label, rows in by_label.items():
        if label not in never_skip and len(rows) < min_count:
            result[label] = {"n": len(rows), "skipped": "n too small"}
        else:
            result[label] = {"n": len(rows), "rows": rows}
    return result


def _failure_rates_for(rows: list[dict], errors_by_key: dict) -> dict | None:
    labelled = [errors_by_key[r["reference"]] for r in rows if r["reference"] in errors_by_key]
    if not labelled:
        return None
    return {ftype: round(sum(ftype in labels for labels in labelled) / len(labelled), 3)
            for ftype in FAILURE_TYPES}


def stratify(predictions: list[dict], rows_by_summary: dict, label_field: str = "topic_label",
             errors_by_key: dict | None = None, min_count: int = DEFAULT_MIN_COUNT,
             never_skip: frozenset = frozenset()) -> dict:
    """Build the per-label report: {n_predictions, n_unmatched, groups: {label: {...}}}."""
    joined, n_unmatched = join_predictions_with_labels(predictions, rows_by_summary, label_field)
    groups = group_by_label(joined, label_field, min_count, never_skip)

    report = {"n_predictions": len(predictions), "n_unmatched": n_unmatched, "label_field": label_field,
              "groups": {}}
    for label, group in groups.items():
        if "skipped" in group:
            report["groups"][label] = group
            continue
        rows = group["rows"]
        entry = {"n": group["n"], "rouge": compute_rouge(rows), "bertscore": compute_bertscore(rows)}
        if errors_by_key:
            rates = _failure_rates_for(rows, errors_by_key)
            if rates:
                entry["failure_rates"] = rates
        report["groups"][label] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description="Break down a predictions file's metrics by topic or style label")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True,
                        help="topics.jsonl (topic_clustering.py) or style_labels.jsonl (style_labels.py)")
    parser.add_argument("--label-field", default="topic_label", choices=("topic_label", "style_label"))
    parser.add_argument("--errors", default="", help="Optional matching *.errors.json for per-group failure rates")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                        help="Minimum matched examples for a group to get a scored entry")
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f]
    for p in predictions:
        p["prediction"] = strip_think(p["prediction"])

    rows_by_summary = load_label_rows(args.labels)
    # The noise bucket only exists for topic_label (HDBSCAN); style_label has no such concept.
    never_skip = frozenset({NOISE_LABEL}) if args.label_field == "topic_label" else frozenset()

    errors_by_key = None
    if args.errors:
        errors_report = json.loads(Path(args.errors).read_text())
        errors_by_key = {e["reference"]: e["labels"] for e in errors_report["examples"]}

    report = stratify(predictions, rows_by_summary, args.label_field, errors_by_key, args.min_count, never_skip)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Stratified report saved to {args.output}")
    if report["n_unmatched"]:
        print(f"WARNING: {report['n_unmatched']}/{report['n_predictions']} predictions had no label match")
    printable = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in report["groups"].items()}
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
