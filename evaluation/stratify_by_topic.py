"""
Evaluation pipeline add-on: breaks down an existing predictions file's ROUGE/BERTScore/failure
rates by topic. Joins predictions to the topics.jsonl artifact produced by the Databricks
notebook (notebooks/cluster_topics_databricks.py, via evaluation/topic_clustering.py) on exact
summary-text match — every prediction's `reference` field is copied verbatim from the original
summary (see evaluation/infer.py, training/train_hf_job.py), so it's a stable join key back to
the whole-corpus topic assignments without touching data/preprocess.py or the Arrow splits.
Reuses evaluate.py's compute_rouge/compute_bertscore per topic group; a matching *.errors.json
(from error_analysis.py) is folded in for per-topic failure-type rates if present. See
docs/superpowers/specs/2026-07-04-topic-clustering-design.md for the full design.

Run: python -m evaluation.stratify_by_topic --predictions outputs/results/predictions-finetuned.jsonl \
    --topics outputs/data/raw/topics.jsonl --errors outputs/results/finetuned-v3.errors.json \
    --output outputs/results/finetuned-by-topic.json
Execution environment: local machine, CPU only (same as evaluate.py) — no GPU/Databricks needed
for this step; topics.jsonl must already be downloaded from the clustering notebook.
"""
import argparse
import json
from pathlib import Path

from evaluation.error_analysis import FAILURE_TYPES
from evaluation.evaluate import compute_bertscore, compute_rouge
from evaluation.gemini_client import strip_think
from evaluation.topic_clustering import NOISE_LABEL

DEFAULT_MIN_COUNT = 10


def load_topics(path: str) -> dict[str, dict]:
    """Map summary text -> {cluster_id, topic_label, keywords} from a topics.jsonl file."""
    topics_by_summary = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            topics_by_summary[row["summary"]] = row
    return topics_by_summary


def join_predictions_with_topics(predictions: list[dict], topics_by_summary: dict) -> tuple[list[dict], int]:
    """Attach topic_label to each prediction by matching reference == summary.

    Returns (joined, n_unmatched) so a broken join key is surfaced instead of silently dropped.
    """
    joined, unmatched = [], 0
    for p in predictions:
        topic = topics_by_summary.get(p["reference"])
        if topic is None:
            unmatched += 1
            continue
        joined.append({**p, "topic_label": topic["topic_label"]})
    return joined, unmatched


def group_by_topic(joined: list[dict], min_count: int = DEFAULT_MIN_COUNT) -> dict[str, dict]:
    """Group joined predictions by topic_label.

    Real topics with fewer than min_count matched examples are marked skipped instead of given
    a noisy per-topic score. The noise bucket is always kept separate and is never skipped or
    merged into a real topic, regardless of size — it's informative by itself.
    """
    by_topic: dict[str, list[dict]] = {}
    for row in joined:
        by_topic.setdefault(row["topic_label"], []).append(row)

    result = {}
    for topic_label, rows in by_topic.items():
        if topic_label != NOISE_LABEL and len(rows) < min_count:
            result[topic_label] = {"n": len(rows), "skipped": "n too small"}
        else:
            result[topic_label] = {"n": len(rows), "rows": rows}
    return result


def _failure_rates_for(rows: list[dict], errors_by_key: dict) -> dict | None:
    labelled = [errors_by_key[r["reference"]] for r in rows if r["reference"] in errors_by_key]
    if not labelled:
        return None
    return {ftype: round(sum(ftype in labels for labels in labelled) / len(labelled), 3)
            for ftype in FAILURE_TYPES}


def stratify(predictions: list[dict], topics_by_summary: dict, errors_by_key: dict | None = None,
             min_count: int = DEFAULT_MIN_COUNT) -> dict:
    """Build the per-topic report: {n_predictions, n_unmatched, topics: {label: {...}}}."""
    joined, n_unmatched = join_predictions_with_topics(predictions, topics_by_summary)
    groups = group_by_topic(joined, min_count)

    report = {"n_predictions": len(predictions), "n_unmatched": n_unmatched, "topics": {}}
    for topic_label, group in groups.items():
        if "skipped" in group:
            report["topics"][topic_label] = group
            continue
        rows = group["rows"]
        entry = {"n": group["n"], "rouge": compute_rouge(rows), "bertscore": compute_bertscore(rows)}
        if errors_by_key:
            rates = _failure_rates_for(rows, errors_by_key)
            if rates:
                entry["failure_rates"] = rates
        report["topics"][topic_label] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description="Break down a predictions file's metrics by topic")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--topics", required=True, help="topics.jsonl from cluster_topics_databricks.py")
    parser.add_argument("--errors", default="", help="Optional matching *.errors.json for per-topic failure rates")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-count", type=int, default=DEFAULT_MIN_COUNT,
                        help="Minimum matched examples for a real topic to get a scored entry")
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f]
    for p in predictions:
        p["prediction"] = strip_think(p["prediction"])

    topics_by_summary = load_topics(args.topics)

    errors_by_key = None
    if args.errors:
        errors_report = json.loads(Path(args.errors).read_text())
        errors_by_key = {e["reference"]: e["labels"] for e in errors_report["examples"]}

    report = stratify(predictions, topics_by_summary, errors_by_key, args.min_count)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Stratified report saved to {args.output}")
    if report["n_unmatched"]:
        print(f"WARNING: {report['n_unmatched']}/{report['n_predictions']} predictions had no topic match")
    printable = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in report["topics"].items()}
    print(json.dumps(printable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
