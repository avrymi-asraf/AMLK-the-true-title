"""
Structural style labeling: classifies each article's summary by its surface FORMAT (single
sentence, multi-sentence, pipe-separated multi-headline digest, or question-style headline) —
a second, independent dimension from evaluation/topic_clustering.py's semantic topic labels.
Motivated by a real pattern in the corpus: ~26% of HeSum summaries are "headline | headline |
headline" digests — a format quirk worth tracking once a model is trained on this data.
Produces the same {summary: label} artifact shape as
topic_clustering.py, so it plugs into evaluation/stratify_by_topic.py's generic --label-field
to break down eval metrics by style instead of (or alongside) topic.

Unlike topic clustering, this is rule-based (regex) and fully local — no embeddings, GPU, or
API calls — so it never needs Databricks; it also has no import-time dependency on `datasets`,
so it runs even in an environment where that import is broken (see AGENTS.md lzma note).

Run: python -m evaluation.style_labels --input outputs/data/raw/combined.jsonl \
    --output outputs/data/raw/style_labels.jsonl
Execution environment: local machine, CPU only, standard library only.
"""
import argparse
import json
import re
from pathlib import Path

PIPE_DIGEST = "pipe_digest"
QUESTION = "question"
MULTI_SENTENCE = "multi_sentence"
SINGLE_SENTENCE = "single_sentence"

_SENTENCE_SPLIT_RE = re.compile(r"[.;]+")


def classify_style(summary: str) -> str:
    """Classify a summary's surface structure (not its topic).

    Checked in this priority order: a pipe-digest or a question can also contain periods
    (each headline segment may end with one), which would otherwise misclassify it as an
    ordinary multi-sentence summary.
    """
    if "|" in summary:
        return PIPE_DIGEST
    if "?" in summary:
        return QUESTION
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(summary) if s.strip()]
    return MULTI_SENTENCE if len(sentences) > 1 else SINGLE_SENTENCE


def label_dataset(records: list[dict]) -> list[dict]:
    """Attach a style_label to each record. Each record needs 'summary' (and 'source').

    Returns rows aligned 1:1 with records (same order), matching topic_clustering.cluster_dataset's
    convention — the two can be zipped together without a join when working from the same records.
    """
    return [
        {"summary": r["summary"], "source": r.get("source"), "style_label": classify_style(r["summary"])}
        for r in records
    ]


def style_summary(rows: list[dict]) -> dict:
    """Count + share per style label, sorted largest-first, for a quick sanity check."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["style_label"]] = counts.get(row["style_label"], 0) + 1
    n = len(rows)
    return {label: {"count": c, "pct": round(c / n, 3)}
            for label, c in sorted(counts.items(), key=lambda kv: -kv[1])}


def plot_style_distribution(summary: dict):
    """Bar chart of classify_style() category counts (style_summary() output) — a quick visual
    complement to the numeric breakdown, e.g. for the Databricks notebook pipeline."""
    import plotly.express as px

    labels = list(summary.keys())
    counts = [summary[label]["count"] for label in labels]
    fig = px.bar(x=labels, y=counts, labels={"x": "style", "y": "count"},
                 title="Summary style distribution")
    return fig


def write_style_labels(rows: list[dict], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Label each summary's structural style (rule-based, local)")
    parser.add_argument("--input", default="outputs/data/raw/combined.jsonl")
    parser.add_argument("--output", default="outputs/data/raw/style_labels.jsonl")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    rows = label_dataset(records)
    write_style_labels(rows, args.output)

    print(f"Labelled {len(rows)} summaries -> {args.output}")
    print(json.dumps(style_summary(rows), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
