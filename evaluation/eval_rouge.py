"""
Evaluation pipeline step 1: ROUGE-1/2/L scoring.
Reads predictions.jsonl from a training checkpoint directory, computes ROUGE
scores against human reference summaries, and writes a JSON report.

Run: python evaluation/eval_rouge.py --predictions outputs/checkpoints/<run>/predictions.jsonl --output outputs/results/<run>-rouge.json
Execution environment: local development machine. Implementation in Stage B.
"""
import argparse
import json
from pathlib import Path


def compute_rouge(predictions: list[dict]) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L for a list of {prediction, reference} dicts.
    Returns {rouge1: {precision, recall, fmeasure}, rouge2: {...}, rougeL: {...}}.
    Stage B: implement using rouge_score.rouge_scorer.RougeScorer.
    """
    raise NotImplementedError("eval_rouge.compute_rouge is implemented in Stage B")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f]

    scores = compute_rouge(predictions)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"ROUGE scores saved to {args.output}")


if __name__ == "__main__":
    main()
