"""
Evaluation pipeline step 2: BERTScore with a multilingual Hebrew-capable model.
Reads predictions.jsonl, computes BERTScore precision/recall/F1 using
xlm-roberta-large, and writes a JSON report.

Run: python evaluation/eval_bertscore.py --predictions outputs/checkpoints/<run>/predictions.jsonl --output outputs/results/<run>-bertscore.json
Execution environment: local development machine. Implementation in Stage B.
"""
import argparse
import json
from pathlib import Path


def compute_bertscore(predictions: list[dict], model_id: str = "xlm-roberta-large") -> dict:
    """
    Compute BERTScore for a list of {prediction, reference} dicts using model_id.
    Returns {precision: float, recall: float, f1: float} (averages over all examples).
    Stage B: implement using bert_score.score().
    """
    raise NotImplementedError("eval_bertscore.compute_bertscore is implemented in Stage B")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="xlm-roberta-large")
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f]

    scores = compute_bertscore(predictions, model_id=args.model)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"BERTScore results saved to {args.output}")


if __name__ == "__main__":
    main()
