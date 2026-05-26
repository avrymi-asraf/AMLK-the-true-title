"""
Evaluation pipeline step 3: LLM-as-judge scoring via Gemini API.
Reads predictions.jsonl, calls Gemini to score each prediction on faithfulness
and fluency (1-5 scale), and writes a JSON report with per-example scores and averages.

Run: python evaluation/eval_llm.py --predictions outputs/checkpoints/<run>/predictions.jsonl --output outputs/results/<run>-llm.json
Execution environment: local machine with GEMINI_API_KEY set. Implementation in Stage B.
"""
import argparse
import json
import os
from pathlib import Path


def score_with_llm(prediction: dict, client) -> dict:
    """
    Score one {text, reference, prediction} entry using Gemini as judge.
    Returns {faithfulness: int, fluency: int} (both 1-5).
    Stage B: implement prompt + Gemini API call.
    """
    raise NotImplementedError("eval_llm.score_with_llm is implemented in Stage B")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        raise EnvironmentError("GEMINI_API_KEY not set. Run: source .env")

    with open(args.predictions, encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f]

    # Stage B: initialize Gemini client, call score_with_llm for each prediction
    raise NotImplementedError("eval_llm.main is implemented in Stage B")


if __name__ == "__main__":
    main()
