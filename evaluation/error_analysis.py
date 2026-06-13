"""
Evaluation pipeline step 3: qualitative error analysis on a prediction sample.
Samples ~50-100 predictions from a predictions.jsonl and asks Gemini to label each
with the failure types reported in the summarization literature (hallucination,
omission, entity/number error, lead copying, fluency problem). Writes per-type rates
plus the labelled examples to outputs/results/, feeding the paper's error-analysis section
and the truncation-probe discussion (e.g. lead-copying rate per model).

Run: python -m evaluation.error_analysis --predictions outputs/results/finetuned-whole.jsonl --output outputs/results/finetuned-whole.errors.json --n 50
Execution environment: local machine with GEMINI_API_KEY set.
"""
import argparse
import json
import os
import random
from pathlib import Path

from evaluation.evaluate import gemini_json
from evaluation.predict import GEMINI_MODEL, strip_think

FAILURE_TYPES = ["hallucination", "omission", "entity_or_number_error", "lead_copying", "fluency_problem"]

LABEL_PROMPT = """You label failure types in Hebrew article summaries.
Read the ARTICLE and the SUMMARY. Choose every label that applies from this exact set:
- "hallucination": states facts not in the article.
- "omission": leaves out the article's main point.
- "entity_or_number_error": wrong name, place, date, or number.
- "lead_copying": just copies the opening sentence instead of summarizing.
- "fluency_problem": ungrammatical or incoherent Hebrew.
If the summary is good, return an empty list.
Reply with ONLY a JSON object: {{"labels": [<zero or more of the labels above>]}}

ARTICLE:
{text}

SUMMARY:
{prediction}
"""


def label_predictions(sample: list[dict], model) -> list[dict]:
    """Attach a 'labels' list (subset of FAILURE_TYPES) to each sampled prediction."""
    labelled = []
    for i, p in enumerate(sample):
        result = gemini_json(model, LABEL_PROMPT.format(text=p["text"][:6000], prediction=p["prediction"]))
        labels = [l for l in result.get("labels", []) if l in FAILURE_TYPES]
        labelled.append({**p, "labels": labels})
        if (i + 1) % 25 == 0:
            print(f"  labelled {i + 1}/{len(sample)}")
    return labelled


def failure_rates(labelled: list[dict]) -> dict:
    """Fraction of the sample exhibiting each failure type."""
    n = len(labelled)
    return {ftype: round(sum(ftype in row["labels"] for row in labelled) / n, 3)
            for ftype in FAILURE_TYPES}


def main():
    parser = argparse.ArgumentParser(description="LLM-labelled failure-type analysis on a prediction sample")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=50, help="Sample size (abstract suggests 50-100)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(GEMINI_MODEL)

    with open(args.predictions, encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f]
    # Label the summary itself, not any leaked Qwen3 <think> reasoning (no-op when absent).
    for p in predictions:
        p["prediction"] = strip_think(p["prediction"])
    random.seed(args.seed)
    sample = random.sample(predictions, min(args.n, len(predictions)))
    print(f"Labelling {len(sample)} of {len(predictions)} predictions from {args.predictions}")

    labelled = label_predictions(sample, model)
    report = {
        "predictions_file": args.predictions,
        "model": predictions[0].get("model"),
        "variant": predictions[0].get("variant"),
        "n_sampled": len(sample),
        "failure_rates": failure_rates(labelled),
        "examples": labelled,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Error analysis saved to {args.output}")
    print(json.dumps(report["failure_rates"], indent=2))


if __name__ == "__main__":
    main()
