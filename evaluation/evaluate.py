"""
Evaluation pipeline step 2: score a predictions file with the full metric battery.
Reads a predictions.jsonl produced by predict.py and computes ROUGE-1/2/L,
BERTScore (xlm-roberta-large, multilingual/Hebrew-capable), and a Gemini LLM-as-judge
rating of faithfulness and fluency (1-5). Writes one JSON report to outputs/results/.
The same report shape is used for every system, so the fine-tuned model, the zero-shot
baseline, and the Gemini baseline drop straight into one comparison table.

Run: python -m evaluation.evaluate --predictions outputs/results/finetuned-whole.jsonl --output outputs/results/finetuned-whole.report.json
Execution environment: local machine; the judge step needs GEMINI_API_KEY (skip with --skip-llm).
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

from evaluation.predict import GEMINI_MODEL, call_with_retry

JUDGE_PROMPT = """You are a strict evaluator of Hebrew text summaries.
Read the ARTICLE and the candidate SUMMARY, then rate the summary:
- "faithfulness" (1-5): 5 = every claim is supported by the article, 1 = mostly hallucinated.
- "fluency" (1-5): 5 = grammatical, natural Hebrew, 1 = broken or incoherent.
Reply with ONLY a JSON object, no prose: {{"faithfulness": <int>, "fluency": <int>}}

ARTICLE:
{text}

SUMMARY:
{prediction}
"""


class _UnicodeTokenizer:
    """ROUGE tokenizer that keeps Hebrew (rouge_score's default strips non-ASCII)."""

    def tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower(), re.UNICODE)


def compute_rouge(predictions: list[dict]) -> dict:
    """Average ROUGE-1/2/L F-measure over {reference, prediction} pairs."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False, tokenizer=_UnicodeTokenizer()
    )
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for p in predictions:
        scores = scorer.score(p["reference"], p["prediction"])
        for k in totals:
            totals[k] += scores[k].fmeasure
    n = len(predictions)
    return {k: round(v / n, 4) for k, v in totals.items()}


def compute_bertscore(predictions: list[dict], model_id: str = "xlm-roberta-large") -> dict:
    """Average BERTScore precision/recall/F1 using a multilingual model.

    Pinned to CPU: this runs locally and must not load a model onto the user's small GPU.
    """
    from bert_score import score as bertscore

    cands = [p["prediction"] for p in predictions]
    refs = [p["reference"] for p in predictions]
    P, R, F1 = bertscore(cands, refs, model_type=model_id, verbose=False, device="cpu")
    return {
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4),
    }


def _parse_json(raw: str) -> dict:
    """Pull the first JSON object out of an LLM reply (tolerates ```json fences)."""
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


def gemini_json(model, prompt: str) -> dict:
    """Call Gemini and parse its reply as a JSON object."""
    return _parse_json(call_with_retry(lambda: model.generate_content(prompt).text))


def judge_with_llm(predictions: list[dict]) -> dict:
    """Score every prediction for faithfulness and fluency (1-5) via Gemini."""
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(GEMINI_MODEL)

    per_example, faith, flu = [], [], []
    for i, p in enumerate(predictions):
        scores = gemini_json(model, JUDGE_PROMPT.format(text=p["text"][:6000], prediction=p["prediction"]))
        f, l = scores.get("faithfulness"), scores.get("fluency")
        per_example.append({"faithfulness": f, "fluency": l})
        if isinstance(f, (int, float)):
            faith.append(f)
        if isinstance(l, (int, float)):
            flu.append(l)
        if (i + 1) % 25 == 0:
            print(f"  judged {i + 1}/{len(predictions)}")

    return {
        "faithfulness_mean": round(sum(faith) / len(faith), 3) if faith else None,
        "fluency_mean": round(sum(flu) / len(flu), 3) if flu else None,
        "scored": len(faith),
        "per_example": per_example,
    }


def main():
    parser = argparse.ArgumentParser(description="Score a predictions file with ROUGE + BERTScore + LLM judge")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bertscore-model", default="xlm-roberta-large")
    parser.add_argument("--skip-llm", action="store_true", help="Skip the Gemini judge (no API needed)")
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f]
    if not predictions:
        print(f"ERROR: {args.predictions} is empty", file=sys.stderr)
        sys.exit(1)
    print(f"Scoring {len(predictions)} predictions from {args.predictions}")

    report = {
        "predictions_file": args.predictions,
        "n": len(predictions),
        "model": predictions[0].get("model"),
        "variant": predictions[0].get("variant"),
        "rouge": compute_rouge(predictions),
        "bertscore": compute_bertscore(predictions, args.bertscore_model),
    }
    if not args.skip_llm:
        report["llm_judge"] = judge_with_llm(predictions)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report saved to {args.output}")
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("rouge", "bertscore") or
                      (k == "llm_judge" and not args.skip_llm)}, indent=2))


if __name__ == "__main__":
    main()
