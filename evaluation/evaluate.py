"""
Evaluation pipeline step 2: score a predictions file with the full metric battery.
Reads a predictions.jsonl produced by predict.py (or the HF Jobs training job) and
computes ROUGE-1/2/L, BERTScore (xlm-roberta-large), and an LLM-as-judge rating of
faithfulness and fluency (1-5). Writes one JSON report to outputs/results/. The judge
supports Gemini (GEMINI_API_KEY) or Hugging Face Inference (HF_TOKEN); use --judge-limit
for a fast subset run.

Run: python -m evaluation.evaluate --predictions outputs/results/predictions-base.jsonl --output outputs/results/zero-shot.report.json
Execution environment: local machine; judge needs GEMINI_API_KEY or HF_TOKEN (skip with --skip-llm).
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

from evaluation.gemini_client import GEMINI_MODEL, GEMINI_TIMEOUT, call_with_retry, strip_think

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
    """Pull a JSON object out of an LLM reply, tolerating fences and malformed values.

    Returns {} when nothing parses so one bad judge reply skips that example instead of
    crashing the whole run.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def sample_for_judge(predictions: list[dict], limit: int, seed: int) -> list[dict]:
    """Return up to `limit` predictions, deterministically sampled."""
    if limit <= 0 or limit >= len(predictions):
        return predictions
    indices = sorted(random.Random(seed).sample(range(len(predictions)), limit))
    return [predictions[i] for i in indices]


def _judge_scores(provider: str, model_id: str, hf_provider: str | None, predictions: list[dict]) -> dict:
    """Score predictions for faithfulness and fluency (1-5)."""
    per_example, faith, flu = [], [], []
    for i, p in enumerate(predictions):
        prompt = JUDGE_PROMPT.format(text=p["text"][:6000], prediction=p["prediction"])
        if provider == "hf":
            from evaluation.hf_client import chat_completion

            raw = chat_completion(prompt, model=model_id, provider=hf_provider or None)
        else:
            import google.generativeai as genai

            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            gemini = genai.GenerativeModel(model_id)
            raw = call_with_retry(lambda: gemini.generate_content(
                prompt, request_options={"timeout": GEMINI_TIMEOUT}).text)

        scores = _parse_json(raw)
        f, l = scores.get("faithfulness"), scores.get("fluency")
        per_example.append({"faithfulness": f, "fluency": l})
        if isinstance(f, (int, float)):
            faith.append(f)
        if isinstance(l, (int, float)):
            flu.append(l)
        if (i + 1) % 25 == 0:
            print(f"  judged {i + 1}/{len(predictions)}")

    return {
        "provider": provider,
        "model": model_id,
        "faithfulness_mean": round(sum(faith) / len(faith), 3) if faith else None,
        "fluency_mean": round(sum(flu) / len(flu), 3) if flu else None,
        "scored": len(faith),
        "per_example": per_example,
    }


def gemini_json(model, prompt: str) -> dict:
    """Call a Gemini model and parse its reply as JSON (used by error_analysis.py)."""
    return _parse_json(call_with_retry(
        lambda: model.generate_content(prompt, request_options={"timeout": GEMINI_TIMEOUT}).text))


def judge_with_llm(predictions: list[dict]) -> dict:
    """Gemini judge over all predictions (legacy entry point for tests)."""
    return _judge_scores("gemini", GEMINI_MODEL, None, predictions)


def main():
    parser = argparse.ArgumentParser(description="Score a predictions file with ROUGE + BERTScore + LLM judge")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bertscore-model", default="xlm-roberta-large")
    parser.add_argument("--skip-llm", action="store_true", help="Skip the LLM judge")
    parser.add_argument("--skip-rouge", action="store_true", help="Skip ROUGE (reuse existing report with --judge-only)")
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip BERTScore")
    parser.add_argument("--judge-only", action="store_true", help="Merge judge scores into an existing report at --output")
    parser.add_argument("--judge-provider", choices=("hf", "gemini"), default="hf")
    parser.add_argument("--judge-model", default="", help="Judge model id (HF repo id or Gemini model name)")
    parser.add_argument("--judge-limit", type=int, default=0, help="Judge a random subset of N examples (0 = all)")
    parser.add_argument("--judge-seed", type=int, default=42)
    parser.add_argument("--hf-inference-provider", default="", help="HF Inference Provider slug (optional)")
    parser.add_argument("--limit", type=int, default=0, help="Cap examples for a quick smoke check")
    args = parser.parse_args()

    output_path = Path(args.output)
    with open(args.predictions, encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f]
    if not predictions:
        print(f"ERROR: {args.predictions} is empty", file=sys.stderr)
        sys.exit(1)
    if args.limit:
        predictions = predictions[:args.limit]
    for p in predictions:
        p["prediction"] = strip_think(p["prediction"])

    if args.judge_only and output_path.exists():
        report = json.loads(output_path.read_text())
        print(f"Loaded existing report from {output_path}")
    else:
        print(f"Scoring {len(predictions)} predictions from {args.predictions}")
        report = {
            "predictions_file": args.predictions,
            "n": len(predictions),
            "model": predictions[0].get("model"),
            "variant": predictions[0].get("variant"),
        }
        if not args.skip_rouge:
            report["rouge"] = compute_rouge(predictions)
        if not args.skip_bertscore:
            report["bertscore"] = compute_bertscore(predictions, args.bertscore_model)

    if not args.skip_llm:
        if args.judge_provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
            print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
            sys.exit(1)
        if args.judge_provider == "hf" and not os.environ.get("HF_TOKEN"):
            print("ERROR: HF_TOKEN not set", file=sys.stderr)
            sys.exit(1)

        if args.judge_provider == "hf":
            from evaluation.hf_client import DEFAULT_JUDGE_MODEL
            judge_model = args.judge_model or DEFAULT_JUDGE_MODEL
        else:
            judge_model = args.judge_model or GEMINI_MODEL

        judge_set = sample_for_judge(predictions, args.judge_limit, args.judge_seed)
        print(f"LLM judge ({args.judge_provider}, {judge_model}) on {len(judge_set)} examples...")
        report["llm_judge"] = _judge_scores(
            args.judge_provider, judge_model, args.hf_inference_provider or None, judge_set
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report saved to {output_path}")
    summary = {k: v for k, v in report.items() if k in ("rouge", "bertscore", "llm_judge")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
