"""
Evaluation pipeline step 1 (local, API-only): generate the Gemini advanced-baseline
summaries for the held-out test split. Uses the same instruction prompt as training
(imported from data.preprocess) so the baseline is comparable to the fine-tuned model.
Writes predictions.jsonl ({text, reference, prediction, model, variant}) that evaluate.py
and error_analysis.py consume; resumes from a partial file so a rate-limit stop is safe.

The fine-tuned and zero-shot-base predictions are NOT produced here — they need a GPU and
are generated on HuggingFace Jobs at the end of the training run (training/train_hf_job.py),
then downloaded from the Hub. This script only does the network-bound Gemini path.

Run: python -m evaluation.predict --variant whole --output outputs/results/gemini-whole.jsonl
Execution environment: any machine with GEMINI_API_KEY set (no GPU, no local model load).
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import datasets as hf_datasets

from data.preprocess import build_prompt, make_variant

GEMINI_MODEL = "gemini-2.5-flash"


def strip_think(text: str) -> str:
    """Drop Qwen3 <think>...</think> reasoning so metrics score the summary, not the reasoning.

    Only well-formed (closed) blocks are removed. A truncated, unclosed <think> means the model
    spent its whole budget reasoning and never wrote a summary — it is left as-is so its low score
    reflects that real failure instead of being hidden. Shared tool, reused by evaluate.py and
    error_analysis.py.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_with_retry(fn, attempts: int = 5):
    """Call fn(), retrying with exponential backoff on transient API errors."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — the API raises a variety of transient errors
            if i == attempts - 1:
                raise
            wait = 2 ** i
            print(f"  API error ({str(e)[:60]}...); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)


def build_gemini_generator():
    """Return a generate(text)->summary fn backed by the Gemini advanced baseline.

    Returns empty string for blocked prompts (PROHIBITED_CONTENT) so resume logic
    (line-count index) stays aligned with the dataset.
    """
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(GEMINI_MODEL)

    def generate(text: str) -> str:
        try:
            return call_with_retry(lambda: model.generate_content(build_prompt(text)).text).strip()
        except Exception as e:
            if "candidates" in str(e).lower() or "blocked" in str(e).lower() or "prohibited" in str(e).lower():
                print(f"  [SKIPPED] Blocked prompt — writing empty prediction.", file=sys.stderr)
                return ""
            raise

    return generate


def main():
    parser = argparse.ArgumentParser(description="Generate Gemini advanced-baseline summaries for the test split")
    parser.add_argument("--variant", choices=("whole", "lead", "body"), default="whole")
    parser.add_argument("--data", default="", help="Test split dir (default: outputs/data/processed/<variant>/test)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0, help="Cap examples for a quick check")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set. Run: source .env", file=sys.stderr)
        sys.exit(1)

    data_dir = Path(args.data or f"outputs/data/processed/{args.variant}/test")
    test_ds = hf_datasets.load_from_disk(str(data_dir))
    if args.limit:
        test_ds = test_ds.select(range(min(args.limit, len(test_ds))))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = sum(1 for _ in open(output_path, encoding="utf-8")) if output_path.exists() else 0
    if done:
        print(f"Resuming: {done} predictions already in {output_path}")

    generate = build_gemini_generator()
    print(f"Generating {len(test_ds) - done} Gemini predictions (variant={args.variant})...")
    with open(output_path, "a", encoding="utf-8") as f:
        for i in range(done, len(test_ds)):
            ex = test_ds[i]
            text = make_variant(ex["text"], args.variant)
            record = {
                "text": text,
                "reference": ex["summary"],
                "prediction": generate(text),
                "model": "gemini",
                "variant": args.variant,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(test_ds)}")

    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
