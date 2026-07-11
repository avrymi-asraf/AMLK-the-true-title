"""
Evaluation pipeline, final step: turn the per-system JSON reports into the D1 comparison
tables. Downloads the reports/*.report.json + reports/*.errors.json that eval_hf_job.py pushed
to the model repo, and the raw base predictions (for the <think>/language leakage finding), and
writes one markdown file with a quality table, a failure-rate table, and behavioural notes —
the numbers that go straight into the presentation. A tool: import build_tables() in a notebook,
or run it from the CLI.

Execution environment: local machine (tiny JSON downloads, no GPU, no API).
"""
import argparse
import json
import os
from pathlib import Path

SYSTEMS = ["finetuned", "base", "gemini"]
LABELS = {
    "finetuned": "DictaLM-2.0-Instruct LoRA (ours)",
    "base": "DictaLM-2.0-Instruct zero-shot",
    "gemini": "Gemini 2.5 Flash (baseline)",
}
FAILURE_TYPES = ["hallucination", "omission", "entity_or_number_error", "lead_copying", "fluency_problem"]


def is_hebrew(s: str) -> bool:
    return any("֐" <= c <= "׿" for c in s)


def base_leakage(base_predictions_path: str) -> dict:
    """Rates of the zero-shot base's raw-output problems, before <think> stripping."""
    rows = [json.loads(l) for l in open(base_predictions_path, encoding="utf-8")]
    raw = [r["prediction"] for r in rows]
    n = len(raw)
    return {
        "n": n,
        "non_hebrew": round(sum(not is_hebrew(x) for x in raw) / n, 3),
        "has_think": round(sum("<think>" in x for x in raw) / n, 3),
        "no_summary": round(sum("<think>" in x and "</think>" not in x for x in raw) / n, 3),
    }


def build_tables(reports: dict, errors: dict, leakage: dict | None) -> str:
    """Render the quality table, failure-rate table, and notes from the loaded reports."""
    out = ["# D1 — Hebrew News Summarization Results (variant: whole, n=1000 test)\n"]

    out.append("## Quality metrics  (↑ = higher is better — all six)\n")
    out.append("| System | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ | BERTScore-F1 ↑ | Faithfulness 1-5 ↑ | Fluency 1-5 ↑ |")
    out.append("|---|---|---|---|---|---|---|")
    for s in SYSTEMS:
        r, j = reports[s], reports[s].get("llm_judge", {})
        out.append(f"| {LABELS[s]} | {r['rouge']['rouge1']} | {r['rouge']['rouge2']} | "
                   f"{r['rouge']['rougeL']} | {r['bertscore']['f1']} | "
                   f"{j.get('faithfulness_mean', '—')} | {j.get('fluency_mean', '—')} |")

    out.append("\n## Failure-type rates (Gemini-labelled sample)  (↓ = lower is better — all five)\n")
    out.append("| System | Hallucination | Omission | Entity/number | Lead-copying | Fluency |")
    out.append("|---|---|---|---|---|---|")
    for s in SYSTEMS:
        fr = errors[s]["failure_rates"]
        out.append(f"| {LABELS[s]} | " + " | ".join(f"{fr[t]:.2f}" for t in FAILURE_TYPES) + " |")

    out.append("\n## Notes\n")
    if leakage:
        out.append(f"- **Zero-shot base raw-output failures** (before `<think>` stripping, n={leakage['n']}): "
                   f"{leakage['non_hebrew']:.0%} not in Hebrew, {leakage['has_think']:.0%} emit a `<think>` "
                   f"reasoning block, {leakage['no_summary']:.0%} run out of budget mid-reasoning and produce "
                   f"**no summary at all**. Metrics score the text after `</think>`; the un-closed cases are "
                   f"left as-is, so their low scores reflect a real failure to summarize.")
    out.append("- **Self-preference caveat:** the LLM-judge and the advanced baseline are the same "
               "model family (Gemini 2.5 Flash); the judge may favour the baseline's own style.")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Assemble the D1 comparison tables from the pushed reports")
    parser.add_argument("--repo", default="avreymi/amlk-dictalm2-instruct-sft")
    parser.add_argument("--variant", default="whole")
    parser.add_argument("--output", default="outputs/results/d1-tables.md")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")

    def fetch(name):
        return hf_hub_download(args.repo, name, repo_type="model", token=token)

    reports = {s: json.load(open(fetch(f"reports/{s}-{args.variant}.report.json"))) for s in SYSTEMS}
    errors = {s: json.load(open(fetch(f"reports/{s}-{args.variant}.errors.json"))) for s in SYSTEMS}
    leakage = base_leakage(fetch("predictions-base.jsonl"))

    md = build_tables(reports, errors, leakage)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(md, encoding="utf-8")
    print(md)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
