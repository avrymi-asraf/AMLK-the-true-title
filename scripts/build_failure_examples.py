"""One-off builder for docs/obsidian/Failure Examples.md — run from repo root."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/failure-examples.md"


def has_hebrew(text: str, n: int = 100) -> bool:
    return any("\u05d0" <= c <= "\u05ea" for c in text[:n])


def pick(examples: list[dict], label_set: set[str]) -> dict | None:
    for e in examples:
        if set(e["labels"]) == label_set:
            return e
    return None


def article_excerpt(text: str, n: int = 600) -> str:
    t = text.replace("\n", " ").strip()
    return t if len(t) <= n else t[:n] + "…"


def pred_block(text: str, limit: int = 0) -> str:
    if limit and len(text) > limit:
        return text[:limit] + "…"
    return text


CLUSTER_DESC = {
    "entity_or_number_error+hallucination+omission": (
        "**Hallucination + wrong entities + missing main points** (20% of finetuned sample). "
        "The model writes a plausible Hebrew media-digest in the right format, but invents newspapers, "
        "people, and claims; it also drops key stories from the article."
    ),
    "hallucination+omission": (
        "**Hallucination + omission** (15%). The summary is on-topic in tone but describes events "
        "that are not in the article and skips the article's actual main points."
    ),
    "entity_or_number_error+omission": (
        "**Wrong entities + omission** (10%). Names/outlets are garbled or wrong; main content missing."
    ),
    "entity_or_number_error+hallucination+lead_copying": (
        "**Hallucination + entity errors + lead bias** (8%). Picks up lead-adjacent names but fabricates the rest."
    ),
    "lead_copying+omission": (
        "**Lead copying + omission** (7%). Echoes opening themes or entities but fails to summarize the full article."
    ),
    "entity_or_number_error+fluency_problem+hallucination": (
        "**Hallucination + entity errors + fluency** (7%). Fluent-looking Hebrew with garbled tokens and invented content."
    ),
    "omission": (
        "**Omission only** (8%). May look reasonable but leaves out the article's central story."
    ),
    "wrong_language": (
        "**Wrong language / non-answer** (97% of base sample, post-hoc tag). "
        "Qwen3 enters an English `<think>` block and never produces a Hebrew summary. "
        "The literature failure taxonomy marks these as \"clean\" because the English reasoning restates the article."
    ),
    "entity_or_number_error+hallucination": (
        "**Hallucination + entity errors in English** (13% of base). "
        "Responds in English with invented facts (e.g. dollar amounts not in the source)."
    ),
    "entity_or_number_error+hallucination+omission_base": (
        "**English summary with wrong facts and missing points** (11% of base)."
    ),
}


def write_example(lines: list[str], e: dict, combo: tuple[str, ...], key: str, pred_limit: int = 0) -> None:
    lines.append(f"**Labels:** `{', '.join(combo) if combo else '(none — wrong_language post-hoc)'}`")
    lines.append("")
    lines.append(CLUSTER_DESC[key])
    lines.append("")
    lines.append("**Article (excerpt):**")
    lines.append("```")
    lines.append(article_excerpt(e["text"]))
    lines.append("```")
    lines.append("")
    lines.append("**Model prediction:**")
    lines.append("```")
    lines.append(pred_block(e["prediction"], pred_limit))
    lines.append("```")
    lines.append("")
    lines.append("**Reference:**")
    lines.append("```")
    lines.append(e["reference"])
    lines.append("```")
    lines.append("")
    lines.append("**What went wrong:**")
    if not combo or key == "wrong_language":
        lines.append("- Model outputs English reasoning inside `<think>` and often hits `max_new_tokens` before any Hebrew summary.")
        lines.append("- Judge taxonomy has no \"wrong language\" label, so this often scores as \"no failure\".")
    else:
        if "hallucination" in combo:
            lines.append("- Prediction invents outlets, people, or events not supported by the article.")
        if "omission" in combo:
            lines.append("- Key stories in the reference (and article) are missing from the prediction.")
        if "entity_or_number_error" in combo:
            lines.append("- Names/numbers are wrong or garbled (e.g. mixed Hebrew/Latin characters).")
        if "lead_copying" in combo:
            lines.append("- Over-relies on lead-adjacent entities without faithful abstraction.")
        if "fluency_problem" in combo:
            lines.append("- Surface fluency breaks down (tokenization artifacts, incoherent phrases).")
        if not has_hebrew(e["prediction"]):
            lines.append("- Responds in English, not Hebrew.")
    lines.append("")


def main() -> None:
    ft = json.loads((ROOT / "outputs/results/finetuned-v3.errors.json").read_text())
    base = json.loads((ROOT / "outputs/results/base-v3.errors.json").read_text())
    ft_ex, base_ex = ft["examples"], base["examples"]

    lines = [
        "# Failure Examples — v3 (Qwen3-2B, whole variant)",
        "",
        "#status/done",
        "",
        "Curated examples of the **biggest failure modes** in the v3 evaluation run "
        "(3-epoch LoRA, anti-degeneration decode). Labels from Gemini `gemini-2.5-flash-lite` "
        "on a 100-example stratified sample (`evaluation/error_analysis.py`). "
        "Full reports: `outputs/results/finetuned-v3.errors.json`, `outputs/results/base-v3.errors.json`.",
        "",
        "See also: [[Prediction Failure Modes]] (v1 repetition-loop analysis), [[Current Results]].",
        "",
        "---",
        "",
        "## Summary rates (n=100 each)",
        "",
        "| Failure type | v3 Finetuned | v3 Base |",
        "|---|---|---|",
    ]
    for k in ["hallucination", "omission", "entity_or_number_error", "lead_copying", "fluency_problem"]:
        lines.append(f"| {k} | {ft['failure_rates'][k]:.0%} | {base['failure_rates'][k]:.0%} |")
    lines += [
        "| wrong_language (post-hoc) | 0% | 97% |",
        "",
        "**Finetuned:** fails on *content* (hallucination/omission) but always answers in Hebrew.",
        "**Base:** fails on *task completion* (English thinking, no Hebrew answer) unless chat template is used.",
        "",
        "---",
        "",
        "## Finetuned v3",
        "",
    ]

    ft_clusters = [
        (("entity_or_number_error", "hallucination", "omission"), "entity_or_number_error+hallucination+omission", "1"),
        (("hallucination", "omission"), "hallucination+omission", "2"),
        (("entity_or_number_error", "omission"), "entity_or_number_error+omission", "3"),
        (("entity_or_number_error", "hallucination", "lead_copying"), "entity_or_number_error+hallucination+lead_copying", "4"),
        (("lead_copying", "omission"), "lead_copying+omission", "5"),
        (("entity_or_number_error", "fluency_problem", "hallucination"), "entity_or_number_error+fluency_problem+hallucination", "6"),
        (("omission",), "omission", "7"),
    ]
    for combo, key, num in ft_clusters:
        e = pick(ft_ex, set(combo))
        if not e:
            continue
        lines.append(f"### {num}. {key.replace('_', ' ')}")
        lines.append("")
        write_example(lines, e, combo, key)
        lines.append("---")
        lines.append("")

    lines += [
        "## Base v3 (zero-shot, raw prompt — known baseline bug)",
        "",
        "> **Note:** A fairer base baseline uses Qwen3 chat template with `enable_thinking=False`. "
        "Regen job `6a48d621` (`--pred-suffix=-v4`) was submitted 2026-07-04.",
        "",
    ]

    e = next(x for x in base_ex if not x["labels"] and not has_hebrew(x["prediction"]))
    lines.append("### 1. wrong_language (non-answer)")
    lines.append("")
    write_example(lines, e, tuple(), "wrong_language", pred_limit=1200)
    lines.append("---")
    lines.append("")

    for combo, key, num in [
        (("entity_or_number_error", "hallucination"), "entity_or_number_error+hallucination", "2"),
        (("entity_or_number_error", "hallucination", "omission"), "entity_or_number_error+hallucination+omission_base", "3"),
    ]:
        e = pick(base_ex, set(combo))
        if not e:
            continue
        lines.append(f"### {num}. {key.replace('_', ' ')}")
        lines.append("")
        write_example(lines, e, combo, key, pred_limit=1200)
        lines.append("---")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
