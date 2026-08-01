"""
E4 metrics + HTML: base vs raw-SFT vs curated-SFT (ROUGE/BERTScore + LLM judge).

Pipeline role: after E4 prediction JSONLs exist, join them with zero-shot base
predictions on the same shared curated test articles, score automatic metrics,
and run Gemini pointwise faithfulness/fluency (T=0) on each arm. Writes a metrics
JSON report and a self-contained HTML viewer (F8-style side-by-side).

Run (from repo root, GEMINI_API_KEY for judge):
  python -m scripts.e4_auto_metrics_html \\
      --raw outputs/results/e4/predictions-e4-raw.jsonl \\
      --curated outputs/results/e4/predictions-e4-curated.jsonl \\
      --base outputs/results/dictalm2-sft-full/predictions-base.jsonl \\
      --output-dir outputs/results/e4

Execution environment: local CPU; BERTScore on CPU; Gemini API for judge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from evaluation.evaluate import _judge_scores, compute_bertscore, compute_rouge
from evaluation.gemini_client import GEMINI_MODEL, strip_think
from rouge_score import rouge_scorer


def load_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["prediction"] = strip_think(row.get("prediction") or "")
            row["reference"] = row.get("reference") or row.get("summary") or ""
            rows.append(row)
    return rows


def index_by_text(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows:
        text = r.get("text") or ""
        if text and text not in out:
            out[text] = r
    return out


def pair_three(
    base_rows: list[dict],
    raw_rows: list[dict],
    cur_rows: list[dict],
) -> list[dict]:
    """Join base / raw / curated on article text; drop unpaired rows."""
    base_ix = index_by_text(base_rows)
    raw_ix = index_by_text(raw_rows)
    cur_ix = index_by_text(cur_rows)
    shared = sorted(set(base_ix) & set(raw_ix) & set(cur_ix))
    pairs: list[dict] = []
    for i, t in enumerate(shared, start=1):
        ref = (
            cur_ix[t].get("reference")
            or cur_ix[t].get("summary")
            or base_ix[t].get("reference")
            or ""
        )
        pairs.append(
            {
                "idx": i,
                "text": t,
                "reference": ref,
                "base": base_ix[t]["prediction"],
                "raw": raw_ix[t]["prediction"],
                "curated": cur_ix[t]["prediction"],
            }
        )
    return pairs


def _scorer(normalize: bool = False) -> rouge_scorer.RougeScorer:
    from evaluation.evaluate import _UnicodeTokenizer

    return rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
        tokenizer=_UnicodeTokenizer(normalize),
    )


def per_example_rouge(pairs: list[dict], normalize: bool = False) -> list[dict]:
    """Attach ROUGE F1 for each arm vs the shared curated reference."""
    scorer = _scorer(normalize)
    suffix = "_norm" if normalize else ""
    out: list[dict] = []
    for p in pairs:
        row = dict(p)
        for arm, key in (("base", "base"), ("raw", "raw"), ("curated", "curated")):
            scores = scorer.score(p["reference"], p[key])
            row[f"{arm}_rouge1{suffix}"] = round(scores["rouge1"].fmeasure, 4)
            row[f"{arm}_rouge2{suffix}"] = round(scores["rouge2"].fmeasure, 4)
            row[f"{arm}_rougeL{suffix}"] = round(scores["rougeL"].fmeasure, 4)
        out.append(row)
    return out


def arm_as_predictions(pairs: list[dict], arm: str) -> list[dict]:
    return [
        {"reference": p["reference"], "prediction": p[arm], "text": p["text"]}
        for p in pairs
    ]


def length_stats(texts: list[str]) -> dict:
    """Word counts via the same Unicode tokenizer used for ROUGE."""
    counts = [len(re.findall(r"\w+", t.lower(), re.UNICODE)) for t in texts]
    if not counts:
        return {"mean_words": 0.0, "median_words": 0.0}
    counts_sorted = sorted(counts)
    n = len(counts_sorted)
    mid = n // 2
    median = (
        counts_sorted[mid]
        if n % 2
        else (counts_sorted[mid - 1] + counts_sorted[mid]) / 2
    )
    return {
        "mean_words": round(sum(counts) / n, 2),
        "median_words": float(median),
    }


def score_arms(pairs: list[dict], skip_bertscore: bool) -> dict:
    systems = {}
    for arm in ("base", "raw", "curated"):
        preds = arm_as_predictions(pairs, arm)
        entry = {
            "n": len(preds),
            "rouge": compute_rouge(preds),
            "rouge_normalized": compute_rouge(preds, normalize=True),
            "length": length_stats([p["prediction"] for p in preds]),
        }
        if not skip_bertscore:
            entry["bertscore"] = compute_bertscore(preds)
        systems[arm] = entry
    return systems


def delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return round(a - b, 4)


def _cache_key(text: str, prediction: str) -> str:
    payload = f"{text}\n---\n{prediction}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def load_judge_cache(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_judge_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def judge_arm_cached(
    pairs: list[dict],
    arm: str,
    cache: dict[str, dict],
    model: str = GEMINI_MODEL,
) -> list[dict]:
    """Pointwise faith/flu for one arm; resume from cache; fill cache in place."""
    prepared: list[dict] = []
    keys: list[str] = []
    need_idx: list[int] = []
    results: list[dict | None] = [None] * len(pairs)

    for i, p in enumerate(pairs):
        key = _cache_key(p["text"], p[arm])
        keys.append(key)
        hit = cache.get(key)
        if (
            isinstance(hit, dict)
            and isinstance(hit.get("faithfulness"), (int, float))
            and isinstance(hit.get("fluency"), (int, float))
        ):
            results[i] = {
                "faithfulness": hit["faithfulness"],
                "fluency": hit["fluency"],
            }
        else:
            need_idx.append(i)
            prepared.append({"text": p["text"], "prediction": p[arm]})

    n_hit = len(pairs) - len(prepared)
    print(
        f"  judge {arm}: cache hit {n_hit}/{len(pairs)}, "
        f"API calls {len(prepared)} ({model})",
        file=sys.stderr,
    )
    if prepared:
        scored = _judge_scores("gemini", model, None, prepared)
        pe = scored.get("per_example") or []
        for j, i in enumerate(need_idx):
            row = pe[j] if j < len(pe) else {}
            faith = row.get("faithfulness")
            flu = row.get("fluency")
            results[i] = {"faithfulness": faith, "fluency": flu}
            if isinstance(faith, (int, float)) and isinstance(flu, (int, float)):
                cache[keys[i]] = {"faithfulness": faith, "fluency": flu}

    return [
        r if r is not None else {"faithfulness": None, "fluency": None}
        for r in results
    ]


def attach_judge_scores(pairs: list[dict], cache: dict[str, dict]) -> list[dict]:
    """Run Gemini pointwise judge for base/raw/curated; attach per-example scores."""
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is required for LLM judge "
            "(or pass --skip-judge)"
        )
    out = [dict(p) for p in pairs]
    for arm in ("base", "raw", "curated"):
        scores = judge_arm_cached(pairs, arm, cache)
        for i, s in enumerate(scores):
            out[i][f"{arm}_faithfulness"] = s.get("faithfulness")
            out[i][f"{arm}_fluency"] = s.get("fluency")
    return out


def judge_arm_means(examples: list[dict], arm: str) -> dict:
    faith = [
        e[f"{arm}_faithfulness"]
        for e in examples
        if isinstance(e.get(f"{arm}_faithfulness"), (int, float))
    ]
    flu = [
        e[f"{arm}_fluency"]
        for e in examples
        if isinstance(e.get(f"{arm}_fluency"), (int, float))
    ]
    return {
        "faithfulness_mean": round(sum(faith) / len(faith), 3) if faith else None,
        "fluency_mean": round(sum(flu) / len(flu), 3) if flu else None,
        "n_scored": len(faith),
    }


def best_arm_by_judge(example: dict, metric: str = "faithfulness") -> str:
    """Highest 1–5 score; ties when two or more share the max (or all missing)."""
    scores = {
        "base": example.get(f"base_{metric}"),
        "raw": example.get(f"raw_{metric}"),
        "curated": example.get(f"curated_{metric}"),
    }
    numeric = {k: v for k, v in scores.items() if isinstance(v, (int, float))}
    if not numeric:
        return "tie"
    top = max(numeric.values())
    winners = [k for k, v in numeric.items() if v == top]
    if len(winners) > 1:
        return "tie"
    return winners[0]


def build_summary(systems: dict, has_judge: bool = False) -> dict:
    """Aggregate table + deltas vs base for the HTML header."""
    rows = []
    base = systems["base"]
    for arm in ("base", "raw", "curated"):
        s = systems[arm]
        row = {
            "system": arm,
            "n": s["n"],
            "rouge1": s["rouge"]["rouge1"],
            "rouge2": s["rouge"]["rouge2"],
            "rougeL": s["rouge"]["rougeL"],
            "rouge1_norm": s["rouge_normalized"]["rouge1"],
            "rouge2_norm": s["rouge_normalized"]["rouge2"],
            "rougeL_norm": s["rouge_normalized"]["rougeL"],
            "mean_words": s["length"]["mean_words"],
            "median_words": s["length"]["median_words"],
        }
        if "bertscore" in s:
            row["bertscore_f1"] = s["bertscore"]["f1"]
            row["bertscore_p"] = s["bertscore"]["precision"]
            row["bertscore_r"] = s["bertscore"]["recall"]
        if has_judge and "judge" in s:
            row["faithfulness_mean"] = s["judge"].get("faithfulness_mean")
            row["fluency_mean"] = s["judge"].get("fluency_mean")
            row["judge_n"] = s["judge"].get("n_scored")
        if arm != "base":
            row["delta_rougeL_vs_base"] = delta(row["rougeL"], base["rouge"]["rougeL"])
            row["delta_rouge1_vs_base"] = delta(row["rouge1"], base["rouge"]["rouge1"])
            if "bertscore" in s and "bertscore" in base:
                row["delta_bertscore_f1_vs_base"] = delta(
                    s["bertscore"]["f1"], base["bertscore"]["f1"]
                )
            if has_judge and "judge" in s and "judge" in base:
                row["delta_faithfulness_vs_base"] = delta(
                    s["judge"].get("faithfulness_mean"),
                    base["judge"].get("faithfulness_mean"),
                )
                row["delta_fluency_vs_base"] = delta(
                    s["judge"].get("fluency_mean"),
                    base["judge"].get("fluency_mean"),
                )
        rows.append(row)
    note = "curated HeSum headlines (shared test)"
    if has_judge:
        note += f"; Gemini {GEMINI_MODEL} faith/flu T=0"
    return {"systems": rows, "reference_note": note, "has_judge": has_judge}


def best_arm_by_metric(example: dict, metric: str = "rougeL") -> str:
    scores = {
        "base": example[f"base_{metric}"],
        "raw": example[f"raw_{metric}"],
        "curated": example[f"curated_{metric}"],
    }
    top = max(scores.values())
    winners = [k for k, v in scores.items() if v == top]
    if len(winners) > 1:
        return "tie"
    return winners[0]


def render_html(
    summary: dict,
    examples: list[dict],
    has_bertscore: bool,
    has_judge: bool = False,
) -> str:
    example_payload = []
    for e in examples:
        row = {
            "idx": e["idx"],
            "text": e["text"],
            "reference": e["reference"],
            "base": e["base"],
            "raw": e["raw"],
            "curated": e["curated"],
            "base_rougeL": e["base_rougeL"],
            "raw_rougeL": e["raw_rougeL"],
            "curated_rougeL": e["curated_rougeL"],
            "base_rouge1": e["base_rouge1"],
            "raw_rouge1": e["raw_rouge1"],
            "curated_rouge1": e["curated_rouge1"],
            "winner_rougeL": best_arm_by_metric(e, "rougeL"),
        }
        if has_judge:
            for arm in ("base", "raw", "curated"):
                row[f"{arm}_faithfulness"] = e.get(f"{arm}_faithfulness")
                row[f"{arm}_fluency"] = e.get(f"{arm}_fluency")
            row["winner_faithfulness"] = best_arm_by_judge(e, "faithfulness")
            row["winner_fluency"] = best_arm_by_judge(e, "fluency")
        example_payload.append(row)

    payload = {
        "summary": summary,
        "examples": example_payload,
        "has_bertscore": has_bertscore,
        "has_judge": has_judge,
        "judge_model": GEMINI_MODEL if has_judge else None,
    }
    data_json = json.dumps(payload, ensure_ascii=False)
    # Escape for embedding in <script type="application/json">
    data_json = data_json.replace("</", "<\\/")

    systems = summary["systems"]
    by_name = {s["system"]: s for s in systems}

    def cell(arm: str, key: str, fmt: str = ".3f") -> str:
        v = by_name[arm].get(key)
        if v is None:
            return "—"
        if isinstance(v, float):
            return format(v, fmt)
        return str(v)

    def dcell(arm: str, key: str) -> str:
        v = by_name[arm].get(key)
        if v is None:
            return "—"
        sign = "+" if v > 0 else ""
        return f"{sign}{v:.3f}"

    bert_header = "<th>BERTScore F1</th><th>Δ F1 vs base</th>" if has_bertscore else ""
    judge_header = (
        "<th>Faith (1–5)</th><th>Δ Faith vs base</th>"
        "<th>Fluency (1–5)</th><th>Δ Flu vs base</th>"
        if has_judge
        else ""
    )
    bert_cells = {}
    judge_cells = {}
    for arm in ("base", "raw", "curated"):
        if has_bertscore:
            if arm == "base":
                bert_cells[arm] = f"<td>{cell(arm, 'bertscore_f1')}</td><td>—</td>"
            else:
                bert_cells[arm] = (
                    f"<td>{cell(arm, 'bertscore_f1')}</td>"
                    f"<td>{dcell(arm, 'delta_bertscore_f1_vs_base')}</td>"
                )
        else:
            bert_cells[arm] = ""
        if has_judge:
            if arm == "base":
                judge_cells[arm] = (
                    f"<td>{cell(arm, 'faithfulness_mean')}</td><td>—</td>"
                    f"<td>{cell(arm, 'fluency_mean')}</td><td>—</td>"
                )
            else:
                judge_cells[arm] = (
                    f"<td>{cell(arm, 'faithfulness_mean')}</td>"
                    f"<td>{dcell(arm, 'delta_faithfulness_vs_base')}</td>"
                    f"<td>{cell(arm, 'fluency_mean')}</td>"
                    f"<td>{dcell(arm, 'delta_fluency_vs_base')}</td>"
                )
        else:
            judge_cells[arm] = ""

    table_rows = []
    labels = {"base": "Base (zero-shot)", "raw": "Raw SFT", "curated": "Curated SFT"}
    for arm in ("base", "raw", "curated"):
        d_r1 = "—" if arm == "base" else dcell(arm, "delta_rouge1_vs_base")
        d_rl = "—" if arm == "base" else dcell(arm, "delta_rougeL_vs_base")
        table_rows.append(
            f"<tr class='arm-{arm}'>"
            f"<td><strong>{labels[arm]}</strong></td>"
            f"<td>{by_name[arm]['n']}</td>"
            f"<td>{cell(arm, 'rouge1')}</td>"
            f"<td>{cell(arm, 'rouge2')}</td>"
            f"<td>{cell(arm, 'rougeL')}</td>"
            f"<td>{d_r1}</td>"
            f"<td>{d_rl}</td>"
            f"{bert_cells[arm]}"
            f"{judge_cells[arm]}"
            f"<td>{cell(arm, 'mean_words', '.1f')}</td>"
            f"</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="he">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>E4 Metrics — Base vs Raw vs Curated{(" + Judge" if has_judge else "")}</title>
<style>
  :root {{
    --bg: #0f1419;
    --surface: #1a2332;
    --surface2: #243044;
    --border: #2d3a4f;
    --text: #e7ecf3;
    --muted: #8b9bb4;
    --base: #818cf8;
    --base-bg: rgba(129, 140, 248, 0.12);
    --raw: #f59e0b;
    --raw-bg: rgba(245, 158, 11, 0.12);
    --curated: #34d399;
    --curated-bg: rgba(52, 211, 153, 0.12);
    --ref: #a78bfa;
    --ref-bg: rgba(167, 139, 250, 0.1);
    --tie: #94a3b8;
    --accent: #60a5fa;
    --up: #34d399;
    --down: #f87171;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.55;
  }}
  header {{
    position: sticky; top: 0; z-index: 20;
    background: rgba(15, 20, 25, 0.94);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border);
    padding: 0.85rem 1.25rem 1rem;
  }}
  header h1 {{
    margin: 0 0 0.25rem;
    font-size: 1.25rem;
    font-weight: 650;
    letter-spacing: -0.02em;
  }}
  header .sub {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 0.75rem; }}
  .metrics-wrap {{
    overflow-x: auto;
    margin-bottom: 0.75rem;
  }}
  table.metrics {{
    border-collapse: collapse;
    width: 100%;
    min-width: 720px;
    font-size: 0.82rem;
  }}
  table.metrics th, table.metrics td {{
    border: 1px solid var(--border);
    padding: 0.4rem 0.55rem;
    text-align: center;
  }}
  table.metrics th {{
    background: var(--surface2);
    color: var(--muted);
    font-weight: 600;
  }}
  table.metrics td:first-child, table.metrics th:first-child {{
    text-align: start;
  }}
  tr.arm-base td:first-child {{ color: var(--base); }}
  tr.arm-raw td:first-child {{ color: var(--raw); }}
  tr.arm-curated td:first-child {{ color: var(--curated); }}
  .controls {{
    display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;
  }}
  .controls input[type="search"] {{
    flex: 1 1 220px; min-width: 180px;
    background: var(--surface); border: 1px solid var(--border);
    color: var(--text); border-radius: 8px;
    padding: 0.5rem 0.75rem; font-size: 0.9rem;
  }}
  .controls input[type="search"]:focus {{
    outline: 2px solid var(--accent); outline-offset: 1px;
  }}
  .filters {{ display: flex; flex-wrap: wrap; gap: 0.35rem; }}
  .filters button {{
    background: var(--surface); border: 1px solid var(--border);
    color: var(--muted); border-radius: 999px;
    padding: 0.35rem 0.75rem; font-size: 0.8rem; cursor: pointer;
  }}
  .filters button:hover {{ border-color: var(--accent); color: var(--text); }}
  .filters button.active {{
    background: rgba(96, 165, 250, 0.15);
    border-color: var(--accent); color: var(--accent);
  }}
  .meta-count {{ color: var(--muted); font-size: 0.8rem; margin-inline-start: auto; }}
  main {{ max-width: 1200px; margin: 0 auto; padding: 1rem 1.25rem 3rem; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; margin-bottom: 1rem; overflow: hidden;
  }}
  .card-head {{
    display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem;
    padding: 0.75rem 1rem; border-bottom: 1px solid var(--border);
    background: var(--surface2);
  }}
  .card-head .num {{
    font-weight: 700; font-variant-numeric: tabular-nums;
    color: var(--muted); min-width: 2.5rem;
  }}
  .badge {{
    display: inline-flex; align-items: center; gap: 0.3rem;
    border-radius: 999px; padding: 0.2rem 0.65rem;
    font-size: 0.75rem; font-weight: 650;
    text-transform: uppercase; letter-spacing: 0.03em;
  }}
  .badge.base {{ background: var(--base-bg); color: var(--base); }}
  .badge.raw {{ background: var(--raw-bg); color: var(--raw); }}
  .badge.curated {{ background: var(--curated-bg); color: var(--curated); }}
  .badge.tie {{ background: rgba(148,163,184,0.12); color: var(--tie); }}
  .score-pills {{ display: flex; flex-wrap: wrap; gap: 0.35rem; margin-inline-start: auto; }}
  .pill {{
    font-size: 0.72rem; font-variant-numeric: tabular-nums;
    border: 1px solid var(--border); border-radius: 6px;
    padding: 0.15rem 0.45rem; color: var(--muted);
  }}
  .pill.best {{ border-color: var(--accent); color: var(--accent); }}
  .card-body {{ padding: 0.9rem 1rem 1.1rem; }}
  .block {{ margin-bottom: 0.85rem; }}
  .label {{
    font-size: 0.72rem; font-weight: 650; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--muted); margin-bottom: 0.3rem;
  }}
  .box {{
    background: var(--bg); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.65rem 0.8rem; font-size: 0.95rem;
  }}
  .box.article {{
    max-height: 5.5em; overflow: hidden; color: var(--muted); font-size: 0.88rem;
  }}
  .box.article.expanded {{ max-height: none; color: var(--text); }}
  .box.ref {{ background: var(--ref-bg); border-color: rgba(167,139,250,0.35); }}
  .box.winner-ring {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
  .rtl {{ direction: rtl; text-align: right; unicode-bidi: plaintext; }}
  .grid3 {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.65rem;
  }}
  @media (max-width: 900px) {{
    .grid3 {{ grid-template-columns: 1fr; }}
  }}
  .toggle-article {{
    margin-top: 0.35rem; background: transparent; border: none;
    color: var(--accent); cursor: pointer; font-size: 0.8rem; padding: 0;
  }}
  .note {{
    color: var(--muted); font-size: 0.8rem; margin: 0.5rem 0 0;
  }}
  .delta-up {{ color: var(--up); }}
  .delta-down {{ color: var(--down); }}
</style>
</head>
<body>
<header>
  <h1>E4 metrics — Base vs Raw SFT vs Curated SFT</h1>
  <div class="sub">
    Shared curated test · ROUGE
    {" + BERTScore" if has_bertscore else ""}
    {" + Gemini LLM-as-judge (faithfulness / fluency, T=0)" if has_judge else " · no LLM judge"}
    · reference = curated HeSum headlines
  </div>
  <div class="metrics-wrap">
    <table class="metrics">
      <thead>
        <tr>
          <th>System</th>
          <th>n</th>
          <th>R-1</th>
          <th>R-2</th>
          <th>R-L</th>
          <th>Δ R-1 vs base</th>
          <th>Δ R-L vs base</th>
          {bert_header}
          {judge_header}
          <th>Mean words</th>
        </tr>
      </thead>
      <tbody>
        {"".join(table_rows)}
      </tbody>
    </table>
  </div>
  <p class="note">
    Δ is system − base on the same n examples. Higher ROUGE / BERTScore = closer to the
    curated reference (style-confounded for raw targets). Judge scores are article-grounded
    faithfulness / fluency (1–5), not reference overlap.
  </p>
  <div class="controls">
    <input type="search" id="q" placeholder="Search Hebrew / English text…" />
    <div class="filters" id="filters">
      <button type="button" data-filter="all" class="active">All</button>
      <button type="button" data-filter="base" data-mode="rougeL">Best R-L: base</button>
      <button type="button" data-filter="raw" data-mode="rougeL">Best R-L: raw</button>
      <button type="button" data-filter="curated" data-mode="rougeL">Best R-L: curated</button>
      <button type="button" data-filter="tie" data-mode="rougeL">R-L ties</button>
      {"<button type='button' data-filter='base' data-mode='faithfulness'>Best faith: base</button>"
       "<button type='button' data-filter='raw' data-mode='faithfulness'>Best faith: raw</button>"
       "<button type='button' data-filter='curated' data-mode='faithfulness'>Best faith: curated</button>"
       if has_judge else ""}
    </div>
    <span class="meta-count" id="count"></span>
  </div>
</header>
<main id="list"></main>
<script id="payload" type="application/json">{data_json}</script>
<script>
(function () {{
  const payload = JSON.parse(document.getElementById("payload").textContent);
  const examples = payload.examples;
  const hasJudge = !!payload.has_judge;
  let filter = "all";
  let filterMode = "rougeL";
  let query = "";

  const listEl = document.getElementById("list");
  const countEl = document.getElementById("count");
  const qEl = document.getElementById("q");

  function esc(str) {{
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }}

  function badge(winner, mode) {{
    const prefix = mode === "faithfulness" ? "Best faith" : "Best ROUGE-L";
    const labels = {{
      base: prefix + ": base",
      raw: prefix + ": raw",
      curated: prefix + ": curated",
      tie: mode === "faithfulness" ? "Faithfulness tie" : "ROUGE-L tie",
    }};
    return `<span class="badge ${{winner}}">${{labels[winner] || winner}}</span>`;
  }}

  function pill(arm, score, best, label) {{
    const cls = best ? "pill best" : "pill";
    const s = (typeof score === "number") ? score.toFixed(3) : "—";
    return `<span class="${{cls}}">${{arm}} ${{label}} ${{s}}</span>`;
  }}

  function judgePill(arm, faith, flu, bestFaith) {{
    if (faith == null && flu == null) return "";
    const cls = bestFaith ? "pill best" : "pill";
    const f = (typeof faith === "number") ? faith : "—";
    const l = (typeof flu === "number") ? flu : "—";
    return `<span class="${{cls}}">${{arm}} F${{f}}/L${{l}}</span>`;
  }}

  function armBlock(colorVar, title, text, bestRl, bestFaith, faith, flu) {{
    const tags = [];
    if (bestRl) tags.push('<span class="pill best">best R-L</span>');
    if (hasJudge && bestFaith) tags.push('<span class="pill best">best faith</span>');
    let judgeLine = "";
    if (hasJudge) {{
      const f = (typeof faith === "number") ? faith : "—";
      const l = (typeof flu === "number") ? flu : "—";
      judgeLine = `<div class="label" style="margin-top:0.35rem;font-weight:500">
        faith ${{f}} · fluency ${{l}}</div>`;
    }}
    const ring = (bestRl || (hasJudge && bestFaith)) ? "winner-ring" : "";
    return `
      <div>
        <div class="label" style="color: var(--${{colorVar}})">
          ${{title}} ${{tags.join(" ")}}
        </div>
        <div class="box rtl ${{ring}}">${{esc(text)}}</div>
        ${{judgeLine}}
      </div>`;
  }}

  function card(ex) {{
    const w = ex.winner_rougeL;
    const wf = hasJudge ? (ex.winner_faithfulness || "tie") : null;
    const bestBase = w === "base";
    const bestRaw = w === "raw";
    const bestCur = w === "curated";
    const bestFaithBase = hasJudge && wf === "base";
    const bestFaithRaw = hasJudge && wf === "raw";
    const bestFaithCur = hasJudge && wf === "curated";
    const judgePills = hasJudge ? `
      ${{judgePill("base", ex.base_faithfulness, ex.base_fluency, bestFaithBase || wf === "tie")}}
      ${{judgePill("raw", ex.raw_faithfulness, ex.raw_fluency, bestFaithRaw || wf === "tie")}}
      ${{judgePill("cur", ex.curated_faithfulness, ex.curated_fluency, bestFaithCur || wf === "tie")}}
    ` : "";
    return `
<article class="card" data-idx="${{ex.idx}}"
  data-winner="${{esc(w)}}" data-winner-faith="${{esc(wf || "")}}">
  <div class="card-head">
    <span class="num">#${{ex.idx}}</span>
    ${{badge(w, "rougeL")}}
    ${{hasJudge ? badge(wf, "faithfulness") : ""}}
    <div class="score-pills">
      ${{pill("base", ex.base_rougeL, bestBase || w === "tie", "R-L")}}
      ${{pill("raw", ex.raw_rougeL, bestRaw || w === "tie", "R-L")}}
      ${{pill("cur", ex.curated_rougeL, bestCur || w === "tie", "R-L")}}
      ${{judgePills}}
    </div>
  </div>
  <div class="card-body">
    <div class="block">
      <div class="label">Article <span class="pill">${{esc(String(ex.text.length))}} chars</span></div>
      <div class="box article rtl" id="art-${{ex.idx}}">${{esc(ex.text)}}</div>
      <button type="button" class="toggle-article" data-toggle="${{ex.idx}}">Show full article</button>
    </div>
    <div class="block">
      <div class="label" style="color: var(--ref)">Reference (curated headline)</div>
      <div class="box ref rtl">${{esc(ex.reference)}}</div>
    </div>
    <div class="grid3 block">
      ${{armBlock("base", "Base (zero-shot)", ex.base, bestBase, bestFaithBase,
        ex.base_faithfulness, ex.base_fluency)}}
      ${{armBlock("raw", "Raw SFT", ex.raw, bestRaw, bestFaithRaw,
        ex.raw_faithfulness, ex.raw_fluency)}}
      ${{armBlock("curated", "Curated SFT", ex.curated, bestCur, bestFaithCur,
        ex.curated_faithfulness, ex.curated_fluency)}}
    </div>
  </div>
</article>`;
  }}

  function matches(ex) {{
    if (filter !== "all") {{
      const winner = filterMode === "faithfulness"
        ? ex.winner_faithfulness
        : ex.winner_rougeL;
      if (winner !== filter) return false;
    }}
    if (!query) return true;
    const q = query.toLowerCase();
    const hay = [ex.text, ex.reference, ex.base, ex.raw, ex.curated, String(ex.idx)]
      .join("\\n").toLowerCase();
    return hay.includes(q);
  }}

  function render() {{
    const shown = examples.filter(matches);
    countEl.textContent = `${{shown.length}} / ${{examples.length}} examples`;
    listEl.innerHTML = shown.map(card).join("") ||
      '<p class="note">No examples match this filter.</p>';
    listEl.querySelectorAll(".toggle-article").forEach((btn) => {{
      btn.addEventListener("click", () => {{
        const id = btn.getAttribute("data-toggle");
        const box = document.getElementById("art-" + id);
        const open = box.classList.toggle("expanded");
        btn.textContent = open ? "Collapse article" : "Show full article";
      }});
    }});
  }}

  document.getElementById("filters").addEventListener("click", (ev) => {{
    const btn = ev.target.closest("button[data-filter]");
    if (!btn) return;
    filter = btn.getAttribute("data-filter");
    filterMode = btn.getAttribute("data-mode") || "rougeL";
    if (filter === "all") filterMode = "rougeL";
    document.querySelectorAll("#filters button").forEach((b) => {{
      b.classList.toggle("active", b === btn);
    }});
    render();
  }});

  qEl.addEventListener("input", () => {{
    query = qEl.value.trim();
    render();
  }});

  render();
}})();
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="E4 base vs raw vs curated metrics + HTML (ROUGE/BERTScore + LLM judge)"
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("outputs/results/e4/predictions-e4-raw.jsonl"),
    )
    parser.add_argument(
        "--curated",
        type=Path,
        default=Path("outputs/results/e4/predictions-e4-curated.jsonl"),
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("outputs/results/dictalm2-sft-full/predictions-base.jsonl"),
        help="Zero-shot base predictions (matched by article text)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/results/e4"),
    )
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore (faster; ROUGE + length only)",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip Gemini faithfulness/fluency judge",
    )
    parser.add_argument(
        "--judge-cache",
        type=Path,
        default=None,
        help="JSON cache for judge scores (default: <output-dir>/e4-judge-cache.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Score only first N paired examples (0 = all; smoke)",
    )
    args = parser.parse_args()

    for path, label in (
        (args.raw, "raw"),
        (args.curated, "curated"),
        (args.base, "base"),
    ):
        if not path.is_file():
            print(f"error: missing {label} predictions: {path}", file=sys.stderr)
            return 1

    print(f"Loading raw={args.raw}")
    raw_rows = load_predictions(args.raw)
    print(f"Loading curated={args.curated}")
    cur_rows = load_predictions(args.curated)
    print(f"Loading base={args.base}")
    base_rows = load_predictions(args.base)

    pairs = pair_three(base_rows, raw_rows, cur_rows)
    print(
        f"Paired n={len(pairs)} "
        f"(base={len(base_rows)} raw={len(raw_rows)} curated={len(cur_rows)})"
    )
    if not pairs:
        print("error: no shared articles across the three prediction files", file=sys.stderr)
        return 1
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]
        print(f"Limited to first n={len(pairs)} (--limit)")

    print("Scoring ROUGE / length" + (" + BERTScore" if not args.skip_bertscore else "") + "…")
    systems = score_arms(pairs, skip_bertscore=args.skip_bertscore)
    examples = per_example_rouge(pairs, normalize=False)

    has_judge = not args.skip_judge
    cache_path = args.judge_cache or (args.output_dir / "e4-judge-cache.json")
    if has_judge:
        print(f"LLM judge ({GEMINI_MODEL}, T=0) on base/raw/curated…")
        cache = load_judge_cache(cache_path)
        try:
            examples = attach_judge_scores(examples, cache)
        except Exception as exc:  # noqa: BLE001 — surface API/env failures clearly
            print(f"error: judge failed: {exc}", file=sys.stderr)
            save_judge_cache(cache_path, cache)
            return 1
        save_judge_cache(cache_path, cache)
        print(f"Wrote judge cache {cache_path} ({len(cache)} keys)")
        for arm in ("base", "raw", "curated"):
            systems[arm]["judge"] = judge_arm_means(examples, arm)

    summary = build_summary(systems, has_judge=has_judge)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "e4-auto-metrics.json"
    html_path = args.output_dir / "e4-base-compare.html"

    metrics_parts = ["rouge", "rouge_normalized", "length"]
    if not args.skip_bertscore:
        metrics_parts.append("bertscore")
    if has_judge:
        metrics_parts.append("gemini_judge_faith_flu")

    report = {
        "n": len(pairs),
        "base_file": str(args.base),
        "raw_file": str(args.raw),
        "curated_file": str(args.curated),
        "metrics": " + ".join(metrics_parts),
        "llm_judge": has_judge,
        "judge_model": GEMINI_MODEL if has_judge else None,
        "judge_temperature": 0.0 if has_judge else None,
        "systems": systems,
        "summary": summary,
        "per_example_winner_counts_rougeL": {
            arm: sum(1 for e in examples if best_arm_by_metric(e, "rougeL") == arm)
            for arm in ("base", "raw", "curated", "tie")
        },
    }
    if has_judge:
        report["per_example_winner_counts_faithfulness"] = {
            arm: sum(1 for e in examples if best_arm_by_judge(e, "faithfulness") == arm)
            for arm in ("base", "raw", "curated", "tie")
        }
        # Compact per-example scores for re-analysis without re-judging.
        report["per_example_judge"] = [
            {
                "idx": e["idx"],
                "base_faithfulness": e.get("base_faithfulness"),
                "base_fluency": e.get("base_fluency"),
                "raw_faithfulness": e.get("raw_faithfulness"),
                "raw_fluency": e.get("raw_fluency"),
                "curated_faithfulness": e.get("curated_faithfulness"),
                "curated_fluency": e.get("curated_fluency"),
                "winner_faithfulness": best_arm_by_judge(e, "faithfulness"),
                "winner_rougeL": best_arm_by_metric(e, "rougeL"),
            }
            for e in examples
        ]

    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(
        render_html(
            summary,
            examples,
            has_bertscore=not args.skip_bertscore,
            has_judge=has_judge,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {metrics_path}")
    print(f"Wrote {html_path}")
    print("\nAggregate (vs curated reference):")
    for row in summary["systems"]:
        extra = ""
        if row["system"] != "base":
            extra = (
                f"  ΔR-L={row.get('delta_rougeL_vs_base'):+.4f}"
                f"  ΔR-1={row.get('delta_rouge1_vs_base'):+.4f}"
            )
            if "delta_bertscore_f1_vs_base" in row:
                extra += f"  ΔBS={row['delta_bertscore_f1_vs_base']:+.4f}"
            if row.get("delta_faithfulness_vs_base") is not None:
                extra += f"  ΔFaith={row['delta_faithfulness_vs_base']:+.3f}"
            if row.get("delta_fluency_vs_base") is not None:
                extra += f"  ΔFlu={row['delta_fluency_vs_base']:+.3f}"
        bs = f"  BS-F1={row['bertscore_f1']:.4f}" if "bertscore_f1" in row else ""
        judge = ""
        if row.get("faithfulness_mean") is not None:
            judge = (
                f"  Faith={row['faithfulness_mean']:.3f}"
                f"  Flu={row['fluency_mean']:.3f}"
            )
        print(
            f"  {row['system']:8s}  R1={row['rouge1']:.4f}  R2={row['rouge2']:.4f}  "
            f"RL={row['rougeL']:.4f}{bs}{judge}  words={row['mean_words']:.1f}{extra}"
        )
    wins = report["per_example_winner_counts_rougeL"]
    print(
        f"\nPer-example best ROUGE-L: base={wins['base']} raw={wins['raw']} "
        f"curated={wins['curated']} tie={wins['tie']}"
    )
    if has_judge:
        fw = report["per_example_winner_counts_faithfulness"]
        print(
            f"Per-example best faithfulness: base={fw['base']} raw={fw['raw']} "
            f"curated={fw['curated']} tie={fw['tie']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
