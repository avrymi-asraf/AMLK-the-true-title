"""
E4 scoring driver: four-dimension rubric deltas + blind pairwise win rate.

Pipeline role: after E4-RAW and E4-CUR adapters produce prediction JSONLs (same
shared curated test articles), this script scores both arms with the Reference
Quality Rubric (`evaluation.rubric_judge` — faithfulness, single-focus,
informativeness, cleanliness), pairs rows by article text, runs pairwise_judge
with randomized sides, and prints Cliff's δ + Wilson CI via
data_curation.analysis.stats — the same instrument and statistics as E1–E3 so
model outputs and dataset references sit on one axis. Also writes a self-contained
HTML results viewer. ROUGE/BERTScore are style-confounded on curated references
and are not the decision rule.

Run (from repo root, with GEMINI_API_KEY):
  python -m scripts.e4_score \\
      --raw outputs/results/e4/predictions-e4-raw.jsonl \\
      --curated outputs/results/e4/predictions-e4-curated.jsonl \\
      --limit 120 --output outputs/results/e4/e4-score-summary.json

Reuse an existing pairwise JSONL without re-calling the API:
  python -m scripts.e4_score ... --skip-pairwise --load-pairwise outputs/results/e4/e4-pairwise.jsonl

Execution environment: local machine, API-bound (Gemini), CPU only. No GPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from data_curation.analysis.stats import (
    bootstrap_cliffs_delta_ci,
    cliffs_delta,
    wilson_ci,
)
from evaluation.gemini_client import GEMINI_MODEL, strip_think
from evaluation.pairwise_judge import compare_headlines
from evaluation.rubric_judge import DIMENSIONS, score_headline

INSTRUMENT = "rubric_v1"
DECISION_RULE = (
    "curation wins if pairwise Wilson CI excludes 50% "
    "OR paired faithfulness Cliff's CI excludes 0"
)


def load_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["prediction"] = strip_think(row.get("prediction") or "")
            rows.append(row)
    return rows


def index_by_text(rows: list[dict]) -> dict[str, dict]:
    """First row per article text (predictions are unique per test article)."""
    out: dict[str, dict] = {}
    for r in rows:
        text = r.get("text") or ""
        if text and text not in out:
            out[text] = r
    return out


def pair_by_text(raw_rows: list[dict], cur_rows: list[dict]) -> list[dict]:
    """Join arms on article text; drop unpaired rows."""
    raw_ix = index_by_text(raw_rows)
    cur_ix = index_by_text(cur_rows)
    shared = sorted(set(raw_ix) & set(cur_ix))
    return [
        {
            "text": t,
            "raw_prediction": raw_ix[t]["prediction"],
            "curated_prediction": cur_ix[t]["prediction"],
            "reference": cur_ix[t].get("reference") or cur_ix[t].get("summary") or "",
        }
        for t in shared
    ]


def cache_key(article: str, prediction: str) -> str:
    """Stable hash for resume cache: article + prediction text."""
    blob = (article + "\0" + prediction).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_json_cache(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_json_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _flatten_scores(scored: dict) -> dict[str, int] | None:
    """Turn rubric_judge output into {dim: score} when all four dims are present."""
    flat: dict[str, int] = {}
    for dim in DIMENSIONS:
        entry = scored.get(dim)
        if not isinstance(entry, dict):
            return None
        score = entry.get("score")
        if not isinstance(score, int) or not (1 <= score <= 5):
            return None
        flat[dim] = score
    return flat


def score_arm_rubric(
    rows: list[dict],
    *,
    model=None,
    temperature: float = 0.0,
    cache: dict | None = None,
    cache_path: Path | None = None,
    arm_label: str = "arm",
) -> list[dict]:
    """Score each {text, prediction} on the four rubric dims; resume via cache.

    Returns one dict per input row: either flat scores for all four dims, or
    empty scores when the judge failed / returned a partial parse. Never invents
    default scores.
    """
    if cache is None:
        cache = {}
    results: list[dict] = []
    n = len(rows)
    for i, row in enumerate(rows):
        text = row.get("text") or ""
        prediction = row.get("prediction") or ""
        key = cache_key(text, prediction)
        hit = cache.get(key)
        if isinstance(hit, dict) and all(
            isinstance(hit.get(d), int) and 1 <= hit[d] <= 5 for d in DIMENSIONS
        ):
            results.append({dim: hit[dim] for dim in DIMENSIONS})
        else:
            raw = score_headline(text, prediction, model=model, temperature=temperature)
            flat = _flatten_scores(raw)
            if flat is None:
                results.append({})
            else:
                results.append(dict(flat))
                cache[key] = dict(flat)
                if cache_path is not None:
                    save_json_cache(cache_path, cache)
        if (i + 1) % 25 == 0 or (i + 1) == n:
            print(f"  rubric {arm_label} {i + 1}/{n}", file=sys.stderr)
    return results


def rubric_paired_summary(
    raw_scores: list[dict],
    cur_scores: list[dict],
) -> dict:
    """Per-dimension means, paired mean deltas, Cliff's δ + bootstrap CI.

    A dimension pair is included only when both arms have that dim scored.
    Positive Cliff's δ means curated tends higher than raw.
    """
    if len(raw_scores) != len(cur_scores):
        raise ValueError(
            f"score list length mismatch: raw={len(raw_scores)} curated={len(cur_scores)}"
        )
    out: dict = {
        "n_pairs_attempted": len(raw_scores),
        "by_dimension": {},
    }
    for dim in DIMENSIONS:
        raw_vals: list[int] = []
        cur_vals: list[int] = []
        for r, c in zip(raw_scores, cur_scores):
            rv, cv = r.get(dim), c.get(dim)
            if isinstance(rv, int) and isinstance(cv, int):
                raw_vals.append(rv)
                cur_vals.append(cv)
        dim_out: dict = {
            "n_pairs": len(raw_vals),
            "raw_mean": round(sum(raw_vals) / len(raw_vals), 3) if raw_vals else None,
            "curated_mean": round(sum(cur_vals) / len(cur_vals), 3) if cur_vals else None,
        }
        if raw_vals:
            dim_out["mean_delta_curated_minus_raw"] = round(
                sum(cur_vals[i] - raw_vals[i] for i in range(len(raw_vals))) / len(raw_vals),
                3,
            )
        if len(raw_vals) >= 2:
            delta = cliffs_delta(cur_vals, raw_vals)
            lo, hi = bootstrap_cliffs_delta_ci(cur_vals, raw_vals)
            dim_out["cliffs_delta_curated_vs_raw"] = round(delta, 4)
            dim_out["cliffs_ci"] = [round(lo, 4), round(hi, 4)]
            dim_out["ci_excludes_0"] = bool(hi < 0 or lo > 0)
        else:
            dim_out["cliffs_delta_curated_vs_raw"] = None
            dim_out["cliffs_ci"] = None
            dim_out["ci_excludes_0"] = False
        out["by_dimension"][dim] = dim_out
        # Flat aliases for the primary decision dim and table convenience.
        out[f"raw_{dim}_mean"] = dim_out["raw_mean"]
        out[f"curated_{dim}_mean"] = dim_out["curated_mean"]
        out[f"{dim}_mean_delta_curated_minus_raw"] = dim_out.get("mean_delta_curated_minus_raw")
        out[f"{dim}_cliffs_delta_curated_vs_raw"] = dim_out.get("cliffs_delta_curated_vs_raw")
        out[f"{dim}_cliffs_ci"] = dim_out.get("cliffs_ci")
        out[f"{dim}_ci_excludes_0"] = dim_out.get("ci_excludes_0", False)
        out[f"n_{dim}_pairs"] = dim_out["n_pairs"]
    return out


def arm_wins(winner: str, curated_is_a: bool) -> str:
    """Map judge winner + side randomization → curated | raw | tie."""
    if winner == "tie":
        return "tie"
    curated_won = (winner == "a") == curated_is_a
    return "curated" if curated_won else "raw"


def run_pairwise(
    pairs: list[dict],
    seed: int = 42,
    model=None,
) -> list[dict]:
    """Blind A/B for each pair; randomize which arm is Headline A."""
    rng = random.Random(seed)
    results: list[dict] = []
    for i, p in enumerate(pairs):
        curated_is_a = rng.random() < 0.5
        if curated_is_a:
            a, b = p["curated_prediction"], p["raw_prediction"]
        else:
            a, b = p["raw_prediction"], p["curated_prediction"]
        verdict = compare_headlines(p["text"], a, b, model=model)
        winner = verdict.get("winner", "")
        outcome = arm_wins(winner, curated_is_a) if winner else "failed"
        results.append({
            "curated_is_a": curated_is_a,
            "winner": winner,
            "outcome": outcome,
            "justification": verdict.get("justification", ""),
            "text_prefix": p["text"][:80],
        })
        if (i + 1) % 25 == 0:
            print(f"  pairwise {i + 1}/{len(pairs)}", file=sys.stderr)
    return results


def pairwise_summary(results: list[dict]) -> dict:
    """Win counts + Wilson CI for curated win rate (excluding ties/failures)."""
    outcomes = [r["outcome"] for r in results if r["outcome"] in ("curated", "raw", "tie")]
    n = len(outcomes)
    curated_n = sum(1 for o in outcomes if o == "curated")
    raw_n = sum(1 for o in outcomes if o == "raw")
    tie_n = sum(1 for o in outcomes if o == "tie")
    decided = curated_n + raw_n
    win_rate = (100.0 * curated_n / decided) if decided else 0.0
    lo, hi = wilson_ci(curated_n, decided) if decided else (0.0, 0.0)
    return {
        "n_judged": n,
        "curated_wins": curated_n,
        "raw_wins": raw_n,
        "ties": tie_n,
        "failed": sum(1 for r in results if r["outcome"] == "failed"),
        "curated_win_rate_pct": round(win_rate, 2),
        "wilson_ci_pct": [round(lo, 2), round(hi, 2)],
        "excludes_50": bool(decided and (hi < 50.0 or lo > 50.0)),
    }


def load_pairwise_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_rubric_jsonl(
    path: Path,
    pairs: list[dict],
    raw_scores: list[dict],
    cur_scores: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p, r, c in zip(pairs, raw_scores, cur_scores):
            f.write(json.dumps({
                "text_prefix": p["text"][:80],
                "raw_scores": r,
                "curated_scores": c,
            }, ensure_ascii=False) + "\n")


def decision_flag(summary: dict) -> bool:
    pairwise_ok = summary.get("pairwise", {}).get("excludes_50", False)
    faith_ok = summary.get("pointwise", {}).get("faithfulness_ci_excludes_0", False)
    return bool(pairwise_ok or faith_ok)


def build_viewer_examples(
    pairs: list[dict],
    pairwise_rows: list[dict],
    raw_scores: list[dict] | None = None,
    cur_scores: list[dict] | None = None,
) -> list[dict]:
    """Join preds + pairwise (and optional rubric scores) into viewer rows."""
    if len(pairs) != len(pairwise_rows):
        raise ValueError(
            f"pairs ({len(pairs)}) vs pairwise ({len(pairwise_rows)}) length mismatch"
        )
    if raw_scores is not None and len(raw_scores) != len(pairs):
        raise ValueError("raw_scores length mismatch")
    if cur_scores is not None and len(cur_scores) != len(pairs):
        raise ValueError("cur_scores length mismatch")

    examples: list[dict] = []
    for i, (p, pw) in enumerate(zip(pairs, pairwise_rows)):
        curated_is_a = bool(pw.get("curated_is_a"))
        if curated_is_a:
            headline_a, headline_b = p["curated_prediction"], p["raw_prediction"]
            label_a, label_b = "curated", "raw"
        else:
            headline_a, headline_b = p["raw_prediction"], p["curated_prediction"]
            label_a, label_b = "raw", "curated"
        row = {
            "idx": i + 1,
            "text": p["text"],
            "reference": p.get("reference") or "",
            "raw": p["raw_prediction"],
            "curated": p["curated_prediction"],
            "outcome": pw.get("outcome", "failed"),
            "winner": pw.get("winner", ""),
            "justification": pw.get("justification", ""),
            "curated_is_a": curated_is_a,
            "headline_a": headline_a,
            "headline_b": headline_b,
            "label_a": label_a,
            "label_b": label_b,
        }
        if raw_scores is not None:
            row["raw_scores"] = raw_scores[i]
        if cur_scores is not None:
            row["curated_scores"] = cur_scores[i]
        examples.append(row)
    return examples


def load_rubric_jsonl(path: Path) -> tuple[list[dict], list[dict]]:
    """Load per-row raw/curated score dicts from e4-rubric-scores.jsonl."""
    raw_scores: list[dict] = []
    cur_scores: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_scores.append(row.get("raw_scores") or {})
            cur_scores.append(row.get("curated_scores") or {})
    return raw_scores, cur_scores


def render_e4_results_html(summary: dict, examples: list[dict]) -> str:
    """Self-contained offline viewer for raw vs curated (rubric + pairwise)."""
    payload = {"summary": summary, "examples": examples, "dimensions": list(DIMENSIONS)}
    data_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    instrument = summary.get("instrument", "rubric_v1")
    n = summary.get("n_pairs", len(examples))
    return f"""<!DOCTYPE html>
<html lang="he">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>E4 Results — Raw vs Curated SFT</title>
<style>
  :root {{
    --bg: #0f1419; --surface: #1a2332; --surface2: #243044; --border: #2d3a4f;
    --text: #e7ecf3; --muted: #8b9bb4; --raw: #f59e0b; --raw-bg: rgba(245,158,11,.12);
    --curated: #34d399; --curated-bg: rgba(52,211,153,.12); --tie: #94a3b8;
    --accent: #60a5fa; --ref: #a78bfa; --ref-bg: rgba(167,139,250,.1);
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: "Segoe UI", system-ui, sans-serif; background: var(--bg); color: var(--text); line-height: 1.55; }}
  header {{ position: sticky; top: 0; z-index: 20; background: rgba(15,20,25,.92); backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border); padding: .85rem 1.25rem 1rem; }}
  header h1 {{ margin: 0 0 .35rem; font-size: 1.25rem; font-weight: 650; letter-spacing: -.02em; }}
  header .sub {{ color: var(--muted); font-size: .85rem; margin-bottom: .75rem; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: .75rem; }}
  .stat {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: .45rem .7rem; font-size: .8rem; }}
  .stat strong {{ color: var(--accent); font-weight: 650; }}
  .stat.win strong {{ color: var(--curated); }}
  .stat.lose strong {{ color: var(--raw); }}
  .controls {{ display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; }}
  .controls input[type="search"] {{ flex: 1 1 220px; min-width: 180px; background: var(--surface); border: 1px solid var(--border);
    color: var(--text); border-radius: 8px; padding: .5rem .75rem; font-size: .9rem; }}
  .filters {{ display: flex; flex-wrap: wrap; gap: .35rem; }}
  .filters button {{ background: var(--surface); border: 1px solid var(--border); color: var(--muted); border-radius: 999px;
    padding: .35rem .75rem; font-size: .8rem; cursor: pointer; }}
  .filters button.active {{ background: rgba(96,165,250,.15); border-color: var(--accent); color: var(--accent); }}
  .meta-count {{ color: var(--muted); font-size: .8rem; margin-inline-start: auto; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 1rem 1.25rem 3rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; margin-bottom: 1rem; overflow: hidden; }}
  .card-head {{ display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; padding: .75rem 1rem;
    border-bottom: 1px solid var(--border); background: var(--surface2); }}
  .card-head .num {{ font-weight: 700; font-variant-numeric: tabular-nums; color: var(--muted); min-width: 2.5rem; }}
  .badge {{ display: inline-flex; border-radius: 999px; padding: .2rem .65rem; font-size: .75rem; font-weight: 650;
    text-transform: uppercase; letter-spacing: .03em; }}
  .badge.raw {{ background: var(--raw-bg); color: var(--raw); }}
  .badge.curated {{ background: var(--curated-bg); color: var(--curated); }}
  .badge.tie, .badge.side {{ background: rgba(148,163,184,.12); color: var(--tie); }}
  .score-pills {{ display: flex; flex-wrap: wrap; gap: .35rem; margin-inline-start: auto; }}
  .pill {{ font-size: .72rem; font-variant-numeric: tabular-nums; border: 1px solid var(--border);
    border-radius: 6px; padding: .15rem .45rem; color: var(--muted); }}
  .pill.raw {{ border-color: rgba(245,158,11,.4); color: var(--raw); }}
  .pill.curated {{ border-color: rgba(52,211,153,.4); color: var(--curated); }}
  .card-body {{ padding: .9rem 1rem 1.1rem; }}
  .block {{ margin-bottom: .85rem; }}
  .label {{ font-size: .72rem; font-weight: 650; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); margin-bottom: .3rem; }}
  .box {{ background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: .65rem .8rem; font-size: .95rem; }}
  .box.article {{ max-height: 5.5em; overflow: hidden; color: var(--muted); font-size: .88rem; }}
  .box.article.expanded {{ max-height: none; color: var(--text); }}
  .box.ref {{ background: var(--ref-bg); border-color: rgba(167,139,250,.35); }}
  .box.winner-ring {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
  .rtl {{ direction: rtl; text-align: right; unicode-bidi: plaintext; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: .65rem; }}
  @media (max-width: 800px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
  .chip {{ font-size: .72rem; font-weight: 600; margin-inline-start: .35rem; }}
  .toggle-article {{ margin-top: .35rem; background: transparent; border: none; color: var(--accent); cursor: pointer; font-size: .8rem; padding: 0; }}
  .pairwise {{ background: var(--surface2); border-radius: 8px; padding: .75rem; border: 1px solid var(--border); }}
  .pairwise h3 {{ margin: 0 0 .5rem; font-size: .85rem; color: var(--muted); font-weight: 650; }}
  .ab-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; margin-bottom: .5rem; }}
  @media (max-width: 800px) {{ .ab-row {{ grid-template-columns: 1fr; }} }}
  .ab {{ background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: .55rem .7rem; font-size: .9rem; }}
  .ab.picked {{ outline: 2px solid var(--accent); }}
  .ab-label {{ font-size: .72rem; color: var(--muted); margin-bottom: .3rem; }}
  .justification {{ background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: .55rem .7rem; font-size: .88rem; color: var(--muted); }}
  footer {{ color: var(--muted); font-size: .75rem; text-align: center; padding: 1rem; }}
  .empty {{ color: var(--muted); padding: 2rem; text-align: center; }}
</style>
</head>
<body>
<header>
  <h1>E4 — Raw vs Curated SFT (DictaLM2)</h1>
  <div class="sub">{n} shared curated-test articles · blind Gemini pairwise · four-dimension rubric ({instrument})</div>
  <div class="stats" id="stats"></div>
  <div class="controls">
    <input type="search" id="q" placeholder="Search Hebrew / English in article, summaries, justification…" />
    <div class="filters" id="filters">
      <button type="button" data-f="all" class="active">All</button>
      <button type="button" data-f="curated">Curated wins</button>
      <button type="button" data-f="raw">Raw wins</button>
      <button type="button" data-f="tie">Ties</button>
    </div>
    <span class="meta-count" id="count"></span>
  </div>
</header>
<main id="list"></main>
<footer>AMLK E4 results viewer · rubric + pairwise · offline</footer>
<script id="data" type="application/json">{data_json}</script>
<script>
(function () {{
  const payload = JSON.parse(document.getElementById("data").textContent);
  const s = payload.summary || {{}};
  const examples = payload.examples || [];
  const dims = payload.dimensions || ["faithfulness","single_focus","informativeness","cleanliness"];
  const dimShort = {{faithfulness:"F", single_focus:"SF", informativeness:"I", cleanliness:"C"}};
  const pw = s.pairwise || {{}};
  const pt = s.pointwise || {{}};
  const byDim = pt.by_dimension || {{}};

  const statsEl = document.getElementById("stats");
  const decision = s.curation_wins_by_decision_rule
    ? '<span class="stat win"><strong>Decision: curation wins</strong></span>'
    : '<span class="stat"><strong>Decision: no clear win</strong></span>';
  const dimStats = dims.map((d) => {{
    const row = byDim[d] || {{}};
    const raw = row.raw_mean ?? pt["raw_" + d + "_mean"];
    const cur = row.curated_mean ?? pt["curated_" + d + "_mean"];
    const delta = row.cliffs_delta_curated_vs_raw ?? pt[d + "_cliffs_delta_curated_vs_raw"];
    const excl = row.ci_excludes_0 ?? pt[d + "_ci_excludes_0"];
    const label = d.replace("_", " ");
    const dStr = (typeof delta === "number") ? delta.toFixed(2) : "—";
    const star = excl ? " *" : "";
    return `<span class="stat">${{label}} raw/cur <strong>${{raw ?? "—"}} / ${{cur ?? "—"}}</strong> (δ=${{dStr}}${{star}})</span>`;
  }});
  statsEl.innerHTML = [
    decision,
    `<span class="stat win">Curated wins <strong>${{pw.curated_wins ?? "—"}}</strong></span>`,
    `<span class="stat lose">Raw wins <strong>${{pw.raw_wins ?? "—"}}</strong></span>`,
    `<span class="stat">Ties <strong>${{pw.ties ?? "—"}}</strong></span>`,
    `<span class="stat">Win rate <strong>${{pw.curated_win_rate_pct ?? "—"}}%</strong> (Wilson ${{
      (pw.wilson_ci_pct || []).join("–") || "—"
    }}%)</span>`,
    ...dimStats,
    `<span class="stat">Judge <strong>${{s.judge_model || "—"}}</strong> · T=${{s.judge_temperature ?? "—"}}</span>`,
    `<span class="stat">* Cliff CI excludes 0</span>`,
  ].join("");

  let filter = "all";
  let query = "";
  const listEl = document.getElementById("list");
  const countEl = document.getElementById("count");
  const qEl = document.getElementById("q");

  function esc(str) {{
    return String(str ?? "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }}

  function badge(outcome) {{
    const cls = outcome === "curated" ? "curated" : outcome === "raw" ? "raw" : "tie";
    const label = outcome === "curated" ? "Curated wins" : outcome === "raw" ? "Raw wins" : "Tie";
    return `<span class="badge ${{cls}}">${{label}}</span>`;
  }}

  function scorePill(arm, scores) {{
    if (!scores || typeof scores !== "object") return "";
    const parts = dims.map((d) => {{
      const v = scores[d];
      return (dimShort[d] || d) + ((typeof v === "number") ? v : "—");
    }});
    return `<span class="pill ${{arm}}">${{arm}} ${{parts.join(" ")}}</span>`;
  }}

  function card(ex) {{
    const winA = ex.winner === "a";
    const winB = ex.winner === "b";
    const isTie = ex.winner === "tie" || ex.outcome === "tie";
    const rawWon = ex.outcome === "raw";
    const curWon = ex.outcome === "curated";
    return `
<article class="card" data-idx="${{ex.idx}}" data-outcome="${{esc(ex.outcome)}}">
  <div class="card-head">
    <span class="num">#${{ex.idx}}</span>
    ${{badge(ex.outcome)}}
    <span class="badge side">Blind: judge picked ${{
      isTie ? "tie" : "Headline " + String(ex.winner || "?").toUpperCase()
    }}</span>
    <span class="badge side">Curated was ${{ex.curated_is_a ? "A" : "B"}}</span>
    <div class="score-pills">
      ${{scorePill("raw", ex.raw_scores)}}
      ${{scorePill("curated", ex.curated_scores)}}
    </div>
  </div>
  <div class="card-body">
    <div class="block">
      <div class="label">Article <span class="chip">${{esc(String(ex.text.length))}} chars</span></div>
      <div class="box article rtl" id="art-${{ex.idx}}">${{esc(ex.text)}}</div>
      <button type="button" class="toggle-article" data-toggle="${{ex.idx}}">Show full article</button>
    </div>
    <div class="block">
      <div class="label" style="color: var(--ref)">Reference (curated headline)</div>
      <div class="box ref rtl">${{esc(ex.reference)}}</div>
    </div>
    <div class="grid2 block">
      <div>
        <div class="label" style="color: var(--raw)">
          Raw SFT prediction
          ${{rawWon ? '<span class="chip" style="color:var(--raw)">pairwise winner</span>' : ""}}
        </div>
        <div class="box rtl ${{rawWon ? "winner-ring" : ""}}">${{esc(ex.raw)}}</div>
        <div class="label" style="margin-top:.35rem;font-weight:500">rubric ${{
          dims.map((d) => (dimShort[d]||d) + ":" + ((ex.raw_scores||{{}})[d] ?? "—")).join(" · ")
        }}</div>
      </div>
      <div>
        <div class="label" style="color: var(--curated)">
          Curated SFT prediction
          ${{curWon ? '<span class="chip" style="color:var(--curated)">pairwise winner</span>' : ""}}
        </div>
        <div class="box rtl ${{curWon ? "winner-ring" : ""}}">${{esc(ex.curated)}}</div>
        <div class="label" style="margin-top:.35rem;font-weight:500">rubric ${{
          dims.map((d) => (dimShort[d]||d) + ":" + ((ex.curated_scores||{{}})[d] ?? "—")).join(" · ")
        }}</div>
      </div>
    </div>
    <div class="pairwise block">
      <h3>Blind pairwise (as shown to the judge)</h3>
      <div class="ab-row">
        <div class="ab ${{winA ? "picked" : ""}}">
          <div class="ab-label">Headline A · arm: <span>${{esc(ex.label_a)}}</span>${{winA ? " · ✓ picked" : ""}}</div>
          <div class="rtl">${{esc(ex.headline_a)}}</div>
        </div>
        <div class="ab ${{winB ? "picked" : ""}}">
          <div class="ab-label">Headline B · arm: <span>${{esc(ex.label_b)}}</span>${{winB ? " · ✓ picked" : ""}}</div>
          <div class="rtl">${{esc(ex.headline_b)}}</div>
        </div>
      </div>
      <div class="label">Judge justification</div>
      <div class="justification">${{esc(ex.justification)}}</div>
    </div>
  </div>
</article>`;
  }}

  function matches(ex) {{
    if (filter !== "all" && ex.outcome !== filter) return false;
    if (!query) return true;
    const q = query.toLowerCase();
    const hay = [ex.text, ex.reference, ex.raw, ex.curated, ex.justification, String(ex.idx), ex.outcome]
      .join("\\n").toLowerCase();
    return hay.includes(q);
  }}

  function render() {{
    const shown = examples.filter(matches);
    countEl.textContent = `${{shown.length}} / ${{examples.length}} examples`;
    listEl.innerHTML = shown.length
      ? shown.map(card).join("")
      : '<div class="empty">No examples match this filter/search.</div>';
  }}

  document.getElementById("filters").addEventListener("click", (e) => {{
    const btn = e.target.closest("button[data-f]");
    if (!btn) return;
    filter = btn.dataset.f;
    document.querySelectorAll("#filters button").forEach((b) => {{
      b.classList.toggle("active", b.dataset.f === filter);
    }});
    render();
  }});
  qEl.addEventListener("input", () => {{ query = qEl.value.trim(); render(); }});
  listEl.addEventListener("click", (e) => {{
    const btn = e.target.closest("[data-toggle]");
    if (!btn) return;
    const el = document.getElementById("art-" + btn.dataset.toggle);
    if (!el) return;
    const open = el.classList.toggle("expanded");
    btn.textContent = open ? "Collapse article" : "Show full article";
  }});
  render();
}})();
</script>
</body>
</html>
"""


def write_results_html(
    path: Path,
    summary: dict,
    pairs: list[dict],
    pairwise_rows: list[dict],
    raw_scores: list[dict] | None = None,
    cur_scores: list[dict] | None = None,
) -> None:
    examples = build_viewer_examples(pairs, pairwise_rows, raw_scores, cur_scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_e4_results_html(summary, examples), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="E4: four-dimension rubric + blind pairwise on raw vs curated preds",
    )
    parser.add_argument("--raw", type=Path, required=True, help="E4-RAW predictions JSONL")
    parser.add_argument("--curated", type=Path, required=True, help="E4-CUR predictions JSONL")
    parser.add_argument(
        "--limit", type=int, default=120,
        help="Judge/pairwise subset size (default 120; 0 = all paired rows)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Subset + side-randomization seed")
    parser.add_argument(
        "--skip-pointwise", action="store_true",
        help="Skip four-dimension rubric judge (pairwise only)",
    )
    parser.add_argument(
        "--skip-pairwise", action="store_true",
        help="Skip blind pairwise A/B (pointwise only)",
    )
    parser.add_argument(
        "--load-pairwise", type=Path, default=None,
        help="Load existing pairwise JSONL into the summary (no API); implies skip live pairwise",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/results/e4/e4-score-summary.json"),
        help="Summary JSON path",
    )
    parser.add_argument(
        "--pairwise-jsonl", type=Path, default=None,
        help="Path to write (or, with --load-pairwise, the file being loaded)",
    )
    parser.add_argument(
        "--rubric-cache", type=Path,
        default=Path("outputs/results/e4/e4-rubric-cache.json"),
        help="Resume cache for four-dimension scores",
    )
    parser.add_argument(
        "--rubric-jsonl", type=Path,
        default=Path("outputs/results/e4/e4-rubric-scores.jsonl"),
        help="Per-row rubric scores JSONL",
    )
    parser.add_argument(
        "--html", type=Path,
        default=Path("outputs/results/e4/e4-results-viewer.html"),
        help="Self-contained results HTML viewer path",
    )
    parser.add_argument(
        "--no-html", action="store_true",
        help="Skip writing the results HTML viewer",
    )
    parser.add_argument(
        "--rebuild-html-only", action="store_true",
        help="Rebuild HTML from existing summary + rubric/pairwise JSONLs (no API)",
    )
    args = parser.parse_args(argv)

    raw_rows = load_predictions(args.raw)
    cur_rows = load_predictions(args.curated)
    if not raw_rows or not cur_rows:
        print("ERROR: empty predictions file", file=sys.stderr)
        return 1

    pairs = pair_by_text(raw_rows, cur_rows)
    print(f"Paired by article text: {len(pairs)} "
          f"(raw={len(raw_rows)}, curated={len(cur_rows)})")
    if not pairs:
        print("ERROR: no shared article texts between arms", file=sys.stderr)
        return 1

    if args.limit and args.limit < len(pairs):
        indices = sorted(random.Random(args.seed).sample(range(len(pairs)), args.limit))
        pairs = [pairs[i] for i in indices]
        print(f"Subset: {len(pairs)} pairs (seed={args.seed})")

    if args.rebuild_html_only:
        if not args.output.is_file():
            print(f"ERROR: missing summary for rebuild: {args.output}", file=sys.stderr)
            return 1
        summary = json.loads(args.output.read_text(encoding="utf-8"))
        pairwise_path = (
            args.load_pairwise
            or args.pairwise_jsonl
            or Path("outputs/results/e4/e4-pairwise.jsonl")
        )
        if not pairwise_path.is_file():
            print(f"ERROR: missing pairwise JSONL: {pairwise_path}", file=sys.stderr)
            return 1
        pairwise_rows = load_pairwise_jsonl(pairwise_path)
        raw_scores: list[dict] | None = None
        cur_scores: list[dict] | None = None
        if args.rubric_jsonl.is_file():
            raw_scores, cur_scores = load_rubric_jsonl(args.rubric_jsonl)
        if len(pairwise_rows) != len(pairs):
            print(
                f"ERROR: pairwise n={len(pairwise_rows)} != pairs n={len(pairs)}",
                file=sys.stderr,
            )
            return 1
        if raw_scores is not None and len(raw_scores) != len(pairs):
            print(
                f"ERROR: rubric n={len(raw_scores)} != pairs n={len(pairs)}",
                file=sys.stderr,
            )
            return 1
        write_results_html(
            args.html, summary, pairs, pairwise_rows, raw_scores, cur_scores,
        )
        print(f"HTML: {args.html}")
        return 0

    need_api = (not args.skip_pointwise) or (
        not args.skip_pairwise and args.load_pairwise is None
    )
    if need_api and not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        return 1

    summary: dict = {
        "instrument": INSTRUMENT,
        "raw_file": str(args.raw),
        "curated_file": str(args.curated),
        "n_pairs": len(pairs),
        "limit": args.limit,
        "seed": args.seed,
        "judge_model": GEMINI_MODEL,
        "judge_temperature": 0.0,
        "decision_rule": DECISION_RULE,
        "dimensions": list(DIMENSIONS),
    }

    raw_scores = None
    cur_scores = None
    pairwise_rows: list[dict] | None = None

    if not args.skip_pointwise:
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(GEMINI_MODEL)
        cache = load_json_cache(args.rubric_cache)
        print(f"Rubric judge on raw arm ({GEMINI_MODEL}, T=0, cache={args.rubric_cache})...")
        raw_subset = [
            {"text": p["text"], "prediction": p["raw_prediction"]} for p in pairs
        ]
        cur_subset = [
            {"text": p["text"], "prediction": p["curated_prediction"]} for p in pairs
        ]
        raw_scores = score_arm_rubric(
            raw_subset, model=model, temperature=0.0,
            cache=cache, cache_path=args.rubric_cache, arm_label="raw",
        )
        print(f"Rubric judge on curated arm ({GEMINI_MODEL}, T=0)...")
        cur_scores = score_arm_rubric(
            cur_subset, model=model, temperature=0.0,
            cache=cache, cache_path=args.rubric_cache, arm_label="curated",
        )
        summary["pointwise"] = rubric_paired_summary(raw_scores, cur_scores)
        write_rubric_jsonl(args.rubric_jsonl, pairs, raw_scores, cur_scores)
        print(f"Rubric rows: {args.rubric_jsonl}")
    elif args.rubric_jsonl.is_file():
        raw_scores, cur_scores = load_rubric_jsonl(args.rubric_jsonl)

    if args.load_pairwise is not None:
        print(f"Loading pairwise from {args.load_pairwise}...")
        pairwise_rows = load_pairwise_jsonl(args.load_pairwise)
        summary["pairwise"] = pairwise_summary(pairwise_rows)
        summary["pairwise_source"] = str(args.load_pairwise)
    elif not args.skip_pairwise:
        print(f"Blind pairwise on {len(pairs)} pairs...")
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(GEMINI_MODEL)
        pairwise_rows = run_pairwise(pairs, seed=args.seed, model=model)
        summary["pairwise"] = pairwise_summary(pairwise_rows)
        out_jsonl = args.pairwise_jsonl or args.output.with_name(
            args.output.stem + "-pairwise.jsonl"
        )
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for row in pairwise_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Pairwise rows: {out_jsonl}")

    summary["curation_wins_by_decision_rule"] = decision_flag(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"Summary: {args.output}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if not args.no_html and pairwise_rows is not None:
        write_results_html(
            args.html, summary, pairs, pairwise_rows, raw_scores, cur_scores,
        )
        print(f"HTML: {args.html}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
