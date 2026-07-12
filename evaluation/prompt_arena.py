"""
Prompt-optimization arena: the local (CPU/API-only) half of the prompt-improvement loop.

Role in the pipeline: a side-loop that sits *before* fine-tuning — it finds the best
zero-shot prompt for the base model instead of training anything. Holds the prompt
candidates for a round, scores a swept predictions file (ROUGE + compliance signals that
match what short summarization prompts actually target), optionally adds the Gemini judge
on the top candidates only, and renders a leaderboard the agent reads before writing the
next round's prompts. Generation itself happens on HF Jobs (evaluation/prompt_sweep_hf_job.py).

Code flow: write_round(candidates) -> sweep job -> load_predictions -> score_round ->
leaderboard -> agent rewrites candidates -> next round. Driven by
notebooks/prompt_optimization.ipynb.

Execution environment: local machine. No GPU and no model load — ROUGE/compliance are pure
CPU, the judge is a Gemini API call.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ARENA_ROOT = Path("outputs/results/prompt-arena")
TEXT_PLACEHOLDER = "{text}"

# What a good summary looks like for this task, in the terms the prompts actually control.
# These bounds define the `compliance` score; they are deliberately the same targets the
# prompt wording asks for (one or two short factual Hebrew sentences, prose, no lists).
MIN_WORDS = 6
MAX_WORDS = 45
MAX_SENTENCES = 2
MIN_HEBREW_RATIO = 0.8

_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_LIST_MARKER_RE = re.compile(r"(\|)|(^\s*[-*•]\s)|(^\s*\d+[.)]\s)", re.MULTILINE)
_HEBREW_RE = re.compile(r"[֐-׿]")
_FOREIGN_LETTER_RE = re.compile(r"[A-Za-zÀ-ɏЀ-ӿͰ-Ͽ؀-ۿ]")


@dataclass(frozen=True)
class PromptCandidate:
    """One prompt under test. `template` must contain a single {text} placeholder."""

    id: str
    template: str
    note: str = ""

    def render(self, text: str) -> str:
        return self.template.replace(TEXT_PLACEHOLDER, text)


def validate_candidates(candidates: Iterable[PromptCandidate]) -> list[str]:
    """Return human-readable problems with a candidate set; empty means OK."""
    problems: list[str] = []
    seen: set[str] = set()
    candidates = list(candidates)
    if not candidates:
        problems.append("no candidates given")
    for c in candidates:
        if not c.id or not c.id.strip():
            problems.append("candidate with empty id")
        if c.id in seen:
            problems.append(f"duplicate candidate id {c.id!r}")
        seen.add(c.id)
        if TEXT_PLACEHOLDER not in c.template:
            problems.append(f"{c.id}: template is missing the {TEXT_PLACEHOLDER} placeholder")
        if not c.template.replace(TEXT_PLACEHOLDER, "").strip():
            problems.append(f"{c.id}: template has no instruction text")
    return problems


# --------------------------------------------------------------------------- round I/O
def round_dir(round_num: int, root: str | Path = ARENA_ROOT) -> Path:
    return Path(root) / f"round-{round_num}"


def write_round(
    round_num: int,
    candidates: Iterable[PromptCandidate],
    root: str | Path = ARENA_ROOT,
) -> Path:
    """Persist a round's candidates to round-<n>/prompts.json (the sweep job's input)."""
    candidates = list(candidates)
    problems = validate_candidates(candidates)
    if problems:
        raise ValueError("invalid prompt candidates: " + "; ".join(problems))
    path = round_dir(round_num, root) / "prompts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(c) for c in candidates], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def read_round(round_num: int, root: str | Path = ARENA_ROOT) -> list[PromptCandidate]:
    path = round_dir(round_num, root) / "prompts.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [PromptCandidate(**c) for c in raw]


def load_predictions(path: str | Path) -> list[dict]:
    """Read a swept predictions.jsonl (rows carry `prompt_id` alongside the usual fields)."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- compliance
def hebrew_ratio(text: str) -> float:
    """Share of letters that are Hebrew (1.0 when there are no letters at all)."""
    hebrew = len(_HEBREW_RE.findall(text))
    foreign = len(_FOREIGN_LETTER_RE.findall(text))
    total = hebrew + foreign
    return 1.0 if total == 0 else hebrew / total


def count_sentences(text: str) -> int:
    return len([s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()])


def compliance_metrics(prediction: str) -> dict:
    """Per-prediction signals for the traits short summarization prompts try to control."""
    pred = prediction.strip()
    words = len(pred.split())
    sentences = count_sentences(pred)
    heb = hebrew_ratio(pred)
    return {
        "words": words,
        "sentences": sentences,
        "hebrew_ratio": round(heb, 4),
        "has_list_markers": bool(_LIST_MARKER_RE.search(pred)),
        "is_empty": not pred,
        "length_ok": MIN_WORDS <= words <= MAX_WORDS,
        "sentences_ok": 1 <= sentences <= MAX_SENTENCES,
        "hebrew_ok": heb >= MIN_HEBREW_RATIO,
    }


def compliance_score(prediction: str) -> float:
    """0-1: fraction of the four format rules a prediction obeys."""
    m = compliance_metrics(prediction)
    checks = [
        m["length_ok"],
        m["sentences_ok"],
        m["hebrew_ok"],
        not m["has_list_markers"] and not m["is_empty"],
    ]
    return sum(checks) / len(checks)


# --------------------------------------------------------------------------- scoring
def group_by_prompt(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row.get("prompt_id", "unknown"), []).append(row)
    return groups


def score_prompt(rows: list[dict], *, skip_rouge: bool = False) -> dict:
    """Aggregate one prompt's predictions: ROUGE + mean compliance signals."""
    from evaluation.evaluate import compute_rouge

    n = len(rows)
    metrics = [compliance_metrics(r["prediction"]) for r in rows]
    result = {
        "n": n,
        "compliance": round(sum(compliance_score(r["prediction"]) for r in rows) / n, 4),
        "mean_words": round(sum(m["words"] for m in metrics) / n, 1),
        "mean_sentences": round(sum(m["sentences"] for m in metrics) / n, 2),
        "hebrew_ratio": round(sum(m["hebrew_ratio"] for m in metrics) / n, 4),
        "list_marker_rate": round(sum(m["has_list_markers"] for m in metrics) / n, 4),
        "empty_rate": round(sum(m["is_empty"] for m in metrics) / n, 4),
    }
    if not skip_rouge:
        result["rouge"] = compute_rouge(rows, normalize=True)
    return result


def score_round(rows: list[dict], *, skip_rouge: bool = False) -> dict[str, dict]:
    """Score every prompt in a swept predictions file. Returns {prompt_id: metrics}."""
    return {
        pid: score_prompt(group, skip_rouge=skip_rouge)
        for pid, group in group_by_prompt(rows).items()
    }


def judge_prompts(
    rows: list[dict],
    prompt_ids: Iterable[str],
    *,
    limit: int = 30,
    seed: int = 42,
) -> dict[str, dict]:
    """Run the Gemini faithfulness/fluency judge on a fixed subset of the named prompts.

    Judging every candidate on every example is the expensive part of the loop, so the
    notebook coarse-ranks on ROUGE + compliance first and judges only the finalists.
    The same example indices are judged for each prompt, so scores are comparable.
    """
    from evaluation.evaluate import judge_with_llm, sample_for_judge

    groups = group_by_prompt(rows)
    judged: dict[str, dict] = {}
    for pid in prompt_ids:
        subset = sample_for_judge(groups[pid], limit, seed)
        report = judge_with_llm(subset)
        judged[pid] = {
            "faithfulness_mean": report["faithfulness_mean"],
            "fluency_mean": report["fluency_mean"],
            "scored": report["scored"],
        }
    return judged


def merge_judge(scores: dict[str, dict], judged: dict[str, dict]) -> dict[str, dict]:
    """Fold judge results into a scores dict (returns a new dict)."""
    merged = {pid: dict(m) for pid, m in scores.items()}
    for pid, j in judged.items():
        if pid in merged:
            merged[pid]["judge"] = j
    return merged


def rank(scores: dict[str, dict]) -> list[str]:
    """Best prompt first.

    Judge scores decide the order whenever they exist — ROUGE alone is a trap here: it
    rewards drifting toward the references' headline/digest register, the very style the
    prompts are meant to suppress. Compliance breaks ties (and ranks un-judged candidates),
    with ROUGE-L last as a weak tiebreak.
    """

    def key(pid: str):
        m = scores[pid]
        judge = m.get("judge") or {}
        faith = judge.get("faithfulness_mean")
        flu = judge.get("fluency_mean")
        judge_mean = (faith + flu) / 2 if faith is not None and flu is not None else -1.0
        rouge_l = (m.get("rouge") or {}).get("rougeL", 0.0)
        return (judge_mean, m["compliance"], rouge_l)

    return sorted(scores, key=key, reverse=True)


def leaderboard(scores: dict[str, dict], candidates: Iterable[PromptCandidate] = ()) -> str:
    """Markdown table of a round's results, best first — what the agent reads to iterate."""
    notes = {c.id: c.note for c in candidates}
    header = (
        "| # | prompt | judge (faith/flu) | compliance | words | sents | heb | lists | ROUGE-L | note |\n"
        "|---|--------|-------------------|-----------|-------|-------|-----|-------|---------|------|\n"
    )
    lines = []
    for i, pid in enumerate(rank(scores), 1):
        m = scores[pid]
        judge = m.get("judge") or {}
        faith, flu = judge.get("faithfulness_mean"), judge.get("fluency_mean")
        judge_cell = f"{faith}/{flu}" if faith is not None else "—"
        rouge_l = (m.get("rouge") or {}).get("rougeL", "—")
        lines.append(
            f"| {i} | `{pid}` | {judge_cell} | {m['compliance']:.2f} | {m['mean_words']} | "
            f"{m['mean_sentences']} | {m['hebrew_ratio']:.2f} | {m['list_marker_rate']:.2f} | "
            f"{rouge_l} | {notes.get(pid, '')} |"
        )
    return header + "\n".join(lines)


def save_scores(round_num: int, scores: dict[str, dict], root: str | Path = ARENA_ROOT) -> Path:
    path = round_dir(round_num, root) / "scores.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def show_examples(rows: list[dict], index: int = 0) -> str:
    """Render one article's reference next to every prompt's summary of it.

    The metrics rank; the text explains. This is what you read before writing the next round.
    """
    groups = group_by_prompt(rows)
    first = next(iter(groups.values()))[index]
    out = [
        f"ARTICLE: {first['text'][:300]}...",
        "",
        f"REFERENCE: {first['reference']}",
        "",
    ]
    for pid, group in groups.items():
        pred = group[index]["prediction"]
        m = compliance_metrics(pred)
        out.append(
            f"--- {pid}  ({m['words']}w, {m['sentences']}s, heb={m['hebrew_ratio']:.2f}, "
            f"lists={m['has_list_markers']})"
        )
        out.append(pred)
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- CLI
def main() -> None:
    """The local half of one loop iteration: --write a round, then --score it once it has run."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Prompt-optimization arena: write a round's candidates, or score a swept round"
    )
    parser.add_argument("--round", type=int, required=True, dest="round_num")
    parser.add_argument("--write", action="store_true",
                        help="Write this round's candidates (from prompt_rounds.py) to prompts.json")
    parser.add_argument("--score", action="store_true",
                        help="Score the downloaded predictions.jsonl and print the leaderboard")
    parser.add_argument("--smoke", action="store_true",
                        help="With --write: use the 2-candidate smoke slice")
    parser.add_argument("--judge", action="store_true",
                        help="With --score: also run the Gemini judge on the top finalists")
    parser.add_argument("--judge-limit", type=int, default=30,
                        help="Examples per finalist for the judge (same examples for each)")
    parser.add_argument("--top-k", type=int, default=3, help="How many finalists to judge")
    parser.add_argument("--show", type=int, default=-1,
                        help="With --score: also print every prompt's summary of example N")
    args = parser.parse_args()

    if args.write:
        from evaluation.prompt_rounds import get_round

        candidates = get_round(args.round_num, smoke=args.smoke)
        path = write_round(args.round_num, candidates)
        print(f"{len(candidates)} candidates -> {path}")
        for c in candidates:
            print(f"  {c.id:16s} {len(c.template.split()):3d} words — {c.note}")
        return

    if args.score:
        preds = round_dir(args.round_num) / "predictions.jsonl"
        if not preds.exists():
            raise SystemExit(
                f"{preds} not found — download it first:\n"
                f"  python -m evaluation.prompt_sweep_hf_job --download --round {args.round_num}"
            )
        rows = load_predictions(preds)
        candidates = read_round(args.round_num)
        scores = score_round(rows)

        if args.judge:
            finalists = rank(scores)[: args.top_k]
            print(f"Judging {finalists} on {args.judge_limit} examples each...")
            scores = merge_judge(scores, judge_prompts(rows, finalists, limit=args.judge_limit))

        save_scores(args.round_num, scores)
        print()
        print(leaderboard(scores, candidates))
        print(f"\nWINNER (round {args.round_num}): {rank(scores)[0]}")
        if args.show >= 0:
            print()
            print(show_examples(rows, args.show))
        return

    raise SystemExit("nothing to do: pass --write or --score")


if __name__ == "__main__":
    main()
