"""
Builds a *distilled* training set: same curated articles, but the targets are Gemini's
summaries instead of the HeSum headline references. Sits parallel to `data/preprocess.py`
in pipeline step 2 and feeds the same training contract (`prompt`/`completion` Arrow splits),
so `training/train.py --dataset-repo …` can train on it unchanged. Motivation lives in
`docs/training-improvement-notebook.md`: the judge scores summaries against the *article*,
and HeSum headlines score 2.9 on that axis while Gemini scores 4.9 — so the reference style,
not the optimizer, is what fine-tuning was failing on.

Execution environment: local machine, API + CPU only (no model load). Needs GEMINI_API_KEY.
Run: python -m data.distill --n-train 1500 --n-val 100 --push avreymi/amlk-distill-data
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluation.gemini_client import GEMINI_MODEL, GEMINI_TIMEOUT, call_with_retry
from evaluation.prompt_arena import compliance_metrics

# Teacher decoding is a measurement-like step, not a creative one: pinned so a rebuild of the
# dataset reproduces the same targets.
TEACHER_GENERATION_CONFIG = {"temperature": 0.0, "max_output_tokens": 256}
# The decode-time Hebrew constraint bans Latin/Cyrillic/Greek/Arabic/CJK tokens, so a target
# containing them is one the student is structurally forbidden to reproduce.


def has_foreign_script(text: str) -> bool:
    """True if the text carries characters the Hebrew decode constraint bans at generation."""
    for ch in text:
        o = ord(ch)
        if (0x41 <= o <= 0x5A) or (0x61 <= o <= 0x7A):  # Latin
            return True
        if 0x0400 <= o <= 0x04FF or 0x0370 <= o <= 0x03FF:  # Cyrillic / Greek
            return True
        if 0x0600 <= o <= 0x06FF:  # Arabic
            return True
        if 0x3000 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF:  # CJK / Hangul
            return True
    return False


def is_usable_target(summary: str, max_words: int = 0, max_sentences: int = 0) -> bool:
    """Keep only teacher summaries that obey every format rule the prompt asks for.

    The arena's `compliance_metrics` bounds (6-45 words, 1-2 sentences) describe the *zero-shot*
    prompt. A distillation run with a longer budget must widen them, or the filter silently
    deletes exactly the targets the run exists to test.
    """
    s = (summary or "").strip()
    if not s or has_foreign_script(s):
        return False
    m = compliance_metrics(s)
    length_ok = (6 <= m["words"] <= max_words) if max_words else m["length_ok"]
    sentences_ok = (1 <= m["sentences"] <= max_sentences) if max_sentences else m["sentences_ok"]
    return length_ok and sentences_ok and m["hebrew_ok"] and not m["has_list_markers"]


def build_template(words: int, sentences: int) -> str:
    """A length-variant of `data.prompts.PROMPT_TEMPLATE`, keeping its winning stop cue.

    The 15-word/one-sentence prompt that won the prompt arena is a *zero-shot* winner. A
    fine-tuned student at that budget produces true-but-generic summaries with the identifying
    specifics compressed out, which the judge scores as unsupported (notebook entry #5). This
    builds the same instruction with more room and an explicit ask for the identifying facts.
    Used for both the teacher's targets and the student's prompt, so the two never disagree.
    """
    if sentences == 1:
        count_he, tail = "במשפט קצר אחד", "משפט אחד"
    else:
        count_he, tail = f"ב-{sentences} משפטים קצרים", f"{sentences} משפטים"
    return (
        f"סכם את כתבת החדשות הבאה בעברית {count_he}, לא יותר מ-{words} מילים. "
        f"כלול את הפרטים המזהים המרכזיים: מי, מה והיכן. "
        f"כתוב {tail} בלבד ועצור מיד בסופם.\n\n"
        "{text}\n\n"
        f"תקציר (עד {tail}, עד {words} מילים):"
    )


def teach(prompts: list[str], workers: int = 8) -> list[str]:
    """Ask the teacher for a summary of every prompt, in order.

    The teacher sees the *student's* prompt verbatim (the promoted PROMPT_TEMPLATE, already
    baked into the `prompt` column) — a target produced from a different instruction would be
    a style the student cannot reach from its own prompt at inference.
    """
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(GEMINI_MODEL)

    def one(prompt: str) -> str:
        try:
            return call_with_retry(lambda: model.generate_content(
                prompt, generation_config=TEACHER_GENERATION_CONFIG,
                request_options={"timeout": GEMINI_TIMEOUT}).text).strip()
        except Exception as exc:  # noqa: BLE001 — a dropped row is cheaper than a dead run
            print(f"  teacher failed ({exc.__class__.__name__}); dropping row")
            return ""

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, prompts))


def build_split(split, n: int, workers: int = 8, template: str = "",
                max_words: int = 0, max_sentences: int = 0):
    """Return a distilled split: same articles, teacher summaries as completions.

    With `template`, the prompt column is rebuilt from the (already truncated) `text` column so
    teacher and student share one instruction; without it the stored prompt is used verbatim.
    """
    from datasets import Dataset

    rows = split.select(range(min(n, len(split))))
    prompts = ([template.format(text=t) for t in rows["text"]] if template
               else list(rows["prompt"]))
    summaries = teach(prompts, workers=workers)
    kept = [
        {
            "text": rows[i]["text"],
            "summary": s,
            "source": "gemini-distill",
            "prompt": prompts[i],
            "completion": s,
            "reference_hesum": rows[i]["summary"],
        }
        for i, s in enumerate(summaries)
        if is_usable_target(s, max_words, max_sentences)
    ]
    print(f"  kept {len(kept)}/{len(summaries)} teacher targets "
          f"({len(summaries) - len(kept)} dropped by the format/script filter)")
    return Dataset.from_list(kept)


def verify_split(split, workers: int = 8, min_faith: int = 5, min_fluency: int = 4):
    """Keep only targets the judge itself rates as fully faithful to their article.

    The format filter above checks shape, not truth: a teacher summary can be well-formed and
    still slip a wrong number. Notebook entry #9 showed the student's remaining gap lives in a
    ~10% tail of bad outputs, not in a broad deficit — so filtering the targets by the *same*
    judge that scores the student attacks exactly that tail. This is the quality half of
    distillation; `is_usable_target` is the format half.
    """
    from datasets import Dataset

    from evaluation.improve_eval import judge_file

    rows = [{"text": t, "prediction": c} for t, c in zip(split["text"], split["completion"])]
    scores = judge_file(rows, workers=workers)["per_example"]
    kept_idx = [
        i for i, s in enumerate(scores)
        if isinstance(s["faithfulness"], (int, float)) and s["faithfulness"] >= min_faith
        and isinstance(s["fluency"], (int, float)) and s["fluency"] >= min_fluency
    ]
    print(f"  judge-verified {len(kept_idx)}/{len(rows)} targets "
          f"(faithfulness >= {min_faith}, fluency >= {min_fluency})")
    return Dataset.from_list([split[i] for i in kept_idx])


def verify_existing(src: Path, out: Path, workers: int = 8):
    """Judge-verify an already-built distilled dataset and write the filtered copy."""
    from datasets import load_from_disk

    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name in ("train", "val"):
        print(f"Judge-verifying {name}...")
        split = load_from_disk(str(src / name))
        kept = verify_split(split, workers=workers)
        kept.save_to_disk(str(out / name))
        stats[name] = {"kept": len(kept), "of": len(split)}
    load_from_disk(str(src / "test")).save_to_disk(str(out / "test"))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def build_preference_pairs(src: Path, verified: Path, out: Path, workers: int = 8):
    """Build DPO pairs from the teacher's own hits and misses on the SAME article.

    SFT teaches "produce this"; preference optimization teaches "this, NOT that". The pairs have
    to isolate faithfulness, so both sides must be the same teacher, same article, same format —
    differing only in whether the judge called them faithful. Source: the ~22% of targets that
    failed judge verification become `rejected`; the teacher is resampled at temperature 1.0 on
    those same articles and a resample that *passes* becomes `chosen`. Articles where the resample
    also fails are dropped — a pair needs a genuinely good side.

    Execution environment: local, API only.
    """
    from datasets import Dataset, load_from_disk

    from evaluation.improve_eval import judge_file

    full = load_from_disk(str(src / "train"))
    kept = set(load_from_disk(str(verified / "train"))["completion"])
    failed = [r for r in full if r["completion"] not in kept]
    print(f"{len(failed)} targets failed verification — resampling the teacher on those articles")

    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(GEMINI_MODEL)
    # Temperature 1.0, unlike every other teacher call in this file: a temperature-0 resample
    # would reproduce the same failed summary and yield no pair at all.
    cfg = {**TEACHER_GENERATION_CONFIG, "temperature": 1.0}

    def one(prompt: str) -> str:
        try:
            return call_with_retry(lambda: model.generate_content(
                prompt, generation_config=cfg,
                request_options={"timeout": GEMINI_TIMEOUT}).text).strip()
        except Exception:  # noqa: BLE001
            return ""

    with ThreadPoolExecutor(max_workers=workers) as pool:
        resampled = list(pool.map(one, [r["prompt"] for r in failed]))

    candidates = [
        (r, s) for r, s in zip(failed, resampled)
        if s and s != r["completion"] and is_usable_target(s, 45, 3)
    ]
    print(f"{len(candidates)} resamples are well-formed and different; judging them")
    scores = judge_file(
        [{"text": r["text"], "prediction": s} for r, s in candidates], workers=workers,
    )["per_example"]

    pairs = [
        {
            "prompt": r["prompt"],
            "chosen": s,
            "rejected": r["completion"],
            "text": r["text"],
        }
        for (r, s), sc in zip(candidates, scores)
        if isinstance(sc["faithfulness"], (int, float)) and sc["faithfulness"] >= 5
    ]
    print(f"{len(pairs)} preference pairs (chosen passed the judge, rejected did not)")
    out.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(pairs)
    ds.save_to_disk(str(out / "train"))
    (out / "stats.json").write_text(json.dumps({"pairs": len(pairs), "failed": len(failed)}, indent=2))
    return ds


def main():
    parser = argparse.ArgumentParser(description="Build a Gemini-distilled training set")
    parser.add_argument("--processed", default="outputs/data/processed/whole",
                        help="Source splits built by data.preprocess")
    parser.add_argument("--out", default="outputs/data/distill/whole")
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--push", default="", help="Hub dataset repo to push to (must be new)")
    parser.add_argument("--target-words", type=int, default=0,
                        help="Rebuild prompts with this word budget (0 = keep the stored prompt)")
    parser.add_argument("--target-sentences", type=int, default=2,
                        help="Sentence budget for --target-words")
    parser.add_argument("--verify-existing", default="",
                        help="Judge-verify an already-built distilled dir into --out (no teacher calls)")
    parser.add_argument("--build-pairs", default="",
                        help="Build DPO preference pairs from a distilled dir (needs --verified)")
    parser.add_argument("--verified", default="",
                        help="The judge-verified copy of --build-pairs' dir")
    args = parser.parse_args()

    from datasets import load_from_disk

    if args.build_pairs:
        ds = build_preference_pairs(
            Path(args.build_pairs), Path(args.verified), Path(args.out), args.workers,
        )
        print("sample pair:", {k: ds[0][k][:120] for k in ("chosen", "rejected")})
        if args.push:
            from huggingface_hub import HfApi

            api = HfApi(token=os.environ["HF_TOKEN"])
            api.create_repo(repo_id=args.push, repo_type="dataset", private=True, exist_ok=True)
            api.upload_folder(folder_path=args.out, repo_id=args.push, repo_type="dataset")
            print(f"Pushed to https://huggingface.co/datasets/{args.push}")
        return

    if args.verify_existing:
        verify_existing(Path(args.verify_existing), Path(args.out), args.workers)
        if args.push:
            from huggingface_hub import HfApi

            api = HfApi(token=os.environ["HF_TOKEN"])
            api.create_repo(repo_id=args.push, repo_type="dataset", private=True, exist_ok=True)
            api.upload_folder(folder_path=args.out, repo_id=args.push, repo_type="dataset")
            print(f"Pushed to https://huggingface.co/datasets/{args.push}")
        return

    src = Path(args.processed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    template = build_template(args.target_words, args.target_sentences) if args.target_words else ""
    if template:
        print(f"Prompt rebuilt for a {args.target_words}-word / {args.target_sentences}-sentence "
              f"budget (teacher and student share it)")

    print(f"Distilling train ({args.n_train}) with {GEMINI_MODEL}...")
    keep_words = int(args.target_words * 1.3) if args.target_words else 0
    keep_sentences = args.target_sentences + 1 if args.target_words else 0
    train = build_split(load_from_disk(str(src / "train")), args.n_train, args.workers, template,
                        keep_words, keep_sentences)
    print(f"Distilling val ({args.n_val})...")
    val = build_split(load_from_disk(str(src / "val")), args.n_val, args.workers, template,
                      keep_words, keep_sentences)
    # The test split keeps its articles and their order — that is what makes every arm's judged
    # subset the same 120 articles — but its prompt column follows the same template as training,
    # since the job generates from that column.
    test = load_from_disk(str(src / "test"))
    if template:
        test = test.map(lambda ex: {**ex, "prompt": template.format(text=ex["text"])})

    train.save_to_disk(str(out / "train"))
    val.save_to_disk(str(out / "val"))
    test.save_to_disk(str(out / "test"))
    stats = {"train": len(train), "val": len(val), "test": len(test),
             "teacher": GEMINI_MODEL, "n_train_requested": args.n_train}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("sample target:", train[0]["completion"])

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ["HF_TOKEN"])
        api.create_repo(repo_id=args.push, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.push, repo_type="dataset")
        print(f"Pushed to https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
