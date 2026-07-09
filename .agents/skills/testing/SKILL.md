---
name: testing
description: Testing philosophy for AMLK — behavior tests only, no implementation dictation, heavy tests (model load / live API) opt-in.
---

# Testing Philosophy

## The Rule

Automated tests answer one cheap question: **is everything still wired and does the
pipeline run after I changed the code?** They do not dictate *how* the code works, and they
are not where you judge summary quality.

The real check is **looking at actual outputs** — read the Hebrew summaries in
`predictions.jsonl`, the metric reports in `outputs/results/`, and the wandb curves. A unit
test cannot tell you whether a summary is faithful or fluent; do not try to make it.

> Keep the suite very small and fast. A few comprehensive "it's connected / it runs" tests
> beat a pile of tiny tests pinning down details nobody cares about. If a test would break on
> a harmless refactor that keeps the same behavior, it is the wrong test — delete it.
> Default `pytest tests/` must finish in seconds on any machine, even with no GPU or API key.

## What to test (a few behavioral contracts, total)

- `data/download.py`: normalizers map raw rows to `{text, summary, source}` and skip empties.
- `data/preprocess.py`: `build_prompt` carries the task + article; `make_variant` makes
  whole/lead/body that actually differ; the split is a clean 80/10/10 with no overlap.
- `evaluation/evaluate.py`: ROUGE scores Hebrew non-zero (guards the tokenizer-strips-non-ASCII
  bug); a messy LLM reply parses into scores.
- `evaluation/error_analysis.py`: failure-rate aggregation counts each type correctly.

## What NOT to test

- Exact prompt wording, internal column names, or JSON report field order.
- CLI argument structure, config/preset values (those are config, not behavior).
- Script source contents (`assert "SFTTrainer" in script` — never).
- Anything that needs the model to actually train well (that's eyeballing outputs, not pytest).

## Heavy tests (model load / live API) — always opt-in

Any test that loads the base model or calls Gemini **must** be gated:

```python
@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") and os.getenv("RUN_LIVE_TESTS")),
    reason="Set GEMINI_API_KEY and RUN_LIVE_TESTS=1 to run live LLM tests",
)
def test_live_gemini_judge_parses_scores(): ...
```

If the default suite ever needs a GPU, an API key, or more than a few seconds, a guard is missing.

## Test files (split by area, so a change in one part runs its own file)

- `tests/test_download.py` — dataset normalization.
- `tests/test_preprocess.py` — prompt building, probe variants, splitting.
- `tests/test_evaluation.py` — ROUGE/Hebrew, judge-reply parsing, failure rates; live Gemini
  test gated behind `RUN_LIVE_TESTS`.

Run: `source .venv/bin/activate && python -m pytest tests/ -v`.
