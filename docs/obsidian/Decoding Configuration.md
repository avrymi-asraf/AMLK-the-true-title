# Decoding Configuration

#status/current

> **SUPERSEDED recommendation (2026-07):** the "planned decode config" below that prescribed
> `repetition_penalty=1.2` + `no_repeat_ngram_size=3` is **wrong for summarization** and was
> measured to *hurt* faithfulness/fluency more than any training change this project made.
>
> On a fixed 120-article judged subset, turning those two knobs **off** (`1.0` / `0`):
>
> | arm | Δ faithfulness | Δ fluency |
> |---|---|---|
> | zero-shot base | **+1.54** | **+0.78** |
> | fine-tuned adapter | **+1.39** | **+0.78** |
>
> Judge-free confirmation: fraction of output words present in the source article went
> **0.204 → 0.685** (paired Δ +0.481). HF applies `repetition_penalty` over the **whole
> sequence, prompt included**; with a ~3,800-token Hebrew article in context, `1.2`
> suppresses the article's own vocabulary at every decode step (misspelled entities like
> `שיוני האקחים` for `שינויי האקלים`). In summarization, copying the input is *correct*.
>
> **Current defaults** (in `training/train_hf_job.py` and `evaluation/infer.py`):
>
> ```python
> max_new_tokens=128          # DEFAULT_MAX_NEW_TOKENS
> min_new_tokens=16
> do_sample=False             # greedy
> repetition_penalty=1.0      # OFF
> no_repeat_ngram_size=0      # OFF
> eos_token_id=tokenizer.eos_token_id
> pad_token_id=tokenizer.pad_token_id
> bad_words_ids=...           # Hebrew-script constraint (always on)
> ```
>
> Degeneration (the original problem) is handled by `min_new_tokens`, explicit EOS,
> the `max_new_tokens` cap, and the instruction-formatted prompt with a stop cue — not by
> penalizing article vocabulary. Env / CLI overrides still exist for A/B re-decode.
>
> See `docs/e4-raw-vs-curated-training-plan.md` §1.1.

---

## Historical (Qwen era, 2026-06) — do not re-apply

> The anti-degeneration settings here reacted to real greedy loops on Qwen3-2B with a
> pre-stop-cue prompt. They are **not** the current DictaLM2 decode config.

Current generation in `training/train_hf_job.py` (post-training inference) **before the fix**:

```python
trained_model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
)
```

No `repetition_penalty`, no `no_repeat_ngram_size`, no explicit `eos_token_id`.

## Why the old config hurt (measured)

| Setting | Effect |
|---------|--------|
| `do_sample=False` (greedy) | Deterministic; prone to loops when EOS isn’t confident |
| `max_new_tokens=128` | ~4× longer than typical ref (~25 words) without a stop cue |
| `repetition_penalty=1.2` | Suppresses article vocabulary → broken entity names |
| `no_repeat_ngram_size=3` | Blocks legitimate multi-token Hebrew entity repeats |

## Old "planned" decode config — SUPERSEDED (do not use)

```python
max_new_tokens=100
min_new_tokens=16
no_repeat_ngram_size=3   # SUPERSEDED → 0
repetition_penalty=1.2   # SUPERSEDED → 1.0
do_sample=False
eos_token_id=tokenizer.eos_token_id
pad_token_id=tokenizer.pad_token_id
```

## Base model: Qwen3 thinking leakage (historical)

Zero-shot base emits `<think>...</think>` and sometimes English.

**Mitigation:** `strip_think()` before scoring; Hebrew-script `bad_words_ids` at generate.
DictaLM2 path does not use `/no_think`.

Related: [[Prediction Failure Modes]], [[Fix Plan]], `docs/e4-raw-vs-curated-training-plan.md`
