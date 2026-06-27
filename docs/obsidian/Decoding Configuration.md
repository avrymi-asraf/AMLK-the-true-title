# Decoding Configuration

Current generation in `training/train_hf_job.py` (post-training inference):

```python
trained_model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
)
```

No `repetition_penalty`, no `no_repeat_ngram_size`, no explicit `eos_token_id`.

## Why this hurts

| Setting | Effect |
|---------|--------|
| `do_sample=False` (greedy) | Deterministic; prone to loops when EOS isn’t confident |
| `max_new_tokens=128` | ~4× longer than typical ref (~25 words); 935/1000 preds truncated mid-sentence |
| No repetition control | 46% degenerate 3-gram loops |
| No EOS emphasis at train | Model doesn’t learn to stop cleanly |

## Planned decode config (Phase 1 re-decode + Phase 2 train)

```python
max_new_tokens=100
min_new_tokens=16
no_repeat_ngram_size=3
repetition_penalty=1.2
do_sample=False
eos_token_id=tokenizer.eos_token_id
pad_token_id=tokenizer.pad_token_id
```

**Phase 1:** apply to **existing** adapter with **old prompt** (adapter was trained on it).  
**Phase 2:** same settings baked into `train_hf_job.py` prediction step.

## Base model: Qwen3 thinking leakage

Zero-shot base emits `<think>...</think>` and sometimes English.

**Mitigation (re-decode script):** strip thinking blocks from decoded text before saving predictions — does not require chat template change if we post-process.

Alternative (not chosen): Qwen3 chat template with `enable_thinking=False`.

## Prompt (unchanged for Phase 1)

`data/prompts.py` — keep for re-decode; Phase 2 adds “up to 3 sentences.”

Related: [[Prediction Failure Modes]], [[Fix Plan]]
