# Training Objective

#status/superseded

> **Qwen era (2026-06). Superseded — see [[Project Pivot]].**
> The core observation holds and is now load-bearing for the whole project: the model minimises
> cross-entropy against its targets, so a noisy target teaches noise faithfully. That is the argument
> for auditing the references at all. Training itself is now owned externally.

What the model is **trained on** vs what we **evaluate**.

## Trained on: cross-entropy (completion-only)

`training/train.py` and `training/train_hf_job.py` use:

```python
completion_only_loss=True
```

- **Input:** `prompt` (English instruction + Hebrew article) + `completion` (gold summary)
- **Loss:** next-token cross-entropy **only on summary tokens** (article/instruction masked)
- **Goal:** maximize likelihood of reproducing reference summary token-by-token

The model is **not** trained on ROUGE, BERTScore, or judge scores.

## Monitored during training

- `train/loss` — training CE loss
- `eval/loss` — same CE loss on validation split (200-example slice on HF job)

Logged to wandb; `eval/loss` summary = min.

## What we do NOT do (gap vs HeSum mLongT5)

| Practice | AMLK now | HeSum mLongT5 |
|----------|----------|---------------|
| Early stopping | No | Yes, on **ROUGE-1** |
| `load_best_model_at_end` | No | Implicit via early stop |
| Generate during training | No | Yes (seq2seq eval) |
| Checkpoint selection metric | Last epoch wins | Best ROUGE-1 |

HF job hardcodes **`num_train_epochs=1`** despite `TrainingConfig.num_train_epochs=3` in `training/config.py`.

## Recommended addition (team consensus)

Keep CE loss for SFT, but add:

```python
load_best_model_at_end=True
metric_for_best_model="eval_loss"
greater_is_better=False
```

Cheap (loss already computed). Avoid ROUGE-based early stopping on HF Jobs unless decode + retrain aren’t enough — generation on val every N steps is expensive.

## Why loss and judge diverge

Optimizing token likelihood does not teach:
- when to emit EOS
- how long to write (refs ~25 words, preds ~89 words)

**Decode note (2026-07-30):** do **not** use `repetition_penalty=1.2` /
`no_repeat_ngram_size=3` as the fix — HF applies the penalty over the whole prompt and it
hurts faithfulness (~+1.4 when turned **off**). Current defaults are `1.0` / `0`. Degeneration
is handled by `min_new_tokens`, explicit EOS, `max_new_tokens`, and the stop-cue prompt.
See [[Decoding Configuration]] and `docs/e4-raw-vs-curated-training-plan.md` §1.1.

Hence high ROUGE + low faithfulness can still appear from target quality / prompt budget,
not from missing decode penalties. See [[Prediction Failure Modes]].

Related: [[Fix Plan#Phase 2]], [[HeSum Paper Insights#ROUGE vs human eval]]
