# Training Objective

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

## Gap vs HeSum mLongT5

| Practice | AMLK now | HeSum mLongT5 |
|----------|----------|---------------|
| Early stopping / best-checkpoint selection | Yes — `load_best_model_at_end` on `eval_loss` (cheap analogue) | Yes, on **ROUGE-1** |
| Generate during training | No | Yes (seq2seq eval) |

Avoid ROUGE-based early stopping on HF Jobs unless `eval_loss` selection proves insufficient —
generation on val every N steps is expensive.

## Why loss and judge can diverge

Optimizing token likelihood does not teach:
- when to emit EOS
- how long to write vs. reference length
- anti-repetition at decode time

Watch for this gap once real training runs and judge scores exist — it's why the decode config
(`no_repeat_ngram_size`, `repetition_penalty`, explicit EOS) and `load_best_model_at_end` are
already wired into `training/train_hf_job.py` rather than left for a later fix pass.

Related: [[HeSum Paper Insights#ROUGE vs human eval]]
