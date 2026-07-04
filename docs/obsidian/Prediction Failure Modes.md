# Prediction Failure Modes

> **v3 curated examples (full article / prediction / reference):** [failure-examples.md](../failure-examples.md)

Analysis of all **1000** fine-tuned predictions (`outputs/results/predictions-finetuned.jsonl`), June 2026.

## Bucket summary

| Category | Count | % |
|----------|-------|---|
| Degenerate repetition loop | 460 | 46% |
| Too long + cut off mid-sentence | 451 | 45% |
| Overlong, no obvious loop | 14 | 1% |
| **Clean** | **75** | **7.5%** |

## Mistake 1: Never stops (length)

- Median prediction: **89 words** / 513 chars
- Median reference: **25 words** / 147 chars
- **893/1000** predictions are >2× reference length
- **935/1000** do not end with `.`, `!`, or `?` — hit `max_new_tokens=128` and get chopped

Root cause: greedy decode + no repetition control + weak EOS learning. See [[Decoding Configuration]].

## Mistake 2: Repetition loops (~46%)

Examples cycle newspaper names with one frozen predicate:

```
"ישראל היום" מתעלם מהטייקון | "ידיעות אחרונות" מתעלם מההתעלמות |
"מעריב" מתעלם מההתעלמות | ... (repeats until token cap)
```

Stats:
- 499/1000: same 3-gram repeated ≥3 times
- 396/1000: same 3-gram repeated ≥5 times
- 440/1000: distinct-word ratio < 0.5

## Mistake 3: Pipe-headline template overfit

HeSum references often use ` | ` between headline clauses (254/1000 refs). The model learned the format but **loops it**:
- 307 preds contain ` | `
- 293 have ≥3 pipes (headline-list degeneration)

## Mistake 4: Lead copying (~14%)

143/1000 predictions have >60% word overlap with the article’s first ~300 characters — extractive, not abstractive. Relevant for [[Lead Bias Probe]].

## What fine-tuning fixed

- **0/1000** `<think>` leakage (base model has this problem)
- **1/1000** majority-English predictions
- Hebrew fluency is generally OK when the output isn’t looping

## Base model failures (for comparison)

`predictions-base.jsonl`:
- Leaks Qwen3 **thinking blocks** and sometimes summarizes in **English**
- Longer median output (733 chars) but judge scores *better* than finetuned — contaminated baseline, not a fair “good zero-shot Hebrew”

Related: [[Fix Plan#Phase 1]], [[Current Results]]
