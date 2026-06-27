# Current Results

#status/done

First full evaluation on the **whole** variant test set (n=1000). Predictions from HF Jobs training run; scored locally.

## Scores (saved reports)

| System | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 (xlm-roberta) | Judge faithfulness | Judge fluency |
|--------|---------|---------|---------|---------------------------|-------------------|---------------|
| **Finetuned** | 11.4 | 4.5 | 9.2 | 0.85 | **1.86** | **3.10** |
| **Base (zero-shot)** | 6.8 | 2.5 | 5.6 | 0.83 | 2.64 | 3.92 |
| Reference length | — | — | — | — | — | med. **25 words** |

Files:
- `outputs/results/finetuned.report.json`
- `outputs/results/zero-shot.report.json`
- `outputs/results/predictions-finetuned.jsonl`
- `outputs/results/predictions-base.jsonl`

Judge: `meta-llama/Meta-Llama-3-8B-Instruct` via HF Inference (100-sample subset).

## Interpretation

- Fine-tuning **raises ROUGE** vs base but **lowers judge faithfulness** — the model looks more like the reference on n-grams while being *less* faithful when read carefully.
- See [[Prediction Failure Modes]] for why (repetition loops, over-length, truncation).
- Compare to [[HeSum Paper Insights#HeSum SOTA (Table 3)]] — our ROUGE-1 11.4 is not far from GPT-4 13.6 on the same dataset family.

## Phase 1 — decoding-only re-decode (2026-06-27)

Re-decoded the **same v1 adapter** (job `6a3f8e18…`, `--inference-only --pred-suffix -v2`) with the
new anti-degeneration config (`no_repeat_ngram_size=3`, `repetition_penalty=1.2`, `min_new_tokens=16`,
explicit EOS), keeping the v1 prompt. Both files re-scored with the **new** metrics (AlephBERT
BERTScore, raw + Hebrew-normalized ROUGE):

| Finetuned | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 (AlephBERT) | median pred chars |
|-----------|---------|---------|---------|--------------------------|-------------------|
| **v1** (old greedy) | 11.4 | 4.5 | 9.2 | **0.449** | 513 |
| **v2** (new decode) | 4.7 | 0.2 | 3.7 | **0.383** | 567 |

**Finding — decoding alone is NOT enough (answers the [[Fix Plan]] gate "Decoding fix enough? → No").**
The decode controls worked *mechanically* (the repetition loops are gone), but every overlap/semantic
metric **dropped**. Reading the outputs: v1 looped a template phrase whose newspaper names accidentally
overlapped the references (inflating ROUGE); v2, forbidden from repeating, produces fluent-looking but
**hallucinated gibberish** (garbled coined words) and still runs to ~256 tokens (the model never emits
EOS). So the v1 repetition was masking a fundamentally **undertrained** adapter (1 epoch, attention-only
LoRA on 6/24 layers). → Proceed to **Phase 2 retrain** (3 epochs, +MLP LoRA on all 24 layers, length-cap
prompt). The decode config stays — it only pays off on a properly trained model.
Note: AlephBERT BERTScore (0.45) is far more discriminative than the old xlm-r number (0.85, ~constant
regardless of quality) — validates the [[Evaluation Metrics|Phase 0 backbone switch]].
Report: `outputs/results/finetuned-v2.report.json`. (base-v2 not generated — the job's 2h budget covered
finetuned; base re-decode was deprioritized since the decoding question is about the finetuned model.)

## Baselines not fully scored

- **Gemini advanced baseline**: `outputs/results/gemini-whole.jsonl` partially generated; run stopped on GCP billing 403 (`gemini-predict.log`). #status/blocker

## Hub artifacts

- Adapter: `avreymi/amlk-qwen3-2b-sft`
- Dataset: `avreymi/amlk-training-data`
- wandb: `amlk-hebrew-summarization`

Related: [[Fix Plan]], [[Evaluation Metrics]]
