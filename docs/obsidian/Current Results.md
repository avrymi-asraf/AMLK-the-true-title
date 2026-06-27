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

## Baselines not fully scored

- **Gemini advanced baseline**: `outputs/results/gemini-whole.jsonl` partially generated; run stopped on GCP billing 403 (`gemini-predict.log`). #status/blocker

## Hub artifacts

- Adapter: `avreymi/amlk-qwen3-2b-sft`
- Dataset: `avreymi/amlk-training-data`
- wandb: `amlk-hebrew-summarization`

Related: [[Fix Plan]], [[Evaluation Metrics]]
