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

## Failure-type clustering (2026-07-04)

Gemini billing unblocked — ran `evaluation.error_analysis` (Gemini `gemini-2.5-flash-lite`, the literature
taxonomy: hallucination / omission / entity-or-number error / lead-copying / fluency) on 100-sample subsets
of both v3 systems. Clustered by exact label combination (co-occurrence), plus a post-hoc `wrong_language`
tag (no Hebrew in first 100 chars) since the taxonomy has no label for a non-Hebrew/non-answer response.

**Marginal failure rates (fraction of 100-sample exhibiting each type):**

| Failure type | v3 Finetuned | v3 Base |
|---|---|---|
| hallucination | **0.67** | 0.28 |
| omission | **0.71** | 0.22 |
| entity_or_number_error | **0.67** | 0.39 |
| lead_copying | 0.26 | 0.04 |
| fluency_problem | 0.17 | 0.03 |
| wrong_language (post-hoc) | 0.00 | **0.97** |

**Top co-occurrence clusters:**
- Finetuned: `entity_error + hallucination + omission` (20%), `hallucination + omission` (15%),
  `entity_error + omission` (10%) — the model reliably produces fluent, correctly-formatted Hebrew
  but gets the facts wrong on ~2/3 of examples. Near-zero `wrong_language`.
- Base: `wrong_language` alone (51%) — the model's `<think>` block correctly restates the article in
  **English** but never emits a Hebrew answer, so it triggers **none** of the literature failure labels
  (it isn't hallucinating or omitting — it just isn't answering in Hebrew). Only when Hebrew *is* produced
  do entity/hallucination errors show up (rates below the finetuned model: 0.39 vs 0.67, 0.28 vs 0.67).

**Key takeaway:** the two systems fail in orthogonal ways. Base fails at the *task* (doesn't produce a Hebrew
summary 97% of the time) but is accurate *when it does* (lower entity/hallucination rates on the Hebrew subset).
Finetuned always attempts the task in the right format/language but hallucinates facts on ~2/3 of examples —
confirms the earlier judge finding (base faithfulness 2.99 > finetuned 1.95) is not a judge artifact: fine-tuning
traded task-compliance for factual grounding. This is the central finding for the paper's error-analysis section.
Reports: `outputs/results/finetuned-v3.errors.json`, `outputs/results/base-v3.errors.json`.

## Phase 2 — 3-epoch retrain, full LoRA (2026-06-28)

Full QLoRA retrain (job `6a3fa247…`, 3 epochs, r=32/alpha=64, MLP+attn modules, "up to 3 sentences" prompt).
Job ran to epoch 3.0 before the 6h timeout; final checkpoint was automatically selected via `load_best_model_at_end`
(best eval_loss 1.712 vs v1's 1.777). Both prediction files pushed to Hub; scored locally with AlephBERT BERTScore.

| Finetuned | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 (AlephBERT) | median pred chars |
|-----------|---------|---------|---------|--------------------------|-------------------|
| **v1** (1ep, attn-only LoRA, old greedy) | 11.4 | 4.5 | 9.2 | **0.449** | 513 |
| **v2** (v1 adapter, new decode) | 4.7 | 0.2 | 3.7 | **0.383** | 567 |
| **v3** (3ep, full LoRA, new decode) | **5.1** | **0.3** | **4.0** | **0.390** | 548 |

**Finding — v3 generates fluent Hebrew in the correct output format but halluccinates content.**
Qualitative inspection: no more repetition loops; produces the correct "|"-separated headline style;
but entities, facts, and events are wrong. The model learned the *style* of summarization from 3 epochs
but not faithfulness to the source article. ROUGE is low because there is no lexical overlap (no lead-copying).
AlephBERT BERTScore (0.39) is slightly above v2 (0.38) — marginal semantic improvement.
LLM judge (faithfulness/fluency) is the right instrument here — **BLOCKED on Gemini API billing**.
Report: `outputs/results/finetuned-v3.report.json`.

Note on BERTScore scale: v1 used `xlm-roberta-large` (scores cluster ~0.85 regardless of quality);
v2/v3 use `onlplab/alephbert-base` (more discriminative, range ~0.35–0.50). NOT directly comparable.

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
