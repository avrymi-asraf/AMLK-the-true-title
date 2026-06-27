# Evaluation Metrics

How AMLK scores predictions and what to trust for Hebrew.

## Pipeline

1. `evaluation/predict.py` or HF Jobs → `predictions.jsonl`
2. `evaluation/evaluate.py` → report JSON (ROUGE, BERTScore, optional judge)
3. `evaluation/error_analysis.py` → failure-type rates (~50 sample)

## ROUGE (Hebrew-aware tokenization)

`evaluation/evaluate.py` uses `_UnicodeTokenizer` so Hebrew isn’t stripped (default `rouge_score` drops non-ASCII).

**Limitations** (HeSum + paper):
- Morphology: same meaning, different inflection → low overlap
- Flexible word order, ktiv haser/male
- **Negative correlation with human judgment** (~−0.16 PCC)

**Planned:** raw + **normalized** ROUGE (strip niqqud, normalize final letters) — `TODO.md` B'.2.

Use ROUGE to compare with HeSum Table 3, not as sole quality signal.

## BERTScore

**Current:** `xlm-roberta-large` (multilingual), CPU-pinned in `evaluate.py`.

**Planned (HeSum-aligned):** `onlplab/alephbert-base` — see [[HeSum Paper Insights#AlephBERT for BERTScore]].

Summaries are short → AlephBERT 512-token limit is fine.

## LLM-as-judge

Faithfulness + fluency (1–5), JSON reply. Default in recent runs: `meta-llama/Meta-Llama-3-8B-Instruct` via HF Inference.

**Caveats:**
- Weak Hebrew → noisy scores
- Gemini judge + Gemini baseline = possible self-preference (`TODO.md` B'.1)
- Judge caught finetuned faithfulness **1.86** vs base **2.64** despite higher ROUGE — aligns with [[Prediction Failure Modes]]

## Error analysis labels

`evaluation/error_analysis.py`: hallucination, omission, entity_or_number_error, **lead_copying**, fluency_problem.

## What to foreground in the paper

1. AlephBERT BERTScore  
2. LLM judge (non-Gemini family preferred)  
3. Qualitative / failure-type rates  
4. ROUGE (comparability + caveat from HeSum)

Related: [[Current Results]], [[Fix Plan#Phase 0]], [[Fix Plan#Phase 3]]
