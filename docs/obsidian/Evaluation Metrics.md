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

**Done:** `evaluate.py` reports raw + **normalized** ROUGE (`normalize_hebrew` strips niqqud and folds
final-form letters).

**Now appendix-only.** Under the dataset review ROUGE never carries a claim — it is reported solely for
comparability with HeSum Table 3. The -0.16 correlation is reason enough, and the [[Project Pivot|v1-to-v2
collapse]] showed it can be inflated outright by repetition artifacts.

## BERTScore

**Done:** default is `onlplab/alephbert-base` (HeSum-aligned, far more discriminative than the old
`xlm-roberta-large`, which clustered everything near 0.85 regardless of quality). Override with
`--bertscore-model`. CPU-pinned in `evaluate.py`. See [[HeSum Paper Insights#AlephBERT for BERTScore]].

Summaries are short → AlephBERT 512-token limit is fine.

**Note on scale:** xlm-r numbers (~0.85) and AlephBERT numbers (~0.35-0.50) are not comparable. Any
table mixing them must say so.

**Role now:** secondary. Cheap, deterministic, reproducible, and it triangulates the rubric judge from a
different instrument — but it compares against a reference, and under the dataset review the reference is
the thing under suspicion.

## LLM-as-judge

Faithfulness + fluency (1–5), JSON reply. Default in recent runs: `meta-llama/Meta-Llama-3-8B-Instruct` via HF Inference.

**Caveats:**
- Weak Hebrew → noisy scores
- Gemini judge + Gemini baseline = possible self-preference (`TODO.md` B'.1)
- Judge caught finetuned faithfulness **1.86** vs base **2.64** despite higher ROUGE — aligns with [[Prediction Failure Modes]]

## Error analysis labels

`evaluation/error_analysis.py`: hallucination, omission, entity_or_number_error, **lead_copying**, fluency_problem.

Applies to model *outputs*. The dataset review needed a taxonomy for *references* instead, which is why
[[Reference Quality Rubric]] exists rather than reusing these labels.

## The rubric judge — primary instrument for the dataset review

Every metric above compares two strings, which is the wrong shape for "is this reference any good?".
[[Reference Quality Rubric]] specifies a judge that reads the **article** and scores a headline directly
on four ordinal dimensions — faithfulness, single-focus, informativeness, cleanliness.

Three properties that matter here:

- **Reference-free.** It never compares against another headline, so it is immune to the confound that
  our curated references were LLM-written.
- **Ordinal, so no means.** Report distributions and use rank-based tests. See
  [[Reference Quality Rubric#Scores are ordinal]].
- **Validated before use.** Test-retest kappa plus a 150-row human round. Unlike the judge described
  above, it is not trusted on assertion.

The same instrument scores dataset references and model outputs, which puts both on one axis.

## What to foreground in the paper

1. **Rubric judge sub-scores** — primary, for both references and model outputs
2. **Blind pairwise win rates** — null-calibrated, the credibility anchor
3. AlephBERT BERTScore — secondary triangulation
4. Qualitative examples and failure-type rates
5. ROUGE — appendix, comparability with HeSum Table 3 only

Related: [[Reference Quality Rubric]], [[Reference Quality Experiment]], [[HeSum Paper Insights]], [[Current Results]]
