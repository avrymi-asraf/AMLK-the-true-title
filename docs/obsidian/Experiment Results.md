# Experiment Results

#status/in-progress

Article-ready log of measured outcomes from E1–E4 and supporting analyses. Numbers are taken
directly from artifacts on disk unless marked *in flight*. Pre-registered design:
`docs/superpowers/specs/2026-07-26-dataset-review-experimental-design.md`. Readable map:
[[Reference Quality Experiment]]. Figure manifest: [[Paper Figures]].

**Last updated:** 2026-07-30.

---

## Status dashboard

| Component | Status | Artifact / figure |
|-----------|--------|-------------------|
| Row labels + F1/F2 | **Done** | `data_curation/artifacts/row_labels.json`, `outputs/figures/f1_*`, `f2_*` |
| Rubric pilot (instrument) | **Done** | `outputs/results/rubric_pilot.json`, `sx16_pilot_kappa`, `sx17_pilot_distributions` |
| **E1** full judge pass | **Done** (9,988 rows) | `outputs/results/e1_rubric_scores.jsonl` → F3–F5 |
| **E2** paired curated vs original | **Done** (3,068 paired) | `outputs/results/e2_repair_summary.json` → F6 |
| **E3** head-to-head + placebo | **Done** (1,159 judgments) | `outputs/results/e3_pairwise_summary.json` → F7 |
| **E4 code path** | **Ready** (2026-07-30) | `docs/e4-raw-vs-curated-training-plan.md` — decode off, 35-word prompt, `download_raw`, `--test-from`, `e4_score`. **Jobs not submitted** |
| **DictaLM2 zero-shot baseline** | **In flight** (648/800 Hub) | `predictions-dictalm2-baseline.jsonl` → F9 (scored on 604) |
| **DictaLM2 fine-tuned Arm B** | **In flight** (~60/586 test) | `avreymi/amlk-dictalm2-instruct-sft` → F8 blocked until E4 arms complete |
| Dual-reference probe (1.7B) | **Done** (1,427 rows) | `outputs/results/probe-dual-reference.json` (narrative support only) |
| Human validation (150 rows) | **Not started** | Original F9 appendix plan |
| Supplementary figures sx01–sx18 | **Done** | `outputs/figures/sx_*`, manifest in `supplementary_manifest.txt` |

---

## E1 — Was the diagnosis right?

**Population:** 9,988 scored rows (10,000 minus 9 anchor rows minus 3 empty text/headline).
**Instrument:** Gemini `gemini-2.5-flash-lite` rubric judge on **original** headlines only.
**Joined to row labels:** 9,958 rows (`load_joined_rubric_results()`).

### Pilot instrument check (n=300 + 60 retest)

Quadratic-weighted Cohen's κ (test-retest): faithfulness **0.863**, single-focus **0.896**,
informativeness **0.757**, cleanliness **0.651**. No dimension degenerate (≥85% on one level).

### Stratum medians vs S0 clean (Cliff's δ vs S0)

| Stratum | n | Faithfulness | Single-focus | Informativeness | Cleanliness |
|---------|---|--------------|--------------|-----------------|-------------|
| **S0 clean** | 2,785 | med 5.0 | med 5.0 | med 5.0 | med 5.0 |
| **S2 multi-pipe** | 2,406 | med 3.0, δ **−0.844516** | med 1.0, δ **−0.949814** | med 3.0, δ **−0.809547** | med 5.0, δ −0.304793 |
| **S3 multi-item** | 567 | med 4.0, δ −0.429952 | med 2.0, δ −0.641520 | med 4.0, δ −0.490321 | med 5.0, δ −0.151429 |
| **S4 rewritten** | 3,058 | med 5.0, δ −0.265556 | med 5.0, δ −0.158323 | med 5.0, δ −0.119620 | med 5.0, δ −0.109905 |

**Pre-registered predictions confirmed (F3/F4):** S2/S3 collapse single-focus (~85% at levels 1–2 for S2);
faithfulness and informativeness also down for S2/S3; S0 at ceiling on all dimensions.

**Surprises kept as findings (not bugs):**

1. **S2/S3 faithfulness** dropped more than predicted ("roughly intact"). Multi-pipe/multi-item headlines
   often assert facts absent from the one article body HeSum paired them with — hallucination relative to
   source, not a rubric artifact.
2. **S4 original headlines** scored close to S0 on the rubric, while the curation pipeline still rewrote
   3,069 of them — inter-instrument disagreement between the OpenAI curator and the independent Gemini judge.

**Cleanliness looks "flat" because of ceiling + definition:** cleanliness measures formatting/scraped-artifact
burden, not semantic focus. S2 med 5.0 vs S0 med 5.0 but only 68.5% at score 5 (vs S0 98.8%); δ still
−0.30 for S2. Single-focus is where the pipe-digest defect shows (S2 med **1**).

### F5 — Length and lead bias

Equal-count (quantile) bins over article length. Headline–lead overlap declines smoothly and
monotonically with article length; **no discontinuity at the 4,000-token filter**. Supports modeling
length as a continuous covariate rather than a stratum ([[Dataset Defect Taxonomy#Why article length is a covariate, not a stratum]]).

**Figures:** `f3_rubric_distributions`, `f4_effect_sizes`, `f5_length_lead_bias`.

---

## E2 — Was the repair right, graded?

**Population:** 3,069 rewritten rows; **3,068** paired (original + curated rubric scores).
**Artifact:** `outputs/results/e2_repair_summary.json`.

### Ceiling compression (why E3 matters)

For **all** rewritten rows, both original and curated medians are **5.0** on every dimension — the
dumbbell plot (F6) will look flat. Wilcoxon is still highly significant because many rows move at the
top of the scale:

| Dimension | share increased (curated > original) | share decreased | Wilcoxon p |
|-----------|--------------------------------------|-----------------|------------|
| Faithfulness | 42.0% | 3.6% | 7.96×10⁻¹⁹² |
| Single-focus | 27.6% | 6.3% | 3.04×10⁻⁹⁵ |
| Informativeness | 33.9% | 10.8% | 2.48×10⁻⁹⁶ |
| Cleanliness | 12.0% | 0.5% | 1.11×10⁻⁶³ |

**Interpretation for the paper:** graded rubric confirms direction (curated ≥ original, rarely worse) but
cannot show magnitude at ceiling. **E3 carries the effect-size story.**

### By edit sub-type (selected)

| Edit type | n | Notable pattern |
|-----------|---|-----------------|
| **pipes_removed** | 48 | Cleanliness: orig med **3.0** → curated med **5.0**, median gap **+2.0**, 81.25% increased |
| **full_rewrite** | 2,627 | Faithfulness 45.0% increased; still med 5 vs 5 |
| **light_edit** | 314 | Weakest lift; E3 win rate ~49.5% curated |
| **boilerplate_stripped** | 57 | Modest gains; small n |

**Figure:** F6 not yet rendered — run `python -m data_curation.analysis.repair_figures`.

---

## E3 — Was the repair preferred head-to-head?

**Population:** ~1,000 rewritten + ~200 placebo (kept rows, identical headlines).
**Artifact:** `outputs/results/e3_pairwise_summary.json` (1,159 valid judgments).

### Main result (rewritten cohort, n=961)

| Outcome | Rate | 95% CI (curated win rate) |
|---------|------|---------------------------|
| **Curated wins** | **73.56919875130072%** | [70.75702393340269%, 76.37877211238293%] |
| Tie | 2.705515088449532% | |
| Original wins | 23.72528616024974% | |

### Placebo calibration (kept rows, n=198)

| Outcome | Rate |
|---------|------|
| Curated wins | 0.5050505050505051% |
| **Tie** | **99.4949494949495%** |
| Original wins | 0.0% |

Placebo near-perfect ties validate the pairwise judge — the 74% curated win rate on rewritten rows is
not instrument noise.

### By edit sub-type (rewritten only)

| Edit type | n | Curated % | Notes |
|-----------|---|-----------|-------|
| **full_rewrite** | 833 | **76.83073229291718%** | Drives the headline number |
| **light_edit** | 97 | 49.48453608247423% | ~coin flip — curation often cosmetic |
| **pipes_removed** | 11 | 72.72727272727273% | Small n |
| **boilerplate_stripped** | 19 | 52.63157894736842% | Small n |

**Figure:** `outputs/figures/f7_win_rate.*`

**If E2 and E3 disagree:** they do, by design — E2 says "significant but ceiling-compressed"; E3 says
"strong preference for curated on full rewrites, not on light edits." Report both.

---

## DictaLM inference

Two separate tracks on **`dicta-il/dictalm2.0-instruct`** (7.3B). Do not conflate with the earlier
**DictaLM-3.0-1.7B** dual-reference ROUGE probe ([[#Dual-reference probe (1.7B, narrative support)]] below).

### A — Zero-shot baseline reliability (ours)

**Question:** On the same E1 rubric axis, how does an off-the-shelf DictaLM2 summary compare to the
HeSum reference headline? Supports "references are the bottleneck" without waiting for fine-tuning.

| Item | Value |
|------|-------|
| Sample | 800 rows stratified (200 per S0/S2/S3/S4), `artifacts/baseline_eval_dataset.json` |
| HF Jobs | Full `6a67863d…` timed out at 604/800; resume `6a67a68b…` **in flight** |
| Hub predictions | `avreymi/amlk-baseline-results/predictions-dictalm2-baseline.jsonl` — **648/800** (2026-07-27) |
| Rubric scored | **604** pairs → `baseline-rubric-comparison.json`, `baseline_model_rubric_scores.jsonl` |
| Figure | `f9_baseline_vs_reference_rubric` (repurposed F9 slot — see [[Paper Figures#F9 note]]) |

**Key findings on 604 pairs** (`baseline-rubric-comparison.json`):

| Stratum | Dimension | Reference med | Model med | Gap (model − ref) | Cliff's δ | Interpretation |
|---------|-----------|---------------|-----------|-------------------|-----------|----------------|
| S0 | faithfulness | 5.0 | 2.0 | −3.0 | −0.825569 | **Task mismatch** — model writes summary-length text, reference is headline-length |
| S2 | **single_focus** | **1.0** | **2.0** | **+1.0** | **+0.411836** | **Core narrative** — zero-shot model more single-focus than defective multi-pipe references |
| S2 | faithfulness | 3.0 | 2.0 | −1.0 | −0.505313 | Model still below reference on faithfulness |
| S3 | faithfulness | 4.0 | 2.0 | −2.0 | −0.726637 | Large gap |
| S4 | faithfulness | 5.0 | 2.0 | −2.0 | −0.716548 | Curated references score high; model summaries score low |

**Paper framing:** Use S2 single-focus reversal as evidence that **defective references** can score worse
than a generic summarizer on dimensions the defect targets. Acknowledge S0 faithfulness gap as
headline-vs-summary format confound, not a clean "model beats gold" claim.

**When resume finishes:** re-run `python -m data_curation.analysis.score_baseline_rubric` and
`baseline_reliability_figures` on full 800.

### B — Fine-tuned Arm B (external, avreymi)

**Question:** E4 — does training on **curated** headlines produce better outputs than Arm A (original headlines)?

| Item | Value |
|------|-------|
| Training job | `6a665c8b7ef3c08464969f28` — **ERROR** exit 143 at step **275/293** (~94% of 1 epoch) |
| Adapter | Pushed to `avreymi/amlk-dictalm2-instruct-sft` (LoRA r=32/α=64, whole variant) |
| Dataset | `avreymi/amlk-training-data` — curated, 4,683 train / 585 val / **586 test** |
| Inference v1 | `6a678afb…` — **CUDA OOM** (batch 8) |
| Inference v2 | `6a67a4f8…` — **RUNNING**, `GEN_BATCH_SIZE=1`, `MAX_NEW_TOKENS=128`, `PRED_SUFFIX=-step200` |
| Hub predictions | `predictions-finetuned-step200.jsonl` — **50/586** rows (partial, timeout-safe pushes) |

**Blockers for F8/E4 (updated 2026-07-30):**

1. **Code path is ready** — decode defaults off, judge T=0, 35-word prompt, `data.download_raw`,
   `preprocess --test-from`, `scripts.e4_score` (see plan). Older Arm B partial preds above used
   the broken 1.2/3 decode and/or 15-word prompt; **do not treat them as the E4 result**.
2. **Jobs still open:** rebuild both datasets with current prompt → Hub
   `amlk-training-data-raw` / `amlk-training-data-e4cur` → train `amlk-e4-raw` /
   `amlk-e4-curated` (`--test-subset 120 --skip-base-arm`) → score.
3. Historical note: prior Arm B inference on Hub used a 586-row curated test without `hesum_id`;
   E4 uses the preprocess 80/10/10 curated test copied through both arms (byte-identical).

---

## Dual-reference probe (1.7B, narrative support)

Separate from DictaLM2 rubric baseline. **Model:** `dicta-il/DictaLM-3.0-1.7B-Instruct` (zero-shot).
**Design:** one prediction per article, scored twice (vs original headline, vs curated headline).
**n=1,427** after stratified subsample. Artifact: `outputs/results/probe-dual-reference.json`.

| Group | ROUGE/BERTScore gap (curated − original) | Placebo check |
|-------|----------------------------------------|---------------|
| **kept** | median gap **0.0** on all metrics, p=1.0 | Method sanity — no artifact |
| **full_rewrite** | small negative gaps (curated scores slightly lower) | ROUGE prefers original wording, not quality |
| **all** | mean ROUGE-1 gap −0.0026, p≈0 | ROUGE weak for this task |

**Paper role:** appendix / triangulation only — shows automatic metrics cannot replace the rubric.
Complements E1; does not replace DictaLM2 baseline or E4.

---

## Figure inventory (2026-07-27)

| Id | Built | Path |
|----|-------|------|
| F1 | yes | `outputs/figures/f1_curation_funnel.*` |
| F2 | yes | `outputs/figures/f2_defect_prevalence.*` |
| F3 | yes | `outputs/figures/f3_rubric_distributions.*` |
| F4 | yes | `outputs/figures/f4_effect_sizes.*` |
| F5 | yes | `outputs/figures/f5_length_lead_bias.*` |
| F6 | yes | `outputs/figures/f6_paired_repair.*`, `f6_transition_heatmap.*` |
| F7 | yes | `outputs/figures/f7_win_rate.*` |
| F8 | blocked | needs finetuned predictions + Arm A |
| F9 | yes (partial n=604) | `outputs/figures/f9_baseline_vs_reference_rubric.*` |
| F9a | yes | `outputs/figures/f9a_judge_validation.*` |
| sx01–sx18 | yes | `outputs/figures/sx_*` |
| Q1 | no | qualitative cards from viewer |

---

## Article-ready narrative bullets

Lift these into the paper/discussion as needed:

1. **Curation scale:** 10,000 → 5,854 usable (41.5% net reduction); 52.4% of survivors needed headline rewrite.
2. **E1:** Defect strata are measurably worse on the rubric; largest effect is S2 single-focus (δ ≈ −0.95).
   Length/lead bias is continuous, not a filter artifact.
3. **E2/E3 tension:** Graded scores sit at ceiling; forced choice shows **~74%** curated preference on
   rewrites with **~99%** ties on placebo. Full rewrites drive the win; light edits do not.
4. **DictaLM2 zero-shot:** On multi-pipe rows, the model is **more single-focus** than the reference
   (median 2 vs 1) — evidence the reference, not the model, carries the multi-headline defect.
5. **DictaLM2 fine-tuned:** Training nearly finished; inference in progress. E4 conclusion pending.
6. **Instrument:** Pilot κ 0.65–0.90; pairwise placebo validates E3. **F9a complete** (3 annotators,
   disjoint 152 rubric + 95 pairwise); pooled judge–human κ 0.79 single-focus, 0.50 informativeness,
   0.28 cleanliness, 0.11 faithfulness — but cleanliness/faithfulness are within ±1 ordinal point
   90%/68% of the time, so low κ reflects compressed score variance, not real disagreement. On
   non-tie pairwise (n=78), humans pick curated 83.3% of the time — *more* decisive than E3's 73.6%
   judge win rate.

---

## F9a — Human judge validation (complete)

**Status:** All three annotators complete — **247 tasks**, **152 unique rubric articles**, **95 pairwise**
(disjoint split; no human–human κ).

| Annotator | Tasks | Rubric | Pairwise |
|-----------|-------|--------|----------|
| amit | 122 | 52 | 70 |
| avreymi | 65 | 52 | 13 |
| ofek | 60 | 48 | 12 |

**Artifacts:** `data_curation/artifacts/human_annotations/{amit,avreymi,ofek}.jsonl`;
summary `outputs/results/human_validation_summary.json`; figure `f9a_judge_validation`
(**two-panel closeness chart**: within ±1 / exact match % + pairwise vs E3 ref; no judge test-retest).

### Rubric: pooled human vs judge closeness, n=152

| Dimension | within ±1 pt | exact match | human mean | judge mean | κ (ref.) |
|-----------|--------------|-------------|------------|------------|----------|
| single_focus | **86%** | 59% | 3.26 | 3.18 | 0.79 |
| informativeness | **76%** | 34% | 3.43 | 3.80 | 0.50 |
| cleanliness | **90%** | 69% | 4.66 | 4.68 | 0.28 |
| faithfulness | **68%** | 40% | 4.33 | 3.85 | 0.11 |

**Readout:** Paper figure leads with **within ±1** and **exact match** % (not κ). All four dimensions
≥68% within one ordinal point; cleanliness means nearly identical. Pairwise panel: human 83.3% curated
on non-tie vs E3 judge 73.6%.

### Pairwise A/B (blind original vs curated), n=95

| Source | n | Curated % | Tie % |
|--------|---|-----------|-------|
| amit | 70 | 66% | 21% |
| avreymi | 13 | 54% | 15% |
| ofek | 12 | 100% | 0% |
| **Pooled (all)** | **95** | **68%** | **18%** |
| **Pooled, non-tie only** | **78** | **83.3%** | — |
| E3 judge (non-tie) | 961 | **73.6%** | — |

vs E3 overlap (n=33): 48% raw agreement. On the non-tie denominator that matches how E3 reports its
win rate, pooled humans are *more* decisive for curated (83.3%) than the automated judge (73.6%) —
the paper uses this comparison instead of the raw 68%-including-ties number.

```bash
python -m data_curation.analysis.human_validation_results
python -m data_curation.analysis.human_validation_figures
```

---

## Remaining work (paper checklist)

- [ ] Finish DictaLM2 baseline inference (800/800) and refresh F9
- [ ] Render **F6** from `e2_repair_summary.json`
- [ ] Finish finetuned inference; score on frozen split with `hesum_id`
- [ ] Arm A training + predictions (external)
- [ ] **F8** once both arms on same test set
- [x] Human validation round → appendix agreement heatmap (**F9a**)
- [ ] **Q1** qualitative exhibit (5–6 Hebrew cards)
- [ ] Reconcile F9 numbering: baseline figure vs judge-validation appendix ([[Paper Figures#F9 note]])

---

## Code entry points

```bash
# E1 figures (already run)
python -m data_curation.analysis.rubric_figures

# E2 summary + F6
python -m data_curation.analysis.repair_results
python -m data_curation.analysis.repair_figures

# E3 summary + F7
python -m data_curation.analysis.repair_pairwise_results
python -m data_curation.analysis.repair_figures   # F7

# DictaLM2 baseline
python -m data_curation.analysis.run_dictalm_baseline_inference --submit-hf --hf-user avreymi
python -m data_curation.analysis.score_baseline_rubric --hf-user avreymi
python -m data_curation.analysis.baseline_reliability_figures

# Supplementary
python -m data_curation.analysis.supplementary_figures
```

---

## Session notes (for provenance)

| Chat / session | What it captured |
|----------------|------------------|
| [7697f9be](7697f9be-9371-4f4f-9f34-3b12a6e75b42) | Pivot to dataset review; E1–E4 pre-registration; rubric judge design; training handoff; figure manifest; frozen split + Arm A redefinition |
| [b77012e9](b77012e9-b533-401d-b55e-708c51b94ed0) | E1 completion; F3–F5; 18 supplementary figures; number formatting fix; HF Arm B job forensics; cleanliness-ceiling analysis |
| [7697f9be follow-on](7697f9be-9371-4f4f-9f34-3b12a6e75b42) | E2/E3 implementation; DictaLM2 baseline + finetuned inference status; F7/F9 |

Related: [[Reference Quality Experiment]], [[Paper Figures]], [[Training Handoff Contract]], [[Project Pivot]]
