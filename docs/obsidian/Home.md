# AMLK — shared research notes

> Hebrew news headline generation, and a review of HeSum as a training resource.
> Last updated: 2026-07-27.

## One-paragraph summary

The project began as "fine-tune a Hebrew summarizer and probe it for lead bias." Two model eras later — Qwen3-2B, then DictaLM — the blocker turned out not to be the model. **HeSum is a scraped corpus whose summaries are news-site subheadings, and a large fraction of them are unfit as training targets.** So the project became a dataset review: audit the defects, repair them, and measure whether the repair matters. Of 10,000 raw rows, 5,854 survive curation, and **52.4% of the survivors needed their headline rewritten**. The lead-bias question survives inside the new frame as a continuous per-row measure rather than a separate training experiment.

Start at [[Project Pivot]] for how we got here.

## Map of content

### Current work — dataset review

- [[Project Pivot]] — the three eras and why the project changed shape
- [[Data Curation Pipeline]] — the eight curation stages, verified counts, design consequences
- [[Dataset Defect Taxonomy]] — the six source labels, what they get right and wrong, the analysis strata
- [[Reference Quality Rubric]] — the four-dimension judge that measures everything
- [[Reference Quality Experiment]] — E1 to E4, the hypotheses and statistical design
- [[Training Handoff Contract]] — the boundary with externally owned training
- [[Paper Figures]] — the nine-figure manifest, conventions, compute placement, sequencing
- [[Experiment Results]] — measured outcomes, exact numbers, and article-ready narrative (E1–E4 + DictaLM)

### Literature & metrics

- [[HeSum Paper Insights]] — arXiv:2406.03897, now read as dataset criticism
- [[Evaluation Metrics]] — ROUGE, AlephBERT BERTScore, judge, and what to trust
- [[Lead Bias Probe]] — the positional-shortcut question, reframed

### Qwen era — superseded, kept for the record

These document the first two eras. The numbers are real and some findings still matter, but the remedies they propose have been overtaken by the dataset review.

- [[Current Results]] — Qwen v1/v2/v3 numbers and the error-analysis clustering
- [[Prediction Failure Modes]] — what went wrong in fine-tuned outputs
- [[Fix Plan]] — the phased decode-then-retrain plan
- [[Decoding Configuration]] — the generation settings that caused repetition loops
- [[Training Objective]] — cross-entropy versus what we evaluate
- [[Failure Examples]] — curated Hebrew failure cases

### Project links

- [[References]] — papers, Hub repos, wandb
- Repo: `data_curation/CURATION_ROADMAP.md`, `AGENTS.md`, `TODO.md`
- Spec: `docs/superpowers/specs/2026-07-26-dataset-review-experimental-design.md`

## What we own, what we do not

Training is owned externally. We own the dataset audit, the figure set, and the post-training evaluation. The interface is [[Training Handoff Contract]], and the one non-negotiable term is that **we produce the split** — arms trained on independently drawn splits cannot be compared.

## Open decisions

- [x] Hebrew rubric anchors drafted (`rubric_anchors.py`) and used in pilot + full E1 pass
- [x] Rubric pilot complete (κ 0.65–0.90; no degenerate dimensions)
- [x] F1–F5 drawn; E2/E3 complete (F6 render pending; F7 done)
- [x] Judge family: Gemini `gemini-2.5-flash-lite` (separate from curator + training base)
- [ ] Finish DictaLM2 baseline inference (648/800) and finetuned Arm B test inference (~50/586)
- [ ] Arm A training + predictions (external); align eval to `frozen_split_v1.json` (1,162 test ids)
- [x] Human-validation UI + frozen worklist on `feature/human-validation-ui` → F9a (see `evaluation/viewer/ANNOTATION.md`)
- [ ] All three annotators (amit, avreymi, ofek) complete the blind round → appendix κ heatmap
- [ ] Render F6; build F8 once E4 predictions land; Q1 qualitative exhibit

See [[Experiment Results]] for the full checklist and measured numbers.
