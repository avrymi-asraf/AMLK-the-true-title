# AMLK — shared research notes

> Hebrew news headline generation, and a review of HeSum as a training resource.
> Last updated: 2026-07-26.

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

- [ ] Draft the Hebrew anchor examples for [[Reference Quality Rubric]] and have both team members review before any full judge run
- [ ] Pilot the rubric on a few hundred rows and check test-retest agreement before spending on the full 10,000-row pass
- [ ] Draw F1 and F2 from existing artifacts — no API cost, immediately useful
- [ ] Decide the judge model family, which must differ from `gpt-5.6-luna` and from the training base model
- [ ] Confirm with the training owner that arm A will be size-matched to arm B
- [ ] Schedule the 150-row human annotation round (both members, independent, blind)
