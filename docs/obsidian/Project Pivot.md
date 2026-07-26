# Project Pivot

#status/done

The project passed through three eras. Each ended because it answered a question we had not meant to ask, and the third answer changed what the paper is about.

The current framing: **AMLK is a review of HeSum as a training resource**, not a model-building exercise. We audit the dataset, repair it, and measure whether the repair matters. The Hebrew headline model is the instrument, not the contribution.

## Era 1 — Qwen3-2B (2026-05 to 2026-06-28)

Fine-tune `Qwen/Qwen3-2B` on HeSum with LoRA/QLoRA and evaluate with ROUGE, BERTScore, and an LLM judge. Full numbers in [[Current Results]].

Three runs, each fixing the previous one's diagnosis:

| Version | Setup | ROUGE-1 | BERTScore F1 (AlephBERT) |
|---------|-------|---------|--------------------------|
| v1 | 1 epoch, attention-only LoRA, old greedy decode | 11.4 | 0.449 |
| v2 | v1 adapter, anti-degeneration decode | 4.7 | 0.383 |
| v3 | 3 epochs, LoRA on all 24 layers, new decode | 5.1 | 0.390 |

**Why it ended.** Two findings that pointed past the model.

The v1-to-v2 drop showed that v1's apparent quality was an artifact: it looped a template phrase whose newspaper names accidentally overlapped the references, inflating ROUGE. Suppressing repetition removed the accident and revealed the real level.

More importantly, v3 produced **fluent, correctly formatted Hebrew with wrong facts** — hallucination on roughly two thirds of sampled examples, and the judge rating the *untuned base* model as more faithful (2.98) than the fine-tuned one (2.64). Fine-tuning had reliably taught the model something. What it taught was HeSum's output *style*, including its `"headline | headline | headline"` digest format, rather than faithfulness to the article.

A model that learns the format of its targets but not their content is usually a signal about the targets.

Related: [[Prediction Failure Modes]], [[Decoding Configuration]], [[Fix Plan]]

## Era 2 — DictaLM (2026-07-09)

Qwen3-2B is not a Hebrew model. Swapping to a Hebrew-native base was the obvious next move, and `notebooks/dictalm_hesum_zero_shot.ipynb` ran a qualitative zero-shot evaluation of `dicta-il/dictalm2.0-instruct` on HeSum articles.

**Result:** clear qualitative improvement. Natural Hebrew, no script leakage, no `<think>` blocks, no failure to answer in the target language — all of which had plagued the Qwen base model, which produced non-Hebrew output on 97% of sampled examples.

**Why it ended.** The improvement was real but the outputs still felt wrong against the references, and in a specific way: the model's headline was often *better* than the reference it was being scored against. Reading enough pairs made the pattern hard to unsee. That moved the investigation from the model to the data.

The DictaLM tokenizer stayed on as the basis for the token-budget filter in the curation pipeline, so this era left a lasting artifact even though the model question moved on.

## Era 3 — Dataset review (2026-07-25 onward)

Reading the HeSum paper alongside the actual rows resolved it. HeSum is a scraped corpus, and its summaries are professional *extended subheadings* lifted from news sites rather than summaries written for the task. That construction has consequences the paper itself discusses, and they surface in the data as concrete defects: multi-headline digests bundled into one target, articles whose substance lives in an embedded video, roundups covering several unrelated stories, and scraped boilerplate tails.

HeSum is not a gold dataset. It is a useful, honestly documented, *noisy* one.

The `data_curation/` pipeline is the response — see [[Data Curation Pipeline]] for the stages and [[Dataset Defect Taxonomy]] for what it found. Headline numbers: 10,000 raw rows reduce to 5,854 usable ones, a 41.5% reduction, and of the survivors **52.4% needed their headline rewritten**.

**Why this is the better project.** The original abstract asked whether a fine-tuned Hebrew summarizer latches onto the lead. That question presupposes trustworthy references — a lead-bias measurement against lead-aligned subheadings mostly measures the references. Auditing the dataset is both the prerequisite for the original question and a more defensible contribution, because it is a claim about a resource other people use rather than a claim about one checkpoint we trained.

The lead-bias question survives inside the new frame, and in better shape: headline-to-lead overlap becomes a per-row covariate computed across the whole corpus rather than a separate training experiment. See [[Lead Bias Probe]].

## What carries forward, what does not

Still valid:

- The evaluation battery in `evaluation/` — ROUGE, BERTScore, judge, error analysis, and the predictions viewer
- [[Evaluation Metrics]] and the AlephBERT BERTScore backbone choice
- [[HeSum Paper Insights]], now read as dataset criticism rather than as a target to beat
- The observation that ROUGE correlates about -0.16 with human judgement on this dataset, which is why the audit uses a rubric judge instead

Superseded, kept for the record:

- [[Current Results]] — Qwen-era numbers, retained because the v1-to-v2 collapse is itself a finding about ROUGE
- [[Fix Plan]], [[Decoding Configuration]], [[Training Objective]] — Qwen-era remedies
- The `--variant whole|lead|body` training probe in `data/preprocess.py`, replaced by the continuous length analysis
- The `--clean` profile in `data/clean.py`, an earlier and much cruder version of what `data_curation/` now does properly

## Current division of labour

Training is owned externally. We own the dataset audit, the figures, and the post-training evaluation. See [[Training Handoff Contract]] for the boundary and [[Reference Quality Experiment]] for the experiments.

Related: [[Home]], [[Data Curation Pipeline]], [[Dataset Defect Taxonomy]], [[Reference Quality Experiment]], [[Paper Figures]]
