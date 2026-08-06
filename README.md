# Auditing HeSum: Reference Quality Defects in a Hebrew News Summarization Corpus

This repository contains the final research code and selected frozen artifacts for auditing and curating the HeSum Hebrew news summarization corpus.

The submitted paper is available at [`Auditing_HeSum.pdf`](Auditing_HeSum.pdf).

This `main` branch is the cleaned final version of the project. The complete research and development history is preserved on `archive/research-complete`.

### Dataset redistribution note

`artifacts/data_curation/final_clean_hesum.json` is a curated derivative of the publicly hosted [`biunlp/HeSum`](https://huggingface.co/datasets/biunlp/HeSum) dataset and is included to support academic review and reproducibility of this project. The official HeSum dataset card does not currently specify an explicit dataset license. This repository does not claim ownership of, relicense, or grant additional rights to the underlying article text; users intending to redistribute or reuse that text should consult the official HeSum dataset page and the applicable original-source terms.

## Repository structure

```text
pipeline/
  common/                              shared I/O, paths, statistics, and plotting
  stage_01_data_curation/              download, deterministic cleanup, saved-output curation, dataset build
  stage_02_evaluation_instrument/      rubric anchors, rubric judge, pairwise judge, pilot
  stage_03_reference_experiments/      E1, E2, E3, and human-validation analyses
  stage_04_training_experiment/        matched E4 LoRA preparation, HF Jobs run, scoring, and figure
  stage_05_supplementary/lead_bias/    article-length and lead-bias analysis
artifacts/                              frozen API-, model-, and human-produced inputs
results/                                locally generated, gitignored figures, tables, and summaries
```

Generated outputs are created locally by the relevant scripts and are not version-controlled. The submitted PDF contains the final reported figures and results.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Execution levels

### 1. Rebuild the curated dataset

```bash
python -m pipeline.stage_01_data_curation.run
```

This separately downloads `biunlp/HeSum`, trims repeated tail boilerplate, applies the 4,000-token DictaLM budget and multi-pipe filters, consumes the frozen source-filter and headline-curation outputs, and writes:

```text
artifacts/data_curation/final_clean_hesum.json
```

It may access Hugging Face and load the `dicta-il/dictalm2.0-instruct` tokenizer. It does not rerun OpenAI curation while the two saved curation outputs are present. The deterministic row ledger can then be built with:

```bash
python -m pipeline.stage_01_data_curation.build_row_ledger
```

### 2. Rerun expensive stages individually

Model-assisted curation:

```bash
python -m pipeline.stage_01_data_curation.model_curation.source_filter
python -m pipeline.stage_01_data_curation.model_curation.headline_curation
```

Evaluation-instrument pilot:

```bash
python -m pipeline.stage_02_evaluation_instrument.run_pilot
```

E4 data preparation and matched HF Jobs runs:

```bash
python -m pipeline.stage_04_training_experiment.prepare_curated_data
python -m pipeline.stage_04_training_experiment.prepare_data \
  --arm curated \
  --input outputs/training_experiment/curated/curated_records.jsonl \
  --force

python -m pipeline.stage_04_training_experiment.prepare_uncleaned_data
python -m pipeline.stage_04_training_experiment.prepare_data \
  --arm uncleaned \
  --input outputs/training_experiment/uncleaned/uncleaned_records.jsonl \
  --test-from outputs/training_experiment/processed/curated \
  --force

python -m pipeline.stage_04_training_experiment.submit --arm uncleaned
python -m pipeline.stage_04_training_experiment.submit --arm curated
python -m pipeline.stage_04_training_experiment.score
```

These commands may require `OPENAI_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`, GPU-backed Hugging Face Jobs, and significant time or cost.

## Frozen artifacts

An artifact is an expensive, human-produced, API-produced, model-produced, or otherwise non-deterministic input required to reproduce a paper result. Deterministic summaries, tables, and figures belong in `results/`; temporary data and checkpoints belong under ignored `outputs/` or remote model storage.

Included now:

- source-usability judgments: `artifacts/data_curation/source_filter_results.json`
- headline curation/repair decisions: `artifacts/data_curation/headline_target_curation_results.json`
- curated HeSum dataset: `artifacts/data_curation/final_clean_hesum.json`
- unified deterministic row ledger: `artifacts/data_curation/row_labels.json`
- frozen human-validation worklist and three completed annotation files under `artifacts/reference_experiments/human_validation/`
- uncleaned-arm E4 predictions: `artifacts/training_experiment/predictions_uncleaned.jsonl`
- curated-arm E4 predictions: `artifacts/training_experiment/predictions_curated.jsonl`
- E4 rubric scores: `artifacts/training_experiment/rubric_scores.jsonl`
- E4 blind-pairwise judgments: `artifacts/training_experiment/pairwise_judgments.jsonl`
- E4 frozen summary: `artifacts/training_experiment/summary.json`

## Research workflow and models

The paper studies data curation, a four-dimensional reference-quality instrument, E1 flagged-strata analysis, E2 paired repair, E3 blind preference, human validation, E4 matched fine-tuning, and supplementary lead-bias analysis.

Additional project-related datasets, training runs, and model repositories are preserved under the `avreymi` Hugging Face account. The repositories used directly in the reported E4 experiment are listed below.

- Dataset: [`biunlp/HeSum`](https://huggingface.co/datasets/biunlp/HeSum)
- Curator: `gpt-5.6-luna`
- Rubric and pairwise judge: `gemini-2.5-flash-lite`
- E4 base model/tokenizer: [`dicta-il/dictalm2.0-instruct`](https://huggingface.co/dicta-il/dictalm2.0-instruct)
- E4 datasets: `avreymi/amlk-training-data-raw`, `avreymi/amlk-training-data-e4cur`
- E4 adapters: `avreymi/amlk-e4-raw`, `avreymi/amlk-e4-curated`

The paper reports that curation retained 5,854 of 10,000 records and rewrote 3,069 retained headlines. E1 found substantially poorer rubric scores in flagged strata; E2 and E3 found the repairs improved and were preferred over original targets; human validation broadly supported the automated instrument; and E4 found gains in faithfulness and informativeness for the model trained on curated data. These are paper findings, not constants used to manufacture generated results.
