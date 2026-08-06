# Auditing HeSum: Reference Quality Defects in a Hebrew News Summarization Corpus

This repository contains the paper-aligned research pipeline and frozen evidence for auditing and curating the HeSum Hebrew news summarization corpus. It separates expensive or non-deterministic research artifacts from deterministic figures, tables, and summaries.

The submitted paper is available at [`Auditing_HeSum.pdf`](Auditing_HeSum.pdf).

## Repository structure

```text
pipeline/
  common/                              shared I/O, paths, statistics, and plotting
  stage_01_data_curation/              download, deterministic cleanup, saved-output curation, dataset build
  stage_02_evaluation_instrument/      rubric anchors, rubric judge, pairwise judge, pilot
  stage_03_reference_experiments/      E1, E2, E3, and human-validation analyses
  stage_04_training_experiment/        matched E4 LoRA preparation, HF Jobs run, scoring, and figure
  stage_05_supplementary/lead_bias/    article-length and lead-bias analysis
  reproduce_results.py                 reviewer-facing deterministic orchestrator
artifacts/                              frozen API-, model-, and human-produced inputs
results/                                deterministic figures, tables, and summaries
```

The historical research repository—including paper source, planning documents, abandoned experiments, notebooks, viewers, and development utilities—remains on `archive/research-complete` and is intentionally excluded from this branch.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The reviewer-facing reproduction command uses only local frozen artifacts. Dataset rebuilding and expensive experiment reruns have additional network, credential, and hardware requirements described below.

## Execution levels

### 1. Reproduce results from frozen artifacts

```bash
python -m pipeline.reproduce_results
```

This command detects the frozen artifacts currently available and runs every supported deterministic analysis. It never calls an API, downloads data or model checkpoints, runs inference, or trains a model. Analyses whose frozen inputs are not currently available are skipped neutrally, while malformed present artifacts and real analysis errors still cause a nonzero exit.

With the artifacts currently included, the command performs a partial reproduction: it regenerates the data-curation and human-annotation summaries together with the E4 rubric, blind-pairwise, and combined summaries and the E4 figure. E4 rubric and blind-pairwise results are reproducible from the distributed frozen artifacts; analyses that require unavailable row labels, pilot outputs, or E1–E3 artifacts continue to be skipped. The human-validation JSON summarizes only the available human annotations; it does not report human–judge agreement.

Generated files go only to:

- `results/figures/`
- `results/tables/`
- `results/summaries/`

The eight final-paper figures are committed at canonical paths. A partial run leaves a figure unchanged when its frozen inputs are unavailable; when those inputs are supplied, the command regenerates that same path.

### 2. Rebuild the curated dataset

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

### 3. Rerun expensive stages individually

Expensive stages are exposed only where the archived repository provides the corresponding execution path. E1, E2, and E3 are reproduced from saved artifacts rather than rerun through a synthetic full-pipeline command.

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

These commands may require `OPENAI_API_KEY`, `GEMINI_API_KEY`, `HF_TOKEN`, GPU-backed Hugging Face Jobs, and significant time or cost. They are not part of normal result reproduction.

## Frozen artifacts

An artifact is an expensive, human-produced, API-produced, model-produced, or otherwise non-deterministic input required to reproduce a paper result. Deterministic summaries, tables, and figures belong in `results/`; temporary data and checkpoints belong under ignored `outputs/` or remote model storage.

Included now:

- source-usability judgments: `artifacts/data_curation/source_filter_results.json`
- headline curation/repair decisions: `artifacts/data_curation/headline_target_curation_results.json`
- curated HeSum dataset: `artifacts/data_curation/final_clean_hesum.json`
- frozen human-validation worklist and three completed annotation files under `artifacts/reference_experiments/human_validation/`
- uncleaned-arm E4 predictions: `artifacts/training_experiment/predictions_uncleaned.jsonl`
- curated-arm E4 predictions: `artifacts/training_experiment/predictions_curated.jsonl`
- E4 rubric scores: `artifacts/training_experiment/rubric_scores.jsonl`
- E4 blind-pairwise judgments: `artifacts/training_experiment/pairwise_judgments.jsonl`
- E4 frozen summary: `artifacts/training_experiment/summary.json`

Documented canonical destinations for frozen files that are not distributed in this repository:

- `artifacts/data_curation/row_labels.json`
- `artifacts/reference_experiments/rubric_pilot.json`
- `artifacts/reference_experiments/e1_rubric_scores.jsonl`
- `artifacts/reference_experiments/e2_repair_summary.json`
- `artifacts/reference_experiments/e3_pairwise.jsonl`
- `artifacts/reference_experiments/e3_pairwise_summary.json`

No placeholder data are committed for unavailable artifacts.

### Dataset redistribution note

`artifacts/data_curation/final_clean_hesum.json` is a curated derivative of the publicly hosted [`biunlp/HeSum`](https://huggingface.co/datasets/biunlp/HeSum) dataset and is included to support academic review and reproducibility of this project. The official HeSum dataset card does not currently specify an explicit dataset license. This repository does not claim ownership of, relicense, or grant additional rights to the underlying article text; users intending to redistribute or reuse that text should consult the official HeSum dataset page and the applicable original-source terms.

## Research workflow and models

The paper studies data curation, a four-dimensional reference-quality instrument, E1 flagged-strata analysis, E2 paired repair, E3 blind preference, human validation, E4 matched fine-tuning, and supplementary lead-bias analysis.

- Dataset: [`biunlp/HeSum`](https://huggingface.co/datasets/biunlp/HeSum)
- Curator: `gpt-5.6-luna`
- Rubric and pairwise judge: `gemini-2.5-flash-lite`
- E4 base model/tokenizer: [`dicta-il/dictalm2.0-instruct`](https://huggingface.co/dicta-il/dictalm2.0-instruct)
- E4 datasets: `avreymi/amlk-training-data-raw`, `avreymi/amlk-training-data-e4cur`
- E4 adapters: `avreymi/amlk-e4-raw`, `avreymi/amlk-e4-curated`

The paper reports that curation retained 5,854 of 10,000 records and rewrote 3,069 retained headlines. E1 found substantially poorer rubric scores in flagged strata; E2 and E3 found the repairs improved and were preferred over original targets; human validation broadly supported the automated instrument; and E4 found gains in faithfulness and informativeness for the model trained on curated data. These are paper findings, not constants used to manufacture generated results.
