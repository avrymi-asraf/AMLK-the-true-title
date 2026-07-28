## Project Goal

* **Description:** AMLK is a Hebrew **news** summarization research project. The goal is to fine-tune `dicta-il/dictalm2.0-instruct` (MODEL_SLUG `dictalm2-instruct`) on **curated HeSum** (main-branch `data_curation` product: `final_clean_hesum.json`), evaluate with ROUGE, BERTScore, and LLM-as-judge, and produce a research paper and presentation. Design choices are informed by **English summarization literature** (lead bias, metric limits, strong baselines) without re-running English experiments. Evaluation includes an **advanced-model baseline** (e.g. Gemini API on the same test set and prompt) so metrics can be interpreted against a stronger system. A **truncation / positional-shortcut probe** trains separate models on Whole text, Lead-only, and Body-only inputs. Optional **headline-style control** varies the instruction (short headline vs longer summary). **Error analysis** labels a sampled set of predictions for failure types common in the literature. Runs locally or on HuggingFace Jobs; all scripts are command-line Python.

---

## Project Structure - remember to update it when you make changes

* **Architecture:** The project is divided into sequential pipelines:
  1. **Data curation** (main worktree, offline / occasional) — `data_curation/` builds the clean HeSum product `final_clean_hesum.json` (source filter + headline repair + deterministic cleanup). Not re-run by every train job.
  2. **Training pipeline** — materialize curated HeSum (`final_clean_hesum.json` → `data.download` → `data.preprocess` → HF Arrow splits with `prompt`/`completion`), load `dicta-il/dictalm2.0-instruct`, and fine-tune with HuggingFace `transformers`/`trl`. Local GPU is insufficient; jobs go to HuggingFace Jobs.
  3. **Evaluation pipeline** — scores fine-tuned and baseline checkpoints on the held-out test set: ROUGE, BERTScore, LLM-as-judge (Gemini), advanced-model baseline, error analysis, plus side tools (topic/style stratification, predictions viewer).
  4. **Results & reporting** — aggregated metrics feed into the paper and presentation.

* **Code Flow:**
  1. Curated HeSum (`data_curation` product) → `data.download` / `data.preprocess` → HuggingFace train/val/test Arrow splits on disk (and Hub on train submit)
  2. Model fine-tuning on HF Jobs → adapter checkpoint on Hub; dual-arm test predictions (`predictions-finetuned.jsonl` / `predictions-base.jsonl`)
  3. Evaluation scripts consume predictions → metric reports
  4. Reports feed the paper / presentation

---

## File Structure - remember to update it with the latest project information

```
/AMLK
├── .agents/
│   └── skills/
│       ├── colab-cli/SKILL.md             # Official Google Colab CLI usage for agent-safe remote runtimes
│       ├── coding-principles/SKILL.md    # Project-local coding standards
│       ├── training/SKILL.md             # AMLK training process (dictalm2, curated data, HF Jobs, wandb)
│       └── testing/SKILL.md              # AMLK testing philosophy
├── data_curation/                        # Offline curation pipeline (source of final_clean_hesum.json)
│   ├── CURATION_ROADMAP.md               # End-to-end curation roadmap
│   ├── build_curated_dataset.py          # Orchestrates curated dataset build steps
│   ├── data_download/download_hesum.py   # Raw biunlp/HeSum download into artifacts
│   ├── pre_model_cleanup/                # Deterministic filters (token budget, multi-pipe, tail boilerplate)
│   ├── model_curation/                   # Batch API source filter + headline target repair
│   ├── final_dataset/                    # Assemble final_clean_hesum.json
│   ├── utils/                            # Shared JSON I/O and paths
│   └── artifacts/                        # Intermediate + final_clean_hesum.json (gitignored bulk)
├── data/
│   ├── __init__.py
│   ├── download.py                       # Pipeline step 1: load curated final_clean_hesum.json → curated_records.jsonl
│   ├── prompts.py                        # build_prompt / format_chat_prompt / make_variant — train+infer prompt source of truth
│   ├── clean.py                          # Legacy digest helpers (style/diagnostics only — not the train path)
│   └── preprocess.py                     # Pipeline step 2: curated → HF Arrow train/val/test (prompt/completion), validate
├── training/
│   ├── __init__.py
│   ├── config.py                         # MODEL_ID, MODEL_SLUG, MAX_LENGTH, METHOD_PRESETS, LoRAConfig, wandb_*/repo helpers
│   ├── train.py                          # Single trainer: --method qlora|lora|full, --submit-hf, resume, TRAIN_CONFIG env
│   ├── train_hf_job.py                   # Self-contained UV script run by HF Jobs (submitted by train.py --submit-hf)
│   └── resume.py                         # Full Trainer checkpoint helpers (Hub list/pick; used by train.py --resume-from)
├── evaluation/
│   ├── __init__.py
│   ├── predict.py                        # Generate the Gemini advanced-baseline summaries (API only)
│   ├── gemini_client.py                  # Shared Gemini API helpers + strip_think()
│   ├── hf_client.py                      # Optional HF-hosted LLM judge (avoid Gemini self-preference)
│   ├── evaluate.py                       # ROUGE-1/2/L + BERTScore (alephbert-base) + Gemini/HF judge → one report
│   ├── error_analysis.py                 # Gemini-labelled failure-type rates on a ~50-sample
│   ├── eval_hf_job.py                    # Full eval battery on HF Jobs (cheap CPU): --submit-hf | cloud runner
│   ├── build_report_tables.py            # Assemble per-system reports into D1 markdown comparison tables
│   ├── infer.py                          # GPU inference helpers (load adapter + generate); observation notebook
│   ├── base_predict.py                   # Zero-shot base helpers: load plan, JSONL I/O, validate_predictions, model_slug
│   ├── hebrew_constraint.py              # Decode constraint: build_bad_words_ids() bans non-Hebrew-script tokens (always on)
│   ├── topic_clustering.py               # Embed + BERTopic cluster + Gemini-name topics; Databricks notebook
│   ├── style_labels.py                   # Rule-based structural style labels — local, no GPU/API
│   ├── stratify_by_topic.py              # ROUGE/BERTScore/failure rates by topic_label or style_label
│   └── viewer/                           # Predictions viewer (self-contained UI feature)
│       ├── __init__.py
│       ├── data.py                       # Streamlit-free helpers: discover/load/keyword-search predictions.jsonl
│       └── app.py                        # Local Streamlit UI: RTL Hebrew, keyword search, multi-system compare
├── notebooks/
│   ├── evaluation_observation.ipynb      # evaluation-observation stage: live per-example view on Colab
│   └── cluster_topics_databricks.py      # Topic-clustering side-analysis on Databricks GPU cluster
├── scripts/
│   ├── __init__.py
│   └── run_nb_cell.py                    # Drive notebook cells on a Colab session via colab-cli
├── tests/                                # Behavioral tests (download/preprocess/eval/viewer/…)
├── docs/
│   ├── obsidian/                         # Shared Obsidian vault (team research notes)
│   ├── ANLP Project abstract.md          # Original proposal (historical — Qwen3-2B era)
│   ├── research-proposal.md              # Original proposal prose (historical — Qwen3-2B era)
│   ├── research-proposal-revised.md      # Current plan of record (base model + probe design)
│   ├── prompt-arena-notebook.md          # Lab notebook for the prompt loop that produced PROMPT_TEMPLATE
│   ├── 2026-06-12-qlora-training-job-postmortem.md  # Historical Qwen full-run post-mortem
│   └── superpowers/
│       ├── specs/…
│       └── plans/…
├── outputs/
│   ├── data/
│   │   ├── curated/final_clean_hesum.json  # Curated HeSum product (or copy from data_curation/artifacts/)
│   │   ├── curated/curated_records.jsonl   # Normalized {text,summary,source,hesum_id} export
│   │   └── processed/<variant>/           # HF Arrow splits train/val/test — train contract
│   ├── checkpoints/                       # LoRA adapter / full model checkpoints
│   └── results/                           # predictions.jsonl + evaluation reports
├── IMPROVEMENT_PLAN.md                   # Training diagnosis + improvement plan (from another-model)
├── .venv/
├── .env                                  # HF_TOKEN, GEMINI_API_KEY — never commit
├── .gitignore
├── AGENTS.md
├── CLAUDE.md                             # Symlink → AGENTS.md
├── README.md
├── requirements.txt
└── TODO.md                               # Milestone tracker
```

* `data_curation/`: Offline pipeline that produces `artifacts/final_clean_hesum.json` (`{hesum_id, text, headline}`). Download → deterministic pre-model cleanup → model curation (source filter + headline rewrite) → final dataset. See `CURATION_ROADMAP.md`. Training never re-runs this; `data.download` only consumes the product.
* `data/download.py`: **Only training-data source path (step 1).** Loads curated HeSum `final_clean_hesum.json` (prefers `outputs/data/curated/`, else `data_curation/artifacts/`), normalizes to `{text, summary, source=hesum-curated, hesum_id}`, writes `outputs/data/curated/curated_records.jsonl`. Does not re-download raw biunlp/HeSum or re-run model curation.
* `data/prompts.py`: Single hardened `PROMPT_TEMPLATE` (Hebrew stop-cue prompt from the prompt-arena loop), `build_prompt(text)`, `make_variant` (whole|lead|body), plus `format_chat_prompt` / `prepare_tokenizer_for_templated_prompts` — wraps instructions in the model's chat template at train/infer time. No Qwen-era `/no_think` injection. Shared by preprocess, train, and evaluation.
* `data/clean.py`: Legacy pure-regex digest helpers (`normalize_summary`, `is_roundup_digest`, `pipe_segments`). **Not used by the train path** (curation already cleaned headlines); kept for style/diagnostics and tests.
* `data/preprocess.py`: **Only training-data path (step 2).** Reads curated JSON/JSONL, builds raw `(prompt, completion)` pairs (chat wrap happens at train/infer), applies `--variant whole|lead|body`, truncates each article to `MAX_LENGTH-256` tokens, splits 80/10/10, **validates the train contract** (`validate_train_dataset`), saves HuggingFace Arrow splits to `outputs/data/processed/<variant>/`. Train `--submit-hf` uploads those splits to `{hf_user}/amlk-training-data`.
* `training/config.py`: Shared constants: `MODEL_ID="dicta-il/dictalm2.0-instruct"`, `MODEL_SLUG="dictalm2-instruct"`, `MAX_LENGTH=4096` (source of truth; preprocess uses `MAX_LENGTH-256=3840` article tokens), `DEFAULT_EPOCHS=1`, `DEFAULT_MAX_NEW_TOKENS=128`, `METHOD_PRESETS` (qlora / lora / full for 7B; default method is **lora** after paired smokes), `LoRAConfig` (r=32, alpha=64, q/k/v/o + gate/up/down_proj), `TrainingConfig`, `wandb_project`/`wandb_run_name`, and `dataset_repo`/`model_repo`/`processed_profile_name` Hub-id helpers (adapter repos `amlk-{MODEL_SLUG}-sft[-variant]`). Self-contained job scripts keep twin fallbacks of `max_length` (must stay in sync).
* `training/train.py`: One trainer for all three regimes (`--method qlora|lora|full`). Trains with `completion_only_loss=True`, 1 epoch by default, logs to model-specific wandb with informative run names, saves the adapter; `--push-to-hub` or `--submit-hf` push to the Hub. Serializes resolved `TRAIN_CONFIG`/`LORA_CONFIG` JSON into the remote job env (so METHOD_PRESETS cannot be ignored). Full-run default flavor **a10g-small**, timeout **8h**. Chat-wraps prompts before SFT. Mid-run stability: creates the model repo before the job starts so `hub_strategy=every_save` can commit checkpoints while training. Cross-job resume: `--resume-from` / `--push-resume-checkpoint` (full Trainer ckpts via `training/resume.py`). Improvement-loop flags: `--dataset-repo`, `--skip-data-upload`, `--max-train`, `--test-subset`, `--skip-base-arm`, `--run-tag`, `--learning-rate`. Inference is NOT here for Gemini — that's `evaluation/predict.py`; dual-arm finetuned/base preds come from the cloud job.
* `training/train_hf_job.py`: Self-contained PEP 723 UV script submitted inline by `train.py --submit-hf`. Reads METHOD/VARIANT/BASE_MODEL/DATASET_REPO/OUTPUT_REPO/WANDB_*/EPOCHS/`TRAIN_CONFIG`/`LORA_CONFIG`/`RESUME_FROM` from env, trains on the cloud GPU (1 epoch default), then generates fine-tuned + zero-shot base test predictions. Chat-wraps train/val/infer for both arms; `add_special_tokens=False` on generate (no double-BOS); Hebrew-script `bad_words_ids` always on. Stability: `/data/output` resume, `hub_strategy=every_save`, immediate prediction uploads. Never run directly.
* `training/resume.py`: Pure helpers for **full** Trainer checkpoints (optimizer + `trainer_state` + weights/adapter) — list Hub dirs, pick `auto`/exact name, validate local dirs. Used by `train.py` for `--resume-from` and `--push-resume-checkpoint`. Inlined twins live inside `train_hf_job.py` (that script cannot import the repo).
* `evaluation/predict.py`: Generates the Gemini advanced-baseline summaries via API (no GPU, no model load), same hardened prompt as training. Resumes from a partial file. Fine-tuned and zero-shot base predictions come from the cloud training job, not here.
* `evaluation/gemini_client.py`: Shared Gemini API helpers (`GEMINI_MODEL`, `call_with_retry`). Also defines `strip_think()` — drops closed `<think>…</think>` reasoning blocks so metrics score the summary (used by evaluate.py and error_analysis.py; residual utility for chat-capable Qwen-family outputs).
* `evaluation/evaluate.py`: Scores a predictions file with raw + Hebrew-normalized ROUGE-1/2/L (`normalize_hebrew` strips niqqud + folds final-form letters), BERTScore (default `onlplab/alephbert-base`; `--bertscore-model` to override), and the faithfulness/fluency judge (`--skip-llm` to skip; `--limit N` for smoke). Applies `strip_think` before scoring. One JSON report per system.
* `evaluation/error_analysis.py`: Samples ~50 predictions (post `strip_think`) and has Gemini label failure types (hallucination, omission, entity/number error, lead copying, fluency), writing per-type rates.
* `evaluation/eval_hf_job.py`: Runs the whole D1 battery on HuggingFace Jobs so Gemini calls + BERTScore happen on the cloud's fast connection. `--submit-hf` uploads itself to a cheap CPU job; with no args it drives the existing `evaluation/` CLIs by subprocess and pushes `reports/*.json` (timeout-safe).
* `evaluation/build_report_tables.py`: Downloads pushed `reports/*.json` and assembles D1 markdown (quality table, failure-rate table, behavioural notes).
* `evaluation/infer.py`: GPU inference helpers — `load_finetuned_model` (base + LoRA adapter, **defaults to 4-bit** so 7B fits a Colab T4; `disable_adapter()` gives zero-shot base), `load_base_model`, and `generate_summaries` (chat-wrap via `format_chat_prompt`, `add_special_tokens=False`, Hebrew-script decode constraint). Twin of `train_hf_job.py` generation. **Remote GPU only — never call locally.**
* `evaluation/base_predict.py`: Pure helpers for multi-model zero-shot baselines (`resolve_load_plan`, `write_predictions_jsonl` / `validate_predictions`, `model_slug` / local paths). Re-exports chat helpers from `data.prompts`. No GPU.
* `evaluation/hebrew_constraint.py`: Decode constraint **always used** at generation. `build_bad_words_ids(tokenizer)` returns ids of tokens whose decoded form contains Latin/Cyrillic/Greek/Arabic/CJK/Hangul. Inlined as a twin in `train_hf_job.py`.
* `evaluation/topic_clustering.py`: Topic-clustering side-analysis (not main pipeline). Embeds article `text` by default with Hebrew-native embed model, BERTopic + Hebrew vectorizer, Gemini naming, optional mega-cluster refinement and duplicate-label merge. See `notebooks/cluster_topics_databricks.py`.
* `evaluation/style_labels.py`: Structural style labels (`single_sentence` / `multi_sentence` / `pipe_digest` / `question`) via pure regex — no GPU/API.
* `evaluation/stratify_by_topic.py`: Joins predictions to topic/style label artifacts and re-scores per group (CPU-only).
* `evaluation/viewer/`: Local Streamlit UI for browsing `predictions.jsonl` (RTL Hebrew, keyword search, multi-system compare). CPU-only, no API.
* `notebooks/evaluation_observation.ipynb`: Colab notebook for live per-example evaluation observation (finetuned/base/gemini).
* `notebooks/cluster_topics_databricks.py`: Databricks source-format notebook for topic + style clustering over the corpus.
* `scripts/run_nb_cell.py`: Agent cell-runner via `colab exec` for cell-by-cell observation.
* `IMPROVEMENT_PLAN.md`: Code audit (C0–C5) + training improvement plan (P0–P5: grounding, chat template, decode, metrics, 7B regime). Historical diagnosis; many code fixes are already applied on this branch.
* `docs/prompt-arena-notebook.md`: Lab notebook for the prompt-optimization loop that produced the current `PROMPT_TEMPLATE` (judge → compliance → ROUGE-L ranking). Prompt-arena **code** may live only on `another-model`; the winning prompt is already in `data/prompts.py`.

---

## Building and Running

**Prerequisites:**
* Python 3.10+
* `uv` package manager (used instead of pip — `uv` is on PATH)
* Install dependencies: `uv pip install -r requirements.txt` (or `uv sync` if using a lockfile)
* Fill in `.env`:
  * `HF_TOKEN` — HuggingFace access token (model download + HF Hub upload)
  * `GEMINI_API_KEY` — Gemini API key (advanced baseline + LLM-judge + error analysis)
  * wandb auth is read from `~/.netrc` (global); the HF job also needs `WANDB_API_KEY`, picked up automatically.
* Source the env and activate venv before running scripts: `source .env && source .venv/bin/activate`
* **Always invoke scripts as modules** (`python -m data.preprocess`, `python -m training.train`, …)
  so package imports resolve from the repo root. This is the one supported way to run them.
* **Never load or run a model on the local GPU** — this machine (8 GB) freezes. All
  training and inference for the 7B base run on **HuggingFace Jobs**. Local is only for:
  data curation materialize/preprocess (CPU), `pytest`, Gemini baseline + judge + error
  analysis (API), and BERTScore (pinned to CPU).

**Running the full pipeline:**
```bash
source .env && source .venv/bin/activate

# 0. Curated product once (from data_curation):
#    data_curation/artifacts/final_clean_hesum.json
#    or copy to outputs/data/curated/final_clean_hesum.json
#    (~5854 rows: hesum_id, text, headline)

# 1. Materialize curated source → curated_records.jsonl   [local, CPU]
python -m data.download --force

# 2. Build HF training dataset: hardened prompt + 80/10/10 + validate. --variant = probe input.
python -m data.preprocess --variant whole --force   # also: --variant lead | body
#    → outputs/data/processed/whole/{train,val,test}
#      columns: text, summary, source, prompt, completion

# 3. Train on HF Jobs (cloud GPU, 1 epoch, default --method lora). Prefer --skip-data-upload
#    when Hub already has current curated whole splits (avreymi/amlk-training-data).
#    Job also generates fine-tuned + zero-shot base test predictions.
#    Mid-run: hub_strategy=every_save; /data/output survives job restarts.
python -m training.train --submit-hf --hf-user avreymi --smoke-test --skip-data-upload
python -m training.train --submit-hf --hf-user avreymi --skip-data-upload   # full 1-epoch

# Cross-job resume (full Trainer ckpt with optimizer — not adapter-only Hub root):
# python -m training.train --push-resume-checkpoint /path/to/checkpoint-200 \
#   --output-repo avreymi/amlk-dictalm2-instruct-sft
# python -m training.train --submit-hf --hf-user avreymi --method lora --skip-data-upload \
#   --resume-from checkpoint-200   # or --resume-from auto

# 4. Full eval battery on HF Jobs (cheap CPU): Gemini baseline + ROUGE/BERTScore/judge/errors
python -m evaluation.eval_hf_job --submit-hf --hf-user avreymi --smoke-test
python -m evaluation.eval_hf_job --submit-hf --hf-user avreymi

# 5. Assemble D1 comparison tables  [local, no GPU/API]
python -m evaluation.build_report_tables --output outputs/results/d1-tables.md
```

**HuggingFace Jobs — submit and monitor:**
```bash
# --submit-hf uploads local processed/<variant>/ unless --skip-data-upload, then submits
# train_hf_job.py inline (a10g-small, 8h full-run timeout, 1-epoch default). Prints a Job ID.
python -m training.train --submit-hf --hf-user avreymi --skip-data-upload
python -m training.train --submit-hf --hf-user avreymi --smoke-test --skip-data-upload
python -m training.train --submit-hf --hf-user avreymi --inference-only  # regen preds from adapter
# Cost: a10g-small = same 24 GB A10G as a10g-large at $1.00/h vs $1.50/h.
# dictalm2.0-instruct is Mistral-7B; default method is lora (bf16) after 2026-07-26 smokes;
# qlora remains available if memory is tight.

hf jobs ps
hf jobs logs <job-id>         # add -f to stream
hf jobs inspect <job-id>

# Adapter (LoRA only, not merged): https://huggingface.co/avreymi/amlk-dictalm2-instruct-sft  (private)
# Training metrics: wandb project "amlk-dictalm2-instruct"; run names include date/method/variant/epochs.
# Dataset: avreymi/amlk-training-data (private, curated whole splits).
```

**Reading model outputs (predictions viewer):**
```bash
source .venv/bin/activate && streamlit run evaluation/viewer/app.py
```

**Running tests:**
```bash
source .venv/bin/activate && python -m pytest tests/ -v
```

---

## Status - remember to update it

**2026-07-28 — Training stack merged from `origin/another-model` into main (docs + skill).**
Primary base is **`dicta-il/dictalm2.0-instruct`** (not Qwen3-2B). Training data path is
**curated HeSum only** via main's `data_curation` → `data.download` → `data.preprocess` →
HF Jobs (`training/train.py` + `train_hf_job.py`). Key training behaviors on this tree:
chat-template wrap at train/infer, `TRAIN_CONFIG`/`LORA_CONFIG` env serialization,
Hebrew decode constraint always on, `MAX_LENGTH=4096`, `DEFAULT_EPOCHS=1`,
`DEFAULT_MAX_NEW_TOKENS=128`, default method **lora**, full-run **a10g-small / 8h**,
`/data/output` + Hub `every_save` stability, and `training/resume.py` cross-job resume.
**Main keeps `data_curation/`** as the source of `final_clean_hesum.json`. Qwen3-2B
pipeline notes below are **historical**. Do not treat a full 1-epoch DictaLM2 run as
completed unless a job ID and Hub adapter are recorded here after a real submission.

**2026-07-26 (from another-model — code/status carried with the training merge).**
GPU cost levers (`group_by_length`, `attn_implementation=sdpa`, val slice 100, gen batch
16 when 4-bit) and paired 10-step smokes promoted **lora** over qlora on a10g-small
(step_time 40.49s vs 55.33s). Curated Hub dataset `avreymi/amlk-training-data` documented
as 4683/585/586 (`source=hesum-curated`). Full 1-epoch train + D.1 eval on the curated
splits remains a **next step**, not a completed deliverable on main.

**2026-07-11 (from another-model — historical DictaLM2 branch work).**
C0–C5 audit fixes (chat template, double-BOS, no `/no_think` on Mistral, 4-bit load
default for Colab, TRAIN_CONFIG wiring, 8h timeout). Prompt-arena loop promoted the
Hebrew stop-cue `PROMPT_TEMPLATE` now in `data/prompts.py` (see
`docs/prompt-arena-notebook.md`). Clean-only / 1-epoch / mid-run Hub stability landed
on that branch. Earlier smokes (e.g. job `6a524384…`) predate the curated-only data
path and/or chat-template fix — re-smoke before a full epoch.

**Historical — Qwen/Qwen3-2B path (Stages A–B, 2026-06).**
Earlier main work fine-tuned `Qwen/Qwen3-2B` on raw/clean HeSum variants (IAHLT
inaccessible). Full QLoRA job `6a2bc974` trained successfully but timed out in
prediction; post-mortem in `docs/2026-06-12-qlora-training-job-postmortem.md`. V1 LoRA
only hit full-attention layers of hybrid Qwen3; later runs added MLP modules. D.1
tables for that era (where present under `outputs/results/`) are not the DictaLM2
baseline. Prefer the DictaLM2 + curated path for new runs.

**Side infrastructure still on main (unchanged by this docs merge):**
predictions viewer, topic clustering + Databricks notebook, style labels, stratification,
evaluation-observation Colab notebook, `eval_hf_job` / `build_report_tables`, Hebrew
ROUGE + AlephBERT BERTScore, optional HF judge client.

**Hub / wandb (current intent):**
- Dataset: `avreymi/amlk-training-data` (private, curated)
- Adapter: `avreymi/amlk-dictalm2-instruct-sft` (private; smoke: `…-smoke`)
- wandb: `amlk-dictalm2-instruct`
- Judge/baseline model family: Gemini `gemini-2.5-flash-lite` (self-preference caveat
  for the paper; HF judge path available via `hf_client.py`)

**Next steps:**
1. Smoke then full 1-epoch train on DictaLM2 + curated Hub data
   (`--skip-data-upload` when Hub is current; default method lora).
2. D.1 full eval battery (`evaluation.eval_hf_job`) + tables.
3. Positional-shortcut probe (whole model, inference ablations) — see TODO F /
   `docs/research-proposal-revised.md`.
4. Literature / paper framing; optional headline-style control (TODO G).

Final submission: **31.07**.

---

## Code Writing Rules
Do not create new documentation files (unless explicitly requested). Only update documentation via the `README` if necessary.

### File Header (Mandatory)
In the header of every code file, you **must** describe how that file relates to the **overall project architecture** and **code flow**.

Each code file **must** include a short description (no more than 4–5 sentences) that explains the following:
- Its role in the **big picture** (as defined in the **Project Structure** section).
- Its connection to the main **code flow** of the project.
- The intended **execution environment** (where this code will run, as defined in the **Project Goal** section).
- The skills, memory, shared docs are very important to continue working on the project. You have all these as live files and currently updating them is very very important. Remember to do it!
