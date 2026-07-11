## Project Goal

* **Description:** AMLK is a Hebrew **news** summarization research project. The goal is to fine-tune `dicta-il/DictaLM-3.0-1.7B-Base` on Hebrew journalism datasets (HeSum, IAHLT summarization_he), evaluate with ROUGE, BERTScore, and LLM-as-judge, and produce a research paper and presentation. Design choices are informed by **English summarization literature** (lead bias, metric limits, strong baselines) without re-running English experiments. Evaluation includes an **advanced-model baseline** (e.g. Gemini API on the same test set and prompt) so metrics can be interpreted against a stronger system. A **truncation / positional-shortcut probe** trains separate models on Whole text, Lead-only, and Body-only inputs. Optional **headline-style control** varies the instruction (short headline vs longer summary). **Error analysis** labels a sampled set of predictions for failure types common in the literature. Runs locally or on HuggingFace Jobs; all scripts are command-line Python.

---

## Project Structure - remember to update it when you make changes

* **Architecture:** The project is divided into three sequential pipelines:
  1. **Training pipeline** — downloads Hebrew summarization datasets (IAHLT summarization_he, HeSum), loads the `dicta-il/DictaLM-3.0-1.7B-Base` base model, and fine-tunes it using the HuggingFace `transformers`/`trl` stack. If local GPU is insufficient, the job is submitted to HuggingFace as a remote training job.
  2. **Evaluation pipeline** — runs fine-tuned and baseline checkpoints on the held-out test set: ROUGE, BERTScore, LLM-as-judge (Gemini), an advanced-model baseline on the same data, and systematic error analysis on a sampled subset.
  3. **Results & reporting** — aggregated metrics feed into the final paper and presentation.

* **Code Flow:**
  1. Dataset download & preprocessing → tokenised dataset saved to disk
  2. Model fine-tuning → checkpoint saved to disk / HF Hub
  3. Inference on test set → predictions saved to disk
  4. Evaluation scripts consume predictions → produce metric reports
  5. Reports feed the paper / presentation

---

## File Structure - remember to update it with the latest project information

```
/AMLK
├── .agents/
│   └── skills/
│       ├── colab-cli/SKILL.md             # Official Google Colab CLI usage for agent-safe remote runtimes
│       ├── coding-principles/SKILL.md    # Project-local coding standards
│       ├── training/SKILL.md             # AMLK training process (train.py, HF Jobs, wandb)
│       └── testing/SKILL.md              # AMLK testing philosophy
├── data/
│   ├── __init__.py
│   ├── download.py                       # Pipeline step 1: downloads & normalizes IAHLT+HeSum datasets
│   ├── prompts.py                        # build_prompt/PROMPT_TEMPLATE(_CLEAN) + make_variant — shared prompt/probe-variant source of truth
│   ├── clean.py                          # Opt-in --clean reference normalization: normalize_summary/is_roundup_digest/pipe_segments
│   └── preprocess.py                     # Pipeline step 2: prompt/completion pairs + probe variants + --clean profile, 80/10/10 split
├── training/
│   ├── __init__.py
│   ├── config.py                         # MODEL_ID, METHOD_PRESETS, LoRAConfig, TrainingConfig, WANDB_PROJECT, repo helpers
│   ├── train.py                          # Single trainer: --method qlora|lora|full, --variant, wandb, --submit-hf
│   └── train_hf_job.py                   # Self-contained UV script run by HF Jobs (submitted by train.py --submit-hf)
├── evaluation/
│   ├── __init__.py
│   ├── predict.py                        # Generate the Gemini advanced-baseline summaries (API only); strip_think() tool
│   ├── evaluate.py                       # ROUGE-1/2/L + BERTScore (xlm-roberta-large) + Gemini judge → one report
│   ├── error_analysis.py                 # Gemini-labelled failure-type rates on a ~50-sample
│   ├── eval_hf_job.py                    # Run the full eval battery on HF Jobs (cheap CPU): --submit-hf | cloud runner
│   ├── build_report_tables.py            # Assemble the per-system reports into the D1 markdown comparison tables
│   ├── infer.py                          # GPU inference helpers (load adapter + generate); used by the observation notebook
│   ├── hebrew_constraint.py              # Opt-in clean-profile decode constraint: build_bad_words_ids() bans non-Hebrew-script tokens
│   ├── topic_clustering.py               # Embed summaries + BERTopic cluster + Gemini-name topics + plot_clusters(); used by the Databricks notebook
│   ├── style_labels.py                   # Rule-based structural style labels (single/multi-sentence, pipe digest, question) — local, no GPU/API
│   ├── stratify_by_topic.py              # Break down a predictions file's ROUGE/BERTScore/failure rates by topic_label or style_label
│   └── viewer/                           # Predictions viewer (its own subfolder — a self-contained UI feature)
│       ├── __init__.py                   # Re-exports data.py's public functions
│       ├── data.py                       # Streamlit-free helpers: discover/load/keyword-search predictions.jsonl files
│       └── app.py                        # Local Streamlit UI: browse predictions.jsonl, RTL Hebrew, keyword search, multi-system compare
├── notebooks/
│   ├── evaluation_observation.ipynb      # evaluation-observation stage: live per-example view (summary/judge/errors) on Colab
│   └── cluster_topics_databricks.py      # Topic-clustering side-analysis: Databricks source-format notebook, GPU cluster
├── scripts/
│   ├── __init__.py
│   └── run_nb_cell.py                    # Drive notebook cells on a Colab session via colab-cli (agent cell-by-cell runner)
├── tests/
│   ├── __init__.py
│   ├── test_download.py                  # normalize_iahlt / normalize_hesum
│   ├── test_preprocess.py                # build_prompt / make_variant / split_dataset
│   ├── test_clean.py                     # normalize_summary / is_roundup_digest / pipe_segments / repo-naming profile helpers
│   ├── test_evaluation.py                # ROUGE-on-Hebrew, judge-reply parsing, failure rates (live test gated)
│   ├── test_stratify_by_topic.py         # join/grouping logic for topic and style stratification
│   ├── test_topic_clustering.py          # BERTopic fit + Gemini naming + plot (live test gated)
│   ├── test_style_labels.py              # rule-based style classification (pipe digest / question / sentence count)
│   └── test_viewer.py                    # predictions-viewer load/keyword-search/discovery logic
├── docs/
│   ├── obsidian/                         # Shared Obsidian vault (team research notes; open folder as vault)
│   ├── ANLP Project abstract.md          # Original submitted proposal (historical record — not updated for the DictaLM swap)
│   ├── research-proposal.md              # Original submitted proposal, prose form (historical record — not updated)
│   ├── research-proposal-revised.md      # Reviewer-updated probe design (current — kept in sync with the base model)
│   └── superpowers/
│       ├── specs/2026-05-26-training-pipeline-design.md
│       └── plans/2026-05-26-stage-a-training-pipeline.md
├── outputs/
│   ├── data/
│   │   ├── raw/combined.jsonl            # Merged normalized dataset — 10,000 records (gitignored)
│   │   └── processed/<variant>/          # Arrow splits train/ val/ test/ per probe variant (gitignored)
│   ├── checkpoints/                      # LoRA adapter / full model checkpoints (gitignored)
│   ├── results/                          # predictions.jsonl + evaluation/error-analysis reports (gitignored)
│   └── manual-dwonloaded/                # Manually downloaded predictions directory (gitignored)
│       ├── predictions-base.jsonl        # Base model test predictions (gitignored)
│       ├── predictions-finetuned.jsonl   # Fine-tuned model test predictions (gitignored)
│       └── compare.html                  # Simple side-by-side comparison HTML page
├── .venv/                                # Python virtual environment (gitignored)
├── .env                                  # HF_TOKEN, GEMINI_API_KEY — never commit
├── .gitignore
├── AGENTS.md
├── CLAUDE.md                             # Symlink → AGENTS.md
├── README.md
├── requirements.txt
└── TODO.md                               # Milestone tracker
```

* `data/download.py`: Downloads Hebrew summarization datasets (biunlp/HeSum; IAHLT/summarization_he inaccessible with current credentials), normalises to `{text, summary, source}`, writes `outputs/data/raw/combined.jsonl`.
* `data/prompts.py`: `PROMPT_TEMPLATE` (original) + `PROMPT_TEMPLATE_CLEAN` (hardened anti-elaboration/no-lists variant), `build_prompt(text, clean=)`, and `make_variant` (whole|lead|body). The single source of truth for prompt construction, reused by `data/preprocess.py` and `evaluation/predict.py`.
* `data/clean.py`: Reference cleaning for the opt-in `--clean` pipeline profile. `normalize_summary` rewrites HeSum's `"headline | headline"` pipe/bullet digests into natural prose; `is_roundup_digest`/`pipe_segments` flag the worst 3+-segment "media roundup" references for removal via `--drop-roundups`. Pure regex, no GPU/API.
* `data/preprocess.py`: Reads `combined.jsonl`, builds `(prompt, completion)` pairs for completion-only SFT, applies the `--variant whole|lead|body` truncation probe (`make_variant`), truncates each article to `MAX_LENGTH-256` tokens so the summary always survives (HeSum articles are long — median ~2500 tokens; without this, completion-only loss goes nan), splits 80/10/10, saves Arrow splits to `outputs/data/processed/<profile>/`. The opt-in `--clean` flag (+ `--drop-roundups`) routes references through `data/clean.py` and selects `build_prompt(clean=True)`, writing to a parallel `<variant>-clean[-drop]` dir so the original artifacts stay reproducible. `build_prompt`/`make_variant` are the single source of truth, reused by `evaluation/predict.py`.
* `training/config.py`: Shared constants: `MODEL_ID="dicta-il/DictaLM-3.0-1.7B-Base"`, `METHOD_PRESETS` (the qlora/lora/full deltas), `LoRAConfig` (r=32, alpha=64, q/k/v/o + gate/up/down_proj), `TrainingConfig`, `WANDB_PROJECT`, and the `dataset_repo`/`model_repo`/`processed_profile_name` Hub-id helpers (each takes `clean`/`drop_roundups` and appends `-clean[-drop]` via `_profile_suffix`).
* `training/train.py`: One trainer for all three regimes (`--method qlora|lora|full`). Trains with `completion_only_loss=True`, logs to wandb, saves the adapter; `--push-to-hub` or `--submit-hf` push to the Hub. `--base-model` swaps the base checkpoint (requires `--output-repo`, so a different base can never overwrite another's adapter); `--skip-data-upload` reuses the splits already on the Hub. `--clean`/`--drop-roundups` (with `--submit-hf`) target the clean-profile data/adapter repos instead of the original ones. Inference is NOT here — that's `evaluation/predict.py`.
* `training/train_hf_job.py`: Self-contained PEP 723 UV script submitted inline by `train.py --submit-hf`. Reads METHOD/VARIANT/BASE_MODEL/DATASET_REPO/OUTPUT_REPO/WANDB_PROJECT/CLEAN from env, trains on the cloud GPU, then generates fine-tuned + zero-shot base test predictions (PEFT `disable_adapter`) and pushes the adapter + `predictions-finetuned.jsonl` / `predictions-base.jsonl` to the Hub. Prints `print_trainable_parameters()` before training so a base-model swap can't silently regress LoRA layer coverage. `BASE_MODEL` defaults to `MODEL_ID` — duplicated from `config.py` on purpose (the script is submitted inline and can't import repo code); keep in sync. Under `CLEAN=1`, `build_input_text` appends a `/no_think` soft switch on chat-capable bases (a no-op on DictaLM's no-chat-template base) and generation applies an inlined Hebrew-script `bad_words_ids` constraint (twin of `evaluation/hebrew_constraint.py`, since this script can't import repo code). Training checkpoints save to `/data/output` — `run_uv_job` auto-mounts a per-job bucket at `/data` to ship this script to the container (`HfApi._create_uv_command_env_and_secrets`), and that bucket, unlike the container's local disk, survives an infra-level restart of the same job, so `trainer.train()` auto-resumes from the last checkpoint there instead of silently restarting from step 0. Never run directly.
* `evaluation/predict.py`: Generates the Gemini advanced-baseline summaries via API (no GPU, no model load), same prompt as training (`--clean` selects the hardened clean-profile prompt). Resumes from a partial file. The fine-tuned and zero-shot predictions come from the cloud training job, not here.
* `evaluation/gemini_client.py`: Shared Gemini API helpers (`GEMINI_MODEL`, `call_with_retry`). Also defines `strip_think()` — the shared tool that drops closed `<think>…</think>` reasoning blocks (emitted by chat-capable Qwen3-family models) so metrics score the summary, not the reasoning (used by evaluate.py and error_analysis.py).
* `evaluation/evaluate.py`: Scores a predictions file with raw + Hebrew-normalized ROUGE-1/2/L (`normalize_hebrew` strips niqqud + folds final-form letters), BERTScore (default `onlplab/alephbert-base`, the HeSum backbone; `--bertscore-model` to override), and the Gemini faithfulness/fluency judge (`--skip-llm` to skip; `--limit N` to cap for a smoke run). Applies `strip_think` before scoring. One JSON report per system.
* `evaluation/error_analysis.py`: Samples ~50 predictions (post `strip_think`) and has Gemini label failure types (hallucination, omission, entity/number error, lead copying, fluency), writing per-type rates.
* `evaluation/eval_hf_job.py`: Runs the whole D1 battery on HuggingFace Jobs so the ~4000 Gemini calls + BERTScore happen on the cloud's fast connection (the user has weak internet). One file, two modes: `--submit-hf` uploads itself to a cheap CPU job; with no args (how HF Jobs invokes it) it fetches the public repo + Hub predictions/dataset and drives the existing `evaluation/` CLIs by subprocess, pushing each report to the model repo under `reports/` as it finishes (timeout-safe). `--clean`/`--drop-roundups` route the job at the clean-profile data/model repos and the hardened Gemini prompt (`DATA_PROFILE`/`CLEAN`/`DROP_ROUNDUPS` env vars).
* `evaluation/build_report_tables.py`: Downloads the pushed `reports/*.json` and assembles the D1 markdown — a quality table (ROUGE/BERTScore/judge), a failure-rate table, and behavioural notes (base `<think>`/language leakage, fine-tuned repetition, judge self-preference caveat).
* `evaluation/infer.py`: GPU inference helpers — `load_finetuned_model` (base + LoRA adapter, `disable_adapter()` gives the zero-shot base) and `generate_summaries` (batched greedy decode over a processed split, `clean=` enables the clean-profile `/no_think` + Hebrew-script decode toggles). The importable twin of `train_hf_job.py`'s inline generation block (that cloud script can't import repo code); keep the two in sync. **Remote GPU only — never call locally.**
* `evaluation/hebrew_constraint.py`: Optional clean-profile decode constraint. `build_bad_words_ids(tokenizer)` scans the vocab once and returns the ids of every token whose decoded form contains a Latin/Cyrillic/Greek/Arabic letter, for `generate(bad_words_ids=...)` to forbid emitting foreign scripts mid-word. Experimental (can hurt fluency), on only under the clean profile. Inlined as a twin in `train_hf_job.py` (that script can't import repo code).
* `evaluation/topic_clustering.py`: Topic-clustering side-analysis (not part of the main pipeline). Embeds truncated article `text` by default (`embed_field='text'`) — summaries alone collapsed ~99% of docs into one media-meta mega-topic — with the Hebrew-native, clustering-tuned `dicta-il/neodictabert-bilingual-embed`, clusters with BERTopic (UMAP + HDBSCAN + Hebrew-only c-TF-IDF vectorizer, `HEBREW_STOPWORDS` + `MEDIA_STOPWORDS` + `BOILERPLATE_STOPWORDS`), names each real cluster with one Gemini call, then optionally `refine_large_clusters()` — a second finer HDBSCAN pass on any cluster holding ≥30% of docs (re-uses embeddings; splits e.g. the politics mega-topic into ביטחון/כלכלה/חברה sub-domains without re-fragmenting sports/legal), then `merge_duplicate_labels()` collapses any clusters Gemini still named identically (on by default via `cluster_dataset(merge_duplicates=True)`) so the report has one row per distinct real-world topic. `fit_topics` tunables: `min_cluster_size`/`min_samples` (default 60/15 — coarser granularity means fewer near-duplicate sub-clusters of the same domain), `outlier_threshold` (only reassign noise above cosine sim — default 0.35; 0 floods the largest cluster), optional `nr_topics` merge (off by default; `auto` over-merged), `language='multilingual'` (required — English mode strips Hebrew). `plot_topic_sizes()` renders a bar chart of `topic_summary()` for the notebook. Output `topics.jsonl` still keyed by `summary` for stratification join. See `notebooks/cluster_topics_databricks.py`.
* `evaluation/style_labels.py`: A second, independent per-summary dimension from topic clustering — not *what topic* an article is about but *what format* its summary takes (`single_sentence` / `multi_sentence` / `pipe_digest` / `question`). Pure regex (`classify_style`), no embeddings/GPU/API, so unlike topic clustering it never needs Databricks and has no `datasets` import (works even if that import is broken locally, see the lzma note below). Motivated by a real corpus pattern: ~26% of HeSum summaries are `"headline | headline | headline"` pipe-separated digests — a format quirk worth tracking once a model is trained on this data. Produces the same `{summary: label}` artifact shape as `topic_clustering.py` so it plugs into the same stratification tool; `plot_style_distribution()` renders a bar chart of `style_summary()` for the notebook.
* `evaluation/stratify_by_topic.py`: Joins a predictions file to a label artifact (`topics.jsonl`'s `topic_label` from `topic_clustering.py`, or `style_labels.jsonl`'s `style_label` from `style_labels.py` — same shape, selected via `--label-field`) on exact `reference`==`summary` text match, and reuses `evaluate.py`'s `compute_rouge`/`compute_bertscore` per group, folding in per-group failure rates if a matching `*.errors.json` exists. Local, CPU-only — no GPU/Databricks needed for this step.
* `evaluation/viewer/`: A local, read-only UI for browsing `predictions.jsonl` files (article/prediction/reference), filling the gap between raw jsonl and the live Colab notebook. `data.py` has the Streamlit-free data logic (`discover_predictions_files`, `load_predictions` — applies `strip_think`, `filter_by_keyword`, `common_length`), importable from a notebook/REPL; `__init__.py` re-exports it; `app.py` is the thin Streamlit script (`streamlit run evaluation/viewer/app.py`) that renders Hebrew right-to-left, supports keyword search, and compares 2+ systems side-by-side for the same article. Local, CPU-only, no GPU/API. See `docs/superpowers/specs/2026-07-04-predictions-viewer-design.md`.
* `notebooks/evaluation_observation.ipynb`: The **evaluation-observation** stage. A self-bootstrapping Colab notebook that runs the *real* evaluation functions live and **displays** the per-example process (article → model summary → reference → judge faithfulness/fluency → error-analysis failure labels) for finetuned/base/gemini. Loads existing Hub predictions (finetuned/base at repo root, gemini under `reports/`) and generates fresh summaries on a T4. Judge/error/browse cells are API+CPU; only the generation cell needs a GPU.
* `notebooks/cluster_topics_databricks.py`: Databricks source-format notebook (`# Databricks notebook source` / `# COMMAND ----------` cell markers) driving `evaluation/topic_clustering.py` and `evaluation/style_labels.py`. Manual, occasional run on a Databricks GPU cluster — the GPU is for speed, not required (the embedding model is 0.4B params, encoder-only, the same class of job as the local-CPU AlephBERT BERTScore step). Clones the repo (or reuses an uploaded Workspace copy) so it calls the same tested functions rather than duplicating logic; computes both `topic_label` (BERTopic) and `style_label` (regex) over the same records. Plots (all inline via `displayHTML`, small enough to skip the DBFS round-trip): a `plot_topic_sizes` cluster-size bar chart, an interactive 2D/3D cluster scatter (`plot_clusters(dimensions=2|3)`, `plot_dimensions` widget; written to DBFS + iframe-embedded since 10k-doc hovers exceed the ~20 MB cell-output cap), a `plot_style_distribution` bar chart, and a topic×style stacked bar chart alongside the crosstab table. Writes one `topics.jsonl`/`topics-summary.json` (carrying both label fields) to DBFS for manual download into `outputs/data/raw/` and `outputs/results/`. Widgets expose `min_cluster_size`/`min_samples`/`reduce_outliers`/`nr_topics`/`merge_duplicate_labels`/`topic_size_plot_top_n`/`plot_dimensions` so noise/near-duplicate-topic tuning (see `topic_clustering.py`) doesn't require editing the notebook. A scoped, one-off departure from AMLK's default local/HF-Jobs/Colab stack — no agent-driven Databricks deployment (no MCP connection today), the notebook is handed off for manual import/run.
* `scripts/run_nb_cell.py`: Agent cell-runner — reads the notebook with `nbformat` and execs a chosen code cell / range against a persistent Colab session via `colab exec` (the Colab CLI has no native `.ipynb` runner). `--list` shows cell indices; the caller owns `colab new`/`stop`. This is how an agent observes the eval cell-by-cell.
* `tests/`: 60 fast behavioral tests + 2 gated live tests (Gemini judge; BERTopic fit + Gemini topic naming + plot), all passing (except the local `plotly`-not-installed gap, see Status).

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
* **Never load or run a model on the local GPU** — this machine (8 GB) freezes. All model
  training and inference run on **HuggingFace Jobs**. Local is only for: data download/preprocess
  (CPU), `pytest`, the Gemini baseline + judge + error analysis (API), and BERTScore (pinned to CPU).

**Running the full pipeline:**
```bash
source .env && source .venv/bin/activate

# 1. Download datasets  →  outputs/data/raw/combined.jsonl (10,000 records)   [local, CPU]
python -m data.download

# 2. Preprocess: prompt/completion pairs + 80/10/10 split. --variant selects the probe input.  [local, CPU]
python -m data.preprocess --variant whole        # also: --variant lead | body

# 3. Train on HF Jobs (cloud GPU). The job also generates fine-tuned + zero-shot base test
#    predictions and pushes predictions-finetuned.jsonl / predictions-base.jsonl to the model repo.
python -m training.train --submit-hf --hf-user avreymi --smoke-test   # verify first (~$0.05)
python -m training.train --submit-hf --hf-user avreymi                # full run

# 4. Run the whole eval battery on HF Jobs (cheap CPU). Generates the Gemini baseline and scores
#    all 3 systems (finetuned/base/gemini) with ROUGE + BERTScore + judge + error analysis, pushing
#    reports/*.json to the model repo. Done on the cloud so the ~4000 Gemini calls + BERTScore are
#    off the user's weak local connection.
python -m evaluation.eval_hf_job --submit-hf --hf-user avreymi --smoke-test   # 5 examples, verify first (~pennies)
python -m evaluation.eval_hf_job --submit-hf --hf-user avreymi                # full run (cpu-upgrade, ~$0.10-0.30)

# 5. Assemble the D1 comparison tables (downloads the tiny report JSONs):  [local, no GPU/API]
python -m evaluation.build_report_tables --output outputs/results/d1-tables.md

# (Local alternative to step 4, if you have a fast connection: run the scripts directly —
#  evaluation.predict for the Gemini baseline, then evaluation.evaluate / evaluation.error_analysis
#  on each of predictions-finetuned.jsonl / predictions-base.jsonl / the gemini file.)
```

**HuggingFace Jobs — submit and monitor:**
```bash
# --submit-hf uploads outputs/data/processed/<variant>/ to the Hub (avreymi/amlk-training-data[-<variant>])
# then submits train_hf_job.py inline (a10g-large, 6h, 3-epoch training by default). It prints a Job ID.
python -m training.train --submit-hf --hf-user avreymi               # full run
python -m training.train --submit-hf --hf-user avreymi --smoke-test  # 10 steps, a10g-small, ~$0.05 — verify first
python -m training.train --submit-hf --hf-user avreymi --inference-only  # regen predictions from pushed adapter (a10g-small, 1h)
# Cost: a10g-small has the SAME 24 GB A10G GPU as a10g-large at $1.00/h vs $1.50/h —
# prefer a10g-small; prefer --method lora over qlora for a model this size (~1.7-2B params).

hf jobs ps                    # list recent jobs
hf jobs logs <job-id>         # snapshot; add -f to stream
hf jobs inspect <job-id>

# Trained adapter (LoRA only, not merged) pushes to: https://huggingface.co/avreymi/amlk-dictalm3-1.7b-sft  (private)
# Evaluation loads it via: predict.py --model finetuned --adapter avreymi/amlk-dictalm3-1.7b-sft
# Training metrics: wandb project "amlk-hebrew-summarization".
```

**Reading model outputs (predictions viewer):**
```bash
source .venv/bin/activate && streamlit run evaluation/viewer/app.py
# Opens a local browser UI over outputs/results/*.jsonl: RTL Hebrew, keyword search,
# side-by-side comparison across systems (finetuned/base/gemini). Local, CPU-only, read-only.
```

**Running tests:**
```bash
source .venv/bin/activate && python -m pytest tests/ -v
```

---

## Status - remember to update it

**Pre-training stage as of 2026-07-09.** The base model is `dicta-il/DictaLM-3.0-1.7B-Base`; the
pipeline (data → train → evaluate → report) is built and its mechanics are validated end-to-end,
but **no full training run has happened yet.** Stack: trl 1.6.0, transformers 5.11, peft 0.19,
wandb 0.27.
- `data/download.py` — 10,000 records from biunlp/HeSum in `outputs/data/raw/combined.jsonl`. IAHLT/summarization_he is inaccessible (not on HF Hub with current credentials).
- `data/preprocess.py` — prompt/completion pairs + `--variant whole|lead|body`; 8,000/1,000/1,000 splits in `outputs/data/processed/<variant>/`.
- `training/train.py` — one trainer for qlora|lora|full, `completion_only_loss=True`, wandb logging, `--submit-hf` to HF Jobs.
- `training/train_hf_job.py` — self-contained HF Jobs script (trl 1.6 API, wandb).
- `evaluation/predict.py` / `evaluate.py` / `error_analysis.py` — full metric battery (ROUGE/BERTScore/Gemini judge), zero-shot + Gemini baselines, failure-type analysis. `strip_think()` (in `gemini_client.py`) drops closed `<think>…</think>` reasoning before scoring; evaluate.py has `--limit` for smoke runs.
- `evaluation/eval_hf_job.py` + `build_report_tables.py` — D1 eval runs on a cheap CPU HF Job (clones the public repo, drives the existing CLIs, pushes `reports/*.json`); the tables tool turns those reports into the presentation markdown. Chosen because the user has weak internet (the ~4000 Gemini calls + BERTScore run cloud-side).
- Test suite: 64 tests (fast behavioral + 2 gated live: Gemini judge, BERTopic fit + naming), all passing except a local `plotly`-not-installed gap in 3 plot tests (`python -m pytest tests/`).
- HF Jobs dataset (uploaded, reusable across base models): `avreymi/amlk-training-data` (private). Model output repo (once a real training run pushes an adapter): `avreymi/amlk-dictalm3-1.7b-sft` (private). wandb project: `amlk-hebrew-summarization`. Advanced baseline + judge: Gemini `gemini-2.5-flash-lite` (full 2.5-flash's ~7s/call thinking latency made the ~4000-call battery ~10h; -lite is ~1s/call, ~6x faster. 2.0-flash is retired.)
- Note: QLoRA `push_to_hub` saves the LoRA adapter only (not merged) — evaluation loads base + adapter via `PeftModel.from_pretrained` (handled in `predict.py`).
- Note: the Gemini LLM-judge and the Gemini advanced baseline are the same model family — flag the possible self-preference bias in the paper.
- `train_hf_job.py` / `evaluation/infer.py` share a decode config: `max_new_tokens=256` (p99 reference length is 187 tokens), `min_new_tokens=16`, `no_repeat_ngram_size=3`, `repetition_penalty=1.2`, explicit `eos_token_id`/`pad_token_id`, greedy. Predictions are pushed immediately after each generation loop (timeout-safe), progress every 10 batches; inference-only jobs use a 1h timeout. `PRED_SUFFIX` env (`--pred-suffix`) appends a suffix so a re-decode doesn't clobber an earlier prediction file.

**2026-07-09 — Diagnosed and fixed a training-checkpoint-loss bug found while verifying the first
real `--clean` LoRA run (job `6a4f55731fba25b8ea3b310b`, 1 epoch).** The job's underlying container
was restarted at the infra level partway through training (confirmed via wandb: the first run,
`afn9wzvk`, reached step 390/500 with a healthy loss curve then died silently around 4h in with no
Python traceback; the retry's logs show the *entire* `uv` venv — torch, CUDA libs, everything —
being reinstalled from scratch, i.e. a full container wipe, not a script-level exception). Root
cause of the wasted progress: `train_hf_job.py`'s `SFTConfig(output_dir="./output")` wrote
checkpoints to the container's ephemeral local disk, which the restart wiped, and the script never
called `resume_from_checkpoint`, so the retry silently restarted training from step 0. Compounding
factor: the job's `running_secs` exceeded its declared `6h` timeout (`train.py`'s full-run flavor)
while still mid-retry — matching a precedent already noted below (job `6a3fa247` also ran past its
declared timeout and still completed) — so timeout enforcement on this account is not reliable
either way. Fixed: `output_dir` now points at `/data/output`. `/data` is a bucket
(`avreymi/jobs-artifacts`) that `run_uv_job` auto-mounts to ship this script into the container
(`HfApi._create_uv_command_env_and_secrets`, confirmed by reading its source — "Local files are
shipped to the job via a bucket mounted at /data"), scoped to a per-job subfolder; unlike local
disk, that bucket survives an infra-level restart of the same job. (A same-turn detour briefly
"corrected" this to a Hub-round-trip approach after grepping `train.py` for `volumes=` and finding
none passed explicitly — that grep missed that `run_uv_job` auto-injects the mount regardless of
what the caller passes; `hf jobs inspect` on both this job and an earlier one confirmed the real
`volumes` entry at `/data`, so the original `/data/output` fix was correct all along and the
detour was reverted.) `trainer.train()` now checks for an existing `checkpoint-*` under
`/data/output` and passes `resume_from_checkpoint=True` when found. A brand-new job submission
gets its own fresh bucket subpath, so this can't cross-contaminate between unrelated runs — only
retries of one job see one another's checkpoints. The job that exposed this (job
`6a4f55731fba25b8ea3b310b`) was canceled rather than left to finish, since a second full restart
was already ~20 min past its own timeout with ~2.5h of training still left; this corrected fix
hasn't been tested against a real restart yet — worth watching the next full submission.

**2026-07-09 — Ported the opt-in "clean" pipeline profile from `main` (`--clean`/`--drop-roundups`),
keeping this branch pinned to DictaLM.** `main` added a parallel pipeline profile addressing the
finding that the model learns HeSum's `"headline | headline"` digest style and hallucinates:
`data/clean.py` (`normalize_summary`/`is_roundup_digest`) rewrites pipe/bullet references into
prose and flags 3+-segment "media roundup" digests for removal; `data/prompts.py` adds a hardened
`PROMPT_TEMPLATE_CLEAN` via `build_prompt(text, clean=)`; `evaluation/hebrew_constraint.py` adds
an optional `bad_words_ids` decode constraint banning non-Hebrew-script tokens. Threaded through
`data/preprocess.py`, `training/train.py`/`train_hf_job.py`, and
`evaluation/predict.py`/`eval_hf_job.py`/`infer.py` via `--clean`/`--drop-roundups`, with a
`-clean[-drop]` suffix on `dataset_repo`/`model_repo`/`processed_profile_name` (`training/config.py`)
so the original raw pipeline stays byte-for-byte reproducible. `main`'s version of this feature was
still on `Qwen/Qwen3-2B` (it treated DictaLM as a to-do "top candidate" swap); ported here with all
repo-naming (`amlk-dictalm3-1.7b-sft`, not `amlk-qwen3-2b-sft`) and model references kept on
`dicta-il/DictaLM-3.0-1.7B-Base` — this branch already completed that swap (see the 2026-07-09 entry
below). The base-model-specific `/no_think` reinforcement in `build_input_text` is guarded by the
existing `tokenizer.chat_template` check, so on DictaLM (no chat template) it's inert; the
Hebrew-script `bad_words_ids` constraint is vocab-based and applies regardless of base model. Not a
git merge — the main-branch diff was read and re-applied by hand. 9 new tests in `tests/test_clean.py`
(64 total, all passing).

**2026-07-09 — Base model switched to `dicta-il/DictaLM-3.0-1.7B-Base`, and pipeline validated
against it on HF Jobs (job `6a4f43e4`, COMPLETED in 4m03s, a10g-small, `--method lora`).**
`dicta-il/DictaLM-3.0-1.7B-Base` is a plain `Qwen3ForCausalLM` (28 dense layers, hidden 2048, GQA
16/8, Qwen3 vocab 151936) — not a hybrid linear-attention architecture, so the existing
`LoRAConfig` target modules (q/k/v/o + gate/up/down) attach to **all 28 layers** out of the box:
`trainable params: 34,865,152 || 1.9861%`. (Some other Qwen3 variants, e.g. `Qwen/Qwen3-2B`, mix
linear- and full-attention layers, where the same target_modules list would only cover the
full-attention subset — not a concern for the current default, but worth checking via
`print_trainable_parameters()` if `--base-model` is ever swapped to one.) Smoke run: loss
1.469→1.439, eval_loss 1.791→1.772 (finite, no nan), all 10 steps, adapter + predictions pushed to
`avreymi/amlk-dictalm3-1.7b-smoke` (a throwaway smoke-test repo, distinct from the real
`amlk-dictalm3-1.7b-sft` output repo). After only 10 steps the fine-tuned output is already fluent
Hebrew in the corpus's `pipe_digest` style — this is a pipeline-mechanics check, not a quality
result. Two integration facts: (1) it ships **no chat template** (pure base checkpoint), so
`build_input_text()` falls back to the raw prompt when `tokenizer.chat_template` is falsy — the
zero-shot `base` system therefore can't be put into "assistant mode" and is expected to drift off
Hebrew; (2) the processed splits are base-model independent (they store text, tokenized at train
time by SFTTrainer) and the Qwen3 vocab is unchanged, so `avreymi/amlk-training-data` is reusable
as-is (`--skip-data-upload`). Reproduce the smoke test with:
`python -m training.train --submit-hf --hf-user avreymi --method lora --smoke-test --output-repo avreymi/amlk-dictalm3-1.7b-smoke --skip-data-upload`

**2026-07-04 — More distinct cluster plot + tighter clustering defaults.** Plot: golden-angle color palette, UMAP `min_dist=0.35`/`spread=1.25`, optional centroid repulsion (`plot_display_spread` widget). Clustering: `min_samples` 15→20, `umap_n_neighbors` 10→15, `outlier_threshold` 0.35→0.40; `umap_n_neighbors` widget on Databricks.

**2026-07-04 — Optional 3D cluster plot.** `plot_clusters(..., dimensions=2|3)` adds a rotatable 3D UMAP view (convex-hull mesh clouds + centroid text labels); Databricks widget `plot_dimensions` defaults to `2`. 2D remains the default for the iframe embed.

**2026-07-04 — Refinement coarsened after 60+ cluster explosion.** The first refinement pass (25/8, no topic cap) split the politics mega-cluster into 60+ near-duplicate "תקשורת ו…" Gemini labels. Defaults now: `refine_min_cluster_size=100`, `refine_min_samples=20`, `refine_nr_topics=12` (BERTopic merge cap on the refinement pass only), stricter refinement naming prompt forbidding "תקשורת/עיתונות/…" meta-labels. Expect ~15–20 topics total (5 pass-1 + ~12 politics sub-domains). Set `refine_oversized=False` to keep ~6 pass-1 topics only.

**2026-07-04 — Two-stage mega-cluster refinement.** Pass 1 still uses coarse HDBSCAN (60/15) for stable top-level domains; `refine_large_clusters()` (on by default, `refine_oversized=True`) re-clusters any topic holding ≥30% of docs with finer settings on the *same* embeddings and a sub-domain Gemini naming prompt — splits the ~7.6k politics blob without re-embedding or re-fragmenting sports/legal. Databricks widgets: `refine_oversized`, `refine_size_fraction`, `refine_min_cluster_size`, `refine_min_samples`, `refine_nr_topics`.

**2026-07-04 — Topic-clustering "fewer, more distinct topics" fix (v3) + notebook plots.** The v2 fix (embed on `text`) still surfaced too many near-duplicate topics (e.g. "תקשורת ומדיה"/"תקשורת וטלוויזיה") once mega-topic collapse was fixed, driven by (a) layout/journalism-meta keywords ("כותרת", "הבוקר", "העיתון"...) dominating c-TF-IDF instead of real subject words, and (b) fine HDBSCAN granularity producing several sub-clusters of the same domain that Gemini then named identically. Fixed in `evaluation/topic_clustering.py`: a `BOILERPLATE_STOPWORDS` set added to the vectorizer; `min_cluster_size`/`min_samples` raised 25/5 → 60/15 (coarser HDBSCAN); a new `merge_duplicate_labels()` post-processing step (on by default, `cluster_dataset(merge_duplicates=True)`) that collapses any clusters Gemini still named identically into one reported topic, keeping the smallest `cluster_id` and the union of keywords — no extra Gemini calls. Databricks widgets: `min_cluster_size`/`min_samples` defaults updated, new `merge_duplicate_labels` toggle. Also added inline Plotly charts to the notebook pipeline (`plot_topic_sizes` in `topic_clustering.py`, `plot_style_distribution` in `style_labels.py`, plus a topic×style stacked bar) alongside the existing big document scatter — small aggregate charts shown directly with `displayHTML(fig.to_html(...))`, no DBFS round-trip needed since they're nowhere near the ~20 MB cell-output cap.

**2026-07-04 — Topic-clustering granularity fix (v2).** After the noise/vectorizer fixes, a second full run collapsed ~99% of docs into one "חדשות ותקשורת" mega-topic — caused by clustering on summaries (outlet-name headlines), `outlier_threshold=0` (force-assign all noise to the largest cluster), and `nr_topics='auto'` over-merging. Defaults now: `embed_field='text'` (first 4k chars of article body), `outlier_threshold=0.35`, `nr_topics=None`, `min_cluster_size=25`/`min_samples=5`, media-outlet stopwords + domain-focused Gemini naming prompt. Databricks widgets updated (`embed_field`, `outlier_threshold`, `max_embed_chars`; `nr_topics` blank by default).

**2026-07-04 — Topic-clustering quality fix (v1).** The first full 10k-doc Databricks run put 51% of docs in the noise cluster (-1) and produced near-duplicate topic names (e.g. "תקשורת ומדיה" / "תקשורת וטלוויזיה") whose c-TF-IDF keywords were mostly years/IDs/Latin site names (`ynet`, `nrg`, `bbc`) — BERTopic's default English-tuned vectorizer let non-Hebrew tokens dominate. Fixed in `evaluation/topic_clustering.py`: a Hebrew-only `CountVectorizer` (`_build_vectorizer`/`HEBREW_TOKEN_PATTERN`/`HEBREW_STOPWORDS`), `min_samples` decoupled from `min_cluster_size` (per BERTopic's FAQ, reduces raw noise), and two opt-out BERTopic post-processing passes — `reduce_outliers` (embedding-similarity reassignment of noise docs) and `nr_topics="auto"` (HDBSCAN-over-topic-vectors merging of only genuinely similar topics). All exposed as Databricks widgets (`min_cluster_size` default lowered 100→40, `min_samples`, `reduce_outliers`, `nr_topics`) so re-tuning doesn't require editing the notebook. 3 new fast unit tests cover the Hebrew token pattern/stopwords.

**2026-07-04 — Predictions viewer added.** `evaluation/viewer/` (`data.py` + `app.py`, its own subfolder): a local Streamlit app (`streamlit run evaluation/viewer/app.py`) for browsing `outputs/results/*.jsonl` — RTL Hebrew rendering, keyword search, side-by-side comparison across systems. Read-only, CPU-only, no GPU/API. Verified end-to-end against the real `predictions-finetuned.jsonl`/`predictions-base.jsonl` files with `streamlit.testing.v1.AppTest` (file discovery, multi-file compare, keyword filtering, navigation — no exceptions). Design: `docs/superpowers/specs/2026-07-04-predictions-viewer-design.md`.

**Next steps:**
1. **Full training run on `dicta-il/DictaLM-3.0-1.7B-Base`** — the smoke test validated the
   pipeline; a real multi-epoch run (`--method lora`, a10g-small) has not happened yet.
2. **D.1 — full eval battery** on the trained adapter (`evaluation.eval_hf_job --submit-hf`),
   scoring finetuned / zero-shot base / Gemini advanced baseline with ROUGE + BERTScore + judge +
   error analysis, assembled via `evaluation.build_report_tables`.
3. **Truncation probe** — train/evaluate whole / lead / body variants (`--variant` is ready in
   `data/preprocess.py` and `training/train.py`).
4. **Literature (English summarization)** — document lessons from English news summarization in the paper (lead bias, ROUGE limits, baseline practices).
5. **Journalism / headline control (optional)** — alternate instruction templates for headline-length vs longer summaries; see `TODO.md` H.

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
