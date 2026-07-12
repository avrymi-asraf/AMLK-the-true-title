## Project Goal

* **Description:** AMLK is a Hebrew **news** summarization research project. The goal is to fine-tune `dicta-il/dictalm2.0-instruct` on Hebrew journalism datasets (HeSum, IAHLT summarization_he), evaluate with ROUGE, BERTScore, and LLM-as-judge, and produce a research paper and presentation. Design choices are informed by **English summarization literature** (lead bias, metric limits, strong baselines) without re-running English experiments. Evaluation includes an **advanced-model baseline** (e.g. Gemini API on the same test set and prompt) so metrics can be interpreted against a stronger system. A **truncation / positional-shortcut probe** trains separate models on Whole text, Lead-only, and Body-only inputs. Optional **headline-style control** varies the instruction (short headline vs longer summary). **Error analysis** labels a sampled set of predictions for failure types common in the literature. Runs locally or on HuggingFace Jobs; all scripts are command-line Python.

---

## Project Structure - remember to update it when you make changes

* **Architecture:** The project is divided into three sequential pipelines:
  1. **Training pipeline** — downloads Hebrew summarization datasets (IAHLT summarization_he, HeSum), loads the `dicta-il/dictalm2.0-instruct` base model, and fine-tunes it using the HuggingFace `transformers`/`trl` stack. If local GPU is insufficient, the job is submitted to HuggingFace as a remote training job.
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
│   ├── prompts.py                        # build_prompt/PROMPT_TEMPLATE + make_variant — shared prompt/probe-variant source of truth
│   ├── clean.py                          # Always-on reference normalization: normalize_summary/is_roundup_digest/pipe_segments
│   └── preprocess.py                     # Pipeline step 2: clean refs + hardened prompt + probe variants, 80/10/10 split
├── training/
│   ├── __init__.py
│   ├── config.py                         # MODEL_ID, METHOD_PRESETS, LoRAConfig, TrainingConfig, wandb_* naming, repo helpers
│   ├── train.py                          # Single trainer: --method qlora|lora|full, --variant, 1 epoch, --submit-hf
│   └── train_hf_job.py                   # Self-contained UV script run by HF Jobs (submitted by train.py --submit-hf)
├── evaluation/
│   ├── __init__.py
│   ├── predict.py                        # Generate the Gemini advanced-baseline summaries (API only); strip_think() tool
│   ├── evaluate.py                       # ROUGE-1/2/L + BERTScore (alephbert-base) + Gemini judge → one report
│   ├── error_analysis.py                 # Gemini-labelled failure-type rates on a ~50-sample
│   ├── eval_hf_job.py                    # Run the full eval battery on HF Jobs (cheap CPU): --submit-hf | cloud runner
│   ├── build_report_tables.py            # Assemble the per-system reports into the D1 markdown comparison tables
│   ├── infer.py                          # GPU inference helpers (load adapter + generate); used by the observation notebook
│   ├── base_predict.py                   # Zero-shot base helpers: load plan, JSONL I/O, validate_predictions, model_slug
│   ├── predict_base_hf_job.py            # Base-only multi-model inference on HF Jobs (no train/adapter); --submit-hf / --download
│   ├── hebrew_constraint.py              # Decode constraint: build_bad_words_ids() bans non-Hebrew-script tokens
│   ├── prompt_arena.py                   # Prompt-optimization loop, local half: compliance+ROUGE scoring, judge, leaderboard, CLI
│   ├── prompt_rounds.py                  # Prompt-optimization loop: the round registry (every candidate set ever tried + its hypothesis)
│   ├── prompt_sweep_hf_job.py            # Prompt-optimization loop, remote half: sweep K prompts x N examples in ONE HF Job
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
│   ├── test_clean.py                     # normalize_summary / roundup filter / repo + wandb naming helpers
│   ├── test_evaluation.py                # ROUGE-on-Hebrew, judge-reply parsing, failure rates (live test gated)
│   ├── test_stratify_by_topic.py         # join/grouping logic for topic and style stratification
│   ├── test_topic_clustering.py          # BERTopic fit + Gemini naming + plot (live test gated)
│   ├── test_style_labels.py              # rule-based style classification (pipe digest / question / sentence count)
│   └── test_viewer.py                    # predictions-viewer load/keyword-search/discovery logic
├── docs/
│   ├── ANLP Project abstract.md          # Original submitted proposal (historical — Qwen3-2B era)
│   ├── research-proposal.md              # Original proposal prose (historical — Qwen3-2B era)
│   ├── research-proposal-revised.md      # Current plan of record (base model + probe design)
│   └── prompt-arena-notebook.md          # Lab notebook for the prompt loop: guidelines, round log, why each change
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
* `data/prompts.py`: Single hardened `PROMPT_TEMPLATE`, `build_prompt(text)`, `make_variant` (whole|lead|body), plus `format_chat_prompt` / `prepare_tokenizer_for_templated_prompts` — the single source of truth for wrapping instructions in the model's chat template (train + both inference arms). No Qwen-era think-switch injection. Reused by preprocess, train, and evaluation.
* `data/clean.py`: Always-on reference cleaning. `normalize_summary` rewrites HeSum's `"headline | headline"` pipe/bullet digests into natural prose; `is_roundup_digest`/`pipe_segments` flag 3+-segment media roundups for removal. Pure regex, no GPU/API.
* `data/preprocess.py`: Reads `combined.jsonl`, drops roundup digests, normalizes remaining references, builds raw `(prompt, completion)` pairs with the hardened prompt (chat wrap happens at train/infer time so multi-model baselines keep their own templates), applies `--variant whole|lead|body`, truncates each article to `MAX_LENGTH-256` tokens so the summary always survives, splits 80/10/10, saves Arrow splits to `outputs/data/processed/<variant>/`.
* `training/config.py`: Shared constants: `MODEL_ID="dicta-il/dictalm2.0-instruct"`, `MODEL_SLUG="dictalm2-instruct"`, `MAX_LENGTH=4096` (source of truth for train/gen seq budget; preprocess uses `MAX_LENGTH-256=3840` article tokens), `DEFAULT_EPOCHS=1`, `METHOD_PRESETS`, `LoRAConfig`, `TrainingConfig`, `wandb_project`/`wandb_run_name` (date + model + method + variant + epochs), and `dataset_repo`/`model_repo`/`processed_profile_name` Hub-id helpers (adapter repos are `amlk-{MODEL_SLUG}-sft[-variant]`). Self-contained job scripts keep twin fallbacks of `max_length` (must stay in sync).
* `training/train.py`: One trainer for all three regimes (`--method qlora|lora|full`). Trains with `completion_only_loss=True`, 1 epoch by default, logs to a model-specific wandb project with informative run names, saves the adapter; `--push-to-hub` or `--submit-hf` push to the Hub. Serializes resolved `TRAIN_CONFIG`/`LORA_CONFIG` JSON into the remote job env (so METHOD_PRESETS cannot be ignored). Full-run default timeout 8h. Chat-wraps prompts before SFT. Mid-run stability: creates the model repo before the job starts so `hub_strategy=every_save` can commit checkpoints while training. Inference is NOT here.
* `training/train_hf_job.py`: Self-contained PEP 723 UV script submitted inline by `train.py --submit-hf`. Reads METHOD/VARIANT/BASE_MODEL/DATASET_REPO/OUTPUT_REPO/WANDB_*/EPOCHS/`TRAIN_CONFIG`/`LORA_CONFIG` from env, trains on the cloud GPU (1 epoch default), then generates fine-tuned + zero-shot base test predictions. Chat-wraps train/val/infer for both arms; `add_special_tokens=False` on generate (no double-BOS); Hebrew-script `bad_words_ids`. Stability: `/data/output` resume, `hub_strategy=every_save`, immediate prediction uploads. Never run directly.
* `evaluation/predict.py`: Generates the Gemini advanced-baseline summaries via API (no GPU, no model load), same hardened prompt as training. Resumes from a partial file. The fine-tuned and zero-shot predictions come from the cloud training job, not here.
* `evaluation/gemini_client.py`: Shared Gemini API helpers (`GEMINI_MODEL`, `call_with_retry`). Also defines `strip_think()` — the shared tool that drops closed `<think>…</think>` reasoning blocks (emitted by chat-capable Qwen3-family models) so metrics score the summary, not the reasoning (used by evaluate.py and error_analysis.py).
* `evaluation/evaluate.py`: Scores a predictions file with raw + Hebrew-normalized ROUGE-1/2/L (`normalize_hebrew` strips niqqud + folds final-form letters), BERTScore (default `onlplab/alephbert-base`, the HeSum backbone; `--bertscore-model` to override), and the Gemini faithfulness/fluency judge (`--skip-llm` to skip; `--limit N` to cap for a smoke run). Applies `strip_think` before scoring. One JSON report per system.
* `evaluation/error_analysis.py`: Samples ~50 predictions (post `strip_think`) and has Gemini label failure types (hallucination, omission, entity/number error, lead copying, fluency), writing per-type rates.
* `evaluation/eval_hf_job.py`: Runs the whole D1 battery on HuggingFace Jobs so the ~4000 Gemini calls + BERTScore happen on the cloud's fast connection (the user has weak internet). One file, two modes: `--submit-hf` uploads itself to a cheap CPU job; with no args (how HF Jobs invokes it) it fetches the public repo + Hub predictions/dataset and drives the existing `evaluation/` CLIs by subprocess, pushing each report to the model repo under `reports/` as it finishes (timeout-safe).
* `evaluation/build_report_tables.py`: Downloads the pushed `reports/*.json` and assembles the D1 markdown — a quality table (ROUGE/BERTScore/judge), a failure-rate table, and behavioural notes (base `<think>`/language leakage, fine-tuned repetition, judge self-preference caveat).
* `evaluation/infer.py`: GPU inference helpers — `load_finetuned_model` (base + LoRA adapter, **defaults to 4-bit** so 7B fits a Colab T4; `disable_adapter()` gives zero-shot base), `load_base_model` (multi-model zero-shot), and `generate_summaries` (chat-wrap both arms via `format_chat_prompt`, `add_special_tokens=False`, Hebrew-script decode constraint). Twin of `train_hf_job.py` generation. **Remote GPU only — never call locally.**
* `evaluation/base_predict.py`: Pure helpers for multi-model zero-shot baselines (`resolve_load_plan`, `write_predictions_jsonl` / `validate_predictions`, `model_slug` / local paths). Re-exports `format_chat_prompt` / `build_input_text_safe` from `data.prompts`. No GPU.
* `evaluation/predict_base_hf_job.py`: Self-contained UV job for base-only predictions on HF Jobs (no training, no adapter). `--submit-hf --model … --limit 100` or `--all-models`; `--download` pulls `predictions-base.jsonl` into `outputs/<slug>/`. Nemotron uses native `NemotronH` + `PreTrainedTokenizerFast` (Hebrew probe); Gemma-4 uses `AutoModelForMultimodalLM`.
* `evaluation/prompt_arena.py`: Local (CPU/API) half of the **prompt-optimization loop** — a side-loop that runs *before* fine-tuning to find the best zero-shot prompt. Holds a round's `PromptCandidate`s (each template needs one `{text}` placeholder; `validate_candidates` rejects a bad set *before* a GPU job is submitted), scores a swept predictions file, and renders the `leaderboard` the agent reads to write the next round. Ranking is deliberately **not** ROUGE-first: `rank()` sorts by Gemini judge (faithfulness+fluency) → `compliance` → ROUGE-L as a weak tiebreak, because ROUGE rewards drifting toward the references' headline/digest register — the very style the prompts exist to suppress. `compliance` is the fraction of four format rules a summary obeys (6–45 words, 1–2 sentences, ≥80% Hebrew script, no pipes/bullets), i.e. exactly what prompt wording controls. `judge_prompts` runs the judge on a fixed subset of the *finalists only* (judging every candidate on every example is the expensive part of the loop). Rounds persist to `outputs/results/prompt-arena/round-<n>/` as the experiment log.
* `evaluation/prompt_rounds.py`: The prompt loop's **round registry** — `ROUNDS[n]` holds every candidate set ever tried plus the `HYPOTHESES[n]` it tests. Candidates live in version control (not in notebook state) so the diff between rounds *is* the record of what changed. A round is never edited after it runs. `--smoke` uses the 2-candidate `SMOKE` slice.
* `evaluation/prompt_sweep_hf_job.py`: Remote half of the prompt-optimization loop. Self-contained UV job that loads the base model **once** and sweeps every candidate in the round over the **same** N test examples (paired comparison), tagging each row with `prompt_id` so one predictions file holds the whole round. Pushes after each candidate, so a timeout still leaves finished prompts scorable. `truncate_article()` cuts the *article* against each template's own token overhead — truncating the assembled prompt instead would chop its tail, i.e. the trailing "Summary:" instruction, and the model would never see the task. Inlines twins of `format_chat_prompt` + `build_bad_words_ids` (HF Jobs ships one file). `--submit-hf --prompts <round>/prompts.json` / `--download`.
* `evaluation/hebrew_constraint.py`: Decode constraint always used at generation. `build_bad_words_ids(tokenizer)` scans the vocab once and returns the ids of every token whose decoded form contains a Latin/Cyrillic/Greek/Arabic letter. Inlined as a twin in `train_hf_job.py` (that script can't import repo code).
* `evaluation/topic_clustering.py`: Topic-clustering side-analysis (not part of the main pipeline). Embeds truncated article `text` by default (`embed_field='text'`) — summaries alone collapsed ~99% of docs into one media-meta mega-topic — with the Hebrew-native, clustering-tuned `dicta-il/neodictabert-bilingual-embed`, clusters with BERTopic (UMAP + HDBSCAN + Hebrew-only c-TF-IDF vectorizer, `HEBREW_STOPWORDS` + `MEDIA_STOPWORDS` + `BOILERPLATE_STOPWORDS`), names each real cluster with one Gemini call, then optionally `refine_large_clusters()` — a second finer HDBSCAN pass on any cluster holding ≥30% of docs (re-uses embeddings; splits e.g. the politics mega-topic into ביטחון/כלכלה/חברה sub-domains without re-fragmenting sports/legal), then `merge_duplicate_labels()` collapses any clusters Gemini still named identically (on by default via `cluster_dataset(merge_duplicates=True)`) so the report has one row per distinct real-world topic. `fit_topics` tunables: `min_cluster_size`/`min_samples` (default 60/15 — coarser granularity means fewer near-duplicate sub-clusters of the same domain), `outlier_threshold` (only reassign noise above cosine sim — default 0.35; 0 floods the largest cluster), optional `nr_topics` merge (off by default; `auto` over-merged), `language='multilingual'` (required — English mode strips Hebrew). `plot_topic_sizes()` renders a bar chart of `topic_summary()` for the notebook. Output `topics.jsonl` still keyed by `summary` for stratification join. See `notebooks/cluster_topics_databricks.py`.
* `evaluation/style_labels.py`: A second, independent per-summary dimension from topic clustering — not *what topic* an article is about but *what format* its summary takes (`single_sentence` / `multi_sentence` / `pipe_digest` / `question`). Pure regex (`classify_style`), no embeddings/GPU/API, so unlike topic clustering it never needs Databricks and has no `datasets` import (works even if that import is broken locally, see the lzma note below). Motivated by a real corpus pattern: ~26% of HeSum summaries are `"headline | headline | headline"` pipe-separated digests — a format quirk worth tracking once a model is trained on this data. Produces the same `{summary: label}` artifact shape as `topic_clustering.py` so it plugs into the same stratification tool; `plot_style_distribution()` renders a bar chart of `style_summary()` for the notebook.
* `evaluation/stratify_by_topic.py`: Joins a predictions file to a label artifact (`topics.jsonl`'s `topic_label` from `topic_clustering.py`, or `style_labels.jsonl`'s `style_label` from `style_labels.py` — same shape, selected via `--label-field`) on exact `reference`==`summary` text match, and reuses `evaluate.py`'s `compute_rouge`/`compute_bertscore` per group, folding in per-group failure rates if a matching `*.errors.json` exists. Local, CPU-only — no GPU/Databricks needed for this step.
* `evaluation/viewer/`: A local, read-only UI for browsing `predictions.jsonl` files (article/prediction/reference), filling the gap between raw jsonl and the live Colab notebook. `data.py` has the Streamlit-free data logic (`discover_predictions_files`, `load_predictions` — applies `strip_think`, `filter_by_keyword`, `common_length`), importable from a notebook/REPL; `__init__.py` re-exports it; `app.py` is the thin Streamlit script (`streamlit run evaluation/viewer/app.py`) that renders Hebrew right-to-left, supports keyword search, and compares 2+ systems side-by-side for the same article. Local, CPU-only, no GPU/API.
* `notebooks/evaluation_observation.ipynb`: The **evaluation-observation** stage. A self-bootstrapping Colab notebook that runs the *real* evaluation functions live and **displays** the per-example process (article → model summary → reference → judge faithfulness/fluency → error-analysis failure labels) for finetuned/base/gemini. Loads existing Hub predictions (finetuned/base at repo root, gemini under `reports/`) and generates fresh summaries on a T4. Judge/error/browse cells are API+CPU; only the generation cell needs a GPU.
* `notebooks/cluster_topics_databricks.py`: Databricks source-format notebook (`# Databricks notebook source` / `# COMMAND ----------` cell markers) driving `evaluation/topic_clustering.py` and `evaluation/style_labels.py`. Manual, occasional run on a Databricks GPU cluster — the GPU is for speed, not required (the embedding model is 0.4B params, encoder-only, the same class of job as the local-CPU AlephBERT BERTScore step). Clones the repo (or reuses an uploaded Workspace copy) so it calls the same tested functions rather than duplicating logic; computes both `topic_label` (BERTopic) and `style_label` (regex) over the same records. Plots (all inline via `displayHTML`, small enough to skip the DBFS round-trip): a `plot_topic_sizes` cluster-size bar chart, an interactive 2D/3D cluster scatter (`plot_clusters(dimensions=2|3)`, `plot_dimensions` widget; written to DBFS + iframe-embedded since 10k-doc hovers exceed the ~20 MB cell-output cap), a `plot_style_distribution` bar chart, and a topic×style stacked bar chart alongside the crosstab table. Writes one `topics.jsonl`/`topics-summary.json` (carrying both label fields) to DBFS for manual download into `outputs/data/raw/` and `outputs/results/`. Widgets expose `min_cluster_size`/`min_samples`/`reduce_outliers`/`nr_topics`/`merge_duplicate_labels`/`topic_size_plot_top_n`/`plot_dimensions` so noise/near-duplicate-topic tuning (see `topic_clustering.py`) doesn't require editing the notebook. A scoped, one-off departure from AMLK's default local/HF-Jobs/Colab stack — no agent-driven Databricks deployment (no MCP connection today), the notebook is handed off for manual import/run.
* `scripts/run_nb_cell.py`: Agent cell-runner — reads the notebook with `nbformat` and execs a chosen code cell / range against a persistent Colab session via `colab exec` (the Colab CLI has no native `.ipynb` runner). `--list` shows cell indices; the caller owns `colab new`/`stop`. This is how an agent observes the eval cell-by-cell.
* `docs/prompt-arena-notebook.md`: The **lab notebook** for the prompt-optimization loop (a written research log, not a Jupyter notebook — the loop is a sequence of long remote jobs, and its record has to survive in git and feed the paper). Holds the guidelines (the 100-example paired-comparison contract, why ranking is judge → compliance → ROUGE-L and never ROUGE-first), the design decisions and their reasons, the per-round log of what was tried and what happened, and the code change log. Append one entry per round; never rewrite a past one.
* `tests/`: ~67 fast behavioral tests + gated live tests (Gemini judge; BERTopic fit + Gemini topic naming + plot). Local `plotly` optional for 3 plot tests.

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

# 2. Preprocess: clean refs + hardened prompt + 80/10/10 split. --variant selects the probe input.  [local, CPU]
python -m data.preprocess --variant whole        # also: --variant lead | body

# 3. Train on HF Jobs (cloud GPU, 1 epoch). The job also generates fine-tuned + zero-shot base test
#    predictions and pushes predictions-finetuned.jsonl / predictions-base.jsonl to the model repo.
#    Mid-run: hub_strategy=every_save commits checkpoints; /data/output survives job restarts.
python -m training.train --submit-hf --hf-user avreymi --smoke-test   # verify first (~$0.05)
python -m training.train --submit-hf --hf-user avreymi                # full 1-epoch run

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
# then submits train_hf_job.py inline (a10g-small, 8h, 1-epoch training by default). It prints a Job ID.
python -m training.train --submit-hf --hf-user avreymi               # full 1-epoch run
python -m training.train --submit-hf --hf-user avreymi --smoke-test  # 10 steps, a10g-small, ~$0.05 — verify first
python -m training.train --submit-hf --hf-user avreymi --inference-only  # regen predictions from pushed adapter (a10g-small, 2h)
# Cost: a10g-small has the SAME 24 GB A10G GPU as a10g-large at $1.00/h vs $1.50/h.
# dictalm2.0-instruct is Mistral-7B → default method is qlora.

hf jobs ps                    # list recent jobs
hf jobs logs <job-id>         # snapshot; add -f to stream
hf jobs inspect <job-id>

# Trained adapter (LoRA only, not merged) pushes to: https://huggingface.co/avreymi/amlk-dictalm2-instruct-sft  (private)
# Evaluation loads it via: predict.py --model finetuned --adapter avreymi/amlk-dictalm2-instruct-sft
# Training metrics: wandb project "amlk-dictalm2-instruct"; run names include date/method/variant/epochs.
```

**Reading model outputs (predictions viewer):**
```bash
source .venv/bin/activate && streamlit run evaluation/viewer/app.py
# Opens a local browser UI over outputs/results/*.jsonl: RTL Hebrew, keyword search,
# side-by-side comparison across systems (finetuned/base/gemini). Local, CPU-only, read-only.
```

**Prompt-optimization loop (find the best zero-shot prompt before fine-tuning):**
```bash
# Full guidelines + the round log live in docs/prompt-arena-notebook.md (read it first).
set -a && source .env && set +a && source .venv/bin/activate    # .env has no `export`

# 1. WRITE round N's candidates (they live in evaluation/prompt_rounds.py):
python -m evaluation.prompt_arena --round 1 --write            # add --smoke for the 2-prompt slice
# 2. RUN — one HF Job: model loads once, every prompt sees the same examples (paired comparison):
python -m evaluation.prompt_sweep_hf_job --submit-hf --round 1 --limit 100 \
    --prompts outputs/results/prompt-arena/round-1/prompts.json
# smoke: --limit 4 --batch-size 2 --timeout 30m   (2 prompts x 4 examples, ~$0.05)
# 3. COMPARE — download, score, rank. --judge adds Gemini on the finalists only;
#    --show N prints every prompt's summary of example N side by side.
python -m evaluation.prompt_sweep_hf_job --download --round 1
python -m evaluation.prompt_arena --round 1 --score --judge --show 0
# 4. IMPROVE — add ROUNDS[2] + its hypothesis to prompt_rounds.py, append a notebook entry, repeat.
# Winner gets promoted into data/prompts.py::PROMPT_TEMPLATE (the prompt fine-tuning trains on).
```

**Running tests:**
```bash
source .venv/bin/activate && python -m pytest tests/ -v
```

---

## Status - remember to update it

**2026-07-11 — Prompt-optimization loop: all 3 rounds complete, guideline recorded, winner
promoted.** Full loop details/tables in `docs/prompt-arena-notebook.md`. Round 1
(job `6a528b54effc02a91cbd9ba1`, 5 prompts × 100 ex.): sentence-count phrasing ("one or two
sentences") does not bind length in any wording/language — all 5 prompts landed at 52-65 words /
3.4-4.2 sentences. Also found the long English "Rules:" prompt provoked a garbled Hangul token
(`[/인스트]`, an apparent hallucinated echo of Mistral's `[/INST]` tag) in 38% of outputs vs 1-2%
for Hebrew prompts — root cause: `evaluation/hebrew_constraint.py`'s decode constraint never
banned CJK/Hangul, only Latin/Cyrillic/Greek/Arabic. **Fixed** in the one source file and its two
inlined twins (`prompt_sweep_hf_job.py`, `train_hf_job.py`) — real fix, not scoped to the loop.
Round 2 (job `6a529bcfe4a4e82c0b58e127`, 5 prompts × 100 ex.): numeric word budgets (15/25/30)
bind only weakly — every capped prompt overshot its own number 2-3x and the three caps barely
separated from each other (47-54 words). Best result: `p6_he_wordcap15`, compliance 0.69 /
judge faithfulness 3.13. Round 3 (job `6a52aa83e4a4e82c0b58e388` submitted then canceled ~4 min in
per user cost instruction; resubmitted smaller as job `6a52b8beeffc02a91cbda075`, 5 prompts × 20
ex.): tested one-shot/two-shot worked examples vs. an explicit stop cue. Worked examples
**underperformed** — both hallucinated an unrelated entity ("ועדת ביטון"), judge faithfulness
dropped to 2.2-2.5. The winner was `p11_he_stopcue` (numeric cap + "write one sentence only and
stop right after it", no exemplar): compliance **0.82**, faithfulness **3.40**, fluency **4.27** —
best of all 15 candidates tested, though only checked at n=20. **Guideline:** for
dictalm2.0-instruct zero-shot Hebrew summarization, abstract length instructions (sentence- or
word-count) barely bind, but a concrete imperative stop cue does much better — and worked examples
are a trap here, risking content contamination rather than constraining format. `p11_he_stopcue`
is promoted into `data/prompts.py::PROMPT_TEMPLATE`. Still short of the loop's 0.9/4.0 target;
fine-tuning (not further prompt engineering) is expected to close the remaining gap.

**2026-07-11 — Cost lever: default `max_new_tokens` 256 → 128 (train+infer path).** C5 showed
dual-arm generation ~1.6 h of a ~5.8 h full a10g-small run because every pred hit the 256-token
cap without EOS. `training/config.py::DEFAULT_MAX_NEW_TOKENS=128` is now the decode budget;
`train.py --submit-hf` ships `MAX_NEW_TOKENS` (CLI `--max-new-tokens`); `train_hf_job.py` +
`infer.py` consume it. Full-run proxy: gen ~1.6 h → ~0.8 h (~$0.80 / ~14% per train+infer job).
Verified real run: inference-only smoke job `6a52aeddeffc02a91cbd9e12` COMPLETED (~5 min,
a10g-small, max_new_tokens=128 logged) → `predictions-{finetuned,base}-cost128.jsonl` (5+5) on
`avreymi/amlk-dictalm2-instruct-smoke`. Char-length proxy vs prior 256-cap smoke preds: mean
474 → 273 chars (ratio 0.575). A train smoke with the same lever (`6a52a461e4a4e82c0b58e27c`)
finished 10/10 steps then hung on post-train hub push and was canceled — decode path re-verified
via inference-only. Override with `--max-new-tokens 256` if a long-form probe needs more.

**2026-07-11 — Prompt-optimization loop built + smoke PASSED.** New side-loop that tunes the
zero-shot prompt *before* fine-tuning: `evaluation/prompt_arena.py` (scoring/ranking + CLI),
`evaluation/prompt_rounds.py` (round registry), `evaluation/prompt_sweep_hf_job.py` (one HF Job,
model loaded once, K prompts × N examples), `docs/prompt-arena-notebook.md` (the lab notebook —
guidelines, round log, reasons), `tests/test_prompt_arena.py` (12 tests, pass).
Contract: 100 examples per prompt; candidates must be short. Ranked by judge → compliance →
ROUGE-L (ROUGE last on purpose — it rewards drifting toward the references' digest register).
Smoke: job `6a5287c2effc02a91cbd9b8c` (a10g-small, 2 prompts × 4 examples, ~6 min) → 8 rows at
`avreymi/amlk-prompt-arena/sweeps/round-1`; download → score → leaderboard verified end-to-end.
**Finding to act on in round 1 proper:** both the current hardened prompt and a minimal Hebrew one
badly overshoot the length target (62w/3.75 sents and 56w/4.0 sents vs. the 1–2 sentence target),
hitting the 160-token cap and truncating mid-sentence; compliance 0.50 / 0.44, ROUGE-L ~0.05.
Neither prompt's length instruction is binding — that is the hypothesis round 2 should attack.

**2026-07-11 — MAX_LENGTH 2048 → 4096 (all sites + re-preprocess + smoke submitted).** Seq budget
is 4096 everywhere that matters: `training/config.py` (`MAX_LENGTH`), `train_hf_job.py` default
`TRAIN_CFG["max_length"]` (+ gen uses `TRAIN_CFG["max_length"]-128`), `evaluation/infer.py`
(`MAX_LENGTH-128`), `evaluation/predict_base_hf_job.py` (`4096-128`). Preprocess
`ARTICLE_TOKEN_BUDGET = MAX_LENGTH-256 = 3840`. Local `outputs/data/processed/whole/` rebuilt
(6073/759/760) and re-uploaded to `avreymi/amlk-training-data`. 4096 smoke submitted:
job `6a52848eeffc02a91cbd9b71` (a10g-small, qlora, 10 steps) →
`avreymi/amlk-dictalm2-instruct-smoke`. Gate before full 1-epoch.

**2026-07-11 — Smoke-path cleanup (dictalm2.0-instruct ready).** After C0–C5, removed dead
branches: no `label`-branched prompt format, no `DATA_PROFILE` dual-path leftover in
`eval_hf_job`, train/infer both call `format_chat_prompt` only. Fixed `MODEL_SLUG` drift
(`dictalm2-0-instruct` → `dictalm2-instruct` via env from `train.py`). Defaults remain
`MODEL_ID=dicta-il/dictalm2.0-instruct`, method `qlora`, 1 epoch, 8h full timeout, 4-bit load.

**2026-07-11 — CODE AUDIT C0–C5 fixed (dictalm2 instruct format + wiring).**
- **C0:** Train + finetuned/base inference both apply the model chat template via
  `data.prompts.format_chat_prompt` (raw `prompt` column still stored for multi-model baselines).
- **C1:** Double-BOS fixed — `add_special_tokens=False` on generate + `add_bos_token=False` when a
  chat template is present.
- **C2:** Removed Qwen-era `/no_think` injection; one shared formatter (+ inlined twins in the two
  self-contained HF job scripts).
- **C3:** `load_finetuned_model(..., quantize=True)` default so 7B fits a Colab T4.
- **C4:** `train.py` serializes `TRAIN_CONFIG`/`LORA_CONFIG` from `METHOD_PRESETS`; `train_hf_job.py`
  consumes them (no more hardcoded batch=2/lr=2e-4 that ignored `--method full`).
- **C5:** Full-run default job timeout 6h → 8h (smoke step-time projected ~5.8h worst case).
  Dataset/P0 grounding and decode-config (P2) left as-is per audit scope.

**2026-07-11 — Clean-only pipeline + 1-epoch runs + mid-run Hub stability (dictalm2 branch).**
Training is clean-only with no dual raw/clean profile: preprocess always drops roundup digests,
normalizes pipe/bullet references, and uses the hardened prompt; generation always applies
Hebrew-script `bad_words_ids`. Removed `--clean` / `--drop-roundups` flags across train/eval/predict.
Default epochs = 1. wandb project is `amlk-{MODEL_SLUG}` (e.g. `amlk-dictalm2-instruct`); run names
are `{date}_{slug}_{method}_{variant}_{N}ep[_tag]`. Stability: `/data/output` resume,
`hub_strategy=every_save`, immediate prediction uploads. Base: `dicta-il/dictalm2.0-instruct`
(Mistral-7B, QLoRA default).

**Post clean-only smoke COMPLETED 2026-07-11** — job `6a524384effc02a91cbd98c6` (~11 min,
a10g-small, qlora, 10 steps). Clean data re-preprocessed (dropped 2408 roundups → 7592) and
re-uploaded to `avreymi/amlk-training-data`. wandb project `amlk-dictalm2-instruct`, run
`2026-07-11_dictalm2-instruct_qlora_whole_1ep_smoke`. LoRA 83.9M / 1.14%, loss 1.04→0.52
(avg 0.779), eval ~1.18–1.30 finite, Hebrew constraint on, adapter + preds at
`avreymi/amlk-dictalm2-instruct-smoke`. That smoke predates the C0–C5 chat-template fix — re-smoke
before a full 1-epoch run.

**Pre-training stage as of 2026-07-11.** Pipeline mechanics validated (clean path); full 1-epoch
run pending. Stack: trl 1.6.0, transformers 5.11, peft 0.19, wandb 0.27–0.28.
- Hub: dataset `avreymi/amlk-training-data` (clean splits as of 2026-07-11). Smoke model:
  `avreymi/amlk-dictalm2-instruct-smoke`. Real adapter repo: `avreymi/amlk-dictalm2-instruct-sft`.
  wandb: `amlk-dictalm2-instruct`. Judge/baseline: Gemini `gemini-2.5-flash-lite`.
- Note: QLoRA `push_to_hub` saves the LoRA adapter only (not merged).
- Note: the Gemini LLM-judge and the Gemini advanced baseline are the same model family — flag
  self-preference bias in the paper.
- Decode config: `max_new_tokens=128` (`DEFAULT_MAX_NEW_TOKENS`; was 256 — gen cost lever),
  `min_new_tokens=min(16, max_new_tokens)`, `no_repeat_ngram_size=3`,
  `repetition_penalty=1.2`, greedy + Hebrew-script constraint.

**2026-07-09 — Diagnosed and fixed a training-checkpoint-loss bug** (job `6a4f55731fba25b8ea3b310b`,
1 epoch; historical dual-profile era, lesson still applies). The job's underlying container
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

**2026-07-09 (historical) — Opt-in clean profile + DictaLM-3.0-1.7B experiments.** An earlier
branch state had dual raw/clean profiles (`--clean`/`--drop-roundups`) and briefly used
`dicta-il/DictaLM-3.0-1.7B-Base` (Qwen3 1.7B). That dual-profile design is **superseded** by the
2026-07-11 clean-only simplification; this branch's base model is **`dicta-il/dictalm2.0-instruct`**
(Mistral-7B instruct, QLoRA default). Keep the checkpoint-resume lesson (`/data/output`) from the
same day's infra-restart diagnosis above.

**2026-07-04 — More distinct cluster plot + tighter clustering defaults.** Plot: golden-angle color palette, UMAP `min_dist=0.35`/`spread=1.25`, optional centroid repulsion (`plot_display_spread` widget). Clustering: `min_samples` 15→20, `umap_n_neighbors` 10→15, `outlier_threshold` 0.35→0.40; `umap_n_neighbors` widget on Databricks.

**2026-07-04 — Optional 3D cluster plot.** `plot_clusters(..., dimensions=2|3)` adds a rotatable 3D UMAP view (convex-hull mesh clouds + centroid text labels); Databricks widget `plot_dimensions` defaults to `2`. 2D remains the default for the iframe embed.

**2026-07-04 — Refinement coarsened after 60+ cluster explosion.** The first refinement pass (25/8, no topic cap) split the politics mega-cluster into 60+ near-duplicate "תקשורת ו…" Gemini labels. Defaults now: `refine_min_cluster_size=100`, `refine_min_samples=20`, `refine_nr_topics=12` (BERTopic merge cap on the refinement pass only), stricter refinement naming prompt forbidding "תקשורת/עיתונות/…" meta-labels. Expect ~15–20 topics total (5 pass-1 + ~12 politics sub-domains). Set `refine_oversized=False` to keep ~6 pass-1 topics only.

**2026-07-04 — Two-stage mega-cluster refinement.** Pass 1 still uses coarse HDBSCAN (60/15) for stable top-level domains; `refine_large_clusters()` (on by default, `refine_oversized=True`) re-clusters any topic holding ≥30% of docs with finer settings on the *same* embeddings and a sub-domain Gemini naming prompt — splits the ~7.6k politics blob without re-embedding or re-fragmenting sports/legal. Databricks widgets: `refine_oversized`, `refine_size_fraction`, `refine_min_cluster_size`, `refine_min_samples`, `refine_nr_topics`.

**2026-07-04 — Topic-clustering "fewer, more distinct topics" fix (v3) + notebook plots.** The v2 fix (embed on `text`) still surfaced too many near-duplicate topics (e.g. "תקשורת ומדיה"/"תקשורת וטלוויזיה") once mega-topic collapse was fixed, driven by (a) layout/journalism-meta keywords ("כותרת", "הבוקר", "העיתון"...) dominating c-TF-IDF instead of real subject words, and (b) fine HDBSCAN granularity producing several sub-clusters of the same domain that Gemini then named identically. Fixed in `evaluation/topic_clustering.py`: a `BOILERPLATE_STOPWORDS` set added to the vectorizer; `min_cluster_size`/`min_samples` raised 25/5 → 60/15 (coarser HDBSCAN); a new `merge_duplicate_labels()` post-processing step (on by default, `cluster_dataset(merge_duplicates=True)`) that collapses any clusters Gemini still named identically into one reported topic, keeping the smallest `cluster_id` and the union of keywords — no extra Gemini calls. Databricks widgets: `min_cluster_size`/`min_samples` defaults updated, new `merge_duplicate_labels` toggle. Also added inline Plotly charts to the notebook pipeline (`plot_topic_sizes` in `topic_clustering.py`, `plot_style_distribution` in `style_labels.py`, plus a topic×style stacked bar) alongside the existing big document scatter — small aggregate charts shown directly with `displayHTML(fig.to_html(...))`, no DBFS round-trip needed since they're nowhere near the ~20 MB cell-output cap.

**2026-07-04 — Topic-clustering granularity fix (v2).** After the noise/vectorizer fixes, a second full run collapsed ~99% of docs into one "חדשות ותקשורת" mega-topic — caused by clustering on summaries (outlet-name headlines), `outlier_threshold=0` (force-assign all noise to the largest cluster), and `nr_topics='auto'` over-merging. Defaults now: `embed_field='text'` (first 4k chars of article body), `outlier_threshold=0.35`, `nr_topics=None`, `min_cluster_size=25`/`min_samples=5`, media-outlet stopwords + domain-focused Gemini naming prompt. Databricks widgets updated (`embed_field`, `outlier_threshold`, `max_embed_chars`; `nr_topics` blank by default).

**2026-07-04 — Topic-clustering quality fix (v1).** The first full 10k-doc Databricks run put 51% of docs in the noise cluster (-1) and produced near-duplicate topic names (e.g. "תקשורת ומדיה" / "תקשורת וטלוויזיה") whose c-TF-IDF keywords were mostly years/IDs/Latin site names (`ynet`, `nrg`, `bbc`) — BERTopic's default English-tuned vectorizer let non-Hebrew tokens dominate. Fixed in `evaluation/topic_clustering.py`: a Hebrew-only `CountVectorizer` (`_build_vectorizer`/`HEBREW_TOKEN_PATTERN`/`HEBREW_STOPWORDS`), `min_samples` decoupled from `min_cluster_size` (per BERTopic's FAQ, reduces raw noise), and two opt-out BERTopic post-processing passes — `reduce_outliers` (embedding-similarity reassignment of noise docs) and `nr_topics="auto"` (HDBSCAN-over-topic-vectors merging of only genuinely similar topics). All exposed as Databricks widgets (`min_cluster_size` default lowered 100→40, `min_samples`, `reduce_outliers`, `nr_topics`) so re-tuning doesn't require editing the notebook. 3 new fast unit tests cover the Hebrew token pattern/stopwords.

**2026-07-04 — Predictions viewer added.** `evaluation/viewer/` (`data.py` + `app.py`, its own subfolder): a local Streamlit app (`streamlit run evaluation/viewer/app.py`) for browsing `outputs/results/*.jsonl` — RTL Hebrew rendering, keyword search, side-by-side comparison across systems. Read-only, CPU-only, no GPU/API. Verified end-to-end against the real `predictions-finetuned.jsonl`/`predictions-base.jsonl` files with `streamlit.testing.v1.AppTest` (file discovery, multi-file compare, keyword filtering, navigation — no exceptions).

**Next steps:**
1. **4096 smoke then full 1-epoch QLoRA** on `dicta-il/dictalm2.0-instruct` (local data already
   re-preprocessed at `MAX_LENGTH=4096`). Smoke first with Hub re-upload:
   `python -m training.train --submit-hf --hf-user avreymi --method qlora --smoke-test --output-repo avreymi/amlk-dictalm2-instruct-smoke`
   then full: `python -m training.train --submit-hf --hf-user avreymi --method qlora`.
2. **D.1 — full eval battery** on the trained adapter (`evaluation.eval_hf_job --submit-hf`),
   scoring finetuned / zero-shot base / Gemini advanced baseline with ROUGE + BERTScore + judge +
   error analysis, assembled via `evaluation.build_report_tables`.
3. **Positional-shortcut probe** — train one whole-article model, then ablate Whole / Lead / Body
   at inference (see `docs/research-proposal-revised.md` and `TODO.md` F).
4. **Literature (English summarization)** — document lessons from English news summarization in the paper (lead bias, ROUGE limits, baseline practices).
5. **Journalism / headline control (optional)** — alternate instruction templates for headline-length vs longer summaries; see `TODO.md` G.

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
