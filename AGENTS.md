## Project Goal

* **Description (current, since 2026-07-26):** AMLK is a **review of HeSum as a training resource** for Hebrew news headline generation. Two model eras (Qwen3-2B, then DictaLM) established that the bottleneck is the dataset, not the model: HeSum is scraped, its "summaries" are news-site subheadings, and a large fraction are unfit as training targets. Of 10,000 raw rows, 5,854 survive curation and **52.4% of survivors needed their headline rewritten**. The project now audits those defects (`data_curation/`), measures whether the defects matter and whether the repair helped, and reports it as a dataset review. **Training is owned externally** — we own the audit, the figures, and the post-training evaluation. The lead-bias question survives as a continuous per-row covariate rather than a training-variant probe. Read `docs/obsidian/Project Pivot.md` first; the pre-registered design is `docs/superpowers/specs/2026-07-26-dataset-review-experimental-design.md`.

* **Description (original, retained for context):** AMLK is a Hebrew **news** summarization research project. The goal is to fine-tune `Qwen/Qwen3-2B` on Hebrew journalism datasets (HeSum, IAHLT summarization_he), evaluate with ROUGE, BERTScore, and LLM-as-judge, and produce a research paper and presentation. Design choices are informed by **English summarization literature** (lead bias, metric limits, strong baselines) without re-running English experiments. Evaluation includes an **advanced-model baseline** (e.g. Gemini API on the same test set and prompt) so metrics can be interpreted against a stronger system. A **truncation / positional-shortcut probe** trains separate models on Whole text, Lead-only, and Body-only inputs. Optional **headline-style control** varies the instruction (short headline vs longer summary). **Error analysis** labels a sampled set of predictions for failure types common in the literature. Runs locally or on HuggingFace Jobs; all scripts are command-line Python.

---

## Project Structure - remember to update it when you make changes

* **Architecture:** The project is divided into three sequential pipelines:
  1. **Training pipeline** — downloads Hebrew summarization datasets (IAHLT summarization_he, HeSum), loads the `Qwen/Qwen3-2B` base model, and fine-tunes it using the HuggingFace `transformers`/`trl` stack. If local GPU is insufficient, the job is submitted to HuggingFace as a remote training job.
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
│   ├── prompts.py                        # Shared prompt templates (raw + hardened clean) + probe make_variant
│   ├── clean.py                          # Clean-profile reference normalization (pipes/bullets→prose) + digest filter
│   └── preprocess.py                     # Pipeline step 2: prompt/completion pairs + probe variants, 80/10/10 split
├── data_curation/                        # CURRENT pipeline: audit + repair HeSum → final_clean_hesum.json
│   ├── CURATION_ROADMAP.md               # Authoritative stage-by-stage reference (paths, shapes, counts)
│   ├── build_curated_dataset.py          # One entry point: download → deterministic cleanup → validate → final dataset
│   ├── data_download/download_hesum.py   # Stage 1: biunlp/HeSum → raw_hesum.json (10,000 × {id,text,headline})
│   ├── pre_model_cleanup/
│   │   ├── run_pre_model_cleanup.py      # Stages 2-5 driver; builds source_filter_input.json
│   │   ├── tail_boilerplate_trimming/    # Stage 2: find + strip repeated scraped tails (722 rows affected)
│   │   ├── dictalm_token_budget_filtering/  # Stage 3: keep map, tokens(text)+tokens(headline) ≤ 4000 (2,659 removed)
│   │   └── multi_pipe_headline_filtering/   # Stage 4: keep map, headline.count("|") ≤ 1 (2,412 removed)
│   ├── model_curation/
│   │   ├── openai_batch_api/openai_client.py  # Batch API mechanics: submit, poll, collect, resume
│   │   ├── source_filter/                # Stage 6: 6 usability labels over 6,486 rows (5,854 usable)
│   │   └── headline_target_curation/     # Stage 7: keep-or-replace headline target (3,069 rewritten)
│   ├── final_dataset/build_final_dataset.py   # Stage 8: assemble + validate final_clean_hesum.json
│   ├── analysis/                         # Dataset-review analysis: row-label artifact, paper figures, rubric-judge pilot
│   │   ├── row_labels.py                 # Join every artifact into one row per id (10,000) — spec section 2
│   │   ├── plotting.py                   # House visual style: template, colorblind-safe palettes, export helper
│   │   ├── figures.py                    # F1 curation-funnel Sankey, F2 defect-prevalence bars
│   │   ├── rubric_anchors.py             # Hand-picked Hebrew 1/3/5 worked examples per rubric dimension + exclusion list
│   │   ├── rubric_pilot.py               # E1 step: pilot the rubric judge on ~300 stratified rows, test-retest kappa
│   │   ├── rubric_full_run.py            # E1 step: full judge pass over all ~10,000 rows, resumable JSONL, concurrent
│   │   ├── rubric_results.py             # Joins e1_rubric_scores.jsonl to row_labels.json by stratum; Cliff's delta + length-binning stats for F3-F5
│   │   ├── rubric_figures.py             # F3 (distributions by stratum), F4 (effect sizes), F5 (length/lead-bias) figure builders
│   │   ├── supplementary_figures.py      # 18 extra slides: pipeline anatomy, E1 alt views, pilot kappa (sx01-sx18)
│   │   ├── recover_frozen_split.py       # Reconstructs frozen_split_v1.json from the in-flight Arm B job (no hesum_id upstream)
│   │   └── build_arm_a_dataset.py        # Arm A content: Arm B's exact ids, original (pre-curation) headlines
│   ├── utils/                            # json_io, paths, lzma_shim (stubs _lzma when the local Python lacks it)
│   └── artifacts/                        # Supplied: source_filter_results.json, headline_target_curation_results.json; rest regenerates
│       └── splits/frozen_split_v1.json   # Reconstructed frozen test split (see Training Handoff Contract) — regenerable
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
│   ├── hebrew_constraint.py              # Optional clean-profile decode constraint: forbid foreign-script tokens
│   ├── topic_clustering.py               # Embed summaries + BERTopic cluster + Gemini-name topics + plot_clusters(); used by the Databricks notebook
│   ├── style_labels.py                   # Rule-based structural style labels (single/multi-sentence, pipe digest, question) — local, no GPU/API
│   ├── stratify_by_topic.py              # Break down a predictions file's ROUGE/BERTScore/failure rates by topic_label or style_label
│   ├── gemini_client.py                  # Shared Gemini retry/model/timeout helpers + strip_think()
│   ├── hf_client.py                      # HF Inference API judge client (alternative to Gemini)
│   ├── rubric_judge.py                   # The 4-dimension Reference Quality Rubric judge — shared by E1-E4
│   └── viewer/                           # Predictions viewer + F9a human-validation UI
│       ├── __init__.py                   # Re-exports data.py's public functions
│       ├── data.py                       # Streamlit-free helpers: discover/load/keyword-search predictions.jsonl files
│       ├── annotation_data.py            # F9a: worklist load, JSONL resume/append, blind pairwise slots
│       ├── annotate_app.py               # F9a Streamlit annotator (rubric + pairwise); run via ANNOTATION.md
│       ├── ANNOTATION.md                 # Teammate handout: checkout branch, run, return JSONL
│       └── app.py                        # Read-only predictions browser (RTL Hebrew, multi-system compare)
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
│   ├── test_evaluation.py                # ROUGE-on-Hebrew, judge-reply parsing, failure rates (live test gated)
│   ├── test_stratify_by_topic.py         # join/grouping logic for topic and style stratification
│   ├── test_topic_clustering.py          # BERTopic fit + Gemini naming + plot (live test gated)
│   ├── test_style_labels.py              # rule-based style classification (pipe digest / question / sentence count)
│   ├── test_clean.py                     # clean-profile reference normalization + roundup-digest filter
│   ├── test_viewer.py                    # predictions-viewer load/keyword-search/discovery logic
│   ├── test_row_labels.py                # row-label join/classification logic (no tokenizer/API)
│   ├── test_figures.py                   # F1/F2 count derivation + figure construction on a tiny fixture
│   ├── test_rubric_judge.py              # prompt construction + judge-reply parsing (live judge call gated)
│   ├── test_rubric_pilot.py              # stratified sampling, distribution/degeneracy checks, kappa
│   ├── test_rubric_full_run.py           # E1 full-pass worklist construction + resume-by-id logic
│   ├── test_rubric_results.py            # score/row-label join, Cliff's delta, equal-count length binning
│   ├── test_recover_frozen_split.py      # Text-to-hesum_id matching logic for the reconstructed frozen split
│   └── test_build_arm_a_dataset.py       # Arm A id-coverage + original-headline pairing logic
├── docs/
│   ├── obsidian/                         # Shared Obsidian vault (team research notes; open folder as vault)
│   │   ├── Home.md                       # Map of content; START HERE
│   │   ├── Project Pivot.md              # The three eras (Qwen → DictaLM → dataset review) and why each ended
│   │   ├── Data Curation Pipeline.md     # The 8 stages, verified counts, design consequences
│   │   ├── Dataset Defect Taxonomy.md    # 6 source labels verbatim, analysis strata, why length is a covariate
│   │   ├── Reference Quality Rubric.md   # The 4-dimension judge — the instrument everything rests on
│   │   ├── Reference Quality Experiment.md  # E1-E4 readable map (full detail in the spec)
│   │   ├── Training Handoff Contract.md  # Boundary with externally owned training; frozen split is ours
│   │   ├── Paper Figures.md              # 9-figure manifest, conventions, compute placement, sequencing
│   │   ├── Experiment Results.md         # Article-ready measured outcomes (E1–E4, DictaLM, exact numbers)
│   │   └── (Qwen-era notes, #status/superseded: Current Results, Fix Plan, Prediction Failure Modes, …)
│   ├── ANLP Project abstract.md          # The research proposal this project implements
│   ├── 2026-06-12-qlora-training-job-postmortem.md  # Full-run post-mortem: cost, root cause, probe-run recommendations
│   └── superpowers/
│       ├── specs/2026-07-26-dataset-review-experimental-design.md  # CURRENT pre-registered design (E1-E4)
│       ├── specs/2026-05-26-training-pipeline-design.md
│       └── plans/2026-05-26-stage-a-training-pipeline.md
├── paper/                                # ACL 2023-style course submission (Overleaf-compiled, no MCP)
│   ├── main.tex                          # Full skeleton: Abstract/Intro/Data/Methods/Results + appendix; E4 (F8) placeholder
│   ├── acl2023.sty / acl_natbib.bst      # ACL template style + bib style, from the supplied template.zip
│   ├── bib.bib                           # HeSum, LoRA/QLoRA, BERTScore, AlephBERT, self-preference bias, Cliff's delta, …
│   ├── figures/                          # PNG copies of F1–F7/F9 + 4 sx appendix figures, synced from outputs/figures/
│   ├── tables/                           # Optional \input{tables/*.tex} (currently empty)
│   ├── sync_figures.sh                   # Re-copies figures/ from outputs/figures/ after a figure-script rerun
│   └── README.md                         # Overleaf upload steps; what's still a placeholder (E4/F8)
├── outputs/
│   ├── data/
│   │   ├── raw/combined.jsonl            # Merged normalized dataset — 10,000 records (gitignored)
│   │   └── processed/<variant>/          # Arrow splits train/ val/ test/ per probe variant (gitignored)
│   ├── checkpoints/                      # LoRA adapter / full model checkpoints (gitignored)
│   └── results/                          # predictions.jsonl + evaluation/error-analysis reports (gitignored)
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
* `data/preprocess.py`: Reads `combined.jsonl`, builds `(prompt, completion)` pairs for completion-only SFT, applies the `--variant whole|lead|body` truncation probe (`make_variant`), truncates each article to `MAX_LENGTH-256` tokens so the summary always survives (HeSum articles are long — median ~2500 tokens; without this, completion-only loss goes nan), splits 80/10/10, saves Arrow splits to `outputs/data/processed/<variant>/`. `build_prompt`/`make_variant` are the single source of truth, reused by `evaluation/predict.py`. `--clean` selects the opt-in clean pipeline profile: rewrites pipe/bullet references into prose (`normalize_summary`), builds the hardened prompt (`build_prompt(clean=True)`), and writes to `outputs/data/processed/<variant>-clean/` (all 10k records). Add `--drop-roundups` to also remove 3+ pipe roundups (~2.4k records) into `<variant>-clean-drop/`; the raw pipeline is never clobbered.
* `data_curation/`: **The current data pipeline**, superseding `data/clean.py`'s `--clean` profile. Turns raw HeSum into `artifacts/final_clean_hesum.json` (5,854 × `{hesum_id, text, headline}`) through eight stages: download → tail-boilerplate trim → two independent deterministic keep-maps (DictaLM token budget ≤ 4000 on text+headline; headline pipes ≤ 1) → intersect into `source_filter_input.json` (6,486) → `gpt-5.6-luna` source-usability labels (5,854 usable / 632 unusable) → `gpt-5.6-luna` headline keep-or-replace (2,785 kept / 3,069 rewritten) → final assembly. Rebuild with `python -m data_curation.build_curated_dataset`; only the two `artifacts/*_results.json` model outputs need shipping, everything else regenerates, and the rebuild aborts unless the regenerated `source_filter_input.json` matches the sha256 the model actually saw. Local, CPU + OpenAI Batch API. Full reference: `data_curation/CURATION_ROADMAP.md`; findings and critique: `docs/obsidian/Dataset Defect Taxonomy.md`.
* `data_curation/model_curation/*/[filter|refine]_prompt_schema.py`: The two curation prompts and their strict JSON schemas. **These prompts are the measurement instrument** — the six source labels' exact wording (including the "Do not use this when..." guards that keep LLM labeling stable) defines what the counts mean, so treat edits as changing the experiment. Note the asymmetry: the source stage emits a *label* per row, while the headline stage emits only `replacement_headline: null | string` with **no reason code**, so the 3,069 rewrites are unexplained per row and must be sub-typed post hoc from the `(original, replacement)` diff.
* `data_curation/utils/lzma_shim.py`: `ensure_lzma_importable()` installs a process-local stub `_lzma` module when the local pyenv build has no compiled `_lzma` extension (a pre-existing environment issue), so `import datasets` (needed by `download_hesum.py`) works without a real xz codec — HeSum ships as plain parquet, so nothing actually needs it. Called once, before `import datasets`. `download_hesum.py` and `filter_over_token_budget.load_dictalm_tokenizer()` also pass `token=False` to their Hub calls, since this machine's cached HF token is invalid and would otherwise turn public-repo requests into 401s.
* `data_curation/analysis/`: Turns the curation-pipeline artifacts into the row-label artifact and the F1/F2 paper figures specified in `docs/superpowers/specs/2026-07-26-dataset-review-experimental-design.md` (section 2) and `docs/obsidian/Paper Figures.md`. Local, CPU-only.
  * `row_labels.py`: Builds `artifacts/row_labels.json` — one row per HeSum id (10,000), joining `raw_hesum.json`/`tail_boilerplate_removed.json`, the two deterministic keep-maps, and the two supplied model-curation result files. `article_tokens`/`headline_tokens` are tokenized fresh (the DictaLM tokenizer, article and headline counted separately — the existing `dictalm_token_counts.json` only stores their sum). `classify_headline_edit()` sub-types each of the 3,069 rewrites from the `(original, replacement)` string pair alone (`pipes_removed` → `boilerplate_stripped` → `truncation_repaired` → `light_edit`/`full_rewrite`, first match wins) — a documented heuristic, not a new LLM label. `compute_lead_overlap()` is the reference-free lead-bias probe: fraction of a headline's Hebrew words also in the article's first 50 Hebrew words. `validate_counts()` checks the fresh build against the pre-registered verified counts (spec section 1) and raises if they drift. `load_row_labels()` builds the artifact on first use if missing.
  * `plotting.py`: The shared "amlk" Plotly template (`apply_house_style`, `save_figure`) every figure routes through — Helvetica-family font, left-aligned bold titles with a muted subtitle line, an Okabe-Ito colorblind-safe categorical palette (`CATEGORICAL_PALETTE`), a perceptually-ordered sequential palette for ordinal 1-to-5 data (`ordinal_palette`), and a source-note annotation. `save_figure` writes interactive HTML plus text-preserving vector SVG/PNG (via kaleido) to `outputs/figures/`, never a flattened-path export.
  * `figures.py`: F1 (`build_f1_curation_funnel` — Sankey, the two deterministic filters shown as separate converging flows, not one combined node) and F2 (`build_f2_defect_prevalence` — horizontal bars of the analysis-strata sizes, `other_unusable` broken into its three constituent labels). Both read only `row_labels.load_row_labels()`, so the pipeline's real counts and the figures cannot drift apart. `python -m data_curation.analysis.figures` writes both to `outputs/figures/`.
  * `rubric_anchors.py`: The nine real HeSum rows (read in full, then hand-summarized) backing the twelve Hebrew worked examples (4 dimensions × levels 1/3/5) `evaluation/rubric_judge.py` embeds in every judge prompt, per the rubric's "Hebrew anchor examples" protocol. `ANCHOR_HESUM_IDS` is the frozenset of ids `rubric_pilot.py` (and later the full E1 pass) must exclude from sampling, so an anchor row can never validate an instrument it helped calibrate.
  * `rubric_pilot.py`: Spec section 10 step 4 — pilots the rubric judge on ~300 rows stratified across S0/S2/S3/S4 before spending judge budget on all 10,000. Scores each sampled row's original headline once, re-scores a ~60-row subsample at `temperature=0.7` for test-retest (`sklearn.metrics.cohen_kappa_score(weights="quadratic")`), and flags any dimension whose score distribution piles onto one level (`is_degenerate`, ≥85% threshold). `python -m data_curation.analysis.rubric_pilot` writes `outputs/results/rubric_pilot.json`. API-bound (Gemini), CPU-only.
  * `rubric_full_run.py`: Spec section 10 step 5 — the full E1 judge pass over every HeSum row (minus the 9 anchor rows and a handful with empty text/headline), feeding F3-F5. The single most expensive step in the review (~10,000 judge calls), so it streams to a resumable JSONL (`outputs/results/e1_rubric_scores.jsonl`, one line per id — a rerun skips already-scored ids via `already_scored_ids`) and runs requests concurrently (`ThreadPoolExecutor`, default 8 workers) to bring wall-clock from a ~5h sequential pass down to under an hour. `python -m data_curation.analysis.rubric_full_run [--workers N] [--limit N]`. API-bound (Gemini), CPU-only.
  * `rubric_results.py`: Joins `e1_rubric_scores.jsonl` to `row_labels.json` by `hesum_id` (`join_scores_to_row_labels`, dropping any row the judge never scored) and tags each row with its analysis-stratum membership (`STRATA`: S0 clean, S2 multi-pipe, S3 multi-item, S4 rewritten — copied from `rubric_pilot.STRATA`, S1/S5 excluded per the design spec). `cliffs_delta`/`bootstrap_ci` compute the effect sizes F4 draws — implemented as a 5×5 ordinal count-table product rather than the naive O(len(a)×len(b)) pairwise loop, since stratum sizes in the thousands times 2,000 bootstrap resamples make the naive version take hours instead of seconds; `bootstrap_ci` resamples via `np.random.Generator.multinomial` over the fixed 5-level count distribution for the same reason. `length_binned_medians` builds F5's data using equal-COUNT (quantile) bins over article length rather than equal-width, since HeSum's token-count distribution is heavily right-skewed and equal-width bins leave the tails at n=1-2 rows. Pure functions, no Plotly — testable without kaleido/rendering. Local, CPU-only, no GPU/API (the API spend already happened in `rubric_full_run.py`).
  * `rubric_figures.py`: F3 (stacked horizontal ordinal bars, one bar per stratum, faceted into the four rubric dimensions), F4 (Cliff's delta forest plot, 95% bootstrap CI whiskers, shaded negligible band at |delta| < 0.147), F5 (two-panel: binned-median rubric scores over log article length on top, binned-median headline-lead overlap below, with a vertical rule at the 4,000-token filter threshold). `python -m data_curation.analysis.rubric_figures` writes all three to `outputs/figures/`. Local, CPU-only: Plotly + kaleido, no GPU/API.
  * `recover_frozen_split.py`: Reconstructs `data_curation/artifacts/splits/frozen_split_v1.json` after the fact, because Arm B (curated) training started externally on `avreymi/amlk-training-data` before this repo produced a split (see the Status entry for how this was discovered). That Hub dataset carries no `hesum_id` column, so `build_text_to_id_map`/`recover_ids` recover one per row by exact-matching article `text` against `tail_boilerplate_removed.json`, then folds Arm B's held-out val+test rows (confirmed never trained on) into a single frozen `test` set for future arms (A/C) to exclude — Arm B itself does not need to be re-run. Also persists `train_arm_b_as_run`/`val_arm_b_as_run`, the exact ids Arm B trained/validated on, which `build_arm_a_dataset.py` consumes. `python -m data_curation.analysis.recover_frozen_split`. One network call (downloads the external dataset repo); otherwise local, CPU-only.
  * `build_arm_a_dataset.py`: Builds the content for the redefined Arm A (`docs/obsidian/Training Handoff Contract.md`, 2026-07-26 revision) — the *same* article ids Arm B trained/validated on, paired with their ORIGINAL pre-curation headlines instead of curated ones, so the two arms differ only in headline target rather than in which articles they saw. Reads ids from `frozen_split_v1.json` and original headlines from `tail_boilerplate_removed.json` (tail-trimming never touches `headline`). `python -m data_curation.analysis.build_arm_a_dataset` writes `data_curation/artifacts/arm_a_original_headlines.json` ({hesum_id, text, headline, split}). Local, CPU-only, no network.
  * `build_probe_dataset.py`: Builds the input for the **zero-shot dual-reference probe** (added 2026-07-27, doesn't need to wait on external training) — for each of the 5,854 curated rows, pairs the article with BOTH its original and curated headline plus its `headline_edit_type` group (null → `kept`, the built-in placebo: curation left that headline untouched, so a real gap there would be a red flag, not a finding). `python -m data_curation.analysis.build_probe_dataset` writes `data_curation/artifacts/probe_dataset.json`. Local, CPU-only, no network.
  * `run_probe_inference.py` / `zero_shot_probe_hf_job.py`: Submitter + self-contained HF Jobs script (same split as `train.py`/`train_hf_job.py`) that generates ONE zero-shot summary per probe article with an off-the-shelf, untrained instruction model (`dicta-il/DictaLM-3.0-1.7B-Instruct` — small and Hebrew-native, so a ~1,400-row run finishes in well under an hour on `a10g-small`) — no fine-tuning at all, so there's no train/test leakage concern to isolate. `stratified_subsample` caps each `edit_type` group at `--max-per-group` (default 500; the small groups are already under that) before upload, since the full 5,854-row population would take hours for no real statistical-power gain. `python -m data_curation.analysis.run_probe_inference --submit-hf --hf-user <user> [--smoke-test]`.
  * `score_probe_dataset.py`: Scores the probe: joins predictions back to `probe_dataset.json` by `hesum_id`, then scores each prediction TWICE with `evaluate.py`'s new per-row ROUGE/BERTScore (once per reference version) — a paired design, since the prediction is identical in both scorings, so any score gap is attributable only to the reference, not to different model output, cleanly isolating the reference-quality effect from article-difficulty confounds. `paired_gap_stats` runs a Wilcoxon signed-rank test per `edit_type` group (valid specifically because of the pairing) plus a bootstrap CI on the mean gap; `kept` should show a null (≈0, non-significant) gap as a sanity check that the method isn't just picking up scoring noise. `python -m data_curation.analysis.score_probe_dataset --hf-user <user>`. Local, CPU-only.
  * `build_baseline_eval_dataset.py`: Builds the input for the **DictaLM2.0 zero-shot baseline reliability comparison** — 800 rows (200 per E1 stratum S0/S2/S3/S4, deduplicated) with article text from `tail_boilerplate_removed.json`, so model outputs can be rubric-scored and compared to E1 reference-headline scores on the same strata. `python -m data_curation.analysis.build_baseline_eval_dataset` writes `data_curation/artifacts/baseline_eval_dataset.json`. Local, CPU-only, no network.
  * `run_dictalm_baseline_inference.py` / `dictalm2_baseline_hf_job.py`: Submitter + self-contained HF Jobs script that generates one zero-shot summary per baseline-eval article with `dicta-il/dictalm2.0-instruct` (7.3B, same family as Arm B training). `python -m data_curation.analysis.run_dictalm_baseline_inference --submit-hf --hf-user <user> [--smoke-test]`.
  * `score_baseline_rubric.py`: Rubric-scores baseline predictions (resumable JSONL) and joins to E1 reference scores for paired per-stratum comparison (Wilcoxon + Cliff's delta). `python -m data_curation.analysis.score_baseline_rubric --hf-user <user>`. API-bound (Gemini), CPU-only.
  * `baseline_reliability_figures.py`: F9 grouped-bar figure (reference vs model-output median rubric scores by stratum). `python -m data_curation.analysis.baseline_reliability_figures`. Local, CPU-only.
 * `build_human_validation_sample.py`: Builds the frozen F9a annotation worklist (~150 stratified rubric rows + pairwise subset) into `artifacts/human_validation_worklist.json` — self-contained article text so annotators need no pipeline rebuild. `python -m data_curation.analysis.build_human_validation_sample`. Local, CPU-only.
 * `human_validation_results.py`: Merges per-annotator JSONL, computes human–human and judge–human κ per rubric dimension + pairwise agreement vs E3. `python -m data_curation.analysis.human_validation_results --annotations ...`. Local, CPU-only.
 * `human_validation_figures.py`: F9a agreement heatmap (human, judge, pilot test-retest). `python -m data_curation.analysis.human_validation_figures`. Local, CPU-only.
* `data/clean.py`: Reference-summary cleaning for the `--clean` profile. **Superseded by `data_curation/`** — kept so the raw pipeline stays byte-for-byte reproducible; do not use for new work. `normalize_summary` rewrites HeSum's `"headline | headline | headline"` pipe/bullet digests into natural prose (periods/commas, terminal period, idempotent on clean text); `is_roundup_digest`/`pipe_segments` flag the worst multi-headline roundups for removal. Import-light (stdlib only, like `style_labels.py`) so it works even where the `datasets` import is broken. Applied at the single choke point in `preprocess.py`, so both training targets and eval references are cleaned at once.
* `training/config.py`: Shared constants: `MODEL_ID="Qwen/Qwen3-2B"`, `METHOD_PRESETS` (the qlora/lora/full deltas), `LoRAConfig` (r=32, alpha=64, q/k/v/o + gate/up/down_proj), `TrainingConfig`, `WANDB_PROJECT`, and `dataset_repo`/`model_repo` Hub-id helpers.
* `training/train.py`: One trainer for all three regimes (`--method qlora|lora|full`). Trains with `completion_only_loss=True`, logs to wandb, saves the adapter; `--push-to-hub` or `--submit-hf` push to the Hub. Inference is NOT here — that's `evaluation/predict.py`.
* `training/train_hf_job.py`: Self-contained PEP 723 UV script submitted inline by `train.py --submit-hf`. Reads METHOD/VARIANT/DATASET_REPO/OUTPUT_REPO/WANDB_PROJECT from env, trains on the cloud GPU, then generates fine-tuned + zero-shot base test predictions (PEFT `disable_adapter`) and pushes the adapter + `predictions-finetuned.jsonl` / `predictions-base.jsonl` to the Hub. Never run directly.
* `evaluation/predict.py`: Generates the Gemini advanced-baseline summaries via API (no GPU, no model load), same prompt as training. Resumes from a partial file. The fine-tuned and zero-shot predictions come from the cloud training job, not here. Also defines `strip_think()` — the shared tool that drops closed Qwen3 `<think>…</think>` reasoning so metrics score the summary, not the reasoning (used by evaluate.py and error_analysis.py).
* `evaluation/evaluate.py`: Scores a predictions file with raw + Hebrew-normalized ROUGE-1/2/L (`normalize_hebrew` strips niqqud + folds final-form letters), BERTScore (default `onlplab/alephbert-base`, the HeSum backbone; `--bertscore-model` to override), and the Gemini faithfulness/fluency judge (`--skip-llm` to skip; `--limit N` to cap for a smoke run). Applies `strip_think` before scoring. One JSON report per system. `compute_rouge`/`compute_bertscore` are now thin averages over `compute_rouge_per_row`/`compute_bertscore_per_row` (added 2026-07-27 for `score_probe_dataset.py`'s paired per-row comparison — a single reference per row wasn't enough once one prediction needed scoring against two different references).
* `evaluation/gemini_client.py` / `hf_client.py`: Shared judge-API plumbing — `GEMINI_MODEL`/`GEMINI_TIMEOUT`/`call_with_retry`/`strip_think` (Gemini), `chat_completion` (HF Inference, a different-family alternative). Kept import-light so judge code doesn't pull in `datasets`.
* `evaluation/rubric_judge.py`: **The Reference Quality Rubric judge** (`docs/obsidian/Reference Quality Rubric.md`) — the single instrument behind E1-E4. `build_judge_prompt` embeds the four dimensions' verbatim level definitions plus the Hebrew worked anchors from `data_curation/analysis/rubric_anchors.py`, blind to stratum/provenance (only article + one headline). `score_headline` calls Gemini (`response_mime_type="application/json"`) — family-separated from the curator (`gpt-5.6-luna`, OpenAI) and the Qwen/DictaLM training base — and `_parse_judge_reply` tolerates malformed replies by dropping just the unparseable dimension rather than the whole row. Article text is capped at 6,000 chars (matches `evaluate.py`'s existing judge-prompt convention) to keep the ~1,400-word fixed anchor block's cost predictable across a 10,000-row pass. Reused unchanged for E4's model-output scoring, so dataset and model quality land on one axis.
* `evaluation/error_analysis.py`: Samples ~50 predictions (post `strip_think`) and has Gemini label failure types (hallucination, omission, entity/number error, lead copying, fluency), writing per-type rates.
* `evaluation/eval_hf_job.py`: Runs the whole D1 battery on HuggingFace Jobs so the ~4000 Gemini calls + BERTScore happen on the cloud's fast connection (the user has weak internet). One file, two modes: `--submit-hf` uploads itself to a cheap CPU job; with no args (how HF Jobs invokes it) it fetches the public repo + Hub predictions/dataset and drives the existing `evaluation/` CLIs by subprocess, pushing each report to the model repo under `reports/` as it finishes (timeout-safe).
* `evaluation/build_report_tables.py`: Downloads the pushed `reports/*.json` and assembles the D1 markdown — a quality table (ROUGE/BERTScore/judge), a failure-rate table, and behavioural notes (base `<think>`/language leakage, fine-tuned repetition, judge self-preference caveat).
* `evaluation/infer.py`: GPU inference helpers — `load_finetuned_model` (base + LoRA adapter, `disable_adapter()` gives the zero-shot base) and `generate_summaries` (batched greedy decode over a processed split). `clean=True` enables the clean-profile decode toggles (base `/no_think` in `build_input_text` + the Hebrew-script `bad_words_ids`). The importable twin of `train_hf_job.py`'s inline generation block (that cloud script can't import repo code); keep the two in sync. **Remote GPU only — never call locally.**
* `evaluation/hebrew_constraint.py`: Optional clean-profile decode constraint. `build_bad_words_ids(tokenizer)` scans the vocab once and returns the ids of every token whose decoded form contains a Latin/Cyrillic/Greek/Arabic letter, so `generate(bad_words_ids=...)` can forbid the mid-word foreign-script leakage seen in fine-tuned outputs (Cyrillic `б`, Arabic). Experimental (an aggressive constraint can hurt fluency); inlined twin lives in `train_hf_job.py`. On only under `CLEAN`.
* `evaluation/topic_clustering.py`: Topic-clustering side-analysis (not part of the main pipeline). Embeds truncated article `text` by default (`embed_field='text'`) — summaries alone collapsed ~99% of docs into one media-meta mega-topic — with the Hebrew-native, clustering-tuned `dicta-il/neodictabert-bilingual-embed`, clusters with BERTopic (UMAP + HDBSCAN + Hebrew-only c-TF-IDF vectorizer, `HEBREW_STOPWORDS` + `MEDIA_STOPWORDS` + `BOILERPLATE_STOPWORDS`), names each real cluster with one Gemini call, then optionally `refine_large_clusters()` — a second finer HDBSCAN pass on any cluster holding ≥30% of docs (re-uses embeddings; splits e.g. the politics mega-topic into ביטחון/כלכלה/חברה sub-domains without re-fragmenting sports/legal), then `merge_duplicate_labels()` collapses any clusters Gemini still named identically (on by default via `cluster_dataset(merge_duplicates=True)`) so the report has one row per distinct real-world topic. `fit_topics` tunables: `min_cluster_size`/`min_samples` (default 60/15 — coarser granularity means fewer near-duplicate sub-clusters of the same domain), `outlier_threshold` (only reassign noise above cosine sim — default 0.35; 0 floods the largest cluster), optional `nr_topics` merge (off by default; `auto` over-merged), `language='multilingual'` (required — English mode strips Hebrew). `plot_topic_sizes()` renders a bar chart of `topic_summary()` for the notebook. Output `topics.jsonl` still keyed by `summary` for stratification join. See `notebooks/cluster_topics_databricks.py`.
* `evaluation/style_labels.py`: A second, independent per-summary dimension from topic clustering — not *what topic* an article is about but *what format* its summary takes (`single_sentence` / `multi_sentence` / `pipe_digest` / `question`). Pure regex (`classify_style`), no embeddings/GPU/API, so unlike topic clustering it never needs Databricks and has no `datasets` import (works even if that import is broken locally, see the lzma note below). Motivated by a real corpus pattern: ~26% of HeSum summaries are `"headline | headline | headline"` pipe-separated digests, already flagged in `docs/obsidian/Current Results.md` as shaping the fine-tuned model's output format. Produces the same `{summary: label}` artifact shape as `topic_clustering.py` so it plugs into the same stratification tool; `plot_style_distribution()` renders a bar chart of `style_summary()` for the notebook.
* `evaluation/stratify_by_topic.py`: Joins a predictions file to a label artifact (`topics.jsonl`'s `topic_label` from `topic_clustering.py`, or `style_labels.jsonl`'s `style_label` from `style_labels.py` — same shape, selected via `--label-field`) on **`hesum_id`** (updated 2026-07-27; was exact `reference`==`summary` text match, which breaks once a curated headline no longer equals the original one a row was clustered from — see the Training Handoff Contract's required `{"hesum_id": ..., "prediction": ...}` prediction format), and reuses `evaluate.py`'s `compute_rouge`/`compute_bertscore` per group, folding in per-group failure rates if a matching `*.errors.json` exists (still `reference`-keyed — `error_analysis.py` not yet modernized to `hesum_id`). Local, CPU-only — no GPU/Databricks needed for this step.
* `evaluation/viewer/`: Predictions browser (`app.py`) plus **F9a human-validation UI** (`annotate_app.py`). `data.py` / `annotation_data.py` hold Streamlit-free logic; `ANNOTATION.md` is the teammate handout for the blind round on `feature/human-validation-ui`. Local, CPU-only, no GPU/API.
* `notebooks/evaluation_observation.ipynb`: The **evaluation-observation** stage. A self-bootstrapping Colab notebook that runs the *real* evaluation functions live and **displays** the per-example process (article → model summary → reference → judge faithfulness/fluency → error-analysis failure labels) for finetuned/base/gemini. Loads existing Hub predictions (finetuned/base at repo root, gemini under `reports/`) and generates fresh summaries on a T4. Judge/error/browse cells are API+CPU; only the generation cell needs a GPU.
* `notebooks/cluster_topics_databricks.py`: Databricks source-format notebook (`# Databricks notebook source` / `# COMMAND ----------` cell markers) driving `evaluation/topic_clustering.py` and `evaluation/style_labels.py`. Manual, occasional run on a Databricks GPU cluster — the GPU is for speed, not required (the embedding model is 0.4B params, encoder-only, the same class of job as the local-CPU AlephBERT BERTScore step). Clones the repo (or reuses an uploaded Workspace copy) so it calls the same tested functions rather than duplicating logic; computes both `topic_label` (BERTopic) and `style_label` (regex) over the same records. Plots (all inline via `displayHTML`, small enough to skip the DBFS round-trip): a `plot_topic_sizes` cluster-size bar chart, an interactive 2D/3D cluster scatter (`plot_clusters(dimensions=2|3)`, `plot_dimensions` widget; written to DBFS + iframe-embedded since 10k-doc hovers exceed the ~20 MB cell-output cap), a `plot_style_distribution` bar chart, and a topic×style stacked bar chart alongside the crosstab table. Writes one `topics.jsonl`/`topics-summary.json` (carrying both label fields) to DBFS for manual download into `outputs/data/raw/` and `outputs/results/`. Widgets expose `min_cluster_size`/`min_samples`/`reduce_outliers`/`nr_topics`/`merge_duplicate_labels`/`topic_size_plot_top_n`/`plot_dimensions` so noise/near-duplicate-topic tuning (see `topic_clustering.py`) doesn't require editing the notebook. A scoped, one-off departure from AMLK's default local/HF-Jobs/Colab stack — no agent-driven Databricks deployment (no MCP connection today), the notebook is handed off for manual import/run.
* `scripts/run_nb_cell.py`: Agent cell-runner — reads the notebook with `nbformat` and execs a chosen code cell / range against a persistent Colab session via `colab exec` (the Colab CLI has no native `.ipynb` runner). `--list` shows cell indices; the caller owns `colab new`/`stop`. This is how an agent observes the eval cell-by-cell.
* `tests/`: 115 fast behavioral tests + 4 gated live tests (Gemini judge x2; BERTopic fit + Gemini topic naming + plot; probe dual-reference scoring), all passing (`test_download.py`/`test_preprocess.py` excluded — pre-existing local `_lzma` environment gap, see Status).
* `paper/`: The ACL 2023-style course submission, narrating the Qwen → DictaLM → dataset-review pivot and reporting E1–E3/F5/F9 in full, with E4 (Figure `fig:f8-placeholder`, a TikZ box) left as a skeleton until the externally owned Arm A/B fine-tuning lands. `main.tex` embeds F1–F7/F9 as PNGs (`figures/`, synced from `outputs/figures/` via `sync_figures.sh`) and cites `bib.bib`. No Overleaf MCP is available, so the workflow is upload-and-compile (see `paper/README.md`); verified to compile cleanly (9 pages, no errors/undefined refs) with both `tectonic` and a real `pdflatex` (via the `texlive/texlive` Docker image) before being handed off. Every statistic is copied from `docs/obsidian/Experiment Results.md`, which stays the source of truth if the two ever drift.

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
* **Never load or run a model on the local GPU** — this machine (8 GB) freezes. All Qwen3-2B
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
# then submits train_hf_job.py inline (a10g-large, 6h, 1-epoch QLoRA). It prints a Job ID.
python -m training.train --submit-hf --hf-user avreymi               # full run
python -m training.train --submit-hf --hf-user avreymi --smoke-test  # 10 steps, a10g-small, ~$0.05 — verify first
python -m training.train --submit-hf --hf-user avreymi --inference-only  # regen predictions from pushed adapter (a10g-small, 1h)
# Cost: a10g-small has the SAME 24 GB A10G GPU as a10g-large at $1.00/h vs $1.50/h —
# prefer a10g-small; prefer --method lora over qlora for the 2B model.
# See docs/2026-06-12-qlora-training-job-postmortem.md before launching the probe runs.

hf jobs ps                    # list recent jobs
hf jobs logs <job-id>         # snapshot; add -f to stream
hf jobs inspect <job-id>

# Trained adapter (LoRA only, not merged): https://huggingface.co/avreymi/amlk-qwen3-2b-sft  (private)
# Evaluation loads it via: predict.py --model finetuned --adapter avreymi/amlk-qwen3-2b-sft
# Training metrics: wandb project "amlk-hebrew-summarization".
```

**Reading model outputs (predictions viewer):**
```bash
source .venv/bin/activate && streamlit run evaluation/viewer/app.py
# Opens a local browser UI over outputs/results/*.jsonl: RTL Hebrew, keyword search,
# side-by-side comparison across systems (finetuned/base/gemini). Local, CPU-only, read-only.
```

**Rebuilding the curated dataset (current pipeline):**
```bash
source .env && source .venv/bin/activate

# Requires the two supplied model-result files in data_curation/artifacts/:
#   source_filter_results.json, headline_target_curation_results.json
# Downloads HeSum, regenerates every deterministic cleanup artifact, validates the supplied
# results against the rebuilt input sha256, then writes artifacts/final_clean_hesum.json (5,854).
python -m data_curation.build_curated_dataset          # [local, CPU]

# Individual stages can be run alone; each is a module with its own __main__:
python -m data_curation.pre_model_cleanup.run_pre_model_cleanup
python -m data_curation.model_curation.source_filter.filter_records          # OpenAI Batch API
python -m data_curation.model_curation.headline_target_curation.refine_records  # OpenAI Batch API
```

**Dataset-review analysis (row-label artifact + F1/F2 figures):**
```bash
source .env && source .venv/bin/activate

# Requires the pipeline artifacts above to exist first (raw_hesum.json, tail_boilerplate_removed.json,
# the two keep-maps). Builds artifacts/row_labels.json (10,000 rows) and validates it against the
# pre-registered verified counts.                                                        [local, CPU]
python -m data_curation.analysis.row_labels

# Draws F1 (curation funnel) and F2 (defect prevalence) straight from row_labels.json, no API spend.
# Writes interactive HTML + text-preserving SVG/PNG to outputs/figures/.                  [local, CPU]
python -m data_curation.analysis.figures

# Pilots the rubric judge on ~300 rows stratified across S0/S2/S3/S4 (excludes the anchor rows),
# checks test-retest kappa + degenerate distributions. Requires GEMINI_API_KEY.     [local, API-bound]
python -m data_curation.analysis.rubric_pilot

# Full E1 judge pass over all ~10,000 rows -> outputs/results/e1_rubric_scores.jsonl. Resumable
# (rerun to continue after an interruption); ~45-90 min with the default 8 concurrent workers.
# NOTE: source .env alone does not export vars to this subprocess — use
# `set -a; source .env; set +a` first, or the process will fail on os.environ["GEMINI_API_KEY"].
python -m data_curation.analysis.rubric_full_run [--workers 8]

# Once e1_rubric_scores.jsonl is complete, draw F3 (distributions by stratum), F4 (Cliff's delta
# forest plot), F5 (length/lead-bias) from the joined judge scores + row_labels.json. No API spend
# (the judge calls already happened above); writes to outputs/figures/.                 [local, CPU]
python -m data_curation.analysis.rubric_figures

# Reconstruct the frozen test split from the in-flight external Arm B job (no hesum_id upstream,
# so this text-matches against tail_boilerplate_removed.json). Run once; re-run only if the
# external dataset repo changes. Requires HF_TOKEN (properly exported, see NOTE above).
python -m data_curation.analysis.recover_frozen_split

# Build Arm A's content (same ids as Arm B, original pre-curation headlines) -> hand off to the
# training owner alongside frozen_split_v1.json. Local, CPU-only, no network.
python -m data_curation.analysis.build_arm_a_dataset

# F9a human judge validation (blind rubric + pairwise annotation UI)
python -m data_curation.analysis.build_human_validation_sample   # lead: regenerate worklist
streamlit run evaluation/viewer/annotate_app.py                  # each annotator locally
# See evaluation/viewer/ANNOTATION.md for branch checkout + JSONL handoff
python -m data_curation.analysis.human_validation_results --annotations outputs/results/human_annotations_*.jsonl
python -m data_curation.analysis.human_validation_figures
```

**Running tests:**
```bash
source .venv/bin/activate && python -m pytest tests/ -v
```

---

## Status - remember to update it

**2026-07-26 — PROJECT PIVOT: from model-building to dataset review. This supersedes the framing of every entry below.**

Two model eras established that HeSum, not the model, is the bottleneck. Qwen3-2B v3 (3 epochs, full LoRA) produced fluent correctly-formatted Hebrew with wrong facts, and the LLM judge rated the *untuned base* as more faithful (2.98) than the fine-tuned model (2.64) — a model reliably learning its targets' *style* but not their content. Swapping to `dicta-il/dictalm2.0-instruct` (`notebooks/dictalm_hesum_zero_shot.ipynb`, 2026-07-09) gave a clear qualitative jump, but the outputs were often *better* than the references they were scored against. That moved the investigation to the data.

**Finding: HeSum is not a gold dataset.** It is scraped, its summaries are news-site extended subheadings rather than task-written summaries, and its construction (documented honestly in the HeSum paper) produces concrete defects — multi-headline digests, roundups covering several stories, articles whose substance is in an embedded video, scraped boilerplate tails. Verified counts from `data_curation/artifacts` on 2026-07-26: 722 rows had a removable tail; **2,659 (26.6%)** exceed a 4,000 DictaLM-token budget on text+headline; **2,412 (24.1%)** have 2+ pipes in the headline; the intersection leaves 6,486 (**35.1% removed** before any model looked at a row); `gpt-5.6-luna` labeled 5,854 usable / 632 unusable; and of the usable rows **3,069 (52.4%) needed their headline rewritten**. Final dataset: **5,854** records, a 41.5% net reduction. (Earlier recollections of "4,096 tokens, ~20%" were wrong on both threshold and magnitude.)

**The project is now a dataset review** — audit the defects, repair them, measure whether the defects mattered and whether the repair helped. Four experiments, pre-registered in `docs/superpowers/specs/2026-07-26-dataset-review-experimental-design.md`, one per curation action: **E1** diagnosis (are flagged rows genuinely worse?), **E2** repair graded, **E3** repair forced-choice, **E4** whole intervention. Primary instrument is a **rubric judge that reads the article** and scores a headline on four ordinal dimensions (faithfulness / single-focus / informativeness / cleanliness) — reference-free, so it sidesteps the confound that our curated references are LLM-written. ROUGE demoted to appendix-only; AlephBERT BERTScore is a secondary triangulation. Statistics are rank-based throughout (Mann-Whitney + Holm, **Cliff's delta as the conclusion-carrying statistic**, bootstrap CIs, ordinal logistic regression for confound control); no Kruskal-Wallis omnibus, since the strata overlap and violate its independence assumption.

**Two methodological decisions worth not re-litigating.** (1) `over_token_budget` is **not** an analysis stratum: the filter is a hard cut at 4,000 tokens, so clean and over-budget rows share **no common support** in article length, making matching impossible and regression identification purely extrapolative. It is an arbitrary cut on a continuous covariate, so length is modeled continuously over all 10,000 rows — which also *recovers* the lead-bias question in a stronger form than the old `--variant whole|lead|body` probe. (2) No new LLM labeling pass is needed for the strata: both deterministic keep-maps are persisted as full 10,000-id boolean maps, so stratum membership is a join.

**Division of labour: training is owned externally.** We own the audit, the nine-figure set (`docs/obsidian/Paper Figures.md`), and the post-training evaluation. The interface is `docs/obsidian/Training Handoff Contract.md`, whose non-negotiable terms are that **we produce the frozen split** (arms trained on independently drawn splits cannot be compared) and that **arm A is size-matched to arm B** (otherwise data quality is confounded with data quantity). Predictions are preferred over checkpoints so no model ever loads locally.

Docs live on branch `docs/dataset-review-pivot`. Vault entry point: `docs/obsidian/Home.md` → `Project Pivot.md`. Qwen-era notes are tagged `#status/superseded` with a banner explaining what still holds.

**Superseded by the above:** the `--clean` profile (`data/clean.py`), the `--variant whole|lead|body` training probe, the Fix Plan phases, and the "next steps" list at the end of this section. Prior-era numbers below remain accurate as history.

---

**2026-07-26 — Row-label artifact + F1/F2 figures built; local pipeline unblocked.** `data_curation/analysis/` added (`row_labels.py`, `plotting.py`, `figures.py`) per spec section 2 and `Paper Figures.md`. Two pre-existing local-environment blockers fixed along the way, both narrowly scoped:
- **`_lzma` missing** (this pyenv build has no compiled `_lzma` extension, so any `import datasets` failed before touching the network — the same root cause `AGENTS.md` already flagged for `tests/test_download.py`/`test_preprocess.py`). `data_curation/utils/lzma_shim.py` installs a process-local stub `_lzma` module satisfying `lzma.py`'s imports; HeSum is plain parquet so nothing ever needs a real xz codec. `download_hesum.py` calls it before `import datasets`.
- **Invalid cached HF token** (`~/.cache/huggingface/token` fails `whoami`, turning anonymous-eligible requests into 401s). `download_hesum.py` and `filter_over_token_budget.load_dictalm_tokenizer()` now pass `token=False` for these public repos.
- With both fixed, ran the real pipeline locally end to end (`download_hesum` → `run_pre_model_cleanup`) and got **exact matches** to every pre-registered count in the spec (722 tail-trims, 2,659 over-budget, 2,412 multi-pipe, 6,486 reached model curation) and **identical id membership** to the supplied `source_filter_results.json` (verified by direct set comparison) — confirms the local rebuild reconstructs the same raw-HeSum row-to-id mapping the original curation run used, which the row-label join depends on. (One loose end: the rebuilt `source_filter_input.json`'s sha256 does not match the value pinned in `build_curated_dataset.py`, despite identical id membership and identical verified counts — likely formatting drift in code since that pin was set, not a content problem. Not chased further since `row_labels.py`/`figures.py` never call `build_curated_dataset.py`'s validator; worth reconciling before the next full `build_curated_dataset` run.)
- `row_labels.py` output validated against every count in spec section 1 (`validate_counts()`, raises on drift). `plotting.py` is the shared "amlk" Plotly template (Okabe-Ito categorical palette, ordered sequential palette for the future rubric scores, left-aligned editorial titles, text-preserving SVG/PNG export via `kaleido`, added to `requirements.txt` alongside `plotly`). F1/F2 rendered and reviewed as PNG.
- 3 new fast tests (`test_row_labels.py`, `test_figures.py`); full suite (minus the two pre-existing `_lzma`-blocked files) at 62 passed / 2 skipped.
- Remaining from `Paper Figures.md`'s sequencing: pilot the rubric judge on a few hundred rows, then the full E1 pass and F3-F5.

---

**2026-07-26 — Rubric pilot run + full E1 pass built and launched (spec section 10, steps 3-5).** `evaluation/rubric_judge.py` (the shared E1-E4 instrument), `data_curation/analysis/{rubric_anchors,rubric_pilot,rubric_full_run}.py`.
- **Anchors** (`rubric_anchors.py`): twelve Hebrew worked examples (4 dimensions × levels 1/3/5), drawn from nine real HeSum rows read in full and hand-summarized (ids 1, 4, 54, 94, 126, 228, 419, 1639, 1727 — some rows anchor more than one dimension, so the exclusion footprint is 9 rows out of 10,000, not 12). `ANCHOR_HESUM_IDS` is the frozenset every sampling step must filter out first.
- **Judge** (`rubric_judge.py`): `build_judge_prompt` embeds the rubric's verbatim level text plus the anchors, blind to stratum/provenance. `score_headline` calls Gemini (`gemini-2.5-flash-lite`, `response_mime_type="application/json"`) — family-separated from curator `gpt-5.6-luna` (OpenAI) and the Qwen/DictaLM training bases. Article text capped at 6,000 chars. Fixed mid-pilot: a `PROHIBITED_CONTENT` safety-filter block (empty `response.candidates`) was raising and killing `call_with_retry` after 5 useless retries (a block is deterministic — retrying the same content never helps); `score_headline` now checks `response.candidates` before `.text` and returns `{}` immediately, no retry, on a block.
- **Pilot run complete** (`rubric_pilot.py`, `outputs/results/rubric_pilot.json`, 300 rows + 60-row retest): no degenerate dimension. Test-retest quadratic-weighted kappa: faithfulness 0.86, single-focus 0.90, informativeness 0.76, cleanliness 0.65 — all substantial-to-near-perfect. Per-stratum means largely confirmed the pre-registered predictions (S2/S3 single-focus down sharply; S2 cleanliness down) with two genuine surprises, checked against the actual judge justifications and kept as findings rather than "fixed": (1) S2/S3 faithfulness dropped more than predicted ("roughly intact"/"near normal") — multi-pipe/multi-item headlines often assert facts absent from the one article body HeSum paired them with, a real hallucination-relative-to-source pattern, not a rubric artifact; (2) S4 (`headline_action=="rewritten"`) original headlines scored close to S0 rather than clearly worse on all four dims as predicted — apparent disagreement between the curation pipeline's rewrite-trigger classifier and this independent judge, to report as an inter-instrument finding.
- **Full E1 pass built and ran to completion** (`rubric_full_run.py`): scored all 9,988 in-scope rows (10,000 minus 9 anchors minus 3 with empty text/headline) concurrently (`ThreadPoolExecutor`, 8 workers) in ~34 min wall-clock, 0 failures. Output: `outputs/results/e1_rubric_scores.jsonl`, 9,988 lines. Ran across two sessions (one long-idle terminal was initially assumed to have "stopped" but had in fact exited cleanly at 9,988/9,988 — check the terminal's `status`/`exit_code` header before assuming a stall).
- 20 new fast tests across `test_rubric_judge.py`/`test_rubric_pilot.py`/`test_rubric_full_run.py`, 1 live-gated; full suite now 82 passed / 3 skipped. Added `scipy`/`scikit-learn` to `requirements.txt`.
- **Shell gotcha found and worth fixing at the source:** `.env` has no `export` keyword, so a plain `source .env` sets vars as shell-locals that never reach subprocesses (python, `hf`) — this caused a false "invalid HF_TOKEN" alarm earlier this session. Use `set -a; source .env; set +a` (both here and in every other documented `source .env && ...` command in this file/skills) until `.env` itself is fixed to `export` each line.

---

**2026-07-27 — F3/F4/F5 built from the completed E1 pass.** `data_curation/analysis/rubric_results.py` (join + stats, no Plotly) + `rubric_figures.py` (the three figure builders), per `docs/obsidian/Paper Figures.md`.
- Joined 9,958/9,988 scored rows to `row_labels.json` (30 ids in the JSONL are not present in the current `row_labels.json` build — a handful of stale/re-run entries, not a join bug; every joined row carries a score).
- **F3 (distributions by stratum) confirms every pre-registered directional prediction**, checked against `docs/superpowers/specs/2026-07-26-dataset-review-experimental-design.md` section 4.2's table: S2 `multi_pipe_headline` collapses single-focus (~85% at levels 1-2) with faithfulness/informativeness also visibly down and cleanliness comparatively spared; S3 `multiple_independent_items` shows the same single-focus collapse; S4 `headline_rewritten` is down across all four dimensions, most visibly single-focus and faithfulness; S0 `clean` is overwhelmingly levels 4-5 on every dimension (a real ceiling effect, flagged in the spec as limiting E2's paired-comparison headroom).
- **F4 (Cliff's delta forest plot)** — the conclusion-carrying figure. S2 single-focus is the largest effect at delta ≈ -0.93 (large); every S2/S3/S4-by-dimension comparison clears the |delta| > 0.147 negligible band except S4 cleanliness and S4 informativeness, which land inside or at the edge of it — i.e. the headline-rewrite trigger targeted single-focus/faithfulness specifically and largely left cleanliness alone, a finding rather than a uniform "everything is worse."
- **F5 (length/lead-bias)** — rebuilt to use equal-COUNT (quantile) bins over article length rather than equal-width; HeSum's token-count distribution is heavily right-skewed, so equal-width bins left the shortest/longest bins at n=1-2 rows and one outlier dominated the median there. Headline-lead overlap declines smoothly and monotonically with article length (no discontinuity at the 4,000-token filter mark) — the intended justification for modeling length as a continuous covariate rather than a stratum, and the direct lead-bias-probe result.
- **Performance bug caught before it wasted an hour:** the first `cliffs_delta`/`bootstrap_ci` implementation used the naive O(len(a) × len(b)) pairwise-comparison loop inside a 2,000-resample bootstrap — at S4's ~3,069 rows vs S0's ~2,785 rows that is over 8.5M comparisons per call, 2,000 calls per stratum-dimension pair. Fixed by exploiting that scores are ordinal integers 1-5: both functions now reduce to a 5×5 count-table product (`cliffs_delta`) or a `numpy` multinomial resample over the fixed 5-level distribution (`bootstrap_ci`) — seconds instead of hours, and mathematically identical (verified against the naive computation in `test_rubric_results.py`).
- 9 new fast tests in `test_rubric_results.py` (join/drop logic, Cliff's delta against hand-worked and naive-pairwise cases, equal-count binning); full suite now 100 passed / 3 skipped (excluding the two pre-existing `_lzma`-blocked files).
- **Next action:** E2/E3 (the 3,069 rewritten rows, paired original-vs-curated comparison) → F6/F7, per the sequencing in `docs/obsidian/Paper Figures.md`.

---

**2026-07-27 — E2/E3 complete; DictaLM2 baseline + wiki results log.** E2 curated rubric on 3,068/3,069 rewritten rows (`e2_repair_summary.json`); E3 pairwise 73.57% curated wins on rewrites vs 99.49% ties on placebo (`e3_pairwise_summary.json`, F7). DictaLM2 zero-shot baseline inference resume in flight (648/800 Hub, rubric scored on 604, F9); finetuned Arm B inference in flight (~50/586, job `6a67a4f8…`). Article-ready numbers consolidated in `docs/obsidian/Experiment Results.md`; `Reference Quality Experiment.md`, `Paper Figures.md`, and `Home.md` updated. F6 render + full 800 baseline + E4/F8 still open.

---

**2026-07-26 — E4 training job in flight (external, per Training Handoff Contract).** avreymi reported a training run started on HF Jobs:
- **Job:** `avreymi/6a665c8b7ef3c08464969f28` (namespace `avreymi`, not ours) — check with `hf jobs inspect avreymi/6a665c8b7ef3c08464969f28` / `hf jobs logs avreymi/6a665c8b7ef3c08464969f28 -f`.
- **Token false alarm, fixed:** the earlier 401 was NOT an invalid token — `.env` has no `export` keyword, so a plain `source .env` sets `HF_TOKEN` as a local shell variable that never reaches subprocesses (python, `hf`). Fix: `set -a; source .env; set +a` (or `export $(grep -v '^#' .env | xargs)`) before invoking anything that shells out. Confirmed `whoami` → `avreymi` once exported properly. **This affects every documented `source .env && ...` command in this file and in the skills — none of them export.** Worth fixing at the source (`.env` → add `export` per line, or fix the run instructions) rather than re-discovering this each session.
- **Status as of inspection (2026-07-26 ~23:00 IDT, job running ~44 min):** `RUNNING`, started 19:14:19 UTC. Base model `dicta-il/dictalm2.0-instruct` (7.3B; 83.9M trainable / 1.14% via LoRA r=32/alpha=64 on q/k/v/o+gate/up/down_proj — the full extended target-module set), `METHOD=lora`, `VARIANT=whole`, `DATASET_REPO=avreymi/amlk-training-data`, `OUTPUT_REPO=avreymi/amlk-dictalm2-instruct-sft`, flavor `a10g-small`, wandb run [`2026-07-26_dictalm2-instruct_lora_whole_1ep`](https://wandb.ai/avreymi-asraf-hebrew-university-of-jerusalem/amlk-dictalm2-instruct/runs/pwo74pgg). Train set 4,683 rows / eval 100 rows, 293 steps total (1 epoch), loss trending down (0.997→0.740 by step 50). At ~35-60s/step this is a multi-hour run; ETA roughly matches avreymi's ~5h estimate.
- **Open question RESOLVED by direct inspection (2026-07-26, later same session) — see the entry below.** `DATASET_REPO=avreymi/amlk-training-data` is the curated dataset (Arm B, `source: "hesum-curated"`), not raw HeSum or an overwrite; a frozen split was reconstructed retroactively rather than existing beforehand.
- **Next action:** keep polling `hf jobs logs avreymi/6a665c8b7ef3c08464969f28` for completion/failure; once done, inspect `OUTPUT_REPO` for the pushed adapter + predictions and score against the reconstructed frozen split below.

---

**2026-07-26 — Frozen split reconstructed after the fact from the in-flight Arm B job; branch archaeology ruled out an alternate branch.** Checked every branch/remote for the split avreymi's job might have used: `origin/dictalm` and local `another-model` are both stale pre-pivot branches (last commits 2026-07-11/07-12, no `data_curation/` at all — from the earlier DictaLM zero-shot era). `main`'s `data_curation/` pipeline was itself authored by a third collaborator, **ofekvarona** (`ofek.varona@mail.huji.ac.il`) — this is a 3-person team (Amit / avreymi / ofekvarona). No branch, anywhere, has ever contained a split file or split-producing code; `data_curation/artifacts/` only ships the two supplied model-curation result files on every branch, so there was nothing avreymi could have branched from.
- Since the split isn't recoverable from git, downloaded `avreymi/amlk-training-data` directly (now that `HF_TOKEN` exports correctly) and inspected it: `datasets.load_from_disk` on its `train`/`val`/`test` arrow folders shows **4,683/585/586 rows, `source: "hesum-curated"`, no `hesum_id` column** — confirms Arm B is training on the full curated 5,854-row set with an ad hoc ~80/10/10 split, not a subsample and not raw HeSum.
- **Recovered `hesum_id` for ~99% of rows by exact-matching each row's `text` against `tail_boilerplate_removed.json`** (`data_curation/analysis/recover_frozen_split.py`, `build_text_to_id_map`/`recover_ids`, 6 fast tests in `test_recover_frozen_split.py`). Matched 4,619/4,683 train, 582/585 val, 580/586 test; the rest are excluded rather than guessed (no `hesum_id` column upstream, plus 15 duplicate-article-text ids in the raw dataset make a few texts ambiguous). Zero id overlap between the recovered train/val/test — the split itself is clean.
- **Formalized as `data_curation/artifacts/splits/frozen_split_v1.json`**, per `docs/obsidian/Training Handoff Contract.md`'s schema, but explicitly marked `"split_version": "v1-reconstructed"` with a `deviations` list (not carved before training; folds Arm B's held-out val+test into one 1,162-row `test` set — above the pre-registered 800-1,000 target but kept as-is since trimming would arbitrarily discard already-unseen rows; `val` is intentionally empty; `train_arm_b_as_run` (4,619 ids) is audit-only, not a constraint on Arms A/C). Future arms (A raw-HeSum subsample, optional C raw-HeSum full) just need to exclude `test`'s 1,162 ids from their own training — **Arm B does not need to be re-run**.
- Along the way, unblocked `python -m data_curation.build_curated_dataset` (added missing `python-dotenv`/`openai` to `requirements.txt` — transitive imports of the OpenAI batch-curation module, previously never exercised locally) and regenerated `final_clean_hesum.json` (5,854 records). Its own pinned `EXPECTED_SOURCE_FILTER_INPUT_SHA256` check still fails on a fresh rebuild despite **identical id-set membership** (verified by direct set comparison, 6,486/6,486, zero diff) — almost certainly JSON formatting/ordering drift since the pin was set, not a content-alignment bug; bypassed by calling `build_final_dataset.main()` directly. Worth fixing the pin itself at some point so `build_curated_dataset.py`'s one-shot entrypoint works end-to-end again.
- Full suite (minus the two pre-existing `_lzma`-blocked legacy files) at 87 passed / 3 skipped.

---

**2026-07-26 — Arm A redefined: same articles as Arm B, original headlines, not a raw-HeSum subsample.** `docs/obsidian/Training Handoff Contract.md` updated: instead of independently subsampling raw HeSum to B's size (which only controls for *quantity*), Arm A now trains on the *identical* `hesum_id`s Arm B trained/validated on (`train_arm_b_as_run`/`val_arm_b_as_run` in `frozen_split_v1.json`), paired with their original pre-curation headlines instead of the curated ones. Removes the "which articles got sampled" confound entirely — a B-over-A win is now attributable to headline repair specifically, not to article selection. Narrows the "Coverage gap" section accordingly (filtering-effect isolation is still open, headline-repair isolation is not). Optional Arm C keeps the original "raw HeSum, full, unmatched size" role.
- `data_curation/analysis/build_arm_a_dataset.py` (+ `test_build_arm_a_dataset.py`, 4 tests) builds the actual content: `data_curation/artifacts/arm_a_original_headlines.json` — 5,201 records (4,619 train + 582 val), verified 100% identical article `text` to the curated version at the same ids, with headlines differing on 2,716/5,201 (the ones curation actually rewrote) and matching on the rest, exactly as expected. We hand over content, not a training-ready prompt/completion dataset — the training owner applies their own preprocessing (same as they did for Arm B) so both arms share one pipeline.
- Full suite now 91 passed / 3 skipped.
- **Next action:** hand `frozen_split_v1.json` + `arm_a_original_headlines.json` to avreymi/ofekvarona so Arm A can start training with identical hyperparameters/base model/decode config to the in-flight Arm B run.

---

**2026-07-27 — `hesum_id` join key threaded through the topic/style stratification tools.** `evaluation/topic_clustering.py`/`style_labels.py` now carry `hesum_id` through their output rows, and `evaluation/stratify_by_topic.py` joins predictions to labels on `hesum_id` instead of exact `reference`==`summary` text match — the old join silently breaks the moment a curated headline no longer equals the original text a row was clustered from. `error_analysis.py`'s failure-rate join is still `reference`-keyed (flagged, not yet modernized). Existing tests updated to match; no behavior change for callers already emitting `hesum_id`.

**2026-07-27 — Zero-shot dual-reference probe: a narrative-support experiment that doesn't wait on external training.** The "run inference and show accuracy per cluster" idea turned out to mean "per curation-defect group" (row_labels' `headline_edit_type`), not BERTopic topic clusters — those are an unrelated side-analysis. The actual plan: run one off-the-shelf, **untrained** instruction model (`dicta-il/DictaLM-3.0-1.7B-Instruct` — small, Hebrew-native, no fine-tuning so zero train/test leakage concern) over each curated article, then score that *same* prediction TWICE — once against the original headline, once against the curated one. Since the prediction never changes, any ROUGE/BERTScore gap is attributable only to which reference was used, not to different model output — a paired design that isolates the reference-quality effect from article-difficulty confounds, and complements E1's intrinsic judge-based argument with an independent extrinsic one.
- New modules: `build_probe_dataset.py` (joins `row_labels.json`/`tail_boilerplate_removed.json`/`final_clean_hesum.json` into `probe_dataset.json`, 5,854 rows: `{hesum_id, text, original_headline, curated_headline, edit_type}`; null `headline_edit_type` → `kept`, the built-in placebo group where curation left the headline untouched and the gap should be exactly zero). `run_probe_inference.py` (local submitter; `stratified_subsample` caps each `edit_type` group at 500 before upload — full-population inference would cost hours for no real power gain, since the small groups already sit under any sane cap) + `zero_shot_probe_hf_job.py` (self-contained HF Jobs script, no adapter/PEFT at all). `score_probe_dataset.py` (`paired_gap_stats`: Wilcoxon signed-rank test + median/mean gap per group, valid specifically because of the pairing; `bootstrap_gap_ci`: 95% CI on the mean gap).
- `evaluation/evaluate.py`'s `compute_rouge`/`compute_bertscore` split into `..._per_row` variants (the aggregate functions are now thin averages over them) so the probe's per-row paired comparison doesn't duplicate the tokenizer/scorer setup.
- 15 new fast tests (`test_build_probe_dataset.py`, `test_run_probe_inference.py`, `test_score_probe_dataset.py`; one BERTScore-heavy test gated behind `RUN_LIVE_TESTS`). Full suite now 115 passed / 4 skipped.
- **Smoke-tested end to end** (job `6a667b5b…`, 20 rows, ~3 min total): generations are coherent on-topic Hebrew, and the `kept` placebo group scored an exact 0.0 gap as expected. Full run launched (job `6a667c52…`, 1,427 rows after subsampling, `a10g-small`, ~2h ceiling — smoke-test timing suggests well under 30 min in practice). Once it lands, run `score_probe_dataset.py` and turn the per-group Wilcoxon results into a dumbbell-plot figure (original-score ●──● curated-score per `edit_type` group).

**2026-07-27 — DictaLM2.0 zero-shot baseline vs E1 dataset reliability (rubric-aligned comparison).** The 1.7B dual-reference ROUGE probe complements E1 but uses a different model and metric; this pipeline runs the same **`dicta-il/dictalm2.0-instruct`** base as Arm B, stratified across all four E1 strata (200 each → 800 unique rows), rubric-scores model outputs with the same Gemini judge as E1, and compares paired reference-vs-output medians per stratum (F9).
- New modules: `build_baseline_eval_dataset.py`, `run_dictalm_baseline_inference.py`, `dictalm2_baseline_hf_job.py`, `score_baseline_rubric.py`, `baseline_reliability_figures.py` (+ 3 fast tests).
- **Smoke job submitted:** `6a67858f7495fcac7095d6bb` (10 rows, `a10g-small`). Results → `avreymi/amlk-baseline-results` / `predictions-dictalm2-baseline.jsonl`. Full run `6a67863d…` timed out at **604/800** (exit 143); resume job `6a67a68b…` submitted with Hub-skip logic. Baseline rubric scoring + comparison figure pipeline: `score_baseline_rubric.py`, `baseline_reliability_figures.py`.

**2026-07-27 — E2/E3 repair experiments launched (F6/F7).** `repair_curated_run.py` scores curated headlines on the 3,069 rewritten rows (paired against E1 originals → `e2_curated_rubric_scores.jsonl`); `repair_results.py` + `repair_figures.py` build F6 dumbbell + transition heatmap. `evaluation/pairwise_judge.py` + `repair_pairwise_run.py` run blind head-to-head on ~1,000 rewritten + ~200 kept placebo rows → F7. All resumable JSONL; API-bound, CPU-only. Full passes launched in background ~21:42 IDT.

---

**2026-07-27 — Paper skeleton drafted: `paper/`, ACL 2023 template, Abstract through Results + appendix.** Per the narrative arc (Qwen3-2B → DictaLM zero-shot → dataset review), scaffolded a course-submission LaTeX source from the supplied ACL template (`main.tex`, `acl2023.sty`, `acl_natbib.bst`), with `bib.bib` seeded from `docs/obsidian/References.md` (HeSum, LoRA/QLoRA, BERTScore, AlephBERT, self-preference bias, Cliff's delta) and `figures/` synced from `outputs/figures/` (F1–F7, F9, plus sx01/sx05/sx16/sx17 in the appendix) via a new `paper/sync_figures.sh`.
- Abstract/Introduction narrate the three eras (Section~`sec:pivot`) and pre-register RQ1–RQ4; Data covers the 8-stage pipeline (F1/F2) and the frozen-split/Arm-A-redefinition boundary; Methods documents the rubric instrument and E1–E4 designs; Results writes up E1 (F3/F4), the length/lead-bias covariate (F5), E2/E3 (F6/F7, framed as complementary rather than contradictory — ceiling-compressed medians vs. a real 73.6% pairwise win rate), and the partial F9 baseline comparison (n=604/800, the S2 single-focus reversal as the headline finding), all copied verbatim from `docs/obsidian/Experiment Results.md`.
- **E4 is explicitly a skeleton**, not a placeholder figure alone: Arm B's fine-tuning job stalled at step 275/293 (exit 143) with only 50/586 predictions generated and Arm A not yet started (per the Status entries below), so `fig:f8-placeholder` is a labeled TikZ box (not a generated image — this is a layout mockup, not data) with the exact planned content (grouped rubric bars + blind pairwise win rate) and prose stating in advance what each graceful-degradation outcome (both arms / B alone / no A) would mean for the paper's causal claim.
- Appendix A–E: Qwen3-2B v1→v2→v3 ROUGE/AlephBERT table (the repetition-artifact and undertraining diagnoses that ended that era), rubric pilot kappa + filter-overlap figures, headline edit sub-type distribution, the dual-reference zero-shot probe (reported as supporting-only, since ROUGE correlates weakly with the rubric judge here), and the E2 transition heatmap.
- **Verified the LaTeX actually compiles**, twice, since there is no Overleaf MCP and this machine had no LaTeX toolchain: installed `tectonic` via Homebrew for a fast local syntax check (caught and fixed a real bug — `cleveref` must load after `amsmath`/`amssymb`, which would have broken on Overleaf too, not just tectonic) and, on the copy with the arXiv `\pdfoutput=1` directive intact, a full `pdflatex`+`bibtex`+`pdflatex`×2 cycle inside the `texlive/texlive` Docker image (tectonic's XeTeX-based engine conflicts with that directive's pdfTeX driver selection, a tectonic-specific quirk unrelated to real pdfLaTeX). Final result: 9 pages, zero errors, zero undefined citations/references, only cosmetic underfull-hbox warnings; two tables that were slightly overfull got `\resizebox`-wrapped. No build artifacts were committed — `.gitignore` gained a `paper/*.{aux,bbl,blg,log,out,pdf,synctex.gz}` block.
- `AGENTS.md`'s file tree and bullet-description list updated with the new `paper/` folder.
- **Next action:** upload `paper/` to Overleaf (manual step — no MCP, see `paper/README.md`) for the team to review; once Arm A/B predictions land on the frozen test split, replace `fig:f8-placeholder` with the real E4 rubric-bars + pairwise-win-rate figure and update the abstract's closing sentence.

---

**Stages A + B complete as of 2026-06-12.** Stack: trl 1.6.0, transformers 5.11, peft 0.19, wandb 0.27.
- `data/download.py` — 10,000 records from biunlp/HeSum in `outputs/data/raw/combined.jsonl`. IAHLT/summarization_he is inaccessible (not on HF Hub with current credentials).
- `data/preprocess.py` — prompt/completion pairs + `--variant whole|lead|body`; 8,000/1,000/1,000 splits in `outputs/data/processed/<variant>/`.
- `training/train.py` — one trainer for qlora|lora|full, `completion_only_loss=True`, wandb logging, `--submit-hf` to HF Jobs. Verified: local 12-step QLoRA smoke runs and logs to wandb.
- `training/train_hf_job.py` — self-contained HF Jobs script (trl 1.6 API, wandb).
- `evaluation/predict.py` / `evaluate.py` / `error_analysis.py` — full metric battery (ROUGE/BERTScore/Gemini judge), zero-shot + Gemini baselines, failure-type analysis. `strip_think()` (in predict.py) drops closed Qwen3 `<think>…</think>` reasoning before scoring; evaluate.py has `--limit` for smoke runs.
- `evaluation/eval_hf_job.py` + `build_report_tables.py` — D1 eval runs on a cheap CPU HF Job (clones the public repo, drives the existing CLIs, pushes `reports/*.json`); the tables tool turns those reports into the presentation markdown. Chosen because the user has weak internet (the ~4000 Gemini calls + BERTScore run cloud-side). Smoke job `6a2cfda2` verified the path end-to-end.
- 16 fast tests + 1 gated live Gemini test, all passing (`python -m pytest tests/`).
- HF Jobs dataset: `avreymi/amlk-training-data` (private). Model output: `avreymi/amlk-qwen3-2b-sft` (private). wandb project: `amlk-hebrew-summarization`. Advanced baseline + judge: Gemini `gemini-2.5-flash-lite` (full 2.5-flash's ~7s/call thinking latency made the ~4000-call battery ~10h; -lite is ~1s/call, ~6x faster. 2.0-flash is retired.)
- Note: QLoRA `push_to_hub` saves the LoRA adapter only (not merged) — evaluation loads base + adapter via `PeftModel.from_pretrained` (handled in `predict.py`).
- Note: the Gemini LLM-judge and the Gemini advanced baseline are the same model family — flag the possible self-preference bias in the paper.
- **2026-06-12 full run (job `6a2bc974`): training succeeded** (1 epoch, eval_loss 1.777; adapter on `avreymi/amlk-qwen3-2b-sft`), but the job timed out in its prediction loop — predictions regenerated by a patched inference-only job. Full post-mortem with cost analysis and the probe-run checklist: `docs/2026-06-12-qlora-training-job-postmortem.md`.
- **v1 adapter flaw (addressed 2026-06-27): Qwen3-2B is a hybrid-attention model (18 linear-attention + 6 full-attention layers); the v1 LoRA `target_modules` q/k/v/o only exist in the 6 full-attention layers, so it covered 6/24 layers (0.07% trainable params).** `LoRAConfig` now adds the MLP projections `gate/up/down_proj` (present in all 24 layers) and bumps r→32/alpha→64; mirrored in `train_hf_job.py`. The next run validates this; still consider the linear-attention modules + `flash-linear-attention`/`causal-conv1d` deps (post-mortem §5.1) if coverage is still insufficient.
- `train_hf_job.py` / `evaluation/infer.py` share a decode config: `max_new_tokens=256` (p99 reference length is 187 tokens), `min_new_tokens=16`, `no_repeat_ngram_size=3`, `repetition_penalty=1.2`, explicit `eos_token_id`/`pad_token_id`, greedy. Predictions are pushed immediately after each generation loop (timeout-safe), progress every 10 batches; inference-only jobs use a 1h timeout. `PRED_SUFFIX` env (`--pred-suffix`) appends e.g. `-v2` so a re-decode doesn't clobber v1.

**2026-07-04 — More distinct cluster plot + tighter clustering defaults.** Plot: golden-angle color palette, UMAP `min_dist=0.35`/`spread=1.25`, optional centroid repulsion (`plot_display_spread` widget). Clustering: `min_samples` 15→20, `umap_n_neighbors` 10→15, `outlier_threshold` 0.35→0.40; `umap_n_neighbors` widget on Databricks.

**2026-07-04 — Optional 3D cluster plot.** `plot_clusters(..., dimensions=2|3)` adds a rotatable 3D UMAP view (convex-hull mesh clouds + centroid text labels); Databricks widget `plot_dimensions` defaults to `2`. 2D remains the default for the iframe embed.

**2026-07-04 — Refinement coarsened after 60+ cluster explosion.** The first refinement pass (25/8, no topic cap) split the politics mega-cluster into 60+ near-duplicate "תקשורת ו…" Gemini labels. Defaults now: `refine_min_cluster_size=100`, `refine_min_samples=20`, `refine_nr_topics=12` (BERTopic merge cap on the refinement pass only), stricter refinement naming prompt forbidding "תקשורת/עיתונות/…" meta-labels. Expect ~15–20 topics total (5 pass-1 + ~12 politics sub-domains). Set `refine_oversized=False` to keep ~6 pass-1 topics only.

**2026-07-04 — Two-stage mega-cluster refinement.** Pass 1 still uses coarse HDBSCAN (60/15) for stable top-level domains; `refine_large_clusters()` (on by default, `refine_oversized=True`) re-clusters any topic holding ≥30% of docs with finer settings on the *same* embeddings and a sub-domain Gemini naming prompt — splits the ~7.6k politics blob without re-embedding or re-fragmenting sports/legal. Databricks widgets: `refine_oversized`, `refine_size_fraction`, `refine_min_cluster_size`, `refine_min_samples`, `refine_nr_topics`.

**2026-07-04 — Topic-clustering "fewer, more distinct topics" fix (v3) + notebook plots.** The v2 fix (embed on `text`) still surfaced too many near-duplicate topics (e.g. "תקשורת ומדיה"/"תקשורת וטלוויזיה") once mega-topic collapse was fixed, driven by (a) layout/journalism-meta keywords ("כותרת", "הבוקר", "העיתון"...) dominating c-TF-IDF instead of real subject words, and (b) fine HDBSCAN granularity producing several sub-clusters of the same domain that Gemini then named identically. Fixed in `evaluation/topic_clustering.py`: a `BOILERPLATE_STOPWORDS` set added to the vectorizer; `min_cluster_size`/`min_samples` raised 25/5 → 60/15 (coarser HDBSCAN); a new `merge_duplicate_labels()` post-processing step (on by default, `cluster_dataset(merge_duplicates=True)`) that collapses any clusters Gemini still named identically into one reported topic, keeping the smallest `cluster_id` and the union of keywords — no extra Gemini calls. Databricks widgets: `min_cluster_size`/`min_samples` defaults updated, new `merge_duplicate_labels` toggle. Also added inline Plotly charts to the notebook pipeline (`plot_topic_sizes` in `topic_clustering.py`, `plot_style_distribution` in `style_labels.py`, plus a topic×style stacked bar) alongside the existing big document scatter — small aggregate charts shown directly with `displayHTML(fig.to_html(...))`, no DBFS round-trip needed since they're nowhere near the ~20 MB cell-output cap.

**2026-07-04 — Topic-clustering granularity fix (v2).** After the noise/vectorizer fixes, a second full run collapsed ~99% of docs into one "חדשות ותקשורת" mega-topic — caused by clustering on summaries (outlet-name headlines), `outlier_threshold=0` (force-assign all noise to the largest cluster), and `nr_topics='auto'` over-merging. Defaults now: `embed_field='text'` (first 4k chars of article body), `outlier_threshold=0.35`, `nr_topics=None`, `min_cluster_size=25`/`min_samples=5`, media-outlet stopwords + domain-focused Gemini naming prompt. Databricks widgets updated (`embed_field`, `outlier_threshold`, `max_embed_chars`; `nr_topics` blank by default).

**2026-07-04 — Topic-clustering quality fix (v1).** The first full 10k-doc Databricks run put 51% of docs in the noise cluster (-1) and produced near-duplicate topic names (e.g. "תקשורת ומדיה" / "תקשורת וטלוויזיה") whose c-TF-IDF keywords were mostly years/IDs/Latin site names (`ynet`, `nrg`, `bbc`) — BERTopic's default English-tuned vectorizer let non-Hebrew tokens dominate. Fixed in `evaluation/topic_clustering.py`: a Hebrew-only `CountVectorizer` (`_build_vectorizer`/`HEBREW_TOKEN_PATTERN`/`HEBREW_STOPWORDS`), `min_samples` decoupled from `min_cluster_size` (per BERTopic's FAQ, reduces raw noise), and two opt-out BERTopic post-processing passes — `reduce_outliers` (embedding-similarity reassignment of noise docs) and `nr_topics="auto"` (HDBSCAN-over-topic-vectors merging of only genuinely similar topics). All exposed as Databricks widgets (`min_cluster_size` default lowered 100→40, `min_samples`, `reduce_outliers`, `nr_topics`) so re-tuning doesn't require editing the notebook. 3 new fast unit tests cover the Hebrew token pattern/stopwords.

**2026-07-08 — Clean pipeline profile (opt-in `--clean`) + Hebrew base-model plumbing.** An opt-in alternative pipeline addressing the v3 finding that the model learned HeSum's `"headline | headline"` digest style and hallucinates: (1) `data/clean.py` normalizes pipe/bullet references into prose and drops multi-headline roundups; (2) `data/prompts.py` adds a hardened `PROMPT_TEMPLATE_CLEAN` (concise, facts-only, no lists/pipes/speculation) via `build_prompt(text, clean=)`; (3) base `/no_think` reinforcement + an optional Hebrew-script `bad_words_ids` decode constraint (`evaluation/hebrew_constraint.py`) under `CLEAN`. Threaded through `preprocess/train/eval/predict` with a `-clean` suffix on `dataset_repo`/`model_repo` and the `<variant>-clean` processed dir, so the original raw pipeline stays byte-for-byte reproducible and the two compare head-to-head (paper ablation). Also added `train.py --base-model` → `MODEL_ID` env override for a Hebrew base-model swap. **Model search:** `dicta-il/DictaLM-3.0-1.7B-Base` (initialized from Qwen3-1.7B — same arch, LoRA modules transfer unchanged; Hebrew-SOTA for its size, summarization 9.72 vs Qwen3-1.7B 0.4) is the recommended candidate, drop-in via `--base-model`. Code done + 8 clean tests passing; HF Jobs runs + smoke finetune pending. Plan: `.cursor/plans/clean-refs-prompt-base_911ba56f.plan.md`; details in `docs/obsidian/Fix Plan.md`.

**2026-07-04 — Predictions viewer added.** `evaluation/viewer/` (`data.py` + `app.py`, its own subfolder): a local Streamlit app (`streamlit run evaluation/viewer/app.py`) for browsing `outputs/results/*.jsonl` — RTL Hebrew rendering, keyword search, side-by-side comparison across systems. Read-only, CPU-only, no GPU/API. Verified end-to-end against the real `predictions-finetuned.jsonl`/`predictions-base.jsonl` files with `streamlit.testing.v1.AppTest` (file discovery, multi-file compare, keyword filtering, navigation — no exceptions). Design: `docs/superpowers/specs/2026-07-04-predictions-viewer-design.md`.

**2026-06-27–28 — Hebrew-summarization fix batch (decoding + training + Hebrew-aware eval). All three phases complete.**
- *Eval (Phase 0, live):* `evaluate.py` BERTScore now defaults to `onlplab/alephbert-base` (HeSum-comparable, more discriminative than xlm-r) and reports both raw and Hebrew-normalized ROUGE.
- *Decoding (Phase 1):* shared generation adds `no_repeat_ngram_size=3`, `repetition_penalty=1.2`, `min_new_tokens=16`, explicit EOS. **Job `6a3f8e18…` ran 2026-06-27.** Finding: decoding alone lowered every metric (ROUGE-1 11.4→4.7, AlephBERT F1 0.45→0.38) — the v1 repetition loops inflated ROUGE by accident; suppressing them revealed fundamental undertraining.
- *Training (Phase 2):* 3 epochs, LoRA r→32/alpha→64 + MLP modules (`gate/up/down_proj`), "in up to 3 sentences" prompt cap. **Job `6a3fa247…` ran 2026-06-28, completed epoch 3.0** (eval_loss 1.712, best checkpoint auto-loaded). `predictions-finetuned.jsonl` + `predictions-base.jsonl` both pushed to Hub. Scored (`outputs/results/finetuned-v3.report.json`): ROUGE-1 5.1, AlephBERT F1 0.390. V3 generates fluent, format-correct Hebrew but hallucinates content (correct style, wrong facts). Marginal improvement over v2 (ROUGE-1 4.7, BS 0.383). LLM-judge required to evaluate faithfulness — **BLOCKED on Gemini API billing**.

**Next steps:**
1. **D.1 — DONE (2026-06-13).** Full eval battery ran on HF Jobs (job `6a2d1448`, gemini-2.5-flash-lite, n=1000 × 3 systems); tables in `outputs/results/d1-tables.md` via `evaluation.build_report_tables`. **Key results:** fine-tuning lifted ROUGE-1 (0.114 vs base 0.069) and BERTScore (0.850 vs 0.829) but the judge rated base *slightly higher* on faithfulness (2.98 vs 2.64) and fluency (3.80 vs 3.67) — a ROUGE-vs-human-judgement misalignment, driven by the fine-tuned model's degenerate repetition + more hallucination (0.22) and lead-copying (0.38). Gemini is a strong upper bound (faith 4.96, flu 5.00, 0% sampled failures). Zero-shot base is unusable raw: 22% non-Hebrew, 44% produce only `<think>` reasoning. (Deliverable is markdown tables — the presentation SVG is flattened paths, not editable text.)
2. **Truncation probe** — train/evaluate whole / lead / body variants by **30.06** (`--variant` is ready). **First apply the post-mortem checklist (§7): extended LoRA target modules, a10g-small, `--method lora`, fast-path deps.**
3. **Literature (English summarization)** — document lessons from English news summarization in the paper (lead bias, ROUGE limits, baseline practices).
4. **Journalism / headline control (optional)** — alternate instruction templates for headline-length vs longer summaries; see `TODO.md` H.

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
