## Project Goal

* **Description:** AMLK is a Hebrew **news** summarization research project. The goal is to fine-tune `Qwen/Qwen3-2B` on Hebrew journalism datasets (HeSum, IAHLT summarization_he), evaluate with ROUGE, BERTScore, and LLM-as-judge, and produce a research paper and presentation. Design choices are informed by **English summarization literature** (lead bias, metric limits, strong baselines) without re-running English experiments. Evaluation includes an **advanced-model baseline** (e.g. Gemini API on the same test set and prompt) so metrics can be interpreted against a stronger system. A **truncation / positional-shortcut probe** trains separate models on Whole text, Lead-only, and Body-only inputs. Optional **headline-style control** varies the instruction (short headline vs longer summary). **Error analysis** labels a sampled set of predictions for failure types common in the literature. Runs locally or on HuggingFace Jobs; all scripts are command-line Python.

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
│   └── preprocess.py                     # Pipeline step 2: prompt/completion pairs + probe variants, 80/10/10 split
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
│   └── infer.py                          # GPU inference helpers (load adapter + generate); used by the observation notebook
├── notebooks/
│   └── evaluation_observation.ipynb      # evaluation-observation stage: live per-example view (summary/judge/errors) on Colab
├── scripts/
│   ├── __init__.py
│   └── run_nb_cell.py                    # Drive notebook cells on a Colab session via colab-cli (agent cell-by-cell runner)
├── tests/
│   ├── __init__.py
│   ├── test_download.py                  # normalize_iahlt / normalize_hesum
│   ├── test_preprocess.py                # build_prompt / make_variant / split_dataset
│   └── test_evaluation.py                # ROUGE-on-Hebrew, judge-reply parsing, failure rates (live test gated)
├── docs/
│   ├── ANLP Project abstract.md          # The research proposal this project implements
│   ├── 2026-06-12-qlora-training-job-postmortem.md  # Full-run post-mortem: cost, root cause, probe-run recommendations
│   └── superpowers/
│       ├── specs/2026-05-26-training-pipeline-design.md
│       └── plans/2026-05-26-stage-a-training-pipeline.md
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
* `data/preprocess.py`: Reads `combined.jsonl`, builds `(prompt, completion)` pairs for completion-only SFT, applies the `--variant whole|lead|body` truncation probe (`make_variant`), truncates each article to `MAX_LENGTH-256` tokens so the summary always survives (HeSum articles are long — median ~2500 tokens; without this, completion-only loss goes nan), splits 80/10/10, saves Arrow splits to `outputs/data/processed/<variant>/`. `build_prompt`/`make_variant` are the single source of truth, reused by `evaluation/predict.py`.
* `training/config.py`: Shared constants: `MODEL_ID="Qwen/Qwen3-2B"`, `METHOD_PRESETS` (the qlora/lora/full deltas), `LoRAConfig` (r=16, alpha=32), `TrainingConfig`, `WANDB_PROJECT`, and `dataset_repo`/`model_repo` Hub-id helpers.
* `training/train.py`: One trainer for all three regimes (`--method qlora|lora|full`). Trains with `completion_only_loss=True`, logs to wandb, saves the adapter; `--push-to-hub` or `--submit-hf` push to the Hub. Inference is NOT here — that's `evaluation/predict.py`.
* `training/train_hf_job.py`: Self-contained PEP 723 UV script submitted inline by `train.py --submit-hf`. Reads METHOD/VARIANT/DATASET_REPO/OUTPUT_REPO/WANDB_PROJECT from env, trains on the cloud GPU, then generates fine-tuned + zero-shot base test predictions (PEFT `disable_adapter`) and pushes the adapter + `predictions-finetuned.jsonl` / `predictions-base.jsonl` to the Hub. Never run directly.
* `evaluation/predict.py`: Generates the Gemini advanced-baseline summaries via API (no GPU, no model load), same prompt as training. Resumes from a partial file. The fine-tuned and zero-shot predictions come from the cloud training job, not here. Also defines `strip_think()` — the shared tool that drops closed Qwen3 `<think>…</think>` reasoning so metrics score the summary, not the reasoning (used by evaluate.py and error_analysis.py).
* `evaluation/evaluate.py`: Scores a predictions file with ROUGE-1/2/L, BERTScore (xlm-roberta-large), and the Gemini faithfulness/fluency judge (`--skip-llm` to skip; `--limit N` to cap for a smoke run). Applies `strip_think` before scoring. One JSON report per system.
* `evaluation/error_analysis.py`: Samples ~50 predictions (post `strip_think`) and has Gemini label failure types (hallucination, omission, entity/number error, lead copying, fluency), writing per-type rates.
* `evaluation/eval_hf_job.py`: Runs the whole D1 battery on HuggingFace Jobs so the ~4000 Gemini calls + BERTScore happen on the cloud's fast connection (the user has weak internet). One file, two modes: `--submit-hf` uploads itself to a cheap CPU job; with no args (how HF Jobs invokes it) it fetches the public repo + Hub predictions/dataset and drives the existing `evaluation/` CLIs by subprocess, pushing each report to the model repo under `reports/` as it finishes (timeout-safe).
* `evaluation/build_report_tables.py`: Downloads the pushed `reports/*.json` and assembles the D1 markdown — a quality table (ROUGE/BERTScore/judge), a failure-rate table, and behavioural notes (base `<think>`/language leakage, fine-tuned repetition, judge self-preference caveat).
* `evaluation/infer.py`: GPU inference helpers — `load_finetuned_model` (base + LoRA adapter, `disable_adapter()` gives the zero-shot base) and `generate_summaries` (batched greedy decode over a processed split). The importable twin of `train_hf_job.py`'s inline generation block (that cloud script can't import repo code); keep the two in sync. **Remote GPU only — never call locally.**
* `notebooks/evaluation_observation.ipynb`: The **evaluation-observation** stage. A self-bootstrapping Colab notebook that runs the *real* evaluation functions live and **displays** the per-example process (article → model summary → reference → judge faithfulness/fluency → error-analysis failure labels) for finetuned/base/gemini. Loads existing Hub predictions (finetuned/base at repo root, gemini under `reports/`) and generates fresh summaries on a T4. Judge/error/browse cells are API+CPU; only the generation cell needs a GPU.
* `scripts/run_nb_cell.py`: Agent cell-runner — reads the notebook with `nbformat` and execs a chosen code cell / range against a persistent Colab session via `colab exec` (the Colab CLI has no native `.ipynb` runner). `--list` shows cell indices; the caller owns `colab new`/`stop`. This is how an agent observes the eval cell-by-cell.
* `tests/`: 16 fast behavioral tests + 1 gated live Gemini test, all passing.

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

**Running tests:**
```bash
source .venv/bin/activate && python -m pytest tests/ -v
```

---

## Status - remember to update it

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
- **KNOWN FLAW in the trained adapter: Qwen3-2B is a hybrid-attention model (18 linear-attention + 6 full-attention layers); LoRA `target_modules` q/k/v/o only exist in the 6 full-attention layers, so the adapter covers 6/24 layers (0.07% trainable params).** Extend `target_modules` (see post-mortem §5.1) and validate with a mini-test before the truncation-probe runs. Also add `flash-linear-attention` + `causal-conv1d` to the job deps (slow-kernel fallback warning).
- `train_hf_job.py` pushes each predictions file immediately after its generation loop (timeout-safe), prints progress every 10 batches, generates with `max_new_tokens=256` (p99 reference length is 187 tokens); inference-only jobs use a 1h timeout.

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
