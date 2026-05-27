## Project Goal

* **Description:** AMLK is a Hebrew text summarization research project. The goal is to fine-tune the `Qwen/Qwen3-2B` language model on Hebrew summarization datasets, evaluate it with ROUGE, BERTScore, and LLM-based metrics, and produce a research paper and presentation. The project runs locally for development and on HuggingFace infrastructure for training jobs when local compute is insufficient; all scripts are executed as command-line Python scripts.

---

## Project Structure - remember to update it when you make changes

* **Architecture:** The project is divided into three sequential pipelines:
  1. **Training pipeline** — downloads Hebrew summarization datasets (IAHLT summarization_he, HeSum), loads the `Qwen/Qwen3-2B` base model, and fine-tunes it using the HuggingFace `transformers`/`trl` stack. If local GPU is insufficient, the job is submitted to HuggingFace as a remote training job.
  2. **Evaluation pipeline** — takes the fine-tuned checkpoint and runs it against a held-out test set, computing ROUGE scores, BERTScore, and an LLM-as-judge evaluation (via the Gemini API).
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
├── .claude/
│   └── skills/
│       └── coding-principles/SKILL.md    # Project-local coding standards
├── data/
│   ├── __init__.py
│   ├── download.py                       # Pipeline step 1: downloads & normalizes IAHLT+HeSum datasets
│   └── preprocess.py                     # Pipeline step 2: formats instruction pairs, splits 80/10/10, saves Arrow datasets
├── training/
│   ├── __init__.py
│   ├── config.py                         # Shared LoRAConfig, TrainingConfig, MODEL_ID, RESPONSE_TEMPLATE
│   ├── train_qlora.py                    # QLoRA 4-bit fine-tuning (~8 GB VRAM)
│   ├── train_lora.py                     # LoRA bf16 fine-tuning (~16 GB VRAM)
│   ├── train_full.py                     # Full fine-tuning locally, or --submit-hf to submit HF Jobs job
│   └── train_hf_job.py                   # Self-contained UV script run by HF Jobs (submitted by train_full.py --submit-hf)
├── evaluation/
│   ├── __init__.py
│   ├── eval_rouge.py                     # ROUGE-1/2/L scorer (Stage B — NotImplementedError stub)
│   ├── eval_bertscore.py                 # BERTScore scorer (Stage B — NotImplementedError stub)
│   └── eval_llm.py                       # Gemini LLM-as-judge scorer (Stage B — NotImplementedError stub)
├── tests/
│   ├── __init__.py
│   ├── test_download.py                  # Unit tests for normalize_iahlt / normalize_hesum
│   └── test_preprocess.py               # Unit tests for format_instruction / split_dataset
├── docs/
│   └── superpowers/
│       ├── specs/2026-05-26-training-pipeline-design.md
│       └── plans/2026-05-26-stage-a-training-pipeline.md
├── outputs/
│   ├── data/
│   │   ├── raw/combined.jsonl            # Merged normalized dataset — 10,000 records (gitignored)
│   │   └── processed/                   # Arrow dataset splits: train/ val/ test/ (gitignored)
│   ├── checkpoints/                     # LoRA adapter / full model checkpoints (gitignored)
│   └── results/                         # Evaluation JSON reports (gitignored)
├── .venv/                               # Python virtual environment (gitignored)
├── .env                                 # HF_TOKEN, GEMINI_API_KEY — never commit
├── .gitignore
├── AGENTS.md
├── CLAUDE.md                            # Claude Code guidance
├── README.md
├── requirements.txt
└── TODO.md                              # Milestone tracker
```

* `data/download.py`: Downloads Hebrew summarization datasets (biunlp/HeSum; IAHLT/summarization_he inaccessible with current credentials), normalises to `{text, summary, source}` schema, writes to `outputs/data/raw/combined.jsonl`.
* `data/preprocess.py`: Reads `combined.jsonl`, formats each example as a Hebrew summarization instruction pair (adds `formatted` column), splits 80/10/10, saves Arrow dataset splits. Does NOT tokenize — SFTTrainer handles that.
* `training/config.py`: Shared constants and dataclasses used by all three training scripts: `MODEL_ID="Qwen/Qwen3-2B"`, `LoRAConfig` (r=16, alpha=32), `TrainingConfig`.
* `training/train_qlora.py`: QLoRA 4-bit fine-tuning using `BitsAndBytesConfig` + LoRA. Saves adapter checkpoint, `training_args.json`, and `predictions.jsonl`.
* `training/train_lora.py`: LoRA bf16 fine-tuning (no quantization). Same outputs as QLoRA variant.
* `training/train_full.py`: Full fine-tuning of all weights. `--submit-hf --hf-user <name>` uploads dataset to HF Hub and submits a real QLoRA training job via `HfApi.run_uv_job()`. Add `--smoke-test` for a quick 10-step verification run.
* `training/train_hf_job.py`: Self-contained PEP 723 UV script submitted inline by `train_full.py --submit-hf`. Downloads dataset from HF Hub, trains Qwen3-2B with QLoRA on the cloud GPU, and pushes the LoRA adapter to Hub. Never run directly.
* `evaluation/eval_*.py`: Stage B stubs — correct function signatures and CLI, body raises `NotImplementedError`.
* `tests/test_download.py` / `tests/test_preprocess.py`: 9 unit tests, all passing.

---

## Building and Running

**Prerequisites:**
* Python 3.10+
* `uv` package manager (used instead of pip — `uv` is on PATH)
* Install dependencies: `uv pip install -r requirements.txt` (or `uv sync` if using a lockfile)
* Fill in `.env`:
  * `HF_TOKEN` — HuggingFace access token (needed for model download and HF Hub upload)
  * `GEMINI_API_KEY` — Gemini API key (needed for LLM-based evaluation in Stage B)
* Source the env and activate venv before running scripts: `source .env && source .venv/bin/activate`

**Running the pipeline (Stage A):**
```bash
source .env && source .venv/bin/activate

# Step 1: Download datasets
python data/download.py

# Step 2: Preprocess (format + split)
python data/preprocess.py

# Step 3a: Fine-tune with QLoRA (~8 GB VRAM)
python training/train_qlora.py --output outputs/checkpoints/run-qlora-01

# Step 3b: Fine-tune with LoRA bf16 (~16 GB VRAM)
python training/train_lora.py --output outputs/checkpoints/run-lora-01

# Step 3c: Full fine-tune locally (~40 GB VRAM)
python training/train_full.py --output outputs/checkpoints/run-full-01

# Step 3c (alternative): Submit QLoRA training job to HuggingFace Jobs
python training/train_full.py --submit-hf --hf-user <your-hf-username>

# Quick smoke-test (10 steps, a10g-small, ~$0.05) to verify the pipeline end-to-end
python training/train_full.py --submit-hf --hf-user <your-hf-username> --smoke-test
```

**Running tests:**
```bash
source .venv/bin/activate && python -m pytest tests/ -v
```

---

## Status - remember to update it

**Stage A complete as of 2026-05-27.** HF Jobs submission verified 2026-05-27.
- `data/download.py` — 10,000 records from biunlp/HeSum in `outputs/data/raw/combined.jsonl`. IAHLT/summarization_he is inaccessible (not on HF Hub with current credentials).
- `data/preprocess.py` — 8,000 train / 1,000 val / 1,000 test splits in `outputs/data/processed/`.
- `training/train_qlora.py`, `train_lora.py`, `train_full.py` — three fine-tuning variants implemented.
- `training/train_hf_job.py` — QLoRA remote training script for HF Jobs infrastructure.
- `evaluation/eval_*.py` — Stage B stubs in place.
- 9 unit tests, all passing.
- HF Jobs dataset: `avreymi/amlk-training-data` (private). Model output: `avreymi/amlk-qwen3-2b-sft`.
- Known limitation: QLoRA `push_to_hub` saves the LoRA adapter only (not merged). Evaluation scripts must load base + adapter together via `PeftModel.from_pretrained`.
- Known limitation: `DataCollatorForCompletionOnlyLM` was removed in trl 1.5.0; training scripts use full-sequence loss. Can be improved with `SFTConfig(completion_only_loss=True)` + prompt/completion column split.

**Next steps:** Stage B (evaluation pipeline) — implement `eval_rouge.py`, `eval_bertscore.py`, `eval_llm.py`. Presentation deadline: 2026-06-14. Final submission: 2026-07-31.

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
