# Training Pipeline Design — Stage A

**Date:** 2026-05-26  
**Project:** AMLK — Hebrew Text Summarization  
**Scope:** Stage A (A.1 Dataset download, A.2 Model prep, A.3 Fine-tuning)

---

## Overview

A modular, idempotent pipeline for fine-tuning `Qwen/Qwen3-2B` on Hebrew summarization. Three scripts (data download, preprocessing, training) run sequentially, each reading from and writing to disk. Multiple training scripts allow comparing fine-tuning strategies. Evaluation scripts are scaffolded for Stage B.

---

## 1. Directory Structure

```
/AMLK
├── data/
│   ├── download.py              # Downloads + merges IAHLT and HeSum datasets
│   └── preprocess.py            # Tokenizes + formats + splits into train/val/test
├── training/
│   ├── train_qlora.py           # QLoRA 4-bit — local, ~8 GB VRAM
│   ├── train_lora.py            # LoRA bf16 — local, ~16 GB VRAM
│   └── train_full.py            # Full fine-tuning — submits HF training job
├── evaluation/
│   ├── eval_rouge.py            # ROUGE-1/2/L evaluation
│   ├── eval_bertscore.py        # BERTScore with Hebrew model
│   └── eval_llm.py              # Gemini-as-judge (faithfulness + fluency)
└── outputs/
    ├── data/                    # Raw + processed datasets (gitignored)
    │   ├── raw/combined.jsonl
    │   └── processed/           # Arrow dataset splits
    ├── checkpoints/             # LoRA adapters / checkpoints (gitignored)
    └── results/                 # Evaluation JSONs (gitignored)
```

---

## 2. Data Handling

### Sources
| Dataset | Access | Text field | Summary field |
|---------|--------|-----------|---------------|
| IAHLT summarization_he | GitHub JSONL download | `text_raw` | `summary` |
| HeSum (biunlp/HeSum) | `load_dataset('biunlp/HeSum')` | `article` | `summary` |

### `data/download.py`
- Downloads IAHLT JSONL files from GitHub via `requests`
- Loads HeSum via `datasets.load_dataset('biunlp/HeSum')`
- Normalizes schema: both become `{text, summary, source}`
- Concatenates into single dataset
- Saves to `outputs/data/raw/combined.jsonl`
- Idempotent: skips download if output file exists

### `data/preprocess.py`
- Loads `combined.jsonl`
- Formats each example as:
  ```
  Summarize the following Hebrew text:
  {text}

  Summary:
  {summary}
  ```
- Tokenizes with `Qwen/Qwen3-2B` tokenizer, max 2048 tokens
- Splits 80% train / 10% val / 10% test (stratified by `source`)
- Saves Arrow dataset to `outputs/data/processed/`
- Idempotent: skips if output directory exists

---

## 3. Training Scripts

### Shared interface
All training scripts accept:
```
python training/train_*.py \
  --data outputs/data/processed/ \
  --output outputs/checkpoints/<run-name>
```
Each script writes a `training_args.json` alongside the checkpoint for reproducibility. After training, each script runs inference on the test split and writes `predictions.jsonl` to the checkpoint directory.

### LoRA configuration (shared across QLoRA + LoRA)
- `r=16`, `lora_alpha=32`, `lora_dropout=0.05`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Trainer: `trl.SFTTrainer`

### `train_qlora.py` — QLoRA 4-bit
- Loads model in 4-bit via `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=bfloat16)`
- `per_device_train_batch_size=2`, `gradient_accumulation_steps=8` → effective batch 16
- VRAM requirement: ~8 GB — runs on consumer GPU or free Colab

### `train_lora.py` — LoRA bf16
- Loads model in `torch.bfloat16`, no quantization
- `per_device_train_batch_size=4`, `gradient_accumulation_steps=4`
- VRAM requirement: ~16 GB — runs on mid-range GPU

### `train_full.py` — Full fine-tuning via HF job
- Does not train locally
- Uploads processed data to a temporary private HF dataset repo
- Submits HF training job via `huggingface_hub` API with the training config
- Streams job logs to terminal until completion
- Downloads checkpoint on success
- Requires `HF_TOKEN` with write permissions

---

## 4. Evaluation Skeleton (Stage B structure)

Scripts are created with function signatures and docstrings — implementation happens in Stage B.

### Shared interface
```
python evaluation/eval_*.py \
  --predictions outputs/checkpoints/<run-name>/predictions.jsonl \
  --output outputs/results/<run-name>-<metric>.json
```

### `eval_rouge.py`
- Computes ROUGE-1, ROUGE-2, ROUGE-L using `rouge_score`

### `eval_bertscore.py`
- Computes BERTScore using a Hebrew-capable multilingual model (`xlm-roberta-large`)

### `eval_llm.py`
- Uses Gemini API (`GEMINI_API_KEY`) as judge
- Scores each prediction on faithfulness and fluency (1-5 scale)

---

## 5. Environment & Dependencies

```bash
pip install transformers trl peft datasets accelerate bitsandbytes \
            evaluate bert-score rouge-score google-generativeai requests
```

Environment variables (from `.env`):
- `HF_TOKEN` — HuggingFace access token
- `GEMINI_API_KEY` — Gemini API key for LLM evaluation

---

## 6. Execution Order

```bash
source .env
python data/download.py
python data/preprocess.py
python training/train_qlora.py --output outputs/checkpoints/run-01
# or
python training/train_lora.py  --output outputs/checkpoints/run-01-lora
# or
python training/train_full.py  --output outputs/checkpoints/run-01-full
```

---

## 7. What Is NOT in Stage A

- Evaluation script implementations (Stage B)
- Hyperparameter sweep / multiple seeds
- Experiment tracking (WandB / MLflow) — can be added later if needed
- Merging LoRA adapters into the base model (done in evaluation step)
