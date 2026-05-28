# Stage A — Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, modular pipeline to download Hebrew summarization datasets, format them, and fine-tune Qwen/Qwen3-2B with QLoRA, LoRA, or full fine-tuning — with evaluation script skeletons for Stage B.

**Architecture:** Three sequential idempotent scripts (download → preprocess → train). `preprocess.py` formats data but does NOT tokenize (SFTTrainer handles tokenization). Three training scripts share `config.py` and differ only in quantization/LoRA strategy. Evaluation scripts are scaffolded with `raise NotImplementedError` for Stage B.

**Tech Stack:** Python 3.10+, `datasets`, `transformers`, `trl>=0.12`, `peft`, `bitsandbytes`, `accelerate`, `rouge-score`, `bert-score`, `google-generativeai`, `requests`, `torch`

---

## File Map

| File | Purpose |
|------|---------|
| `requirements.txt` | All Python dependencies |
| `data/download.py` | Download IAHLT + HeSum, normalize to `{text, summary, source}`, save raw JSONL |
| `data/preprocess.py` | Format as instruction pairs, split 80/10/10, save Arrow dataset (no tokenization) |
| `training/config.py` | Shared `LoRAConfig` and `TrainingConfig` dataclasses |
| `training/train_qlora.py` | QLoRA 4-bit fine-tuning (~8 GB VRAM) |
| `training/train_lora.py` | LoRA bf16 fine-tuning (~16 GB VRAM) |
| `training/train_full.py` | Full fine-tuning locally, or `--submit-hf` to upload + print Space instructions |
| `evaluation/eval_rouge.py` | ROUGE skeleton (Stage B) |
| `evaluation/eval_bertscore.py` | BERTScore skeleton (Stage B) |
| `evaluation/eval_llm.py` | LLM-as-judge skeleton (Stage B) |
| `tests/test_download.py` | Unit tests for IAHLT + HeSum normalization functions |
| `tests/test_preprocess.py` | Unit tests for instruction formatting and dataset splitting |

---

### Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `data/__init__.py`, `training/__init__.py`, `evaluation/__init__.py`, `tests/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create directories and package files**

```bash
mkdir -p data training evaluation outputs/data/raw outputs/data/processed outputs/checkpoints outputs/results tests
touch data/__init__.py training/__init__.py evaluation/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create requirements.txt**

```
transformers>=4.45.0
trl>=0.12.0
peft>=0.13.0
datasets>=3.0.0
accelerate>=1.0.0
bitsandbytes>=0.44.0
evaluate>=0.4.0
bert-score>=0.3.13
rouge-score>=0.1.2
google-generativeai>=0.8.3
requests>=2.32.0
torch>=2.4.0
```

- [ ] **Step 3: Add outputs/ to .gitignore**

Add to `.gitignore`:
```
outputs/
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt data/__init__.py training/__init__.py evaluation/__init__.py tests/__init__.py .gitignore
git commit -m "feat: scaffold Stage A project structure and dependencies"
```

---

### Task 2: data/download.py — normalize + download

**Files:**
- Create: `data/download.py`
- Create: `tests/test_download.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_download.py`:
```python
"""Tests for data/download.py normalization functions."""
from data.download import normalize_iahlt, normalize_hesum


def test_normalize_iahlt_happy_path():
    record = {
        "text_raw": "כותרת המאמר וכל התוכן שלו בעברית",
        "summary": "סיכום קצר",
        "metadata": {"source": "haaretz", "doc_id": "123"},
    }
    result = normalize_iahlt(record)
    assert result == {
        "text": "כותרת המאמר וכל התוכן שלו בעברית",
        "summary": "סיכום קצר",
        "source": "iahlt",
    }


def test_normalize_iahlt_skips_empty_text():
    assert normalize_iahlt({"text_raw": "", "summary": "סיכום", "metadata": {}}) is None


def test_normalize_iahlt_skips_empty_summary():
    assert normalize_iahlt({"text_raw": "טקסט", "summary": "", "metadata": {}}) is None


def test_normalize_hesum_happy_path():
    record = {"article": "תוכן המאמר המלא", "summary": "הכותרת"}
    result = normalize_hesum(record)
    assert result == {"text": "תוכן המאמר המלא", "summary": "הכותרת", "source": "hesum"}


def test_normalize_hesum_skips_empty_article():
    assert normalize_hesum({"article": "", "summary": "סיכום"}) is None
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_download.py -v
```

Expected: `ImportError: cannot import name 'normalize_iahlt' from 'data.download'`

- [ ] **Step 3: Implement data/download.py**

Create `data/download.py`:
```python
"""
Pipeline step 1 of 3: dataset acquisition.
Downloads Hebrew summarization datasets from HuggingFace Hub (IAHLT/summarization_he
and biunlp/HeSum), normalises their schemas to {text, summary, source}, and writes
the merged dataset to outputs/data/raw/combined.jsonl.

Run: python data/download.py
Execution environment: local development machine with HF_TOKEN in environment.
"""
import json
import os
import sys
from pathlib import Path

import datasets


OUTPUT_PATH = Path("outputs/data/raw/combined.jsonl")


def normalize_iahlt(record: dict) -> dict | None:
    """Return {text, summary, source='iahlt'} from an IAHLT JSONL record, or None to skip."""
    text = record.get("text_raw", "").strip()
    summary = record.get("summary", "").strip()
    if not text or not summary:
        return None
    return {"text": text, "summary": summary, "source": "iahlt"}


def normalize_hesum(record: dict) -> dict | None:
    """Return {text, summary, source='hesum'} from a HeSum HF row, or None to skip."""
    text = record.get("article", "").strip()
    summary = record.get("summary", "").strip()
    if not text or not summary:
        return None
    return {"text": text, "summary": summary, "source": "hesum"}


def _load_iahlt(hf_token: str) -> list[dict]:
    print("Loading IAHLT/summarization_he from HuggingFace Hub...")
    ds = datasets.load_dataset("IAHLT/summarization_he", token=hf_token)
    records = []
    for split in ds.values():
        for row in split:
            norm = normalize_iahlt(row)
            if norm:
                records.append(norm)
    print(f"  IAHLT: {len(records)} usable records")
    return records


def _load_hesum() -> list[dict]:
    print("Loading biunlp/HeSum from HuggingFace Hub...")
    ds = datasets.load_dataset("biunlp/HeSum")
    records = []
    for split in ds.values():
        for row in split:
            norm = normalize_hesum(row)
            if norm:
                records.append(norm)
    print(f"  HeSum: {len(records)} usable records")
    return records


def main():
    if OUTPUT_PATH.exists():
        print(f"Output already exists at {OUTPUT_PATH}. Delete it to re-download.")
        sys.exit(0)

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN not set. Run: source .env", file=sys.stderr)
        sys.exit(1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = _load_iahlt(hf_token) + _load_hesum()
    print(f"Total: {len(records)} records")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_download.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Run the download script**

```bash
source .env && python data/download.py
```

Expected: prints record counts for IAHLT and HeSum, writes `outputs/data/raw/combined.jsonl`.

- [ ] **Step 6: Commit**

```bash
git add data/download.py tests/test_download.py
git commit -m "feat: add data/download.py with IAHLT+HeSum normalization"
```

---

### Task 3: data/preprocess.py — format + split

**Files:**
- Create: `data/preprocess.py`
- Create: `tests/test_preprocess.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_preprocess.py`:
```python
"""Tests for data/preprocess.py formatting and splitting functions."""
import datasets as hf_datasets
from data.preprocess import format_instruction, split_dataset


def test_format_instruction_contains_text_and_summary():
    result = format_instruction("המאמר הגדול", "סיכום קצר")
    assert "המאמר הגדול" in result
    assert "סיכום קצר" in result


def test_format_instruction_contains_prompt_and_label():
    result = format_instruction("א", "ב")
    assert "Summarize" in result
    assert "Summary" in result


def test_split_dataset_ratios():
    data = hf_datasets.Dataset.from_dict({
        "text": [f"text {i}" for i in range(1000)],
        "summary": [f"summary {i}" for i in range(1000)],
        "source": ["iahlt"] * 500 + ["hesum"] * 500,
    })
    train, val, test = split_dataset(data, seed=42)
    assert len(train) == 800
    assert len(val) == 100
    assert len(test) == 100


def test_split_dataset_no_overlap():
    data = hf_datasets.Dataset.from_dict({
        "text": [f"text {i}" for i in range(100)],
        "summary": [f"summary {i}" for i in range(100)],
        "source": ["iahlt"] * 100,
    })
    train, val, test = split_dataset(data, seed=42)
    train_set = set(train["text"])
    val_set = set(val["text"])
    test_set = set(test["text"])
    assert not train_set & val_set
    assert not train_set & test_set
    assert not val_set & test_set
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_preprocess.py -v
```

Expected: `ImportError: cannot import name 'format_instruction' from 'data.preprocess'`

- [ ] **Step 3: Implement data/preprocess.py**

Create `data/preprocess.py`:
```python
"""
Pipeline step 2 of 3: instruction formatting and dataset splitting.
Reads outputs/data/raw/combined.jsonl, formats each example as a Hebrew
summarization instruction pair (adding a 'formatted' column), splits 80/10/10,
and saves Arrow dataset splits to outputs/data/processed/.
Does NOT tokenize — tokenization is handled by SFTTrainer at training time.

Run: python data/preprocess.py
Execution environment: local development machine.
"""
import json
import sys
from pathlib import Path

import datasets as hf_datasets


INPUT_PATH = Path("outputs/data/raw/combined.jsonl")
OUTPUT_DIR = Path("outputs/data/processed")


def format_instruction(text: str, summary: str) -> str:
    """Format a text+summary pair as a causal LM instruction for Hebrew summarization."""
    return (
        f"Summarize the following Hebrew text:\n\n"
        f"{text}\n\n"
        f"Summary:\n{summary}"
    )


def split_dataset(
    dataset: hf_datasets.Dataset,
    seed: int = 42,
) -> tuple[hf_datasets.Dataset, hf_datasets.Dataset, hf_datasets.Dataset]:
    """Split dataset 80% train / 10% val / 10% test. Returns (train, val, test)."""
    split = dataset.train_test_split(test_size=0.2, seed=seed)
    val_test = split["test"].train_test_split(test_size=0.5, seed=seed)
    return split["train"], val_test["train"], val_test["test"]


def main():
    if OUTPUT_DIR.exists():
        print(f"Output already exists at {OUTPUT_DIR}. Delete it to re-preprocess.")
        sys.exit(0)

    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run data/download.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {INPUT_PATH}...")
    records = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} records")

    dataset = hf_datasets.Dataset.from_dict({
        "text": [r["text"] for r in records],
        "summary": [r["summary"] for r in records],
        "source": [r["source"] for r in records],
        "formatted": [format_instruction(r["text"], r["summary"]) for r in records],
    })

    print("Splitting 80/10/10...")
    train, val, test = split_dataset(dataset)
    print(f"  train: {len(train)}, val: {len(val)}, test: {len(test)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train.save_to_disk(str(OUTPUT_DIR / "train"))
    val.save_to_disk(str(OUTPUT_DIR / "val"))
    test.save_to_disk(str(OUTPUT_DIR / "test"))
    print(f"Saved splits to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
python -m pytest tests/test_preprocess.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run the preprocess script**

```bash
python data/preprocess.py
```

Expected: prints split sizes, creates `outputs/data/processed/train/`, `val/`, `test/`.

- [ ] **Step 6: Commit**

```bash
git add data/preprocess.py tests/test_preprocess.py
git commit -m "feat: add data/preprocess.py with instruction formatting and 80/10/10 split"
```

---

### Task 4: training/config.py — shared LoRA + training config

**Files:**
- Create: `training/config.py`

- [ ] **Step 1: Implement training/config.py**

Create `training/config.py`:
```python
"""
Shared configuration for all fine-tuning scripts (train_qlora.py, train_lora.py,
train_full.py). Defines model ID, LoRA hyperparameters, and training settings.
Changing values here affects all three training approaches uniformly.

Execution environment: imported by training scripts on local machine or remote GPU.
"""
from dataclasses import dataclass, field


MODEL_ID = "Qwen/Qwen3-2B"
DATA_DIR = "outputs/data/processed"
MAX_SEQ_LENGTH = 2048
RESPONSE_TEMPLATE = "\nSummary:\n"  # used by DataCollatorForCompletionOnlyLM


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    bf16: bool = True
    report_to: str = "none"
```

- [ ] **Step 2: Verify import**

```bash
python -c "from training.config import LoRAConfig, TrainingConfig, MODEL_ID; print(LoRAConfig()); print(TrainingConfig())"
```

Expected: prints both dataclass instances with default values.

- [ ] **Step 3: Commit**

```bash
git add training/config.py
git commit -m "feat: add training/config.py with shared LoRA and training hyperparameters"
```

---

### Task 5: training/train_qlora.py — QLoRA 4-bit fine-tuning

**Files:**
- Create: `training/train_qlora.py`

- [ ] **Step 1: Implement training/train_qlora.py**

Create `training/train_qlora.py`:
```python
"""
Pipeline step 3 of 3 (variant A): QLoRA 4-bit fine-tuning.
Loads Qwen/Qwen3-2B in 4-bit quantization, attaches LoRA adapters, and fine-tunes
on preprocessed Hebrew summarization data. Loss is computed only on summary tokens
(completion-only SFT). Saves the LoRA adapter checkpoint and test predictions.

Run: python training/train_qlora.py --output outputs/checkpoints/run-qlora-01
Execution environment: local GPU with CUDA, minimum ~8 GB VRAM.
"""
import argparse
import json
import os
from pathlib import Path

import datasets as hf_datasets
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from training.config import (
    DATA_DIR,
    LORA_CONFIG,
    MAX_SEQ_LENGTH,
    MODEL_ID,
    RESPONSE_TEMPLATE,
    LoRAConfig,
    TrainingConfig,
)


def build_model_and_tokenizer(hf_token: str):
    """Load Qwen3-2B in 4-bit NF4 quantization with LoRA adapters attached."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token or None,
    )
    model.config.use_cache = False

    lora_cfg = LoRAConfig()
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )
    return get_peft_model(model, peft_config), tokenizer


def run_inference(model, tokenizer, test_dataset, output_dir: Path, max_new_tokens: int = 128):
    """Generate summaries for the test split and write predictions.jsonl."""
    model.eval()
    predictions = []
    for example in test_dataset:
        prompt = (
            f"Summarize the following Hebrew text:\n\n{example['text']}\n\nSummary:\n"
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH - max_new_tokens,
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        predictions.append({
            "text": example["text"],
            "reference": example["summary"],
            "prediction": generated.strip(),
        })

    pred_path = output_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(predictions)} predictions to {pred_path}")


def main():
    parser = argparse.ArgumentParser(description="QLoRA 4-bit fine-tuning for Hebrew summarization")
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=-1, help="Set to a small number for smoke tests")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN", "")
    data_dir = Path(args.data)

    print("Loading data...")
    train_ds = hf_datasets.load_from_disk(str(data_dir / "train"))
    val_ds = hf_datasets.load_from_disk(str(data_dir / "val"))
    test_ds = hf_datasets.load_from_disk(str(data_dir / "test"))

    print("Loading model (QLoRA 4-bit)...")
    model, tokenizer = build_model_and_tokenizer(hf_token)
    model.print_trainable_parameters()

    collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tokenizer)

    train_cfg = TrainingConfig()
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        eval_steps=train_cfg.eval_steps,
        bf16=train_cfg.bf16,
        report_to=train_cfg.report_to,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="formatted",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    print("Starting QLoRA training...")
    trainer.train()
    trainer.save_model(str(output_dir))

    with open(output_dir / "training_args.json", "w") as f:
        json.dump({
            "model_id": MODEL_ID,
            "method": "qlora_4bit",
            "lora": vars(LoRAConfig()),
            "training": vars(train_cfg),
        }, f, indent=2)

    print("Running inference on test set...")
    run_inference(model, tokenizer, test_ds, output_dir)


if __name__ == "__main__":
    main()
```

**Note:** `training/config.py` does not export `LORA_CONFIG` — remove that import line. Only `LoRAConfig`, `TrainingConfig`, `MODEL_ID`, `DATA_DIR`, `MAX_SEQ_LENGTH`, and `RESPONSE_TEMPLATE` are exported.

- [ ] **Step 2: Fix the import line**

The import in `train_qlora.py` Step 1 above erroneously includes `LORA_CONFIG`. The correct import block is:
```python
from training.config import (
    DATA_DIR,
    MAX_SEQ_LENGTH,
    MODEL_ID,
    RESPONSE_TEMPLATE,
    LoRAConfig,
    TrainingConfig,
)
```

- [ ] **Step 3: Verify import (no GPU needed)**

```bash
python -c "from training.train_qlora import build_model_and_tokenizer, run_inference; print('Import OK')"
```

Expected: `Import OK`

- [ ] **Step 4: Smoke test (requires CUDA GPU, ~8 GB VRAM)**

```bash
source .env && python training/train_qlora.py \
  --data outputs/data/processed/ \
  --output outputs/checkpoints/smoke-qlora \
  --max-steps 3
```

Expected: 3 training steps complete, `outputs/checkpoints/smoke-qlora/` contains adapter weights, `training_args.json`, and `predictions.jsonl`.

- [ ] **Step 5: Commit**

```bash
git add training/train_qlora.py
git commit -m "feat: add train_qlora.py — QLoRA 4-bit fine-tuning for Hebrew summarization"
```

---

### Task 6: training/train_lora.py — LoRA bf16 fine-tuning

**Files:**
- Create: `training/train_lora.py`

- [ ] **Step 1: Implement training/train_lora.py**

Create `training/train_lora.py`:
```python
"""
Pipeline step 3 of 3 (variant B): LoRA bf16 fine-tuning.
Loads Qwen/Qwen3-2B in bfloat16 (no quantization), attaches LoRA adapters,
and fine-tunes on preprocessed Hebrew summarization data. Faster convergence
than QLoRA but requires more VRAM. Saves adapter checkpoint and test predictions.

Run: python training/train_lora.py --output outputs/checkpoints/run-lora-01
Execution environment: local GPU with CUDA, minimum ~16 GB VRAM.
"""
import argparse
import json
import os
from pathlib import Path

import datasets as hf_datasets
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from training.config import (
    DATA_DIR,
    MAX_SEQ_LENGTH,
    MODEL_ID,
    RESPONSE_TEMPLATE,
    LoRAConfig,
    TrainingConfig,
)


def build_model_and_tokenizer(hf_token: str):
    """Load Qwen3-2B in bfloat16 with LoRA adapters (no quantization)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token or None,
    )
    model.config.use_cache = False

    lora_cfg = LoRAConfig()
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=lora_cfg.target_modules,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )
    return get_peft_model(model, peft_config), tokenizer


def run_inference(model, tokenizer, test_dataset, output_dir: Path, max_new_tokens: int = 128):
    """Generate summaries for the test split and write predictions.jsonl."""
    model.eval()
    predictions = []
    for example in test_dataset:
        prompt = (
            f"Summarize the following Hebrew text:\n\n{example['text']}\n\nSummary:\n"
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH - max_new_tokens,
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        predictions.append({
            "text": example["text"],
            "reference": example["summary"],
            "prediction": generated.strip(),
        })

    pred_path = output_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(predictions)} predictions to {pred_path}")


def main():
    parser = argparse.ArgumentParser(description="LoRA bf16 fine-tuning for Hebrew summarization")
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=-1, help="Set to a small number for smoke tests")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_token = os.environ.get("HF_TOKEN", "")
    data_dir = Path(args.data)

    print("Loading data...")
    train_ds = hf_datasets.load_from_disk(str(data_dir / "train"))
    val_ds = hf_datasets.load_from_disk(str(data_dir / "val"))
    test_ds = hf_datasets.load_from_disk(str(data_dir / "test"))

    print("Loading model (LoRA bf16)...")
    model, tokenizer = build_model_and_tokenizer(hf_token)
    model.print_trainable_parameters()

    collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tokenizer)

    train_cfg = TrainingConfig()
    train_cfg.per_device_train_batch_size = 4
    train_cfg.gradient_accumulation_steps = 4

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        eval_steps=train_cfg.eval_steps,
        bf16=train_cfg.bf16,
        report_to=train_cfg.report_to,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="formatted",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    print("Starting LoRA bf16 training...")
    trainer.train()
    trainer.save_model(str(output_dir))

    with open(output_dir / "training_args.json", "w") as f:
        json.dump({
            "model_id": MODEL_ID,
            "method": "lora_bf16",
            "lora": vars(LoRAConfig()),
            "training": vars(train_cfg),
        }, f, indent=2)

    print("Running inference on test set...")
    run_inference(model, tokenizer, test_ds, output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import**

```bash
python -c "from training.train_lora import build_model_and_tokenizer, run_inference; print('Import OK')"
```

Expected: `Import OK`

- [ ] **Step 3: Smoke test (requires ~16 GB VRAM)**

```bash
source .env && python training/train_lora.py \
  --data outputs/data/processed/ \
  --output outputs/checkpoints/smoke-lora \
  --max-steps 3
```

Expected: 3 steps complete, checkpoint + `predictions.jsonl` in `outputs/checkpoints/smoke-lora/`.

- [ ] **Step 4: Commit**

```bash
git add training/train_lora.py
git commit -m "feat: add train_lora.py — LoRA bf16 fine-tuning for Hebrew summarization"
```

---

### Task 7: training/train_full.py — full fine-tuning or HF job submission

**Files:**
- Create: `training/train_full.py`

- [ ] **Step 1: Implement training/train_full.py**

Create `training/train_full.py`:
```python
"""
Pipeline step 3 of 3 (variant C): full fine-tuning of all model weights.
Without flags: trains locally (requires ~40 GB VRAM, bf16, no LoRA).
With --submit-hf: uploads processed data to HuggingFace Hub and prints
step-by-step instructions to run training on a HF Space with A100/H100 GPU.
Saves full model checkpoint and test predictions.

Run (local):  python training/train_full.py --output outputs/checkpoints/run-full-01
Run (HF job): python training/train_full.py --submit-hf --hf-user <username>
Execution environment: high-VRAM local GPU, or HuggingFace Space for remote runs.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import datasets as hf_datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from training.config import (
    DATA_DIR,
    MAX_SEQ_LENGTH,
    MODEL_ID,
    RESPONSE_TEMPLATE,
    TrainingConfig,
)


def build_model_and_tokenizer(hf_token: str):
    """Load Qwen3-2B in bfloat16, all weights unfrozen (no LoRA, no quantization)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token or None,
    )
    return model, tokenizer


def submit_hf_job(data_dir: Path, hf_token: str, hf_user: str):
    """Upload processed data to HF Hub and print Space training instructions."""
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    dataset_repo = f"{hf_user}/amlk-training-data"

    print(f"Uploading dataset to {dataset_repo}...")
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=dataset_repo,
        repo_type="dataset",
    )
    print(f"\nDataset uploaded to: https://huggingface.co/datasets/{dataset_repo}")
    print("\nTo run full fine-tuning on HuggingFace:")
    print("  1. Create a Space at https://huggingface.co/new-space with A100 GPU runtime")
    print("  2. Clone this repo into the Space")
    print("  3. In the Space terminal:")
    print("     pip install -r requirements.txt")
    print(f"     huggingface-cli download {dataset_repo} --local-dir outputs/data/processed/ --repo-type dataset")
    print("     python training/train_full.py --output outputs/checkpoints/run-full-hf")


def run_inference(model, tokenizer, test_dataset, output_dir: Path, max_new_tokens: int = 128):
    """Generate summaries for the test split and write predictions.jsonl."""
    model.eval()
    predictions = []
    for example in test_dataset:
        prompt = (
            f"Summarize the following Hebrew text:\n\n{example['text']}\n\nSummary:\n"
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH - max_new_tokens,
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        predictions.append({
            "text": example["text"],
            "reference": example["summary"],
            "prediction": generated.strip(),
        })

    pred_path = output_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Saved {len(predictions)} predictions to {pred_path}")


def main():
    parser = argparse.ArgumentParser(description="Full fine-tuning for Hebrew summarization")
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--output", default="outputs/checkpoints/run-full")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--submit-hf", action="store_true", help="Upload data to HF Hub and print Space instructions")
    parser.add_argument("--hf-user", default="", help="HuggingFace username (required with --submit-hf)")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN not set. Run: source .env", file=sys.stderr)
        sys.exit(1)

    if args.submit_hf:
        if not args.hf_user:
            print("ERROR: --hf-user required with --submit-hf", file=sys.stderr)
            sys.exit(1)
        submit_hf_job(Path(args.data), hf_token, args.hf_user)
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data)

    print("Loading data...")
    train_ds = hf_datasets.load_from_disk(str(data_dir / "train"))
    val_ds = hf_datasets.load_from_disk(str(data_dir / "val"))
    test_ds = hf_datasets.load_from_disk(str(data_dir / "test"))

    print("Loading model (full bf16, no LoRA)...")
    model, tokenizer = build_model_and_tokenizer(hf_token)

    collator = DataCollatorForCompletionOnlyLM(RESPONSE_TEMPLATE, tokenizer=tokenizer)

    train_cfg = TrainingConfig()
    train_cfg.per_device_train_batch_size = 1
    train_cfg.gradient_accumulation_steps = 16
    train_cfg.learning_rate = 5e-5

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        eval_steps=train_cfg.eval_steps,
        bf16=train_cfg.bf16,
        report_to=train_cfg.report_to,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="formatted",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    print("Starting full fine-tuning...")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    with open(output_dir / "training_args.json", "w") as f:
        json.dump({
            "model_id": MODEL_ID,
            "method": "full_finetune",
            "training": vars(train_cfg),
        }, f, indent=2, default=str)

    print("Running inference on test set...")
    run_inference(model, tokenizer, test_ds, output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import**

```bash
python -c "from training.train_full import submit_hf_job, run_inference, build_model_and_tokenizer; print('Import OK')"
```

Expected: `Import OK`

- [ ] **Step 3: Test HF submission path (no GPU, no data needed)**

```bash
source .env && python training/train_full.py --submit-hf --hf-user testuser 2>&1 | head -5
```

Expected: error about `outputs/data/processed` not existing (tries to upload), or upload proceeds if data is ready. No training starts.

- [ ] **Step 4: Commit**

```bash
git add training/train_full.py
git commit -m "feat: add train_full.py — full fine-tuning with optional HF job submission"
```

---

### Task 8: Evaluation skeletons — Stage B scaffolding

**Files:**
- Create: `evaluation/eval_rouge.py`
- Create: `evaluation/eval_bertscore.py`
- Create: `evaluation/eval_llm.py`

- [ ] **Step 1: Create evaluation/eval_rouge.py**

Create `evaluation/eval_rouge.py`:
```python
"""
Evaluation pipeline step 1: ROUGE-1/2/L scoring.
Reads predictions.jsonl from a training checkpoint directory, computes ROUGE
scores against human reference summaries, and writes a JSON report.

Run: python evaluation/eval_rouge.py --predictions outputs/checkpoints/<run>/predictions.jsonl --output outputs/results/<run>-rouge.json
Execution environment: local development machine. Implementation in Stage B.
"""
import argparse
import json
from pathlib import Path


def compute_rouge(predictions: list[dict]) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L for a list of {prediction, reference} dicts.
    Returns {rouge1: {precision, recall, fmeasure}, rouge2: {...}, rougeL: {...}}.
    Stage B: implement using rouge_score.rouge_scorer.RougeScorer.
    """
    raise NotImplementedError("eval_rouge.compute_rouge is implemented in Stage B")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f]

    scores = compute_rouge(predictions)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"ROUGE scores saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create evaluation/eval_bertscore.py**

Create `evaluation/eval_bertscore.py`:
```python
"""
Evaluation pipeline step 2: BERTScore with a multilingual Hebrew-capable model.
Reads predictions.jsonl, computes BERTScore precision/recall/F1 using
xlm-roberta-large, and writes a JSON report.

Run: python evaluation/eval_bertscore.py --predictions outputs/checkpoints/<run>/predictions.jsonl --output outputs/results/<run>-bertscore.json
Execution environment: local development machine. Implementation in Stage B.
"""
import argparse
import json
from pathlib import Path


def compute_bertscore(predictions: list[dict], model_id: str = "xlm-roberta-large") -> dict:
    """
    Compute BERTScore for a list of {prediction, reference} dicts using model_id.
    Returns {precision: float, recall: float, f1: float} (averages over all examples).
    Stage B: implement using bert_score.score().
    """
    raise NotImplementedError("eval_bertscore.compute_bertscore is implemented in Stage B")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="xlm-roberta-large")
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f]

    scores = compute_bertscore(predictions, model_id=args.model)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"BERTScore results saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create evaluation/eval_llm.py**

Create `evaluation/eval_llm.py`:
```python
"""
Evaluation pipeline step 3: LLM-as-judge scoring via Gemini API.
Reads predictions.jsonl, calls Gemini to score each prediction on faithfulness
and fluency (1–5 scale), and writes a JSON report with per-example scores and averages.

Run: python evaluation/eval_llm.py --predictions outputs/checkpoints/<run>/predictions.jsonl --output outputs/results/<run>-llm.json
Execution environment: local machine with GEMINI_API_KEY set. Implementation in Stage B.
"""
import argparse
import json
import os
from pathlib import Path


def score_with_llm(prediction: dict, client) -> dict:
    """
    Score one {text, reference, prediction} entry using Gemini as judge.
    Returns {faithfulness: int, fluency: int} (both 1–5).
    Stage B: implement prompt + Gemini API call.
    """
    raise NotImplementedError("eval_llm.score_with_llm is implemented in Stage B")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        raise EnvironmentError("GEMINI_API_KEY not set. Run: source .env")

    with open(args.predictions, encoding="utf-8") as f:
        predictions = [json.loads(line) for line in f]

    # Stage B: initialize Gemini client, call score_with_llm for each prediction
    raise NotImplementedError("eval_llm.main is implemented in Stage B")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify all eval signatures import correctly**

```bash
python -c "
from evaluation.eval_rouge import compute_rouge
from evaluation.eval_bertscore import compute_bertscore
from evaluation.eval_llm import score_with_llm
print('All eval signatures OK')
"
```

Expected: `All eval signatures OK`

- [ ] **Step 5: Commit**

```bash
git add evaluation/eval_rouge.py evaluation/eval_bertscore.py evaluation/eval_llm.py
git commit -m "feat: add evaluation script skeletons for Stage B (ROUGE, BERTScore, LLM judge)"
```

---

## Self-Review

**Spec coverage check:**
- A.1 Download datasets (IAHLT + HeSum) → Task 2 ✓
- A.2 Download pretrained model (Qwen3-2B) → handled automatically inside Task 5/6/7 training scripts ✓
- A.3 Fine-tune with QLoRA → Task 5 ✓; LoRA → Task 6 ✓; Full/HF job → Task 7 ✓
- Multiple training approaches → Tasks 5, 6, 7 ✓
- Evaluation skeleton → Task 8 ✓
- 80/10/10 split → Task 3 ✓
- Merged IAHLT + HeSum → Task 2 ✓
- `predictions.jsonl` output from training → all train scripts ✓
- `training_args.json` for reproducibility → Tasks 5, 6, 7 ✓

**Placeholder scan:** No TBDs or incomplete steps. Eval files use `raise NotImplementedError` explicitly to mark Stage B work — this is intentional, not accidental omission.

**Type consistency:**
- `normalize_iahlt` / `normalize_hesum` → return `dict | None` — consistent across Task 2 tests and implementation ✓
- `format_instruction(text, summary) -> str` — consistent across Task 3 tests and implementation ✓
- `split_dataset(dataset, seed) -> tuple[Dataset, Dataset, Dataset]` — consistent ✓
- `run_inference(model, tokenizer, test_dataset, output_dir, max_new_tokens)` — same signature in Tasks 5, 6, 7 ✓
- `predictions.jsonl` schema: `{text, reference, prediction}` — consistent across all train scripts and eval skeletons ✓
