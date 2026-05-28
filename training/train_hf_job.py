#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.45.0",
#     "accelerate>=1.0.0",
#     "bitsandbytes>=0.44.0",
#     "datasets>=3.0.0",
#     "huggingface_hub",
#     "trackio",
# ]
# ///
"""
Self-contained UV script for QLoRA fine-tuning of Qwen/Qwen3-2B on Hebrew summarization,
designed to run on HuggingFace Jobs infrastructure. This is pipeline step 3 (remote variant).

Submitted by training/train_full.py --submit-hf; not run directly by the user.
Reads HF_USER / DATASET_REPO / OUTPUT_REPO from environment variables injected by the job.
Output: LoRA adapter pushed to HF Hub at OUTPUT_REPO (base model + adapter, not merged).
Execution environment: HuggingFace Jobs GPU (a10g-large or similar), ephemeral container.
"""

import os
import warnings
from pathlib import Path

import torch
import trackio
from datasets import load_from_disk
from huggingface_hub import snapshot_download
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

warnings.filterwarnings("ignore", category=UserWarning)

HF_USER = os.environ["HF_USER"]
DATASET_REPO = os.environ.get("DATASET_REPO", f"{HF_USER}/amlk-training-data")
OUTPUT_REPO = os.environ.get("OUTPUT_REPO", f"{HF_USER}/amlk-qwen3-2b-sft")
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
MODEL_ID = "Qwen/Qwen3-2B"

print(f"Dataset:    {DATASET_REPO}")
print(f"Output:     {OUTPUT_REPO}")
print(f"Smoke test: {SMOKE_TEST}")

# Download Arrow dataset from Hub (uploaded by train_full.py --submit-hf)
local_data = Path("./data")
print(f"\nDownloading dataset from {DATASET_REPO}...")
snapshot_download(repo_id=DATASET_REPO, repo_type="dataset", local_dir=str(local_data))
train_ds = load_from_disk(str(local_data / "train"))
val_ds = load_from_disk(str(local_data / "val"))
print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

if SMOKE_TEST:
    train_ds = train_ds.select(range(50))
    val_ds = val_ds.select(range(20))
    print(f"  [Smoke] Truncated to train={len(train_ds)}, val={len(val_ds)}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    output_dir="./output",
    push_to_hub=True,
    hub_model_id=OUTPUT_REPO,
    hub_strategy="every_save",
    hub_private_repo=True,
    num_train_epochs=1 if SMOKE_TEST else 3,
    max_steps=10 if SMOKE_TEST else -1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=5 if SMOKE_TEST else 10,
    save_strategy="steps",
    save_steps=5 if SMOKE_TEST else 200,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=5 if SMOKE_TEST else 200,
    bf16=True,
    gradient_checkpointing=True,
    report_to="trackio",
    project="amlk-hebrew-summarization",
    run_name="qwen3-2b-qlora-smoke" if SMOKE_TEST else "qwen3-2b-qlora",
    dataset_text_field="formatted",
    max_length=2048,
)

trainer = SFTTrainer(
    model=MODEL_ID,
    model_init_kwargs={"quantization_config": bnb_config, "device_map": "auto"},
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=sft_config,
    peft_config=peft_config,
)

print("\nStarting training...")
trainer.train()
trainer.push_to_hub()
trackio.finish()

print(f"\nDone. Adapter saved to: https://huggingface.co/{OUTPUT_REPO}")
