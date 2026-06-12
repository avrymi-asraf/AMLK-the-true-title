#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=1.6.0",
#     "peft>=0.17.0",
#     "transformers>=5.0.0",
#     "accelerate>=1.0.0",
#     "bitsandbytes>=0.44.0",
#     "datasets>=3.0.0",
#     "huggingface_hub",
#     "wandb",
# ]
# ///
"""
Pipeline step 3 (remote variant): self-contained QLoRA fine-tuning of Qwen/Qwen3-2B,
run on HuggingFace Jobs. Submitted inline by training/train.py --submit-hf; never run
directly. All settings arrive as environment variables (the repo is NOT uploaded with
the script). Downloads the processed splits from the Hub, trains with completion_only_loss,
logs to Weights & Biases, and pushes the trained adapter back to the Hub.

INFERENCE_ONLY=1 skips training and loads an already-pushed adapter from OUTPUT_REPO —
use this to re-generate predictions when the training job ran out of time.

Execution environment: ephemeral HuggingFace Jobs GPU container (a10g-large by default;
a10g-small is sufficient for inference-only runs).
"""
import json
import os
import warnings
from pathlib import Path

import torch
import wandb
from datasets import load_from_disk
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

METHOD = os.environ.get("METHOD", "qlora")
VARIANT = os.environ.get("VARIANT", "whole")
DATASET_REPO = os.environ["DATASET_REPO"]
OUTPUT_REPO = os.environ["OUTPUT_REPO"]
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
MINI_TEST = os.environ.get("MINI_TEST", "0") == "1"
INFERENCE_ONLY = os.environ.get("INFERENCE_ONLY", "0") == "1"
MODEL_ID = "Qwen/Qwen3-2B"
os.environ.setdefault("WANDB_PROJECT", os.environ.get("WANDB_PROJECT", "amlk-hebrew-summarization"))

mode = "inference-only" if INFERENCE_ONLY else f"train+infer  Method={METHOD}"
print(f"Mode: {mode}  Variant: {VARIANT}  Smoke: {SMOKE_TEST}  Mini: {MINI_TEST}")
print(f"Dataset: {DATASET_REPO}  ->  Output: {OUTPUT_REPO}")

local_data = Path("./data")
snapshot_download(repo_id=DATASET_REPO, repo_type="dataset", local_dir=str(local_data))
train_ds = load_from_disk(str(local_data / "train"))
val_ds = load_from_disk(str(local_data / "val"))
test_ds = load_from_disk(str(local_data / "test"))
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

if SMOKE_TEST:
    train_ds = train_ds.select(range(min(50, len(train_ds))))
    val_ds = val_ds.select(range(min(20, len(val_ds))))
    test_ds = test_ds.select(range(min(5, len(test_ds))))
    print(f"[Smoke] Truncated to train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
elif MINI_TEST:
    # 80 train + 20 val gives ~25 optimizer steps over 5 epochs (batch=2, grad_accum=8) —
    # enough to show a real loss curve in wandb without burning full-run budget.
    train_ds = train_ds.select(range(min(80, len(train_ds))))
    val_ds = val_ds.select(range(min(20, len(val_ds))))
    test_ds = test_ds.select(range(min(10, len(test_ds))))
    print(f"[Mini] Truncated to train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
else:
    # Eval on a fixed 200-example slice — full 1000-example eval several times would blow the budget.
    val_ds = val_ds.select(range(min(200, len(val_ds))))

quantize = METHOD == "qlora"
use_lora = METHOD != "full"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

load_kwargs = dict(device_map="auto")
if quantize:
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
else:
    load_kwargs["torch_dtype"] = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
model.config.use_cache = False

if INFERENCE_ONLY:
    print(f"Loading adapter from {OUTPUT_REPO}...")
    trained_model = PeftModel.from_pretrained(model, OUTPUT_REPO).eval()
    device = next(trained_model.parameters()).device
    use_lora = True  # adapter is always a LoRA adapter
else:
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    ) if use_lora else None

    if SMOKE_TEST:
        run_suffix = "-smoke"
    elif MINI_TEST:
        run_suffix = "-mini"
    else:
        run_suffix = ""
    run_name = f"{METHOD}-{VARIANT}-hfjob{run_suffix}"

    # Mini: 5 epochs over 80 examples → ~25 optimizer steps (batch=2, grad_accum=8);
    # log every step and eval every 5 to get ≥5 eval points visible in wandb.
    if MINI_TEST:
        n_epochs, max_steps_cfg = 5, -1
        log_steps, eval_steps_cfg, save_steps_cfg = 1, 5, 20
    elif SMOKE_TEST:
        n_epochs, max_steps_cfg = 1, 10
        log_steps, eval_steps_cfg, save_steps_cfg = 5, 5, 5
    else:
        n_epochs, max_steps_cfg = 1, -1
        log_steps, eval_steps_cfg, save_steps_cfg = 10, 200, 200

    sft_config = SFTConfig(
        output_dir="./output",
        push_to_hub=True,
        hub_model_id=OUTPUT_REPO,
        hub_strategy="every_save",
        hub_private_repo=True,
        num_train_epochs=n_epochs,
        max_steps=max_steps_cfg,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,   # eval defaults to 8 → OOM at seq-len 2048; keep it small
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=log_steps,
        save_strategy="steps",
        save_steps=save_steps_cfg,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=eval_steps_cfg,
        bf16=True,
        gradient_checkpointing=True,
        completion_only_loss=True,
        max_length=2048,
        report_to="wandb",
        run_name=run_name,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting training...")
    trainer.train()
    trainer.push_to_hub()

    trained_model = trainer.model.eval()
    device = next(trained_model.parameters()).device

# Restore KV cache for faster generation (disabled during gradient checkpointing).
trained_model.config.use_cache = True


def generate_predictions(label: str) -> list[dict]:
    """Generate test-set summaries in batches of 8 with progress logging.

    Left-padding ensures all sequences in a batch are right-aligned so that
    the slice `out[:, input_len:]` consistently extracts only generated tokens.
    """
    tokenizer.padding_side = "left"
    rows = []
    batch_size = 8
    for i in range(0, len(test_ds), batch_size):
        batch = test_ds[i:i + batch_size]
        prompts: list[str] = batch["prompt"]
        inputs = tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=2048 - 128, padding=True,
        ).to(device)
        with torch.no_grad():
            outs = trained_model.generate(
                # 256 covers p99 of reference-summary lengths (p95=151, p99=187 tokens);
                # the old cap of 128 truncated ~9% of summaries mid-sentence.
                **inputs, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for j, prompt in enumerate(prompts):
            pred = tokenizer.decode(outs[j][input_len:], skip_special_tokens=True)
            rows.append({"text": batch["text"][j], "reference": batch["summary"][j],
                         "prediction": pred.strip(), "model": label, "variant": VARIANT})
        end = min(i + batch_size, len(test_ds))
        if (i // batch_size) % 10 == 0 or end == len(test_ds):
            print(f"  [{label}] {end}/{len(test_ds)}")
    tokenizer.padding_side = "right"
    return rows


api = HfApi(token=os.environ.get("HF_TOKEN"))


def generate_and_push(label: str):
    """Generate one system's predictions and push the file immediately — a later
    timeout must never destroy finished work (see the 2026-06-12 post-mortem)."""
    rows = generate_predictions(label)
    path = Path(f"predictions-{label}.jsonl")
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                    repo_id=OUTPUT_REPO, repo_type="model")
    print(f"  Pushed {path.name} ({len(rows)} rows) to {OUTPUT_REPO}")


# Two systems from one loaded model: the fine-tuned adapter, and the zero-shot base
# (PEFT's disable_adapter() turns the adapter off without reloading the base model).
print("\nGenerating fine-tuned predictions...")
generate_and_push("finetuned")
if use_lora:
    print("Generating zero-shot base predictions...")
    with trained_model.disable_adapter():
        generate_and_push("base")

if not INFERENCE_ONLY:
    wandb.finish()
print(f"\nDone. Adapter + predictions at https://huggingface.co/{OUTPUT_REPO}")
