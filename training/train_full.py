"""
Pipeline step 3 of 3 (variant C): full fine-tuning of all model weights.
Without flags: trains locally (requires ~40 GB VRAM, bf16, no LoRA).
With --submit-hf: uploads processed data to HuggingFace Hub and submits a real
QLoRA training job via HuggingFace Jobs API (train_hf_job.py runs remotely).
Saves full model checkpoint and test predictions.

Run (local):       python training/train_full.py --output outputs/checkpoints/run-full-01
Run (HF job):      python training/train_full.py --submit-hf --hf-user <username>
Run (smoke test):  python training/train_full.py --submit-hf --hf-user <username> --smoke-test
Execution environment: high-VRAM local GPU, or HuggingFace Jobs GPU for remote runs.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import datasets as hf_datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

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


def submit_hf_job(data_dir: Path, hf_token: str, hf_user: str, smoke_test: bool = False):
    """Upload processed data to HF Hub and submit a QLoRA training job via HF Jobs API."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    dataset_repo = f"{hf_user}/amlk-training-data"
    output_repo = f"{hf_user}/amlk-qwen3-2b-sft"

    # Step 1: Upload Arrow dataset to HF Hub (job downloads it with snapshot_download)
    print(f"Uploading dataset to {dataset_repo}...")
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=dataset_repo,
        repo_type="dataset",
    )
    print(f"  Dataset uploaded: https://huggingface.co/datasets/{dataset_repo}")

    # Step 2: Read training script and submit as inline UV job
    script_path = Path(__file__).parent / "train_hf_job.py"
    script = script_path.read_text()

    flavor = "a10g-small" if smoke_test else "a10g-large"
    timeout = "30m" if smoke_test else "4h"
    label = "smoke test" if smoke_test else "full training"

    print(f"\nSubmitting {label} job (hardware={flavor}, timeout={timeout})...")
    job = api.run_uv_job(
        script=script,
        flavor=flavor,
        timeout=timeout,
        secrets={"HF_TOKEN": "$HF_TOKEN"},
        env={
            "HF_USER": hf_user,
            "DATASET_REPO": dataset_repo,
            "OUTPUT_REPO": output_repo,
            "SMOKE_TEST": "1" if smoke_test else "0",
            "HF_HUB_DISABLE_EXPERIMENTAL_WARNING": "1",
        },
        token=hf_token,
    )

    print(f"\nJob submitted successfully!")
    print(f"  Job ID:  {job.id}")
    print(f"  Status:  {job.status.stage}")
    print(f"  Monitor: https://huggingface.co/jobs/{hf_user}/{job.id}")
    print(f"  Model:   https://huggingface.co/{output_repo}  (available after training)")
    print(f"\nCheck logs: source .venv/bin/activate && hf jobs logs {job.id}")
    return job


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
    parser.add_argument("--submit-hf", action="store_true", help="Upload data to HF Hub and submit a remote QLoRA training job")
    parser.add_argument("--hf-user", default="", help="HuggingFace username (required with --submit-hf)")
    parser.add_argument("--smoke-test", action="store_true", help="Submit a quick smoke-test job (50 examples, 10 steps, a10g-small)")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN not set. Run: source .env", file=sys.stderr)
        sys.exit(1)

    if args.submit_hf:
        if not args.hf_user:
            print("ERROR: --hf-user required with --submit-hf", file=sys.stderr)
            sys.exit(1)
        submit_hf_job(Path(args.data), hf_token, args.hf_user, smoke_test=args.smoke_test)
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
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="formatted",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
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
