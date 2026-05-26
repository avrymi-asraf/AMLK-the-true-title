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
