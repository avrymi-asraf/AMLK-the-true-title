"""
Pipeline step 3 of 3: fine-tune dicta-il/dictalm2.0-instruct on Hebrew summarization.
One entry point for all three regimes the paper compares — --method qlora | lora | full
— differing only by the small METHOD_PRESETS deltas in config.py. Trains with the trl
SFT trainer using completion_only_loss=True (loss on the summary only), logs every step
to Weights & Biases, and saves / optionally pushes the adapter (or full model) to the Hub.
Inference lives separately in evaluation/predict.py, so this script only trains.

Run (HF Jobs):   python -m training.train --submit-hf --hf-user avreymi [--method qlora] [--smoke-test]
Execution environment: HuggingFace Jobs GPU via --submit-hf (preferred); local CUDA only if available.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Heavy training deps (datasets/torch/peft/transformers/trl/wandb) are imported lazily inside
# the functions that need them, so `--submit-hf` can run on a minimal local env (no GPU stack)
# — submission only needs huggingface_hub.

from training.config import (
    DEFAULT_EPOCHS,
    DEFAULT_MAX_NEW_TOKENS,
    MAX_LENGTH,
    METHOD_PRESETS,
    MODEL_ID,
    MODEL_SLUG,
    PROCESSED_DIR,
    LoRAConfig,
    TrainingConfig,
    dataset_repo,
    model_repo,
    processed_profile_name,
    wandb_project,
    wandb_run_name,
)


def build_model_and_tokenizer(method: str, hf_token: str):
    """Load the base model for the chosen regime: 4-bit (qlora) or bf16 (lora, full)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from data.prompts import prepare_tokenizer_for_templated_prompts

    preset = METHOD_PRESETS[method]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepare_tokenizer_for_templated_prompts(tokenizer)

    load_kwargs = dict(token=hf_token or None, device_map="auto")
    if preset["quantize"]:
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
    return model, tokenizer


def _train_config_payload(method: str) -> dict:
    """Serialize METHOD_PRESETS + shared TrainingConfig for the remote UV job."""
    preset = METHOD_PRESETS[method]
    base = TrainingConfig()
    return {
        "quantize": preset["quantize"],
        "use_lora": preset["use_lora"],
        "per_device_train_batch_size": preset["per_device_train_batch_size"],
        "gradient_accumulation_steps": preset["gradient_accumulation_steps"],
        "learning_rate": preset["learning_rate"],
        "warmup_ratio": base.warmup_ratio,
        "lr_scheduler_type": base.lr_scheduler_type,
        "bf16": base.bf16,
        "max_length": MAX_LENGTH,
    }


def _lora_config_payload() -> dict:
    cfg = LoRAConfig()
    return {
        "r": cfg.r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "target_modules": list(cfg.target_modules),
        "bias": cfg.bias,
        "task_type": cfg.task_type,
    }


def lora_config():
    from peft import LoraConfig

    cfg = LoRAConfig()
    return LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias=cfg.bias,
        task_type=cfg.task_type,
    )


def wandb_api_key() -> str:
    """The wandb key, from $WANDB_API_KEY or the api.wandb.ai entry in ~/.netrc."""
    if os.environ.get("WANDB_API_KEY"):
        return os.environ["WANDB_API_KEY"]
    try:
        import netrc
        auth = netrc.netrc().authenticators("api.wandb.ai")
        return auth[2] if auth else ""
    except (FileNotFoundError, netrc.NetrcParseError):
        return ""


def submit_hf_job(method: str, variant: str, hf_token: str, hf_user: str,
                  smoke_test: bool, mini_test: bool = False, inference_only: bool = False,
                  pred_suffix: str = "", epochs: int = 0, base_model: str = "",
                  output_repo: str = "", skip_data_upload: bool = False,
                  timeout: str = "", max_new_tokens: int = 0):
    """Upload the processed splits to the Hub and submit train_hf_job.py to HF Jobs.

    inference_only=True skips dataset re-upload and training; loads the already-pushed
    adapter and regenerates predictions only (fast: a10g-small, 2h timeout). pred_suffix
    (e.g. "-v2") keeps a re-decode from clobbering the v1 predictions. epochs overrides the
    default (1). base_model swaps the base checkpoint (defaults to config.MODEL_ID);
    output_repo must then be set too. skip_data_upload reuses the splits already on the Hub.
    max_new_tokens overrides DEFAULT_MAX_NEW_TOKENS for dual-arm generation (cost lever).
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    data_repo = dataset_repo(hf_user, variant)
    out_repo = output_repo or model_repo(hf_user, variant)
    n_epochs = epochs or DEFAULT_EPOCHS
    base = base_model or MODEL_ID

    if not inference_only and not skip_data_upload:
        data_dir = Path(PROCESSED_DIR) / processed_profile_name(variant)
        if not data_dir.exists():
            print(f"ERROR: {data_dir} not found. Run: python -m data.preprocess --variant {variant}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Uploading {data_dir} to {data_repo}...")
        api.create_repo(repo_id=data_repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(data_dir), repo_id=data_repo, repo_type="dataset")

    # Ensure the model repo exists before the job starts so mid-run hub_strategy=every_save works.
    if not inference_only:
        api.create_repo(repo_id=out_repo, repo_type="model", private=True, exist_ok=True)

    script_path = Path(__file__).parent / "train_hf_job.py"
    if inference_only:
        flavor, label = "a10g-small", "infer"
        timeout = timeout or "2h"
    elif smoke_test:
        flavor, label = "a10g-small", "smoke"
        timeout = timeout or "30m"
    elif mini_test:
        flavor, label = "a10g-small", "mini"
        timeout = timeout or "1h"
    else:
        # 7B QLoRA ~5.8h worst-case at smoke step-time; 8h leaves headroom (C5).
        flavor, label = "a10g-small", ""
        timeout = timeout or "8h"
    wandb_key = wandb_api_key()
    project = wandb_project(MODEL_SLUG)
    run_name = wandb_run_name(
        method, variant, model_slug=MODEL_SLUG, epochs=n_epochs, tag=label or "",
    )
    train_cfg = _train_config_payload(method)
    lora_cfg = _lora_config_payload() if train_cfg["use_lora"] else {}
    n_new_tokens = max_new_tokens or DEFAULT_MAX_NEW_TOKENS

    tag = f"{label} " if label else ""
    print(f"Submitting {tag}{method} job (flavor={flavor}, timeout={timeout})...")
    print(f"  Base model: {base}")
    print(f"  Output repo: {out_repo}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Train config: batch={train_cfg['per_device_train_batch_size']} "
          f"accum={train_cfg['gradient_accumulation_steps']} lr={train_cfg['learning_rate']}")
    print(f"  max_new_tokens={n_new_tokens} (post-train dual-arm decode budget)")
    print(f"  wandb: {project} / {run_name}")
    print(f"  Stability: hub_strategy=every_save (checkpoint commits mid-run) + /data/output resume")
    job = api.run_uv_job(
        script=str(script_path),
        flavor=flavor,
        timeout=timeout,
        secrets={"HF_TOKEN": hf_token, "WANDB_API_KEY": wandb_key},
        env={
            "METHOD": method,
            "VARIANT": variant,
            "BASE_MODEL": base,
            "MODEL_SLUG": MODEL_SLUG,
            "DATASET_REPO": data_repo,
            "OUTPUT_REPO": out_repo,
            "WANDB_PROJECT": project,
            "WANDB_RUN_NAME": run_name,
            "SMOKE_TEST": "1" if smoke_test else "0",
            "MINI_TEST": "1" if mini_test else "0",
            "INFERENCE_ONLY": "1" if inference_only else "0",
            "PRED_SUFFIX": pred_suffix,
            "EPOCHS": str(n_epochs),
            # Decode budget for dual-arm generation (cost lever; was hardcoded 256).
            "MAX_NEW_TOKENS": str(n_new_tokens),
            # Resolved presets from config.py — never hardcode batch/lr in train_hf_job.
            "TRAIN_CONFIG": json.dumps(train_cfg),
            "LORA_CONFIG": json.dumps(lora_cfg),
        },
        token=hf_token,
    )
    print(f"\nJob submitted. ID: {job.id}  Status: {job.status.stage}")
    print(f"  Monitor: https://huggingface.co/jobs/{hf_user}/{job.id}")
    print(f"  Logs:    hf jobs logs {job.id}")
    print(f"  Model:   https://huggingface.co/{out_repo}  (after training)")
    return job


def train_local(method: str, variant: str, output_dir: Path, max_steps: int,
                max_length: int, batch_size: int, push_to_hub: bool, hf_user: str, hf_token: str,
                epochs: int = DEFAULT_EPOCHS):
    """Run the SFT loop locally and save the adapter / model to output_dir."""
    import datasets as hf_datasets
    import wandb
    from trl import SFTConfig, SFTTrainer

    preset = METHOD_PRESETS[method]
    base = TrainingConfig()
    per_device_batch = batch_size or preset["per_device_train_batch_size"]
    data_dir = Path(PROCESSED_DIR) / processed_profile_name(variant)
    n_epochs = epochs or DEFAULT_EPOCHS
    project = wandb_project(MODEL_SLUG)
    run_name = wandb_run_name(method, variant, model_slug=MODEL_SLUG, epochs=n_epochs)

    from data.prompts import format_chat_prompt

    print(f"Loading data from {data_dir}...")
    train_ds = hf_datasets.load_from_disk(str(data_dir / "train"))
    val_ds = hf_datasets.load_from_disk(str(data_dir / "val"))

    print(f"Loading model ({method})...")
    model, tokenizer = build_model_and_tokenizer(method, hf_token)

    def _wrap(example):
        example["prompt"] = format_chat_prompt(tokenizer, example["prompt"])
        return example

    print("Wrapping prompts in chat template (if the base has one)...")
    train_ds = train_ds.map(_wrap)
    val_ds = val_ds.map(_wrap)

    run = wandb.init(
        project=project,
        name=run_name,
        group=f"{MODEL_SLUG}-{variant}",
        tags=[method, variant, MODEL_SLUG, f"{n_epochs}ep"],
        config={"model_id": MODEL_ID, "method": method, "variant": variant,
                "max_length": max_length, "epochs": n_epochs, **preset},
    )
    run.define_metric("train/*", step_metric="step")
    run.define_metric("eval/*", step_metric="step")
    run.define_metric("eval/loss", summary="min")

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=n_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=preset["gradient_accumulation_steps"],
        learning_rate=preset["learning_rate"],
        warmup_ratio=base.warmup_ratio,
        lr_scheduler_type=base.lr_scheduler_type,
        logging_steps=base.logging_steps,
        save_strategy="steps",
        save_steps=base.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=base.eval_steps,
        bf16=base.bf16,
        gradient_checkpointing=True,
        completion_only_loss=True,
        max_length=max_length,
        report_to="wandb",
        run_name=run_name,
        push_to_hub=push_to_hub,
        hub_model_id=model_repo(hf_user, variant) if push_to_hub else None,
        hub_strategy="every_save" if push_to_hub else "end",
        hub_private_repo=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora_config() if preset["use_lora"] else None,
    )

    print(f"Starting {method} training (variant={variant}, epochs={n_epochs})...")
    if push_to_hub:
        print(f"  Mid-run Hub push: every save_steps={base.save_steps} → {model_repo(hf_user, variant)}")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    if push_to_hub:
        trainer.push_to_hub()

    (output_dir / "training_args.json").write_text(json.dumps(
        {"model_id": MODEL_ID, "method": method, "variant": variant, "epochs": n_epochs,
         "preset": preset},
        indent=2,
    ))
    (output_dir / "wandb_run_info.json").write_text(json.dumps(
        {"run_id": run.id, "run_name": run.name, "project": run.project,
         "entity": run.entity, "url": run.url},
        indent=2,
    ))
    wandb.finish()
    print(f"Done. Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune dictalm2.0-instruct for Hebrew summarization")
    parser.add_argument("--method", choices=list(METHOD_PRESETS), default="qlora")
    parser.add_argument("--variant", choices=("whole", "lead", "body"), default="whole")
    parser.add_argument("--output", default=None, help="Local checkpoint dir (default: outputs/checkpoints/<method>-<variant>)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Cap steps for a smoke run")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=0, help="Override per-device batch size (for memory-limited local GPUs)")
    parser.add_argument("--push-to-hub", action="store_true", help="Push the trained adapter to the Hub")
    parser.add_argument("--submit-hf", action="store_true", help="Submit a remote training job to HF Jobs instead of training locally")
    parser.add_argument("--hf-user", default="", help="HuggingFace username (required with --submit-hf or --push-to-hub)")
    parser.add_argument("--smoke-test", action="store_true", help="With --submit-hf: quick 10-step job on a10g-small")
    parser.add_argument("--mini-test", action="store_true", help="With --submit-hf: small-data 1-epoch job on a10g-small")
    parser.add_argument("--inference-only", action="store_true", help="With --submit-hf: skip training, regenerate predictions from the already-pushed adapter")
    parser.add_argument("--pred-suffix", default="", help="With --submit-hf: suffix for pushed prediction files (e.g. -v2)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--base-model", default="", help=f"Base checkpoint to fine-tune (default: {MODEL_ID}). Requires --output-repo.")
    parser.add_argument("--output-repo", default="", help="Hub repo for the adapter (default: derived from --hf-user/--variant)")
    parser.add_argument("--skip-data-upload", action="store_true", help="With --submit-hf: reuse the splits already on the Hub instead of re-uploading")
    parser.add_argument("--timeout", default="", help="With --submit-hf: override the job timeout (e.g. 8h).")
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Post-train dual-arm decode budget (default: {DEFAULT_MAX_NEW_TOKENS}; "
             "lower = cheaper GPU time when the model hits the cap)",
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN not set. Run: source .env", file=sys.stderr)
        sys.exit(1)

    if args.submit_hf:
        if not args.hf_user:
            print("ERROR: --hf-user required with --submit-hf", file=sys.stderr)
            sys.exit(1)
        if args.base_model and not args.output_repo:
            print("ERROR: --base-model requires --output-repo (refusing to overwrite "
                  f"{model_repo(args.hf_user, args.variant)})", file=sys.stderr)
            sys.exit(1)
        submit_hf_job(args.method, args.variant, hf_token, args.hf_user,
                      args.smoke_test, args.mini_test, args.inference_only,
                      args.pred_suffix, args.epochs, args.base_model,
                      args.output_repo, args.skip_data_upload, args.timeout,
                      args.max_new_tokens)
        return

    if args.push_to_hub and not args.hf_user:
        print("ERROR: --hf-user required with --push-to-hub", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output or f"outputs/checkpoints/{args.method}-{args.variant}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_local(args.method, args.variant, output_dir, args.max_steps,
                args.max_length, args.batch_size, args.push_to_hub, args.hf_user, hf_token,
                args.epochs)


if __name__ == "__main__":
    main()
