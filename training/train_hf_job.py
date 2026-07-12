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
Pipeline step 3 (remote variant): self-contained QLoRA fine-tuning of
dicta-il/dictalm2.0-instruct, run on HuggingFace Jobs. Submitted inline by
training/train.py --submit-hf; never run directly. All settings arrive as
environment variables (the repo is NOT uploaded with the script).

Hyperparameters come from TRAIN_CONFIG / LORA_CONFIG JSON (serialized by train.py
from METHOD_PRESETS) so --method lora|full cannot silently use qlora batch/lr.
Train and serve both apply the model chat template (C0); generation tokenizes with
add_special_tokens=False to avoid double-BOS (C1).

Stability (so a long run is not lost on crash/timeout):
  1. Checkpoints write to /data/output — the per-job bucket volume that survives
     infra restarts of the same job; trainer.train(resume_from_checkpoint=True)
     picks them up automatically.
  2. hub_strategy="every_save" pushes each checkpoint as a Hub commit mid-run,
     so partial adapters exist on OUTPUT_REPO even if the job dies later.
  3. Predictions files are uploaded immediately after each generation loop.

Execution environment: ephemeral HuggingFace Jobs GPU container.
"""
import json
import os
import warnings
from datetime import date
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
PRED_SUFFIX = os.environ.get("PRED_SUFFIX", "")
# One epoch per run by default (override via EPOCHS env / train.py --epochs).
EPOCHS = int(os.environ.get("EPOCHS") or 1)
# Base checkpoint + slug — duplicated from training/config.py on purpose (this script
# is submitted inline and cannot import the repo). train.py passes both as env; keep
# the fallbacks in sync with config.MODEL_ID / config.MODEL_SLUG.
# Do NOT derive the slug with .replace(".", "-") alone: dictalm2.0-instruct would
# become dictalm2-0-instruct and drift from wandb/Hub naming.
MODEL_ID = os.environ.get("BASE_MODEL") or "dicta-il/dictalm2.0-instruct"
MODEL_SLUG = os.environ.get("MODEL_SLUG") or "dictalm2-instruct"
WANDB_PROJECT = os.environ.get("WANDB_PROJECT") or f"amlk-{MODEL_SLUG}"
if SMOKE_TEST:
    _tag = "smoke"
elif MINI_TEST:
    _tag = "mini"
else:
    _tag = ""
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME") or "_".join(
    p for p in [date.today().isoformat(), MODEL_SLUG, METHOD, VARIANT, f"{EPOCHS}ep", _tag] if p
)
os.environ["WANDB_PROJECT"] = WANDB_PROJECT

# Resolved presets from train.py (METHOD_PRESETS). Fallbacks match the qlora preset so a
# hand-fired job without TRAIN_CONFIG still trains sanely.
_DEFAULT_TRAIN = {
    "quantize": True,
    "use_lora": True,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "bf16": True,
    "max_length": 4096,
}
_DEFAULT_LORA = {
    "r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
try:
    TRAIN_CFG = {**_DEFAULT_TRAIN, **json.loads(os.environ.get("TRAIN_CONFIG") or "{}")}
except json.JSONDecodeError:
    TRAIN_CFG = dict(_DEFAULT_TRAIN)
try:
    LORA_CFG = {**_DEFAULT_LORA, **json.loads(os.environ.get("LORA_CONFIG") or "{}")}
except json.JSONDecodeError:
    LORA_CFG = dict(_DEFAULT_LORA)

quantize = bool(TRAIN_CFG["quantize"])
use_lora = bool(TRAIN_CFG["use_lora"])

mode = "inference-only" if INFERENCE_ONLY else f"train+infer  Method={METHOD}"
print(f"Mode: {mode}  Variant: {VARIANT}  Smoke: {SMOKE_TEST}  Mini: {MINI_TEST}")
print(f"Base model: {MODEL_ID}")
print(f"Dataset: {DATASET_REPO}  ->  Output: {OUTPUT_REPO}")
print(f"Epochs: {EPOCHS}")
print(f"Train config: quantize={quantize} use_lora={use_lora} "
      f"batch={TRAIN_CFG['per_device_train_batch_size']} "
      f"accum={TRAIN_CFG['gradient_accumulation_steps']} lr={TRAIN_CFG['learning_rate']}")
# Decode budget for dual-arm generation. Default 128 (was 256) — cost lever; see
# training/config.py DEFAULT_MAX_NEW_TOKENS. train.py passes MAX_NEW_TOKENS via env.
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS") or 128)
print(f"max_new_tokens={MAX_NEW_TOKENS} (post-train dual-arm decode budget)")
print(f"wandb: {WANDB_PROJECT} / {WANDB_RUN_NAME}")

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
    # 80 train examples / 1 epoch — enough to show a real loss curve without full-run budget.
    train_ds = train_ds.select(range(min(80, len(train_ds))))
    val_ds = val_ds.select(range(min(20, len(val_ds))))
    test_ds = test_ds.select(range(min(10, len(test_ds))))
    print(f"[Mini] Truncated to train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
else:
    # Eval on a fixed 200-example slice — full 1000-example eval several times would blow the budget.
    val_ds = val_ds.select(range(min(200, len(val_ds))))

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# Twin of data.prompts.prepare_tokenizer_for_templated_prompts (this script can't import repo).
if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "add_bos_token"):
    tokenizer.add_bos_token = False


def format_chat_prompt(prompt: str) -> str:
    """Twin of data.prompts.format_chat_prompt — keep in sync by hand.

    Applies the model chat template for train and both inference arms. Does not
    inject family-specific control tokens; enable_thinking=False is attempted for
    templates that support it, ignored (TypeError) for Mistral/dictalm2.
    """
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


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
        r=int(LORA_CFG["r"]),
        lora_alpha=int(LORA_CFG["lora_alpha"]),
        lora_dropout=float(LORA_CFG["lora_dropout"]),
        target_modules=list(LORA_CFG["target_modules"]),
        bias=LORA_CFG["bias"],
        task_type=LORA_CFG["task_type"],
    ) if use_lora else None

    # Wrap train/val so SFT sees [INST]…[/INST], matching inference.
    print("Wrapping train/val prompts in chat template (if present)...")
    train_ds = train_ds.map(lambda ex: {**ex, "prompt": format_chat_prompt(ex["prompt"])})
    val_ds = val_ds.map(lambda ex: {**ex, "prompt": format_chat_prompt(ex["prompt"])})

    # Mini: log every step; smoke: 10 steps; full 1-epoch: save every 100 steps so Hub
    # gets several mid-run commits (hub_strategy=every_save).
    if MINI_TEST:
        n_epochs, max_steps_cfg = 1, -1
        log_steps, eval_steps_cfg, save_steps_cfg = 1, 5, 20
    elif SMOKE_TEST:
        n_epochs, max_steps_cfg = 1, 10
        log_steps, eval_steps_cfg, save_steps_cfg = 5, 5, 5
    else:
        n_epochs, max_steps_cfg = EPOCHS, -1
        log_steps, eval_steps_cfg, save_steps_cfg = 10, 100, 100

    # /data is the bucket run_uv_job auto-mounts — survives infra restarts of this job.
    output_dir = "/data/output"
    print(f"Stability: checkpoints → {output_dir} (bucket resume)")
    print(f"Stability: hub_strategy=every_save → {OUTPUT_REPO} every {save_steps_cfg} steps")

    sft_config = SFTConfig(
        output_dir=output_dir,
        push_to_hub=True,
        hub_model_id=OUTPUT_REPO,
        hub_strategy="every_save",
        hub_private_repo=True,
        num_train_epochs=n_epochs,
        max_steps=max_steps_cfg,
        per_device_train_batch_size=int(TRAIN_CFG["per_device_train_batch_size"]),
        per_device_eval_batch_size=1,   # eval defaults to 8 → OOM at long seq lengths
        gradient_accumulation_steps=int(TRAIN_CFG["gradient_accumulation_steps"]),
        learning_rate=float(TRAIN_CFG["learning_rate"]),
        warmup_ratio=float(TRAIN_CFG["warmup_ratio"]),
        lr_scheduler_type=TRAIN_CFG["lr_scheduler_type"],
        logging_steps=log_steps,
        save_strategy="steps",
        save_steps=save_steps_cfg,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=eval_steps_cfg,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bool(TRAIN_CFG["bf16"]),
        gradient_checkpointing=True,
        # TRL auto-appends EOS to each completion; completion_only_loss keeps it in the mask.
        completion_only_loss=True,
        max_length=int(TRAIN_CFG["max_length"]),
        report_to="wandb",
        run_name=WANDB_RUN_NAME,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    if use_lora:
        # Print so a base-model swap can never silently regress LoRA layer coverage.
        trainer.model.print_trainable_parameters()

    resume_from_checkpoint = None
    if os.path.isdir(output_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(output_dir)
    ):
        resume_from_checkpoint = True
        print(f"Found existing checkpoint(s) in {output_dir} — resuming training.")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    # Final Hub commit with the best (or last) weights — mid-run saves already pushed via every_save.
    trainer.push_to_hub()
    print(f"Final adapter push complete → {OUTPUT_REPO}")

    trained_model = trainer.model.eval()
    device = next(trained_model.parameters()).device

# Restore KV cache for faster generation (disabled during gradient checkpointing).
trained_model.config.use_cache = True


def _build_bad_words_ids():
    """Forbid tokens containing foreign scripts (Latin/Cyrillic/Greek/Arabic/CJK/Hangul).

    Inlined twin of evaluation/hebrew_constraint.py (this script can't import repo code).
    """
    import re
    forbidden = re.compile(
        "[A-Za-zÀ-ɏЀ-ӿͰ-Ͽ؀-ۿ぀-ヿ㐀-鿿가-힯ᄀ-ᇿㄱ-ㆿ]")
    special_ids = set(tokenizer.all_special_ids)
    bad = []
    for token_id in range(tokenizer.vocab_size):
        if token_id in special_ids:
            continue
        piece = tokenizer.decode([token_id])
        if piece and forbidden.search(piece):
            bad.append([token_id])
    print(f"Hebrew-script constraint: forbidding {len(bad)} foreign-script tokens")
    return bad or None


BAD_WORDS_IDS = _build_bad_words_ids()


def generate_predictions(label: str) -> list[dict]:
    """Generate test-set summaries in batches of 8 with progress logging.

    Left-padding ensures all sequences in a batch are right-aligned so that
    the slice `out[:, input_len:]` consistently extracts only generated tokens.
    Finetuned and base use the same chat-wrapped prompts.

    Importable twin: evaluation/infer.py:generate_summaries — keep in sync by hand.
    """
    tokenizer.padding_side = "left"
    rows = []
    batch_size = 8
    for i in range(0, len(test_ds), batch_size):
        batch = test_ds[i:i + batch_size]
        prompts: list[str] = [format_chat_prompt(p) for p in batch["prompt"]]
        # Chat template already includes BOS — do not prepend another.
        # Leave headroom for max_new_tokens under the same seq budget as training.
        gen_max_input = int(TRAIN_CFG["max_length"]) - MAX_NEW_TOKENS
        inputs = tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=gen_max_input, padding=True, add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            outs = trained_model.generate(
                # DEFAULT_MAX_NEW_TOKENS=128 (config); short news summaries rarely need more,
                # and smoke preds always hit the old 256 cap without EOS (wasted GPU decode).
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=min(16, MAX_NEW_TOKENS), do_sample=False,
                no_repeat_ngram_size=3, repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                bad_words_ids=BAD_WORDS_IDS,
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
    job timeout must never destroy finished work."""
    rows = generate_predictions(label)
    path = Path(f"predictions-{label}{PRED_SUFFIX}.jsonl")
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
