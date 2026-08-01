"""
Pipeline step 3 of 3: fine-tune dicta-il/dictalm2.0-instruct on Hebrew summarization.
One entry point for all three regimes the paper compares — --method qlora | lora | full
— differing only by the small METHOD_PRESETS deltas in config.py. Trains with the trl
SFT trainer using completion_only_loss=True (loss on the summary only), logs every step
to Weights & Biases, and saves / optionally pushes the adapter (or full model) to the Hub.
Remote body is always training/train_hf_job.py via --submit-hf (HF Jobs) or
--submit-colab (Colab T4; see training/colab_submit.py). Inference lives separately
in evaluation/predict.py.

Run (HF Jobs):  python -m training.train --submit-hf --hf-user avreymi [--method lora] [--smoke-test]
Run (Colab):    python -m training.train --submit-colab --hf-user avreymi --method qlora --smoke-test
Execution environment: HF Jobs or Colab GPU for real runs; never load 7B on local 8 GB GPU.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Heavy training deps (datasets/torch/peft/transformers/trl/wandb) are imported lazily inside
# the functions that need them, so `--submit-hf` / `--submit-colab` can run on a minimal local
# env (no GPU stack) — submission only needs huggingface_hub (+ colab CLI for Colab path).

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
from training.resume import (
    checkpoint_step,
    hub_full_trainer_checkpoints,
    is_full_trainer_checkpoint_dir,
    pick_resume_checkpoint,
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

    # Pin SDPA rather than trusting the transformers default: at max_length=4096 eager attention
    # is a large multiplier on step time. Twin of the same kwarg in train_hf_job.py.
    load_kwargs = dict(token=hf_token or None, device_map="auto", attn_implementation="sdpa")
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


def push_resume_checkpoint(local_dir: str, output_repo: str, hf_token: str,
                           checkpoint_name: str = "") -> str:
    """Upload a local full Trainer checkpoint dir to OUTPUT_REPO/checkpoint-N.

    Returns the Hub checkpoint name (e.g. checkpoint-200). Used so a killed job's
    /data/output (or a bucket copy) can be resumed from a *new* HF Job via RESUME_FROM.
    """
    from huggingface_hub import HfApi

    src = Path(local_dir)
    if not is_full_trainer_checkpoint_dir(src):
        print(
            f"ERROR: {src} is not a full Trainer checkpoint "
            f"(need trainer_state.json, optimizer.pt, scheduler.pt, adapter/model weights)",
            file=sys.stderr,
        )
        sys.exit(1)
    name = checkpoint_name or src.name
    if checkpoint_step(name) < 0:
        print(f"ERROR: checkpoint name must look like checkpoint-N, got {name!r}", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=output_repo, repo_type="model", private=True, exist_ok=True)
    print(f"Uploading full Trainer checkpoint {src} → {output_repo}/{name} ...")
    api.upload_folder(
        folder_path=str(src),
        repo_id=output_repo,
        repo_type="model",
        path_in_repo=name,
        commit_message=f"Add full Trainer {name} for cross-job resume",
    )
    print(f"Done. Resume with: --resume-from {name}")
    return name


def build_job_env(
    method: str,
    variant: str,
    *,
    data_repo: str,
    out_repo: str,
    base: str,
    n_epochs: int,
    smoke_test: bool,
    mini_test: bool,
    inference_only: bool,
    resume_key: str,
    pred_suffix: str,
    max_new_tokens: int,
    max_train: int,
    test_subset: int,
    skip_base_arm: bool,
    run_tag: str,
    learning_rate: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    batch_size: int,
    label: str,
    output_dir: str = "",
    data_dir: str = "",
) -> tuple[dict[str, str], dict, str, str]:
    """Serialize METHOD_PRESETS + run knobs into the env dict train_hf_job.py reads.

    Shared by --submit-hf and --submit-colab so both backends get identical
    TRAIN_CONFIG / LORA_CONFIG / wandb naming. Returns (env, train_cfg, project, run_name).
    """
    project = wandb_project(MODEL_SLUG)
    run_name = wandb_run_name(
        method, variant, model_slug=MODEL_SLUG, epochs=n_epochs,
        tag="-".join(t for t in (label, run_tag) if t),
    )
    train_cfg = _train_config_payload(method)
    if learning_rate:
        train_cfg["learning_rate"] = learning_rate
    if batch_size and batch_size > 0:
        # Hold effective batch fixed so LR / step count stay comparable to the preset.
        eff = (int(train_cfg["per_device_train_batch_size"])
               * int(train_cfg["gradient_accumulation_steps"]))
        if eff % batch_size != 0:
            print(
                f"WARNING: effective batch {eff} not divisible by --batch-size {batch_size}; "
                f"using accum={max(1, eff // batch_size)} "
                f"(new effective={batch_size * max(1, eff // batch_size)})",
                file=sys.stderr,
            )
        train_cfg["per_device_train_batch_size"] = int(batch_size)
        train_cfg["gradient_accumulation_steps"] = max(1, eff // batch_size)
    lora_cfg = _lora_config_payload() if train_cfg["use_lora"] else {}
    n_new_tokens = max_new_tokens or DEFAULT_MAX_NEW_TOKENS

    env: dict[str, str] = {
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
        "RESUME_FROM": resume_key,
        "PRED_SUFFIX": pred_suffix,
        "EPOCHS": str(n_epochs),
        "MAX_NEW_TOKENS": str(n_new_tokens),
        "MAX_TRAIN_EXAMPLES": str(max_train or 0),
        "TEST_SUBSET_N": str(test_subset or 0),
        "SKIP_BASE_ARM": "1" if skip_base_arm else "0",
        "REPETITION_PENALTY": str(repetition_penalty) if repetition_penalty else "",
        "NO_REPEAT_NGRAM_SIZE": (
            str(no_repeat_ngram_size) if no_repeat_ngram_size >= 0 else ""
        ),
        "TRAIN_CONFIG": json.dumps(train_cfg),
        "LORA_CONFIG": json.dumps(lora_cfg),
    }
    if output_dir:
        env["OUTPUT_DIR"] = output_dir
    if data_dir:
        env["DATA_DIR"] = data_dir
    return env, train_cfg, project, run_name


def prepare_remote_submit(
    method: str, variant: str, hf_token: str, hf_user: str,
    smoke_test: bool, mini_test: bool = False, inference_only: bool = False,
    pred_suffix: str = "", epochs: int = 0, base_model: str = "",
    output_repo: str = "", skip_data_upload: bool = False,
    max_new_tokens: int = 0, resume_from: str = "",
    dataset_repo_override: str = "", max_train: int = 0, test_subset: int = 0,
    skip_base_arm: bool = False, run_tag: str = "", learning_rate: float = 0.0,
    repetition_penalty: float = 0.0, no_repeat_ngram_size: int = -1,
    batch_size: int = 0, output_dir: str = "", data_dir: str = "",
    upload_data: bool = True, create_model_repo: bool = True,
) -> dict:
    """Hub prep + shared env for remote train_hf_job backends (HF Jobs or Colab).

    Returns a dict with keys: env, train_cfg, project, run_name, data_repo, out_repo,
    n_epochs, base, resume_key, label, n_new_tokens.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    data_repo = dataset_repo_override or dataset_repo(hf_user, variant)
    out_repo = output_repo or model_repo(hf_user, variant)
    n_epochs = epochs or DEFAULT_EPOCHS
    base = base_model or MODEL_ID
    resume_key = (resume_from or "").strip()

    if resume_key and inference_only:
        print("ERROR: --resume-from cannot be combined with --inference-only", file=sys.stderr)
        sys.exit(1)

    if dataset_repo_override and not skip_data_upload:
        print(f"ERROR: --dataset-repo {dataset_repo_override} requires --skip-data-upload",
              file=sys.stderr)
        sys.exit(1)

    if upload_data and not inference_only and not skip_data_upload:
        data_path = Path(PROCESSED_DIR) / processed_profile_name(variant)
        if not data_path.exists():
            print(f"ERROR: {data_path} not found. Run: python -m data.preprocess --variant {variant}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"Uploading {data_path} to {data_repo}...")
        api.create_repo(repo_id=data_repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=str(data_path), repo_id=data_repo, repo_type="dataset")

    if create_model_repo and not inference_only:
        api.create_repo(repo_id=out_repo, repo_type="model", private=True, exist_ok=True)

    if resume_key:
        repo_files = api.list_repo_files(out_repo, repo_type="model")
        available = hub_full_trainer_checkpoints(repo_files)
        try:
            chosen = pick_resume_checkpoint(available, resume_key)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        if chosen is None:
            print(
                f"ERROR: --resume-from={resume_key!r} but {out_repo} has no full Trainer "
                f"checkpoint-* (need trainer_state.json + optimizer.pt + adapter). "
                f"Upload with: python -m training.train --push-resume-checkpoint DIR "
                f"--output-repo {out_repo}",
                file=sys.stderr,
            )
            sys.exit(1)
        resume_key = chosen
        print(f"  Will resume from Hub full Trainer checkpoint: {out_repo}/{resume_key}")

    if inference_only:
        label = "infer"
    elif smoke_test:
        label = "smoke"
    elif mini_test:
        label = "mini"
    elif resume_key:
        label = f"resume{checkpoint_step(resume_key)}"
    else:
        label = ""

    env, train_cfg, project, run_name = build_job_env(
        method, variant,
        data_repo=data_repo, out_repo=out_repo, base=base, n_epochs=n_epochs,
        smoke_test=smoke_test, mini_test=mini_test, inference_only=inference_only,
        resume_key=resume_key, pred_suffix=pred_suffix,
        max_new_tokens=max_new_tokens, max_train=max_train, test_subset=test_subset,
        skip_base_arm=skip_base_arm, run_tag=run_tag, learning_rate=learning_rate,
        repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
        batch_size=batch_size, label=label, output_dir=output_dir, data_dir=data_dir,
    )
    n_new_tokens = int(env["MAX_NEW_TOKENS"])
    return {
        "env": env,
        "train_cfg": train_cfg,
        "project": project,
        "run_name": run_name,
        "data_repo": data_repo,
        "out_repo": out_repo,
        "n_epochs": n_epochs,
        "base": base,
        "resume_key": resume_key,
        "label": label,
        "n_new_tokens": n_new_tokens,
        "api": api,
    }


def submit_hf_job(method: str, variant: str, hf_token: str, hf_user: str,
                  smoke_test: bool, mini_test: bool = False, inference_only: bool = False,
                  pred_suffix: str = "", epochs: int = 0, base_model: str = "",
                  output_repo: str = "", skip_data_upload: bool = False,
                  timeout: str = "", max_new_tokens: int = 0,
                  resume_from: str = "", dataset_repo_override: str = "",
                  max_train: int = 0, test_subset: int = 0, skip_base_arm: bool = False,
                  run_tag: str = "", learning_rate: float = 0.0,
                  repetition_penalty: float = 0.0, no_repeat_ngram_size: int = -1,
                  batch_size: int = 0):
    """Upload the processed splits to the Hub and submit train_hf_job.py to HF Jobs.

    inference_only=True skips dataset re-upload and training; loads the already-pushed
    adapter and regenerates predictions only. resume_from downloads a full Trainer
    checkpoint into OUTPUT_DIR (/data/output) for cross-job resume. Unchanged Colab path:
    use --submit-colab instead.
    """
    prepared = prepare_remote_submit(
        method, variant, hf_token, hf_user, smoke_test, mini_test, inference_only,
        pred_suffix, epochs, base_model, output_repo, skip_data_upload,
        max_new_tokens, resume_from, dataset_repo_override, max_train, test_subset,
        skip_base_arm, run_tag, learning_rate, repetition_penalty, no_repeat_ngram_size,
        batch_size,
    )
    resume_key = prepared["resume_key"]
    if inference_only:
        flavor = "a10g-small"
        # Batch-1 4-bit dual-arm on ~586 test examples is ~2.5–3.5h; 2h was too tight.
        timeout = timeout or "5h"
    elif smoke_test:
        flavor = "a10g-small"
        timeout = timeout or "30m"
    elif mini_test:
        flavor = "a10g-small"
        timeout = timeout or "1h"
    elif resume_key:
        flavor = "a10g-small"
        timeout = timeout or "4h"
    else:
        flavor = "a10g-small"
        timeout = timeout or "8h"

    script_path = Path(__file__).parent / "train_hf_job.py"
    wandb_key = wandb_api_key()
    label = prepared["label"]
    train_cfg = prepared["train_cfg"]
    tag = f"{label} " if label else ""
    print(f"Submitting {tag}{method} job (flavor={flavor}, timeout={timeout})...")
    print(f"  Base model: {prepared['base']}")
    print(f"  Output repo: {prepared['out_repo']}")
    print(f"  Epochs: {prepared['n_epochs']}")
    print(f"  Train config: batch={train_cfg['per_device_train_batch_size']} "
          f"accum={train_cfg['gradient_accumulation_steps']} lr={train_cfg['learning_rate']}")
    print(f"  max_new_tokens={prepared['n_new_tokens']} (post-train dual-arm decode budget)")
    if resume_key:
        print(f"  Resume from: {resume_key} (Hub full Trainer ckpt → finish epoch + gen)")
    print(f"  wandb: {prepared['project']} / {prepared['run_name']}")
    print(
        f"  Stability: hub_strategy=all_checkpoints (full Trainer ckpt every save; "
        f"soft-fail Hub I/O) + /data/output resume"
    )
    job = prepared["api"].run_uv_job(
        script=str(script_path),
        flavor=flavor,
        timeout=timeout,
        secrets={"HF_TOKEN": hf_token, "WANDB_API_KEY": wandb_key},
        env=prepared["env"],
        token=hf_token,
    )
    print(f"\nJob submitted. ID: {job.id}  Status: {job.status.stage}")
    print(f"  Monitor: https://huggingface.co/jobs/{hf_user}/{job.id}")
    print(f"  Logs:    hf jobs logs {job.id}")
    print(f"  Model:   https://huggingface.co/{prepared['out_repo']}  (after training)")
    return job


def submit_colab_job_from_cli(
    method: str, variant: str, hf_token: str, hf_user: str,
    smoke_test: bool, mini_test: bool = False, inference_only: bool = False,
    pred_suffix: str = "", epochs: int = 0, base_model: str = "",
    output_repo: str = "", skip_data_upload: bool = False,
    timeout: str = "", max_new_tokens: int = 0,
    resume_from: str = "", dataset_repo_override: str = "",
    max_train: int = 0, test_subset: int = 0, skip_base_arm: bool = False,
    run_tag: str = "", learning_rate: float = 0.0,
    repetition_penalty: float = 0.0, no_repeat_ngram_size: int = -1,
    batch_size: int = 0, colab_gpu: str = "", colab_session: str = "",
    colab_mode: str = "auto", colab_env_file: str = "", dry_run: bool = False,
):
    """Hub prep + Colab launch of train_hf_job.py (OUTPUT_DIR=/content/amlk-output)."""
    from training.colab_submit import submit_colab_job
    from training.config import COLAB_DATA_DIR, COLAB_DEFAULT_GPU, COLAB_OUTPUT_DIR

    prepared = prepare_remote_submit(
        method, variant, hf_token, hf_user, smoke_test, mini_test, inference_only,
        pred_suffix, epochs, base_model, output_repo, skip_data_upload,
        max_new_tokens, resume_from, dataset_repo_override, max_train, test_subset,
        skip_base_arm, run_tag, learning_rate, repetition_penalty, no_repeat_ngram_size,
        batch_size, output_dir=COLAB_OUTPUT_DIR, data_dir=COLAB_DATA_DIR,
        # Dry-run must not hit Hub (create_repo / resume listing).
        upload_data=not dry_run,
        create_model_repo=not dry_run,
    )
    # Secrets for remote: HF Jobs uses secrets=; Colab loads uploaded .env. Also inject
    # wandb key into env JSON if local has it but .env might not (entry merges both).
    env = dict(prepared["env"])
    wb = wandb_api_key()
    if wb:
        env["WANDB_API_KEY"] = wb
    # HF_TOKEN is in .env upload; also put on env for bootstrap path if needed.
    env["HF_TOKEN"] = hf_token

    train_cfg = prepared["train_cfg"]
    print(f"Submitting Colab {method} job (gpu={colab_gpu or COLAB_DEFAULT_GPU})...")
    print(f"  Base model: {prepared['base']}")
    print(f"  Output repo: {prepared['out_repo']}")
    print(f"  Epochs: {prepared['n_epochs']}")
    print(f"  Train config: batch={train_cfg['per_device_train_batch_size']} "
          f"accum={train_cfg['gradient_accumulation_steps']} lr={train_cfg['learning_rate']}")
    print(f"  OUTPUT_DIR={COLAB_OUTPUT_DIR}  DATA_DIR={COLAB_DATA_DIR}")
    print(f"  wandb: {prepared['project']} / {prepared['run_name']}")
    print(
        "  Stability: hub_strategy=all_checkpoints + Hub soft-push "
        "(Colab /content is ephemeral; Hub is cross-session durability)"
    )
    rc = submit_colab_job(
        env,
        smoke_test=smoke_test,
        mini_test=mini_test,
        inference_only=inference_only,
        run_tag=run_tag or prepared["label"] or "colab",
        gpu=colab_gpu or COLAB_DEFAULT_GPU,
        timeout=timeout,
        session=colab_session,
        mode=colab_mode,
        env_file=colab_env_file,
        dry_run=dry_run,
    )
    if rc != 0:
        print(f"ERROR: Colab job exited with code {rc}", file=sys.stderr)
        sys.exit(rc)
    print(f"  Model:   https://huggingface.co/{prepared['out_repo']}")
    return rc


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
        # Cost lever (twin of train_hf_job.py): length-grouped batches pay ~the mean article
        # length instead of E[max of batch] — measured 20% fewer padded tokens at micro-batch 2.
        # transformers 5 replaced the `group_by_length` bool with this enum (default "random").
        train_sampling_strategy="group_by_length",
        completion_only_loss=True,
        max_length=max_length,
        report_to="wandb",
        run_name=run_name,
        push_to_hub=push_to_hub,
        hub_model_id=model_repo(hf_user, variant) if push_to_hub else None,
        # all_checkpoints (not every_save): full Trainer checkpoint-* folders on Hub
        # for --resume-from (optimizer+trainer_state). every_save only copies adapter root.
        hub_strategy="all_checkpoints" if push_to_hub else "end",
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
    # Default is bf16 lora, not qlora: measured 1.37x faster per optimizer step on a paired
    # smoke (see METHOD_PRESETS). Pass --method qlora to fall back if a bigger seq/batch OOMs.
    parser.add_argument("--method", choices=list(METHOD_PRESETS), default="lora")
    parser.add_argument("--variant", choices=("whole", "lead", "body"), default="whole")
    parser.add_argument("--output", default=None, help="Local checkpoint dir (default: outputs/checkpoints/<method>-<variant>)")
    parser.add_argument("--max-steps", type=int, default=-1, help="Cap steps for a smoke run")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="Override per-device train batch size (local or --submit-hf). "
             "On HF, rescales gradient_accumulation to keep effective batch = preset product "
             "(lora: 16). Try 2 on a10g if memory allows — unlocks group_by_length padding savings.",
    )
    parser.add_argument("--push-to-hub", action="store_true", help="Push the trained adapter to the Hub")
    parser.add_argument("--submit-hf", action="store_true", help="Submit a remote training job to HF Jobs instead of training locally")
    parser.add_argument(
        "--submit-colab", action="store_true",
        help="Submit the same train_hf_job body to Google Colab (T4 default; prefer --method qlora). "
             "Does not replace --submit-hf.",
    )
    parser.add_argument("--hf-user", default="", help="HuggingFace username (required with --submit-hf/--submit-colab or --push-to-hub)")
    parser.add_argument("--smoke-test", action="store_true", help="With --submit-hf/--submit-colab: quick 10-step job")
    parser.add_argument("--mini-test", action="store_true", help="With --submit-hf/--submit-colab: small-data 1-epoch job")
    parser.add_argument("--inference-only", action="store_true", help="With --submit-hf/--submit-colab: skip training, regenerate predictions from the already-pushed adapter")
    parser.add_argument("--pred-suffix", default="", help="With remote submit: suffix for pushed prediction files (e.g. -v2)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Training epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--base-model", default="", help=f"Base checkpoint to fine-tune (default: {MODEL_ID}). Requires --output-repo.")
    parser.add_argument("--output-repo", default="", help="Hub repo for the adapter (default: derived from --hf-user/--variant)")
    parser.add_argument("--skip-data-upload", action="store_true", help="With remote submit: reuse the splits already on the Hub instead of re-uploading")
    parser.add_argument("--timeout", default="", help="With --submit-hf: e.g. 8h. With --submit-colab: seconds or 2h/30m.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Post-train dual-arm decode budget (default: {DEFAULT_MAX_NEW_TOKENS}; "
             "lower = cheaper GPU time when the model hits the cap)",
    )
    parser.add_argument(
        "--resume-from", default="",
        help="With remote submit: resume a killed run from a full Trainer checkpoint on "
             "OUTPUT_REPO. Use 'auto' (latest) or 'checkpoint-N'. Requires a Hub upload "
             "with optimizer+trainer_state (not adapter-only root) — see --push-resume-checkpoint.",
    )
    parser.add_argument(
        "--colab-gpu", default="",
        help="With --submit-colab: accelerator (default T4). Prefer qlora on T4.",
    )
    parser.add_argument(
        "--colab-session", default="",
        help="With --submit-colab: durable/run session name (default amlk-colab-smoke / amlk-colab-<tag>).",
    )
    parser.add_argument(
        "--colab-mode", choices=("auto", "durable", "run"), default="auto",
        help="With --submit-colab: durable (new+upload+exec+stop), run (one-shot self-clean), "
             "or auto (durable for smoke, run otherwise).",
    )
    parser.add_argument(
        "--colab-env-file", default="",
        help="With --submit-colab: local secrets file to upload as /content/.env (default: repo .env).",
    )
    parser.add_argument(
        "--colab-dry-run", action="store_true",
        help="With --submit-colab: print plan and exit without allocating a Colab VM.",
    )
    parser.add_argument(
        "--push-resume-checkpoint", default="",
        help="Upload a local full Trainer checkpoint dir (e.g. .../checkpoint-200) to "
             "--output-repo for cross-job --resume-from. Does not submit a job.",
    )
    # --- training-improvement loop (docs/training-improvement-notebook.md) ---
    parser.add_argument("--dataset-repo", default="",
                        help="Train on an alternative Hub dataset (e.g. a distilled target set). "
                             "Requires --skip-data-upload.")
    parser.add_argument("--max-train", type=int, default=0,
                        help="Cap train examples so cheap arms match on step count (0 = all)")
    parser.add_argument("--test-subset", type=int, default=0,
                        help="Generate predictions for the fixed judged subset only "
                             "(evaluation.improve_eval.SUBSET_N; 0 = whole test split)")
    parser.add_argument("--skip-base-arm", action="store_true",
                        help="Skip zero-shot base generation (greedy base preds already scored)")
    parser.add_argument("--run-tag", default="", help="Extra tag in the wandb run name")
    parser.add_argument("--repetition-penalty", type=float, default=0.0,
                        help="Decode repetition penalty (default 1.0 = off; HF applies it to "
                             "the prompt too, which penalizes copying the article's entities)")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=-1,
                        help="Decode no-repeat n-gram size (default 0 = off)")
    parser.add_argument("--learning-rate", type=float, default=0.0,
                        help="Override the method preset's learning rate")
    parser.add_argument(
        "--checkpoint-name", default="",
        help="With --push-resume-checkpoint: Hub folder name (default: the local dir name).",
    )
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN not set. Run: source .env", file=sys.stderr)
        sys.exit(1)

    if args.push_resume_checkpoint:
        out_repo = args.output_repo or (
            model_repo(args.hf_user, args.variant) if args.hf_user else ""
        )
        if not out_repo:
            print("ERROR: --push-resume-checkpoint needs --output-repo or --hf-user",
                  file=sys.stderr)
            sys.exit(1)
        push_resume_checkpoint(
            args.push_resume_checkpoint, out_repo, hf_token, args.checkpoint_name,
        )
        return

    if args.submit_hf and args.submit_colab:
        print("ERROR: pass only one of --submit-hf or --submit-colab", file=sys.stderr)
        sys.exit(1)

    if args.submit_hf or args.submit_colab:
        if not args.hf_user:
            print("ERROR: --hf-user required with --submit-hf / --submit-colab", file=sys.stderr)
            sys.exit(1)
        if args.base_model and not args.output_repo:
            print("ERROR: --base-model requires --output-repo (refusing to overwrite "
                  f"{model_repo(args.hf_user, args.variant)})", file=sys.stderr)
            sys.exit(1)
        common = dict(
            method=args.method, variant=args.variant, hf_token=hf_token, hf_user=args.hf_user,
            smoke_test=args.smoke_test, mini_test=args.mini_test,
            inference_only=args.inference_only, pred_suffix=args.pred_suffix,
            epochs=args.epochs, base_model=args.base_model, output_repo=args.output_repo,
            skip_data_upload=args.skip_data_upload, timeout=args.timeout,
            max_new_tokens=args.max_new_tokens, resume_from=args.resume_from,
            dataset_repo_override=args.dataset_repo, max_train=args.max_train,
            test_subset=args.test_subset, skip_base_arm=args.skip_base_arm,
            run_tag=args.run_tag, learning_rate=args.learning_rate,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size, batch_size=args.batch_size,
        )
        if args.submit_hf:
            submit_hf_job(**common)
        else:
            submit_colab_job_from_cli(
                **common,
                colab_gpu=args.colab_gpu,
                colab_session=args.colab_session,
                colab_mode=args.colab_mode,
                colab_env_file=args.colab_env_file,
                dry_run=args.colab_dry_run,
            )
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
