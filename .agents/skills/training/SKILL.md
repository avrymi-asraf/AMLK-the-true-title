---
name: training
description: HF Jobs training process — LoRA SFT on PT dataset, W&B logging, model push.
---

# Training Process

Training runs on Hugging Face Jobs via `scripts/train_pt_dataset_job.py`. The script
is model-agnostic: all settings come from a `configs/train/*.yaml` file selected via
the `TRAINING_CONFIG` env var.

## What the script does

1. Loads the training config YAML (or falls back to defaults).
2. Calls `whoami()` to resolve the HF namespace for default `hub_model_id`.
3. Loads the Hub PT dataset using `load_dataset(repo, config_name, split="train")` —
   the `config_name` is `"training"` (set by `dataset_config` in the YAML).
4. Builds a 4-bit QLoRA config with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=bfloat16, bnb_4bit_quant_type="nf4")`.
5. Trains with `SFTTrainer` using `completion_only_loss=True` on `prompt`/`completion` columns.
6. Logs every step to W&B (`WANDB_PROJECT`, `WANDB_RUN_GROUP`).
7. Saves the adapter at `max_steps` and pushes to Hub if `HF_TOKEN` is present.

## Key config fields (`configs/train/*.yaml`)

```yaml
base_model: <any HF model id>
hub_dataset_id: <HF dataset repo with canonical+training configs>
hub_model_id: <target adapter repo>
dataset_config: training          # config_name used in push_to_hub
output_dir: <local adapter dir>
max_steps: 200
learning_rate: 0.0001
per_device_train_batch_size: 1
gradient_accumulation_steps: 4   # effective batch = batch × accum
max_length: 1024
report_to: [wandb]
project: reasoning-pruning
run_name: <W&B run name>
```

## Job submission

`run_uv_job` uploads only the script file — config YAMLs in `configs/train/` are NOT
available on the HF Jobs server. Pass every config value as an explicit env var:

```python
api.run_uv_job(
    script="scripts/train_pt_dataset_job.py",
    flavor="t4-small",           # T4 16GB sufficient for 4-bit QLoRA
    secrets={"HF_TOKEN": ..., "WANDB_API_KEY": ...},
    env={
        "BASE_MODEL": "google/gemma-4-E2B-it",
        "HUB_DATASET_ID": "<hub-dataset-repo>",
        "HUB_MODEL_ID": "<hub-model-repo>",
        "HUB_DATASET_CONFIG": "training",
        "OUTPUT_DIR": "<local-adapter-dir>",
        "MAX_STEPS": "200",
        "LEARNING_RATE": "0.0001",
        "PER_DEVICE_TRAIN_BATCH_SIZE": "1",
        "GRADIENT_ACCUMULATION_STEPS": "4",
        "MAX_LENGTH": "1024",
        "REPORT_TO": "wandb",
        "WANDB_PROJECT": "reasoning-pruning",
        "RUN_NAME": "<run-name>",
        "ROUND_ID": "<round-id>",
    },
    timeout="3h",
)
```

## Known lessons

- `run_uv_job` uploads only the script file, NOT the repo. Config YAML files in
  `configs/train/` are not accessible on the HF Jobs server. Always pass all config
  values as explicit env vars when submitting — never rely on `TRAINING_CONFIG` pointing
  to a local file path. The `_load_config()` call in the script silently returns `{}`
  when the file doesn't exist, causing wrong defaults (e.g. wrong dataset ID).

- `torchvision` is required even for text-only training when the base model is
  multimodal (e.g. Gemma-4). Include it in the PEP 723 deps.
- `completion_only_loss=True` in `SFTConfig` trains only on the `completion` column —
  critical for PT rows so the model learns to generate `target_y`, not restate `input_x`.
- `bf16=True` and `bnb_4bit_compute_dtype=bfloat16` must be consistent; mixing
  float16 and bfloat16 causes silent precision issues on A10G/T4.
- Trackio export fails when `rank_pattern` is empty in the LoRA config — use W&B only.
- The Hub dataset must be published with `config_name="training"` (not as a split
  named `"training"`); load downstream with `load_dataset(repo, "training", split="train")`.
- `gradient_accumulation_steps=4` with batch_size=1 gives effective batch 4 — important
  for stable LoRA training when the dataset is small (~30–100 rows).

## Job monitoring

**Check current logs (non-blocking snapshot):**
```bash
source .env && .venv/bin/hf jobs logs <job-id>
source .env && .venv/bin/hf jobs logs --tail 50 <job-id>   # last 50 lines
```

**Stream live logs until job ends (blocking):**
```bash
source .env && .venv/bin/hf jobs logs -f <job-id>
```
In Claude Code's Bash tool, use `run_in_background: true` with `-f` and read the output file for progress, OR poll with `--tail` without `-f`.

**Check job status and metadata:**
```bash
source .env && .venv/bin/hf jobs inspect <job-id>
source .env && .venv/bin/hf jobs ps          # list recent jobs
```

**Cancel a running job:**
```bash
source .env && .venv/bin/hf jobs cancel <job-id>
```

**Monitor URL:** `https://huggingface.co/jobs/avreymi/<job-id>`

## Completed runs

- Smoke run: `avreymi/gemma-4-E2B-it-reasoning-pruning-smoke`, HF job `6a187ee53a4b8cae6044d45f`,
  W&B `https://wandb.ai/avreymi-asraf-hebrew-university-of-jerusalem/reasoning-pruning/runs/mdv0vre3`.
- GSM8K-100-r1: dataset `avreymi/reasoning-pruning-pt-gsm8k-100-gemma4-r1` (33 rows, completed),
  training job `6a18aca83a4b8cae6044d618` FAILED (config file not accessible on HF Jobs),
  resubmitted as job `6a18ad855c8d10ffa110633e` on 2026-05-29 with explicit env vars,
  config `configs/train/training_gemma4_gsm8k_100.yaml`,
  target model `avreymi/gemma-4-E2B-it-reasoning-pruning-gsm8k-100-r1`.
