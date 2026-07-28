---
name: training
description: AMLK training process — fine-tune dicta-il/dictalm2.0-instruct for Hebrew summarization (qlora|lora|full) on HF Jobs, with informative wandb names, 1-epoch runs, and mid-run Hub checkpoint pushes.
---

# AMLK Training Process

One script trains all three regimes the paper compares: `training/train.py`, selected with
`--method qlora|lora|full`. The methods differ only by the small `METHOD_PRESETS` deltas in
`training/config.py`. The self-contained `training/train_hf_job.py` is the same logic packaged
for HuggingFace Jobs.

**One data path only:** curated HeSum → HuggingFace Arrow training splits.
`data.download` resolves `final_clean_hesum.json` from (1)
`outputs/data/curated/` or (2) `data_curation/artifacts/` (this worktree). Then:

```bash
python -m data.download                 # → curated_records.jsonl
python -m data.preprocess --variant whole --force   # → processed/whole/{train,val,test}
```

Preprocess builds columns `text, summary, source, prompt, completion` (raw instruction
prompts; chat-wrap later), truncates articles to `MAX_LENGTH-256`, splits 80/10/10, and
runs `validate_train_dataset` before save. Default **1 epoch** per train run. Hebrew decode
constraint is always on at generation.

## How a run is wired

1. Load the processed Arrow splits from `outputs/data/processed/<variant>/{train,val}`
   (built by `data/preprocess.py` from curated HeSum — stores **raw** instruction prompts).
2. Load the base model (`dicta-il/dictalm2.0-instruct` by default): 4-bit NF4 (`qlora`) or
   bf16 (`lora`, `full`).
3. **Chat-wrap** train/val prompts with `format_chat_prompt` so instruct models see
   `[INST]…[/INST]` at train time (same wrap at inference for finetuned + base). Disable
   double-BOS (`add_bos_token=False` / generate with `add_special_tokens=False`).
4. `SFTTrainer` with `peft_config` + `processing_class=tokenizer`, `completion_only_loss=True`.
   Hyperparameters come from `METHOD_PRESETS` via `TRAIN_CONFIG`/`LORA_CONFIG` env JSON.
5. wandb: project `amlk-{MODEL_SLUG}`, run name `{date}_{slug}_{method}_{variant}_{N}ep[_tag]`.
6. **Stability:**
   - Checkpoints → `/data/output` (per-job bucket; survives infra restart; auto-resume).
   - `hub_strategy="every_save"` → each checkpoint is a Hub commit mid-run.
   - Predictions uploaded immediately after each generation loop.
   - Full-run timeout default **8h** (7B QLoRA worst-case ~5.8h at smoke step-time).

## Run it (always `python -m` from repo root)

> **Do NOT train or run model inference on the local machine — it freezes (8 GB GPU).**
> Everything model-related runs on HF Jobs.

```bash
source .env && source .venv/bin/activate

# Data (curated only — no IAHLT/raw HeSum download in this repo)
python -m data.download
python -m data.preprocess --variant whole --force

# Hub already has curated whole data (2026-07-26). Re-upload only after rebuild:
# omit --skip-data-upload (or after MAX_LENGTH / preprocess changes).
# Default --method is lora (measured ~1.37x faster than qlora on 10-step smokes).
python -m training.train --submit-hf --hf-user avreymi --smoke-test \
  --skip-data-upload --output-repo avreymi/amlk-dictalm2-instruct-smoke

python -m training.train --submit-hf --hf-user avreymi --skip-data-upload
python -m training.train --submit-hf --hf-user avreymi --inference-only

# Cross-job resume (full Trainer ckpt with optimizer — not adapter-only Hub root):
# 1) Upload a killed job's checkpoint-N (from bucket or local) to OUTPUT_REPO:
python -m training.train --push-resume-checkpoint /path/to/checkpoint-200 \
  --output-repo avreymi/amlk-dictalm2-instruct-sft
# 2) Finish remaining steps + dual-arm generation:
python -m training.train --submit-hf --hf-user avreymi --skip-data-upload \
  --resume-from checkpoint-200
#    --resume-from auto  → highest full checkpoint-* on OUTPUT_REPO
```

> **Cost note:** `dictalm2.0-instruct` is Mistral-7B on a10g-small (same 24 GB GPU as
> a10g-large, $1.00/h vs $1.50/h). Default method is **lora** (bf16, micro-batch 1 /
> accum 16); use `--method qlora` if memory is tight. Seq budget is `MAX_LENGTH=4096`
> (config source of truth; twin fallbacks in `train_hf_job.py` + gen sites).

## Train-data contract (do not break)

- Local: `outputs/data/processed/<variant>/{train,val,test}` Arrow dirs from `datasets.save_to_disk`.
- Columns: `text`, `summary`, `source`, `prompt`, `completion` (all non-empty strings).
- `completion == summary`; `prompt` contains `text` (hardened Hebrew instruction + article).
- `source` is `hesum-curated`.
- Hub: private `{hf_user}/amlk-training-data` (or `-lead`/`-body`). **As of 2026-07-26**,
  `avreymi/amlk-training-data` holds curated whole splits (4683/585/586, `hesum-curated`).
  Upload via `train.py --submit-hf` or a one-shot `HfApi.upload_folder` of
  `outputs/data/processed/<variant>/`.
- `completion_only_loss=True` requires `prompt`/`completion` columns.
- Curated input: `outputs/data/curated/final_clean_hesum.json` (or
  `data_curation/artifacts/final_clean_hesum.json`) rows
  `{hesum_id, text, headline}` (headline becomes `summary`/`completion`).

## trl 1.6.0 / transformers 5.x API (verified — do not regress)

- `max_length=` (NOT `max_seq_length=`), `processing_class=tokenizer` (NOT `tokenizer=`).
- Load the model object ourselves and pass `model=<object>`.
- `completion_only_loss=True` requires `prompt`/`completion` columns.

## HF Jobs submission — the hard rule

`run_uv_job` uploads **only the script file**. Pass every setting as env
(`METHOD`, `VARIANT`, `BASE_MODEL`, `MODEL_SLUG`, `DATASET_REPO`, `OUTPUT_REPO`,
`WANDB_PROJECT`, `WANDB_RUN_NAME`, `EPOCHS`, `SMOKE_TEST`, `TRAIN_CONFIG`, `LORA_CONFIG`).
Secrets must be real token strings via the Python API (not `"$HF_TOKEN"`). Never hardcode
batch/lr in `train_hf_job.py` — always resolve from `METHOD_PRESETS` through `TRAIN_CONFIG`.
`MODEL_SLUG` must be passed (not derived with naive `.`→`-` replace — that turns
`dictalm2.0-instruct` into the wrong `dictalm2-0-instruct`).

## Monitoring

```bash
hf jobs ps
hf jobs logs <job-id>            # add -f to stream
hf jobs inspect <job-id>
```

wandb: project `amlk-dictalm2-instruct` (see `training.config.wandb_project`).

## Lessons (keep these true)

- Instruct models must train and serve under their chat template (C0) — raw completion prompts
  silently break dictalm2.0-instruct.
- Never inject `/no_think` into Mistral prompts; never double-BOS after `apply_chat_template`.
- `per_device_eval_batch_size=1` — eval default 8 OOMs at long seq lengths on A10G 24 GB.
- HeSum articles are long — preprocess truncates to `MAX_LENGTH-256` (3840 at MAX_LENGTH=4096)
  so the summary survives. Changing `MAX_LENGTH` requires twins in `train_hf_job` defaults +
  gen truncation (`infer.py` / `predict_base_hf_job.py`), then re-preprocess + Hub re-upload.
- Hub adapter is LoRA only (not merged).
- Cloud-job crash economics: mid-run Hub commits + immediate prediction pushes are non-negotiable.
- Training data is **curated HeSum only** — do not reintroduce raw biunlp/HeSum + IAHLT merge
  or in-repo roundup-drop as the train path (that work lives in main-branch `data_curation`).
- For wandb axis alignment, see the global `wandb-for-trl` skill.

## Completed runs

- 2026-07-11 **post clean-only** smoke (qlora, `dicta-il/dictalm2.0-instruct`, 10 steps):
  HF job `6a524384effc02a91cbd98c6` COMPLETED (~11 min, a10g-small). Clean Hub data
  (7592 refs after drop-roundups — **pre-curated-path**), wandb project `amlk-dictalm2-instruct`,
  run `2026-07-11_dictalm2-instruct_qlora_whole_1ep_smoke`. Finite loss 1.04→0.52 (avg 0.779),
  eval ~1.18–1.30, Hebrew constraint 27848 tokens, adapter + 5+5 preds pushed to
  `avreymi/amlk-dictalm2-instruct-smoke`.
- 2026-07-26 **curated HeSum training data** built locally and **pushed to Hub**: 5854 records →
  train/val/test = 4683/585/586, columns validated for SFT; re-download of Hub train verified.
  Local: `outputs/data/curated/final_clean_hesum.json` → `outputs/data/processed/whole`.
  Hub: private `avreymi/amlk-training-data` (replaces pre-curated clean splits). Replaces the
  old IAHLT+raw-HeSum path entirely.
