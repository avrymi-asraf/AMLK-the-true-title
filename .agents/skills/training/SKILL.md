---
name: training
description: AMLK training process — fine-tune dicta-il/dictalm2.0-instruct for Hebrew summarization (qlora|lora|full) on HF Jobs or Colab, with informative wandb names, 1-epoch runs, and mid-run Hub checkpoint pushes.
---

# AMLK Training Process

One script trains all three regimes the paper compares: `training/train.py`, selected with
`--method qlora|lora|full`. The methods differ only by the small `METHOD_PRESETS` deltas in
`training/config.py`. The self-contained `training/train_hf_job.py` is the **sole remote
training body** for both backends:
- `--submit-hf` → HuggingFace Jobs (`run_uv_job`)
- `--submit-colab` → Google Colab (`training/colab_submit.py` + `scripts/colab_train_entry.py`)

**Primary data path:** curated HeSum → HuggingFace Arrow training splits.
`data.download` resolves `final_clean_hesum.json` from (1)
`outputs/data/curated/` or (2) `data_curation/artifacts/` (this worktree). Then:

```bash
python -m data.download                 # → curated_records.jsonl
python -m data.preprocess --variant whole --force   # → processed/whole/{train,val,test}
```

**E4 raw-vs-curated** (secondary path — see `docs/e4-raw-vs-curated-training-plan.md`):

```bash
python -m data.download_raw --force     # leakage-safe raw pool → outputs/data/raw/raw_records.jsonl
python -m data.preprocess --variant whole --force \
  --output outputs/data/processed/e4cur
python -m data.preprocess --input outputs/data/raw/raw_records.jsonl --variant whole --force \
  --test-from outputs/data/processed/e4cur --output outputs/data/processed/e4raw
# Upload each dir to Hub (amlk-training-data-e4cur / amlk-training-data-raw), then:
python -m training.train --submit-hf --hf-user avreymi --method lora \
  --dataset-repo avreymi/amlk-training-data-raw --output-repo avreymi/amlk-e4-raw \
  --skip-data-upload --test-subset 120 --skip-base-arm --run-tag e4-raw
# …same for e4cur / amlk-e4-curated
python -m scripts.e4_score --raw <raw-preds.jsonl> --curated <cur-preds.jsonl> --limit 120
```

Preprocess builds columns `text, summary, source, prompt, completion` (raw instruction
prompts; chat-wrap later), truncates articles to `MAX_LENGTH-256`, splits 80/10/10, and
runs `validate_train_dataset` before save. Default **1 epoch** per train run. Hebrew decode
constraint is always on at generation. **Decode defaults: no repetition_penalty /
no_repeat_ngram_size** (1.0 / 0) — penalties suppress article vocabulary in summarization.

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
   - Checkpoints → `OUTPUT_DIR` (`/data/output` on HF Jobs bucket; `/content/amlk-output`
     on Colab — same-session only). Cross-session durability is always Hub.
   - `hub_strategy="all_checkpoints"` → each `checkpoint-N/` (optimizer + trainer_state +
     adapter) lands on Hub mid-run so `--resume-from` works after SIGTERM. Full-run
     `save_steps=50` / `eval_steps=50`. SoftHubSFTTrainer soft-fails Hub I/O so a push
     OSError cannot kill the epoch (local OUTPUT_DIR still has the save).
   - Predictions written under OUTPUT_DIR and uploaded periodically (soft-fail too).
   - Full-run HF Jobs timeout default **8h** (7B QLoRA worst-case ~5.8h at smoke step-time).

## Run it (always `python -m` from repo root)

> **Do NOT train or run model inference on the local machine — it freezes (8 GB GPU).**
> Everything model-related runs on HF Jobs or Colab.

```bash
source .env && source .venv/bin/activate

# Data (main path: curated — no IAHLT merge; E4 raw arm is separate, see above)
python -m data.download
python -m data.preprocess --variant whole --force

# Hub already has curated whole data (2026-07-26). Re-upload only after rebuild:
# omit --skip-data-upload (or after MAX_LENGTH / preprocess changes).
# Default --method is lora (measured ~1.37x faster than qlora on 10-step smokes).
python -m training.train --submit-hf --hf-user avreymi --smoke-test \
  --skip-data-upload --output-repo avreymi/amlk-dictalm2-instruct-smoke

python -m training.train --submit-hf --hf-user avreymi --skip-data-upload
python -m training.train --submit-hf --hf-user avreymi --inference-only

# Colab path (same train_hf_job body; OUTPUT_DIR=/content/amlk-output). Prefer qlora on T4.
# Dry-run (no VM): add --colab-dry-run. First smoke uses durable session + always stop.
python -m training.train --submit-colab --hf-user avreymi --method qlora --smoke-test \
  --skip-data-upload --skip-base-arm \
  --output-repo avreymi/amlk-dictalm2-instruct-colab-smoke --run-tag colab
# Optional: --colab-mode durable|run|auto  --colab-gpu T4  --colab-session NAME

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
- `source` is `hesum-curated` on the main path; E4-RAW uses `hesum-raw` (same columns otherwise).
- Hub: private `{hf_user}/amlk-training-data` (or `-lead`/`-body`). **As of 2026-07-26**,
  `avreymi/amlk-training-data` holds curated whole splits (4683/585/586, `hesum-curated`).
  E4 Hub datasets: `amlk-training-data-raw`, `amlk-training-data-e4cur` (upload folder manually
  then train with `--dataset-repo … --skip-data-upload`).
  Upload via `train.py --submit-hf` or a one-shot `HfApi.upload_folder` of
  `outputs/data/processed/<variant>/`.
- `completion_only_loss=True` requires `prompt`/`completion` columns.
- Curated input: `outputs/data/curated/final_clean_hesum.json` (or
  `data_curation/artifacts/final_clean_hesum.json`) rows
  `{hesum_id, text, headline}` (headline becomes `summary`/`completion`).
  E4-RAW input: `outputs/data/raw/raw_records.jsonl` from `data.download_raw`.

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

## Colab submission — additive path

- Same env payload as HF Jobs via `training.train.build_job_env` / `prepare_remote_submit`.
- Extra env: `OUTPUT_DIR=/content/amlk-output`, `DATA_DIR=/content/amlk-data`.
- Secrets: upload local `.env` to `/content/.env` (not userdata). Always
  `colab --auth=oauth2` before the subcommand; isolate `--config /tmp/amlk-colab-<tag>.json`.
- Smoke default mode **durable**: `new -s NAME --gpu T4` → upload → high-timeout `exec` →
  always `stop -s NAME` and verify `sessions` empty.
- Steady-state: `colab run --gpu T4` without `--keep` (self-clean; Hub-only artifacts).
- Prefer **`--method qlora`** on T4 (16 GB); bf16 lora often OOMs for 7B.
- Pin `jupyter-kernel-client==0.15.0` if KernelClient import breaks after CLI upgrade.
- Production full-epoch still prefers HF Jobs until Colab is proven end-to-end.
- See `.agents/skills/colab-cli/SKILL.md` for agent safety rules.

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
- Training data is **curated HeSum** for the main pipeline. E4-RAW is a deliberate experiment
  arm via `data.download_raw` + shared curated test (`--test-from`); do not reintroduce
  raw biunlp/HeSum + IAHLT merge as the default train path.
- Never re-enable default `repetition_penalty=1.2` / `no_repeat_ngram_size=3` for decode —
  measured to hurt faithfulness ~1.4–1.5 points (see Decoding Configuration / E4 plan §1.1).
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
