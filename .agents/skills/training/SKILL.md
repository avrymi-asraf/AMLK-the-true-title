---
name: training
description: AMLK training process — fine-tune Qwen3-2B for Hebrew summarization (qlora|lora|full) locally or on HF Jobs, with wandb logging and adapter push to the Hub.
---

# AMLK Training Process

One script trains all three regimes the paper compares: `training/train.py`, selected with
`--method qlora|lora|full`. The methods differ only by the small `METHOD_PRESETS` deltas in
`training/config.py` (quantize, use_lora, batch size, lr); everything else is shared. The
self-contained `training/train_hf_job.py` is the same logic packaged for HuggingFace Jobs.

## How a run is wired

1. Load the processed Arrow splits from `outputs/data/processed/<variant>/{train,val}`.
   Each row is a `(prompt, completion)` pair, so SFT uses `completion_only_loss=True`
   (loss on the summary only — the model learns to generate the summary, not restate the article).
2. Load Qwen3-2B: 4-bit NF4 (`qlora`) or bf16 (`lora`, `full`).
3. `SFTTrainer` with `peft_config` (LoRA for qlora/lora, `None` for full) +
   `processing_class=tokenizer`.
4. wandb logs every step (project `amlk-hebrew-summarization`, group = variant).
5. Save the adapter to `--output`; `--push-to-hub` / `--submit-hf` push to the Hub.

After training, the **HF Jobs script also generates the test-set predictions** (fine-tuned via
the adapter, zero-shot base via PEFT `disable_adapter()`) and pushes
`predictions-finetuned.jsonl` / `predictions-base.jsonl` to the model repo — so no separate
GPU inference job is needed. Metrics are computed locally (CPU/API) by `evaluation/`.

## Run it (always `python -m` from repo root, so package imports resolve)

> **Do NOT train or run model inference on the local machine — it freezes (8 GB GPU).**
> Everything model-related runs on HF Jobs. See memory `no-local-gpu-runs`.

```bash
source .env && source .venv/bin/activate

# Verify the whole cloud pipeline cheaply first, then the real run:
python -m training.train --submit-hf --hf-user avreymi --smoke-test # a10g-small, 10 steps, ~$0.05
python -m training.train --submit-hf --hf-user avreymi              # a10g-large, 6h, 1-epoch QLoRA
```

`train.py` does keep a local code path (`--method ... --output ...`) for machines with a real
GPU, but it is not used here.

## trl 1.6.0 / transformers 5.x API (verified — do not regress)

- `max_length=` (NOT `max_seq_length=`), `processing_class=tokenizer` (NOT `tokenizer=`).
- `model_init_kwargs` is a **`SFTConfig`** field, not a `SFTTrainer` arg. We sidestep it by
  loading the model object ourselves and passing `model=<object>` — the remote script does the same.
- `completion_only_loss=True` requires `prompt`/`completion` columns (produced by `data/preprocess.py`).

## HF Jobs submission — the hard rule

`run_uv_job` uploads **only the script file**; the repo is NOT available on the job server.
Every value the job needs is passed as an explicit env var (`METHOD`, `VARIANT`, `DATASET_REPO`,
`OUTPUT_REPO`, `WANDB_PROJECT`, `SMOKE_TEST`). `--submit-hf` also uploads the processed splits to
`avreymi/amlk-training-data[-<variant>]` first, then the job `snapshot_download`s them.
Secrets passed: `HF_TOKEN` and `WANDB_API_KEY` — the wandb key is read from `~/.netrc` when it
isn't in the environment (`wandb_api_key()` in `train.py`), so wandb logs from the cloud.

- **`run_uv_job(script=...)` takes a file PATH, not the script contents** (huggingface_hub ≥1.17;
  it `stat()`s the argument to decide whether to upload it). Passing `script_path.read_text()`
  raises a `File name too long` error. Pass `str(script_path)`.
- **Secrets must be the real values via the Python API.** `secrets={"HF_TOKEN": "$HF_TOKEN"}`
  only works on the `hf jobs` CLI (the shell expands it). Through `HfApi.run_uv_job` the `"$HF_TOKEN"`
  string is passed literally → the job gets a bogus token → **401 on the private dataset
  `snapshot_download`**, then `FileNotFoundError: data/train`. Pass the actual token string.

## Monitoring

```bash
hf jobs ps                       # list recent jobs
hf jobs logs <job-id>            # snapshot; add -f to stream
hf jobs inspect <job-id>
hf jobs cancel <job-id>
```
Monitor URL: `https://huggingface.co/jobs/avreymi/<job-id>`. wandb dashboard:
`https://wandb.ai/<entity>/amlk-hebrew-summarization`.

## Lessons (keep these true)

- `bf16=True` and `bnb_4bit_compute_dtype=bfloat16` must agree — mixing fp16/bf16 causes silent
  precision issues on A10G/T4.
- **Set `per_device_eval_batch_size=1`.** It defaults to 8; at `max_length=2048` the eval step tries
  to allocate ~15 GB and OOMs the 24 GB A10G even though training (batch 2 + gradient checkpointing)
  fits fine. Training getting to step 5 then crashing at the first eval is this bug.
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in the job to reduce fragmentation.
- **HeSum articles are long (median ~2500 tokens, p90 ~6400).** `data/preprocess.py` truncates
  each article to `MAX_LENGTH-256` tokens so the summary survives — otherwise right-truncation cuts
  the completion and `completion_only_loss` produces `eval_loss=nan` (and learns from a minority of rows).
- The Hub adapter is the **LoRA adapter only** (not merged into the base). Evaluation loads
  base 4-bit + `PeftModel.from_pretrained(model, adapter)` — see `evaluation/predict.py`.
- All training and model inference run on HF Jobs — never on the local 8 GB GPU (it freezes).
- For wandb specifics (axis alignment, downloading curves), see the global `wandb-for-trl` skill.

## Completed runs

<!-- Append run IDs + wandb URLs here as runs complete. -->
- 2026-06-12 smoke (qlora-whole, 10 steps): HF job `6a2bc7558806fe3ef5852983` COMPLETED —
  verified train → eval → predict (finetuned + zero-shot base via `disable_adapter`) → push to
  `avreymi/amlk-qwen3-2b-sft`. Surfaced + fixed: secret literal-401, run_uv_job path, eval-batch OOM,
  English-output prompt, and the long-article truncation→nan-eval-loss bug.
- 2026-06-12 full (qlora-whole, 1 epoch): HF job `6a2bc974822d86c524179991` (a10g-large, 6h).
