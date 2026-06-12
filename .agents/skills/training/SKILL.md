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
python -m training.train --submit-hf --hf-user avreymi --inference-only  # regen predictions from pushed adapter (a10g-small, 1h)
```

> **Cost note (2026-06-12 post-mortem):** a10g-small and a10g-large have the **same 24 GB
> A10G GPU** — large only adds vCPUs/RAM ($1.50/h vs $1.00/h). For this GPU-bound 2B job,
> prefer a10g-small for full runs, and prefer `--method lora` (bf16) over qlora — the 2B
> model doesn't need quantization on 24 GB and nf4 dequant slows every step ~20-40%.
> Full analysis: `docs/2026-06-12-qlora-training-job-postmortem.md`.

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
- **Qwen3-2B is a HYBRID-attention model** (`model_type: qwen3_5`): 24 text layers = 18
  linear-attention (Gated DeltaNet: `linear_attn.{in_proj_qkv,in_proj_z,out_proj,...}`) + 6
  full-attention (`self_attn.{q,k,v,o}_proj`), plus an unused vision tower. The current LoRA
  `target_modules=["q_proj","k_proj","v_proj","o_proj"]` therefore attaches to **only 6 of 24
  layers** (1.48 M trainable params, 0.07% — the 2.96 MB adapter is the tell). Before the next
  training run, extend `target_modules` to include `in_proj_qkv`, `in_proj_z`, `out_proj`,
  `gate_proj`, `up_proj`, `down_proj` (≈15.6 M params) and A/B it at mini-test scale. Do NOT
  use PEFT's `"all-linear"` — it can catch the vision tower and `mtp` head. See post-mortem §5.1.
- The same hybrid layers have an optimized kernel path that needs `flash-linear-attention` and
  `causal-conv1d` installed — they are missing from the job deps, so transformers logs a
  "fast path is not available" warning and falls back to slow torch kernels. Add them to the
  PEP 723 block and benchmark in a mini-test before the probe runs.
- **Cloud-job crash economics:** any artifact not yet pushed has zero value when the timeout
  hits. `train_hf_job.py` pushes each `predictions-*.jsonl` immediately after its loop (since
  d8bf268) — keep it that way. Generation uses `max_new_tokens=256` (p99 of reference summaries
  is 187 tokens; the old 128 cap truncated ~9%).
- For wandb specifics (axis alignment, downloading curves), see the global `wandb-for-trl` skill.

## Completed runs

<!-- Append run IDs + wandb URLs here as runs complete. -->
- 2026-06-12 smoke (qlora-whole, 10 steps): HF job `6a2bc7558806fe3ef5852983` COMPLETED —
  verified train → eval → predict (finetuned + zero-shot base via `disable_adapter`) → push to
  `avreymi/amlk-qwen3-2b-sft`. Surfaced + fixed: secret literal-401, run_uv_job path, eval-batch OOM,
  English-output prompt, and the long-article truncation→nan-eval-loss bug.
- 2026-06-12 full (qlora-whole, 1 epoch): HF job `6a2bc974822d86c524179991` (a10g-large, 6h) —
  **training succeeded** (500 steps, 3h54m, eval_loss 1.777; adapter on `avreymi/amlk-qwen3-2b-sft`)
  but the job was CANCELED at the 6h timeout inside an unbatched, push-at-the-end prediction loop,
  losing all predictions. Post-mortem: `docs/2026-06-12-qlora-training-job-postmortem.md`.
- 2026-06-12 mini (qlora-whole, 100 examples / 5 epochs): HF job `6a2bcd887c68f455eff13113` (a10g-small, 1h) — validation run; check wandb for real loss curves and finite eval_loss.
- 2026-06-12 inference-only rerun #1: `6a2c2088871c005b5352b4ac` — canceled at ~24m; its 30-min
  timeout could not fit 2,000 generations and it still pushed only at the end.
- 2026-06-12 inference-only rerun #2: `6a2c26ea7c68f455eff13d1c` (a10g-small, 1h) — patched
  script: incremental per-file pushes, `max_new_tokens=256`, fixed progress prints.
