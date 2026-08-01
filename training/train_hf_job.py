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
Pipeline step 3 (remote variant): self-contained fine-tuning of
dicta-il/dictalm2.0-instruct (qlora|lora|full), run on HuggingFace Jobs.
Submitted inline by training/train.py --submit-hf; never run directly. All
settings arrive as environment variables (the repo is NOT uploaded with the script).

Hyperparameters come from TRAIN_CONFIG / LORA_CONFIG JSON (serialized by train.py
from METHOD_PRESETS) so --method cannot silently use wrong batch/lr. Default
method when env is missing is **lora** (matches train.py CLI).
Train and serve both apply the model chat template (C0); generation tokenizes with
add_special_tokens=False to avoid double-BOS (C1).

Stability (so a long run is not lost on crash/timeout):
  1. Checkpoints write to /data/output — the per-job bucket volume that survives
     infra restarts of the same job; trainer.train(resume_from_checkpoint=True)
     picks them up automatically.
  2. hub_strategy="all_checkpoints" pushes each checkpoint-* folder (optimizer +
     trainer_state + adapter) mid-run so --resume-from works after SIGTERM. Soft
     Hub push: I/O errors on push never kill training (local /data/output remains).
  3. Predictions files are uploaded periodically during generation (soft-fail too).
  4. Cross-job Hub resume: RESUME_FROM=auto|checkpoint-N downloads a *full*
     Trainer checkpoint (optimizer+trainer_state) from OUTPUT_REPO into
     /data/output before train — adapter-only Hub root is not enough.

Execution environment: ephemeral HuggingFace Jobs GPU container.
"""
import json
import os
import re
import statistics
import time
import warnings
from datetime import date
from pathlib import Path

import torch
import wandb
from datasets import load_from_disk
from huggingface_hub import HfApi, snapshot_download
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

METHOD = os.environ.get("METHOD", "lora")
VARIANT = os.environ.get("VARIANT", "whole")
DATASET_REPO = os.environ["DATASET_REPO"]
OUTPUT_REPO = os.environ["OUTPUT_REPO"]
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
MINI_TEST = os.environ.get("MINI_TEST", "0") == "1"
INFERENCE_ONLY = os.environ.get("INFERENCE_ONLY", "0") == "1"
# Cross-job resume from a full Trainer checkpoint on OUTPUT_REPO:
#   "" (default) → only /data/output local resume (same job restart)
#   "auto"       → highest checkpoint-* with optimizer+trainer_state on Hub
#   "checkpoint-N" → that exact dir
RESUME_FROM = (os.environ.get("RESUME_FROM") or "").strip()
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

# Resolved presets from train.py (METHOD_PRESETS). Fallbacks match the **lora** preset
# (CLI default) so a hand-fired job without TRAIN_CONFIG still matches train.py.
_DEFAULT_TRAIN = {
    "quantize": False,
    "use_lora": True,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
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
# Inference-only always 4-bit: bf16 7B + gen batch left no KV headroom on a10g (22 GB) —
# dual-arm OOM'd on first batch (job 6a678afb, GEN_BATCH_SIZE=8). Matches infer.py quantize=True.
if INFERENCE_ONLY:
    quantize = True

# --- Hub full-Trainer resume (twin of training/resume.py; this script cannot import the repo) ---
_CKPT_NAME_RE = re.compile(r"^checkpoint-(\d+)$")
_REQUIRED_CKPT_FILES = frozenset({"trainer_state.json", "optimizer.pt", "scheduler.pt"})
_ADAPTER_OR_MODEL = frozenset({
    "adapter_model.safetensors", "adapter_model.bin",
    "model.safetensors", "pytorch_model.bin",
})


def _checkpoint_step(name: str) -> int:
    m = _CKPT_NAME_RE.match(name)
    return int(m.group(1)) if m else -1


def _is_full_trainer_checkpoint_files(filenames: set[str]) -> bool:
    return _REQUIRED_CKPT_FILES.issubset(filenames) and bool(filenames & _ADAPTER_OR_MODEL)


def _hub_full_trainer_checkpoints(repo_files: list[str]) -> list[str]:
    by_ckpt: dict[str, set[str]] = {}
    for path in repo_files:
        parts = path.split("/")
        if len(parts) < 2 or not parts[0].startswith("checkpoint-"):
            continue
        by_ckpt.setdefault(parts[0], set()).add(parts[-1])
    names = [n for n, files in by_ckpt.items() if _is_full_trainer_checkpoint_files(files)]
    return sorted(names, key=_checkpoint_step)


def _pick_resume_checkpoint(names: list[str], prefer: str) -> str | None:
    if not names:
        return None
    key = (prefer or "auto").strip()
    if key in ("", "auto"):
        return names[-1]
    if key in names:
        return key
    raise ValueError(
        f"Resume checkpoint {key!r} not among full Trainer checkpoints on Hub: {names}"
    )


def _materialize_hub_checkpoint(repo_id: str, checkpoint_name: str, output_dir: str) -> str:
    dest = Path(output_dir) / checkpoint_name
    if dest.is_dir() and _is_full_trainer_checkpoint_files(
        {p.name for p in dest.iterdir() if p.is_file()}
    ):
        print(f"Resume checkpoint already on disk: {dest}")
        return str(dest)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Downloading full Trainer {checkpoint_name} from {repo_id} → {output_dir}/...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{checkpoint_name}/*"],
        local_dir=output_dir,
    )
    if not dest.is_dir() or not _is_full_trainer_checkpoint_files(
        {p.name for p in dest.iterdir() if p.is_file()}
    ):
        raise FileNotFoundError(
            f"{dest} is not a full Trainer checkpoint after Hub download "
            f"(need trainer_state.json + optimizer.pt + adapter weights)"
        )
    print(f"Resume checkpoint ready: {dest}")
    return str(dest)


class SoftHubSFTTrainer(SFTTrainer):
    """SFTTrainer that never aborts training because a Hub checkpoint push failed.

    E4 curated full run died on OSError during mid-run hub push (storage/network I/O)
    even though /data/output already held the checkpoint. Hub is best-effort; local
    bucket checkpoints + a later soft final push are enough to resume or finish.
    """

    def _push_from_checkpoint(self, checkpoint_folder: str) -> None:
        try:
            super()._push_from_checkpoint(checkpoint_folder)
        except Exception as exc:
            print(
                f"WARNING: Hub checkpoint push failed for {checkpoint_folder} "
                f"({type(exc).__name__}: {exc}); training continues. "
                f"Local checkpoint remains under {self.args.output_dir}."
            )

    def push_to_hub(self, *args, **kwargs):
        try:
            return super().push_to_hub(*args, **kwargs)
        except Exception as exc:
            print(
                f"WARNING: final Hub push failed ({type(exc).__name__}: {exc}); "
                f"local weights remain under {self.args.output_dir}."
            )
            return None


class StepTimeCallback(TrainerCallback):
    """Median seconds per optimizer step, warmup excluded — the number that makes two runs
    comparable on GPU cost.

    `train_runtime` cannot serve this purpose: it folds in container start, the first-step
    CUDA/kernel warmup, and every eval pass, and it scales with dataset size — so a 10-step
    smoke of method A and one of method B are not comparable through it. Median-of-steps on
    the *same* 50-example smoke subset is, which is what gates the qlora-vs-bf16-lora choice
    (a 1.3-2x effect, resolvable at 10 steps; a ~20% padding effect is not, and is instead
    computed offline from the token-length distribution — see AGENTS.md).
    """

    def __init__(self, full_epoch_steps: int = 0, warmup_steps: int = 2):
        # full_epoch_steps: optimizer steps a full (untruncated) epoch would take, so a 10-step
        # smoke can project the real run's hours instead of only reporting s/step.
        self.full_epoch_steps = full_epoch_steps
        self.warmup_steps = warmup_steps
        self.step_times: list[float] = []
        self._started: float | None = None

    def on_step_begin(self, args, state, control, **kwargs):
        self._started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self._started is not None:
            self.step_times.append(time.perf_counter() - self._started)
            self._started = None

    def report(self) -> dict:
        """Print and return the timing summary; {} if too few steps to be meaningful."""
        timed = self.step_times[self.warmup_steps:]
        if not timed:
            print(f"Step timing: only {len(self.step_times)} step(s) — no median reported.")
            return {}
        median = statistics.median(timed)
        summary = {
            "step_time_median_s": round(median, 3),
            "step_time_min_s": round(min(timed), 3),
            "step_time_n": len(timed),
            "projected_epoch_h": round(median * self.full_epoch_steps / 3600, 2)
            if self.full_epoch_steps else None,
            # Peak memory turns "it did not OOM" into a number. A smoke only proves the regime
            # survives the longest example *it happened to see* (the 50-example subset tops out
            # at 3926 tokens vs the full split's 4031), so headroom cannot be claimed without this.
            "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            if torch.cuda.is_available() else None,
        }
        print(f"Step timing (warmup {self.warmup_steps} excluded): "
              f"median {median:.2f}s over {len(timed)} steps (min {min(timed):.2f}s)")
        if summary["peak_mem_gb"]:
            print(f"  Peak CUDA memory: {summary['peak_mem_gb']} GB")
        if summary["projected_epoch_h"]:
            print(f"  Projected full epoch (training only, no eval/generation): "
                  f"{summary['projected_epoch_h']}h over {self.full_epoch_steps} optimizer steps")
        return summary

mode = "inference-only" if INFERENCE_ONLY else f"train+infer  Method={METHOD}"
print(f"Mode: {mode}  Variant: {VARIANT}  Smoke: {SMOKE_TEST}  Mini: {MINI_TEST}")
print(f"Base model: {MODEL_ID}")
print(f"Dataset: {DATASET_REPO}  ->  Output: {OUTPUT_REPO}")
if INFERENCE_ONLY and quantize:
    print("Inference-only: 4-bit base load (VRAM headroom for dual-arm gen on a10g)")
if RESUME_FROM and not INFERENCE_ONLY:
    print(f"Hub resume: RESUME_FROM={RESUME_FROM!r} (full Trainer checkpoint from OUTPUT_REPO)")
print(f"Epochs: {EPOCHS}")
print(f"Train config: quantize={quantize} use_lora={use_lora} "
      f"batch={TRAIN_CFG['per_device_train_batch_size']} "
      f"accum={TRAIN_CFG['gradient_accumulation_steps']} lr={TRAIN_CFG['learning_rate']}")
# Decode budget for dual-arm generation. Default 128 (was 256) — cost lever; see
# training/config.py DEFAULT_MAX_NEW_TOKENS. train.py passes MAX_NEW_TOKENS via env.
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS") or 128)
print(f"max_new_tokens={MAX_NEW_TOKENS} (post-train dual-arm decode budget)")
# Decode defaults: NO repetition penalty. HF applies repetition_penalty over the whole
# sequence (prompt included); at 1.2 every word already in the ~3.8k-token article is
# suppressed, so the model emits near-miss entity names. Measured +1.4–1.5 faithfulness
# on a 120-article subset when 1.2/ngram-3 → 1.0/0. Degeneration is handled by min_new_tokens,
# explicit eos, max_new_tokens=128, and the stop-cue prompt. Env overrides still work.
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY") or 1.0)
NO_REPEAT_NGRAM_SIZE = int(os.environ.get("NO_REPEAT_NGRAM_SIZE") or 0)
print(f"decode: repetition_penalty={REPETITION_PENALTY} "
      f"no_repeat_ngram_size={NO_REPEAT_NGRAM_SIZE}")
print(f"wandb: {WANDB_PROJECT} / {WANDB_RUN_NAME}")

local_data = Path("./data")
snapshot_download(repo_id=DATASET_REPO, repo_type="dataset", local_dir=str(local_data))
train_ds = load_from_disk(str(local_data / "train"))
val_ds = load_from_disk(str(local_data / "val"))
test_ds = load_from_disk(str(local_data / "test"))
print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
# Kept before the smoke/mini truncation below so a 10-step smoke can project the real epoch.
FULL_TRAIN_SIZE = len(train_ds)

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
    # Eval on a fixed 100-example slice. Eval runs at per_device_eval_batch_size=1 (OOM at
    # higher, see below), so it is pure added GPU time; with ~293 optimizer steps on the
    # curated 4683-example train set, 100 examples is enough signal for load_best_model_at_end.
    val_ds = val_ds.select(range(min(100, len(val_ds))))

# Improvement-loop knobs (docs/training-improvement-notebook.md). MAX_TRAIN_EXAMPLES keeps
# cheap arms matched on optimizer steps; TEST_SUBSET_N generates only the fixed judged subset
# (deterministic twin of evaluation.improve_eval.subset_indices — keep the seed in sync), which
# is what makes a ~120-example arm cost minutes of GPU instead of hours.
MAX_TRAIN_EXAMPLES = int(os.environ.get("MAX_TRAIN_EXAMPLES") or 0)
TEST_SUBSET_N = int(os.environ.get("TEST_SUBSET_N") or 0)
TEST_SUBSET_SEED = int(os.environ.get("TEST_SUBSET_SEED") or 1234)
SKIP_BASE_ARM = os.environ.get("SKIP_BASE_ARM", "0") == "1"

if MAX_TRAIN_EXAMPLES and MAX_TRAIN_EXAMPLES < len(train_ds):
    train_ds = train_ds.select(range(MAX_TRAIN_EXAMPLES))
    print(f"[Improve] Capped train to {len(train_ds)} examples")
if TEST_SUBSET_N and TEST_SUBSET_N < len(test_ds):
    import random as _random
    idx = sorted(_random.Random(TEST_SUBSET_SEED).sample(range(len(test_ds)), TEST_SUBSET_N))
    test_ds = test_ds.select(idx)
    print(f"[Improve] Test restricted to the fixed judged subset: {len(test_ds)} examples "
          f"(seed={TEST_SUBSET_SEED})")

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


# attn_implementation is pinned rather than left to the transformers default: at max_length=4096
# eager attention is a large multiplier on step time (and therefore on billed GPU minutes).
# Not flash_attention_2 — that needs a source build inside the UV job for marginal gain over SDPA.
# Single-GPU map ({"": 0}) avoids multi-shard surprises on a10g-small.
load_kwargs = dict(device_map={"": 0}, attn_implementation="sdpa")
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
if quantize:
    # Fail loud if bnb silently didn't quantize (bf16 7B ~14–19 GiB looks "loaded" but
    # leaves no headroom for generate — seen on inference jobs 6a678afb / 6a678ef5).
    n_4bit = sum(
        1 for m in model.modules()
        if "4bit" in m.__class__.__name__.lower() or "4Bit" in m.__class__.__name__
    )
    print(f"4-bit Linear modules after load: {n_4bit}")
    if n_4bit == 0:
        raise RuntimeError(
            "quantize=True but no 4-bit modules found — bitsandbytes path failed"
        )
if torch.cuda.is_available():
    print(f"CUDA mem after base load: "
          f"{torch.cuda.memory_allocated()/1024**3:.2f} GiB allocated")

if INFERENCE_ONLY:
    print(f"Loading adapter from {OUTPUT_REPO}...")
    trained_model = PeftModel.from_pretrained(model, OUTPUT_REPO).eval()
    device = next(trained_model.parameters()).device
    use_lora = True  # adapter is always a LoRA adapter
    if torch.cuda.is_available():
        print(f"CUDA mem after adapter: "
              f"{torch.cuda.memory_allocated()/1024**3:.2f} GiB allocated")
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

    # Mini: log every step; smoke: 10 steps; full 1-epoch: eval+save every 50 steps → ~5–6
    # mid-run full Trainer checkpoint-* folders on Hub (hub_strategy=all_checkpoints) over the
    # ~293 optimizer steps of a 4683-example epoch — denser resume surface after SIGTERM.
    # load_best_model_at_end requires save_steps to be a multiple of eval_steps, so these two
    # move together — the eval cost lever is the val slice size above, not the cadence.
    if MINI_TEST:
        n_epochs, max_steps_cfg = 1, -1
        log_steps, eval_steps_cfg, save_steps_cfg = 1, 5, 20
    elif SMOKE_TEST:
        n_epochs, max_steps_cfg = 1, 10
        log_steps, eval_steps_cfg, save_steps_cfg = 5, 5, 5
    else:
        n_epochs, max_steps_cfg = EPOCHS, -1
        log_steps, eval_steps_cfg, save_steps_cfg = 10, 50, 50

    # /data is the bucket run_uv_job auto-mounts — survives infra restarts of this job.
    output_dir = "/data/output"
    print(f"Stability: checkpoints → {output_dir} (bucket resume)")
    print(
        f"Stability: hub_strategy=all_checkpoints → {OUTPUT_REPO} every "
        f"{save_steps_cfg} steps (full Trainer ckpt; soft-fail Hub I/O)"
    )

    # Cross-job resume: pull a full Trainer checkpoint from the Hub into /data/output.
    # Adapter-only root files on OUTPUT_REPO are not enough (no optimizer/trainer_state).
    if RESUME_FROM:
        api_hub = HfApi()
        repo_files = api_hub.list_repo_files(OUTPUT_REPO, repo_type="model")
        available = _hub_full_trainer_checkpoints(repo_files)
        if not available:
            raise RuntimeError(
                f"RESUME_FROM={RESUME_FROM!r} but {OUTPUT_REPO} has no full Trainer "
                f"checkpoint-* (need trainer_state.json + optimizer.pt + adapter). "
                f"Upload one with: python -m training.train --push-resume-checkpoint DIR "
                f"--output-repo {OUTPUT_REPO}"
            )
        chosen = _pick_resume_checkpoint(available, RESUME_FROM)
        print(f"Hub full Trainer checkpoints: {available} → resuming from {chosen}")
        _materialize_hub_checkpoint(OUTPUT_REPO, chosen, output_dir)

    # When finishing a killed run from Hub, keep *last* weights after the remaining steps.
    # load_best_model_at_end would otherwise reload the mid-run best (e.g. step 100) and
    # discard the final stretch of the epoch if no later eval/save lands on a save_steps multiple.
    load_best = not bool(RESUME_FROM)

    sft_config = SFTConfig(
        output_dir=output_dir,
        push_to_hub=True,
        hub_model_id=OUTPUT_REPO,
        # all_checkpoints (not every_save): every_save only copies adapter weights to the
        # repo root; resume needs checkpoint-N/{optimizer,trainer_state,adapter}. SoftHub
        # wraps push so Hub I/O OSError cannot kill the epoch (E4 curated death mode).
        hub_strategy="all_checkpoints",
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
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bool(TRAIN_CFG["bf16"]),
        gradient_checkpointing=True,
        # Cost lever: batches are padded to their longest member, and article lengths are wide
        # (mean 1981 tokens, p50 1887, max 4031). Length-grouped sampling pays ~the mean instead
        # of E[max of batch]: measured 20% fewer padded tokens at micro-batch 2, no effect on
        # what is learned. NOTE the API — transformers 5 removed the `group_by_length` bool in
        # favour of this enum (default "random"); passing the old kwarg is a hard error.
        # Worth nothing at micro-batch 1 (a batch of one has no padding) — see the lora preset.
        train_sampling_strategy="group_by_length",
        # TRL auto-appends EOS to each completion; completion_only_loss keeps it in the mask.
        completion_only_loss=True,
        max_length=int(TRAIN_CFG["max_length"]),
        report_to="wandb",
        run_name=WANDB_RUN_NAME,
    )

    trainer = SoftHubSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    effective_batch = (int(TRAIN_CFG["per_device_train_batch_size"])
                       * int(TRAIN_CFG["gradient_accumulation_steps"]))
    step_timer = StepTimeCallback(
        full_epoch_steps=-(-FULL_TRAIN_SIZE // effective_batch),   # ceil division
    )
    trainer.add_callback(step_timer)

    if use_lora:
        # Print so a base-model swap can never silently regress LoRA layer coverage.
        trainer.model.print_trainable_parameters()

    resume_from_checkpoint = None
    if os.path.isdir(output_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(output_dir)
    ):
        resume_from_checkpoint = True
        print(f"Found existing checkpoint(s) in {output_dir} — resuming training.")
    elif RESUME_FROM:
        raise RuntimeError(
            f"RESUME_FROM={RESUME_FROM!r} set but no checkpoint-* under {output_dir}"
        )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Cost accounting: median s/step (warmup excluded) is what makes runs comparable. Mirrored
    # into the wandb run summary so two smokes can be diffed in the dashboard, not just the logs.
    timing = step_timer.report()
    for key, value in timing.items():
        if value is not None and wandb.run is not None:
            wandb.run.summary[key] = value

    # Resume tails often end between save_steps; run a final eval + save so metrics and Hub
    # get the completed-epoch weights (not only the last all_checkpoints save).
    if RESUME_FROM:
        print("Post-resume final eval + save (end of remaining steps)...")
        metrics = trainer.evaluate()
        print(f"Final eval: {metrics}")
        trainer.save_model(output_dir)

    # Final Hub commit with the best (or last) weights — mid-run saves already pushed
    # via all_checkpoints (soft-fail inside SoftHubSFTTrainer.push_to_hub).
    trainer.push_to_hub()
    print(f"Final adapter push complete (or soft-failed) → {OUTPUT_REPO}")

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


# Cost lever: wider batches cut dual-arm wall-clock on ~586x2 test preds. Reality on a10g
# (22 GB): batch 8 bf16 OOM'd (job 6a678afb); batch 16 under 4-bit still OOM'd on the first
# left-padded long prefill (job 6a678ef5, ~19 GiB resident + 3.3 GiB alloc). Keep gen at 1
# so padding never multiplies a 4k context across the batch — slower but completes.
GEN_BATCH_SIZE = 1
print(f"GEN_BATCH_SIZE={GEN_BATCH_SIZE} (quantize={quantize})")
if torch.cuda.is_available():
    print(f"CUDA mem after model load: "
          f"{torch.cuda.memory_allocated()/1024**3:.2f} GiB allocated, "
          f"{torch.cuda.memory_reserved()/1024**3:.2f} GiB reserved")


api = HfApi(token=os.environ.get("HF_TOKEN"))
# Mid-arm Hub checkpoints so a SIGTERM (exit 143) does not erase hours of generation —
# full train job 6a665c8b and infer job 6a67930f both died with 143 mid-work and zero
# prediction files, because we only uploaded after the whole arm finished.
PRED_PUSH_EVERY = int(os.environ.get("PRED_PUSH_EVERY") or 50)


def _pred_filename(label: str) -> str:
    return f"predictions-{label}{PRED_SUFFIX}.jsonl"


def _load_partial_predictions(label: str) -> list[dict]:
    """Resume from Hub (or local) partial jsonl if a prior job left one mid-arm."""
    name = _pred_filename(label)
    path = Path(name)
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            OUTPUT_REPO, name, repo_type="model", token=os.environ.get("HF_TOKEN"),
        )
        path = Path(local)
    except Exception as exc:
        print(f"  No Hub partial for {name} ({exc.__class__.__name__}); starting fresh")
        if not path.exists():
            return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    print(f"  Resuming {name} from {len(rows)} existing rows")
    return rows


def _write_and_push_predictions(label: str, rows: list[dict]) -> None:
    path = Path(_pred_filename(label))
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8",
    )
    try:
        api.upload_file(
            path_or_fileobj=str(path), path_in_repo=path.name,
            repo_id=OUTPUT_REPO, repo_type="model",
        )
        print(f"  Pushed {path.name} ({len(rows)}/{len(test_ds)} rows) to {OUTPUT_REPO}")
    except Exception as exc:
        # Soft-fail: keep generating; local path still has the jsonl for a later retry.
        print(
            f"  WARNING: Hub push of {path.name} failed "
            f"({type(exc).__name__}: {exc}); generation continues with local file."
        )


def generate_and_push(label: str):
    """Generate test predictions with periodic Hub pushes + resume.

    Left-padding keeps batch right-aligned so `out[:, input_len:]` is completion-only.
    Importable twin of evaluation/infer.py:generate_summaries — keep decode knobs in sync.
    """
    rows = _load_partial_predictions(label)
    start = len(rows)
    if start >= len(test_ds):
        print(f"  [{label}] already complete ({start}/{len(test_ds)}); re-pushing")
        _write_and_push_predictions(label, rows)
        return

    tokenizer.padding_side = "left"
    batch_size = GEN_BATCH_SIZE
    for i in range(start, len(test_ds), batch_size):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        batch = test_ds[i:i + batch_size]
        prompts: list[str] = [format_chat_prompt(p) for p in batch["prompt"]]
        # Chat template already includes BOS — do not prepend another.
        gen_max_input = int(TRAIN_CFG["max_length"]) - MAX_NEW_TOKENS
        inputs = tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=gen_max_input, padding=True, add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            outs = trained_model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                min_new_tokens=min(16, MAX_NEW_TOKENS), do_sample=False,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                repetition_penalty=REPETITION_PENALTY,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                bad_words_ids=BAD_WORDS_IDS,
            )
        input_len = inputs["input_ids"].shape[1]
        for j in range(len(prompts)):
            pred = tokenizer.decode(outs[j][input_len:], skip_special_tokens=True)
            rows.append({
                "text": batch["text"][j], "reference": batch["summary"][j],
                "prediction": pred.strip(), "model": label, "variant": VARIANT,
            })
        end = min(i + batch_size, len(test_ds))
        if end % 10 == 0 or end == len(test_ds):
            print(f"  [{label}] {end}/{len(test_ds)}")
        # Checkpoint to Hub so SIGTERM mid-arm still leaves scorable partials.
        if end % PRED_PUSH_EVERY == 0 or end == len(test_ds):
            _write_and_push_predictions(label, rows)
    tokenizer.padding_side = "right"


# Two systems from one loaded model: the fine-tuned adapter, and the zero-shot base
# (PEFT's disable_adapter() turns the adapter off without reloading the base model).
print(f"\nGenerating fine-tuned predictions (push every {PRED_PUSH_EVERY})...")
generate_and_push("finetuned")
if use_lora and not SKIP_BASE_ARM:
    print(f"Generating zero-shot base predictions (push every {PRED_PUSH_EVERY})...")
    with trained_model.disable_adapter():
        generate_and_push("base")

if not INFERENCE_ONLY:
    wandb.finish()
print(f"\nDone. Adapter + predictions at https://huggingface.co/{OUTPUT_REPO}")
