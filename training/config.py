"""
Shared configuration for the training pipeline (training/train.py and the remote
training/train_hf_job.py). Defines the model id, the per-method presets (qlora |
lora | full), LoRA hyperparameters, common training settings, wandb naming helpers,
and the Hub repo-id helpers. Changing a value here changes every training method at once.

Execution environment: imported by the local trainer and read (values only) by the
self-contained HF Jobs script.
"""
from dataclasses import dataclass, field
from datetime import date


MODEL_ID = "dicta-il/dictalm2.0-instruct"
PROCESSED_DIR = "outputs/data/processed"   # actual data dir is <PROCESSED_DIR>/<variant>
MAX_LENGTH = 4096
# Short slug used in Hub adapter repos and wandb names (not the full model id).
MODEL_SLUG = "dictalm2-instruct"
# One epoch per run is the default; override with --epochs only when deliberately multi-epoch.
DEFAULT_EPOCHS = 1
# Decode budget for post-train dual-arm generation (train_hf_job + infer twins).
# 128 is enough for the short news-summary target (median ref ~157 chars) and cuts GPU
# decode time vs the old 256 cap: smoke preds always hit 256 without EOS, so dual inference
# was ~1.6h of a ~5.8h a10g-small job (C5, measured on the pre-curation 6073/760 splits —
# the curated splits are smaller, so re-measure before quoting a wall-clock).
# Override with --max-new-tokens / MAX_NEW_TOKENS when a long-form probe needs more.
DEFAULT_MAX_NEW_TOKENS = 128


def wandb_project(model_slug: str = MODEL_SLUG) -> str:
    """One wandb project per base model so runs stay grouped and discoverable."""
    return f"amlk-{model_slug}"


def wandb_run_name(
    method: str,
    variant: str,
    *,
    model_slug: str = MODEL_SLUG,
    epochs: int = DEFAULT_EPOCHS,
    tag: str = "",
    run_date: str | None = None,
) -> str:
    """Informative wandb run name: date + model + method + variant + epochs [+ tag].

    Example: 2026-07-11_dictalm2-instruct_qlora_whole_1ep_smoke
    """
    day = run_date or date.today().isoformat()
    parts = [day, model_slug, method, variant, f"{epochs}ep"]
    if tag:
        parts.append(tag)
    return "_".join(parts)


def _profile_suffix(variant: str) -> str:
    """Repo/dir suffix for a probe variant. `whole` = no suffix."""
    return "" if variant == "whole" else f"-{variant}"


def processed_profile_name(variant: str) -> str:
    """Local processed-data dir name under outputs/data/processed/."""
    return variant


def dataset_repo(hf_user: str, variant: str = "whole") -> str:
    """Hub dataset repo holding the processed Arrow splits for a probe variant."""
    return f"{hf_user}/amlk-training-data{_profile_suffix(variant)}"


def model_repo(hf_user: str, variant: str = "whole") -> str:
    """Hub repo for the trained LoRA adapter of a probe variant."""
    return f"{hf_user}/amlk-{MODEL_SLUG}-sft{_profile_suffix(variant)}"


@dataclass
class LoRAConfig:
    # r=32 + the MLP projections (gate/up/down), not just attention, gives the adapter enough
    # capacity for abstractive generation rather than degenerating into lead-copying/looping.
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    num_train_epochs: int = DEFAULT_EPOCHS
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    # Match train_hf_job full-run cadence: denser mid-run Hub full Trainer checkpoints
    # for cross-job --resume-from after SIGTERM (was 200; cloud twin uses 50).
    save_steps: int = 50
    eval_steps: int = 50
    bf16: bool = True


# Per-method deltas — everything not listed here is shared (see TrainingConfig).
# `quantize` loads the base model in 4-bit; `use_lora` attaches LoRA adapters.
METHOD_PRESETS = {
    "qlora": dict(quantize=True,  use_lora=True,  per_device_train_batch_size=2,
                  gradient_accumulation_steps=8,  learning_rate=2e-4),
    # bf16 LoRA is the cheapest regime if it fits (no 4-bit dequant per matmul), but a bf16 7B is
    # ~14.5 GB of the A10G's 24 GB before activations, so batch is 1 — the old 4/4 was set when
    # MAX_LENGTH was 2048. Effective batch stays 16, matching qlora.
    # MEASURED 2026-07-26 (paired 10-step smokes, same 50 examples): 40.49 s/step vs qlora's
    # 55.33 — 1.37x faster, and both numbers already include each regime's group_by_length
    # effect (20% for qlora at micro-batch 2, nil for lora at micro-batch 1). Default method.
    "lora":  dict(quantize=False, use_lora=True,  per_device_train_batch_size=1,
                  gradient_accumulation_steps=16, learning_rate=2e-4),
    "full":  dict(quantize=False, use_lora=False, per_device_train_batch_size=1,
                  gradient_accumulation_steps=16, learning_rate=5e-5),
}
