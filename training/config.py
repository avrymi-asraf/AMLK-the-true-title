"""
Shared configuration for the training pipeline (training/train.py and the remote
training/train_hf_job.py). Defines the model id, the per-method presets (qlora |
lora | full), LoRA hyperparameters, common training settings, the wandb project, and
the Hub repo-id helpers. Changing a value here changes every training method at once.

Execution environment: imported by the local trainer and read (values only) by the
self-contained HF Jobs script.
"""
from dataclasses import dataclass, field


MODEL_ID = "Qwen/Qwen3-2B"
PROCESSED_DIR = "outputs/data/processed"   # actual data dir is <PROCESSED_DIR>/<variant>
MAX_LENGTH = 2048
WANDB_PROJECT = "amlk-hebrew-summarization"


def dataset_repo(hf_user: str, variant: str = "whole") -> str:
    """Hub dataset repo holding the processed Arrow splits for a probe variant."""
    suffix = "" if variant == "whole" else f"-{variant}"
    return f"{hf_user}/amlk-training-data{suffix}"


def model_repo(hf_user: str, variant: str = "whole") -> str:
    """Hub repo for the trained LoRA adapter of a probe variant."""
    suffix = "" if variant == "whole" else f"-{variant}"
    return f"{hf_user}/amlk-qwen3-2b-sft{suffix}"


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    num_train_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    bf16: bool = True


# Per-method deltas — everything not listed here is shared (see TrainingConfig).
# `quantize` loads the base model in 4-bit; `use_lora` attaches LoRA adapters.
METHOD_PRESETS = {
    "qlora": dict(quantize=True,  use_lora=True,  per_device_train_batch_size=2,
                  gradient_accumulation_steps=8,  learning_rate=2e-4),
    "lora":  dict(quantize=False, use_lora=True,  per_device_train_batch_size=4,
                  gradient_accumulation_steps=4,  learning_rate=2e-4),
    "full":  dict(quantize=False, use_lora=False, per_device_train_batch_size=1,
                  gradient_accumulation_steps=16, learning_rate=5e-5),
}
