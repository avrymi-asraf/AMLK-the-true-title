"""
Shared configuration for the training pipeline (training/train.py and the remote
training/train_hf_job.py). Defines the model id, the per-method presets (qlora |
lora | full), LoRA hyperparameters, common training settings, the wandb project, and
the Hub repo-id helpers. Changing a value here changes every training method at once.

Execution environment: imported by the local trainer and read (values only) by the
self-contained HF Jobs script.
"""
from dataclasses import dataclass, field


MODEL_ID = "dicta-il/DictaLM-3.0-1.7B-Base"
PROCESSED_DIR = "outputs/data/processed"   # actual data dir is <PROCESSED_DIR>/<variant>
MAX_LENGTH = 2048
WANDB_PROJECT = "amlk-hebrew-summarization"


def _profile_suffix(variant: str, clean: bool = False, drop_roundups: bool = False) -> str:
    """Repo suffix for a (variant, clean, drop_roundups) profile. `whole` + raw = no suffix
    (the original artifacts). Clean adds `-clean`; `--drop-roundups` adds `-drop` on top."""
    suffix = "" if variant == "whole" else f"-{variant}"
    if clean:
        suffix += "-clean"
        if drop_roundups:
            suffix += "-drop"
    return suffix


def processed_profile_name(variant: str, clean: bool = False, drop_roundups: bool = False) -> str:
    """Local processed-data dir name under outputs/data/processed/."""
    name = variant
    if clean:
        name += "-clean"
        if drop_roundups:
            name += "-drop"
    return name


def dataset_repo(hf_user: str, variant: str = "whole", clean: bool = False,
                 drop_roundups: bool = False) -> str:
    """Hub dataset repo holding the processed Arrow splits for a probe variant/profile."""
    return f"{hf_user}/amlk-training-data{_profile_suffix(variant, clean, drop_roundups)}"


def model_repo(hf_user: str, variant: str = "whole", clean: bool = False,
               drop_roundups: bool = False) -> str:
    """Hub repo for the trained LoRA adapter of a probe variant/profile."""
    return f"{hf_user}/amlk-dictalm3-1.7b-sft{_profile_suffix(variant, clean, drop_roundups)}"


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
