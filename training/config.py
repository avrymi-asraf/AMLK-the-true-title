"""
Shared configuration for all fine-tuning scripts (train_qlora.py, train_lora.py,
train_full.py). Defines model ID, LoRA hyperparameters, and training settings.
Changing values here affects all three training approaches uniformly.

Execution environment: imported by training scripts on local machine or remote GPU.
"""
from dataclasses import dataclass, field


MODEL_ID = "Qwen/Qwen3-2B"
DATA_DIR = "outputs/data/processed"
MAX_SEQ_LENGTH = 2048
RESPONSE_TEMPLATE = "\nSummary:\n"  # used by DataCollatorForCompletionOnlyLM


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
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    bf16: bool = True
    report_to: str = "none"
