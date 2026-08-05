"""Exact configuration used by the final E4 matched LoRA experiment."""

from dataclasses import asdict, dataclass, field


MODEL_ID = "dicta-il/dictalm2.0-instruct"
MODEL_SLUG = "dictalm2-instruct"
MAX_LENGTH = 4096
ARTICLE_TOKEN_BUDGET = 3840
DEFAULT_MAX_NEW_TOKENS = 128
TEST_SUBSET_N = 120
TEST_SUBSET_SEED = 1234

DATASET_REPOS = {
    "uncleaned": "avreymi/amlk-training-data-raw",
    "curated": "avreymi/amlk-training-data-e4cur",
}
MODEL_REPOS = {
    "uncleaned": "avreymi/amlk-e4-raw",
    "curated": "avreymi/amlk-e4-curated",
}


@dataclass(frozen=True)
class LoRAConfig:
    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(frozen=True)
class TrainingConfig:
    quantize: bool = False
    use_lora: bool = True
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    max_length: int = MAX_LENGTH


def training_payload() -> dict:
    return asdict(TrainingConfig())


def lora_payload() -> dict:
    return asdict(LoRAConfig())
