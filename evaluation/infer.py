"""
Evaluation pipeline, GPU inference helpers: load the fine-tuned Qwen3-2B + LoRA adapter and
generate test-set summaries. This is the importable twin of the generation block inside
training/train_hf_job.py (which is a self-contained cloud script and cannot import repo code).
It exists so the evaluation-observation notebook (notebooks/evaluation_observation.ipynb) can
watch the *real* model produce summaries live, using the same code path the HF job uses.

Execution environment: remote GPU only (Colab T4 / HF Jobs) — NEVER call locally; this machine's
8 GB GPU freezes on a Qwen3-2B load. Keep generate_summaries() in sync with
train_hf_job.py:generate_predictions().
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from training.config import MODEL_ID


def load_finetuned_model(adapter_repo: str, model_id: str = MODEL_ID, quantize: bool = False):
    """Load the base model with its LoRA adapter attached, ready for generation.

    Returns (model, tokenizer, device). The adapter can be toggled off with
    `model.disable_adapter()` to read the zero-shot base — the same trick the HF job uses to
    get both systems from one loaded model. Mirrors train_hf_job.py's load + adapter block.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(device_map="auto")
    if quantize:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    base = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model = PeftModel.from_pretrained(base, adapter_repo).eval()
    model.config.use_cache = True  # KV cache on for faster generation
    device = next(model.parameters()).device
    return model, tokenizer, device


def generate_summaries(
    model, tokenizer, dataset, variant: str, device,
    batch_size: int = 8, max_new_tokens: int = 256, label: str = "finetuned",
) -> list[dict]:
    """Generate summaries for a (processed) test split, in batches, greedily.

    Reads the dataset's precomputed `prompt`/`text`/`summary` columns (the variant is already
    baked into them by data/preprocess.py — do NOT re-apply make_variant). Left-padding keeps
    every sequence in a batch right-aligned so `out[:, input_len:]` extracts only the generated
    tokens. Returns the standard prediction rows that evaluate.py / error_analysis.py consume.
    Mirrors train_hf_job.py:generate_predictions().
    """
    tokenizer.padding_side = "left"
    rows = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        prompts: list[str] = batch["prompt"]
        inputs = tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=2048 - 128, padding=True,
        ).to(device)
        with torch.no_grad():
            outs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for j in range(len(prompts)):
            pred = tokenizer.decode(outs[j][input_len:], skip_special_tokens=True)
            rows.append({"text": batch["text"][j], "reference": batch["summary"][j],
                         "prediction": pred.strip(), "model": label, "variant": variant})
        print(f"  [{label}] {min(i + batch_size, len(dataset))}/{len(dataset)}")
    tokenizer.padding_side = "right"
    return rows
