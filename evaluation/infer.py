"""
Evaluation pipeline, GPU inference helpers: load a base model (optionally + LoRA adapter) and
generate test-set summaries. This is the importable twin of the generation blocks inside
training/train_hf_job.py and evaluation/predict_base_hf_job.py (self-contained cloud scripts
that cannot import repo code). Used by the evaluation-observation notebook and by local
helpers that prepare multi-model zero-shot baselines.

Chat formatting uses data.prompts.format_chat_prompt for *both* finetuned and base arms
(instruct models require [INST]…[/INST] at train and serve). Tokenize with
add_special_tokens=False after templating to avoid double-BOS. Defaults to 4-bit load so a
7B base fits a Colab T4.

Execution environment: remote GPU only (Colab T4 / HF Jobs) — NEVER call locally; this machine's
8 GB GPU freezes on a model load of this size. Keep generate_summaries() in sync with
train_hf_job.py:generate_predictions() and predict_base_hf_job.py.
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from data.prompts import format_chat_prompt, prepare_tokenizer_for_templated_prompts
from evaluation.base_predict import resolve_load_plan
from training.config import DEFAULT_MAX_NEW_TOKENS, MAX_LENGTH, MODEL_ID


def _quant_or_bf16_kwargs(quantize: bool) -> dict:
    load_kwargs: dict = dict(device_map="auto")
    if quantize:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    return load_kwargs


def load_finetuned_model(adapter_repo: str, model_id: str = MODEL_ID, quantize: bool = True):
    """Load the base model with its LoRA adapter attached, ready for generation.

    Defaults to 4-bit: dictalm2.0-instruct is 7B; bf16 alone is ~14.7 GB and OOMs a 16 GB
    Colab T4 (plus adapter + KV). Pass quantize=False only on larger GPUs.

    Returns (model, tokenizer, device). The adapter can be toggled off with
    `model.disable_adapter()` to read the zero-shot base — the same trick the HF job uses to
    get both systems from one loaded model. Mirrors train_hf_job.py's load + adapter block.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepare_tokenizer_for_templated_prompts(tokenizer)

    load_kwargs = _quant_or_bf16_kwargs(quantize)
    base = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model = PeftModel.from_pretrained(base, adapter_repo).eval()
    model.config.use_cache = True  # KV cache on for faster generation
    device = next(model.parameters()).device
    return model, tokenizer, device


def load_base_model(model_id: str, quantize: bool | None = None):
    """Load a zero-shot base checkpoint with no adapter (multi-model baseline path).

    Handles causal LMs (incl. Nemotron with trust_remote_code) and Gemma-4 multimodal
    unified models (AutoProcessor + AutoModelForMultimodalLM). Returns a dict:
      {model, tokenizer, device, kind, processor?}
    Never call this on the local 8 GB machine — remote GPU only.
    """
    plan = resolve_load_plan(model_id)
    if quantize is None:
        quantize = bool(plan["quantize_default"])
    load_kwargs = _quant_or_bf16_kwargs(quantize)
    if plan["trust_remote_code"]:
        load_kwargs["trust_remote_code"] = True

    if plan["kind"] == "multimodal":
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs).eval()
        model.config.use_cache = True
        device = next(model.parameters()).device
        # Processor exposes a tokenizer for pad/eos ids and chat templating.
        tokenizer = getattr(processor, "tokenizer", None) or processor
        if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token
        prepare_tokenizer_for_templated_prompts(tokenizer)
        return {
            "model": model,
            "tokenizer": tokenizer,
            "processor": processor,
            "device": device,
            "kind": "multimodal",
        }

    # Nemotron ships tokenizer.json only; AutoTokenizer can bind a slow LlamaTokenizer that
    # encodes Hebrew to empty ids. Prefer fast; fall back to PreTrainedTokenizerFast.
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=plan["trust_remote_code"], use_fast=True
    )
    if not tokenizer.encode("שלום", add_special_tokens=False):
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepare_tokenizer_for_templated_prompts(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs).eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    return {
        "model": model,
        "tokenizer": tokenizer,
        "processor": None,
        "device": device,
        "kind": "causal",
    }


def build_input_text(tokenizer, prompt: str) -> str:
    """Format a prompt for generation (chat template when the model has one).

    Same path for finetuned and zero-shot base. Twin of train_hf_job.py:format_chat_prompt.
    """
    return format_chat_prompt(tokenizer, prompt)


def generate_summaries(
    model, tokenizer, dataset, variant: str, device,
    batch_size: int = 8, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    label: str = "finetuned",
) -> list[dict]:
    """Generate summaries for a (processed) test split, in batches, greedily.

    Reads the dataset's precomputed `prompt`/`text`/`summary` columns (the variant is already
    baked into them by data/preprocess.py — do NOT re-apply make_variant). Left-padding keeps
    every sequence in a batch right-aligned so `out[:, input_len:]` extracts only the generated
    tokens. Always applies the Hebrew-script decode constraint. Returns the standard prediction
    rows that evaluate.py / error_analysis.py consume. Mirrors train_hf_job.py:generate_predictions().
    Default max_new_tokens matches training/config.DEFAULT_MAX_NEW_TOKENS (cost lever: 128).
    """
    from evaluation.hebrew_constraint import build_bad_words_ids

    tokenizer.padding_side = "left"
    bad_words_ids = build_bad_words_ids(tokenizer)
    rows = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        prompts: list[str] = [build_input_text(tokenizer, p) for p in batch["prompt"]]
        # Chat template already includes BOS — do not prepend another.
        # Leave headroom for max_new_tokens under the same seq budget as training.
        inputs = tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=MAX_LENGTH - max_new_tokens, padding=True, add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            outs = model.generate(
                # no_repeat_ngram_size + repetition_penalty kill greedy degeneration loops;
                # min_new_tokens + explicit eos let the model stop instead of running to the cap.
                **inputs, max_new_tokens=max_new_tokens,
                min_new_tokens=min(16, max_new_tokens), do_sample=False,
                no_repeat_ngram_size=3, repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                bad_words_ids=bad_words_ids,
            )
        input_len = inputs["input_ids"].shape[1]
        for j in range(len(prompts)):
            pred = tokenizer.decode(outs[j][input_len:], skip_special_tokens=True)
            rows.append({"text": batch["text"][j], "reference": batch["summary"][j],
                         "prediction": pred.strip(), "model": label, "variant": variant})
        print(f"  [{label}] {min(i + batch_size, len(dataset))}/{len(dataset)}")
    tokenizer.padding_side = "right"
    return rows
