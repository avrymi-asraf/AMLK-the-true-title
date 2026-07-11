"""
Evaluation pipeline, GPU inference helpers: load a base model (optionally + LoRA adapter) and
generate test-set summaries. This is the importable twin of the generation blocks inside
training/train_hf_job.py and evaluation/predict_base_hf_job.py (self-contained cloud scripts
that cannot import repo code). Used by the evaluation-observation notebook and by local
helpers that prepare multi-model zero-shot baselines.

Execution environment: remote GPU only (Colab T4 / HF Jobs) — NEVER call locally; this machine's
8 GB GPU freezes on a model load of this size. Keep generate_summaries() in sync with
train_hf_job.py:generate_predictions() and predict_base_hf_job.py.
"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from evaluation.base_predict import build_input_text_safe, resolve_load_plan
from training.config import MODEL_ID


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


def load_finetuned_model(adapter_repo: str, model_id: str = MODEL_ID, quantize: bool = False):
    """Load the base model with its LoRA adapter attached, ready for generation.

    Returns (model, tokenizer, device). The adapter can be toggled off with
    `model.disable_adapter()` to read the zero-shot base — the same trick the HF job uses to
    get both systems from one loaded model. Mirrors train_hf_job.py's load + adapter block.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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


def build_input_text(tokenizer, prompt: str, label: str) -> str:
    """Format the prompt for generation, per system. Mirrors train_hf_job.py:build_input_text —
    keep the two in sync by hand (that script can't import repo code).

    The LoRA adapter is trained on the raw completion-style prompt, so "finetuned" keeps
    using it verbatim. The zero-shot "base" system never sees that format in training — on a
    chat-capable model, feed the real chat template with enable_thinking=False and a
    `/no_think` soft switch. Older Mistral / DictaLM-2 templates reject enable_thinking= —
    build_input_text_safe falls back. Pure base checkpoints with no chat template use the
    raw prompt.
    """
    if label != "base":
        return prompt
    return build_input_text_safe(tokenizer, prompt)


def generate_summaries(
    model, tokenizer, dataset, variant: str, device,
    batch_size: int = 8, max_new_tokens: int = 256, label: str = "finetuned",
) -> list[dict]:
    """Generate summaries for a (processed) test split, in batches, greedily.

    Reads the dataset's precomputed `prompt`/`text`/`summary` columns (the variant is already
    baked into them by data/preprocess.py — do NOT re-apply make_variant). Left-padding keeps
    every sequence in a batch right-aligned so `out[:, input_len:]` extracts only the generated
    tokens. Always applies the Hebrew-script decode constraint. Returns the standard prediction
    rows that evaluate.py / error_analysis.py consume. Mirrors train_hf_job.py:generate_predictions().
    """
    from evaluation.hebrew_constraint import build_bad_words_ids

    tokenizer.padding_side = "left"
    bad_words_ids = build_bad_words_ids(tokenizer)
    rows = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        prompts: list[str] = [build_input_text(tokenizer, p, label) for p in batch["prompt"]]
        inputs = tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=2048 - 128, padding=True,
        ).to(device)
        with torch.no_grad():
            outs = model.generate(
                # no_repeat_ngram_size + repetition_penalty kill greedy degeneration loops;
                # min_new_tokens + explicit eos let the model stop instead of running to the cap.
                **inputs, max_new_tokens=max_new_tokens, min_new_tokens=16, do_sample=False,
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
