#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "transformers>=5.10.0",
#     "accelerate>=1.0.0",
#     "bitsandbytes>=0.44.0",
#     "datasets>=3.0.0",
#     "huggingface_hub",
#     "torch",
#     "sentencepiece",
#     "protobuf",
#     "pillow",
#     "einops",
#     "torchvision",
# ]
# ///
"""
Evaluation pipeline, remote zero-shot base predictions: load a named Hub base model (no
training, no LoRA adapter) and generate predictions-base.jsonl on the first N test examples.

Role: multi-model baseline generation for AMLK (DictaLM-2 instruct, DictaLM-3 Nemotron,
Gemma-4, etc.) so systems can be compared without fine-tuning. Two modes in one file:
  --submit-hf  (local) uploads this script to an HF Jobs GPU and passes settings as env vars
  no args      (cloud) HF Jobs entry — downloads the processed test split, generates, pushes

Execution environment: submitted from a machine with HF_TOKEN; generation runs in an
ephemeral HuggingFace Jobs GPU container (a10g-small, 4-bit for ~7–12B). Never load these
models on the local 8 GB machine.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

# --------------------------------------------------------------------------- pure helpers
# Duplicated from evaluation/base_predict.py on purpose: the UV job only ships this file.


def model_slug(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def resolve_load_plan(model_id: str) -> dict:
    mid = model_id.lower()
    if "gemma-4" in mid or "gemma4" in mid:
        return {"kind": "multimodal", "trust_remote_code": False, "quantize_default": True}
    if "nemotron" in mid:
        # Prefer native transformers NemotronH (no mamba-ssm). Remote Hub code needs mamba-ssm.
        return {"kind": "causal", "trust_remote_code": False, "quantize_default": True}
    return {"kind": "causal", "trust_remote_code": False, "quantize_default": True}


def build_input_text_safe(tokenizer, prompt: str) -> str:
    """Twin of data.prompts.format_chat_prompt — no think-switch injection (Mistral-safe)."""
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def write_predictions_jsonl(rows: list[dict], path: Path) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _load_causal_tokenizer(base_model: str, plan: dict, hf_token: str | None):
    """Load a causal LM tokenizer that actually encodes Hebrew.

    DictaLM-3 Nemotron ships only tokenizer.json (no sentencepiece .model). AutoTokenizer
    can still bind LlamaTokenizer (slow) which encodes Hebrew to *empty* ids and then
    produces mojibake predictions. Prefer the fast JSON tokenizer; verify with a Hebrew
    probe and fall back to PreTrainedTokenizerFast if needed.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    kwargs = dict(token=hf_token, trust_remote_code=plan["trust_remote_code"])
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, **kwargs)
    probe = "שלום עולם"
    if not tokenizer.encode(probe, add_special_tokens=False):
        print(
            f"WARNING: {type(tokenizer).__name__} failed Hebrew encode probe; "
            "forcing PreTrainedTokenizerFast(tokenizer.json)"
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model, token=hf_token)
        # Chat template lives in chat_template.jinja / tokenizer_config — re-load via Auto
        # config if the fast-only path dropped it.
        if not getattr(tokenizer, "chat_template", None):
            try:
                cfg_tok = AutoTokenizer.from_pretrained(base_model, use_fast=True, **kwargs)
                if getattr(cfg_tok, "chat_template", None):
                    tokenizer.chat_template = cfg_tok.chat_template
            except Exception as err:
                print(f"chat_template attach skipped: {err}")
    ids = tokenizer.encode(probe, add_special_tokens=False)
    roundtrip = tokenizer.decode(ids, skip_special_tokens=True)
    print(f"Tokenizer={type(tokenizer).__name__} hebrew_probe_ids={len(ids)} "
          f"roundtrip_ok={probe in roundtrip or all(c in roundtrip for c in probe if not c.isspace())}")
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    # Twin of data.prompts.prepare_tokenizer_for_templated_prompts (C1).
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False
    return tokenizer


# --------------------------------------------------------------------------- cloud side
def run_cloud_job() -> None:
    """HF Jobs entry: load BASE_MODEL, generate LIMIT base predictions, push to OUTPUT_REPO."""
    import torch
    from datasets import load_from_disk
    from huggingface_hub import HfApi, snapshot_download
    from transformers import BitsAndBytesConfig

    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    base_model = os.environ["BASE_MODEL"]
    dataset_repo = os.environ["DATASET_REPO"]
    output_repo = os.environ["OUTPUT_REPO"]
    variant = os.environ.get("VARIANT", "whole")
    limit = int(os.environ.get("LIMIT", "100"))
    quantize = os.environ.get("QUANTIZE", "1") == "1"
    batch_size = int(os.environ.get("BATCH_SIZE", "0"))  # 0 = auto
    # MAX_INPUT caps article tokens; Nemotron hybrid Mamba prefill scales badly with length.
    max_input = int(os.environ.get("MAX_INPUT", "0"))  # 0 = auto
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "256"))
    hf_token = os.environ.get("HF_TOKEN")

    plan = resolve_load_plan(base_model)
    mid = base_model.lower()
    if batch_size <= 0:
        # Nemotron hybrid + 12B multimodal: keep batch 1 on 24 GB A10G (4-bit still spikes
        # during Mamba dequant / multimodal prefill). Smaller causal instruct models can batch.
        if plan["kind"] == "multimodal" or "nemotron" in mid or "12b" in mid:
            batch_size = 1
        else:
            batch_size = 4
    if max_input <= 0:
        # With a working Hebrew tokenizer, HeSum articles are long; hybrid Mamba OOM'd at 1536
        # on A10G 24 GB (tried to allocate 7.5 GiB mid-prefill). Keep Nemotron short.
        if "nemotron" in mid:
            max_input = 768
        elif plan["kind"] == "multimodal" or "12b" in mid:
            max_input = 1536
        else:
            max_input = 4096 - 128

    print("=== zero-shot base predictions (no training) ===")
    print(f"Base model: {base_model}")
    print(f"Plan: {plan}  quantize={quantize}  batch_size={batch_size}")
    print(f"max_input={max_input}  max_new_tokens={max_new_tokens}")
    print(f"Dataset: {dataset_repo}  limit={limit}  variant={variant}")
    print(f"Output: {output_repo}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    local_data = Path("./data")
    snapshot_download(
        repo_id=dataset_repo, repo_type="dataset", local_dir=str(local_data), token=hf_token
    )
    test_ds = load_from_disk(str(local_data / "test"))
    n = min(limit, len(test_ds))
    test_ds = test_ds.select(range(n))
    print(f"Test slice: {len(test_ds)} examples (of original split)")

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
    if plan["trust_remote_code"]:
        load_kwargs["trust_remote_code"] = True

    processor = None
    if plan["kind"] == "multimodal":
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        print("Loading multimodal model + processor...")
        processor = AutoProcessor.from_pretrained(base_model, token=hf_token)
        model = AutoModelForMultimodalLM.from_pretrained(
            base_model, token=hf_token, **load_kwargs
        ).eval()
        tokenizer = getattr(processor, "tokenizer", None) or processor
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading causal LM + tokenizer...")
        tokenizer = _load_causal_tokenizer(base_model, plan, hf_token)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model, token=hf_token, **load_kwargs
            ).eval()
        except Exception as err:
            # Hub auto_map may still pull remote code; if native load fails for Nemotron-class
            # models, retry with trust_remote_code (requires mamba-ssm in the UV deps).
            if plan.get("trust_remote_code"):
                raise
            print(f"Native load failed ({type(err).__name__}: {err}); retrying with trust_remote_code")
            load_kwargs = {**load_kwargs, "trust_remote_code": True}
            tokenizer = _load_causal_tokenizer(
                base_model, {**plan, "trust_remote_code": True}, hf_token
            )
            model = AutoModelForCausalLM.from_pretrained(
                base_model, token=hf_token, **load_kwargs
            ).eval()

    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = True
    device = next(model.parameters()).device
    print(f"Loaded on {device}")

    rows: list[dict] = []
    if plan["kind"] == "multimodal":
        rows = _generate_multimodal(
            model, processor, tokenizer, test_ds, variant, device, batch_size,
            max_new_tokens=max_new_tokens,
        )
    else:
        rows = _generate_causal(
            model, tokenizer, test_ds, variant, device, batch_size,
            max_input=max_input, max_new_tokens=max_new_tokens,
        )

    out_path = Path("predictions-base.jsonl")
    write_predictions_jsonl(rows, out_path)
    print(f"Wrote {out_path} ({len(rows)} rows)")

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=output_repo, repo_type="model", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=out_path.name,
        repo_id=output_repo,
        repo_type="model",
    )
    print(f"Pushed {out_path.name} to https://huggingface.co/{output_repo}")
    print("Done (base-only inference; no training).")


def _generate_causal(
    model, tokenizer, test_ds, variant, device, batch_size,
    max_input: int = 1920, max_new_tokens: int = 256,
) -> list[dict]:
    import torch

    tokenizer.padding_side = "left"
    rows: list[dict] = []
    for i in range(0, len(test_ds), batch_size):
        batch = test_ds[i : i + batch_size]
        prompts = [build_input_text_safe(tokenizer, p) for p in batch["prompt"]]
        # add_special_tokens=False: chat template already includes BOS (C1 double-BOS fix).
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=max_input,
            padding=True,
            add_special_tokens=False,
        ).to(device)
        if i == 0:
            print(f"  first-batch input_ids shape={tuple(inputs['input_ids'].shape)} "
                  f"max_input={max_input}", flush=True)
        with torch.no_grad():
            outs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min(16, max_new_tokens),
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        input_len = inputs["input_ids"].shape[1]
        for j in range(len(prompts)):
            pred = tokenizer.decode(outs[j][input_len:], skip_special_tokens=True)
            rows.append(
                {
                    "text": batch["text"][j],
                    "reference": batch["summary"][j],
                    "prediction": pred.strip(),
                    "model": "base",
                    "variant": variant,
                }
            )
        # Free activation peaks between batches on memory-tight 12B jobs.
        if device.type == "cuda":
            del outs, inputs
            torch.cuda.empty_cache()
        end = min(i + batch_size, len(test_ds))
        if (i // batch_size) % 5 == 0 or end == len(test_ds):
            print(f"  [base] {end}/{len(test_ds)}", flush=True)
            if end <= batch_size and rows:
                sample = rows[0]["prediction"][:120].replace("\n", " ")
                print(f"  sample pred[0]: {sample!r}", flush=True)
    tokenizer.padding_side = "right"
    return rows


def _generate_multimodal(
    model, processor, tokenizer, test_ds, variant, device, batch_size,
    max_new_tokens: int = 256,
) -> list[dict]:
    """Gemma-4 unified path: processor.apply_chat_template + model.generate (batch 1 preferred)."""
    import torch

    rows: list[dict] = []
    # Multimodal processors rarely left-pad batches cleanly; default to sequential.
    for i in range(len(test_ds)):
        prompt = test_ds[i]["prompt"]
        messages = [{"role": "user", "content": prompt}]
        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            min_new_tokens=min(16, max_new_tokens),
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )
        eos = getattr(tokenizer, "eos_token_id", None)
        pad = getattr(tokenizer, "pad_token_id", None) or eos
        if eos is not None:
            gen_kwargs["eos_token_id"] = eos
        if pad is not None:
            gen_kwargs["pad_token_id"] = pad
        with torch.no_grad():
            outs = model.generate(**inputs, **gen_kwargs)
        pred = processor.decode(outs[0][input_len:], skip_special_tokens=True)
        rows.append(
            {
                "text": test_ds[i]["text"],
                "reference": test_ds[i]["summary"],
                "prediction": pred.strip(),
                "model": "base",
                "variant": variant,
            }
        )
        if i % 10 == 0 or i + 1 == len(test_ds):
            print(f"  [base] {i + 1}/{len(test_ds)}", flush=True)
    return rows


# --------------------------------------------------------------------------- local side
def submit(
    model_id: str,
    hf_user: str,
    *,
    limit: int = 100,
    variant: str = "whole",
    quantize: bool = True,
    flavor: str = "a10g-small",
    timeout: str = "2h",
    output_repo: str = "",
    dataset_repo: str = "",
    max_input: int = 0,
    max_new_tokens: int = 256,
) -> object:
    """Submit this script to HF Jobs for one base model."""
    from huggingface_hub import HfApi

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN not set. Run: source .env", file=sys.stderr)
        sys.exit(1)

    data_repo = dataset_repo or f"{hf_user}/amlk-training-data"
    out_repo = output_repo or f"{hf_user}/amlk-preds-{model_slug(model_id)}"
    plan = resolve_load_plan(model_id)
    # Nemotron hybrid needs more VRAM headroom once Hebrew tokenization is correct.
    if flavor == "a10g-small" and "nemotron" in model_id.lower():
        flavor = "a100-large"
        if max_input <= 0:
            max_input = 1024

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=out_repo, repo_type="model", private=True, exist_ok=True)

    print(f"Submitting base-only prediction job for {model_id}")
    print(f"  flavor={flavor} timeout={timeout} limit={limit} quantize={quantize}")
    print(f"  max_input={max_input or 'auto'} max_new_tokens={max_new_tokens}")
    print(f"  plan={plan}")
    print(f"  dataset={data_repo} -> output={out_repo}")

    env = {
        "BASE_MODEL": model_id,
        "DATASET_REPO": data_repo,
        "OUTPUT_REPO": out_repo,
        "VARIANT": variant,
        "LIMIT": str(limit),
        "QUANTIZE": "1" if quantize else "0",
        "MAX_NEW_TOKENS": str(max_new_tokens),
    }
    if max_input > 0:
        env["MAX_INPUT"] = str(max_input)

    job = api.run_uv_job(
        script=str(Path(__file__).resolve()),
        flavor=flavor,
        timeout=timeout,
        secrets={"HF_TOKEN": hf_token},
        env=env,
        token=hf_token,
    )
    print(f"\nJob submitted. ID: {job.id}  Status: {job.status.stage}")
    print(f"  Monitor: https://huggingface.co/jobs/{hf_user}/{job.id}")
    print(f"  Logs:    hf jobs logs {job.id}")
    print(f"  Preds:   https://huggingface.co/{out_repo}")
    return job


def download_predictions(model_id: str, hf_user: str, dest_root: str | Path = "outputs") -> Path:
    """Pull predictions-base.jsonl from the Hub into outputs/<slug>/."""
    from huggingface_hub import hf_hub_download

    hf_token = os.environ.get("HF_TOKEN", "")
    out_repo = f"{hf_user}/amlk-preds-{model_slug(model_id)}"
    dest_dir = Path(dest_root) / model_slug(model_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        out_repo,
        "predictions-base.jsonl",
        repo_type="model",
        local_dir=str(dest_dir),
        token=hf_token,
    )
    path = Path(local)
    print(f"Downloaded {path} ({sum(1 for _ in open(path, encoding='utf-8'))} lines)")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Zero-shot base predictions on HF Jobs (no training)"
    )
    parser.add_argument(
        "--submit-hf",
        action="store_true",
        help="Submit this script to HF Jobs (GPU) for base-only generation",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Hub model id (required with --submit-hf / --download)",
    )
    parser.add_argument("--hf-user", default="avreymi", help="HuggingFace username")
    parser.add_argument("--limit", type=int, default=100, help="Number of test examples")
    parser.add_argument("--variant", choices=("whole", "lead", "body"), default="whole")
    parser.add_argument("--no-quantize", action="store_true", help="Load bf16 instead of 4-bit")
    parser.add_argument("--flavor", default="a10g-small", help="HF Jobs GPU flavor")
    parser.add_argument("--timeout", default="2h", help="Job timeout")
    parser.add_argument("--output-repo", default="", help="Override Hub model repo for preds")
    parser.add_argument("--dataset-repo", default="", help="Override Hub dataset repo")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download predictions-base.jsonl from Hub into outputs/<slug>/",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="With --submit-hf: submit the three default comparison models",
    )
    args = parser.parse_args()

    default_models = [
        "dicta-il/dictalm2.0-instruct",
        "dicta-il/DictaLM-3.0-Nemotron-12B-Instruct",
        "google/gemma-4-12B-it",
    ]

    if args.download:
        models = default_models if args.all_models else ([args.model] if args.model else [])
        if not models:
            print("ERROR: --download needs --model or --all-models", file=sys.stderr)
            sys.exit(1)
        for m in models:
            download_predictions(m, args.hf_user)
        return

    if args.submit_hf:
        models = default_models if args.all_models else ([args.model] if args.model else [])
        if not models:
            print("ERROR: --submit-hf needs --model or --all-models", file=sys.stderr)
            sys.exit(1)
        for m in models:
            submit(
                m,
                args.hf_user,
                limit=args.limit,
                variant=args.variant,
                quantize=not args.no_quantize,
                flavor=args.flavor,
                timeout=args.timeout,
                output_repo=args.output_repo if len(models) == 1 else "",
                dataset_repo=args.dataset_repo,
            )
        return

    # Cloud entry (HF Jobs invokes the script with no CLI args).
    run_cloud_job()


if __name__ == "__main__":
    main()
