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
# ]
# ///
"""
Prompt-optimization arena, remote half: sweep K prompt candidates over the same N test
examples on one base model, in ONE HF Job that loads the model once.

Role: the generation step of the prompt-improvement loop that precedes fine-tuning. The
local half (evaluation/prompt_arena.py) writes a round's candidates and scores the results;
this script turns candidates into predictions. Rows carry `prompt_id` so one predictions
file holds the whole round. Two modes in one file:
  --submit-hf  (local) uploads this script to an HF Jobs GPU, passing the round as env vars
  no args      (cloud) HF Jobs entry — downloads the test split, generates, pushes results

Execution environment: submitted from a machine with HF_TOKEN; generation runs on an
ephemeral HF Jobs GPU (a10g-small, 4-bit). Never load the model on the local 8 GB machine.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

TEXT_PLACEHOLDER = "{text}"

# --------------------------------------------------------------------------- inlined twins
# Duplicated on purpose: HF Jobs ships this single file, so it cannot import repo code.
# Twins of data.prompts.format_chat_prompt and evaluation.hebrew_constraint.

_FORBIDDEN_SCRIPT_RE = re.compile("[A-Za-zÀ-ɏЀ-ӿͰ-Ͽ؀-ۿ぀-ヿ㐀-鿿가-힯ᄀ-ᇿㄱ-ㆿ]")


def build_input_text_safe(tokenizer, prompt: str) -> str:
    """Wrap a user instruction in the model's chat template (dictalm2: [INST]…[/INST])."""
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_bad_words_ids(tokenizer):
    """Ban every vocab token whose decoded form carries a non-Hebrew-script letter."""
    special_ids = set(tokenizer.all_special_ids)
    bad = []
    for token_id in range(tokenizer.vocab_size):
        if token_id in special_ids:
            continue
        piece = tokenizer.decode([token_id])
        if piece and _FORBIDDEN_SCRIPT_RE.search(piece):
            bad.append([token_id])
    return bad or None


def truncate_article(tokenizer, template: str, text: str, max_input: int) -> str:
    """Fit the article to the budget the *template* leaves inside max_input.

    Truncating the assembled prompt instead would cut its tail — i.e. the trailing
    instruction ("Summary:") — and the model would never see the task. So we measure the
    template's own token cost and cut only the article.
    """
    overhead = len(tokenizer(template.replace(TEXT_PLACEHOLDER, ""),
                             add_special_tokens=False).input_ids)
    budget = max_input - overhead - 32  # margin for chat-template markers
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= budget:
        return text
    return tokenizer.decode(ids[:budget], skip_special_tokens=True)


# --------------------------------------------------------------------------- cloud side
def run_cloud_job() -> None:
    """HF Jobs entry: load BASE_MODEL once, generate for every prompt candidate, push."""
    import torch
    from datasets import load_from_disk
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base_model = os.environ["BASE_MODEL"]
    dataset_repo = os.environ["DATASET_REPO"]
    output_repo = os.environ["OUTPUT_REPO"]
    candidates = json.loads(os.environ["PROMPTS_JSON"])
    round_num = int(os.environ.get("ROUND", "1"))
    limit = int(os.environ.get("LIMIT", "100"))
    variant = os.environ.get("VARIANT", "whole")
    quantize = os.environ.get("QUANTIZE", "1") == "1"
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    max_input = int(os.environ.get("MAX_INPUT", "3968"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "160"))
    hf_token = os.environ.get("HF_TOKEN")

    print("=== prompt sweep (zero-shot, no training) ===")
    print(f"Base model: {base_model}  quantize={quantize}")
    print(f"Round {round_num}: {len(candidates)} prompts x {limit} examples")
    print(f"batch_size={batch_size} max_input={max_input} max_new_tokens={max_new_tokens}")
    print(f"Dataset: {dataset_repo} -> Output: {output_repo}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    local_data = Path("./data")
    snapshot_download(repo_id=dataset_repo, repo_type="dataset",
                      local_dir=str(local_data), token=hf_token)
    test_ds = load_from_disk(str(local_data / "test"))
    n = min(limit, len(test_ds))
    test_ds = test_ds.select(range(n))
    print(f"Test slice: {len(test_ds)} examples")

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

    print("Loading model + tokenizer (once for the whole sweep)...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    # The chat template emits BOS itself; pair with add_special_tokens=False on encode.
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False
    model = AutoModelForCausalLM.from_pretrained(base_model, token=hf_token, **load_kwargs).eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    print(f"Loaded on {device}")

    bad_words_ids = build_bad_words_ids(tokenizer)
    print(f"Hebrew constraint: banning {len(bad_words_ids or [])} foreign-script tokens")

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=output_repo, repo_type="model", private=True, exist_ok=True)
    remote_dir = f"sweeps/round-{round_num}"

    rows: list[dict] = []
    for c in candidates:
        pid = c["id"]
        print(f"\n--- prompt {pid!r} ---", flush=True)
        rows.extend(
            _generate_for_prompt(
                model, tokenizer, test_ds, c, variant, device, batch_size,
                max_input=max_input, max_new_tokens=max_new_tokens,
                bad_words_ids=bad_words_ids,
            )
        )
        # Push after every candidate so a timeout still leaves the finished prompts scored.
        out_path = Path("predictions.jsonl")
        out_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=f"{remote_dir}/predictions.jsonl",
            repo_id=output_repo,
            repo_type="model",
        )
        print(f"  pushed {len(rows)} rows so far -> {output_repo}/{remote_dir}", flush=True)

    prompts_path = Path("prompts.json")
    prompts_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(prompts_path),
        path_in_repo=f"{remote_dir}/prompts.json",
        repo_id=output_repo,
        repo_type="model",
    )
    print(f"\nDone. Round {round_num}: {len(rows)} rows from {len(candidates)} prompts.")


def _generate_for_prompt(
    model, tokenizer, test_ds, candidate: dict, variant: str, device, batch_size: int,
    *, max_input: int, max_new_tokens: int, bad_words_ids,
) -> list[dict]:
    import torch

    template = candidate["template"]
    pid = candidate["id"]
    tokenizer.padding_side = "left"
    rows: list[dict] = []
    for i in range(0, len(test_ds), batch_size):
        batch = test_ds[i : i + batch_size]
        prompts = [
            build_input_text_safe(
                tokenizer,
                template.replace(
                    TEXT_PLACEHOLDER,
                    truncate_article(tokenizer, template, t, max_input),
                ),
            )
            for t in batch["text"]
        ]
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            outs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=8,
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                bad_words_ids=bad_words_ids,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        input_len = inputs["input_ids"].shape[1]
        for j in range(len(prompts)):
            pred = tokenizer.decode(outs[j][input_len:], skip_special_tokens=True)
            rows.append({
                "text": batch["text"][j],
                "reference": batch["summary"][j],
                "prediction": pred.strip(),
                "model": "base",
                "variant": variant,
                "prompt_id": pid,
            })
        if device.type == "cuda":
            del outs, inputs
            torch.cuda.empty_cache()
        end = min(i + batch_size, len(test_ds))
        print(f"  [{pid}] {end}/{len(test_ds)}", flush=True)
        if i == 0 and rows:
            print(f"  sample: {rows[0]['prediction'][:140]!r}", flush=True)
    tokenizer.padding_side = "right"
    return rows


# --------------------------------------------------------------------------- local side
def submit(
    prompts_file: str,
    hf_user: str,
    *,
    round_num: int = 1,
    limit: int = 100,
    model_id: str = "dicta-il/dictalm2.0-instruct",
    variant: str = "whole",
    flavor: str = "a10g-small",
    timeout: str = "2h",
    max_new_tokens: int = 160,
    batch_size: int = 4,
    dataset_repo: str = "",
    output_repo: str = "",
) -> object:
    """Submit one round (all its prompt candidates) as a single HF Job."""
    from huggingface_hub import HfApi

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN not set. Run: source .env", file=sys.stderr)
        sys.exit(1)

    candidates = json.loads(Path(prompts_file).read_text(encoding="utf-8"))
    for c in candidates:
        if TEXT_PLACEHOLDER not in c["template"]:
            print(f"ERROR: prompt {c['id']!r} has no {TEXT_PLACEHOLDER} placeholder", file=sys.stderr)
            sys.exit(1)

    data_repo = dataset_repo or f"{hf_user}/amlk-training-data"
    out_repo = output_repo or f"{hf_user}/amlk-prompt-arena"

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=out_repo, repo_type="model", private=True, exist_ok=True)

    print(f"Submitting prompt sweep: round {round_num}, "
          f"{len(candidates)} prompts x {limit} examples on {model_id}")
    print(f"  flavor={flavor} timeout={timeout} -> {out_repo}/sweeps/round-{round_num}")

    job = api.run_uv_job(
        script=str(Path(__file__).resolve()),
        flavor=flavor,
        timeout=timeout,
        secrets={"HF_TOKEN": hf_token},
        env={
            "BASE_MODEL": model_id,
            "DATASET_REPO": data_repo,
            "OUTPUT_REPO": out_repo,
            "PROMPTS_JSON": json.dumps(candidates, ensure_ascii=False),
            "ROUND": str(round_num),
            "LIMIT": str(limit),
            "VARIANT": variant,
            "MAX_NEW_TOKENS": str(max_new_tokens),
            "BATCH_SIZE": str(batch_size),
        },
        token=hf_token,
    )
    print(f"\nJob submitted. ID: {job.id}  Status: {job.status.stage}")
    print(f"  Logs:  hf jobs logs {job.id}")
    print(f"  Preds: https://huggingface.co/{out_repo}/tree/main/sweeps/round-{round_num}")
    return job


def download_round(
    round_num: int, hf_user: str, *, output_repo: str = "", dest_root: str = "outputs/results/prompt-arena",
) -> Path:
    """Pull a finished round's predictions.jsonl into outputs/results/prompt-arena/round-<n>/."""
    from huggingface_hub import hf_hub_download

    out_repo = output_repo or f"{hf_user}/amlk-prompt-arena"
    dest = Path(dest_root) / f"round-{round_num}"
    dest.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        out_repo,
        f"sweeps/round-{round_num}/predictions.jsonl",
        repo_type="model",
        token=os.environ.get("HF_TOKEN", ""),
    )
    path = dest / "predictions.jsonl"
    path.write_text(Path(local).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Downloaded {path} ({sum(1 for _ in path.open(encoding='utf-8'))} rows)")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Sweep prompt candidates on HF Jobs (no training)")
    parser.add_argument("--submit-hf", action="store_true", help="Submit this round to HF Jobs")
    parser.add_argument("--download", action="store_true", help="Download a finished round")
    parser.add_argument("--prompts", default="", help="Path to the round's prompts.json")
    parser.add_argument("--round", type=int, default=1, dest="round_num")
    parser.add_argument("--limit", type=int, default=100, help="Test examples per prompt")
    parser.add_argument("--model", default="dicta-il/dictalm2.0-instruct")
    parser.add_argument("--hf-user", default="avreymi")
    parser.add_argument("--variant", choices=("whole", "lead", "body"), default="whole")
    parser.add_argument("--flavor", default="a10g-small")
    parser.add_argument("--timeout", default="2h")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset-repo", default="")
    parser.add_argument("--output-repo", default="")
    args = parser.parse_args()

    if args.download:
        download_round(args.round_num, args.hf_user, output_repo=args.output_repo)
        return

    if args.submit_hf:
        if not args.prompts:
            print("ERROR: --submit-hf needs --prompts <round-N/prompts.json>", file=sys.stderr)
            sys.exit(1)
        submit(
            args.prompts,
            args.hf_user,
            round_num=args.round_num,
            limit=args.limit,
            model_id=args.model,
            variant=args.variant,
            flavor=args.flavor,
            timeout=args.timeout,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            dataset_repo=args.dataset_repo,
            output_repo=args.output_repo,
        )
        return

    run_cloud_job()


if __name__ == "__main__":
    main()
