#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=1.6.0",
#     "peft>=0.17.0",
#     "transformers>=5.0.0",
#     "accelerate>=1.0.0",
#     "bitsandbytes>=0.44.0",
#     "datasets>=3.0.0",
#     "huggingface_hub",
#     "wandb",
# ]
# ///
"""
Pipeline step 3b (remote): preference optimization (DPO) on top of an SFT adapter, on HF Jobs.

Where SFT (train_hf_job.py) teaches "produce this summary", DPO teaches "prefer this summary over
that one" from pairs that differ only in faithfulness. It is the one technique in the improvement
loop whose mechanism targets the failure tail rather than the average — see
docs/training-improvement-notebook.md entry #12 for why the SFT arms plateaued. Loads the SFT
adapter as the trainable policy (its frozen copy is the implicit reference model), trains on a
{prompt, chosen, rejected} dataset, then generates test predictions exactly like the SFT job.

Submit with: python -m training.dpo_hf_job --submit-hf --pairs <repo> --sft-adapter <repo> ...
Execution environment: ephemeral HuggingFace Jobs GPU container (never run the training half
locally — this machine cannot hold the model).
"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path


def _run_job():
    """The remote half: everything below runs inside the HF Jobs container."""
    import torch
    import wandb
    from datasets import load_from_disk
    from huggingface_hub import HfApi, snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    MODEL_ID = os.environ.get("BASE_MODEL") or "dicta-il/dictalm2.0-instruct"
    PAIRS_REPO = os.environ["PAIRS_REPO"]
    SFT_ADAPTER = os.environ["SFT_ADAPTER"]
    OUTPUT_REPO = os.environ["OUTPUT_REPO"]
    DATASET_REPO = os.environ["DATASET_REPO"]        # for the test split used by generation
    BETA = float(os.environ.get("DPO_BETA") or 0.1)
    LR = float(os.environ.get("DPO_LR") or 5e-6)
    EPOCHS = float(os.environ.get("EPOCHS") or 1)
    MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS") or 128)
    MAX_LENGTH = int(os.environ.get("MAX_LENGTH") or 4096)
    TEST_SUBSET_N = int(os.environ.get("TEST_SUBSET_N") or 0)
    TEST_SUBSET_SEED = int(os.environ.get("TEST_SUBSET_SEED") or 1234)
    SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
    REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY") or 1.0)
    NO_REPEAT_NGRAM_SIZE = int(os.environ.get("NO_REPEAT_NGRAM_SIZE") or 0)

    print(f"DPO: base={MODEL_ID} adapter={SFT_ADAPTER} pairs={PAIRS_REPO} -> {OUTPUT_REPO}")
    print(f"beta={BETA} lr={LR} epochs={EPOCHS} smoke={SMOKE}")

    pairs_dir = Path("./pairs")
    snapshot_download(repo_id=PAIRS_REPO, repo_type="dataset", local_dir=str(pairs_dir))
    pairs = load_from_disk(str(pairs_dir / "train"))
    data_dir = Path("./data")
    snapshot_download(repo_id=DATASET_REPO, repo_type="dataset", local_dir=str(data_dir))
    test_ds = load_from_disk(str(data_dir / "test"))
    if SMOKE:
        pairs = pairs.select(range(min(32, len(pairs))))
        test_ds = test_ds.select(range(4))
    elif TEST_SUBSET_N and TEST_SUBSET_N < len(test_ds):
        import random as _random
        idx = sorted(_random.Random(TEST_SUBSET_SEED).sample(range(len(test_ds)), TEST_SUBSET_N))
        test_ds = test_ds.select(idx)
    print(f"Pairs: {len(pairs)}  Test: {len(test_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Twin of data.prompts.prepare_tokenizer_for_templated_prompts (no repo import here).
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False

    def format_chat_prompt(prompt: str) -> str:
        if not getattr(tokenizer, "chat_template", None):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    pairs = pairs.map(lambda ex: {**ex, "prompt": format_chat_prompt(ex["prompt"])})

    # 4-bit so the policy, its LoRA and the KV cache for generation all fit an a10g (22 GB).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map={"": 0}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        ),
    )
    # The SFT adapter is the starting policy AND (frozen, via adapter disabling inside TRL) the
    # reference model — so no second 7B copy is loaded.
    model = PeftModel.from_pretrained(model, SFT_ADAPTER, is_trainable=True)
    model.config.use_cache = False
    print("Loaded SFT adapter as the trainable DPO policy")

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=OUTPUT_REPO, repo_type="model", private=True, exist_ok=True)
    wandb.init(project=os.environ.get("WANDB_PROJECT") or "amlk-dictalm2-instruct",
               name=os.environ.get("WANDB_RUN_NAME") or "dpo", reinit=True)

    # DPOConfig's field set moves between trl releases (this job hit a container where
    # `max_prompt_length` no longer exists), and a self-contained job script cannot pin the
    # version it will get. Filter to the fields this build actually accepts, and say which were
    # dropped, so an API change degrades to a logged omission instead of a crash 12 minutes in.
    import inspect

    wanted = dict(
        output_dir="/data/output",
        beta=BETA,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        max_steps=10 if SMOKE else -1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_LENGTH - 256,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=1,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb",
        push_to_hub=True,
        hub_model_id=OUTPUT_REPO,
        hub_strategy="every_save",
        hub_private_repo=True,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
    )
    accepted = set(inspect.signature(DPOConfig.__init__).parameters)
    dropped = sorted(k for k in wanted if k not in accepted)
    if dropped:
        print(f"DPOConfig in this trl build does not accept: {dropped} — omitting")
    cfg = DPOConfig(**{k: v for k, v in wanted.items() if k in accepted})
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=pairs, processing_class=tokenizer)
    trainer.train()
    trainer.save_model("/data/output/final")
    trainer.push_to_hub()
    print("DPO training done; adapter pushed")

    # ---- generation, mirroring train_hf_job.py's decode config exactly ----
    def build_bad_words_ids():
        bad = []
        for token, tid in tokenizer.get_vocab().items():
            s = tokenizer.convert_tokens_to_string([token])
            if any(
                (0x41 <= ord(c) <= 0x5A) or (0x61 <= ord(c) <= 0x7A)
                or 0x0400 <= ord(c) <= 0x04FF or 0x0370 <= ord(c) <= 0x03FF
                or 0x0600 <= ord(c) <= 0x06FF
                or 0x3000 <= ord(c) <= 0x9FFF or 0xAC00 <= ord(c) <= 0xD7AF
                for c in s
            ):
                bad.append([tid])
        print(f"Hebrew-script constraint: forbidding {len(bad)} foreign-script tokens")
        return bad

    bad_words_ids = build_bad_words_ids()
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    rows = []
    for i in range(len(test_ds)):
        ex = test_ds[i]
        inputs = tokenizer(
            [format_chat_prompt(ex["prompt"])], return_tensors="pt", truncation=True,
            max_length=MAX_LENGTH - MAX_NEW_TOKENS, add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=min(16, MAX_NEW_TOKENS),
                do_sample=False, no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                repetition_penalty=REPETITION_PENALTY,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                bad_words_ids=bad_words_ids,
            )
        pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rows.append({"text": ex["text"], "reference": ex["summary"],
                     "prediction": pred.strip(), "model": "dpo", "variant": "whole"})
        if (i + 1) % 10 == 0 or i + 1 == len(test_ds):
            print(f"  [dpo] {i + 1}/{len(test_ds)}")
        if (i + 1) % 50 == 0 or i + 1 == len(test_ds):
            path = Path("predictions-dpo.jsonl")
            path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                            encoding="utf-8")
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                            repo_id=OUTPUT_REPO, repo_type="model")
            print(f"  Pushed {len(rows)} predictions")
    wandb.finish()
    print(f"Done → https://huggingface.co/{OUTPUT_REPO}")


def _submit(args):
    """The local half: ship this file to HF Jobs with the settings in its environment."""
    from huggingface_hub import HfApi

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training.config import DEFAULT_MAX_NEW_TOKENS, MAX_LENGTH, MODEL_ID
    from training.train import wandb_api_key

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    env = {
        "BASE_MODEL": MODEL_ID,
        "PAIRS_REPO": args.pairs,
        "SFT_ADAPTER": args.sft_adapter,
        "DATASET_REPO": args.dataset_repo,
        "OUTPUT_REPO": args.output_repo,
        "DPO_BETA": str(args.beta),
        "DPO_LR": str(args.learning_rate),
        "EPOCHS": str(args.epochs),
        "MAX_LENGTH": str(MAX_LENGTH),
        "MAX_NEW_TOKENS": str(DEFAULT_MAX_NEW_TOKENS),
        "TEST_SUBSET_N": str(args.test_subset),
        "SMOKE_TEST": "1" if args.smoke_test else "0",
        "WANDB_PROJECT": "amlk-dictalm2-instruct",
        "WANDB_RUN_NAME": args.run_name or ("dpo-smoke" if args.smoke_test else "dpo"),
    }
    job = api.run_uv_job(
        script=str(Path(__file__)),
        flavor="a10g-small",
        timeout=args.timeout,
        secrets={"HF_TOKEN": token, "WANDB_API_KEY": wandb_api_key()},
        env=env,
        token=token,
    )
    print(f"Job submitted. ID: {job.id}  Status: {job.status.stage}")
    print(f"  Logs:  hf jobs logs {job.id}")
    print(f"  Model: https://huggingface.co/{args.output_repo}")


if __name__ == "__main__":
    if os.environ.get("PAIRS_REPO"):          # running inside the job container
        _run_job()
    else:
        p = argparse.ArgumentParser(description="DPO on top of an SFT adapter (HF Jobs)")
        p.add_argument("--submit-hf", action="store_true", required=True)
        p.add_argument("--pairs", required=True, help="Hub dataset of {prompt,chosen,rejected}")
        p.add_argument("--sft-adapter", required=True, help="Hub repo of the SFT LoRA to start from")
        p.add_argument("--dataset-repo", required=True, help="Dataset providing the test split")
        p.add_argument("--output-repo", required=True, help="NEW Hub repo for the DPO adapter")
        p.add_argument("--beta", type=float, default=0.1)
        p.add_argument("--learning-rate", type=float, default=5e-6)
        p.add_argument("--epochs", type=float, default=1)
        p.add_argument("--test-subset", type=int, default=120)
        p.add_argument("--smoke-test", action="store_true")
        p.add_argument("--timeout", default="3h")
        p.add_argument("--run-name", default="")
        _submit(p.parse_args())
