"""Submit one final-paper E4 LoRA arm to Hugging Face Jobs.

This is the only supported E4 launcher. It submits the self-contained
``train_and_generate.py`` body with the exact matched-arm configuration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pipeline.stage_04_training_experiment.config import (
    DEFAULT_MAX_NEW_TOKENS,
    DATASET_REPOS,
    MODEL_ID,
    MODEL_REPOS,
    MODEL_SLUG,
    TEST_SUBSET_N,
    TEST_SUBSET_SEED,
    lora_payload,
    training_payload,
)
from pipeline.stage_04_training_experiment.resume import (
    hub_full_trainer_checkpoints,
    pick_resume_checkpoint,
)


def build_job_environment(arm: str, *, resume_from: str = "", inference_only: bool = False) -> dict[str, str]:
    if arm not in DATASET_REPOS:
        raise ValueError(f"unknown E4 arm: {arm!r}")
    return {
        "METHOD": "lora",
        "VARIANT": "whole",
        "BASE_MODEL": MODEL_ID,
        "MODEL_SLUG": MODEL_SLUG,
        "DATASET_REPO": DATASET_REPOS[arm],
        "OUTPUT_REPO": MODEL_REPOS[arm],
        "WANDB_PROJECT": "amlk-dictalm2-instruct",
        "WANDB_RUN_NAME": f"e4-{arm}",
        "INFERENCE_ONLY": "1" if inference_only else "0",
        "RESUME_FROM": resume_from,
        "PRED_SUFFIX": "",
        "EPOCHS": "1",
        "MAX_NEW_TOKENS": str(DEFAULT_MAX_NEW_TOKENS),
        "TEST_SUBSET_N": str(TEST_SUBSET_N),
        "TEST_SUBSET_SEED": str(TEST_SUBSET_SEED),
        "REPETITION_PENALTY": "1.0",
        "NO_REPEAT_NGRAM_SIZE": "0",
        "TRAIN_CONFIG": json.dumps(training_payload()),
        "LORA_CONFIG": json.dumps(lora_payload()),
    }


def resolve_resume(api, output_repo: str, requested: str) -> str:
    if not requested:
        return ""
    available = hub_full_trainer_checkpoints(api.list_repo_files(output_repo, repo_type="model"))
    chosen = pick_resume_checkpoint(available, requested)
    if chosen is None:
        raise ValueError(f"{output_repo} has no complete Trainer checkpoint")
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(DATASET_REPOS), required=True)
    parser.add_argument("--resume-from", default="", help="auto or checkpoint-N")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--timeout", default="", help="Override the HF Jobs timeout")
    args = parser.parse_args(argv)

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required to submit an E4 job")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    output_repo = MODEL_REPOS[args.arm]
    resume_from = resolve_resume(api, output_repo, args.resume_from)
    environment = build_job_environment(
        args.arm,
        resume_from=resume_from,
        inference_only=args.inference_only,
    )
    secrets = {"HF_TOKEN": token}
    if os.environ.get("WANDB_API_KEY"):
        secrets["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    script = Path(__file__).with_name("train_and_generate.py")
    timeout = args.timeout or ("5h" if args.inference_only else "4h" if resume_from else "8h")
    job = api.run_uv_job(
        script=str(script),
        flavor="a10g-small",
        timeout=timeout,
        secrets=secrets,
        env=environment,
        token=token,
    )
    print(f"Submitted E4 {args.arm} job: {job.id} ({job.status.stage})")
    print(f"Dataset: {DATASET_REPOS[args.arm]}")
    print(f"Adapter and predictions: {output_repo}")
    if resume_from:
        print(f"Resume checkpoint: {resume_from}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
