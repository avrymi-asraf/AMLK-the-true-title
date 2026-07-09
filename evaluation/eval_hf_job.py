#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets>=3.0.0",
#     "transformers>=4.45.0",
#     "torch>=2.4.0",
#     "bert-score>=0.3.13",
#     "rouge-score>=0.1.2",
#     "google-generativeai>=0.8.3",
#     "sentencepiece",
#     "requests>=2.32.0",
#     "huggingface_hub",
# ]
# ///
"""
Evaluation pipeline, remote variant: run the full D1 metric battery on HuggingFace Jobs.
The user's machine has weak internet, so the ~3000 Gemini API round-trips (baseline + judge +
error analysis) and the heavy BERTScore pass run in the cloud instead of locally; only small
JSON reports come back. This one file has two modes: run locally with --submit-hf it uploads
*itself* to HF Jobs (cheap CPU flavor); run with no args (how HF Jobs invokes it) it fetches
the public repo + the Hub predictions/dataset and drives the existing evaluation/ CLIs by
subprocess, pushing each report to the model repo under reports/ as soon as it is produced.

Execution environment: submitted from any machine with HF_TOKEN + GEMINI_API_KEY; the job runs
in an ephemeral HuggingFace Jobs CPU container (no GPU — BERTScore is CPU-pinned, the cheapest path).
"""
import os
import sys

REPO_TARBALL = "https://codeload.github.com/avrymi-asraf/AMLK-the-true-title/tar.gz/refs/heads/main"
SYSTEMS = ("finetuned", "base", "gemini")  # one predictions-<name>.jsonl per system


# --------------------------------------------------------------------------- cloud side
def run_cloud_job():
    """Driven by HF Jobs: fetch code + data, run the eval CLIs, push reports to the Hub."""
    import io
    import json
    import subprocess
    import tarfile
    import urllib.request
    from pathlib import Path
    from shutil import copyfile

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    model_repo = os.environ["MODEL_REPO"]
    dataset_repo = os.environ["DATASET_REPO"]
    variant = os.environ.get("VARIANT", "whole")
    data_profile = os.environ.get("DATA_PROFILE", variant)
    clean = os.environ.get("CLEAN", "0") == "1"
    drop_roundups = os.environ.get("DROP_ROUNDUPS", "0") == "1"
    limit = int(os.environ.get("LIMIT", "0"))
    n_errors = limit if limit else 50
    hf_token = os.environ["HF_TOKEN"]
    api = HfApi(token=hf_token)
    print(f"Eval job: model={model_repo} dataset={dataset_repo} profile={data_profile} "
          f"clean={clean} drop_roundups={drop_roundups} limit={limit or 'full'}")

    # 1. Unpack the public repo so we can reuse evaluation/ + data/ verbatim.
    print("Downloading repo tarball...")
    with urllib.request.urlopen(REPO_TARBALL) as resp:
        tar = tarfile.open(fileobj=io.BytesIO(resp.read()), mode="r:gz")
    root = tar.getnames()[0].split("/")[0]
    tar.extractall()
    repo = Path(root).resolve()
    os.chdir(repo)
    print(f"Repo at {repo}")

    # 2. Pull the model-generated predictions and the test split from the Hub.
    results = repo / "outputs" / "results"
    results.mkdir(parents=True, exist_ok=True)
    for name in ("finetuned", "base"):
        hf_hub_download(model_repo, f"predictions-{name}.jsonl", repo_type="model",
                        local_dir=str(results), token=hf_token)
    snapshot_download(dataset_repo, repo_type="dataset", token=hf_token,
                      local_dir=str(repo / "outputs" / "data" / "processed" / data_profile))

    env = {**os.environ, "PYTHONPATH": str(repo)}

    def step(args):
        print(f"\n$ python -m {' '.join(args)}", flush=True)
        subprocess.run([sys.executable, "-m", *args], cwd=str(repo), env=env, check=True)

    def push(local_name):
        api.upload_file(path_or_fileobj=str(results / local_name), path_in_repo=f"reports/{local_name}",
                        repo_id=model_repo, repo_type="model")
        print(f"  pushed reports/{local_name}")

    def hub_file(name):
        """Local path to reports/<name> on the model repo, or None if it isn't there yet."""
        try:
            return hf_hub_download(model_repo, f"reports/{name}", repo_type="model", token=hf_token)
        except Exception:
            return None

    def line_count(path):
        return sum(1 for _ in open(path, encoding="utf-8"))

    lim = ["--limit", str(limit)] if limit else []

    # 3. Gemini advanced baseline — reuse a complete one already on the Hub (so a re-run after a
    #    crash skips the ~40-min generation).
    cached = hub_file("predictions-gemini.jsonl")
    if cached and not limit and line_count(cached) >= 1000:
        copyfile(cached, results / "predictions-gemini.jsonl")
        print(f"Reusing Gemini baseline from the Hub ({line_count(cached)} rows)")
    else:
        clean_flag = ["--clean"] if clean else []
        step(["evaluation.predict", "--variant", variant,
              "--data", f"outputs/data/processed/{data_profile}/test",
              "--output", "outputs/results/predictions-gemini.jsonl", *clean_flag, *lim])
        push("predictions-gemini.jsonl")

    # 4. Score + error-analyse every system; skip any whose report is already complete (resume),
    #    push each report immediately (timeout-safe).
    for name in SYSTEMS:
        preds = f"outputs/results/predictions-{name}.jsonl"
        n_pred = line_count(results / f"predictions-{name}.jsonl")
        report, errors = f"{name}-{variant}.report.json", f"{name}-{variant}.errors.json"
        done = hub_file(report)
        if done and json.load(open(done)).get("n") == n_pred:
            print(f"Skipping {name}: report already complete (n={n_pred})")
            continue
        step(["evaluation.evaluate", "--predictions", preds,
              "--output", f"outputs/results/{report}", *lim])
        push(report)
        step(["evaluation.error_analysis", "--predictions", preds,
              "--output", f"outputs/results/{errors}", "--n", str(n_errors)])
        push(errors)

    print("\nEvaluation job complete — reports under reports/ in", model_repo)


# --------------------------------------------------------------------------- local side
def submit(hf_user: str, variant: str, smoke: bool, clean: bool = False,
           drop_roundups: bool = False, output_repo: str = "", limit: int | None = None):
    """Upload this script to HF Jobs on a cheap CPU flavor and pass settings as env vars.

    clean=True targets the clean data/model repos and hardened Gemini prompt.
    drop_roundups=True (requires clean) targets the -clean-drop repos. output_repo overrides
    the model repo scored (e.g. a one-off --output-repo repo from train.py that doesn't
    follow the standard -sft[-clean] naming, such as a smoke/mini validation run); the
    dataset repo is still derived normally since the processed splits are shared across
    such repos. limit overrides the example cap (default: 5 with smoke, unlimited otherwise) —
    e.g. matching a --mini-test model's smaller predictions-finetuned/base.jsonl row count so
    all three systems (finetuned/base/gemini) score the same N examples.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    from huggingface_hub import HfApi

    from training.config import dataset_repo, model_repo, processed_profile_name

    if drop_roundups and not clean:
        print("ERROR: drop_roundups requires clean=True", file=sys.stderr)
        sys.exit(1)

    profile = processed_profile_name(variant, clean, drop_roundups)

    hf_token = os.environ.get("HF_TOKEN", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not hf_token or not gemini_key:
        print("ERROR: HF_TOKEN and GEMINI_API_KEY must be set. Run: source .env", file=sys.stderr)
        sys.exit(1)

    out_repo = output_repo or model_repo(hf_user, variant, clean, drop_roundups)
    data_repo = dataset_repo(hf_user, variant, clean, drop_roundups)
    # ~4000 sequential Gemini calls (baseline + judge) can be rate-limited, so allow plenty of
    # wall-clock; each report is pushed as soon as it's ready, so a timeout never loses finished work.
    if limit is not None:
        flavor, timeout, limit_str = "cpu-basic", "30m", str(limit)
    else:
        flavor, timeout, limit_str = ("cpu-basic", "30m", "5") if smoke else ("cpu-upgrade", "5h", "0")
    api = HfApi(token=hf_token)
    drop_label = " drop-roundups" if drop_roundups else ""
    print(f"Submitting {'SMOKE ' if smoke else ''}eval job (flavor={flavor}, timeout={timeout}, limit={limit_str}, clean={clean}{drop_label})...")
    job = api.run_uv_job(
        script=str(__import__("pathlib").Path(__file__).resolve()),
        flavor=flavor,
        timeout=timeout,
        secrets={"HF_TOKEN": hf_token, "GEMINI_API_KEY": gemini_key},
        env={
            "MODEL_REPO": out_repo,
            "DATASET_REPO": data_repo,
            "VARIANT": variant,
            "DATA_PROFILE": profile,
            "CLEAN": "1" if clean else "0",
            "DROP_ROUNDUPS": "1" if drop_roundups else "0",
            "LIMIT": limit_str,
        },
        token=hf_token,
    )
    print(f"\nJob submitted. ID: {job.id}  Status: {job.status.stage}")
    print(f"  Monitor: https://huggingface.co/jobs/{hf_user}/{job.id}")
    print(f"  Logs:    hf jobs logs {job.id} -f")
    print(f"  Reports: https://huggingface.co/{out_repo}/tree/main/reports  (after the run)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the D1 evaluation battery on HuggingFace Jobs")
    parser.add_argument("--submit-hf", action="store_true", help="Submit this script to HF Jobs (cheap CPU)")
    parser.add_argument("--hf-user", default="", help="HuggingFace username (required with --submit-hf)")
    parser.add_argument("--variant", choices=("whole", "lead", "body"), default="whole")
    parser.add_argument("--smoke-test", action="store_true", help="Cap to 5 examples to verify the path cheaply")
    parser.add_argument("--clean", action="store_true",
                        help="Clean profile: score the clean data/model repos with the hardened prompt.")
    parser.add_argument("--drop-roundups", action="store_true",
                        help="With --clean: target the -clean-drop repos (requires --clean).")
    parser.add_argument("--output-repo", default="",
                        help="Override the model repo to score (default: derived from --hf-user/--variant/--clean)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap to N examples (default: 5 with --smoke-test, unlimited otherwise). "
                             "Use to match a --mini-test model's smaller prediction files.")
    args = parser.parse_args()

    if args.drop_roundups and not args.clean:
        print("ERROR: --drop-roundups requires --clean", file=sys.stderr)
        sys.exit(1)

    if args.submit_hf:
        if not args.hf_user:
            print("ERROR: --hf-user required with --submit-hf", file=sys.stderr)
            sys.exit(1)
        submit(args.hf_user, args.variant, args.smoke_test, args.clean, args.drop_roundups,
               args.output_repo, args.limit)
    else:
        run_cloud_job()


if __name__ == "__main__":
    main()
