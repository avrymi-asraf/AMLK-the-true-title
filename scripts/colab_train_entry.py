"""
Remote Colab entry for AMLK training (pipeline step 3 body launcher).

Uploaded and executed by `training/colab_submit.py` on a Colab GPU VM. Loads
secrets from `/content/.env` (never userdata), applies `/content/amlk_job_env.json`
(METHOD, TRAIN_CONFIG, OUTPUT_DIR=/content/amlk-output, …), pip-installs the
PEP 723 deps declared in `train_hf_job.py`, then runs that script as the sole
training body (chat-wrap, SoftHub, dual-arm preds, Hebrew bad_words_ids).

Execution environment: Colab runtime only. Not for local 7B training.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ENV_FILE = Path(os.environ.get("AMLK_COLAB_ENV_FILE", "/content/.env"))
JOB_ENV_JSON = Path(os.environ.get("AMLK_COLAB_JOB_ENV", "/content/amlk_job_env.json"))
JOB_SCRIPT = Path(os.environ.get("AMLK_COLAB_JOB_SCRIPT", "/content/train_hf_job.py"))
DEFAULT_OUTPUT_DIR = "/content/amlk-output"
DEFAULT_DATA_DIR = "/content/amlk-data"


def load_dotenv(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE parser (no export, supports optional quotes)."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        out[key] = val
    return out


def pep723_dependencies(script_path: Path) -> list[str]:
    text = script_path.read_text(encoding="utf-8")
    m = re.search(r"# /// script\n(.*?)# ///", text, re.DOTALL)
    if not m:
        return []
    block = m.group(1)
    deps: list[str] = []
    in_deps = False
    for line in block.splitlines():
        raw = line.lstrip("#").strip()
        if raw.startswith("dependencies"):
            in_deps = True
            if "[" in raw and "]" in raw and raw.index("]") > raw.index("["):
                inner = raw[raw.index("[") + 1 : raw.index("]")]
                for part in inner.split(","):
                    part = part.strip().strip("\"'")
                    if part:
                        deps.append(part)
                in_deps = False
            continue
        if in_deps:
            if raw.startswith("]"):
                in_deps = False
                continue
            part = raw.strip().rstrip(",").strip("\"'")
            if part:
                deps.append(part)
    return deps


def install_deps(script_path: Path) -> None:
    deps = pep723_dependencies(script_path)
    if not deps:
        print("WARNING: no PEP 723 dependencies found in job script", flush=True)
        return
    print(f"Installing {len(deps)} deps from PEP 723 header...", flush=True)
    # Prefer uv if present (faster); fall back to pip.
    if subprocess.run(["which", "uv"], capture_output=True).returncode == 0:
        cmd = ["uv", "pip", "install", "--system", *deps]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-q", *deps]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)


def apply_env(dotenv: dict[str, str], job_env: dict) -> None:
    # Job JSON first (run config), then .env secrets win for tokens if both set.
    for k, v in job_env.items():
        if v is None:
            continue
        os.environ[str(k)] = str(v)
    for k, v in dotenv.items():
        os.environ[k] = v
    os.environ.setdefault("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    os.environ.setdefault("DATA_DIR", DEFAULT_DATA_DIR)
    Path(os.environ["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["DATA_DIR"]).mkdir(parents=True, exist_ok=True)


def main() -> int:
    if not JOB_SCRIPT.is_file():
        print(f"ERROR: job script missing: {JOB_SCRIPT}", file=sys.stderr)
        return 1
    if not JOB_ENV_JSON.is_file():
        print(f"ERROR: job env JSON missing: {JOB_ENV_JSON}", file=sys.stderr)
        return 1

    import json

    job_env = json.loads(JOB_ENV_JSON.read_text(encoding="utf-8"))
    dotenv = load_dotenv(ENV_FILE)
    if not dotenv.get("HF_TOKEN") and not os.environ.get("HF_TOKEN"):
        print(
            "ERROR: HF_TOKEN not found in /content/.env or environment. "
            "Upload the local .env via colab_submit.",
            file=sys.stderr,
        )
        return 1

    apply_env(dotenv, job_env)
    print(
        f"Colab entry: METHOD={os.environ.get('METHOD')} "
        f"OUTPUT_REPO={os.environ.get('OUTPUT_REPO')} "
        f"OUTPUT_DIR={os.environ.get('OUTPUT_DIR')} "
        f"SMOKE_TEST={os.environ.get('SMOKE_TEST')}",
        flush=True,
    )

    install_deps(JOB_SCRIPT)
    print(f"Running job body: {JOB_SCRIPT}", flush=True)
    # Subprocess so newly installed packages are importable and job top-level runs cleanly.
    rc = subprocess.call([sys.executable, str(JOB_SCRIPT)], env=os.environ.copy())
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
