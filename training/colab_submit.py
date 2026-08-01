"""
Colab remote submit path for pipeline step 3 (training).

Shares the same job env payload and sole training body (`training/train_hf_job.py`)
as `--submit-hf`, but launches on Google Colab via the official `colab` CLI instead
of HuggingFace Jobs. Local code: preflight auth, isolate session config, upload
`.env` + scripts, run the remote entry; never loads the 7B model on this machine.

Execution environment: local machine only (shells out to `colab`). Called by
`python -m training.train --submit-colab`. Prefer `--method qlora` on T4.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from training.config import COLAB_DATA_DIR, COLAB_DEFAULT_GPU, COLAB_OUTPUT_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_HF_JOB = Path(__file__).resolve().parent / "train_hf_job.py"
REMOTE_ENTRY = REPO_ROOT / "scripts" / "colab_train_entry.py"
REMOTE_ENV_JSON = "/content/amlk_job_env.json"
REMOTE_JOB_SCRIPT = "/content/train_hf_job.py"
REMOTE_ENTRY_PATH = "/content/colab_train_entry.py"
REMOTE_ENV_FILE = "/content/.env"

# jupyter-kernel-client 1.x renamed JupyterKernelClient; pin until google-colab-cli catches up.
KERNEL_CLIENT_PIN = "jupyter-kernel-client==0.15.0"


def pep723_dependencies(script_path: Path | str) -> list[str]:
    """Parse PEP 723 dependency list from a UV script header."""
    text = Path(script_path).read_text(encoding="utf-8")
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
                # single-line list
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


def colab_base_cmd(auth: str, config_path: str) -> list[str]:
    """Global flags before the subcommand: --auth=oauth2 --config /tmp/..."""
    return ["colab", f"--auth={auth}", f"--config={config_path}"]


def run_colab(
    args: list[str],
    *,
    auth: str,
    config_path: str,
    check: bool = True,
    capture: bool = False,
) -> subprocess.CompletedProcess:
    """Run `colab --auth=... --config=... <subcommand...>`. Never uses unpiped repl/console."""
    cmd = colab_base_cmd(auth, config_path) + args
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture,
    )


def ensure_kernel_client_pin() -> None:
    """Pin jupyter-kernel-client if import path would break google-colab-cli exec/run."""
    try:
        import jupyter_kernel_client  # type: ignore

        ver = getattr(jupyter_kernel_client, "__version__", "")
        # 0.15.x is known-good; 1.x renames JupyterKernelClient.
        if ver.startswith("0.15") or ver.startswith("0.14") or ver.startswith("0.13"):
            return
        print(
            f"WARNING: jupyter-kernel-client={ver!r}; pinning {KERNEL_CLIENT_PIN} "
            f"(1.x breaks colab exec/run until google-colab-cli updates).",
            flush=True,
        )
    except ImportError:
        print(f"Installing {KERNEL_CLIENT_PIN} for colab CLI kernel client...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", KERNEL_CLIENT_PIN],
        check=False,
    )


def preflight(auth: str, config_path: str) -> None:
    """Verify colab binary, oauth2 identity, and empty-or-listed sessions before GPU work."""
    if shutil.which("colab") is None:
        print(
            "ERROR: `colab` CLI not on PATH. Install: uv tool install google-colab-cli",
            file=sys.stderr,
        )
        sys.exit(1)
    ensure_kernel_client_pin()
    who = run_colab(["whoami"], auth=auth, config_path=config_path, capture=True)
    print(who.stdout or who.stderr or "(whoami ok)", flush=True)
    sess = run_colab(["sessions"], auth=auth, config_path=config_path, capture=True)
    print(sess.stdout or sess.stderr or "(sessions ok)", flush=True)


def session_name_for(smoke_test: bool, run_tag: str, session_override: str) -> str:
    if session_override:
        return session_override
    tag = (run_tag or "train").replace(" ", "-")
    if smoke_test:
        return f"amlk-colab-smoke-{tag}" if tag != "smoke" else "amlk-colab-smoke"
    return f"amlk-colab-{tag}"


def default_timeout_seconds(
    smoke_test: bool, mini_test: bool, inference_only: bool, timeout_override: str,
) -> int:
    """Map HF-style timeout strings or empty → seconds for colab --timeout."""
    if timeout_override:
        s = timeout_override.strip().lower()
        if s.endswith("h"):
            return int(float(s[:-1]) * 3600)
        if s.endswith("m"):
            return int(float(s[:-1]) * 60)
        if s.endswith("s"):
            return int(float(s[:-1]))
        return int(float(s))
    if inference_only:
        return 5 * 3600
    if smoke_test:
        return 2 * 3600  # 10-step qlora + dual-arm headroom on T4
    if mini_test:
        return 3 * 3600
    return 10 * 3600  # full epoch on T4 is slow; prefer HF Jobs for production


def write_job_env_file(env: dict[str, str], path: Path) -> None:
    """Write non-secret job env for the remote entry (secrets come from uploaded .env)."""
    # Do not put token material in the JSON that lands in colab logs via cat.
    safe = {k: v for k, v in env.items() if k not in ("HF_TOKEN", "WANDB_API_KEY")}
    path.write_text(json.dumps(safe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _merge_secrets_into_dotenv(env_text: str, secrets: dict[str, str]) -> str:
    """Ensure HF_TOKEN / WANDB_API_KEY are present in the uploaded .env (from netrc etc.)."""
    if not secrets:
        return env_text
    lines = env_text.splitlines()
    keys_present = set()
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        keys_present.add(s.split("=", 1)[0].strip())
    out = list(lines)
    for key, val in secrets.items():
        if not val or key in keys_present:
            continue
        out.append(f"{key}={val}")
    return "\n".join(out) + ("\n" if out else "")


def upload_file(
    local: Path, remote: str, *, auth: str, config_path: str, session: str | None,
) -> None:
    args = ["upload"]
    if session:
        args += ["-s", session]
    args += [str(local), remote]
    run_colab(args, auth=auth, config_path=config_path)


def stop_session(session: str, *, auth: str, config_path: str) -> None:
    try:
        run_colab(["stop", "-s", session], auth=auth, config_path=config_path, check=False)
    except Exception as exc:
        print(f"WARNING: colab stop failed: {exc}", file=sys.stderr)
    sess = run_colab(["sessions"], auth=auth, config_path=config_path, capture=True, check=False)
    print("Post-stop sessions:", (sess.stdout or sess.stderr or "").strip(), flush=True)


def submit_colab_job(
    env: dict[str, str],
    *,
    smoke_test: bool = False,
    mini_test: bool = False,
    inference_only: bool = False,
    run_tag: str = "",
    gpu: str = COLAB_DEFAULT_GPU,
    timeout: str = "",
    session: str = "",
    mode: str = "auto",
    auth: str = "oauth2",
    env_file: str = "",
    dry_run: bool = False,
) -> int:
    """Launch train_hf_job.py on Colab. mode: auto|durable|run. Returns CLI exit code.

    auto → durable for smoke (upload + multi-exec friendly), colab run otherwise.
    Secrets: upload local .env (or env_file) to /content/.env — never userdata.
    """
    if not TRAIN_HF_JOB.is_file():
        print(f"ERROR: missing job body {TRAIN_HF_JOB}", file=sys.stderr)
        sys.exit(1)
    if not REMOTE_ENTRY.is_file():
        print(f"ERROR: missing remote entry {REMOTE_ENTRY}", file=sys.stderr)
        sys.exit(1)

    # Colab path overrides (HF Jobs leave these unset → defaults in train_hf_job.py).
    job_env = dict(env)
    job_env.setdefault("OUTPUT_DIR", COLAB_OUTPUT_DIR)
    job_env.setdefault("DATA_DIR", COLAB_DATA_DIR)

    tag = run_tag or ("smoke" if smoke_test else "train")
    config_path = f"/tmp/amlk-colab-{tag}.json"
    sess_name = session_name_for(smoke_test, run_tag, session)
    timeout_s = default_timeout_seconds(smoke_test, mini_test, inference_only, timeout)

    if mode == "auto":
        mode = "durable" if smoke_test else "run"
    if mode not in ("durable", "run"):
        print(f"ERROR: unknown colab mode {mode!r} (use auto|durable|run)", file=sys.stderr)
        sys.exit(1)

    local_env = Path(env_file) if env_file else REPO_ROOT / ".env"
    if not local_env.is_file():
        print(
            f"ERROR: {local_env} not found — Colab injects secrets by uploading .env "
            f"(not userdata). Create it or pass --colab-env-file.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Colab submit: mode={mode} gpu={gpu} session={sess_name} timeout={timeout_s}s")
    print(f"  OUTPUT_DIR={job_env.get('OUTPUT_DIR')}  DATA_DIR={job_env.get('DATA_DIR')}")
    print(f"  METHOD={job_env.get('METHOD')}  OUTPUT_REPO={job_env.get('OUTPUT_REPO')}")
    print(f"  config={config_path}  auth={auth}")
    if job_env.get("METHOD") == "lora" and not inference_only:
        print(
            "WARNING: bf16 lora on Colab T4 (16 GB) often OOMs for 7B; prefer --method qlora.",
            file=sys.stderr,
        )

    if dry_run:
        print("Dry-run: would preflight, upload .env + scripts, and run remote entry.")
        print(f"  pep723 deps: {pep723_dependencies(TRAIN_HF_JOB)}")
        return 0

    preflight(auth, config_path)

    with tempfile.TemporaryDirectory(prefix="amlk-colab-") as tmp:
        tmp_path = Path(tmp)
        env_json_path = tmp_path / "amlk_job_env.json"
        # Secrets stay out of the non-secret JSON; merge token keys into the uploaded .env.
        write_job_env_file(job_env, env_json_path)
        env_text = _merge_secrets_into_dotenv(
            local_env.read_text(encoding="utf-8"),
            {k: job_env[k] for k in ("HF_TOKEN", "WANDB_API_KEY") if job_env.get(k)},
        )
        env_upload = tmp_path / ".env"
        env_upload.write_text(env_text, encoding="utf-8")

        if mode == "run":
            # One-shot self-cleaning VM: bootstrap embeds files (colab run only sends one script).
            bootstrap = tmp_path / "colab_bootstrap.py"
            _write_run_bootstrap(
                bootstrap,
                job_env={k: v for k, v in job_env.items() if k not in ("HF_TOKEN", "WANDB_API_KEY")},
                job_script_text=TRAIN_HF_JOB.read_text(encoding="utf-8"),
                entry_text=REMOTE_ENTRY.read_text(encoding="utf-8"),
                env_text=env_text,
            )
            result = run_colab(
                [
                    "run",
                    "--gpu", gpu,
                    "--timeout", str(timeout_s),
                    "-s", sess_name,
                    str(bootstrap),
                ],
                auth=auth,
                config_path=config_path,
                check=False,
            )
            sess = run_colab(
                ["sessions"], auth=auth, config_path=config_path, capture=True, check=False,
            )
            print("Post-run sessions:", (sess.stdout or sess.stderr or "").strip(), flush=True)
            return int(result.returncode)

        # durable: new → upload → exec → always stop
        try:
            run_colab(
                ["new", "-s", sess_name, "--gpu", gpu],
                auth=auth,
                config_path=config_path,
            )
            upload_file(env_upload, REMOTE_ENV_FILE, auth=auth, config_path=config_path, session=sess_name)
            upload_file(env_json_path, REMOTE_ENV_JSON, auth=auth, config_path=config_path, session=sess_name)
            upload_file(TRAIN_HF_JOB, REMOTE_JOB_SCRIPT, auth=auth, config_path=config_path, session=sess_name)
            upload_file(REMOTE_ENTRY, REMOTE_ENTRY_PATH, auth=auth, config_path=config_path, session=sess_name)
            result = run_colab(
                [
                    "exec",
                    "-s", sess_name,
                    "-f", str(REMOTE_ENTRY),
                    "--timeout", str(timeout_s),
                ],
                auth=auth,
                config_path=config_path,
                check=False,
            )
            return int(result.returncode)
        finally:
            stop_session(sess_name, auth=auth, config_path=config_path)


def _write_run_bootstrap(
    path: Path,
    *,
    job_env: dict[str, str],
    job_script_text: str,
    entry_text: str,
    env_text: str,
) -> None:
    """Self-contained script for `colab run`: materialize files then exec entry main."""
    # Embed payloads as base64 to avoid quote hell with Hebrew/special chars in .env.
    import base64

    def b64(s: str) -> str:
        return base64.b64encode(s.encode("utf-8")).decode("ascii")

    path.write_text(
        f'''# Auto-generated AMLK Colab run bootstrap — do not commit.
import base64, json, os, runpy, sys
from pathlib import Path

def _w(p, b64s):
    Path(p).write_bytes(base64.b64decode(b64s))

_w("/content/.env", {b64(env_text)!r})
_w("/content/amlk_job_env.json", {b64(json.dumps(job_env, ensure_ascii=False))!r})
_w("/content/train_hf_job.py", {b64(job_script_text)!r})
_w("/content/colab_train_entry.py", {b64(entry_text)!r})
sys.argv = ["colab_train_entry.py"]
runpy.run_path("/content/colab_train_entry.py", run_name="__main__")
''',
        encoding="utf-8",
    )
