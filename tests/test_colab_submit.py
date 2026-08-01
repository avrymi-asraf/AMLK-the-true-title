"""
Unit tests for the Colab training submit path (no GPU, no live Colab allocate).

Covers: PEP 723 parse, job-env serialization, argparse --submit-colab, dry-run
wiring, and OUTPUT_DIR/DATA_DIR defaults. Does not load dictalm2 or start a VM.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.colab_submit import (  # noqa: E402
    _merge_secrets_into_dotenv,
    colab_base_cmd,
    default_timeout_seconds,
    pep723_dependencies,
    session_name_for,
    submit_colab_job,
    write_job_env_file,
)
from training.config import (  # noqa: E402
    COLAB_DATA_DIR,
    COLAB_DEFAULT_GPU,
    COLAB_OUTPUT_DIR,
    METHOD_PRESETS,
)
from training.train import build_job_env  # noqa: E402


def test_pep723_deps_from_train_hf_job():
    deps = pep723_dependencies(REPO / "training" / "train_hf_job.py")
    assert "trl>=1.6.0" in deps
    assert "peft>=0.17.0" in deps
    assert "bitsandbytes>=0.44.0" in deps
    assert "wandb" in deps


def test_colab_path_constants():
    assert COLAB_OUTPUT_DIR == "/content/amlk-output"
    assert COLAB_DATA_DIR == "/content/amlk-data"
    assert COLAB_DEFAULT_GPU == "T4"


def test_build_job_env_includes_train_config_and_optional_dirs():
    env, train_cfg, project, run_name = build_job_env(
        "qlora",
        "whole",
        data_repo="avreymi/amlk-training-data",
        out_repo="avreymi/amlk-dictalm2-instruct-colab-smoke",
        base="dicta-il/dictalm2.0-instruct",
        n_epochs=1,
        smoke_test=True,
        mini_test=False,
        inference_only=False,
        resume_key="",
        pred_suffix="",
        max_new_tokens=128,
        max_train=0,
        test_subset=0,
        skip_base_arm=True,
        run_tag="colab",
        learning_rate=0.0,
        repetition_penalty=0.0,
        no_repeat_ngram_size=-1,
        batch_size=0,
        label="smoke",
        output_dir=COLAB_OUTPUT_DIR,
        data_dir=COLAB_DATA_DIR,
    )
    assert env["METHOD"] == "qlora"
    assert env["MODEL_SLUG"] == "dictalm2-instruct"
    assert env["SMOKE_TEST"] == "1"
    assert env["SKIP_BASE_ARM"] == "1"
    assert env["OUTPUT_DIR"] == COLAB_OUTPUT_DIR
    assert env["DATA_DIR"] == COLAB_DATA_DIR
    assert env["OUTPUT_REPO"] == "avreymi/amlk-dictalm2-instruct-colab-smoke"
    cfg = json.loads(env["TRAIN_CONFIG"])
    assert cfg["quantize"] is True
    assert cfg["use_lora"] is True
    assert train_cfg["per_device_train_batch_size"] == METHOD_PRESETS["qlora"]["per_device_train_batch_size"]
    assert project == "amlk-dictalm2-instruct"
    assert "qlora" in run_name and "smoke" in run_name


def test_build_job_env_omits_dirs_when_empty_for_hf_jobs():
    env, _, _, _ = build_job_env(
        "lora",
        "whole",
        data_repo="u/data",
        out_repo="u/model",
        base="dicta-il/dictalm2.0-instruct",
        n_epochs=1,
        smoke_test=False,
        mini_test=False,
        inference_only=False,
        resume_key="",
        pred_suffix="",
        max_new_tokens=64,
        max_train=0,
        test_subset=0,
        skip_base_arm=False,
        run_tag="",
        learning_rate=0.0,
        repetition_penalty=0.0,
        no_repeat_ngram_size=-1,
        batch_size=0,
        label="",
    )
    assert "OUTPUT_DIR" not in env
    assert "DATA_DIR" not in env
    assert env["MAX_NEW_TOKENS"] == "64"


def test_write_job_env_file_strips_secrets(tmp_path):
    p = tmp_path / "env.json"
    write_job_env_file(
        {"METHOD": "qlora", "HF_TOKEN": "secret", "WANDB_API_KEY": "w", "EPOCHS": "1"},
        p,
    )
    data = json.loads(p.read_text())
    assert data["METHOD"] == "qlora"
    assert "HF_TOKEN" not in data
    assert "WANDB_API_KEY" not in data


def test_merge_secrets_into_dotenv():
    text = "HF_TOKEN=already\nFOO=1\n"
    merged = _merge_secrets_into_dotenv(text, {"HF_TOKEN": "new", "WANDB_API_KEY": "wb"})
    assert "HF_TOKEN=already" in merged
    assert "WANDB_API_KEY=wb" in merged
    assert "HF_TOKEN=new" not in merged


def test_session_name_and_timeouts():
    assert session_name_for(True, "colab", "") == "amlk-colab-smoke-colab"
    assert session_name_for(True, "smoke", "") == "amlk-colab-smoke"
    assert session_name_for(False, "e4", "") == "amlk-colab-e4"
    assert session_name_for(False, "", "custom") == "custom"
    assert default_timeout_seconds(True, False, False, "") == 2 * 3600
    assert default_timeout_seconds(False, False, False, "90m") == 90 * 60
    assert default_timeout_seconds(False, False, True, "1h") == 3600


def test_colab_base_cmd_auth_before_subcommand():
    cmd = colab_base_cmd("oauth2", "/tmp/amlk-colab-x.json")
    assert cmd[:3] == ["colab", "--auth=oauth2", "--config=/tmp/amlk-colab-x.json"]


def test_train_hf_job_source_has_output_dir_env():
    src = (REPO / "training" / "train_hf_job.py").read_text(encoding="utf-8")
    assert 'OUTPUT_DIR = os.environ.get("OUTPUT_DIR") or "/data/output"' in src
    assert 'DATA_DIR = os.environ.get("DATA_DIR") or "./data"' in src
    assert "_pred_local_path" in src


def test_argparse_submit_colab_help():
    r = subprocess.run(
        [sys.executable, "-m", "training.train", "--help"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env={**os.environ, "HF_TOKEN": "dummy"},
    )
    assert r.returncode == 0
    assert "--submit-colab" in r.stdout
    assert "--colab-gpu" in r.stdout
    assert "--colab-dry-run" in r.stdout
    assert "--submit-hf" in r.stdout


def test_submit_colab_dry_run_no_vm(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=fake\n", encoding="utf-8")
    env = {
        "METHOD": "qlora",
        "VARIANT": "whole",
        "DATASET_REPO": "u/d",
        "OUTPUT_REPO": "u/m",
        "OUTPUT_DIR": COLAB_OUTPUT_DIR,
        "DATA_DIR": COLAB_DATA_DIR,
        "SMOKE_TEST": "1",
    }
    rc = submit_colab_job(
        env,
        smoke_test=True,
        run_tag="test",
        env_file=str(env_file),
        dry_run=True,
    )
    assert rc == 0


def test_entry_load_dotenv_and_pep723():
    # Import remote entry helpers without executing main.
    import importlib.util

    path = REPO / "scripts" / "colab_train_entry.py"
    spec = importlib.util.spec_from_file_location("colab_train_entry", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    deps = mod.pep723_dependencies(REPO / "training" / "train_hf_job.py")
    assert "transformers>=5.0.0" in deps
    d = mod.load_dotenv  # type: ignore[attr-defined]
    # write temp
    import tempfile

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".env") as f:
        f.write("# c\nHF_TOKEN=abc\nWANDB_API_KEY='xyz'\n")
        name = f.name
    try:
        got = d(Path(name))
        assert got["HF_TOKEN"] == "abc"
        assert got["WANDB_API_KEY"] == "xyz"
    finally:
        Path(name).unlink(missing_ok=True)


def test_mutual_exclusion_submit_flags():
    r = subprocess.run(
        [
            sys.executable, "-m", "training.train",
            "--submit-hf", "--submit-colab",
            "--hf-user", "avreymi", "--smoke-test",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env={**os.environ, "HF_TOKEN": "dummy"},
    )
    assert r.returncode != 0
    assert "only one" in (r.stderr + r.stdout).lower()
