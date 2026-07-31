"""
Behavior tests for Hub full-Trainer checkpoint resume helpers.

These cover the pure pick/validate logic used before submitting a resume job
(and the twin in train_hf_job.py). No GPU, no Hub network.
"""
from pathlib import Path

import pytest

from training.resume import (
    checkpoint_step,
    hub_full_trainer_checkpoints,
    is_full_trainer_checkpoint_dir,
    is_full_trainer_checkpoint_files,
    pick_resume_checkpoint,
)


def test_checkpoint_step_parses_name():
    assert checkpoint_step("checkpoint-200") == 200
    assert checkpoint_step("checkpoint-0") == 0
    assert checkpoint_step("adapter") == -1
    assert checkpoint_step("checkpoint-") == -1


def test_is_full_trainer_checkpoint_files_requires_optimizer_and_adapter():
    full = {
        "trainer_state.json", "optimizer.pt", "scheduler.pt",
        "adapter_model.safetensors", "adapter_config.json",
    }
    assert is_full_trainer_checkpoint_files(full)
    assert not is_full_trainer_checkpoint_files(full - {"optimizer.pt"})
    assert not is_full_trainer_checkpoint_files(full - {"adapter_model.safetensors"})
    # Adapter-only Hub root style — not resumable.
    assert not is_full_trainer_checkpoint_files({
        "adapter_model.safetensors", "adapter_config.json", "training_args.bin",
    })


def test_hub_full_trainer_checkpoints_filters_and_sorts():
    files = [
        "adapter_model.safetensors",
        "checkpoint-100/trainer_state.json",
        "checkpoint-100/optimizer.pt",
        "checkpoint-100/scheduler.pt",
        "checkpoint-100/adapter_model.safetensors",
        "checkpoint-200/trainer_state.json",
        "checkpoint-200/optimizer.pt",
        "checkpoint-200/scheduler.pt",
        "checkpoint-200/adapter_model.safetensors",
        # incomplete mid-save — ignored
        "checkpoint-150/adapter_model.safetensors",
        "checkpoint-150/trainer_state.json",
    ]
    assert hub_full_trainer_checkpoints(files) == ["checkpoint-100", "checkpoint-200"]


def test_pick_resume_checkpoint_auto_and_explicit():
    names = ["checkpoint-100", "checkpoint-200"]
    assert pick_resume_checkpoint(names, "auto") == "checkpoint-200"
    assert pick_resume_checkpoint(names, "") == "checkpoint-200"
    assert pick_resume_checkpoint(names, "checkpoint-100") == "checkpoint-100"
    assert pick_resume_checkpoint([], "auto") is None
    with pytest.raises(ValueError, match="checkpoint-999"):
        pick_resume_checkpoint(names, "checkpoint-999")


def test_is_full_trainer_checkpoint_dir(tmp_path: Path):
    d = tmp_path / "checkpoint-200"
    d.mkdir()
    for name in (
        "trainer_state.json", "optimizer.pt", "scheduler.pt",
        "adapter_model.safetensors",
    ):
        (d / name).write_text("x")
    assert is_full_trainer_checkpoint_dir(d)
    (d / "optimizer.pt").unlink()
    assert not is_full_trainer_checkpoint_dir(d)


def test_train_hf_job_wires_resume_from_env():
    """Structural guard: remote job honors RESUME_FROM and materializes Hub ckpts."""
    src = Path("training/train_hf_job.py").read_text(encoding="utf-8")
    assert 'RESUME_FROM = (os.environ.get("RESUME_FROM")' in src
    assert "_materialize_hub_checkpoint" in src
    assert "_hub_full_trainer_checkpoints" in src
    assert "load_best = not bool(RESUME_FROM)" in src
    assert "Post-resume final eval" in src


def test_train_py_ships_resume_from():
    src = Path("training/train.py").read_text(encoding="utf-8")
    assert '"RESUME_FROM": resume_key' in src
    assert "--resume-from" in src
    assert "--push-resume-checkpoint" in src
    assert "push_resume_checkpoint" in src
