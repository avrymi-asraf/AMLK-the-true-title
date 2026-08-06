"""Provide Hub and local full-Trainer checkpoint helpers for cross-job resume.

This pure-Python API is used by `pipeline.stage_04_training_experiment.submit` and mirrored inside
the self-contained remote job body. The flow is: list Hub files →
filter complete checkpoint-* dirs → pick auto/exact name → materialize or train
with resume_from_checkpoint. Execution environment: local CLI and HF Jobs CPU
submit path; no GPU required.
"""
from __future__ import annotations

import re
from pathlib import Path

_CKPT_NAME_RE = re.compile(r"^checkpoint-(\d+)$")
_REQUIRED_CKPT_FILES = frozenset({"trainer_state.json", "optimizer.pt", "scheduler.pt"})
_ADAPTER_OR_MODEL = frozenset({
    "adapter_model.safetensors", "adapter_model.bin",
    "model.safetensors", "pytorch_model.bin",
})


def checkpoint_step(name: str) -> int:
    """Parse ``checkpoint-N`` → N; return -1 if the name is not a step checkpoint."""
    m = _CKPT_NAME_RE.match(name)
    return int(m.group(1)) if m else -1


def is_full_trainer_checkpoint_files(filenames: set[str]) -> bool:
    """True when *filenames* include trainer state, optimizer, scheduler, and weights."""
    return _REQUIRED_CKPT_FILES.issubset(filenames) and bool(filenames & _ADAPTER_OR_MODEL)


def is_full_trainer_checkpoint_dir(path: Path | str) -> bool:
    """True when *path* is a directory with a full Trainer checkpoint on disk."""
    p = Path(path)
    if not p.is_dir():
        return False
    return is_full_trainer_checkpoint_files({f.name for f in p.iterdir() if f.is_file()})


def hub_full_trainer_checkpoints(repo_files: list[str]) -> list[str]:
    """Return checkpoint-* names that look like full Trainer checkpoints, sorted by step."""
    by_ckpt: dict[str, set[str]] = {}
    for path in repo_files:
        parts = path.split("/")
        if len(parts) < 2 or not parts[0].startswith("checkpoint-"):
            continue
        by_ckpt.setdefault(parts[0], set()).add(parts[-1])
    names = [n for n, files in by_ckpt.items() if is_full_trainer_checkpoint_files(files)]
    return sorted(names, key=checkpoint_step)


def pick_resume_checkpoint(names: list[str], prefer: str) -> str | None:
    """Pick a resume checkpoint from *names* (already sorted).

    ``prefer`` empty or ``"auto"`` → last (highest step); exact name if present;
    ``ValueError`` if *prefer* is a concrete name not in *names*.
    """
    if not names:
        return None
    key = (prefer or "auto").strip()
    if key in ("", "auto"):
        return names[-1]
    if key in names:
        return key
    raise ValueError(
        f"Resume checkpoint {key!r} not among full Trainer checkpoints on Hub: {names}"
    )
