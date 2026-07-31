"""
Hub Trainer-checkpoint helpers for crash-resume of long SFT jobs.

Fits the training pipeline: full runs save PEFT + optimizer under /data/output and
push adapters with hub_strategy=every_save, but only a *full* Trainer checkpoint
(trainer_state + optimizer) can continue the same epoch. These pure helpers pick
and validate such checkpoints (from a Hub file list or a local dir) so train.py
can upload them and train_hf_job can materialize them under /data/output before
trainer.train(resume_from_checkpoint=...).

Execution environment: local CPU (upload/list) or the HF Jobs container (download).
"""
from __future__ import annotations

import re
from pathlib import Path

# Minimum files for true HF Trainer resume (not adapter-only Hub root).
_REQUIRED_CKPT_FILES = frozenset({
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
})
_ADAPTER_OR_MODEL = frozenset({
    "adapter_model.safetensors",
    "adapter_model.bin",
    "model.safetensors",
    "pytorch_model.bin",
})

_CKPT_NAME_RE = re.compile(r"^checkpoint-(\d+)$")


def checkpoint_step(name: str) -> int:
    """Numeric step from 'checkpoint-200', or -1 if the name does not match."""
    m = _CKPT_NAME_RE.match(name)
    return int(m.group(1)) if m else -1


def is_full_trainer_checkpoint_files(filenames: set[str]) -> bool:
    """True if the file basenames look like a resumable Trainer checkpoint dir."""
    if not _REQUIRED_CKPT_FILES.issubset(filenames):
        return False
    return bool(filenames & _ADAPTER_OR_MODEL)


def is_full_trainer_checkpoint_dir(path: Path | str) -> bool:
    """True if path is a local dir with trainer_state + optimizer + adapter/model."""
    p = Path(path)
    if not p.is_dir():
        return False
    names = {c.name for c in p.iterdir() if c.is_file()}
    return is_full_trainer_checkpoint_files(names)


def hub_full_trainer_checkpoints(repo_files: list[str]) -> list[str]:
    """
    From list_repo_files paths, return checkpoint-* names that include a full
    Trainer state (resumable), sorted by step ascending.
    """
    by_ckpt: dict[str, set[str]] = {}
    for path in repo_files:
        parts = path.split("/")
        if len(parts) < 2 or not parts[0].startswith("checkpoint-"):
            continue
        by_ckpt.setdefault(parts[0], set()).add(parts[-1])

    names = [
        name for name, files in by_ckpt.items()
        if is_full_trainer_checkpoint_files(files)
    ]
    return sorted(names, key=checkpoint_step)


def pick_resume_checkpoint(names: list[str], prefer: str = "auto") -> str | None:
    """
    Choose which checkpoint-* to resume from.

    prefer:
      - "" / "auto" → highest step (or None if names empty)
      - "checkpoint-N" → that name if present, else ValueError
    """
    if not names:
        return None
    key = (prefer or "auto").strip()
    if key in ("", "auto"):
        return names[-1]
    if key in names:
        return key
    raise ValueError(
        f"Resume checkpoint {key!r} not among full Trainer checkpoints: {names}"
    )


def materialize_hub_checkpoint(
    repo_id: str,
    checkpoint_name: str,
    output_dir: str | Path,
    *,
    token: str | None = None,
) -> Path:
    """
    Download OUTPUT_REPO/checkpoint_name/* into output_dir/checkpoint_name.

    Skips the download if a full Trainer checkpoint is already on disk.
    Returns the local checkpoint path.
    """
    from huggingface_hub import snapshot_download

    output_dir = Path(output_dir)
    dest = output_dir / checkpoint_name
    if is_full_trainer_checkpoint_dir(dest):
        return dest

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[f"{checkpoint_name}/*"],
        local_dir=str(output_dir),
        token=token,
    )
    if not is_full_trainer_checkpoint_dir(dest):
        raise FileNotFoundError(
            f"After download, {dest} is not a full Trainer checkpoint "
            f"(need trainer_state.json + optimizer.pt + adapter/model weights)"
        )
    return dest
