"""
Evaluation helpers for zero-shot base-model predictions (no training, no LoRA adapter).

Role in the pipeline: multi-model baseline generation that sits next to the trained
DictaLM path — load a named Hub checkpoint as-is, run the shared Hebrew summarization
prompts from the processed test split, and write predictions-base.jsonl for scoring.

Code flow: resolve_load_plan → load_base_model (in infer.py / HF job) → generate →
write_predictions_jsonl → outputs/<model-slug>/predictions-base.jsonl. The HF Jobs
entry point is evaluation/predict_base_hf_job.py (self-contained; cannot import this
module on the cloud side — keep pure helpers here for local tests + notebook reuse).

Execution environment: pure helpers run locally (CPU); model load/generate is remote GPU
only (never on the 8 GB local machine).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


REQUIRED_ROW_FIELDS = ("text", "reference", "prediction", "model", "variant")


def model_slug(model_id: str) -> str:
    """Last path segment of a Hub model id, safe as a local directory name."""
    return model_id.rstrip("/").split("/")[-1]


def local_predictions_dir(model_id: str, root: str | Path = "outputs") -> Path:
    """outputs/<slug>/ — holds predictions-base.jsonl for one base checkpoint."""
    return Path(root) / model_slug(model_id)


def local_predictions_path(model_id: str, root: str | Path = "outputs") -> Path:
    return local_predictions_dir(model_id, root) / "predictions-base.jsonl"


def output_repo_for(hf_user: str, model_id: str) -> str:
    """Private Hub model repo that receives the pushed predictions-base.jsonl."""
    return f"{hf_user}/amlk-preds-{model_slug(model_id)}"


def resolve_load_plan(model_id: str) -> dict[str, Any]:
    """Decide how to load a base checkpoint for zero-shot generation.

    Returns a plain dict so the HF Jobs UV script can re-implement the same branches
    without importing this module:
      kind: "causal" | "multimodal"
      trust_remote_code: bool  (Nemotron custom modeling_*.py)
      quantize_default: bool   (4-bit on 24 GB A10G for ~7–12B models)
    """
    mid = model_id.lower()
    if "gemma-4" in mid or "gemma4" in mid:
        return {
            "kind": "multimodal",
            "trust_remote_code": False,
            "quantize_default": True,
        }
    if "nemotron" in mid:
        # transformers>=5.10 ships native NemotronHForCausalLM (no mamba-ssm remote dep).
        # Hub auto_map still works with trust_remote_code=True, but that path needs mamba-ssm.
        return {
            "kind": "causal",
            "trust_remote_code": False,
            "quantize_default": True,
        }
    return {
        "kind": "causal",
        "trust_remote_code": False,
        "quantize_default": True,
    }


def prediction_row(
    text: str,
    reference: str,
    prediction: str,
    *,
    model: str = "base",
    variant: str = "whole",
) -> dict[str, str]:
    """One JSONL row in the project's standard predictions schema."""
    return {
        "text": text,
        "reference": reference,
        "prediction": prediction,
        "model": model,
        "variant": variant,
    }


def write_predictions_jsonl(rows: Iterable[dict], path: str | Path) -> Path:
    """Write prediction rows as UTF-8 JSONL (one object per line, no trailing blank)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def load_predictions_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def validate_predictions(
    rows: list[dict],
    *,
    expected_n: int | None = 100,
    model: str = "base",
) -> list[str]:
    """Return a list of human-readable problems; empty means OK."""
    problems: list[str] = []
    if expected_n is not None and len(rows) != expected_n:
        problems.append(f"expected {expected_n} rows, got {len(rows)}")
    for i, row in enumerate(rows):
        for field in REQUIRED_ROW_FIELDS:
            if field not in row:
                problems.append(f"row {i}: missing field {field!r}")
                continue
            val = row[field]
            if not isinstance(val, str) or not val.strip():
                problems.append(f"row {i}: empty/non-string {field!r}")
        if row.get("model") != model:
            problems.append(f"row {i}: model={row.get('model')!r}, expected {model!r}")
    if len(rows) >= 2:
        preds = [r.get("prediction", "") for r in rows]
        if len(set(preds)) == 1 and preds[0]:
            problems.append("all predictions are identical (likely a fixture/hardcoded stub)")
    return problems


def build_input_text_safe(tokenizer, prompt: str) -> str:
    """Zero-shot base prompt formatting with chat-template when available.

    Mirrors evaluation.infer.build_input_text(..., label='base') but tolerates templates
    that do not accept enable_thinking= (e.g. older Mistral / DictaLM-2 templates).
    Always appends `/no_think` as soft reinforcement on chat-capable models.
    """
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    content = f"{prompt}\n/no_think"
    messages = [{"role": "user", "content": content}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)
