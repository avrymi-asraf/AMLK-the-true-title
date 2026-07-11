"""Behavior tests for zero-shot base prediction helpers (no GPU, no Hub).

Covers load-plan routing for the three comparison models, JSONL write/validate schema,
chat-template formatting, and structural guarantees that the HF Jobs script is base-only
(no training entry points).
"""
from pathlib import Path

import pytest

from evaluation.base_predict import (
    build_input_text_safe,
    load_predictions_jsonl,
    local_predictions_path,
    model_slug,
    output_repo_for,
    prediction_row,
    resolve_load_plan,
    validate_predictions,
    write_predictions_jsonl,
)


MODELS = [
    "dicta-il/dictalm2.0-instruct",
    "dicta-il/DictaLM-3.0-Nemotron-12B-Instruct",
    "google/gemma-4-12B-it",
]


def test_model_slug_and_paths():
    assert model_slug("dicta-il/dictalm2.0-instruct") == "dictalm2.0-instruct"
    assert model_slug("google/gemma-4-12B-it") == "gemma-4-12B-it"
    path = local_predictions_path("dicta-il/dictalm2.0-instruct", root="/tmp/out")
    assert path == Path("/tmp/out/dictalm2.0-instruct/predictions-base.jsonl")
    assert output_repo_for("avreymi", "google/gemma-4-12B-it") == "avreymi/amlk-preds-gemma-4-12B-it"


def test_resolve_load_plan_for_three_models():
    dicta2 = resolve_load_plan("dicta-il/dictalm2.0-instruct")
    assert dicta2["kind"] == "causal"
    assert dicta2["trust_remote_code"] is False

    nemo = resolve_load_plan("dicta-il/DictaLM-3.0-Nemotron-12B-Instruct")
    assert nemo["kind"] == "causal"
    # Native transformers NemotronH — avoid mamba-ssm remote-code path by default.
    assert nemo["trust_remote_code"] is False

    gemma = resolve_load_plan("google/gemma-4-12B-it")
    assert gemma["kind"] == "multimodal"
    assert gemma["trust_remote_code"] is False


def test_write_and_validate_predictions_schema(tmp_path):
    rows = [
        prediction_row(f"article {i}", f"ref {i}", f"pred {i} unique-{i}", variant="whole")
        for i in range(100)
    ]
    path = write_predictions_jsonl(rows, tmp_path / "predictions-base.jsonl")
    loaded = load_predictions_jsonl(path)
    assert len(loaded) == 100
    assert validate_predictions(loaded, expected_n=100) == []
    assert all(r["model"] == "base" for r in loaded)
    assert all(r["variant"] == "whole" for r in loaded)


def test_validate_rejects_empty_prediction_and_wrong_count():
    rows = [prediction_row("t", "r", "ok")]
    problems = validate_predictions(rows, expected_n=100)
    assert any("expected 100" in p for p in problems)

    bad = [prediction_row("t", "r", "   ")]
    problems = validate_predictions(bad, expected_n=1)
    assert any("empty" in p for p in problems)


def test_validate_rejects_identical_predictions():
    rows = [prediction_row("t1", "r1", "SAME"), prediction_row("t2", "r2", "SAME")]
    problems = validate_predictions(rows, expected_n=2)
    assert any("identical" in p for p in problems)


class _FakeTok:
    """Minimal stand-in that exercises chat-template vs raw-prompt branches."""

    def __init__(self, template=None):
        self.chat_template = template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        assert tokenize is False
        user = messages[0]["content"]
        return f"<user>{user}</user><assistant>" if add_generation_prompt else f"<user>{user}</user>"


def test_build_input_text_safe_uses_chat_template_when_present():
    tok = _FakeTok(template="yes")
    out = build_input_text_safe(tok, "Summarize:\nhello")
    assert out.startswith("<user>Summarize:\nhello")
    assert "/no_think" in out
    assert out.endswith("<assistant>")


def test_build_input_text_safe_raw_when_no_template():
    tok = _FakeTok(template=None)
    assert build_input_text_safe(tok, "raw prompt") == "raw prompt"


def test_build_input_text_safe_tolerates_no_enable_thinking():
    class NoThinkingTok:
        chat_template = "x"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
            if "enable_thinking" in kw:
                raise TypeError("unexpected enable_thinking")
            return f"FMT:{messages[0]['content']}"

    assert build_input_text_safe(NoThinkingTok(), "p") == "FMT:p\n/no_think"


def test_hf_job_script_is_base_only_no_training():
    """Structural guard: the remote job must not train or attach LoRA for training."""
    src = Path("evaluation/predict_base_hf_job.py").read_text(encoding="utf-8")
    assert "SFTTrainer" not in src
    assert "trainer.train" not in src
    assert "LoraConfig" not in src
    assert "PeftModel" not in src
    assert "predictions-base" in src
    assert "run_cloud_job" in src
    # Nemotron ships tokenizer.json only; job must probe Hebrew and force fast tokenizer.
    assert "_load_causal_tokenizer" in src
    assert "PreTrainedTokenizerFast" in src
    assert "hebrew" in src.lower() or "שלום" in src
    for mid in MODELS:
        # Script must know how to route each target (via resolve_load_plan or listed defaults).
        assert mid.split("/")[-1] in src or mid in src


def test_infer_load_base_model_uses_resolve_plan():
    """Import path exists and reuses resolve_load_plan (no GPU call)."""
    from evaluation import infer
    from evaluation.base_predict import resolve_load_plan as rp

    assert hasattr(infer, "load_base_model")
    assert infer.resolve_load_plan("google/gemma-4-12B-it") == rp("google/gemma-4-12B-it")
