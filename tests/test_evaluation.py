"""Tests for the evaluation pipeline: metric wiring and judge-reply parsing.

Fast and offline by default — ROUGE on Hebrew, JSON extraction from an LLM reply, and
failure-rate aggregation. The one test that calls Gemini is gated behind RUN_LIVE_TESTS.
"""
import os

import pytest

from evaluation.evaluate import _parse_json, compute_rouge
from evaluation.error_analysis import failure_rates


def test_rouge_handles_hebrew():
    # The bug this guards: rouge_score's default tokenizer strips non-ASCII, zeroing Hebrew.
    preds = [{"reference": "החתול ישב על המחצלת", "prediction": "החתול ישב על המחצלת"}]
    scores = compute_rouge(preds)
    assert scores["rouge1"] == 1.0
    assert scores["rougeL"] == 1.0


def test_rouge_partial_overlap_between_zero_and_one():
    preds = [{"reference": "החתול ישב על המחצלת", "prediction": "החתול רץ"}]
    scores = compute_rouge(preds)
    assert 0.0 < scores["rouge1"] < 1.0


def test_parse_json_tolerates_code_fences():
    assert _parse_json('```json\n{"faithfulness": 4, "fluency": 5}\n```') == {"faithfulness": 4, "fluency": 5}


def test_parse_json_returns_empty_on_garbage():
    assert _parse_json("the model refused to answer") == {}


def test_failure_rates_counts_each_type():
    labelled = [
        {"labels": ["hallucination", "omission"]},
        {"labels": ["hallucination"]},
        {"labels": []},
        {"labels": ["lead_copying"]},
    ]
    rates = failure_rates(labelled)
    assert rates["hallucination"] == 0.5
    assert rates["omission"] == 0.25
    assert rates["fluency_problem"] == 0.0


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") and os.getenv("RUN_LIVE_TESTS")),
    reason="Set GEMINI_API_KEY and RUN_LIVE_TESTS=1 to run the live Gemini judge test",
)
def test_live_gemini_judge_parses_scores():
    from evaluation.evaluate import judge_with_llm

    preds = [{"text": "ראש הממשלה נפגש היום עם נשיא צרפת בפריז.",
              "prediction": "ראש הממשלה נפגש עם נשיא צרפת."}]
    result = judge_with_llm(preds)
    assert 1 <= result["faithfulness_mean"] <= 5
    assert 1 <= result["fluency_mean"] <= 5
