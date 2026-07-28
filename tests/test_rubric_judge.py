"""Tests for evaluation.rubric_judge: prompt construction and judge-reply parsing, offline. The one
test that calls Gemini is gated behind RUN_LIVE_TESTS, matching tests/test_evaluation.py's convention.
"""
import os

import pytest

from evaluation.rubric_judge import DIMENSIONS, _parse_judge_reply, build_judge_prompt


def _well_formed_reply() -> str:
    return """{
  "faithfulness": {"score": 4, "justification": "mostly supported"},
  "single_focus": {"score": 5, "justification": "one clear subject"},
  "informativeness": {"score": 3, "justification": "names topic only"},
  "cleanliness": {"score": 2, "justification": "has a pipe fragment"}
}"""


def test_parse_judge_reply_extracts_all_four_dimensions():
    scores = _parse_judge_reply(_well_formed_reply())

    assert set(scores.keys()) == set(DIMENSIONS)
    assert scores["faithfulness"]["score"] == 4
    assert scores["cleanliness"]["justification"] == "has a pipe fragment"


def test_parse_judge_reply_tolerates_markdown_fences():
    fenced = f"```json\n{_well_formed_reply()}\n```"
    scores = _parse_judge_reply(fenced)

    assert scores["single_focus"]["score"] == 5


def test_parse_judge_reply_drops_out_of_range_or_missing_scores():
    raw = """{
      "faithfulness": {"score": 7, "justification": "bad"},
      "single_focus": {"score": 3, "justification": "ok"}
    }"""
    scores = _parse_judge_reply(raw)

    assert "faithfulness" not in scores  # 7 is out of the 1-5 range
    assert scores["single_focus"]["score"] == 3
    assert "informativeness" not in scores  # dimension absent from the reply entirely


def test_parse_judge_reply_returns_empty_on_garbage():
    assert _parse_judge_reply("I cannot help with that.") == {}


def test_build_judge_prompt_includes_article_headline_and_anchors():
    prompt = build_judge_prompt("מאמר לדוגמה על אירוע חדשותי כלשהו.", "כותרת לדוגמה")

    assert "מאמר לדוגמה על אירוע חדשותי כלשהו." in prompt
    assert "כותרת לדוגמה" in prompt
    for dimension in DIMENSIONS:
        assert dimension in prompt
    assert "Example (score 5)" in prompt  # at least one worked anchor is present


def test_build_judge_prompt_truncates_long_articles():
    from evaluation.rubric_judge import ARTICLE_CHAR_LIMIT

    long_article = "א" * (ARTICLE_CHAR_LIMIT + 5000)
    prompt = build_judge_prompt(long_article, "כותרת")

    assert "א" * (ARTICLE_CHAR_LIMIT + 1) not in prompt


def test_score_headline_returns_empty_dict_on_blocked_prompt_without_retrying():
    """A safety-filter block (empty response.candidates) is deterministic for the given content —
    score_headline must return {} immediately, not retry five times and raise (this exact crash hit
    the rubric pilot at row 276/300 on a real HeSum article, see rubric_judge.py's docstring)."""
    from evaluation.rubric_judge import score_headline

    class _BlockedResponse:
        candidates = []

    class _FakeModel:
        call_count = 0

        def generate_content(self, *args, **kwargs):
            _FakeModel.call_count += 1
            return _BlockedResponse()

    scores = score_headline("מאמר כלשהו", "כותרת כלשהי", model=_FakeModel())

    assert scores == {}
    assert _FakeModel.call_count == 1  # no retries for a non-transient block


@pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") and os.getenv("RUN_LIVE_TESTS")),
    reason="Set GEMINI_API_KEY and RUN_LIVE_TESTS=1 to run the live rubric-judge test",
)
def test_live_rubric_judge_scores_a_clean_headline():
    from evaluation.rubric_judge import score_headline

    article = "ראש הממשלה נפגש היום עם נשיא צרפת בפריז ודן עמו בהסכם סחר חדש בין המדינות."
    headline = "ראש הממשלה נפגש עם נשיא צרפת ודן בהסכם סחר חדש"

    scores = score_headline(article, headline)

    assert set(scores.keys()) == set(DIMENSIONS)
    for dimension in DIMENSIONS:
        assert 1 <= scores[dimension]["score"] <= 5
