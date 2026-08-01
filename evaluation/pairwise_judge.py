"""E3 forced-choice judge: blind head-to-head comparison of two headlines for the same article.
Shared by `repair_pairwise_run.py` (E3) and future E4 arm comparisons. Family-separated from
the curator (OpenAI) and training bases, same as `rubric_judge.py`. Local, API-bound, CPU-only.
"""

from __future__ import annotations

import json
import os
import re

from evaluation.gemini_client import GEMINI_MODEL, GEMINI_TIMEOUT, call_with_retry

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
VALID_WINNERS = frozenset({"a", "b", "tie"})


def build_pairwise_prompt(article: str, headline_a: str, headline_b: str) -> str:
    """Blind pairwise prompt — no provenance labels, only Headline A / Headline B."""
    truncated = article[:6000]
    return f"""You are comparing two Hebrew news headlines for the same article. Read the article,
then decide which headline is the better training target for a summarization model: more faithful
to the article, more focused, more informative, and cleaner. If they are genuinely equivalent,
say tie.

# Article
{truncated}

# Headline A
{headline_a}

# Headline B
{headline_b}

Reply with ONLY a JSON object:
{{"winner": "a" | "b" | "tie", "justification": "<one sentence in English>"}}"""


def _parse_pairwise_reply(raw: str) -> dict:
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    winner = str(parsed.get("winner", "")).lower().strip()
    if winner not in VALID_WINNERS:
        return {}
    return {"winner": winner, "justification": parsed.get("justification", "")}


def compare_headlines(
    article: str, headline_a: str, headline_b: str, model=None,
) -> dict:
    """Return {winner: a|b|tie, justification} or {} on failure/block."""
    import google.generativeai as genai

    if model is None:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = build_pairwise_prompt(article, headline_a, headline_b)

    def _generate():
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.0,
            },
            request_options={"timeout": GEMINI_TIMEOUT},
        )
        if not response.candidates:
            return None
        return response.text

    raw = call_with_retry(_generate)
    if raw is None:
        return {}
    return _parse_pairwise_reply(raw)


def curated_wins(winner: str, curated_is_a: bool) -> str:
    """Map judge winner + position randomization to curated_wins / original_wins / tie."""
    if winner == "tie":
        return "tie"
    curated_won = (winner == "a") == curated_is_a
    return "curated" if curated_won else "original"
