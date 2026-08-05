"""The Reference Quality Rubric judge used as the measurement instrument behind E1-E4.

Reads a Hebrew article
and one headline, blind to stratum or provenance, and returns four ordinal 1-to-5 sub-scores -
faithfulness, single-focus, informativeness, cleanliness - plus a one-sentence justification per
dimension. Reused unchanged across the dataset-review scripts (`data_curation/analysis/rubric_pilot.py`
today, the full E1-E3 passes later) and E4's model-output scoring, so dataset quality and model
quality land on one axis. Family separation: the judge is Gemini, distinct from the curator
(`gpt-5.6-luna`, OpenAI) and from the Qwen/DictaLM training base models.

Execution environment: local machine with GEMINI_API_KEY set. API-bound but CPU-only (no GPU, no
local model load), per the design spec's compute-placement rules.
"""

from __future__ import annotations

import json
import os
import re

from pipeline.stage_02_evaluation_instrument.gemini_client import (
    GEMINI_MODEL,
    GEMINI_TIMEOUT,
    call_with_retry,
)
from pipeline.stage_02_evaluation_instrument.rubric_anchors import ANCHORS

DIMENSIONS = ("faithfulness", "single_focus", "informativeness", "cleanliness")

# Frozen label boundaries used by the final paper's measurement instrument.
RUBRIC_LEVELS: dict[str, dict[int, str]] = {
    "faithfulness": {
        5: "Every element is directly supported. Attribution and uncertainty preserved where the article hedges.",
        4: "Fully supported, with one minor framing choice the article does not quite make explicit.",
        3: "Mostly supported, but one detail is an inference beyond what the text states.",
        2: "Contains a claim the article does not support, or states a hedged claim as fact.",
        1: "Contradicts the article, or centres on something absent from it.",
    },
    "single_focus": {
        5: "One clear subject.",
        4: "One subject with a subordinate clause that supports it.",
        3: "Two related aspects given roughly equal weight, still recognisably one story.",
        2: "Two or more separable stories joined together.",
        1: "A list of unrelated items with no dominant subject.",
    },
    "informativeness": {
        5: "Names the who and what, and enough of the specifics to be useful alone.",
        4: "Clear subject and event, some specifics missing.",
        3: "Identifies the topic but not the development — what it is about, not what happened.",
        2: "Gestures at a topic without content. Teaser-like.",
        1: "Pure hook, empty question, or promotional line carrying no information.",
    },
    "cleanliness": {
        5: "Clean headline text, nothing extraneous.",
        4: "One minor punctuation or spacing oddity.",
        3: "Contains a removable fragment such as a site label, category tag, or credit.",
        2: "Multiple artifacts, or separator characters used as structure.",
        1: "Largely boilerplate, metadata, or broken fragments rather than a headline.",
    },
}

DIMENSION_QUESTIONS = {
    "faithfulness": "Is every claim in the headline supported by the article?",
    "single_focus": "Does the headline name one dominant subject, or bundle several?",
    "informativeness": "Could a reader tell what happened?",
    "cleanliness": "Is it free of scraping artifacts?",
}

# Character cap on the article text sent to the judge, matching the existing judge-prompt convention
# in evaluation/evaluate.py (JUDGE_PROMPT truncates to 6000 chars). Gemini's context window does not
# require this, but every one of 10,000 rows repeats the ~1,400-word anchor block below, so capping
# article length keeps per-call cost and latency predictable at that scale.
ARTICLE_CHAR_LIMIT = 6000

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _format_dimension_block(dimension: str) -> str:
    """Render one dimension's question, five-level scale, and worked anchors as prompt text."""
    levels = RUBRIC_LEVELS[dimension]
    scale_lines = "\n".join(f"  {score}: {desc}" for score, desc in sorted(levels.items(), reverse=True))

    anchor_lines = []
    for level, example in sorted(ANCHORS[dimension].items(), reverse=True):
        anchor_lines.append(
            f"  Example (score {level}):\n"
            f"    Article context: {example['article_context']}\n"
            f"    Headline: \"{example['headline']}\"\n"
            f"    Why: {example['justification']}"
        )

    return (
        f"### {dimension} ({DIMENSION_QUESTIONS[dimension]})\n"
        f"{scale_lines}\n\n"
        + "\n".join(anchor_lines)
    )


def build_judge_prompt(article: str, headline: str) -> str:
    """Build the full blinded rubric prompt for one (article, headline) pair.

    Blinding: only the article and one headline are shown. No stratum, filter outcome, or whether
    the headline is original or curated — per the rubric's "Judge protocol" section.
    """
    dimension_blocks = "\n\n".join(_format_dimension_block(d) for d in DIMENSIONS)
    truncated_article = article[:ARTICLE_CHAR_LIMIT]

    return f"""You are scoring a Hebrew news headline against its source article on four independent
1-to-5 ordinal dimensions. Each dimension is scored separately; a headline can be clean but
unfaithful, or faithful but uninformative. Use the worked examples to calibrate your scale, but score
the actual pair on its own merits — do not just imitate the closest example.

{dimension_blocks}

# Article
{truncated_article}

# Headline
{headline}

# Output
Reply with ONLY a JSON object, no prose, no markdown fences, in this exact shape:
{{
  "faithfulness": {{"score": <int 1-5>, "justification": "<one sentence>"}},
  "single_focus": {{"score": <int 1-5>, "justification": "<one sentence>"}},
  "informativeness": {{"score": <int 1-5>, "justification": "<one sentence>"}},
  "cleanliness": {{"score": <int 1-5>, "justification": "<one sentence>"}}
}}"""


def _parse_judge_reply(raw: str) -> dict:
    """Parse a judge reply into {dimension: {score, justification}}, tolerating malformed output.

    Returns an empty dict (rather than raising) on any parse failure, so one bad reply skips that
    row at run time instead of crashing a 10,000-row pass; callers are expected to check for missing
    dimensions and report a scored/attempted count, the same convention as evaluation/evaluate.py.
    """
    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

    result = {}
    for dimension in DIMENSIONS:
        entry = parsed.get(dimension)
        if not isinstance(entry, dict):
            continue
        score = entry.get("score")
        if isinstance(score, int) and 1 <= score <= 5:
            result[dimension] = {"score": score, "justification": entry.get("justification", "")}
    return result


def score_headline(article: str, headline: str, model=None, temperature: float | None = None) -> dict:
    """Score one (article, headline) pair on all four rubric dimensions.

    `model` accepts an already-constructed `genai.GenerativeModel` so callers scoring many rows (the
    pilot, the full E1 pass) build it once instead of per call. Returns
    {dimension: {"score": int, "justification": str}} for every dimension the judge answered; a
    dimension is simply absent if parsing failed for it, never a default score. Also returns {} (no
    retry) if Gemini's safety filter blocks the prompt (empty `response.candidates`, seen in practice
    on a handful of HeSum rows) — that outcome is deterministic for the given content, so retrying
    would just fail identically five more times and abort a 10,000-row pass over one unlucky row.
    """
    import google.generativeai as genai

    if model is None:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = build_judge_prompt(article, headline)
    generation_config = {"response_mime_type": "application/json"}
    if temperature is not None:
        generation_config["temperature"] = temperature

    def _generate():
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            request_options={"timeout": GEMINI_TIMEOUT},
        )
        if not response.candidates:
            return None
        return response.text

    raw = call_with_retry(_generate)
    if raw is None:
        return {}

    return _parse_judge_reply(raw)
