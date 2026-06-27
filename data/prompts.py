"""
Shared Hebrew summarization prompt and probe-variant helpers (build_prompt, make_variant).
Used by data/preprocess.py at training-data build time and by evaluation/predict.py at
inference. Kept free of datasets/transformers imports so API-only evaluation scripts
can import it on minimal Python builds.

Execution environment: imported locally by preprocess and predict.
"""
import re

# The "in Hebrew" instruction matters for the zero-shot baselines (base Qwen, Gemini):
# without it they summarize in English and score near-zero against the Hebrew references.
# "in up to 3 sentences" caps length (borrowed from HeSum's GPT prompt, Figure 2) — v1 ran on
# for hundreds of tokens; an explicit budget anchors the model toward reference-length summaries.
PROMPT_TEMPLATE = "Summarize the following Hebrew text in up to 3 sentences. Write the summary in Hebrew:\n\n{text}\n\nSummary:\n"


def build_prompt(text: str) -> str:
    """Render the Hebrew summarization instruction prompt for an article."""
    return PROMPT_TEMPLATE.format(text=text)


def _split_lead_body(text: str) -> tuple[str, str]:
    """Split an article into (lead, body): the first paragraph vs. the rest."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) >= 2:
        return paragraphs[0], "\n\n".join(paragraphs[1:])
    sentences = [s for s in re.split(r"(?<=[.!?。])\s+", text.strip()) if s]
    if len(sentences) >= 2:
        return sentences[0], " ".join(sentences[1:])
    return text, text  # too short to split — probe falls back to whole text


def make_variant(text: str, variant: str) -> str:
    """Return the article input for a probe variant: whole, lead-only, or body-only."""
    if variant == "whole":
        return text
    lead, body = _split_lead_body(text)
    return lead if variant == "lead" else body
