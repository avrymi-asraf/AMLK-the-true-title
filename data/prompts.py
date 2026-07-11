"""
Shared Hebrew summarization prompt and probe-variant helpers (build_prompt, make_variant).
Used by data/preprocess.py at training-data build time and by evaluation/predict.py at
inference. Kept free of datasets/transformers imports so API-only evaluation scripts
can import it on minimal Python builds.

Execution environment: imported locally by preprocess and predict.
"""
import re

# Hardened anti-elaboration prompt: caps length, forbids lists/pipes/speculation.
# Error analysis traced much hallucination to the model learning HeSum's
# "headline | headline" digest style and running on.
PROMPT_TEMPLATE = (
    "Summarize the following Hebrew news article in one or two short, factual sentences. "
    "Write the summary in Hebrew.\n"
    "Rules:\n"
    "- Use only information stated in the article; do not add, infer, or speculate.\n"
    "- Do not elaborate, editorialize, or list unrelated items.\n"
    "- Write plain prose with periods and commas — no bullet points, lists, or '|' separators.\n\n"
    "{text}\n\nSummary:\n"
)


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
