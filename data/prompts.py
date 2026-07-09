"""
Shared Hebrew summarization prompt and probe-variant helpers (build_prompt, make_variant).
Used by data/preprocess.py at training-data build time and by evaluation/predict.py at
inference. Kept free of datasets/transformers imports so API-only evaluation scripts
can import it on minimal Python builds.

Execution environment: imported locally by preprocess and predict.
"""
import re

# The "in Hebrew" instruction matters for the zero-shot baselines (base model, Gemini):
# without it they risk summarizing in English and scoring near-zero against the Hebrew
# references. "in up to 3 sentences" caps length (borrowed from HeSum's GPT prompt, Figure 2)
# — an explicit budget anchors the model toward reference-length summaries instead of running on.
PROMPT_TEMPLATE = "Summarize the following Hebrew text in up to 3 sentences. Write the summary in Hebrew:\n\n{text}\n\nSummary:\n"

# Hardened prompt for the opt-in "clean" pipeline profile. Adds an explicit anti-elaboration
# cap and negative instructions (no lists / pipes / added detail) — error analysis traced much
# of the model's hallucination to it running on and reproducing HeSum's "headline | headline"
# digest style. Kept as a separate template so the original PROMPT_TEMPLATE stays reproducible.
PROMPT_TEMPLATE_CLEAN = (
    "Summarize the following Hebrew news article in one or two short, factual sentences. "
    "Write the summary in Hebrew.\n"
    "Rules:\n"
    "- Use only information stated in the article; do not add, infer, or speculate.\n"
    "- Do not elaborate, editorialize, or list unrelated items.\n"
    "- Write plain prose with periods and commas — no bullet points, lists, or '|' separators.\n\n"
    "{text}\n\nSummary:\n"
)


def build_prompt(text: str, clean: bool = False) -> str:
    """Render the Hebrew summarization instruction prompt for an article.

    clean=True selects the hardened, anti-elaboration template used by the clean pipeline
    profile; the default reproduces the original prompt.
    """
    template = PROMPT_TEMPLATE_CLEAN if clean else PROMPT_TEMPLATE
    return template.format(text=text)


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
