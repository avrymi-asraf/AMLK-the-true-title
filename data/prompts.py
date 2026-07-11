"""
Shared Hebrew summarization prompt, probe-variant helpers, and chat-template wrapping.
Used by data/preprocess.py at training-data build time and by evaluation/train paths at
inference. `build_prompt` stays free of model-specific control tokens; `format_chat_prompt`
is the single source of truth for wrapping instructions in a tokenizer's chat template
(instruct models like dictalm2.0-instruct require [INST]…[/INST]). Kept free of
datasets/transformers imports so API-only scripts can import build_prompt on minimal builds.

Execution environment: imported locally by preprocess, train, and evaluation helpers.
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


def format_chat_prompt(tokenizer, prompt: str) -> str:
    """Wrap a user instruction in the model's chat template when one exists.

    Instruct checkpoints (e.g. dictalm2.0-instruct) must see their native format
    (`[INST] … [/INST]`) at both train and inference. Pure base checkpoints with no
    chat template get the raw prompt. Does not inject model-family extras like
    `/no_think` — those corrupt templates that have no notion of them (Mistral).
    """
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def prepare_tokenizer_for_templated_prompts(tokenizer):
    """Avoid double-BOS: chat templates already emit BOS; disable a second one on encode.

    dictalm2's template starts with `{{bos_token}}`; with add_bos_token=True the next
    tokenizer(...) call would produce ['<s>', '<s>', '[', 'INST', …]. Call this after
    load whenever generation/SFT will encode already-templated strings.
    """
    if getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "add_bos_token"):
        tokenizer.add_bos_token = False
    return tokenizer


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
