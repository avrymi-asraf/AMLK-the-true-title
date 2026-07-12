"""
Shared Hebrew summarization prompt, probe-variant helpers, and chat-template wrapping.
Preprocess stores raw `build_prompt` text; train and inference wrap with
`format_chat_prompt` (dictalm2.0-instruct needs [INST]…[/INST]). Single source of truth
for instruct formatting — no per-arm branches. Free of datasets/transformers imports so
API-only scripts can import build_prompt on minimal builds.

Execution environment: imported locally by preprocess, train, and evaluation helpers.
"""
import re

# Prompt-arena round-3 winner (see docs/prompt-arena-notebook.md): short Hebrew instruction +
# numeric word cap + an explicit imperative stop cue ("write one sentence only and stop right
# after it"). Beat every other candidate across all 3 rounds on every axis (compliance 0.82,
# judge faithfulness 3.40, fluency 4.27 — n=20, not yet re-checked at n=100). Round 1 found
# worked-example (one-shot/two-shot) prompts actually hurt faithfulness here — they hallucinated
# an unrelated entity apparently primed by the example's own content — so no exemplar is used.
# Also chosen over the long hardened English "Rules:" prompt for stability: that prompt
# provoked a garbled Hangul near-token (an apparent hallucinated echo of Mistral's own [/INST]
# tag) in 38% of round-1 outputs vs 1-2% for Hebrew prompts, a failure mode traced to
# evaluation/hebrew_constraint.py's decode constraint not covering CJK/Hangul (since fixed).
# Caveat: still short of the loop's 0.9/4.0 target — no zero-shot prompt tested reached it.
# Round 1 traced much of the base overshoot to the model ignoring length instructions outright,
# not to digest-style copying, so the anti-digest/no-lists rule was dropped as
# untested-necessary; re-add it if fine-tuned output regresses toward HeSum's pipe-digest style.
PROMPT_TEMPLATE = (
    "סכם את כתבת החדשות הבאה בעברית במשפט קצר אחד, לא יותר מ-15 מילים. "
    "כתוב משפט אחד בלבד ועצור מיד בסופו.\n\n"
    "{text}\n\nתקציר (משפט אחד, עד 15 מילים):"
)


def build_prompt(text: str) -> str:
    """Render the Hebrew summarization instruction prompt for an article."""
    return PROMPT_TEMPLATE.format(text=text)


def format_chat_prompt(tokenizer, prompt: str) -> str:
    """Wrap a user instruction in the model's chat template when one exists.

    Instruct checkpoints (dictalm2.0-instruct: `[INST] … [/INST]`) must see this format
    at train and inference. Pure base checkpoints with no chat template get the raw prompt.
    Does not inject family-specific control tokens. `enable_thinking=False` is attempted
    when supported and ignored (TypeError) on Mistral-style templates.
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
    """Avoid double-BOS after chat templating.

    dictalm2's template starts with `{{bos_token}}`; with add_bos_token=True the next
    tokenizer(...) would produce ['<s>', '<s>', …]. Call after load whenever generation
    or SFT will encode already-templated strings; pair with add_special_tokens=False.
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
