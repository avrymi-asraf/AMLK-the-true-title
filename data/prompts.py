"""
Shared Hebrew summarization prompt, probe-variant helpers, and chat-template wrapping.
Preprocess stores raw `build_prompt` text; train and inference wrap with
`format_chat_prompt` (dictalm2.0-instruct needs [INST]…[/INST]). Single source of truth
for instruct formatting — no per-arm branches. Free of datasets/transformers imports so
API-only scripts can import build_prompt on minimal builds.

Execution environment: imported locally by preprocess, train, and evaluation helpers.
"""
import re

# Prompt-arena round-3 winner (full experiment log: docs/prompt-arena-notebook.md). A numeric
# word cap alone bound length only weakly across 3 rounds; adding an explicit stop cue ("write
# one sentence only and stop right after it") beat every other candidate tested, including
# worked examples (which hallucinated content). Still short of the loop's 0.9 compliance / 4.0
# faithfulness target — re-run the loop if a stronger prompt is needed.
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
