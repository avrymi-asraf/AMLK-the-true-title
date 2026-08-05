"""Hebrew E4 summarization prompt and DictaLM chat-template helpers.
Preprocess stores raw `build_prompt` text; train and inference wrap with
`format_chat_prompt` (dictalm2.0-instruct needs [INST]…[/INST]). Single source of truth
for instruct formatting — no per-arm branches. Free of datasets/transformers imports so
API-only scripts can import build_prompt on minimal builds.

Execution environment: imported locally by preprocess, train, and evaluation helpers.
"""
# Fine-tuned student budget used by both final E4 arms. The earlier
# 15-word / 1-sentence winner was tuned on the zero-shot base; at 15 words a fine-tuned model
# compresses out identifying specifics the judge needs (faithfulness ~2.56). Widening to 2
# sentences / 35 words + who/what/where reached ~4.52 on a correctly-decoded base. Same stop
# cue; must be identical across E4 arms (baked into the prompt column at preprocess time).
PROMPT_TEMPLATE = (
    "סכם את כתבת החדשות הבאה בעברית ב-2 משפטים קצרים, לא יותר מ-35 מילים. "
    "כלול את הפרטים המזהים המרכזיים: מי, מה והיכן. "
    "כתוב 2 משפטים בלבד ועצור מיד בסופם.\n\n"
    "{text}\n\nתקציר (עד 2 משפטים, עד 35 מילים):"
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


