"""
Hebrew-script decode constraint for generation. Fine-tuned outputs occasionally leak
foreign scripts mid-word (Cyrillic, Arabic, Latin, etc.) — a vocab/decoding artifact.
build_bad_words_ids scans the tokenizer vocabulary once and returns the ids of every
token whose decoded surface form contains a Latin/Cyrillic/Greek/Arabic letter, so
generate(bad_words_ids=...) can forbid emitting them. Hebrew letters, digits, punctuation,
and whitespace stay allowed.

Wired into evaluation/infer.py and training/train_hf_job.py generation (always on).
Execution environment: wherever generation runs (remote GPU: HF Jobs / Colab).
"""
import re

# Scripts we forbid inside generated summaries. Hebrew (0x0590-0x05FF) is intentionally absent.
_FORBIDDEN_SCRIPT_RE = re.compile(
    "["
    "A-Za-z"        # Latin A-Z a-z
    "À-ɏ"                       # Latin-1 supplement + extended (accented Latin)
    "Ѐ-ӿ"                       # Cyrillic
    "Ͱ-Ͽ"                       # Greek
    "؀-ۿ"                       # Arabic
    "]"
)


def _has_forbidden_script(text: str) -> bool:
    return bool(_FORBIDDEN_SCRIPT_RE.search(text))


def build_bad_words_ids(tokenizer) -> list[list[int]] | None:
    """Return bad_words_ids (list of single-token id lists) for every vocab token whose decoded
    form contains a forbidden foreign-script letter, or None if none are found.

    Special tokens are skipped so EOS/pad and chat-template markers stay usable.
    """
    special_ids = set(tokenizer.all_special_ids)
    bad: list[list[int]] = []
    for token_id in range(tokenizer.vocab_size):
        if token_id in special_ids:
            continue
        piece = tokenizer.decode([token_id])
        if piece and _has_forbidden_script(piece):
            bad.append([token_id])
    return bad or None
