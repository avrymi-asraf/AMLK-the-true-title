"""
Reference-summary cleaning used by every preprocess run (data/preprocess.py).
HeSum references are frequently "headline | headline | headline" media digests with
pipes and bullets; error analysis traced much of the fine-tuned model's hallucination
and over-long, split outputs to it learning that format. This module rewrites those
digests into natural prose (normalize_summary) and flags the worst multi-headline
roundups for removal (is_roundup_digest).

Kept import-light (standard library only, no `datasets`/`transformers`) so it can be
imported anywhere in the pipeline. Execution environment: local machine, CPU only.
"""
import re

# A leading list marker at the very start of a segment ("- item" / "* item").
_LEADING_MARKER_RE = re.compile(r"^\s*[-*]\s+")
_MULTISPACE_RE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,;:!?])")
_MULTI_PERIOD_RE = re.compile(r"\.\s*\.(?:\s*\.)*")


def pipe_segments(summary: str) -> int:
    """Number of '|'-separated segments (1 if there is no pipe)."""
    return len([s for s in summary.split("|") if s.strip()]) if "|" in summary else 1


def is_roundup_digest(summary: str, min_segments: int = 3) -> bool:
    """True for the worst multi-headline "media roundup" digests — several unrelated
    headlines stitched with pipes. These are not single-article summaries, so
    preprocess drops them rather than rewriting them into one."""
    return pipe_segments(summary) >= min_segments


def normalize_summary(summary: str) -> str:
    """Rewrite a digest-style summary into natural prose.

    Pipes and bullets become sentence breaks (". "), leading list markers are dropped,
    whitespace is collapsed, stray space-before-punctuation is fixed, and the result ends
    with a single terminal period. Idempotent on already-clean prose.
    """
    if not summary:
        return summary
    text = summary.strip()
    text = _LEADING_MARKER_RE.sub("", text)
    # Split on pipes and bullets, treat each piece as its own clause/sentence.
    parts = re.split(r"\s*\|\s*|\s*[•·●‣◦▪]\s*", text)
    parts = [_LEADING_MARKER_RE.sub("", p).strip() for p in parts]
    parts = [p for p in parts if p]
    text = ". ".join(parts)
    text = _MULTISPACE_RE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _MULTI_PERIOD_RE.sub(".", text)
    text = text.strip()
    if text and text[-1] not in ".!?…":
        text += "."
    return text
