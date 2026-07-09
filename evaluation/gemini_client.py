"""
Shared Gemini API helpers for the evaluation pipeline (predict.py, evaluate.py,
error_analysis.py). Kept separate from predict.py so judge/baseline code can import
call_with_retry and GEMINI_MODEL without pulling in datasets (load_from_disk).

Execution environment: local machine with GEMINI_API_KEY set.
"""
import re
import sys
import time

# gemini-2.5-flash-lite, not 2.5-flash: full 2.5-flash "thinks" before answering (~7s/call
# measured), which makes the ~4000-call evaluation battery take ~10h on HF Jobs. The -lite
# variant has no thinking step (~1s/call measured), ~6x faster, and is still a capable
# advanced baseline + judge.
GEMINI_MODEL = "gemini-2.5-flash-lite"
# Per-request deadline (seconds) for every generate_content call.
GEMINI_TIMEOUT = 60


def strip_think(text: str) -> str:
    """Drop <think>...</think> reasoning blocks so metrics score the summary, not the reasoning.

    Only well-formed (closed) blocks are removed. A truncated, unclosed block is left as-is
    so its low score reflects that real failure instead of being hidden.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def call_with_retry(fn, attempts: int = 5):
    """Call fn(), retrying with exponential backoff on transient API errors."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — the API raises a variety of transient errors
            if i == attempts - 1:
                raise
            wait = 2 ** i
            print(f"  API error ({str(e)[:60]}...); retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
