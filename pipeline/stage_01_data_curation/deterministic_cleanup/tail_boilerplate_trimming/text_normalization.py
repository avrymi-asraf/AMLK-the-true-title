"""Text normalization helpers for tail boilerplate trimming."""

from __future__ import annotations


def is_hebrew_char(value: str) -> bool:
    """Return whether a character is inside the Hebrew Unicode block."""
    return "\u0590" <= value <= "\u05FF"


def is_split_delimiter(value: str) -> bool:
    """Return whether a character should split text into coarse tokens."""
    return value in {" ", "\n", "\r", "\t"}


def split_hebrew_words(text: str) -> list[str]:
    """Extract normalized Hebrew words from text while discarding non-Hebrew characters."""
    words, _ = hebrew_words_with_original_spans(text)
    return words


def hebrew_words_with_original_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Extract Hebrew words and their original character spans in the source text."""
    words = []
    spans = []
    current_word = []
    current_token_start = None
    current_word_start = None
    current_word_end = None

    for index, char in enumerate(text):
        if is_split_delimiter(char):
            if current_word:
                words.append("".join(current_word))
                spans.append((current_word_start, current_word_end))
            current_word = []
            current_token_start = None
            current_word_start = None
            current_word_end = None
            continue

        if current_token_start is None:
            current_token_start = index

        if is_hebrew_char(char):
            if not current_word:
                current_word_start = current_token_start
            current_word.append(char)
            current_word_end = index + 1

    if current_word:
        words.append("".join(current_word))
        spans.append((current_word_start, current_word_end))

    return words, spans
