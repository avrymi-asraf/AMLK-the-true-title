"""Remove detected repeated tail boilerplate from raw HeSum texts.

Run:
    python -m data_curation.pre_model_cleanup.tail_boilerplate_trimming.remove_tail_boilerplates

Inputs:
    data_curation/artifacts/raw_hesum.json

Output:
    data_curation/artifacts/tail_boilerplate_removed.json
"""

from __future__ import annotations

import re

from data_curation.data_download.download_hesum import RAW_HESUM_PATH
from data_curation.utils.json_io import load_json, save_json
from data_curation.utils.paths import ARTIFACTS_DIR
from data_curation.pre_model_cleanup.tail_boilerplate_trimming.find_tail_boilerplates import (
    TAIL_BOILERPLATE_CANDIDATES_PATH,
    find_tail_boilerplate_candidates,
)
from data_curation.pre_model_cleanup.tail_boilerplate_trimming.text_normalization import (
    hebrew_words_with_original_spans,
)


ASCII_LETTER_OR_DIGIT_RE = re.compile(r"[A-Za-z0-9]")
SENTENCE_PUNCTUATION = set(".!?;:)]}")
TAIL_BOILERPLATE_REMOVED_PATH = ARTIFACTS_DIR / "tail_boilerplate_removed.json"


def tail_by_record_id(tail_rows: list[dict]) -> dict[str, tuple[str, ...]]:
    """Build a record-id lookup using the longest detected tail per record."""
    record_tails: dict[str, tuple[str, ...]] = {}

    for row in tail_rows:
        tail_words = tuple(row["tail"].split())
        for record_id in row["ids"]:
            previous_tail = record_tails.get(record_id)
            if previous_tail is None or len(tail_words) > len(previous_tail):
                record_tails[record_id] = tail_words

    return record_tails


def gap_looks_like_prior_metadata(value: str) -> bool:
    """Return whether text between Hebrew words looks like scraped metadata."""
    return (
        bool(ASCII_LETTER_OR_DIGIT_RE.search(value))
        and any(char in SENTENCE_PUNCTUATION for char in value)
    )


def remove_tail_from_text(text: str, tail_words: tuple[str, ...]) -> tuple[str, bool]:
    """Remove a matching tail from one text and report whether it matched."""
    words, spans = hebrew_words_with_original_spans(text)
    tail_length = len(tail_words)

    if len(words) < tail_length or tuple(words[-tail_length:]) != tail_words:
        return text, False

    tail_word_index = len(words) - tail_length
    while tail_word_index < len(words) - 1:
        _, first_word_end = spans[tail_word_index]
        second_word_start, _ = spans[tail_word_index + 1]
        gap = text[first_word_end:second_word_start]
        if not gap_looks_like_prior_metadata(gap):
            break
        tail_word_index += 1

    tail_start, _ = spans[tail_word_index]
    return text[:tail_start].rstrip(), True


def remove_tail_boilerplates(
    records: list[dict],
    record_tails: dict[str, tuple[str, ...]],
) -> tuple[list[dict], list[str]]:
    """Apply detected tail removals to records and return unmatched ids."""
    cleaned_records = []
    unmatched_ids = []

    for record in records:
        cleaned_record = dict(record)
        tail_words = record_tails.get(record["id"])
        cleaned_record["tail_boilerplate_removed"] = False

        if tail_words is not None:
            cleaned_text, matched = remove_tail_from_text(record["text"], tail_words)
            if matched:
                cleaned_record["text"] = cleaned_text
                cleaned_record["tail_boilerplate_removed"] = True
            else:
                unmatched_ids.append(record["id"])

        cleaned_records.append(cleaned_record)

    return cleaned_records, unmatched_ids


def main() -> None:
    """Run tail detection and write the tail-trimmed public artifact."""
    records = load_json(RAW_HESUM_PATH)
    tail_rows = find_tail_boilerplate_candidates(records)
    save_json(TAIL_BOILERPLATE_CANDIDATES_PATH, tail_rows)

    record_tails = tail_by_record_id(tail_rows)
    cleaned_records, unmatched_ids = remove_tail_boilerplates(records, record_tails)

    save_json(TAIL_BOILERPLATE_REMOVED_PATH, cleaned_records)

    print(f"Records loaded: {len(records)}")
    print(f"Tail candidates saved: {len(tail_rows)}")
    print(f"Tail candidates path: {TAIL_BOILERPLATE_CANDIDATES_PATH}")
    print(f"Records targeted for tail removal: {len(record_tails)}")
    print(f"Records with unmatched tails: {len(unmatched_ids)}")
    print(f"Records saved: {len(cleaned_records)}")
    print(f"Saved to: {TAIL_BOILERPLATE_REMOVED_PATH}")


if __name__ == "__main__":
    main()
