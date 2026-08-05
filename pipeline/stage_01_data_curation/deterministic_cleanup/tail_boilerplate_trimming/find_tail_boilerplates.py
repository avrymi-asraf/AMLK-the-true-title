"""Helper functions for detecting repeated article-tail boilerplate."""

from __future__ import annotations

from pipeline.common.paths import CURATION_WORK_DIR
from pipeline.stage_01_data_curation.deterministic_cleanup.tail_boilerplate_trimming.text_normalization import (
    split_hebrew_words,
)


TAIL_BOILERPLATE_CANDIDATES_PATH = CURATION_WORK_DIR / "tail_boilerplate_candidates.json"
TAIL_SIZE = 5
MIN_COUNT = 5

TokenizedRecord = tuple[str, list[str]]


def tail_words(words: list[str], size: int = TAIL_SIZE) -> tuple[str, ...]:
    """Return the final words used as the seed tail for one record."""
    if len(words) < size:
        return ()
    return tuple(words[-size:])


def format_result(tail: tuple[str, ...], records: list[TokenizedRecord]) -> dict:
    """Format one repeated tail candidate and the matching record ids."""
    return {
        "tail": " ".join(tail),
        "count": len(records),
        "length": len(tail),
        "ids": [record_id for record_id, _ in records],
    }


def expand_tail_branches(
    tail: tuple[str, ...],
    matching_records: list[TokenizedRecord],
) -> list[dict]:
    """Recursively expand a repeated tail backward while enough records still match."""
    previous_word_groups: dict[str, list[TokenizedRecord]] = {}
    current_length = len(tail)

    for record_id, words in matching_records:
        if len(words) > current_length:
            previous_word = words[-current_length - 1]
            previous_word_groups.setdefault(previous_word, []).append((record_id, words))

    expandable_groups = {
        previous_word: records
        for previous_word, records in previous_word_groups.items()
        if len(records) >= MIN_COUNT
    }
    if not expandable_groups:
        return [format_result(tail, matching_records)]

    results = []
    expanded_record_ids = set()
    for previous_word, records in expandable_groups.items():
        expanded_tail = (previous_word,) + tail
        results.extend(expand_tail_branches(expanded_tail, records))
        expanded_record_ids.update(record_id for record_id, _ in records)

    leftover_records = [
        record
        for record in matching_records
        if record[0] not in expanded_record_ids
    ]
    if len(leftover_records) >= MIN_COUNT:
        results.append(format_result(tail, leftover_records))

    return results


def find_tail_boilerplate_candidates(records: list[dict]) -> list[dict]:
    """Find repeated article-ending candidates from normalized Hebrew word tails."""
    tokenized_records = [
        (record["id"], split_hebrew_words(record["text"]))
        for record in records
    ]

    seed_groups: dict[tuple[str, ...], list[TokenizedRecord]] = {}
    for record_id, words in tokenized_records:
        tail = tail_words(words)
        if tail:
            seed_groups.setdefault(tail, []).append((record_id, words))

    results = []
    sorted_seed_groups = sorted(
        seed_groups.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    for seed_tail, matching_records in sorted_seed_groups:
        if len(matching_records) < MIN_COUNT:
            break
        results.extend(expand_tail_branches(seed_tail, matching_records))

    return sorted(
        results,
        key=lambda result: (-result["count"], -result["length"], result["tail"]),
    )
