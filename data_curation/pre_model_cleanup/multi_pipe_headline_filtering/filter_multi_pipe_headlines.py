"""Build a keep/remove map for records with too many headline pipes.

Run:
    python -m data_curation.pre_model_cleanup.multi_pipe_headline_filtering.filter_multi_pipe_headlines

Input:
    data_curation/artifacts/tail_boilerplate_removed.json

Output:
    data_curation/artifacts/headline_pipes_upto_1.json
"""

from __future__ import annotations

from data_curation.pre_model_cleanup.tail_boilerplate_trimming.remove_tail_boilerplates import (
    TAIL_BOILERPLATE_REMOVED_PATH,
)
from data_curation.utils.json_io import load_json, save_json
from data_curation.utils.paths import ARTIFACTS_DIR


MAX_HEADLINE_PIPES = 1
PIPE = "|"
HEADLINE_PIPE_FILTER_PATH = ARTIFACTS_DIR / f"headline_pipes_upto_{MAX_HEADLINE_PIPES}.json"


def build_headline_pipe_filter(
    records: list[dict],
    max_headline_pipes: int = MAX_HEADLINE_PIPES,
) -> dict[str, bool]:
    """Build a record-id keep map based on headline pipe count."""
    return {
        str(record["id"]): record["headline"].count(PIPE) <= max_headline_pipes
        for record in records
    }


def main() -> None:
    """Generate the public multi-pipe-headline keep/remove artifact."""
    records = load_json(TAIL_BOILERPLATE_REMOVED_PATH)
    headline_pipe_filter = build_headline_pipe_filter(records)

    save_json(HEADLINE_PIPE_FILTER_PATH, headline_pipe_filter)

    kept = sum(1 for keep in headline_pipe_filter.values() if keep)
    removed = len(headline_pipe_filter) - kept

    print(f"Records loaded: {len(records)}")
    print(f"Records kept: {kept}")
    print(f"Records removed: {removed}")
    print(f"Saved to: {HEADLINE_PIPE_FILTER_PATH}")


if __name__ == "__main__":
    main()
