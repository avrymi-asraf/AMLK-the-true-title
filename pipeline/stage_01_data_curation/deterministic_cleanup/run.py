"""Run all pre-model cleanup stages and build model-curation input.

Run:
    python -m pipeline.stage_01_data_curation.deterministic_cleanup.run

Input:
    outputs/data_curation/raw_hesum.json

Outputs:
    outputs/data_curation/tail_boilerplate_removed.json
    outputs/data_curation/token_budget_upto_4000.json
    outputs/data_curation/headline_pipes_upto_1.json
    outputs/data_curation/source_filter_input.json
"""

from __future__ import annotations

from pipeline.common.json_io import load_json, save_json
from pipeline.common.paths import CURATION_WORK_DIR
from pipeline.stage_01_data_curation.download_hesum import RAW_HESUM_PATH
from pipeline.stage_01_data_curation.deterministic_cleanup.token_budget import filter_over_token_budget
from pipeline.stage_01_data_curation.deterministic_cleanup.multi_pipe import filter_multi_pipe_headlines
from pipeline.stage_01_data_curation.deterministic_cleanup.tail_boilerplate_trimming import remove_tail_boilerplates
from pipeline.stage_01_data_curation.deterministic_cleanup.token_budget.filter_over_token_budget import (
    TOKEN_BUDGET_FILTER_PATH,
)
from pipeline.stage_01_data_curation.deterministic_cleanup.multi_pipe.filter_multi_pipe_headlines import (
    HEADLINE_PIPE_FILTER_PATH,
)
from pipeline.stage_01_data_curation.deterministic_cleanup.tail_boilerplate_trimming.remove_tail_boilerplates import (
    TAIL_BOILERPLATE_REMOVED_PATH,
)
SOURCE_FILTER_INPUT_PATH = CURATION_WORK_DIR / "source_filter_input.json"


def should_keep_record(record_id: str, filters: list[dict[str, bool]]) -> bool:
    """Return whether a record passes every keep/remove filter map."""
    missing_filters = [
        index
        for index, filter_map in enumerate(filters, start=1)
        if record_id not in filter_map
    ]
    if missing_filters:
        raise ValueError(
            f"Record {record_id} is missing from filter maps: {missing_filters}"
        )

    return all(filter_map[record_id] for filter_map in filters)


def build_source_filter_input(
    records: list[dict],
    token_budget_filter: dict[str, bool],
    headline_pipe_filter: dict[str, bool],
) -> list[dict]:
    """Build source-filter input records that passed all pre-model filters."""
    filters = [token_budget_filter, headline_pipe_filter]
    source_filter_input = []

    for record in records:
        record_id = str(record["id"])
        if should_keep_record(record_id, filters):
            source_filter_input.append(
                {
                    "id": record_id,
                    "text": record["text"],
                    "headline": record["headline"],
                }
            )

    return source_filter_input


def write_source_filter_input() -> None:
    """Write the model-curation input from tail-trimmed records and filter maps."""
    records = load_json(TAIL_BOILERPLATE_REMOVED_PATH)
    token_budget_filter = load_json(TOKEN_BUDGET_FILTER_PATH)
    headline_pipe_filter = load_json(HEADLINE_PIPE_FILTER_PATH)

    source_filter_input = build_source_filter_input(
        records,
        token_budget_filter,
        headline_pipe_filter,
    )
    save_json(SOURCE_FILTER_INPUT_PATH, source_filter_input)

    print(f"Tail-trimmed records loaded: {len(records)}")
    print(f"Source-filter input records saved: {len(source_filter_input)}")
    print(f"Saved to: {SOURCE_FILTER_INPUT_PATH}")


def main() -> None:
    """Run all pre-model cleanup stages and build the source-filter input artifact."""
    if not RAW_HESUM_PATH.exists():
        raise FileNotFoundError(
            f"Missing raw input artifact: {RAW_HESUM_PATH}. "
            "Run python -m pipeline.stage_01_data_curation.download_hesum first."
        )

    print("Step 1/4: removing tail boilerplate")
    remove_tail_boilerplates.main()

    print("Step 2/4: filtering records over token budget")
    filter_over_token_budget.main()

    print("Step 3/4: filtering multi-pipe headlines")
    filter_multi_pipe_headlines.main()

    print("Step 4/4: building source-filter input")
    write_source_filter_input()


if __name__ == "__main__":
    main()
