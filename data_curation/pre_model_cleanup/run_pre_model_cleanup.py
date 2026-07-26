"""Run all pre-model cleanup stages and build model-curation input.

Run:
    python -m data_curation.pre_model_cleanup.run_pre_model_cleanup

Input:
    data_curation/artifacts/raw_hesum.json

Outputs:
    data_curation/artifacts/tail_boilerplate_removed.json
    data_curation/artifacts/token_budget_upto_4000.json
    data_curation/artifacts/headline_pipes_upto_1.json
    data_curation/artifacts/source_filter_input.json
"""

from __future__ import annotations

from data_curation.pre_model_cleanup.dictalm_token_budget_filtering import (
    filter_over_token_budget,
)
from data_curation.pre_model_cleanup.multi_pipe_headline_filtering import (
    filter_multi_pipe_headlines,
)
from data_curation.pre_model_cleanup.tail_boilerplate_trimming import (
    remove_tail_boilerplates,
)
from data_curation.data_download.download_hesum import RAW_HESUM_PATH
from data_curation.pre_model_cleanup.dictalm_token_budget_filtering.filter_over_token_budget import (
    TOKEN_BUDGET_FILTER_PATH,
)
from data_curation.pre_model_cleanup.multi_pipe_headline_filtering.filter_multi_pipe_headlines import (
    HEADLINE_PIPE_FILTER_PATH,
)
from data_curation.pre_model_cleanup.tail_boilerplate_trimming.remove_tail_boilerplates import (
    TAIL_BOILERPLATE_REMOVED_PATH,
)
from data_curation.utils.json_io import load_json, save_json
from data_curation.utils.paths import ARTIFACTS_DIR


SOURCE_FILTER_INPUT_PATH = ARTIFACTS_DIR / "source_filter_input.json"


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
            "Run data_curation.data_download.download_hesum first."
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
