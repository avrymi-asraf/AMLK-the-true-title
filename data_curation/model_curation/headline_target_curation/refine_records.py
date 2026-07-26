"""Submit, collect, and finalize headline target refinement API results."""

from __future__ import annotations

from pathlib import Path
import json

from data_curation.model_curation.headline_target_curation.refine_prompt_schema import (
    PROMPT,
    TARGET_REPAIR_SCHEMA,
)
from data_curation.model_curation.openai_batch_api.openai_client import OpenAIClient
from data_curation.utils.json_io import load_json, save_json
from data_curation.utils.paths import ARTIFACTS_DIR


BATCH_SIZE = 600
MAX_OUTPUT_TOKENS = 1000
CURATION_RUN_NAME = "headline_refine"
MODEL = "gpt-5.6-luna"

SOURCE_FILTER_INPUT_PATH = ARTIFACTS_DIR / "source_filter_input.json"
SOURCE_FILTER_RESULTS_PATH = ARTIFACTS_DIR / "source_filter_results.json"
FINAL_RESULTS_PATH = ARTIFACTS_DIR / "headline_target_curation_results.json"
OUTPUTS_DIR = Path(__file__).resolve().parent / "outputs"
RESULTS_PATH = OUTPUTS_DIR / "refine_results.json"
ERRORS_PATH = OUTPUTS_DIR / "refine_errors.jsonl"


def numeric_id_sort_key(record_id: str) -> tuple[int, str]:
    """Return a stable sort key for numeric-string and nonnumeric record ids."""
    record_id = str(record_id)
    if record_id.isdecimal():
        return (0, f"{int(record_id):020d}")
    return (1, record_id)


def result_sort_key(row: dict) -> tuple[int, str]:
    """Return a stable sort key for headline-refine result rows."""
    return numeric_id_sort_key(str(row["id"]))


def ensure_results_file() -> None:
    """Create the internal result file if this stage has no results yet."""
    if not RESULTS_PATH.exists():
        save_results([])


def save_results(results: list[dict]) -> None:
    """Save internal headline-refine results sorted by record id."""
    save_json(RESULTS_PATH, sorted(results, key=result_sort_key))


def load_results() -> list[dict]:
    """Load internal headline-refine results or return an empty list."""
    if not RESULTS_PATH.exists():
        return []

    return load_json(RESULTS_PATH)


def load_records() -> list[dict]:
    """Load usable source-filter records as headline-refine input."""
    records = load_json(SOURCE_FILTER_INPUT_PATH)
    source_filter_results = load_json(SOURCE_FILTER_RESULTS_PATH)
    usable_ids = {
        str(row["id"])
        for row in source_filter_results
        if row["filter_label"] == "usable"
    }
    usable_records = [
        {
            "id": str(record["id"]),
            "text": record["text"],
            "headline": record["headline"],
        }
        for record in records
        if str(record["id"]) in usable_ids
    ]
    return sorted(usable_records, key=lambda record: numeric_id_sort_key(record["id"]))


def completed_ids(results: list[dict]) -> set[str]:
    """Return ids that already have successful headline-refine results."""
    return {str(row["id"]) for row in results}


def load_failed_ids() -> set[str]:
    """Return ids that already have failed headline-refine result lines."""
    if not ERRORS_PATH.exists():
        return set()

    failed_ids = set()
    with ERRORS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            failed_ids.add(str(row.get("id") or row["custom_id"]))

    return failed_ids


def select_batch_records(records: list[dict], skipped_ids: set[str]) -> list[dict]:
    """Select the next unprocessed headline-refine records for one batch."""
    batch_records = []

    for record in records:
        if str(record["id"]) in skipped_ids:
            continue

        batch_records.append(record)
        if len(batch_records) >= BATCH_SIZE:
            break

    return batch_records


def build_user_input(record: dict) -> str:
    """Build the source-text and existing-headline prompt body for one record."""
    return (
        "<source_text>\n"
        f"{record['text']}\n"
        "</source_text>\n\n"
        "<existing_headline>\n"
        f"{record['headline']}\n"
        "</existing_headline>"
    )


def build_batch_inputs(batch_records: list[dict]) -> list[tuple[str, str]]:
    """Build custom-id and prompt-input pairs for headline-refine submission."""
    return [(str(record["id"]), build_user_input(record)) for record in batch_records]


def append_new_results(results: list[dict], new_results: list[dict]) -> int:
    """Append unseen successful results to the internal result file."""
    seen = completed_ids(results)
    added = 0
    for result in new_results:
        record_id = str(result["id"])
        if record_id in seen:
            continue
        results.append(result)
        seen.add(record_id)
        added += 1

    save_results(results)
    return added


def clean_nullable_string(name: str, value: str | None) -> str | None:
    """Normalize nullable model string fields and reject invalid types."""
    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")

    value = value.strip()
    if not value:
        return None
    return value


def clean_and_validate_result(record_id: str, result: dict) -> dict:
    """Validate one model result and normalize it to the stored result shape."""
    replacement_headline = clean_nullable_string(
        "replacement_headline",
        result["replacement_headline"],
    )

    return {
        "id": str(record_id),
        "replacement_headline": replacement_headline,
    }


def load_error_keys() -> set[tuple[str, str]]:
    """Load batch-id and record-id pairs already present in the error file."""
    if not ERRORS_PATH.exists():
        return set()

    error_keys = set()
    with ERRORS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            error_keys.add((row["batch_id"], str(row.get("id") or row["custom_id"])))

    return error_keys


def append_failed_results(failed_results: list[dict], batch_state: dict) -> int:
    """Append unseen failed result lines to the internal error file."""
    if not failed_results:
        return 0

    ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_error_keys = load_error_keys()
    added = 0
    with ERRORS_PATH.open("a", encoding="utf-8") as file:
        for result in failed_results:
            record_id = str(result["custom_id"])
            error_key = (batch_state["id"], record_id)
            if error_key in existing_error_keys:
                continue

            error_row = {
                "id": record_id,
                "batch_id": batch_state["id"],
                "custom_id": result["custom_id"],
                "error": result["error"],
            }
            json.dump(error_row, file, ensure_ascii=False)
            file.write("\n")
            existing_error_keys.add(error_key)
            added += 1

    return added


def finalize_if_complete(records: list[dict], results: list[dict], failed_ids: set[str]) -> bool:
    """Write the public artifact and stats only when every input record is processed."""
    input_ids = {str(record["id"]) for record in records}
    done_ids = completed_ids(results)
    processed_ids = done_ids | failed_ids
    remaining_ids = input_ids - processed_ids

    if remaining_ids:
        print(f"Remaining records to refine: {len(remaining_ids)}")
        return False

    final_results = sorted(results, key=result_sort_key)
    save_json(FINAL_RESULTS_PATH, final_results)
    print_final_stats(records, final_results, failed_ids)
    return True


def print_final_stats(records: list[dict], results: list[dict], failed_ids: set[str]) -> None:
    """Print final headline-refine counts after the public artifact is saved."""
    replaced = sum(1 for row in results if row["replacement_headline"] is not None)
    kept = len(results) - replaced

    print("Headline target curation is complete.")
    print(f"Input records: {len(records)}")
    print(f"Successful results: {len(results)}")
    print(f"Failed records: {len(failed_ids)}")
    print(f"Headlines kept unchanged: {kept}")
    print(f"Headlines replaced: {replaced}")
    print(f"Saved final artifact to: {FINAL_RESULTS_PATH}")


def submit_next_batch(openai_client: OpenAIClient) -> bool:
    """Submit the next headline-refine batch unless all records are processed."""
    ensure_results_file()
    results = load_results()
    records = load_records()
    done_ids = completed_ids(results)
    failed_ids = load_failed_ids()
    skipped_ids = done_ids | failed_ids
    batch_records = select_batch_records(records, skipped_ids)

    print(f"Eligible records: {len(records)}")
    print(f"Already completed in this output: {len(done_ids)}")
    print(f"Already failed in this output: {len(failed_ids)}")

    if not batch_records:
        print("No unprocessed eligible records left.")
        finalize_if_complete(records, results, failed_ids)
        return False

    batch_id = openai_client.submit_batch(build_batch_inputs(batch_records))

    print(f"Submitted batch: {batch_id}")
    print(f"Submitted records: {len(batch_records)}")
    print("Rerun this script later to check status and collect results.")
    return True


def collect_completed_batch(openai_client: OpenAIClient, batch_state: dict) -> None:
    """Collect a completed batch and update internal headline-refine outputs."""
    ensure_results_file()
    results = load_results()
    new_results = []
    failed_results = []

    for result in openai_client.get_last_batch_results():
        if result["ok"]:
            try:
                record_id = str(result["custom_id"])
                new_results.append(clean_and_validate_result(record_id, result["data"]))
            except (KeyError, TypeError, ValueError) as error:
                failed_results.append({
                    "custom_id": result["custom_id"],
                    "error": {"code": "validation_error", "message": str(error)},
                })
        else:
            failed_results.append(result)

    added = append_new_results(results, new_results)
    failed_added = append_failed_results(failed_results, batch_state)
    openai_client.clean_batch_state()
    print(f"New results saved: {added}")
    print(f"Failed result lines saved: {failed_added}")
    print(f"Total results: {len(load_results())}")

    finalize_if_complete(load_records(), load_results(), load_failed_ids())


def handle_existing_batch_state(openai_client: OpenAIClient, batch_state: dict) -> None:
    """Print current batch state and collect results when the batch is complete."""
    counts = batch_state["counts"]
    print(f"Batch: {batch_state['id']}")
    print(f"Status: {batch_state['status']}")
    if counts is not None:
        print(
            f"Requests: {counts['completed']}/{counts['total']} completed, "
            f"{counts['failed']} failed",
        )

    if batch_state["status"] == "completed":
        print("Batch completed successfully.")
        print(f"Output file: {batch_state.get('output_file_id')}")
        print(f"Error file: {batch_state.get('error_file_id')}")
        collect_completed_batch(openai_client, batch_state)

    elif batch_state["status"] in {"failed", "cancelled", "expired"}:
        print("Fatal batch status. Check the OpenAI batch state before continuing.")

    else:
        print("Batch is still running. Rerun this script later.")


def main() -> None:
    """Submit, collect, or finalize headline-refine batch results."""
    openai_client = OpenAIClient(
        CURATION_RUN_NAME,
        model=MODEL,
        prompt=PROMPT,
        schema=TARGET_REPAIR_SCHEMA,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        cache_prompt=True,
    )
    ensure_results_file()

    batch_state = openai_client.get_current_batch_state()

    if batch_state is None:
        submit_next_batch(openai_client)
    else:
        handle_existing_batch_state(openai_client, batch_state)


if __name__ == "__main__":
    main()
