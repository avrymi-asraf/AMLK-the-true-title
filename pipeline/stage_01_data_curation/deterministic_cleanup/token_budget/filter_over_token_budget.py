"""Count DictaLM tokens and build a keep/remove map for the token budget.

Run:
    python -m pipeline.stage_01_data_curation.deterministic_cleanup.token_budget.filter_over_token_budget

Input:
    outputs/data_curation/tail_boilerplate_removed.json

Outputs:
    outputs/data_curation/dictalm_token_counts.json
    outputs/data_curation/token_budget_upto_4000.json
"""

from __future__ import annotations

from tqdm import tqdm

from pipeline.common.json_io import load_json, save_json
from pipeline.common.paths import CURATION_WORK_DIR
from pipeline.stage_01_data_curation.deterministic_cleanup.tail_boilerplate_trimming.remove_tail_boilerplates import (
    TAIL_BOILERPLATE_REMOVED_PATH,
)
TOKENIZER_NAME = "dicta-il/dictalm2.0-instruct"
MAX_TOTAL_TOKENS = 4000
DICTALM_TOKEN_COUNTS_PATH = CURATION_WORK_DIR / "dictalm_token_counts.json"
TOKEN_BUDGET_FILTER_PATH = CURATION_WORK_DIR / f"token_budget_upto_{MAX_TOTAL_TOKENS}.json"


def load_dictalm_tokenizer():
    """Load the DictaLM tokenizer used for all token-budget decisions."""
    from transformers import AutoTokenizer

    # token=False: the tokenizer repo is public; some local setups carry an invalid cached HF
    # token that would otherwise turn this anonymous-eligible request into a 401.
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=False)


def count_tokens(tokenizer, text: str) -> int:
    """Count tokenizer tokens for one text without special tokens."""
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def count_record_tokens(tokenizer, record: dict) -> int:
    """Count total source-text and headline tokens for one record."""
    return count_tokens(tokenizer, record["text"]) + count_tokens(
        tokenizer,
        record["headline"],
    )


def build_token_counts(records: list[dict]) -> dict[str, int]:
    """Count total DictaLM tokens for each record id."""
    tokenizer = load_dictalm_tokenizer()
    token_counts: dict[str, int] = {}

    for record in tqdm(records, desc="Counting DictaLM tokens"):
        record_id = record["id"]
        token_counts[record_id] = count_record_tokens(tokenizer, record)

    return token_counts


def build_token_budget_filter(
    token_counts: dict[str, int],
    max_total_tokens: int = MAX_TOTAL_TOKENS,
) -> dict[str, bool]:
    """Build a record-id keep map based on the configured token budget."""
    return {
        record_id: tokens_amount <= max_total_tokens
        for record_id, tokens_amount in token_counts.items()
    }


def main() -> None:
    """Generate the internal token counts and public token-budget filter artifact."""
    print(f"Loading records from: {TAIL_BOILERPLATE_REMOVED_PATH}")
    records = load_json(TAIL_BOILERPLATE_REMOVED_PATH)

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    token_counts = build_token_counts(records)
    save_json(DICTALM_TOKEN_COUNTS_PATH, token_counts)

    token_budget_filter = build_token_budget_filter(token_counts)

    save_json(TOKEN_BUDGET_FILTER_PATH, token_budget_filter)

    kept = sum(1 for keep in token_budget_filter.values() if keep)
    removed = len(token_budget_filter) - kept

    print(f"Records loaded: {len(records)}")
    print(f"Token counts saved: {len(token_counts)}")
    print(f"Token counts path: {DICTALM_TOKEN_COUNTS_PATH}")
    print(f"Records kept: {kept}")
    print(f"Records removed: {removed}")
    print(f"Saved to: {TOKEN_BUDGET_FILTER_PATH}")


if __name__ == "__main__":
    main()
