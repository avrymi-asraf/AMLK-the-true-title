"""Build the unified row-label artifact: one row per HeSum id, joining every curation-pipeline
artifact already on disk. This is the single source of truth for the dataset-review analysis
(`docs/superpowers/specs/2026-07-26-dataset-review-experimental-design.md`, section 2) — the
paper figures (`figures.py`) and the later statistical tests all read this file instead of
re-deriving defect membership from the raw artifacts. Local, CPU-only: needs the DictaLM
tokenizer (for `article_tokens`/`headline_tokens`) but no GPU and no API calls.

Run:
    python -m data_curation.analysis.row_labels

Requires the pipeline artifacts to already exist (raw_hesum.json, tail_boilerplate_removed.json,
the two keep maps) — run `python -m data_curation.pre_model_cleanup.run_pre_model_cleanup` first
if they are missing. `source_filter_results.json` / `headline_target_curation_results.json` are
the two supplied model-curation artifacts and must already be in `data_curation/artifacts/`.

Output:
    data_curation/artifacts/row_labels.json
"""

from __future__ import annotations

from data_curation.data_download.download_hesum import RAW_HESUM_PATH
from data_curation.final_dataset.build_final_dataset import (
    HEADLINE_TARGET_CURATION_RESULTS_PATH,
    SOURCE_FILTER_RESULTS_PATH,
)
from data_curation.pre_model_cleanup.dictalm_token_budget_filtering.filter_over_token_budget import (
    TOKEN_BUDGET_FILTER_PATH,
    load_dictalm_tokenizer,
)
from data_curation.pre_model_cleanup.multi_pipe_headline_filtering.filter_multi_pipe_headlines import (
    HEADLINE_PIPE_FILTER_PATH,
)
from data_curation.pre_model_cleanup.tail_boilerplate_trimming.remove_tail_boilerplates import (
    TAIL_BOILERPLATE_REMOVED_PATH,
)
from data_curation.pre_model_cleanup.tail_boilerplate_trimming.text_normalization import (
    split_hebrew_words,
)
from data_curation.utils.json_io import load_json, save_json
from data_curation.utils.paths import ARTIFACTS_DIR


ROW_LABELS_PATH = ARTIFACTS_DIR / "row_labels.json"
LEAD_WORD_COUNT = 50  # first N Hebrew words of the article treated as its "lead"
LIGHT_EDIT_OVERLAP_THRESHOLD = 0.5  # Hebrew-word Jaccard above which a rewrite counts as "light"
BOILERPLATE_OVERLAP_THRESHOLD = 0.8  # overlap above which a shorter replacement looks like a trim
DANGLING_LAST_WORDS = {"של", "עם", "אל", "כי", "אשר", "וכן", "וגם", "אבל", "או", "גם", "רק"}

# Pre-registered verified counts (spec section 1) — a fresh pipeline run must reproduce these
# exactly, or the artifact does not mean what the paper claims it means.
EXPECTED_COUNTS = {
    "total": 10000,
    "tail_boilerplate_removed": 722,
    "over_token_budget": 2659,
    "multi_pipe": 2412,
    "reached_model_curation": 6486,
    "usable": 5854,
    "unusable": 632,
    "headline_kept": 2785,
    "headline_rewritten": 3069,
}


def count_pipes(headline: str) -> int:
    """Count the raw number of pipe characters in a headline."""
    return headline.count("|")


def compute_lead_overlap(article_text: str, headline_text: str, lead_word_count: int = LEAD_WORD_COUNT) -> float:
    """Fraction of the headline's Hebrew words that also appear in the article's lead window.

    A simple, reference-free lead-bias probe: 1.0 means every headline word is drawn from the
    first `lead_word_count` words of the article, 0.0 means none are. Word-set based (not order
    or position sensitive) and Hebrew-only by construction (reuses the same word extraction the
    tail-trimming stage uses), so punctuation/site-name noise cannot inflate the overlap.
    """
    headline_words = set(split_hebrew_words(headline_text))
    if not headline_words:
        return 0.0

    lead_words = set(split_hebrew_words(article_text)[:lead_word_count])
    return round(len(headline_words & lead_words) / len(headline_words), 4)


def classify_headline_edit(original: str, replacement: str) -> str:
    """Classify a curated headline rewrite from the (original, replacement) string pair alone.

    Priority order matches `docs/superpowers/specs/2026-07-26-dataset-review-experimental-design.md`
    section 2: a rewrite is filed under the first sub-type it matches. This is a documented
    heuristic, not a new LLM label — it exists so `full_rewrite` (little usable signal in the
    original) can be told apart from every other repair (the original was salvageable).
    """
    original_words = split_hebrew_words(original)
    replacement_words = split_hebrew_words(replacement)
    original_word_set = set(original_words)
    replacement_word_set = set(replacement_words)

    if count_pipes(original) > 0 and count_pipes(replacement) == 0:
        return "pipes_removed"

    union = original_word_set | replacement_word_set
    overlap = len(original_word_set & replacement_word_set) / len(union) if union else 0.0

    is_shorter = len(replacement_words) < len(original_words)
    looks_boilerplate_trim = replacement.strip() and replacement.strip() in original
    if is_shorter and (looks_boilerplate_trim or overlap >= BOILERPLATE_OVERLAP_THRESHOLD):
        return "boilerplate_stripped"

    stripped_original = original.rstrip()
    ends_without_punctuation = not stripped_original.endswith((".", "!", "?", '"', "”", "׳"))
    last_word = original_words[-1] if original_words else ""
    looks_cut_off = stripped_original.endswith(("...", "…")) or last_word in DANGLING_LAST_WORDS
    if ends_without_punctuation and looks_cut_off:
        return "truncation_repaired"

    return "light_edit" if overlap >= LIGHT_EDIT_OVERLAP_THRESHOLD else "full_rewrite"


def build_row_labels(
    records: list[dict],
    token_counts: dict[str, tuple[int, int]],
    token_budget_filter: dict[str, bool],
    headline_pipe_filter: dict[str, bool],
    source_labels: dict[str, str],
    headline_replacements: dict[str, str | None],
) -> list[dict]:
    """Join every per-id artifact into one row-label record per id.

    `records` are the tail-trimmed records (all 10,000). `token_counts` maps id to
    `(article_tokens, headline_tokens)`, computed separately so this function stays pure and
    testable without a tokenizer. The rest are the deterministic keep maps and the two supplied
    model-curation result files, exactly as described in the design spec.
    """
    rows = []
    for record in records:
        record_id = record["id"]
        article_tokens, headline_tokens = token_counts[record_id]
        source_label = source_labels.get(record_id)
        reached_model_curation = record_id in source_labels

        headline_action = None
        headline_edit_type = None
        if record_id in headline_replacements:
            replacement = headline_replacements[record_id]
            if replacement is None:
                headline_action = "kept"
            else:
                headline_action = "rewritten"
                headline_edit_type = classify_headline_edit(record["headline"], replacement)

        rows.append({
            "hesum_id": record_id,
            "article_tokens": article_tokens,
            "headline_tokens": headline_tokens,
            "n_pipes": count_pipes(record["headline"]),
            "over_token_budget": not token_budget_filter[record_id],
            "multi_pipe": not headline_pipe_filter[record_id],
            "tail_boilerplate_removed": record["tail_boilerplate_removed"],
            "source_label": source_label,
            "reached_model_curation": reached_model_curation,
            "headline_action": headline_action,
            "headline_edit_type": headline_edit_type,
            "headline_lead_overlap": compute_lead_overlap(record["text"], record["headline"]),
        })

    return rows


def load_source_labels() -> dict[str, str]:
    """Load the supplied source-filter labels, keyed by id."""
    return {
        str(row["id"]): row["filter_label"]
        for row in load_json(SOURCE_FILTER_RESULTS_PATH)
    }


def load_headline_replacements() -> dict[str, str | None]:
    """Load the supplied headline replacements (None means kept unchanged), keyed by id."""
    return {
        str(row["id"]): row["replacement_headline"]
        for row in load_json(HEADLINE_TARGET_CURATION_RESULTS_PATH)
    }


def build_token_counts(records: list[dict]) -> dict[str, tuple[int, int]]:
    """Tokenize each record's article and headline separately with the DictaLM tokenizer."""
    from tqdm import tqdm

    tokenizer = load_dictalm_tokenizer()

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False).input_ids)

    return {
        record["id"]: (count(record["text"]), count(record["headline"]))
        for record in tqdm(records, desc="Counting DictaLM tokens (article/headline split)")
    }


def validate_counts(rows: list[dict]) -> None:
    """Check the freshly built artifact against the pre-registered verified counts."""
    counts = {
        "total": len(rows),
        "tail_boilerplate_removed": sum(r["tail_boilerplate_removed"] for r in rows),
        "over_token_budget": sum(r["over_token_budget"] for r in rows),
        "multi_pipe": sum(r["multi_pipe"] for r in rows),
        "reached_model_curation": sum(r["reached_model_curation"] for r in rows),
        "usable": sum(r["source_label"] == "usable" for r in rows),
        "unusable": sum(
            r["source_label"] is not None and r["source_label"] != "usable" for r in rows
        ),
        "headline_kept": sum(r["headline_action"] == "kept" for r in rows),
        "headline_rewritten": sum(r["headline_action"] == "rewritten" for r in rows),
    }
    mismatches = {
        key: (counts[key], expected)
        for key, expected in EXPECTED_COUNTS.items()
        if counts[key] != expected
    }
    if mismatches:
        raise ValueError(
            "Row-label counts do not match the pre-registered verified inputs "
            f"(docs/superpowers/specs/2026-07-26-dataset-review-experimental-design.md section 1). "
            f"(actual, expected) per field: {mismatches}"
        )


def load_row_labels() -> list[dict]:
    """Load the row-label artifact, building it first if it does not exist yet."""
    if not ROW_LABELS_PATH.exists():
        main()
    return load_json(ROW_LABELS_PATH)


def main() -> None:
    """Build, validate, and save the row-label artifact."""
    for path in (RAW_HESUM_PATH, TAIL_BOILERPLATE_REMOVED_PATH, TOKEN_BUDGET_FILTER_PATH, HEADLINE_PIPE_FILTER_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing pipeline artifact: {path}. Run "
                "`python -m data_curation.pre_model_cleanup.run_pre_model_cleanup` first."
            )

    records = load_json(TAIL_BOILERPLATE_REMOVED_PATH)
    token_counts = build_token_counts(records)
    token_budget_filter = load_json(TOKEN_BUDGET_FILTER_PATH)
    headline_pipe_filter = load_json(HEADLINE_PIPE_FILTER_PATH)
    source_labels = load_source_labels()
    headline_replacements = load_headline_replacements()

    rows = build_row_labels(
        records, token_counts, token_budget_filter, headline_pipe_filter,
        source_labels, headline_replacements,
    )
    validate_counts(rows)
    save_json(ROW_LABELS_PATH, rows)

    print(f"Row-label records: {len(rows)}")
    print("All counts matched the pre-registered verified inputs.")
    print(f"Saved to: {ROW_LABELS_PATH}")


if __name__ == "__main__":
    main()
