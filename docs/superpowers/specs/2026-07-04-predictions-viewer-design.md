# Predictions Viewer — Design Spec

**Date:** 2026-07-04
**Status:** Approved

## Problem

`outputs/results/` accumulates several `predictions-*.jsonl` files (finetuned / base / gemini,
across versions), each a list of `{text, reference, prediction, model, variant}` rows. Today the
only ways to read them are: raw `jsonl` in a text editor (no RTL, no navigation), or the
`evaluation_observation.ipynb` notebook, which prints a few examples to stdout on a live Colab
session. There is no lightweight, local, browsable way to read what the model actually produced —
which is needed constantly during error analysis and write-up.

## Goal

A local Streamlit app that browses one or more `predictions.jsonl` files, rendering Hebrew
correctly (RTL), with keyword search and side-by-side comparison across systems for the same
article. Read-only, CPU-only, no GPU/API — it only reads files already on disk.

## Architecture

A small package, `evaluation/viewer/`, so the whole feature is easy to find in one place, split
so the data logic stays reusable outside the UI (the project's existing pattern of separating
"tools" from their notebook/UI callers, e.g. `evaluation/infer.py`):

- `evaluation/viewer/data.py` — plain functions, no Streamlit import. Discoverable/testable/importable
  from a notebook or REPL.
- `evaluation/viewer/__init__.py` — re-exports `data.py`'s public functions so callers use
  `from evaluation.viewer import ...` without knowing the internal module layout.
- `evaluation/viewer/app.py` — the Streamlit script. Thin: only widgets, wired to `data.py`
  functions via the package import. Run with `streamlit run evaluation/viewer/app.py`.

## Components

### `evaluation/viewer/data.py`

- `discover_predictions_files(results_dir="outputs/results") -> list[Path]`
  Globs `*.jsonl` directly under `results_dir` (non-recursive — skips the `.cache` subdir),
  sorted by name, so the sidebar has candidates without the user typing paths.
- `load_predictions(path) -> list[dict]`
  Reads the jsonl, applies the existing `strip_think()` (from `evaluation.gemini_client`) to each
  `prediction`, so the viewer shows exactly what `evaluate.py` scores, not raw `<think>` leakage.
- `filter_by_keyword(rows: list[dict], keyword: str) -> list[int]`
  Case-insensitive substring match over `text` + `prediction` + `reference`. Returns matching row
  indices; empty keyword returns all indices (`range(len(rows))`).
- `common_length(files_rows: dict[str, list[dict]]) -> int`
  Returns the shortest row count across the selected files, so comparison mode can't index out of
  range when two files come from mismatched runs (e.g. one file resumed a partial job).

### `evaluation/viewer/app.py`

- Sidebar:
  - Multiselect of files from `discover_predictions_files()` (plus a manual path text input
    fallback if none are found or the user wants a file elsewhere).
  - Keyword search box, applied via `filter_by_keyword` against the *first* selected file's rows
    (the shared index space) — matching indices become the navigable set.
  - Index navigation: a number input plus Prev / Next / Random buttons, bounded by
    `common_length(...)` (or by the filtered match count, if a keyword is active).
- Main panel, per current index:
  1. Article — collapsed in an expander by default (articles are long); RTL.
  2. Reference summary — RTL.
  3. One block per selected file, labeled with that row's `model` / `variant`, prediction text
     rendered RTL, plus a small char/word count caption.
- RTL rendering: a block of injected CSS (`direction: rtl; text-align: right`) applied via a small
  `st.markdown(..., unsafe_allow_html=True)` wrapper, reused for every Hebrew text block.
- Caching: `load_predictions` wrapped with `st.cache_data`, keyed on `(path, mtime)` so a
  regenerated file (same path, new content) reloads instead of serving stale cached rows.

## Data Flow

```
outputs/results/*.jsonl
        │  discover_predictions_files()
        ▼
   sidebar multiselect ──► load_predictions() (cached) ──► rows per file
        │                                                        │
        ▼                                                        ▼
  keyword search ──► filter_by_keyword() ──► navigable indices   │
        │                                                        │
        └──────────────► index picker ─────────────────────────►│
                                                                  ▼
                                          main panel: article / reference /
                                          per-file prediction, at current index
```

## Error Handling

- No `.jsonl` files found under `outputs/results/` → friendly message + manual path text input.
- Selected files have different lengths → clamp navigation to `common_length(...)`, show a
  `st.warning` with the per-file counts so the mismatch is visible, not silent.
- Keyword search with zero matches → "No matches for '<keyword>'" message instead of an
  index-out-of-range crash.
- Empty prediction string (a known base-model failure mode — Qwen3 sometimes never closes its
  `<think>` block) → rendered as a visible `(empty)` placeholder, not blank space, so it reads as
  a failure rather than a UI bug.

## Testing

`tests/test_viewer.py` — pure-function tests against `evaluation/viewer/data.py` only (no Streamlit
runtime, no GPU/API), using small `tmp_path` jsonl fixtures:

- `load_predictions` strips a `<think>...</think>` block from `prediction`.
- `filter_by_keyword` matches a Hebrew substring across text/prediction/reference and returns
  `[]` for a keyword with no match.
- `common_length` returns the minimum across files of different lengths.
- `discover_predictions_files` finds `.jsonl` files in a temp dir and ignores non-jsonl files.

Fits the existing suite: fast, no network, no GPU, runs in the default `pytest tests/`.

## Dependencies

Add `streamlit` to `requirements.txt`.

## Out of Scope (this iteration)

- Loading `*.report.json` aggregate metrics or `*.errors.json` failure labels into the viewer.
- Topic-stratified browsing (`stratify_by_topic.py` output).
- Any write/edit capability — this is read-only.
