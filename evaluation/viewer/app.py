"""
Evaluation pipeline, results-reading step: a local Streamlit UI for browsing predictions.jsonl
files (the {text, reference, prediction, model, variant} rows written by predict.py /
train_hf_job.py / infer.py, scored by evaluate.py). Fills the gap between raw jsonl in a text
editor and the live-generation notebook (notebooks/evaluation_observation.ipynb) — a fast,
read-only, RTL-aware way to page through what the model actually produced, with keyword search
and side-by-side comparison across systems. All data logic lives in evaluation/viewer/data.py;
this file only wires those functions to widgets.

Run: streamlit run evaluation/viewer/app.py
Execution environment: local machine, CPU-only, no GPU/API — reads files already in outputs/results/.
"""
import html
import random
import sys
from pathlib import Path

# `streamlit run` executes this file directly (as __main__), which only puts this file's own
# directory on sys.path — not the repo root — so the absolute `evaluation.viewer` import below
# would fail regardless of the caller's cwd. Insert the repo root (three levels up: app.py ->
# viewer/ -> evaluation/ -> root) so this script works from any invocation directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from evaluation.viewer import common_length, discover_predictions_files, filter_by_keyword, load_predictions

st.set_page_config(page_title="AMLK Predictions Viewer", layout="wide")


@st.cache_data(show_spinner="Loading predictions…")
def _load_cached(path_str: str, _mtime: float) -> list[dict]:
    """_mtime is only there to bust the cache when the file on disk changes."""
    return load_predictions(path_str)


def load_cached(path: Path) -> list[dict]:
    return _load_cached(str(path), path.stat().st_mtime)


def rtl_block(label: str, text: str) -> None:
    """Render a labeled Hebrew text block right-to-left, with a char/word count caption."""
    st.markdown(f"**{label}**")
    body = html.escape(text) if text else "<i>(empty)</i>"
    st.markdown(
        f'<div dir="rtl" style="text-align: right; white-space: pre-wrap; '
        f'font-size: 1.05rem; line-height: 1.7; padding: 0.25rem 0;">{body}</div>',
        unsafe_allow_html=True,
    )
    if text:
        st.caption(f"{len(text)} characters · {len(text.split())} words")


def main() -> None:
    st.title("AMLK Predictions Viewer")
    st.caption("Browse model outputs from outputs/results/*.jsonl — read-only, local, CPU-only.")

    with st.sidebar:
        st.header("Files")
        options = [str(p) for p in discover_predictions_files()]
        selected = st.multiselect(
            "Predictions files (pick 2+ to compare side-by-side)", options,
            default=options[:1] if options else [],
        )
        manual_path = st.text_input("…or add a path manually")
        if manual_path:
            selected = selected + [manual_path]

        st.header("Search")
        keyword = st.text_input("Keyword (article / prediction / reference)")

    if not selected:
        st.info("No files found under outputs/results/. Pick or type a predictions.jsonl path in the sidebar.")
        return

    files_rows: dict[str, list[dict]] = {}
    for path_str in selected:
        path = Path(path_str)
        if not path.exists():
            st.sidebar.error(f"Not found: {path}")
            continue
        files_rows[path.name] = load_cached(path)

    if not files_rows:
        st.warning("None of the selected files could be loaded.")
        return

    lengths = {name: len(rows) for name, rows in files_rows.items()}
    n = common_length(files_rows)
    if len(set(lengths.values())) > 1:
        st.warning(f"Selected files have different lengths — navigation is clamped to {n}. Counts: {lengths}")

    first_name = next(iter(files_rows))
    indices = filter_by_keyword(files_rows[first_name][:n], keyword)
    if not indices:
        st.warning(f"No matches for '{keyword}'.")
        return

    if "pos" not in st.session_state:
        st.session_state.pos = 0
    st.session_state.pos = max(0, min(st.session_state.pos, len(indices) - 1))

    nav_cols = st.columns([1, 1, 1, 3])
    if nav_cols[0].button("⟵ Prev", use_container_width=True):
        st.session_state.pos = max(0, st.session_state.pos - 1)
    if nav_cols[1].button("Next ⟶", use_container_width=True):
        st.session_state.pos = min(len(indices) - 1, st.session_state.pos + 1)
    if nav_cols[2].button("🎲 Random", use_container_width=True):
        st.session_state.pos = random.randrange(len(indices))
    picked = nav_cols[3].number_input(
        f"Jump to example (1-{len(indices)})", min_value=1, max_value=len(indices),
        value=st.session_state.pos + 1, step=1,
    )
    st.session_state.pos = picked - 1

    idx = indices[st.session_state.pos]
    st.caption(f"Showing example {st.session_state.pos + 1} / {len(indices)}  ·  row index {idx}")

    row0 = files_rows[first_name][idx]
    with st.expander("Article", expanded=False):
        rtl_block("Article", row0.get("text", ""))
    rtl_block("Reference summary", row0.get("reference", ""))

    st.divider()
    cols = st.columns(len(files_rows))
    for col, (name, rows) in zip(cols, files_rows.items()):
        with col:
            row = rows[idx]
            st.markdown(f"##### {name}")
            st.caption(f"{row.get('model', '?')} / {row.get('variant', '?')}")
            rtl_block("Prediction", row.get("prediction", ""))


if __name__ == "__main__":
    main()
