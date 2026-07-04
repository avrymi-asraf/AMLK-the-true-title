"""
Predictions viewer package: a local, read-only UI for browsing outputs/results/*.jsonl (see
evaluation/viewer/data.py for the data logic, evaluation/viewer/app.py for the Streamlit UI).
This file just re-exports the public data functions so callers can use
`from evaluation.viewer import load_predictions` without knowing the internal module layout.
"""
from evaluation.viewer.data import common_length, discover_predictions_files, filter_by_keyword, load_predictions

__all__ = ["discover_predictions_files", "load_predictions", "filter_by_keyword", "common_length"]
