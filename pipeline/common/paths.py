"""Canonical repository paths used by every retained research stage."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_CURATION_ARTIFACTS_DIR = ARTIFACTS_DIR / "data_curation"
REFERENCE_EXPERIMENT_ARTIFACTS_DIR = ARTIFACTS_DIR / "reference_experiments"
TRAINING_EXPERIMENT_ARTIFACTS_DIR = ARTIFACTS_DIR / "training_experiment"

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
SUMMARIES_DIR = RESULTS_DIR / "summaries"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CURATION_WORK_DIR = OUTPUTS_DIR / "data_curation"
TRAINING_WORK_DIR = OUTPUTS_DIR / "training_experiment"
