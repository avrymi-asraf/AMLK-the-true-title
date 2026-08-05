"""Regenerate every deterministic result supported by available frozen artifacts.

This reviewer-facing command never downloads data or models, calls an API,
performs inference, or starts training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pipeline.common.json_io import load_json, save_json
from pipeline.common.paths import (
    DATA_CURATION_ARTIFACTS_DIR,
    REFERENCE_EXPERIMENT_ARTIFACTS_DIR,
    SUMMARIES_DIR,
    TRAINING_EXPERIMENT_ARTIFACTS_DIR,
)


Runner = Callable[[], None]


@dataclass(frozen=True)
class Analysis:
    name: str
    inputs: tuple[Path, ...]
    runner: Runner


def _copy_json_summary(source: Path, destination: Path) -> None:
    data = load_json(source)
    if not isinstance(data, dict):
        raise ValueError(f"{source} must contain a JSON object")
    save_json(destination, data)


def _curation_summary() -> None:
    from pipeline.stage_01_data_curation.summarize import main

    main()


def _curation_outputs() -> None:
    from pipeline.stage_01_data_curation.make_curation_figure import main as make_figure
    from pipeline.stage_01_data_curation.make_strata_table import main as make_table

    make_figure()
    make_table()


def _pilot_summary() -> None:
    source = REFERENCE_EXPERIMENT_ARTIFACTS_DIR / "rubric_pilot.json"
    _copy_json_summary(source, SUMMARIES_DIR / "evaluation_instrument_pilot.json")


def _e1() -> None:
    from pipeline.stage_03_reference_experiments.e1_flagged_strata.make_figures import main as make_figures
    from pipeline.stage_03_reference_experiments.e1_flagged_strata.summarize import main as summarize

    summarize()
    make_figures()


def _e2() -> None:
    from pipeline.stage_03_reference_experiments.e2_paired_repair.make_figure import main as make_figure

    source = REFERENCE_EXPERIMENT_ARTIFACTS_DIR / "e2_repair_summary.json"
    make_figure()
    _copy_json_summary(source, SUMMARIES_DIR / "e2_paired_repair.json")


def _e3() -> None:
    from pipeline.stage_03_reference_experiments.e3_blind_preference.make_figure import main as make_figure

    source = REFERENCE_EXPERIMENT_ARTIFACTS_DIR / "e3_pairwise_summary.json"
    make_figure()
    _copy_json_summary(source, SUMMARIES_DIR / "e3_blind_preference.json")


def _human_summary() -> None:
    from pipeline.stage_03_reference_experiments.human_validation.summarize import main

    main()


def _human_figure() -> None:
    from pipeline.stage_03_reference_experiments.human_validation.make_figure import main

    main()


def _e4_rubric() -> None:
    from pipeline.stage_04_training_experiment.make_figure import main as make_figure
    from pipeline.stage_04_training_experiment.score import summarize_frozen_rubric

    summary = summarize_frozen_rubric()
    print(f"E4 rubric summary saved: {SUMMARIES_DIR / 'e4_rubric.json'} (n={summary['n_pairs']})")
    make_figure()


def _e4_pairwise() -> None:
    from pipeline.stage_04_training_experiment.score import (
        RUBRIC_SCORES_PATH,
        summarize_frozen,
        summarize_frozen_pairwise,
    )

    summary = summarize_frozen_pairwise()
    print(f"E4 pairwise summary saved: {SUMMARIES_DIR / 'e4_pairwise.json'} (n={summary['n_pairs']})")
    if RUBRIC_SCORES_PATH.is_file():
        summarize_frozen()
        print(f"Combined E4 summary saved: {SUMMARIES_DIR / 'e4.json'}")


def _lead_bias() -> None:
    from pipeline.stage_05_supplementary.lead_bias.make_figure import main

    main()


def analyses() -> tuple[Analysis, ...]:
    row_labels = DATA_CURATION_ARTIFACTS_DIR / "row_labels.json"
    e1_scores = REFERENCE_EXPERIMENT_ARTIFACTS_DIR / "e1_rubric_scores.jsonl"
    human_dir = REFERENCE_EXPERIMENT_ARTIFACTS_DIR / "human_validation"
    human_inputs = (
        human_dir / "worklist.json",
        human_dir / "amit.jsonl",
        human_dir / "avreymi.jsonl",
        human_dir / "ofek.jsonl",
    )
    return (
        Analysis(
            "data-curation summary",
            (
                DATA_CURATION_ARTIFACTS_DIR / "source_filter_results.json",
                DATA_CURATION_ARTIFACTS_DIR / "headline_target_curation_results.json",
            ),
            _curation_summary,
        ),
        Analysis("curation funnel and strata table", (row_labels,), _curation_outputs),
        Analysis(
            "evaluation-instrument pilot summary",
            (REFERENCE_EXPERIMENT_ARTIFACTS_DIR / "rubric_pilot.json",),
            _pilot_summary,
        ),
        Analysis("E1 flagged-strata analysis", (row_labels, e1_scores), _e1),
        Analysis(
            "E2 paired-repair figure and summary",
            (REFERENCE_EXPERIMENT_ARTIFACTS_DIR / "e2_repair_summary.json",),
            _e2,
        ),
        Analysis(
            "E3 blind-preference figure and summary",
            (REFERENCE_EXPERIMENT_ARTIFACTS_DIR / "e3_pairwise_summary.json",),
            _e3,
        ),
        Analysis("human-validation annotation summary", human_inputs, _human_summary),
        Analysis("human-validation agreement figure", human_inputs + (e1_scores,), _human_figure),
        Analysis(
            "E4 rubric summary and figure",
            (TRAINING_EXPERIMENT_ARTIFACTS_DIR / "rubric_scores.jsonl",),
            _e4_rubric,
        ),
        Analysis(
            "E4 blind-pairwise summary",
            (TRAINING_EXPERIMENT_ARTIFACTS_DIR / "pairwise_judgments.jsonl",),
            _e4_pairwise,
        ),
        Analysis("supplementary lead-bias figure", (row_labels, e1_scores), _lead_bias),
    )


def main() -> int:
    completed: list[str] = []
    skipped: list[str] = []
    print("Reproducing deterministic paper outputs from available frozen artifacts.\n")
    for analysis in analyses():
        unavailable = [path for path in analysis.inputs if not path.is_file()]
        if unavailable:
            skipped.append(analysis.name)
            paths = ", ".join(str(path) for path in unavailable)
            print(f"SKIP  {analysis.name} — frozen inputs not currently available: {paths}")
            continue
        print(f"RUN   {analysis.name}")
        analysis.runner()
        completed.append(analysis.name)

    print("\nReproduction status")
    print(f"  completed analyses: {len(completed)}")
    for name in completed:
        print(f"    - {name}")
    print(f"  skipped analyses: {len(skipped)}")
    for name in skipped:
        print(f"    - {name}")

    if skipped:
        print(
            "\nPartial reproduction completed. Available figures and summaries were generated "
            "successfully. Some analyses were skipped because their frozen artifacts were unavailable."
        )
    else:
        print("\nFull deterministic reproduction completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
