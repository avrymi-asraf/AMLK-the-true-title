"""F9a human-validation analysis: Cohen's kappa human–human and judge–human per rubric dimension,
plus pairwise agreement vs E3. Consumes per-annotator JSONL from the annotation UI and joins to
`e1_rubric_scores.jsonl` / `e3_pairwise.jsonl`. Local, CPU-only.

Run:
    python -m data_curation.analysis.human_validation_results \\
        --annotations outputs/results/human_annotations_amit.jsonl \\
                    outputs/results/human_annotations_avreymi.jsonl
    python -m data_curation.analysis.human_validation_results --check --annotations ...

Output:
    outputs/results/human_validation_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from data_curation.analysis.build_human_validation_sample import WORKLIST_PATH
from data_curation.analysis.rubric_pilot import weighted_kappa
from evaluation.pairwise_judge import curated_wins
from evaluation.rubric_judge import DIMENSIONS
from evaluation.viewer.annotation_data import (
    DEFAULT_WORKLIST_PATH,
    TEAM_ANNOTATOR_IDS,
    default_team_annotation_paths,
    expand_tasks,
    load_worklist,
)

E1_SCORES_PATH = Path(__file__).resolve().parents[2] / "outputs" / "results" / "e1_rubric_scores.jsonl"
E3_PAIRWISE_PATH = Path(__file__).resolve().parents[2] / "outputs" / "results" / "e3_pairwise.jsonl"
SUMMARY_PATH = Path(__file__).resolve().parents[2] / "outputs" / "results" / "human_validation_summary.json"


def load_human_annotations(paths: list[Path]) -> list[dict]:
    """Merge annotation records from one or more JSONL files."""
    records = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def group_by_annotator(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        grouped[rec["annotator_id"]].append(rec)
    return dict(grouped)


def rubric_scores_by_id(annotations: list[dict]) -> dict[str, dict[str, int]]:
    """Map hesum_id -> {dimension: score} from rubric task records (latest per id)."""
    from evaluation.viewer.annotation_data import dedupe_annotations

    out: dict[str, dict[str, int]] = {}
    for rec in dedupe_annotations(annotations):
        if rec.get("task") != "rubric":
            continue
        out[rec["hesum_id"]] = rec["scores"]
    return out


def load_judge_rubric_scores(path: Path = E1_SCORES_PATH) -> dict[str, dict[str, int]]:
    scores: dict[str, dict[str, int]] = {}
    if not path.exists():
        return scores
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            hesum_id = row["hesum_id"]
            scores[hesum_id] = {
                dim: row["scores"][dim]["score"]
                for dim in DIMENSIONS
                if dim in row.get("scores", {})
            }
    return scores


def paired_dimension_scores(
    scores_a: dict[str, dict[str, int]],
    scores_b: dict[str, dict[str, int]],
) -> dict[str, tuple[list[int], list[int]]]:
    """Align rubric scores on shared hesum_ids per dimension."""
    shared_ids = sorted(set(scores_a) & set(scores_b))
    paired: dict[str, tuple[list[int], list[int]]] = {}
    for dim in DIMENSIONS:
        a_list, b_list = [], []
        for hesum_id in shared_ids:
            if dim in scores_a[hesum_id] and dim in scores_b[hesum_id]:
                a_list.append(scores_a[hesum_id][dim])
                b_list.append(scores_b[hesum_id][dim])
        paired[dim] = (a_list, b_list)
    return paired


def rubric_agreement(
    human_a: dict[str, dict[str, int]],
    human_b: dict[str, dict[str, int]],
    judge_scores: dict[str, dict[str, int]],
) -> dict:
    """Quadratic-weighted Cohen's kappa per dimension for human–human and judge–human."""
    hh = paired_dimension_scores(human_a, human_b)
    jh_a = paired_dimension_scores(judge_scores, human_a)
    jh_b = paired_dimension_scores(judge_scores, human_b)

    result = {"human_human": {}, "judge_human_a": {}, "judge_human_b": {}, "n_pairs": {}}
    for dim in DIMENSIONS:
        a, b = hh[dim]
        if len(a) >= 2:
            result["human_human"][dim] = weighted_kappa(a, b)
            result["n_pairs"][f"human_human_{dim}"] = len(a)
        ja, ha = jh_a[dim]
        if len(ja) >= 2:
            result["judge_human_a"][dim] = weighted_kappa(ja, ha)
            result["n_pairs"][f"judge_human_a_{dim}"] = len(ja)
        jb, hb = jh_b[dim]
        if len(jb) >= 2:
            result["judge_human_b"][dim] = weighted_kappa(jb, hb)
            result["n_pairs"][f"judge_human_b_{dim}"] = len(jb)
    return result


def human_pairwise_outcome(record: dict) -> str:
    """Map blind human winner + slot_map to curated / original / tie."""
    curated_is_a = record["slot_map"].get("a") == "curated"
    return curated_wins(record["winner"], curated_is_a)


def load_judge_pairwise_outcomes(path: Path = E3_PAIRWISE_PATH) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    if not path.exists():
        return outcomes
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("outcome") in {"curated", "original", "tie"}:
                outcomes[row["hesum_id"]] = row["outcome"]
    return outcomes


def pairwise_agreement(
    human_records: list[dict],
    judge_outcomes: dict[str, str],
) -> dict:
    """Percent agreement and kappa on curated/original/tie between human and judge."""
    human_pairwise = {
        r["hesum_id"]: human_pairwise_outcome(r)
        for r in human_records
        if r.get("task") == "pairwise"
    }
    shared = sorted(set(human_pairwise) & set(judge_outcomes))
    if not shared:
        return {"n": 0, "insufficient": True}

    label_order = ["original", "tie", "curated"]
    label_to_int = {label: i for i, label in enumerate(label_order)}
    human_labels = [label_to_int[human_pairwise[i]] for i in shared]
    judge_labels = [label_to_int[judge_outcomes[i]] for i in shared]
    agree = sum(h == j for h, j in zip(human_labels, judge_labels)) / len(shared)

    from sklearn.metrics import cohen_kappa_score

    kappa = round(cohen_kappa_score(human_labels, judge_labels, weights="quadratic"), 3)
    return {
        "n": len(shared),
        "percent_agreement": round(100 * agree, 2),
        "quadratic_kappa": kappa,
    }


def completion_check(
    annotations: list[dict],
    worklist_path: Path = DEFAULT_WORKLIST_PATH,
) -> dict:
    """Report missing (hesum_id, task) pairs vs the frozen worklist."""
    worklist = load_worklist(worklist_path)
    by_annotator = group_by_annotator(annotations)

    if worklist.get("split_mode") == "disjoint":
        annotator_ids = worklist.get("annotators", list(TEAM_ANNOTATOR_IDS))
        per_annotator: dict[str, dict] = {}
        total_expected = 0
        total_done = 0
        all_complete = True
        for aid in annotator_ids:
            expected = {
                (item["hesum_id"], item["task"])
                for item in expand_tasks(worklist, annotator_id=aid)
            }
            done = {
                (a["hesum_id"], a["task"])
                for a in by_annotator.get(aid, [])
            }
            missing = expected - done
            matched = len(done & expected)
            total_expected += len(expected)
            total_done += matched
            per_annotator[aid] = {
                "expected_tasks": len(expected),
                "submitted_tasks": matched,
                "missing_tasks": len(missing),
                "complete": len(missing) == 0,
            }
            if missing:
                all_complete = False
        return {
            "split_mode": "disjoint",
            "expected_tasks": total_expected,
            "submitted_tasks": total_done,
            "missing_tasks": total_expected - total_done,
            "per_annotator": per_annotator,
            "annotators": {aid: len(recs) for aid, recs in by_annotator.items()},
            "complete": all_complete,
        }

    expected = {(item["hesum_id"], item["task"]) for item in expand_tasks(worklist)}
    done = {(a["hesum_id"], a["task"]) for a in annotations}
    missing = sorted(expected - done)
    return {
        "expected_tasks": len(expected),
        "submitted_tasks": len(done),
        "missing_tasks": len(missing),
        "missing": [{"hesum_id": h, "task": t} for h, t in missing[:50]],
        "annotators": {aid: len(recs) for aid, recs in by_annotator.items()},
        "complete": len(missing) == 0,
    }


def judge_human_kappa(
    judge_scores: dict[str, dict[str, int]],
    human: dict[str, dict[str, int]],
) -> dict[str, float | None]:
    paired = paired_dimension_scores(judge_scores, human)
    return {
        dim: weighted_kappa(*paired[dim]) if len(paired[dim][0]) >= 2 else None
        for dim in DIMENSIONS
    }


def human_human_kappa_pair(
    human_a: dict[str, dict[str, int]],
    human_b: dict[str, dict[str, int]],
) -> dict[str, float | None]:
    paired = paired_dimension_scores(human_a, human_b)
    return {
        dim: weighted_kappa(*paired[dim]) if len(paired[dim][0]) >= 2 else None
        for dim in DIMENSIONS
    }


def build_summary(
    annotation_paths: list[Path],
    *,
    worklist_path: Path = DEFAULT_WORKLIST_PATH,
    e1_path: Path = E1_SCORES_PATH,
    e3_path: Path = E3_PAIRWISE_PATH,
) -> dict:
    records = load_human_annotations(annotation_paths)
    by_annotator = group_by_annotator(records)
    annotator_ids = sorted(by_annotator)
    worklist = load_worklist(worklist_path)
    split_mode = worklist.get("split_mode") == "disjoint"
    judge_scores = load_judge_rubric_scores(e1_path)
    judge_pairwise = load_judge_pairwise_outcomes(e3_path)

    summary: dict = {
        "annotators": annotator_ids,
        "completion": completion_check(records, worklist_path),
        "rubric": {},
        "pairwise": {},
    }

    human_by_annotator = {
        aid: rubric_scores_by_id(by_annotator[aid]) for aid in annotator_ids
    }

    judge_human = {
        aid: judge_human_kappa(judge_scores, human_by_annotator[aid])
        for aid in annotator_ids
    }

    pooled_human: dict[str, dict[str, int]] = {}
    for aid in annotator_ids:
        pooled_human.update(human_by_annotator[aid])

    rubric_block: dict = {
        "annotator_ids": annotator_ids,
        "split_mode": worklist.get("split_mode"),
        "judge_human": judge_human,
        "judge_human_pooled": judge_human_kappa(judge_scores, pooled_human),
    }

    if split_mode:
        rubric_block["note"] = (
            "Disjoint split — each annotator scored a unique subset; "
            "human–human κ not computed (no shared rows)."
        )
    else:
        human_human_pairs: dict[str, dict[str, float | None]] = {}
        for i, a in enumerate(annotator_ids):
            for b in annotator_ids[i + 1:]:
                key = f"{a}_vs_{b}"
                human_human_pairs[key] = human_human_kappa_pair(
                    human_by_annotator[a], human_by_annotator[b],
                )
        rubric_block["human_human_pairs"] = human_human_pairs
        if len(annotator_ids) >= 2:
            first_pair = f"{annotator_ids[0]}_vs_{annotator_ids[1]}"
            rubric_block["human_human"] = human_human_pairs.get(first_pair, {})
            rubric_block["judge_human_a"] = judge_human.get(annotator_ids[0], {})
            rubric_block["judge_human_b"] = judge_human.get(annotator_ids[1], {})
        elif len(annotator_ids) == 1:
            rubric_block["note"] = "Single annotator — no human–human κ"

    summary["rubric"] = rubric_block

    for aid in annotator_ids:
        summary["pairwise"][aid] = pairwise_agreement(by_annotator[aid], judge_pairwise)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="F9a human-validation agreement summary")
    parser.add_argument(
        "--annotations", nargs="*",
        help="human_annotations/*.jsonl files (default: all team paths in artifacts/)",
    )
    parser.add_argument("--worklist", default=str(WORKLIST_PATH))
    parser.add_argument("--e1-scores", default=str(E1_SCORES_PATH))
    parser.add_argument("--e3-pairwise", default=str(E3_PAIRWISE_PATH))
    parser.add_argument("--output", default=str(SUMMARY_PATH))
    parser.add_argument("--check", action="store_true", help="Only print completion check")
    args = parser.parse_args()

    paths = [Path(p) for p in args.annotations] if args.annotations else default_team_annotation_paths()
    if args.check:
        records = load_human_annotations(paths)
        report = completion_check(records, Path(args.worklist))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    summary = build_summary(
        paths,
        worklist_path=Path(args.worklist),
        e1_path=Path(args.e1_scores),
        e3_path=Path(args.e3_pairwise),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote summary → {output}")


if __name__ == "__main__":
    main()
