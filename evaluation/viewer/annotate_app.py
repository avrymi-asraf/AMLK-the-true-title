"""
F9a human-validation UI: blind rubric scoring and pairwise comparison over the frozen
human_validation_worklist.json. Extends the evaluation viewer package alongside the read-only
predictions browser (evaluation/viewer/app.py). Annotators run locally, CPU-only; progress
Progress appends to per-annotator JSONL under `data_curation/artifacts/human_annotations/` (git-tracked).

Run: streamlit run evaluation/viewer/annotate_app.py
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from evaluation.rubric_judge import DIMENSIONS, DIMENSION_QUESTIONS, RUBRIC_LEVELS
from evaluation.viewer.annotation_data import (
    DEFAULT_WORKLIST_PATH,
    TEAM_ANNOTATOR_IDS,
    annotations_git_path,
    annotation_lookup,
    append_annotation,
    build_pairwise_record,
    build_rubric_record,
    completed_keys,
    default_annotations_path,
    expand_tasks,
    export_summary,
    filter_task_items,
    load_annotations,
    load_worklist,
    pairwise_presentation,
    upsert_annotation,
)


def rtl_block(label: str, text: str) -> None:
    st.markdown(f"**{label}**")
    body = html.escape(text) if text else "<i>(empty)</i>"
    st.markdown(
        f'<div dir="rtl" style="text-align: right; white-space: pre-wrap; '
        f'font-size: 1.05rem; line-height: 1.7; padding: 0.25rem 0;">{body}</div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="AMLK Human Validation (F9a)", layout="wide")
    st.title("Human Judge Validation")
    st.caption("Blind rubric + pairwise annotation for F9a — local, CPU-only.")

    with st.sidebar:
        st.header("Annotator")
        preset = st.selectbox("Annotator ID", TEAM_ANNOTATOR_IDS, index=None, placeholder="Choose your id…")
        custom_id = st.text_input("Or type a custom id", value="", placeholder="only if not listed above")
        annotator_id = (custom_id.strip() or preset or "").strip()
        st.session_state.annotator_id = annotator_id

        worklist_path = st.text_input("Worklist path", value=str(DEFAULT_WORKLIST_PATH))
        if not Path(worklist_path).exists():
            st.error("Worklist not found. Run: python -m data_curation.analysis.build_human_validation_sample")
            return

        worklist = load_worklist(worklist_path)
        st.caption(f"Worklist {worklist.get('version', '?')} · seed {worklist.get('seed', '?')}")

        if annotator_id:
            ann_path = default_annotations_path(annotator_id)
        else:
            ann_path = Path("outputs/results/human_annotations.jsonl")
        ann_path = Path(st.text_input("Annotations file", value=str(ann_path)))

        annotations = load_annotations(ann_path) if annotator_id else []
        completed = completed_keys(annotations)
        lookup = annotation_lookup(annotations)
        all_items = expand_tasks(worklist, annotator_id=annotator_id) if annotator_id else []
        summary = export_summary(annotations, worklist, annotator_id=annotator_id or None)
        if worklist.get("split_mode") == "disjoint" and annotator_id:
            assignment = worklist.get("assignment", {})
            if annotator_id in assignment:
                st.caption(f"Your assigned share: {assignment[annotator_id]} rows (disjoint split)")
        st.metric("Progress", f"{summary['total_done']} / {summary.get('total_tasks', len(all_items))}")
        st.write(
            f"Rubric: {summary['rubric_done']} / {summary.get('rubric_total', 0)} · "
            f"Pairwise: {summary['pairwise_done']} / {summary.get('pairwise_total', 0)}"
        )

        task_filter = st.selectbox("Show tasks", ["all", "rubric", "pairwise"])
        only_remaining = st.checkbox("Only remaining", value=True)
        allow_editing = st.checkbox("Allow editing submitted items", value=False)
        show_admin = st.checkbox("Admin: show judge scores (post-hoc)", value=False)

        if ann_path.exists():
            with open(ann_path, "rb") as f:
                st.download_button(
                    "Download annotations JSONL (backup)",
                    data=f.read(),
                    file_name=ann_path.name,
                    mime="application/json",
                )

        if annotator_id and summary.get("total_tasks") and summary["total_done"] >= summary["total_tasks"]:
            rel = annotations_git_path(annotator_id)
            st.success("All tasks complete — push your file to git:")
            st.code(
                f"git add {rel}\n"
                f"git commit -m \"Add F9a human annotations ({annotator_id})\"\n"
                "git push origin main",
                language="bash",
            )

    if not annotator_id:
        st.info("Enter your annotator ID in the sidebar to begin.")
        return

    if not all_items:
        st.warning(
            "No rows assigned to this annotator ID. Pick amit, avreymi, or ofek from the sidebar "
            "(disjoint split — each person scores a different subset)."
        )
        return

    navigable = filter_task_items(
        all_items, completed, task_filter=task_filter, only_remaining=only_remaining,
    )
    if not navigable and only_remaining:
        st.success("All tasks in this filter are complete.")
        navigable = filter_task_items(all_items, completed, task_filter=task_filter, only_remaining=False)
    if not navigable:
        st.warning("No items to show.")
        return

    if "ann_pos" not in st.session_state:
        st.session_state.ann_pos = 0
    st.session_state.ann_pos = max(0, min(st.session_state.ann_pos, len(navigable) - 1))

    nav = st.columns([1, 1, 1, 3])
    if nav[0].button("⟵ Prev", use_container_width=True):
        st.session_state.ann_pos = max(0, st.session_state.ann_pos - 1)
    if nav[1].button("Next ⟶", use_container_width=True):
        st.session_state.ann_pos = min(len(navigable) - 1, st.session_state.ann_pos + 1)
    if nav[2].button("Skip", use_container_width=True):
        st.session_state.ann_pos = min(len(navigable) - 1, st.session_state.ann_pos + 1)
    jump = nav[3].number_input(
        f"Item (1-{len(navigable)})",
        min_value=1,
        max_value=len(navigable),
        value=st.session_state.ann_pos + 1,
    )
    st.session_state.ann_pos = int(jump) - 1

    item = navigable[st.session_state.ann_pos]
    hesum_id = item["hesum_id"]
    task = item["task"]
    is_done = (hesum_id, task) in completed
    can_edit = allow_editing or not is_done
    existing = lookup.get((hesum_id, task))

    st.caption(f"Item {st.session_state.ann_pos + 1}/{len(navigable)} · id {hesum_id} · task {task}")
    if is_done and not allow_editing:
        st.info("Already submitted — enable 'Allow editing submitted items' in the sidebar to revise.")
    elif is_done and allow_editing:
        st.warning("Editing mode — saving will replace your previous submission for this item.")

    with st.expander("Article", expanded=False):
        rtl_block("Article", item.get("text", ""))

    if task == "rubric":
        st.subheader("Rubric — score the headline")
        rtl_block("Headline", item.get("original_headline", ""))

        scores = {}
        existing_scores = existing.get("scores", {}) if existing and existing.get("task") == "rubric" else {}
        for dim in DIMENSIONS:
            with st.expander(f"{dim.replace('_', ' ').title()} — {DIMENSION_QUESTIONS[dim]}", expanded=False):
                for level in sorted(RUBRIC_LEVELS[dim].keys(), reverse=True):
                    st.caption(f"{level}: {RUBRIC_LEVELS[dim][level]}")
            prev = existing_scores.get(dim)
            scores[dim] = st.radio(
                dim,
                options=list(range(1, 6)),
                index=(prev - 1) if prev in range(1, 6) else 0,
                horizontal=True,
                key=f"rubric_{hesum_id}_{dim}",
                disabled=not can_edit,
            )

        submit_label = "Update rubric scores" if is_done else "Submit rubric scores"
        if st.button(submit_label, disabled=not can_edit):
            record = build_rubric_record(annotator_id, hesum_id, scores)
            upsert_annotation(ann_path, record)
            st.success("Saved.")
            st.rerun()

    elif task == "pairwise":
        st.subheader("Pairwise — which headline is better?")
        presentation = pairwise_presentation(
            annotator_id,
            hesum_id,
            item["original_headline"],
            item["curated_headline"],
        )
        rtl_block("Headline A", presentation["headline_a"])
        rtl_block("Headline B", presentation["headline_b"])

        cols = st.columns(3)
        for col, label, winner in zip(cols, ["A better", "B better", "Tie"], ["a", "b", "tie"]):
            if col.button(label, disabled=not can_edit, use_container_width=True):
                record = build_pairwise_record(
                    annotator_id, hesum_id, winner, presentation["slot_map"],
                )
                upsert_annotation(ann_path, record)
                st.success("Saved.")
                st.rerun()

    if show_admin and is_done:
        st.divider()
        st.markdown("**Admin (judge reference)**")
        e1_path = Path("outputs/results/e1_rubric_scores.jsonl")
        if e1_path.exists() and task == "rubric":
            for line in e1_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("hesum_id") == hesum_id:
                    st.json(row.get("scores", {}))
                    break
        e3_path = Path("outputs/results/e3_pairwise.jsonl")
        if e3_path.exists() and task == "pairwise":
            for line in e3_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("hesum_id") == hesum_id:
                    st.json({k: row[k] for k in ("winner", "curated_wins") if k in row})
                    break


if __name__ == "__main__":
    main()
