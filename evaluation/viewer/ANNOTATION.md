# Human annotation round (F9a)

Blind rubric + pairwise validation of the LLM judge. **Three annotators** (amit, avreymi, ofek) split the worklist **disjointly** — each person scores a different subset (~50 rows each), not the same 150 rows three times.

## Setup (once per machine)

```bash
git fetch origin
git checkout main
git pull origin main
uv pip install -r requirements.txt
source .venv/bin/activate   # required — streamlit is not on PATH without the venv
```

The worklist is committed at `data_curation/artifacts/human_validation_worklist.json` — **do not rebuild** unless the lead regenerates it for everyone.

## Annotate

```bash
source .venv/bin/activate
streamlit run evaluation/viewer/annotate_app.py
```

Or without activating the venv:

```bash
uv run streamlit run evaluation/viewer/annotate_app.py
```

1. Pick **your** annotator ID in the sidebar (`amit`, `avreymi`, or `ofek` — do not use someone else's).
2. Confirm worklist version `v1-split` and seed `42` match what the lead shared.
3. The app shows only rows **assigned to you** (~50 rubric items plus pairwise tasks on your rewritten subset).
4. Score blind — do not discuss specific rows until all three finish their shares.
5. Progress saves to `data_curation/artifacts/human_annotations/{your_id}.jsonl` (tracked in git).
6. To **fix a mistake**, check **Allow editing submitted items** in the sidebar, uncheck **Only remaining**, filter to **rubric**, and update scores (saves replace the old record).
7. When finished, **commit and push** (the app shows the exact commands when your share is complete):

```bash
git pull origin main
git add data_curation/artifacts/human_annotations/amit.jsonl   # use your id
git commit -m "Add F9a human annotations (amit)"
git push origin main
```

You can push partial progress too — `git pull` before you annotate again so you stay in sync with teammates' pushes.

## Lead: after all three finish

```bash
git pull origin main

python -m data_curation.analysis.human_validation_results
python -m data_curation.analysis.human_validation_figures
```

(`human_validation_results` defaults to all three files under `data_curation/artifacts/human_annotations/`.)

Analysis reports **judge–human κ per annotator** on each person's slice, plus **pooled judge–human κ** over the combined disjoint set. Human–human κ is not computed (no overlapping rows).

Check completion for one annotator:

```bash
python -m data_curation.analysis.human_validation_results --check \
  --annotations data_curation/artifacts/human_annotations/ofek.jsonl
```

## Regenerate worklist (lead only)

Requires local pipeline artifacts (`row_labels.json`, `tail_boilerplate_removed.json`, `final_clean_hesum.json`):

```bash
python -m data_curation.analysis.build_human_validation_sample
```

Commit the new JSON if the sample changes — all three annotators must use the same file (with the same per-person assignments).
