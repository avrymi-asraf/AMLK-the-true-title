# Human annotation round (F9a)

Blind rubric + pairwise validation of the LLM judge. Two annotators, independent, same frozen worklist.

## Setup (once per machine)

```bash
git fetch origin
git checkout feature/human-validation-ui
uv pip install -r requirements.txt
```

The worklist is committed at `data_curation/artifacts/human_validation_worklist.json` — **do not rebuild** unless the lead regenerates it for everyone.

## Annotate

```bash
streamlit run evaluation/viewer/annotate_app.py
```

1. Enter **your** annotator ID in the sidebar (e.g. `amit`, `avreymi`).
2. Confirm worklist version `v1` and seed `42` match what the lead shared.
3. Score blind — do not discuss specific rows until both finish.
4. Progress saves to `outputs/results/human_annotations_{your_id}.jsonl`.
5. Use **Download annotations JSONL** when done and send the file to the lead (Slack/Drive — not git).

## Lead: after both finish

```bash
python -m data_curation.analysis.human_validation_results \
  --annotations outputs/results/human_annotations_amit.jsonl \
               outputs/results/human_annotations_avreymi.jsonl

python -m data_curation.analysis.human_validation_figures
```

Check completion only:

```bash
python -m data_curation.analysis.human_validation_results --check \
  --annotations outputs/results/human_annotations_amit.jsonl
```

## Regenerate worklist (lead only)

Requires local pipeline artifacts (`row_labels.json`, `tail_boilerplate_removed.json`, `final_clean_hesum.json`):

```bash
python -m data_curation.analysis.build_human_validation_sample
```

Commit the new JSON if the sample changes — both annotators must use the same file.
