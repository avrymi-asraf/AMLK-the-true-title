# Human annotation round (F9a)

Blind rubric + pairwise validation of the LLM judge. **Three annotators** (amit, avreymi, ofek), independent, same frozen worklist.

## Setup (once per machine)

```bash
git fetch origin
git checkout feature/human-validation-ui
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
2. Confirm worklist version `v1` and seed `42` match what the lead shared.
3. Score blind — do not discuss specific rows until all three finish.
4. Progress saves to `outputs/results/human_annotations_{your_id}.jsonl`.
5. Use **Download annotations JSONL** when done and send the file to the lead (Slack/Drive — not git).

## Lead: after all three finish

```bash
python -m data_curation.analysis.human_validation_results \
  --annotations outputs/results/human_annotations_amit.jsonl \
               outputs/results/human_annotations_avreymi.jsonl \
               outputs/results/human_annotations_ofek.jsonl

python -m data_curation.analysis.human_validation_figures
```

Check completion for one annotator:

```bash
python -m data_curation.analysis.human_validation_results --check \
  --annotations outputs/results/human_annotations_ofek.jsonl
```

## Regenerate worklist (lead only)

Requires local pipeline artifacts (`row_labels.json`, `tail_boilerplate_removed.json`, `final_clean_hesum.json`):

```bash
python -m data_curation.analysis.build_human_validation_sample
```

Commit the new JSON if the sample changes — all three annotators must use the same file.
