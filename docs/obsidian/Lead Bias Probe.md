# Lead Bias Probe

Research question: does the fine-tuned model **aggregate global context** or **latch onto the lead**?

## Reframed for the dataset review (2026-07-26)

The question survives the [[Project Pivot]], in better shape, but the *design* below is superseded.

**Why the old design cannot work as written.** It presupposes trustworthy references. Measuring lead
reliance against gold summaries that are themselves lead-aligned news subheadings mostly measures the
references, not the model — and [[Dataset Defect Taxonomy]] establishes that a large fraction of those
references are unfit as targets at all.

**Tension with the curation pipeline.** The token-budget filter in [[Data Curation Pipeline]] drops every
row over 4,000 tokens — 2,659 rows, 26.6% of the corpus — which is exactly the long-article tail where
lead bias is most measurable. A short article has little distinction between lead and body; a 6,000-token
one has a great deal. Curation is convenient for training and destructive for this probe.

**The resolution.** Stop treating input slicing as a training variant and treat lead overlap as a
**continuous per-row covariate** computed over all 10,000 rows, including the ones the filter removed:

- `headline_lead_overlap` — token overlap between the headline and the article's first N tokens
- Regressed against `log(article_tokens)` alongside the rubric sub-scores from [[Reference Quality Rubric]]
- Reported as figure F5 in [[Paper Figures]], two panels on a shared length axis

This is strictly more informative than the three-variant training probe. It uses every row rather than
three subsets, it needs no extra training runs, and it separates two things the old design conflated:
whether *references* are lead-aligned (a dataset property, measurable now) and whether a *model* relies
on the lead (a model property, measurable from the arms described in [[Training Handoff Contract]]).

The `--variant whole|lead|body` machinery in `data/preprocess.py` stays in the tree but is not used.

Everything below is the Qwen-era design, retained for the record.

---

## Reviewer redesign (`docs/research-proposal-revised.md`)

**Old design (data question):** train separate models on whole / lead / body slices.  
**New design (model question):** train **one** model on whole articles; **ablate input at inference**:

| Input at test | What it tests |
|---------------|---------------|
| Whole article | Full capability |
| Lead-only | Reliance on opening |
| Body-only | Can it use non-lead content? |

## Controls (TODO F)

1. **Body-supported subset** — gold summary content appears in body, not only lead (summary↔body overlap filter)
2. **Length-matched cut** — remove same #tokens as lead from random post-lead span (length confound)
3. **Sanity:** Gemini/advanced baseline still summarizes body-supported examples without lead

## Code already in repo

- `data/prompts.py` → `make_variant(text, "whole"|"lead"|"body")`
- `data/preprocess.py` → `--variant`
- `evaluation/predict.py` → `--variant` for Gemini baseline

## Empirical signal from current run

From [[Prediction Failure Modes]]: **143/1000** finetuned preds have >60% word overlap with article lead — quantify as **lead_copying** rate in error analysis, not only a bug.

HeSum gold summaries are **journalistic subheadings** — often lead-aligned by construction (`TODO.md` B'.4 data characterization).

## Training-distribution experiment (optional F.7)

Train two whole-article models:
- Low summary↔lead overlap subset
- Matched random subset  

Compare inference-time lead reliance between them.

Related: [[HeSum Paper Insights]], [[Fix Plan#Phase 3]], `TODO.md` section F
