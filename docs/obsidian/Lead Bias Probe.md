# Lead Bias Probe

Research question: does the fine-tuned model **aggregate global context** or **latch onto the lead**?

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

## Where to look once trained

`evaluation/error_analysis.py`'s `lead_copying` rate (word overlap between prediction and article lead) is the metric to watch once a model is trained — HeSum gold summaries are **journalistic subheadings**, often lead-aligned by construction (`TODO.md` section F).

## Training-distribution experiment (optional F.7)

Train two whole-article models:
- Low summary↔lead overlap subset
- Matched random subset  

Compare inference-time lead reliance between them.

Related: [[HeSum Paper Insights]], `TODO.md` section F
