# TODO — AMLK Hebrew Summarization

Implements `docs/ANLP Project abstract.md`, `docs/research-proposal.md`, and the reviewer-revised
`docs/research-proposal-revised.md` (current plan of record). Milestones from the abstract are dated below.

## A. Training pipeline — DONE
- [x] A.1 Download datasets (HeSum 10,000 records; IAHLT inaccessible with current credentials)
- [x] A.2 Base model: dicta-il/dictalm2.0-instruct
- [x] A.3 Fine-tune via HF `trl` SFT — one `training/train.py` for qlora | lora | full,
      completion-only loss, **clean-only** data path, **1 epoch** default, informative wandb
      names (`amlk-{model}`, date/method/variant/epochs), mid-run Hub checkpoint commits
      (`hub_strategy=every_save` + `/data/output` resume), HF Jobs (`--submit-hf`)

## B. Evaluation pipeline — DONE (Stage B, due 07.06)
- [x] B.1 ROUGE-1/2/L (Hebrew-aware tokenizer)
- [x] B.2 BERTScore (default AlephBERT `onlplab/alephbert-base`, HeSum-comparable; `--bertscore-model` to override)
- [x] B.3 Gemini LLM-as-judge (faithfulness + fluency, 1-5)
- [x] B.4 Advanced-model baseline: Gemini API on the same Hebrew test set + prompt; score with B.1–B.3
- [x] B.5 Error analysis: failure-type labelling on a ~50–100 sample (`evaluation/error_analysis.py`)

### B'. Reviewer-driven evaluation upgrades
- [x] B'.1 Switch LLM judge OFF the Gemini family — `evaluation/hf_client.py` enables a HF-hosted judge (`--judge-provider hf --judge-model …`) to avoid self-preference bias vs the Gemini baseline
- [x] B'.2 ROUGE-Hebrew: HeSum-style morphological normalization (niqqud + final-form folding); `evaluate.py` reports raw + normalized
- [ ] B'.3 Add a simple extractive Lead-N baseline (first N sentences) scored with B.1–B.3
- [ ] B'.4 Data characterization before runs: abstractiveness (novel n-grams, extractive
      coverage/density, small manual check) + summary↔lead and summary↔body overlap

## C. Literature & framing — DONE (24.05)
- [x] C.1 Survey English news summarization (datasets, models, lead bias, metric limits) and map lessons to Hebrew setup
- [x] C.2 Abstract / research proposal — see `docs/ANLP Project abstract.md`, `docs/research-proposal.md`
- [x] C.3 Goals and milestones

## D. Initial results — IN PROGRESS
- [ ] D.1 Full **1-epoch clean QLoRA** run on `dicta-il/dictalm2.0-instruct` (HF Jobs) +
      evaluation battery (finetuned vs zero-shot vs Gemini)
- [ ] D.2 Improve training (regime comparison: lora / full FT; still 1 epoch per run)

## E. Present results — 14.06
- [ ] E.1 Paper draft
- [ ] E.2 Presentation: QLoRA/LoRA/full vs baselines, news/journalism framing

## F. Positional-shortcut probe — 30.06 (REDESIGNED per reviewer feedback)
Reframed from "train one model per input slice" (a question about the data) to "train one
whole-article model, ablate the input at inference" (what the trained model relies on). See
`docs/research-proposal-revised.md`.
- [ ] F.1 Train ONE model on whole articles (reuse the main fine-tuned model)
- [ ] F.2 Inference ablation: evaluate that model on Whole / Lead-only / Body-only inputs
- [ ] F.3 Control — information availability: restrict the primary probe to a "body-supported"
      subset (gold summary content present in the body, by summary↔body overlap)
- [ ] F.4 Control — input length: length-matched cut (remove #tokens == lead length from a random
      post-lead span) so Body-only is compared at equal length
- [ ] F.5 Sanity check: confirm the advanced baseline can still summarize the body-supported subset
      without the lead (otherwise a Body-only drop is uninformative)
- [ ] F.6 Hypothesis: large Body-only drop + small Lead-only drop (after controls) = lead reliance
- [ ] F.7 Training-distribution experiment: train two whole-article models (low summary↔lead overlap
      subset vs matched random subset); run the F.2 ablation on both; compare lead reliance

## G. Hebrew news / headline control (optional)
- [ ] G.1 Emphasize journalism subset in analysis (HeSum + IAHLT; stratify or report by source)
- [ ] G.2 Optional: alternate instructions (one-line headline vs multi-sentence summary) and compare metrics

## H. Finalize — 31.07
- [ ] Final paper and presentation



# add observation of the process.
in this stage I dont see the summary of my model, the output of the model on test/eval set, the what the llm judge said about the summary, and the error analysis. I want to see them. 
add evaluation-observation - it means I want to see the porcess of the evaluation so I can understand what happened, how the output look like and according that understand what works good in the train and what not.
we will do it - 1. by colab notebook that run and display everything.
to do it, you need to use the real functions that you use in the code. create it according the skill notebook-observability.
implement it and validate that it works.
I also want that agent can run the cells of the notebook by the colab-cli, in this way it can run cell by cell ant the agent can do the reaserch and observation by self. 