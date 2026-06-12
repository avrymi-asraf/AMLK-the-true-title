# TODO — AMLK Hebrew Summarization

Implements `docs/ANLP Project abstract.md` and `docs/research-proposal.md`. Milestones from the abstract are dated below.

## A. Training pipeline — DONE
- [x] A.1 Download datasets (HeSum 10,000 records; IAHLT inaccessible with current credentials)
- [x] A.2 Base model: Qwen/Qwen3-2B
- [x] A.3 Fine-tune via HF `trl` SFT — one `training/train.py` for qlora | lora | full,
      completion-only loss, wandb logging, local or HF Jobs (`--submit-hf`)

## B. Evaluation pipeline — DONE (Stage B, due 07.06)
- [x] B.1 ROUGE-1/2/L (Hebrew-aware tokenizer)
- [x] B.2 BERTScore (xlm-roberta-large)
- [x] B.3 Gemini LLM-as-judge (faithfulness + fluency, 1-5)
- [x] B.4 Advanced-model baseline: Gemini API on the same Hebrew test set + prompt; score with B.1–B.3
- [x] B.5 Error analysis: failure-type labelling on a ~50–100 sample (`evaluation/error_analysis.py`)

## C. Literature & framing — DONE (24.05)
- [x] C.1 Survey English news summarization (datasets, models, lead bias, metric limits) and map lessons to Hebrew setup
- [x] C.2 Abstract / research proposal — see `docs/ANLP Project abstract.md`, `docs/research-proposal.md`
- [x] C.3 Goals and milestones

## D. Initial results — IN PROGRESS
- [ ] D.1 Full QLoRA run on HF Jobs + evaluation battery (finetuned vs zero-shot vs Gemini)
- [ ] D.2 Improve training (regime comparison: lora / full FT)

## E. Present results — 14.06
- [ ] E.1 Paper draft
- [ ] E.2 Presentation: QLoRA/LoRA/full vs baselines, news/journalism framing

## F. Truncation / positional-shortcut probe — 30.06
- [ ] F.1 Preprocess whole / lead / body variants (`--variant`; code ready)
- [ ] F.2 Train one model per variant with identical hyperparameters
- [ ] F.3 Evaluate each model on its matching test split (ROUGE / BERTScore / LLM-judge)
- [ ] F.4 Hypothesis: Body-only drop vs Whole/Lead-only indicates positional shortcuts (lead overlap with reference)

## G. Hebrew news / headline control (optional)
- [ ] G.1 Emphasize journalism subset in analysis (HeSum + IAHLT; stratify or report by source)
- [ ] G.2 Optional: alternate instructions (one-line headline vs multi-sentence summary) and compare metrics

## H. Finalize — 31.07
- [ ] Final paper and presentation
