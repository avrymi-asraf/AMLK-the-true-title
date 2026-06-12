# TODO — AMLK Hebrew Summarization

Implements `docs/ANLP Project abstract.md`. Milestones from the abstract are dated below.

## A. Training pipeline — DONE
- [x] A.1 Download datasets (HeSum 10,000 records; IAHLT inaccessible with current credentials)
- [x] A.2 Base model: Qwen/Qwen3-2B
- [x] A.3 Fine-tune via HF `trl` SFT — one `training/train.py` for qlora | lora | full,
      completion-only loss, wandb logging, local or HF Jobs (`--submit-hf`)

## B. Evaluation pipeline — DONE (Stage B, due 07.06)
- [x] B.1 ROUGE-1/2/L (Hebrew-aware tokenizer)
- [x] B.2 BERTScore (xlm-roberta-large)
- [x] B.3 Gemini LLM-as-judge (faithfulness + fluency, 1-5)
- [x] Zero-shot Qwen3-2B baseline + Gemini advanced baseline (`evaluation/predict.py`)
- [x] Error analysis: failure-type labelling on a sample (`evaluation/error_analysis.py`)

## C. Literature & framing — DONE (24.05)
- [x] Related work, abstract, goals/milestones — see `docs/ANLP Project abstract.md`

## D. Initial results — IN PROGRESS
- [ ] D.1 Full QLoRA run on HF Jobs + evaluation battery (finetuned vs zero-shot vs Gemini)
- [ ] D.2 Improve training (regime comparison: lora / full FT)

## E. Present results — 14.06
- [ ] E.1 Paper draft
- [ ] E.2 Presentation: QLoRA/LoRA/full vs baselines, news/journalism framing

## F. Truncation / positional-shortcut probe — 30.06
- [ ] Train one model per variant (whole / lead / body) and evaluate each on its matching split.
      Code is ready (`--variant`); training runs deferred to this milestone.

## G. Finalize — 31.07
- [ ] Final paper and presentation
