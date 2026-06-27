# AMLK — shared research notes

> Hebrew news abstractive summarization with Qwen3-2B on HeSum.  
> Last updated from team discussion: 2026-06-27.

## Map of content

### Problem & diagnosis
- [[Current Results]] — numbers from the first full HF Jobs run
- [[Prediction Failure Modes]] — what went wrong in fine-tuned outputs (92% broken)
- [[Training Objective]] — what the model is actually optimized on (CE loss, not ROUGE)

### Literature & dataset
- [[HeSum Paper Insights]] — arXiv:2406.03897 takeaways for our setup
- [[Evaluation Metrics]] — ROUGE, AlephBERT BERTScore, judge, caveats
- [[Lead Bias Probe]] — positional-shortcut experiment (reviewer redesign)

### Plan & implementation
- [[Fix Plan]] — phased fixes (decode → retrain → report)
- [[Decoding Configuration]] — generation settings that caused repetition loops

### Project links
- [[References]] — papers, Hub repos, wandb
- Repo: `docs/research-proposal-revised.md`, `TODO.md`, `AGENTS.md`

## One-sentence summary

The fine-tuned model writes fluent Hebrew but **does not stop generating** under greedy decode; most failures are **decoding + undertraining**, not “Hebrew is impossible.” Recalibrate against HeSum SOTA and trust **AlephBERT BERTScore + judge** over ROUGE.

## Open decisions

- [ ] Run Phase 1 re-decode on existing adapter before paying for retrain
- [ ] Add `load_best_model_at_end` on `eval_loss` in Phase 2 (recommended)
- [ ] Restore Gemini API billing for advanced baseline (`gemini-predict.log` 403)
