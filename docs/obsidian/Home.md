# AMLK — shared research notes

> Hebrew news abstractive summarization with dicta-il/DictaLM-3.0-1.7B-Base on HeSum.
> Last updated: 2026-07-09 (base model switched to DictaLM-3.0-1.7B-Base; pre-training stage).

## Map of content

### Literature & dataset
- [[HeSum Paper Insights]] — arXiv:2406.03897 takeaways for our setup
- [[Evaluation Metrics]] — ROUGE, AlephBERT BERTScore, judge, caveats
- [[Lead Bias Probe]] — positional-shortcut experiment (reviewer redesign)

### Training design
- [[Training Objective]] — what the model is trained on (CE loss) vs what we evaluate

### Project links
- [[References]] — papers, Hub repos, wandb
- Repo: `docs/research-proposal-revised.md`, `TODO.md`, `AGENTS.md`

## Status

Pre-training stage: pipeline validated end-to-end (a LoRA smoke run on
dicta-il/DictaLM-3.0-1.7B-Base completed successfully on HF Jobs), but no full training run
has happened yet. See `AGENTS.md` Status section for the current milestone.
