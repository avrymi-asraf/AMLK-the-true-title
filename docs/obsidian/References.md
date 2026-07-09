# References

## Papers

| Resource | Link |
|----------|------|
| HeSum dataset paper | [arXiv:2406.03897](https://arxiv.org/pdf/2406.03897) |
| AlephBERT | Seker et al., ACL 2022 |
| BERTScore | Zhang et al., ICLR 2020 |

## Repo docs

| File | Purpose |
|------|---------|
| `docs/ANLP Project abstract.md` | Original proposal |
| `docs/research-proposal-revised.md` | Reviewer-updated probe design |
| `TODO.md` | Milestone tracker |
| `AGENTS.md` / `CLAUDE.md` | Project memory for agents |

## Hugging Face

| Artifact | ID |
|----------|-----|
| Base model | `dicta-il/DictaLM-3.0-1.7B-Base` |
| HeSum dataset | `biunlp/HeSum` |
| Trained adapter | `avreymi/amlk-dictalm3-1.7b-sft` |
| Training data (processed) | `avreymi/amlk-training-data` |
| HeSum mLongT5 (paper) | `biunlp/mT5LongHeSum-large` |
| AlephBERT | `onlplab/alephbert-base` |

## Monitoring

- wandb project: `amlk-hebrew-summarization`
- HF Jobs: `hf jobs ps` / `hf jobs logs <id>`

## Key code paths

| Component | Path |
|-----------|------|
| Prompt / variants | `data/prompts.py` |
| Preprocess | `data/preprocess.py` |
| Train (local submit) | `training/train.py` |
| Train (HF Jobs) | `training/train_hf_job.py` |
| Config | `training/config.py` |
| Gemini baseline | `evaluation/predict.py` |
| Metrics | `evaluation/evaluate.py` |
| Error analysis | `evaluation/error_analysis.py` |

## Prior chat context

Research proposal revision and presentation prep: agent transcript `f3cdd137-74db-4222-aaa4-63f906607d75` (June 2026).

Related: [[Home]]
