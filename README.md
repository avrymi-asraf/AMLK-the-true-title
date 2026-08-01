# AMLK — Hebrew news summarization

Fine-tune **`dicta-il/dictalm2.0-instruct`** on **curated** Hebrew journalism data (HeSum),
evaluate with ROUGE / BERTScore / LLM-as-judge, and probe lead bias. Full project docs live
in **[AGENTS.md](AGENTS.md)** (architecture, file map, runbook, status). Plan of record:
[`docs/research-proposal-revised.md`](docs/research-proposal-revised.md). Training diagnosis
notes: [`IMPROVEMENT_PLAN.md`](IMPROVEMENT_PLAN.md).

## Pipeline (main train path)

1. **Curate HeSum** (main `data_curation/`) → `final_clean_hesum.json`
   (`{hesum_id, text, headline}` under `data_curation/artifacts/` or
   `outputs/data/curated/`).
2. **Materialize + preprocess** — `data.download` + `data.preprocess` build Arrow
   splits `{text, summary, source, prompt, completion}` at
   `outputs/data/processed/<variant>/` (the only train contract). Hub copy (private):
   [`avreymi/amlk-training-data`](https://huggingface.co/datasets/avreymi/amlk-training-data).
3. **Train (1 epoch default)** — LoRA / QLoRA / full on HF Jobs (`a10g-small`, 8h);
   can reuse Hub data with `--skip-data-upload`; wandb project `amlk-dictalm2-instruct`.
   Decode defaults: **no** `repetition_penalty` / `no_repeat_ngram_size` (1.0 / 0).
4. **Stability** — checkpoints on `/data/output` (resume after infra restart);
   `hub_strategy=every_save` commits adapters mid-run; predictions upload as soon as generated.
5. **Evaluate** — finetuned / zero-shot base / Gemini baseline

```bash
source .env && source .venv/bin/activate
# Requires final_clean_hesum.json (data_curation product)
python -m data.download
python -m data.preprocess --variant whole --force
python -m training.train --submit-hf --hf-user avreymi --smoke-test --skip-data-upload
python -m training.train --submit-hf --hf-user avreymi --skip-data-upload   # 1-epoch full
python -m evaluation.eval_hf_job --submit-hf --hf-user avreymi
```

### E4 secondary path (raw vs curated SFT)

Paper experiment F8: two matched 1-epoch LoRA runs that differ only in training corpus
(raw HeSum vs curated), judged on the **same** curated test articles. Code is in-tree;
jobs not yet submitted. Plan + commands:
[`docs/e4-raw-vs-curated-training-plan.md`](docs/e4-raw-vs-curated-training-plan.md)
(`data.download_raw`, `preprocess --test-from`, `scripts.e4_score`). Also see the
[training skill](.agents/skills/training/SKILL.md) E4 block and TODO D'.

**Never train or load the model on the local 8 GB GPU** — use HuggingFace Jobs only.

See [AGENTS.md](AGENTS.md) for Hub repo names (including E4), wandb, resume flags, monitoring,
and the full eval battery.
