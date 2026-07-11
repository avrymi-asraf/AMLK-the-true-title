# AMLK — Hebrew news summarization

Fine-tune **`dicta-il/dictalm2.0-instruct`** on Hebrew journalism data (HeSum), evaluate with
ROUGE / BERTScore / LLM-as-judge, and probe lead bias. Full project docs live in
**[AGENTS.md](AGENTS.md)** (architecture, file map, runbook, status). Plan of record:
[`docs/research-proposal-revised.md`](docs/research-proposal-revised.md).

## Pipeline (one path)

1. **Preprocess (always clean)** — drop multi-headline roundups, normalize pipe digests to prose,
   hardened summarization prompt → `outputs/data/processed/<variant>/`
2. **Train (1 epoch default)** — QLoRA / LoRA / full on HF Jobs; wandb project
   `amlk-dictalm2-instruct`; run names `{date}_{model}_{method}_{variant}_{N}ep[_tag]`
3. **Stability** — checkpoints on `/data/output` (resume after infra restart);
   `hub_strategy=every_save` commits adapters mid-run; predictions upload as soon as generated
4. **Evaluate** — finetuned / zero-shot base / Gemini baseline

```bash
source .env && source .venv/bin/activate
python -m data.download
python -m data.preprocess --variant whole
python -m training.train --submit-hf --hf-user avreymi --smoke-test   # verify
python -m training.train --submit-hf --hf-user avreymi                # 1-epoch full
python -m evaluation.eval_hf_job --submit-hf --hf-user avreymi
```

**Never train or load the model on the local 8 GB GPU** — use HuggingFace Jobs only.

See AGENTS.md for Hub repo names, wandb, monitoring, and the full eval battery.
