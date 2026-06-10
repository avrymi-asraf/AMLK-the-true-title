---
name: colab-pipeline-inspection
description: Use when running pipeline inspection with the real generator model on a Colab GPU — iterating D prompts, testing unit-split strategies, or trying different G checkpoints (any inspection that needs the full model on GPU, not the API proxy used by scripts/pipeline_inspection.py).
---

# Colab Pipeline Inspection

The local `scripts/pipeline_inspection.py` uses an API proxy (fast, cheap, inspection-only). To inspect with the **real generator model on a GPU** — whichever model is under investigation — use **colab-cli** to provision a T4 session, load the model once, then iterate live without restarting.

G is never hardcoded: the setup script builds it from `config.generator`, so swapping the model under test is just editing `config.generator["model_id"]` — the same live-edit pattern used for D prompts below.

## Critical Constraints

| Rule | Why |
|------|-----|
| `--timeout 600` for model load | Default `exec` timeout is 10 s; model takes 3-5 min |
| Never use `colab repl` / `colab console` interactively | Both require a TTY and hang in agent context |
| Pipe stdin for all iterative code: `echo "..." \| colab exec` | Only mode that works headlessly |
| Kernel state persists across `exec` calls | Load G once; rebuild D fast with stdin snippets |
| `colab upload` requires parent dir to exist on VM | Create dir first via kernel exec if needed |
| Always `colab stop` when done | Idle VMs burn compute units |

## Phase 1: Setup (one-time, ~7 min)

```bash
# Provision T4 GPU session
colab new -s inspect --gpu T4

# Inject secrets from local env (never hardcode)
echo "import os; os.environ['HF_TOKEN']='${HF_TOKEN}'; os.environ['GEMINI_API_KEY']='${GEMINI_API_KEY}'" \
  | colab exec -s inspect --timeout 30

# Run setup script — clones repo, installs deps, loads G + config + questions + D
# After this finishes, kernel holds: generator, config, questions, decision_model, run_pipeline_inspection
colab exec -s inspect -f scripts/colab_inspect_setup.py --timeout 600
```

**Verify setup completed:**
```bash
echo "print(f'G={config.generator[\"model_id\"]}, Q={len(questions)}, D-prompt={config.decision[\"prompt_version\"]}')" \
  | colab exec -s inspect --timeout 15
```

**Swap the G model under test (no file edits):**
```bash
echo "
from dataclasses import replace
config.generator['model_id'] = 'some-org/some-other-model'
generator = create_generator_from_config(config.generator, config.generation, max_units_per_batch=2)
print('G ready:', config.generator['model_id'])
" | colab exec -s inspect --timeout 600
```

## Phase 2: Iterative Inspection (fast, no model reload)

### Run inspection on a question

```bash
mkdir -p output/pipeline_inspection
echo "
rows = run_pipeline_inspection(
    question=questions[3],
    generator=generator,
    decision_model=decision_model,
    config=config,
)
" | colab exec -s inspect --timeout 180 | tee output/pipeline_inspection/q3_base.txt
```

### Swap D prompt and re-run

```bash
# Edit prompt locally, then upload (parent dir already exists from git clone)
colab upload -s inspect prompts/my-new-prompt.txt /content/reasoning-pruning/prompts/my-new-prompt.txt

# Rebuild D — fast, no model reload
echo "
config.decision['prompt_version'] = 'my-new-prompt'
decision_model = create_decision_model_from_config(
    config.decision, config.pruning, prompts_dir='/content/reasoning-pruning/prompts'
)
print('D ready:', config.decision['prompt_version'])
" | colab exec -s inspect --timeout 30

# Run inspection and capture
echo "
rows = run_pipeline_inspection(question=questions[3], generator=generator, decision_model=decision_model, config=config)
" | colab exec -s inspect --timeout 180 | tee output/pipeline_inspection/q3_my-new-prompt.txt
```

### Change unit-split strategy

```bash
echo "
from dataclasses import replace
config = replace(config, unit_split_strategy='clauses')
print('unit_split_strategy:', config.unit_split_strategy)
" | colab exec -s inspect --timeout 15
```

### Run multiple questions

```bash
echo "
for qi in [1, 3, 5, 7]:
    print(f'\n=== Q{qi} ===')
    run_pipeline_inspection(question=questions[qi], generator=generator, decision_model=decision_model, config=config)
" | colab exec -s inspect --timeout 600 | tee output/pipeline_inspection/multi_run.txt
```

## Phase 3: Save Output

Stdout from `colab exec` is the inspection output — always `tee` to `output/pipeline_inspection/`.

To download a file written on the VM:
```bash
colab download -s inspect /content/reasoning-pruning/output/result.json output/pipeline_inspection/result.json
```

Export full session history as markdown:
```bash
colab log -s inspect -o output/pipeline_inspection/session.md
```

## Phase 4: Cleanup

```bash
colab stop -s inspect
```

## Relationship to the notebook

`scripts/colab_inspect_setup.py` is the headless equivalent of the browser notebook's setup cells (`notebooks/data_creation_playground.ipynb`): both load the config, build G and D, load questions, and call the **same** loop entry point `run_pipeline_inspection` / `build_rows_for_question`. Each `colab exec` stdin snippet is the equivalent of running one notebook cell against the persistent kernel. Humans use the notebook in a browser; agents use this script + stdin snippets. Keep the two in sync when the library API changes (Notebook Alignment Rule).

(The setup script builds G config-driven via `create_generator_from_config` so any model works; the notebook still constructs its generator inline — making it config-driven too is a worthwhile follow-up for full model-variety parity.)

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `exec` times out immediately | Add `--timeout 600` — default is 10 s |
| "Session not found" | `colab sessions` to check; re-run Phase 1 if pruned |
| `repl` / `console` hangs | Needs TTY — always pipe stdin instead |
| Kernel deadlocked | `colab restart-kernel -s inspect`; re-run model-load block |
| Upload 500 error | Parent dir doesn't exist on VM — create it first via exec |
| `create_decision_model_from_config` not defined | Run setup or import it: `echo "from reasoning_pruning.clients import create_decision_model_from_config" \| colab exec` |
| G produces wrong output | Verify HF_TOKEN: `echo "import os; print(os.environ.get('HF_TOKEN','MISSING')[:8])" \| colab exec` |
