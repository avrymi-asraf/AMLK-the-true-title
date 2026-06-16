---
name: colab-cli
description: Use when operating the official Google Colab CLI from an agent or terminal, including remote Colab GPU/TPU sessions, colab run jobs, file transfer, logs, authentication, cleanup, or avoiding local GPU use.
---

# Colab CLI

Use this skill for the official `googlecolab/google-colab-cli`, not older community tools
with similar names. The CLI lets local code run on remote Colab CPU/GPU/TPU runtimes from a
terminal, which is useful for AMLK when a task needs cloud acceleration without using the
local 8 GB GPU.

Validated locally on 2026-06-16 with `google-colab-cli` 0.5.11. The working path on this
machine is `--auth=oauth2` with cached Colab CLI OAuth credentials for
`avreymi.asraf@gmail.com`; `--auth=adc` fails because ADC is not configured. A short
CPU-only session and a CPU-only `colab run` were exercised and stopped; no GPU/TPU runtime
was allocated.

## First Checks

The official CLI is young and command flags can move between releases.

```bash
colab version
colab help
colab help run
colab help exec
```

If `colab` is missing or stale, use `uv tool install google-colab-cli` or
`uv tool install -U google-colab-cli`. Current official docs say Linux and macOS are
supported; Windows is not.

Observed version drift matters:

- `colab readme` from 0.5.11 says the default auth is `adc`.
- `colab help` from the same install says the default auth is `oauth2`.
- Treat `colab help` on the installed binary as authoritative. In this project, pass
  `--auth=oauth2` explicitly unless the user asks for ADC.
- `whoami` is hidden from `colab help`, but `colab help whoami` works in 0.5.11.

## Mental Model

- A session is a live Jupyter kernel on a rented Colab VM. `colab new` allocates it;
  `colab stop` releases it.
- `colab exec -f script.py` reads the local file and sends its contents to the remote
  kernel. Do not upload source code just to execute it.
- Kernel state persists across `exec` calls in the same session. Use `restart-kernel` for
  clean Python state without releasing the VM.
- Commands run in `/content` by default. Use absolute `/content/...` paths for remote files.
- GPU/TPU sessions burn compute until stopped. Prefer `colab run` for unattended jobs
  because it tears the VM down automatically unless `--keep` is set.

## Agent Safety Rules

| Rule | Reason |
| --- | --- |
| Pass `--auth=oauth2` before the subcommand in this repo | Cached OAuth works here; ADC does not |
| Always name durable sessions with `-s NAME` | Later cleanup and logs are unambiguous |
| Prefer `colab run` for one-shot jobs | It provisions, runs, downloads outputs, and stops |
| Always `colab stop -s NAME` for manual sessions | Idle accelerators cost compute units |
| Never run unpiped `colab repl` or `colab console` from an agent | They expect a TTY and can hang |
| Treat `colab auth` and `colab drivemount` as human-interactive | They are VM-side auth/mount flows |
| Isolate parallel jobs with `--config /tmp/<job>.json` | Prevent session-state collisions |
| Check `colab sessions` before and after GPU work | Finds orphaned backend sessions |

## Cost Control

Run this order before any paid or scarce accelerator work:

1. `colab version` and `colab update`.
2. `colab --auth=oauth2 whoami` to verify identity and scopes.
3. `colab --auth=oauth2 sessions` to find existing allocations.
4. `colab --auth=oauth2 run /tmp/missing.py` as a preflight only if checking local script
   existence behavior; a missing script exits before allocation.
5. Use CPU first, then T4, and only then larger GPUs if the task really needs them.
6. Prefer `colab run` without `--keep` for unattended work; it self-cleans.
7. For manual sessions, run `colab stop -s NAME` in a `finally`/cleanup step and verify
   with `colab sessions`.

Do not start A100/H100/G4/L4 sessions while debugging auth, script paths, argument parsing,
or small data issues.

## Authentication

Use the Colab CLI OAuth cache in this repo. Do not use `gcloud` unless the user explicitly
asks for ADC setup.

```bash
colab --auth=oauth2 whoami
colab --auth=oauth2 sessions
```

The validated identity is `avreymi.asraf@gmail.com`, with the `colaboratory` scope present.
`colab --auth=oauth2 sessions` was tested and returned "No active sessions found on server"
after cleanup.

Put the global auth flag before the command:

```bash
colab --auth=oauth2 sessions
colab --auth=oauth2 new -s amlk-test --gpu T4
```

Known auth behavior on this machine:

- `--auth=oauth2` works with the cached token in `~/.config/colab-cli/token.json`.
- `--auth=adc` fails with `DefaultCredentialsError: Your default credentials were not found`.
- `~/.colab-cli-oauth-config.json` is missing, but the cached OAuth token is still enough for
  current CLI use.

Do not use `colab auth` to fix CLI login errors. `colab auth` injects credentials into the
remote VM for GCS/BigQuery code; it does not authenticate the local CLI to allocate runtimes.

## Common Workflows

### One-shot GPU job

Use this when no persistent kernel state is needed:

```bash
colab --auth=oauth2 run --gpu T4 --timeout 120 scripts/check_gpu.py
```

Arguments after the script are forwarded to Python:

```bash
colab --auth=oauth2 run --gpu T4 --timeout 600 train_probe.py --variant lead --limit 100
```

Use `--keep` only when you need to inspect the VM afterward, then stop it manually.

`colab run` checks that the local script exists before allocating. A typo like this exits
non-zero without spending compute:

```bash
colab --auth=oauth2 run /tmp/definitely-missing.py
```

CPU `colab run` was tested with a tiny local script. It created `run-d0566c`, printed the
script output, stopped the session, and `colab --auth=oauth2 sessions` was empty afterward.

### Durable session

Use this when loading a model once and iterating:

```bash
colab --auth=oauth2 new -s amlk-probe --gpu T4
colab --auth=oauth2 install -s amlk-probe -r requirements.txt
colab --auth=oauth2 exec -s amlk-probe -f scripts/setup_remote_probe.py --timeout 600
echo "print('state is still alive')" | colab --auth=oauth2 exec -s amlk-probe --timeout 30
colab --auth=oauth2 log -s amlk-probe -n 50
colab --auth=oauth2 stop -s amlk-probe
```

CPU durable session `amlk-cpu-smoke` was tested with `new`, `status`, `sessions`, stdin
`exec`, file `exec`, piped `repl`, piped `console`, `upload`, `ls`, `download`, `edit` with
`EDITOR=true`, `rm`, `install packaging`, `restart-kernel`, `url`, `log`, and `stop`.
`sessions` was empty after cleanup.

### File transfer

Transfers are source first, destination second:

```bash
colab --auth=oauth2 upload -s amlk-probe outputs/data/sample.jsonl /content/sample.jsonl
colab --auth=oauth2 download -s amlk-probe /content/result.jsonl outputs/results/result.jsonl
```

### Logs and recovery

```bash
colab --auth=oauth2 sessions
colab --auth=oauth2 status -s amlk-probe
colab --auth=oauth2 log -s amlk-probe -n 100
colab --auth=oauth2 log -s amlk-probe -o outputs/results/colab-session.md
colab --auth=oauth2 restart-kernel -s amlk-probe
```

If `exec` reports 401, 403, 404, or "Session not found", run `colab sessions`, inspect
`colab log`, refresh ADC scopes if needed, and recreate the session if the backend pruned it.

In 0.5.11, some missing-session paths are still rough:

- `exec`, `repl`, `console`, `edit`, `ls`, `upload`, `download`, `rm`, `url`, and `stop` report
  "Session not found" cleanly.
- `install`, `auth`, `drivemount`, and `restart-kernel` can raise
  `AttributeError: 'NoneType' object has no attribute 'url'` if the session name is not in
  local state. Check `colab status -s NAME` first.
- `repl` was tested only with piped stdin. Do not test unpiped `repl` or `console` from an
  agent; those are interactive TTY modes and can hang.
- A real CPU session log showed keep-alive failing with a 403 against
  `colab.pa.googleapis.com`, even though command execution worked. Do not leave manual
  sessions idle; finish work and `stop` them promptly. If long-lived sessions are required,
  investigate account/project permission for keep-alive before relying on them.

## AMLK Guidance

- Keep the default AMLK path on HuggingFace Jobs unless the task specifically benefits from
  Colab. HF Jobs is still the established training/evaluation pipeline in this repo.
- Never run Qwen3-2B training or inference on the local GPU. If not using HF Jobs, use
  Colab or another remote runtime.
- For unattended experiments, prefer `colab run` over a persistent session so failed jobs
  self-clean.
- Checkpoint or download artifacts frequently. VM storage is ephemeral and backend lifetime
  caps still apply even with keep-alive.
- Do not rely on `google.colab.userdata` for agent-run scripts. Use environment variables,
  uploaded config files, or explicit secret injection patterns instead.
- For AMLK societal-bias checks, treat gender bias as a prominent first probe but keep it
  small and cheap: start with CPU/API scripts or a tiny `colab run` smoke sample, inspect
  outputs manually, and only scale to GPU if the probe actually requires local model
  inference.

### Driving a notebook cell-by-cell (evaluation-observation)

`scripts/run_nb_cell.py` lets an agent run `notebooks/evaluation_observation.ipynb` one cell at
a time against a *persistent* session — the Colab CLI has no native `.ipynb` runner, so the
script reads the notebook locally with `nbformat` and `colab exec`s each cell's source. Kernel
state carries across calls, so cells build on each other. Validated 2026-06-16 on a CPU session:

```bash
colab --auth=oauth2 new -s amlk-eval-obs                 # CPU first (judge/error are API+CPU)
# Inject secrets WITHOUT leaking into colab log: upload the .env, the bootstrap reads /content/.env.
colab --auth=oauth2 upload -s amlk-eval-obs .env /content/.env
# To validate UNcommitted code, skip the notebook's git clone by pre-seeding the repo dir:
tar --exclude=.git --exclude=.venv --exclude=outputs -czf /tmp/repo.tgz .
colab --auth=oauth2 upload -s amlk-eval-obs /tmp/repo.tgz /content/repo.tgz
printf 'import os,subprocess;d="/content/AMLK-the-true-title";os.makedirs(d,exist_ok=True);subprocess.run(["tar","-xzf","/content/repo.tgz","-C",d],check=True)' \
  | colab --auth=oauth2 exec -s amlk-eval-obs --timeout 120
# run_nb_cell reads the LOCAL .ipynb and sends source remotely — edits don't need re-upload.
python -m scripts.run_nb_cell notebooks/evaluation_observation.ipynb --list
python -m scripts.run_nb_cell notebooks/evaluation_observation.ipynb --session amlk-eval-obs --cell 0 --timeout 1200   # bootstrap (pip install)
python -m scripts.run_nb_cell notebooks/evaluation_observation.ipynb --session amlk-eval-obs --range 1:6   # load..error-analysis (CPU)
colab --auth=oauth2 stop -s amlk-eval-obs
```

The last code cell (live generation) needs a separate **T4** session. `run_nb_cell` is local —
pass `--timeout` ≥1200 for the dep-install cell. Use `colab log` if a cell's stdout is truncated.

#### Gotcha: `google.generativeai` hangs on Colab (the runtime-proxy import hook)

On Colab, `import google.generativeai` is wrapped by `google.colab._import_hooks._GenerativeAIImportHook`
(the source of the `FutureWarning` about the deprecated package). The hook routes every
`generate_content` call through Colab's **runtime service proxy** at `localhost:45545`, which needs
the `serviceusage.services.use` permission on the runtime's GCP project. On this account that
permission is **missing** — it's the same 403 that kills keep-alive — so the proxied call **hangs
forever** (no error, kernel stuck `BUSY`) instead of reaching Google. Verified 2026-06-16: a raw
REST call to `generativelanguage.googleapis.com` and the SDK *with the hook removed* both answer in
~0.5s; the hooked SDK times out on `localhost:45545`. This is unrelated to `GEMINI_API_KEY` being
valid. Fix (done in the eval-obs notebook bootstrap) — drop the finder before the SDK is first
imported:

```python
import sys
sys.meta_path = [f for f in sys.meta_path if type(f).__name__ != "_GenerativeAIImportHook"]
for _m in [m for m in sys.modules if m.startswith("google.generativeai")]:
    del sys.modules[_m]
```

Any Colab notebook in this repo that calls Gemini (judge / error analysis / baseline) must do this
in its first cell. A per-request `request_options={"timeout": N}` (now `GEMINI_TIMEOUT` in
`evaluation/predict.py`) is a second line of defence: it converts a hung call into a fast failure
that `call_with_retry` can surface, but it does **not** make the proxied call succeed — only
removing the hook does.

## Low-Cost Validation Checklist

This checklist exercises most CLI surfaces without starting a VM:

```bash
colab version
colab help
colab help new
colab help run
colab help exec
colab help whoami
colab skill
colab readme
colab update
colab --auth=oauth2 --config /tmp/colab-test.json run /tmp/missing.py
printf "print('repl smoke')\n" | colab --auth=oauth2 --config /tmp/colab-test.json repl -s missing-session
colab --auth=oauth2 --config /tmp/colab-test.json edit -s missing-session /content/test.txt
colab --auth=oauth2 --config /tmp/colab-test.json whoami
colab --auth=oauth2 --config /tmp/colab-test.json sessions
```

Expected in this repo: `oauth2 whoami` succeeds, `oauth2 sessions` works, and missing-script
`run` exits before allocation. `adc whoami` fails with `DefaultCredentialsError`.

## Installation and Reference

```bash
uv tool install google-colab-cli
uv tool install -U google-colab-cli
colab update
colab skill
colab readme
```

Primary references: `https://github.com/googlecolab/google-colab-cli`,
`https://pypi.org/project/google-colab-cli/`,
`https://developers.googleblog.com/introducing-the-google-colab-cli/`, the bundled
`COLAB_SKILL.md` printed by `colab skill`, and local `colab help <command>`.
