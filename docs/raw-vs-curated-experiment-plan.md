# Raw HeSum vs. Curated HeSum — a controlled 2-arm plan

**Purpose:** the training-improvement loop (`docs/training-improvement-notebook.md`) established that
the biggest cost to faithfulness/fluency was a decode bug, now fixed everywhere
(`repetition_penalty=1.0`, `no_repeat_ngram_size=0`). This is a **separate, narrower question**:
with that fix in place, and with everything else held constant, does training on the **curated**
HeSum product beat training on **raw** `biunlp/HeSum`? Only two arms, no distillation, no DPO, no
prompt changes — a clean data-source comparison.

**Rule for this plan:** the *only* code change either arm depends on is the decode fix already
merged (see below). No new decode parameters, no new prompt template, no new training method.

---

## The two experiments

### Experiment 1 — SFT on raw `biunlp/HeSum`

- **Source:** `datasets.load_dataset("biunlp/HeSum")` — a public Hub dataset, no local file needed.
  Confirmed live: `train` 8000 rows, `validation` 1000, `test` 1000; columns `article`, `summary`.
  Unlike the curated product, these are **not** deduplicated against roundup/pipe-digest style and
  are not the same underlying article set as the curated product (no shared `hesum_id`) — this is
  a genuinely different, unfiltered corpus, which is the point of the comparison.
- **Size match:** subsample `train` to **4683** rows (same count as `avreymi/amlk-training-data`'s
  train split) with a fixed seed so the run is reproducible. Val/test subsampled to **585/586** for
  the same reason — matching split sizes end to end, not just train.
- **Preprocessing:** identical pipeline to the curated path — `data.prompts.build_prompt` (the
  hardened, stop-cue prompt already promoted from the prompt-arena loop), `variant="whole"`,
  `ARTICLE_TOKEN_BUDGET = MAX_LENGTH - 256 = 3840` token truncation, same `TRAIN_COLUMNS` contract,
  same `validate_train_dataset` gate. The only thing that changes is which `{text, summary}` pairs
  go in.
- **New Hub dataset repo (non-destructive):** `avreymi/amlk-training-data-raw` — a new repo, does
  not touch `avreymi/amlk-training-data`.

### Experiment 2 — SFT on curated HeSum

- **Source:** the same 5854 curated rows already used to build `avreymi/amlk-training-data`
  (`outputs/data/curated/final_clean_hesum.json` / `curated_records.jsonl`).
- **Update since the prompt change:** `data/prompts.py::PROMPT_TEMPLATE` now uses the 35-word /
  2-sentence template found in the training-improvement loop (entry #8, A4) instead of the
  original 15-word one. The `prompt` column is baked in at `data.preprocess` build time and read
  verbatim by `train_hf_job.py` — it is **not** rebuilt from `build_prompt` at train time — so the
  already-uploaded `avreymi/amlk-training-data` still carries the *old* 15-word prompt and no
  longer reflects the current `PROMPT_TEMPLATE`. This arm therefore needs a **local rebuild**
  (`python -m data.preprocess --variant whole --force`, same 4683/585/586 split/seed) and push to
  a **new** repo — reusing the old repo name would silently retrain on stale prompt text, and
  overwriting it would violate the non-overwrite constraint anyway.
- Variant, truncation, and everything else in `data.preprocess` stay as-is — only the `prompt`
  (and downstream `completion`, unchanged) column content differs from the old upload.

### What is held identical across both arms

| | Experiment 1 (raw) | Experiment 2 (curated) |
|---|---|---|
| train / val / test size | 4683 / 585 / 586 | 4683 / 585 / 586 |
| prompt template | `data/prompts.py::PROMPT_TEMPLATE` — 35-word/2-sentence stop-cue winner (entry #8) | same |
| variant | `whole` | same |
| article truncation | 3840 tokens | same |
| base model | `dicta-il/dictalm2.0-instruct` | same |
| method | `lora` (bf16, current default — see `METHOD_PRESETS`) | same |
| epochs | 1 | same |
| decode at generation | `repetition_penalty=1.0`, `no_repeat_ngram_size=0` (already the default in `train_hf_job.py`, `evaluation/infer.py`, `predict_base_hf_job.py`, `prompt_sweep_hf_job.py`) | same |
| judge | Gemini, `temperature=0.0` (`JUDGE_GENERATION_CONFIG`) | same |

The **only** deliberate difference is the source article/summary pairs. This isolates curation as
a variable the way B1 isolated target-text quality in the earlier loop.

---

## Why the earlier "fixed 120-article" instrument does not carry over unchanged

`evaluation/improve_eval.py`'s paired design (`subset_indices(total, n=120, seed=1234)`) assumes
both arms are scored on **the same articles** — that's what makes a *paired* delta valid and
sensitive. Raw HeSum's test split and curated HeSum's test split are different articles (no shared
`hesum_id`), so a paired-by-article comparison across arms is not available here.

**What this plan does instead:** each arm gets its own fixed 120-example subset drawn from *its
own* test split (same seed=1234, same `subset_indices` function, applied to each dataset's own
586-row test set) — so each arm's number is still a stable, reproducible measurement, pinned
exactly like every earlier entry in the notebook. The two resulting means are then compared as an
**independent-groups** difference (not paired): report both means with a 95% CI on each (or a
CI on the unpaired difference, `pooled SE = sqrt(SE1² + SE2²)`), and state plainly in the writeup
that this is a between-corpus comparison, not a within-article one. This is a weaker statistical
design than the paired subset (more examples would be needed to detect the same effect size), but
it is the honest design given that the two corpora don't share articles.

---

## Resources required

### Code changes (data prep only — no decode/training-loop changes)

1. **A raw-HeSum loader**, parallel to `data/download.py`'s curated path (that module is
   curated-only by design per the 2026-07-26 status note — do not repurpose it, add a sibling):
   - Load `biunlp/HeSum` via `datasets.load_dataset`, normalize `{article, summary}` →
     `{text, summary, source="hesum-raw", hesum_id=""}` (mirrors `normalize_curated_record`'s
     output shape so it drops straight into `data.preprocess`'s existing `build_train_dataset`).
   - Subsample train/val/test to 4683/585/586 with a fixed seed (e.g. `random.Random(42).sample`),
     writing a `curated_records.jsonl`-shaped file so `data.preprocess --input <file>` "just works"
     unmodified — reuse, not reimplementation, of the whole preprocessing/validation path.
2. **No changes to `data/preprocess.py`, `training/train.py`, `training/train_hf_job.py`,
   `evaluation/infer.py`, or the decode paths.** They already default to the fixed decode and
   already support everything this plan needs (`--dataset-repo`, `--skip-data-upload`,
   `--test-subset`, `--skip-base-arm`, `--run-tag`). `data/prompts.py::PROMPT_TEMPLATE` is already
   updated to the 35-word/2-sentence version — both arms pick it up automatically the next time
   `data.preprocess` runs, since it's read at preprocess build time.
3. **A small eval script or a few lines of `improve_eval.py` reuse** to run `subset_indices` +
   `judge_file` once per arm's own test predictions and print the two means with CIs
   side by side — no change to `improve_eval.py` itself, just a second call site (or a thin CLI
   flag) since its current `arm` subcommand assumes the shared curated-subset framing.

### HuggingFace Hub resources (all new, non-destructive)

| repo | kind | purpose |
|---|---|---|
| `avreymi/amlk-training-data-raw` | dataset | raw-HeSum splits, 4683/585/586, 35-word prompt |
| `avreymi/amlk-training-data-w35` | dataset | curated splits rebuilt with the 35-word prompt, 4683/585/586 |
| `avreymi/amlk-rawcompare-raw` | model | Experiment 1 adapter + predictions |
| `avreymi/amlk-rawcompare-curated` | model | Experiment 2 adapter + predictions |

`avreymi/amlk-training-data` (the original, 15-word-prompt upload) and every existing model repo
are untouched. `train.py`'s existing guard (refuses `--dataset-repo` without `--skip-data-upload`)
already prevents an accidental overwrite of any curated Hub dataset.

### HF Jobs (a10g-small, ~$1/h) — command sketch

```bash
# one-time, local, CPU/API only: rebuild curated splits with the new 35-word prompt
python -m data.preprocess --variant whole --force          # picks up new PROMPT_TEMPLATE
# push outputs/data/processed/whole/ to a NEW repo (not amlk-training-data) — done by
# training.train --submit-hf below via --dataset-repo, or manually beforehand

# one-time, local, CPU/API only: build + upload the raw-HeSum splits
python -m data.download_raw --force                       # new sibling script (see above)
python -m data.preprocess --input outputs/data/raw/curated_records_raw.jsonl \
    --variant whole --force

# Experiment 1 — raw
python -m training.train --submit-hf --hf-user avreymi --method lora \
    --dataset-repo avreymi/amlk-training-data-raw \
    --output-repo avreymi/amlk-rawcompare-raw \
    --test-subset 120 --run-tag raw-vs-curated

# Experiment 2 — curated, rebuilt with the 35-word prompt
python -m training.train --submit-hf --hf-user avreymi --method lora \
    --dataset-repo avreymi/amlk-training-data-w35 \
    --output-repo avreymi/amlk-rawcompare-curated \
    --test-subset 120 --run-tag raw-vs-curated
```

`--test-subset 120` caps dual-arm generation to the judged 120 rather than the full 586, which is
the main lever on job cost (generation was ~40–45% of total job time in earlier full-test runs).

### Cost estimate

Based on directly comparable prior jobs on this exact model/method/epoch count (A4/A6, ~1.5–1.75h
each including dual-arm generation on a full 586-row test set):

| item | basis | estimate |
|---|---|---|
| Experiment 1 training + gen (120-row test subset, not 586) | ~1h training + ~15–20 min gen at batch 8 (bf16 lora) | ~$1.20–1.40 |
| Experiment 2 training + gen (same) | same | ~$1.20–1.40 |
| Gemini judge calls, 120 × 2 arms | prior judged-subset runs cost effectively $0 of the HF budget (API only) | ~$0.00 (HF budget) |
| **total** | | **~$2.40–2.80 of the ~$4.9 remaining** |

This leaves roughly $2–2.5 of headroom for one retry if a job ERRORs mid-run (the recurring
infra-kill pattern seen in every earlier long job) — recoverable cheaply via inference-only reruns
from the pushed adapter, as in every prior recovery in this project (~$0.45–0.75 each).

### What "resources" excludes on purpose

No new prompt candidates, no distillation targets, no DPO pairs, no additional decode parameter —
per the instruction to change only what's needed to fix decode (already done) and hold everything
else fixed. If Experiment 1 vs 2 shows a real gap, *that* result is what should motivate any further
arm — not this plan.

---

## Reporting format

Once both jobs complete, report in the same shape as every other entry in
`docs/training-improvement-notebook.md`:

| arm | n (judged subset) | faithfulness | fluency | note |
|---|---|---|---|---|
| Experiment 1 — raw HeSum SFT | 120 (raw test) | — | — | own subset, own test split |
| Experiment 2 — curated HeSum SFT | 120 (curated test) | — | — | own subset, own test split |
| difference (unpaired) | — | Δ [95% CI] | Δ [95% CI] | pooled-SE CI, see note above |

Decision rule, unchanged from the rest of the loop: a source is "better" only if the unpaired CI on
the difference excludes zero.
