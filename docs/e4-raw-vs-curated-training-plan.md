# E4 — Does curation actually train a better model? (plan + required changes)

**Status:** code implemented (2026-07-30). Decode defaults, judge pin, 35-word prompt,
`data/download_raw.py`, `preprocess --test-from/--output`, and `scripts/e4_score.py` are in the
tree. **Jobs not yet submitted** — rebuild both datasets with the new prompt, upload Hub repos,
train, then score.

**Why this exists.** The paper's E1–E3 show that curated headlines are *better targets* (rubric
scores, 73.6% pairwise preference on rewrites). They do not show that a model trained on them is
better. E4 closes that loop with the smallest experiment that can: **two SFT runs that differ only
in which corpus they train on**, judged on the same articles. `evaluation/pairwise_judge.py`
already anticipates this — its docstring says "future E4 arm comparisons".

The changes below fall into two groups: a **decode fix that is not optional** (it was measured to
be worth more than any training change this project has made), and a **prompt change that is
optional but recommended**. Everything else is held fixed on purpose.

---

## Part 1 — Repo changes (implemented checklist)

These came from the training-improvement loop on the `another-model` worktree. **All of Part 1
is applied in this tree as of 2026-07-30.** Remaining work is Part 2/3 operations: rebuild
datasets, Hub upload, train jobs, score — not further code ports.

| § | Item | Status |
|---|------|--------|
| 1.1 | Decode defaults `1.0` / `0` (`train_hf_job.py`, `infer.py`) | **done** |
| 1.1 | Decoding Configuration superseded banner | **done** |
| 1.2 | `JUDGE_GENERATION_CONFIG` + evaluate / pairwise pin | **done** |
| 1.3 | `PROMPT_TEMPLATE` 2 sent / 35 words | **done** |
| 3.1 | `data/download_raw.py` | **done** (verified: held_out=1171, sample=5854) |
| 3.2 | `preprocess --test-from` / `--output` | **done** |
| 3.3 | `scripts/e4_score.py` | **done** |
| 3 | Hub upload + train jobs + score | **open** |

### 1.1 Decode: turn the repetition penalties off (mandatory, blocking) — **DONE**

HF `transformers` applies `repetition_penalty` over the **whole sequence, prompt included**. With a
~3,800-token Hebrew article in context, `1.2` suppresses the article's own vocabulary at every
decode step, so the model emits near-miss tokens — misspelled entity names like `שיוני האקחים` for
`שינויי האקלים`. In summarization, copying the input is *correct*; penalizing it is a bug.

Measured on a fixed 120-article judged subset, turning the `1.2` / `no_repeat_ngram_size=3` pair
off:

| arm | Δ faithfulness | Δ fluency |
|---|---|---|
| zero-shot base | **+1.54** | **+0.78** |
| fine-tuned adapter | **+1.39** | **+0.78** |

Confirmed judge-free: the fraction of output words present in the source article went
**0.204 → 0.685** (paired Δ +0.481). Consequence for this repo: every Faith/Flu number produced
before this fix was measured through a broken decode, and the zero-shot base is a ~4.5-faithfulness
model, not the ~2.9 the old tables report.

| file | was | **now (applied)** |
|---|---|---|
| `training/train_hf_job.py` | defaults `1.2` / `3` | **`1.0` / `0`** |
| `evaluation/infer.py` | `3` / `1.2` | **`0` / `1.0`** |
| `training/train.py` comments/help | "defaults 1.2 / 3" | **"1.0 / 0"** |

Env-var plumbing (`--repetition-penalty` / `--no-repeat-ngram-size`) unchanged — empty env still
falls through to the new defaults. **`docs/obsidian/Decoding Configuration.md` has the superseded
banner** with the measured numbers above.

Degeneration was the real problem that note was reacting to; it is now handled by the settings that
remain: `min_new_tokens`, explicit `eos_token_id`, the `max_new_tokens=128` cap, and — the actual
fix — an instruction-formatted prompt with a stop cue, which the Qwen-era note predates.

### 1.2 Pin the judge temperature (mandatory, cheap) — **DONE**

An unpinned judge temperature makes re-scoring the same file drift, which shows up as fake deltas
between arms. Applied:

- `evaluation/gemini_client.py`: `JUDGE_GENERATION_CONFIG = {"temperature": 0.0}`
- `evaluation/evaluate.py`: passes it on both Gemini `generate_content` sites
- `evaluation/pairwise_judge.py`: `"temperature": 0.0` in its config
- `rubric_judge.score_headline` still accepts optional `temperature` — pass `0.0` from E4 call
  sites (E1 pilot deliberately used non-zero to measure variance)

**Do not** flip `evaluate.py`'s `--judge-provider` default from `hf` to `gemini` (B'.1 self-preference
guard). E4 passes Gemini explicitly via `scripts.e4_score`.

### 1.3 Prompt template — 35 words, not 15 (recommended) — **DONE (chosen)**

Arena round-3 was 1 sentence / ≤15 words (zero-shot tuned). Fine-tuned at 15 words compressed out
identifying specifics (faithfulness ~2.56). **Current** `data/prompts.py::PROMPT_TEMPLATE`:

```python
PROMPT_TEMPLATE = (
    "סכם את כתבת החדשות הבאה בעברית ב-2 משפטים קצרים, לא יותר מ-35 מילים. "
    "כלול את הפרטים המזהים המרכזיים: מי, מה והיכן. "
    "כתוב 2 משפטים בלבד ועצור מיד בסופם.\n\n"
    "{text}\n\nתקציר (עד 2 משפטים, עד 35 מילים):"
)
```

Baked into the `prompt` column at preprocess time — **rebuild both E4 datasets** after any change;
keep identical across arms.

### 1.4 What is *not* being ported

`data/distill.py`, `training/dpo_hf_job.py`, `evaluation/improve_eval.py`, the prompt-arena sweep
scripts. Those served a different question (can a better *target text* beat the base model — answer:
not at distillation quality). E4 needs none of them. Statistics come from
`data_curation/analysis/stats.py`, which this repo already uses for E1–E3.

---

## Part 2 — The two experiments

### The design in one line

Both arms train the same model, the same way, for the same number of examples, and are **judged on
the same 586 curated test articles**. The only thing that varies is which corpus supplies the
training pairs.

### E4-RAW — SFT on raw HeSum, as-is

- **Source:** `data_curation/artifacts/raw_hesum.json` — already on this worktree, 10,000 rows of
  `{id, text, headline}`. No Hub download, no `datasets.load_dataset` call needed.
- **Targets:** the original HeSum headlines, unfiltered and unrewritten. For scale: **25.9%** of raw
  headlines are pipe-separated multi-story digests (`"headline | headline | headline"`) versus
  **0.07%** after curation, and raw headlines average 28.7 words versus 21.7 curated.
- **Size:** 4,683 train / 585 val — matched to the curated split exactly.

### E4-CUR — SFT on curated HeSum

- **Source:** `data_curation/artifacts/final_clean_hesum.json` — 5,854 rows, the E1–E3 product.
- **Size:** 4,683 / 585 / 586 from the existing deterministic 80/10/10 split (`split_dataset`,
  seed 42). Verified locally: those are exactly the sizes it produces.

### What curation is, quantitatively

Both arms' provenance, so the reader knows what the manipulation actually contains. Verified
against the artifacts on this worktree:

| stage | rows |
|---|---|
| raw HeSum | 10,000 |
| after source filter input | 6,486 |
| final curated | 5,854 |

Of the 5,854 curated rows, **all 5,854 ids exist in raw**; **3,068 (52.4%) have a rewritten
headline**, 2,786 keep the original; **368 (6.3%) have modified article text** (tail-boilerplate
removal). So E4 measures the *combined* effect of source filtering + target rewriting + text
cleanup — not any one of them alone.

### The leakage control (do not skip this)

Curated is a strict **subset** of raw. If E4-RAW sampled 4,683 rows uniformly from the 10,000, then
about `4683/10000 × 1171 ≈ 548` of the held-out val+test articles would land in its *training* set
— it would have seen roughly 47% of the evaluation articles. That alone could produce the entire
result, in either direction.

**Fix:** build E4-RAW's pool as `raw_ids − curated_val_ids − curated_test_ids`, then sample.
Verified locally: 1,171 val+test ids, **pool = 8,829**, comfortably more than the 5,854 needed
(see §3.1 for why the sample is 5,854 and not 4,683 + 585).

Two properties of that pool, both checked locally so the loader can rely on them:

- **0** pool texts equal a held-out curated text under a different id — id-based exclusion is
  sufficient, no text-level exclusion needed on top.
- **14** exact-duplicate texts *within* the pool (raw HeSum was never deduplicated; curation removed
  these upstream). These must be dropped before splitting, or two copies of one article can land in
  different splits and `validate_train_dataset`'s overlap check will raise — after the sample, not
  before, so it fails late and confusingly. Dedup on exact `text`, keeping the first occurrence, and
  sample from the deduplicated pool.

Getting the excluded id list is slightly awkward because `data/preprocess.py::build_train_dataset`
**drops `hesum_id`** — saved columns are `text, summary, source, prompt, completion`. Text-matching
back is unreliable (the split's `text` is token-truncated, and 368 curated texts differ from their
raw twin anyway). Instead, reproduce the split deterministically on the ids: `train_test_split`
partitions by index from the seed, so a `Dataset` with the same row order and `seed=42` yields the
same partition regardless of which columns it carries.

```python
recs = [c for c in curated if c["text"].strip() and c["headline"].strip()]  # same order as data.download
ds = hf_datasets.Dataset.from_dict({"hesum_id": [r["hesum_id"] for r in recs]})
tr, va, te = split_dataset(ds, seed=42)          # 4683 / 585 / 586 — confirmed
held_out = set(va["hesum_id"]) | set(te["hesum_id"])   # 1171 — confirmed
```

Id-based exclusion is exact and also covers the 368 text-edited articles, since raw `id` and
curated `hesum_id` are the same key.

### The shared test split

E4-RAW's dataset carries the **curated test split, copied through untouched**. There is precedent
for exactly this pattern in the sibling branch's `data/distill.py` ("the `test` split is copied
through untouched so every arm is judged on the same articles").

This is what buys back a **paired** comparison. The judge scores a summary against the **article**,
not against the reference, so judging a raw-trained model on curated-test articles is legitimate.

**Caveat to state in the paper:** the references on that split are curated headlines, so
**ROUGE and BERTScore are confounded** — they reward matching curated *style*, which E4-CUR gets
for free. Report them for completeness, but they are not the outcome. The article-grounded judge is.

### Metrics, in priority order

1. **Faithfulness / fluency judge** (`evaluation/evaluate.py`, Gemini, temperature 0.0) on a fixed
   subset of the shared test split. 120 examples matches the sibling loop's instrument and was
   enough to resolve a ~1.5-point effect; use the full 586 only if the gap looks small.
   Statistic: paired per-article difference, plus `cliffs_delta` +
   `bootstrap_cliffs_delta_ci` from `data_curation/analysis/stats.py` — same statistics as E1.
2. **Blind pairwise A/B** (`evaluation/pairwise_judge.py::compare_headlines`) between the two arms'
   outputs for the same article, with side assignment randomized per row. Report the win rate with
   `wilson_ci`, exactly as E2/E3 do. This is the headline result and the most robust one: paired by
   construction, blind, and directly comparable to the paper's existing preference numbers.
3. **Rubric scores** (`evaluation/rubric_judge.py`, `temperature=0.0`) on both arms' outputs —
   optional, but it puts *model outputs* on the same axes as the *reference* scores in E1, which is
   a table the paper can use directly.
4. **ROUGE / BERTScore** — reported, flagged as style-confounded, not decisive.

**Decision rule:** curation wins only if the pairwise win-rate Wilson CI excludes 50% **or** the
paired judge CI excludes 0. A null result is a publishable finding here, not a failure — "better
targets by rubric, no measurable downstream gain at 4.7k examples / 1 epoch" is a real claim,
provided the decode fix is in (a broken decode would floor both arms and manufacture a null).

### Held identical across both arms

| | E4-RAW | E4-CUR |
|---|---|---|
| train | 4,683 | 4,683 |
| val | raw-sampled 585 (see below) | curated 585 |
| test (judged) | curated test, 586 — **same rows** | curated test, 586 — **same rows** |
| prompt template | `data/prompts.py::PROMPT_TEMPLATE` | same |
| variant | `whole` | same |
| article truncation | 3,840 tokens (`MAX_LENGTH − 256`) | same |
| base model | `dicta-il/dictalm2.0-instruct` | same |
| method / epochs | `lora`, 1 epoch, `METHOD_PRESETS` | same |
| decode | `repetition_penalty=1.0`, `no_repeat_ngram_size=0`, `max_new_tokens=128`, greedy, Hebrew constraint | same |
| judge | Gemini, `temperature=0.0` | same |

**Val split — a deliberate asymmetry.** Each arm validates on its *own* distribution. Since
`load_best_model_at_end` is on, the two arms select their checkpoint against different yardsticks,
which is a real (small) confound. The alternative — copying the curated val through as well, so both
arms are early-stopped on a common yardstick — is equally defensible and arguably cleaner given that
curated-test is what gets judged. **Recommendation: keep each arm's own val**, because eval loss on
curated references would systematically favour the curated arm for style reasons, reintroducing the
same confound §"shared test split" flags for ROUGE, but in the checkpoint-selection path where it is
invisible. At 1 epoch with ~293 optimizer steps the practical difference is small either way. Record
whichever is chosen in the run manifest.

---

## Part 3 — Resources required

### Code to write (data prep only)

1. **A raw-HeSum loader**, a sibling to `data/download.py` (do not repurpose that module — it is
   curated-only by design). It must:
   - read `data_curation/artifacts/raw_hesum.json`;
   - compute the held-out curated id set as shown above and exclude it;
   - drop the 14 exact-duplicate texts;
   - sample **5,854** rows — *not* 4,683 + 585 — from the remaining pool with a fixed seed,
     recorded in the output. `split_dataset` is unconditionally 80/10/10: hand it 5,268 records and
     it silently returns 4,214 / 527 / 527 and the matched-size premise dies without an error.
     Handing it 5,854 gives exactly 4,683 / 585 / 586, verified. The raw test 586 is then discarded
     and replaced by the curated one, which costs nothing;
   - normalize to the same record shape `data/preprocess.py` already consumes —
     `{text, summary, source: "hesum-raw", hesum_id}` — written as a
     `curated_records.jsonl`-shaped file so `python -m data.preprocess --input <file>` works
     unmodified.
2. **A test-split swap** — a `--test-from <dir>` flag on `data.preprocess` (cleaner and testable
   than a post-hoc directory edit). It substitutes the curated `test` split for the freshly built
   one and **re-runs `validate_train_dataset(train, val, curated_test)` afterwards**. This second
   validation is not optional: the built-in call runs *before* any swap, so the one check that
   would catch raw-train ↔ curated-test overlap is otherwise bypassed on the exact arm that needs
   it. It compares `set(ds["text"])`, so it also catches same-article-different-id collisions that
   id exclusion structurally cannot.
3. **An E4 scoring script** under `scripts/`, matching how E1–E3 are driven: run the judge on both
   prediction files, call `pairwise_judge` on the paired rows with randomized sides, and print
   Cliff's δ + Wilson CI via `data_curation/analysis/stats.py`.

No changes needed to `data/preprocess.py` internals, `training/train.py`, `training/train_hf_job.py`
(beyond the two default lines in 1.1), or the Hebrew decode constraint. The improvement-loop flags
E4 needs — `--dataset-repo`, `--max-train`, `--test-subset`, `--skip-base-arm`, `--run-tag` — are
**already on this branch** (`training/train.py:492–508`), merged with the 2026-07-28 training-stack
merge.

Tests worth adding, in the style of the existing ones: the id-exclusion set is disjoint from the raw
sample, the raw sample is text-deduplicated, the two arms' test splits are byte-identical, the
post-swap `validate_train_dataset` actually runs, and the sampler is reproducible under its seed.

### Hub repos (all new — nothing existing is overwritten)

| repo | kind | contents |
|---|---|---|
| `avreymi/amlk-training-data-raw` | dataset | E4-RAW splits, 4683/585 raw + 586 curated test |
| `avreymi/amlk-training-data-e4cur` | dataset | curated splits rebuilt with the final prompt |
| `avreymi/amlk-e4-raw` | model | E4-RAW adapter + predictions |
| `avreymi/amlk-e4-curated` | model | E4-CUR adapter + predictions |

A rebuild is needed for the curated arm too if §1.3 is taken: the prompt is baked at preprocess
time, so anything already on the Hub carries the old 15-word prompt. Push it to the **new** repo
above; `train.py`'s existing guard (refuses `--dataset-repo` without `--skip-data-upload`) already
blocks an accidental overwrite of `avreymi/amlk-training-data`.

### HF Jobs

```bash
source .env && source .venv/bin/activate

# local, CPU only — build both datasets AFTER the prompt decision is final
python -m data.download_raw --force            # new sibling script (§3.1)
python -m data.preprocess --input outputs/data/raw/raw_records.jsonl --variant whole --force
python -m data.preprocess --variant whole --force        # curated, current PROMPT_TEMPLATE

# E4-RAW
python -m training.train --submit-hf --hf-user avreymi --method lora \
    --dataset-repo avreymi/amlk-training-data-raw \
    --output-repo avreymi/amlk-e4-raw \
    --test-subset 120 --skip-base-arm --run-tag e4-raw

# E4-CUR
python -m training.train --submit-hf --hf-user avreymi --method lora \
    --dataset-repo avreymi/amlk-training-data-e4cur \
    --output-repo avreymi/amlk-e4-curated \
    --test-subset 120 --skip-base-arm --run-tag e4-cur
```

`--test-subset 120` caps dual-arm generation to the judged rows rather than all 586 — generation was
40–45% of job wall time on full-test runs, so this is the main cost lever. `--skip-base-arm` drops
the zero-shot arm; run it **once** separately if the paper wants a base row, since it is identical
for both arms by construction.

The 120 rows match across arms **by construction**, no post-hoc check needed:
`train_hf_job.py:322–325` selects them with `sorted(random.Random(TEST_SUBSET_SEED).sample(...))`
at a fixed default seed of 1234, over a test split that is byte-identical in both arms. Same seed,
same length, same sorted indices. This only holds because the test split is shared — if that swap is
ever dropped, the pairwise join (which matches on article text) would silently lose rows instead of
failing.

### Cost

Based on comparable a10g-small runs of this exact model/method/epoch count (~1.5–1.75 h each with
full-test generation):

| item | estimate |
|---|---|
| E4-RAW train + 120-row generation | ~$1.20–1.40 |
| E4-CUR train + 120-row generation | ~$1.20–1.40 |
| Gemini judging (120 × 2 arms, plus pairwise) | API only, negligible |
| **total** | **~$2.40–2.80** |

Budget one retry (~$0.45–0.75 via an `--inference-only` rerun from the pushed adapter) — long jobs
on this account have been killed at the infra level mid-run before, and the adapter survives because
`hub_strategy=every_save` commits it.

### Out of scope on purpose

No distillation targets, no DPO, no new prompt candidates beyond the single decision in §1.3, no
new decode parameters, no third arm. In particular the **id-matched variant** — same 5,854 articles,
raw headlines vs curated headlines, isolating target rewriting from source filtering — is the
obvious follow-up and the artifacts support it (3,068 rewritten pairs), but it answers a different
question and should wait for E4's result.

---

## Order of work

1. ~~Branch off `main`.~~
2. ~~Apply §1.1 + §1.2 (decode + judge pin); update Decoding Configuration.~~ **done**
3. ~~Decide §1.3, write raw loader + test-split swap + scorer + tests.~~ **done**
4. **Build both datasets** with current `PROMPT_TEMPLATE` (`e4cur` then `e4raw --test-from`).
5. Upload Hub repos; submit E4-RAW, confirm it completes, then E4-CUR.
6. Score (`scripts.e4_score`), write into `docs/obsidian/Experiment Results.md` and the paper.
