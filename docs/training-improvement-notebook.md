# Training-Improvement Notebook

Lab notebook for the "fine-tuning makes the model *worse*" problem. Append-only: one entry per
experiment, never rewrite a past one. Sibling of `docs/prompt-arena-notebook.md`.

**Success axis (fixed by the user): Gemini-judge faithfulness and fluency only.** ROUGE and
BERTScore are recorded when free but never used to pick a winner — the references are
headline/digest style, so ROUGE rewards exactly the register that costs faithfulness.

**Budget:** $16 of HF Jobs. Running tally at the bottom.

**Hard constraints**
- Never overwrite `avreymi/amlk-dictalm2-instruct-sft` or `avreymi/amlk-training-data`
  → every submission passes an explicit `--output-repo` / new dataset repo.
- Nothing runs on the local GPU. Training/generation = HF Jobs; judge/scoring = local API+CPU.
- Short runs first; a change is only believed if the paired delta clears the noise bar (entry #0).

---

## The problem, stated precisely

`evaluate.py::JUDGE_PROMPT` scores the prediction **against the article**, not against the
reference. Training targets are curated HeSum headlines (median ~157 chars, ~26% pipe digests)
whose register asserts more than the article states. So SFT toward those targets moves the model
*away* from the metric being measured. The observed step-200 result is consistent with that
(entry #0 below reproduces it): faithfulness 2.93 → 2.24, fluency 4.20 → 3.46.

Working hypothesis: **the targets, not the hyperparameters, are the problem.** Primary experiment
is therefore to change what the model is trained to imitate, not how fast it is trained.

---

## Entry #0 — Instrument: fixed subset, pinned judge, noise floor (2026-07-28, $0)

**Hypothesis:** before spending GPU budget, confirm the judge can resolve the effect sizes we hope
to see.

**What was built** — `evaluation/improve_eval.py`:
- One fixed eval subset for every arm forever: `subset_indices` = `random.Random(1234).sample`,
  **n = 120** of the 586-row test split. All arms are scored on the same articles, so deltas are
  **paired**.
- `judge_file` → per-example faithfulness/fluency; `paired_delta` → mean(arm − control) with SE
  and 95% CI over the paired examples. That CI is the decision rule.
- **Judge temperature pinned to 0** (`gemini_client.JUDGE_GENERATION_CONFIG`). It was previously
  unpinned, i.e. re-scoring the same file could drift and manufacture fake deltas.
- Bug found and fixed while wiring this: `str.splitlines()` on a predictions file splits rows in
  half, because Hebrew article text contains U+2028/U+0085. Now splitting on `"\n"` only.

**Measurements**

| measurement | faithfulness | fluency |
|---|---|---|
| judge-repeat noise (same base file judged twice) | **0.000** | **0.000** |
| per-example sd (base arm) | 1.228 | 0.856 |
| SE of a paired mean at n=120 | ≈0.11 | ≈0.08 |
| zero-shot base (control) | 2.933 | 4.200 |
| LoRA step-200 on HeSum refs (paired delta vs base) | **−0.692** [−0.99, −0.40] | **−0.742** [−0.99, −0.50] |

**Verdict / decision rule adopted:** with temperature 0 the judge is *deterministic* — repeat
noise is exactly zero, so all uncertainty is example-sampling variance, already captured by the
paired SE. **A run counts as a real improvement only if its paired delta vs the zero-shot base has
a 95% CI excluding 0** (≈ |delta| > 0.22 faith / 0.16 flu at n=120). The step-200 regression is
real by that rule and is reproduced on the subset (means match the full 586-row report), so the
instrument is validated against a known-sign effect.

---

## Entry #1 — Arm A: Gemini-teacher distillation (2026-07-28)

**Hypothesis.** If the *targets* are what break faithfulness, then swapping HeSum headlines for
teacher summaries of the **same articles under the same prompt** should move the paired delta from
−0.69/−0.74 (entry #0) to ≥ 0. The teacher scores 4.90/4.99 on this judge, so the target ceiling
is high; the question is how much of it a 1.9k-example LoRA can reach.

**Dataset** — new repo `avreymi/amlk-distill-data` (the original `amlk-training-data` is untouched):
- Built by `data/distill.py` from the *same* curated articles; the teacher is fed the row's
  existing `prompt` column verbatim, i.e. the exact prompt the student sees at inference.
- Filter: `prompt_arena.compliance_metrics` (6–45 words, 1–2 sentences, ≥80% Hebrew, no list
  markers) **and** no Latin/Cyrillic/Greek/Arabic/CJK — the decode constraint bans those scripts,
  so a target containing them is unreachable by the student at generation time.
- Kept **1941/2000** train, **112/120** val (drops are mostly teacher API refusals). `test` is
  copied through byte-identical, so the judged subset is the same 120 articles as every other arm.
- Teacher temperature pinned to 0, so a rebuild reproduces the same targets.

**Run plumbing added** (so an arm costs minutes, not hours): `train.py --dataset-repo / --max-train
/ --test-subset / --skip-base-arm / --run-tag / --learning-rate`. Generation was the dominant cost
before — 586×2 arms at `GEN_BATCH_SIZE=1` took 2h38m. Generating only the 120 judged examples for
the adapter arm alone is ~1/10 of that. The base arm is not regenerated: decoding is greedy and
already scored in entry #0, so it is a valid paired control.

Smoke first (job `6a68e74b`, `--smoke-test`, ~$0.15) to gate the new code paths, then the real arm.

## Entry #2 — The gold targets are NOT the problem (2026-07-28, $0). Hypothesis revised.

Cheap check that should have come first: score the **HeSum reference summaries themselves** as if
they were system outputs, on the same 120 articles, same pinned judge.

| "system" on the judged subset | faithfulness | fluency |
|---|---|---|
| HeSum reference (the SFT target) | **4.233** | **4.983** |
| Gemini teacher (re-judged on this exact subset) | 4.908 | 4.992 |
| zero-shot base | 2.933 | 4.200 |
| LoRA step-200 trained toward those references | 2.242 | 3.458 |

**This contradicts entry #1's premise** (mine and the advisor's): the targets score *far above* the
base model on the very axis being measured. "Train toward the references and you necessarily lose
faithfulness" is false — a model that reproduced the references perfectly would score 4.23/4.98.

Revised hypothesis: the targets are **unreachable**, not bad. The references are terse
headline-register assertions (median 19 words) that name specific entities and numbers; coverage of
reference tokens by the article is only median **0.70** (p25 0.59), so a large minority of
reference content is not literally derivable from the input. SFT teaches the model the *register*
— confident, specific, terse — long before it can supply the right specifics, and a confident wrong
specific is exactly what the judge punishes (the step-200 arm's entity/number error rate was 0.60
vs 0.40 zero-shot). That is a style-before-substance failure, not a bad-label failure.

Two consequences for the arm queue:
- **A1 (distillation) is still the right first arm**, but for a different reason than entry #1
  gave. Teacher targets are generated *from the article the student sees*, so they are reachable:
  every specific in them is present in the input. The ceiling is the teacher's 4.90, not a rescue
  from bad labels.
- A new arm is now justified on its own evidence — **A3: grounding-filtered HeSum**. Keep the
  original references but drop rows whose summary tokens are poorly covered by the article
  (≥0.7 retains 51% of train, ≥0.6 retains 74%). If unreachability is the mechanism, filtering
  should recover most of the regression without any teacher at all — and it is the arm that keeps
  the project's own data, which matters for the paper.

## Entry #3 — A3 (grounding filter) demoted: coverage barely predicts label quality (2026-07-28, $0)

Before funding A3 (train on high-coverage references only), test what its filter actually selects.
Correlate per-reference article-coverage against the judge's faithfulness *of that reference*, on
the same 120 rows, with and without `evaluate.normalize_hebrew` (to rule out morphology inflating
the "uncovered" tail):

| | median coverage | Pearson r (coverage vs ref faithfulness) | faith: low-cov half | high-cov half |
|---|---|---|---|---|
| raw tokens | 0.692 | **0.148** | 4.11 (n=57) | 4.35 (n=63) |
| Hebrew-normalized | 0.692 | 0.133 | 4.11 | 4.35 |

Hebrew normalization moves nothing, so the uncovered tail is not a morphology artifact. But the
relationship with label quality is weak (r≈0.14; ~0.24 of a judge point between halves), for the
simple reason entry #2 already established: **the references are faithful almost regardless of
coverage** (mean 4.23) — a human wrote them correctly from the whole article, and low token overlap
mostly means paraphrase, not invention. So a coverage filter would buy ≈0.2 points of label quality
while discarding half the data. **A3 is demoted, not funded.**

What this does *not* settle: whether low-coverage examples teach the *student* to assert specifics
it cannot derive. That is a claim about the gradient, not about the label, and this correlation
cannot see it. It stays an open question, cheaper to answer as a by-product of A1 than by a
dedicated arm — the teacher targets are, by construction, fully derivable from the input.

## Entry #4 — New suspect found by *reading the outputs*: the decode penalty (2026-07-28, $0)

Looking at the step-200 predictions next to the base ones, the fine-tuned failures are not vague
summaries — they are **character-level corruptions of words that appear in the article**:
`פלפליפ צהוביפ` for `פלפלים צהובים`, `ביקופ לתוכניפ`, `הרייטנגיפ`, `ידיעת אחרונת`. Final mem (ם)
turning into פ is not a summarization error; it is a token-selection error.

Mechanism: HF's `RepetitionPenaltyLogitsProcessor` applies the penalty over the **entire
sequence, prompt included**. Generation runs at `repetition_penalty=1.2` with a ~3.8k-token article
in the context, so essentially every content word in the article is pushed down at every step — in
*summarization* that is a penalty on copying the correct entity, name or number. The model settles
on a near-miss token instead. This predicts exactly what the error analysis found: entity/number
error rate 0.60 for the fine-tuned arm vs 0.40 zero-shot, and it predicts the fine-tuned arm suffers
more (it wants to copy specifics in a terse register; the base arm paraphrases at length).

If this is right, **part of the "training makes it worse" result is a decode bug, not training** —
and it is fixable without retraining. `repetition_penalty` and `no_repeat_ngram_size` are now env-
overridable in `train_hf_job.py` (`train.py --repetition-penalty / --no-repeat-ngram-size`). The
test is an inference-only job on the A1 adapter at `repetition_penalty=1.0`, generating **both**
arms (adapter + `disable_adapter()` base) on the judged subset: ~240 generations, ~$0.6, and it
yields two paired comparisons at once — A1-with-penalty vs A1-without, and base-with vs base-
without. Queued as **E1**, to run once A1 lands (it needs A1's adapter, and it must not write into
the protected `amlk-dictalm2-instruct-sft` repo).

## Entry #5 — A1 RESULT: distillation halves the regression but does not clear zero (2026-07-28)

Job `6a68ea3915e81eca66a8d3f2`, COMPLETED 1h29m (~$1.50). LoRA, 1941 distilled targets, 121 steps,
loss 0.78 → 0.62. Judged on the fixed 120, paired against the zero-shot base:

| arm | faithfulness | Δ vs base [95% CI] | fluency | Δ vs base [95% CI] |
|---|---|---|---|---|
| base (control) | 2.933 | — | 4.200 | — |
| **A1 distilled** | **2.558** | **−0.375 [−0.71, −0.05]** | **4.175** | −0.025 [−0.24, 0.19] |
| step-200 on HeSum refs (entry #0) | 2.242 | −0.692 [−0.99, −0.40] | 3.458 | −0.742 [−0.99, −0.50] |

**Verdict: A1 fails its own bar** (CI excludes 0 on the wrong side) but it is clearly *better* than
training on HeSum references — it recovers the entire fluency loss (−0.03 vs −0.74, i.e. fluency is
now statistically indistinguishable from the base) and roughly halves the faithfulness loss. So the
target style *does* matter, just not enough to make SFT a net win on its own.

**What the outputs show.** A1 learned the teacher's format perfectly: mean **10.6 words** vs the
base's 30.9, one sentence, no digests. Faithfulness is strongly bimodal — 45/120 score 1, but
20/120 score 5. The 5s are clean (`"פלפל צהוב" הוכיחה שאיכות טלוויזיונית יכולה להביא רייטניג`).
Two distinct failure modes in the 1s:
1. **Over-compression drops the identifying specifics** — `קורא אלקטרונית חדשה מוצעת למכירה` where
   the article is about a specific publisher's device at a specific price. Generic-but-not-wrong
   reads as unsupported to the judge.
2. **Character-level corruption of Hebrew words**: `שיוני האקחים` for `שינויי האקלים`, `פטירתיות`
   for `פטריוטיות`, `מקצו` cut mid-word.

Failure mode 2 is the important one, because **the base arm has it too** (`ניופאין`, `התחחם`,
`מקצווים`, `הואר`) — so it is not something training introduced. It is the decode path, shared by
every arm ever measured in this project, and it is exactly what entry #4 predicted. That makes E1
the next experiment, not a scale-up of A1.

E1 submitted: job `6a68ffc4a9f4e0ab00b2bc85`, inference-only on the A1 adapter,
`repetition_penalty=1.0`, `no_repeat_ngram_size=0`, both arms, same 120 articles, preds suffixed
`-nopen` so nothing is overwritten (~$0.75).

## Entry #6 — B1 control: the step-200 regression was NOT under-training (2026-07-28)

Job `6a68eae1a9f4e0ab00b2ba5d`: trained a full matched epoch on HeSum references (1941 examples,
122 steps, eval_loss 0.765, `step_time_median 35.77s`, `peak_mem 17.44 GB`) — then **ERRORed during
generation at 70/120**, after pushing the first 50 rows. bf16 base (14.47 GiB resident) plus a 4k
prefill on a 22 GB a10g is the likely cause; A1's identical config survived, so this is marginal
headroom, not a code fault. Regeneration submitted as an inference-only 4-bit job
(`6a69018815e81eca66a8d526`, suffix `-r2`) rather than resuming, so all 120 rows share one decode
precision.

Final numbers on the full 120 (`predictions-finetuned-r2.jsonl`, job `6a69018815e81eca66a8d526`,
31m43s; B1 = 2.167 faith / 3.558 flu). All four systems here decode at penalty 1.2, so the
comparison is internally consistent:

| comparison (n=120, paired) | faithfulness Δ [95% CI] | fluency Δ [95% CI] |
|---|---|---|
| B1 (HeSum refs, full epoch) vs base | **−0.767 [−1.02, −0.52]** | **−0.642 [−0.87, −0.42]** |
| **A1 (distilled) vs B1 — same recipe, same steps, only the targets differ** | **+0.392 [+0.09, +0.70]** | **+0.617 [+0.37, +0.87]** |

**A1 beats B1 on both axes with CIs excluding zero.** That is the clean answer to "does changing
what we train on improve the model": yes — swapping HeSum headline targets for teacher summaries of
the same articles, at identical step count, is worth ~+0.4 faithfulness and ~+0.6 fluency. It is
also the honest limit of the claim: it improves on *the project's existing training recipe*, and
(entry #7) both still sit below a correctly-decoded zero-shot base.

**This settles the confound entry #0 flagged.** The step-200 arm was a SIGTERM'd mid-run
checkpoint, so its −0.69 could have been under-training. A *completed*, step-matched epoch on the
same references lands at −0.92 — no better, if anything worse. Training longer on HeSum references
does not recover faithfulness; the reference target is genuinely the wrong thing to imitate for
this metric, even though the references themselves score 4.23 (entry #2).

Against that control, distillation is a real gain on fluency (+0.56, CI excludes 0) and directionally
positive on faithfulness (+0.22, not resolved at n=50 — the full 120 will settle it).

**Confound to keep in view:** A1's predictions were generated in bf16 (the `lora` preset does not
quantize), while the base control and the `-r2` regeneration are 4-bit. Greedy decoding can differ
slightly between precisions. E1 generates *both* arms in one 4-bit job, so its internal comparison
is free of this; A1-vs-base carries it.

## Entry #7 — E1: THE DECODE PENALTY WAS THE MAIN PROBLEM (2026-07-28) ⚑

Job `6a68ffc4a9f4e0ab00b2bc85`, inference-only on the A1 adapter, 22m41s (~$0.38). Identical
adapter, identical 120 articles, identical greedy decode — the *only* change is
`repetition_penalty 1.2 → 1.0` and `no_repeat_ngram_size 3 → 0`. Both arms regenerated in one job.

| system | faithfulness | fluency |
|---|---|---|
| base, penalty 1.2 (the control every earlier entry used) | 2.933 | 4.200 |
| **base, penalty 1.0** | **4.475** | **4.975** |
| A1 distilled, penalty 1.2 | 2.558 | 4.175 |
| **A1 distilled, penalty 1.0** | **3.950** | **4.950** |
| Gemini teacher (ceiling) | 4.908 | 4.992 |

Paired deltas, same 120 articles:

| comparison | faithfulness Δ [95% CI] | fluency Δ [95% CI] |
|---|---|---|
| base: penalty 1.0 vs 1.2 | **+1.542 [1.29, 1.79]** | **+0.775 [0.62, 0.93]** |
| A1: penalty 1.0 vs 1.2 | **+1.392 [1.07, 1.71]** | **+0.775 [0.59, 0.96]** |
| A1 vs base, both at penalty 1.0 | −0.525 [−0.80, −0.25] | −0.025 [−0.08, 0.03] |

**Entry #4's hypothesis is confirmed, and it is the largest effect in this notebook by a wide
margin.** `repetition_penalty=1.2` applied over a 3.8k-token prompt was suppressing the article's
own vocabulary at every decode step, forcing near-miss tokens — the misspellings in entry #5 — and
costing **~1.5 judge points of faithfulness and ~0.8 of fluency on every arm this project has ever
measured**. The zero-shot base is not a 2.93-faithfulness model; it is a **4.48** model that was
being decoded badly. `no_repeat_ngram_size=3` moved with it in the same job, so the two are not
separated — but the penalty is the one with the mechanism, and 1-2 sentence summaries rarely repeat
a trigram legitimately.

**Consequences, in order of importance:**
1. Every faithfulness/fluency number in `d1-tables-step200.md`, and in entries #0/#5/#6 here, was
   measured through a broken decode. The *relative* orderings inside a fixed decode still hold
   (all arms shared the config), but no absolute number should be quoted going forward.
2. **The bar for "training improves the model" moved up by 1.5 points.** Beating the base now means
   beating 4.475/4.975 — close to the Gemini teacher's 4.908/4.992 and leaving little headroom.
   A1, re-decoded, is 3.95: still **below** a properly-decoded base.
3. The re-decode is free of the bf16/4-bit confound entry #6 raised: both arms came from the same
   4-bit job.

**Attribution caveat, stated deliberately:** E1 moved `repetition_penalty` 1.2→1.0 *and*
`no_repeat_ngram_size` 3→0 in the same job, so what is measured is the **penalty pair**, not
`repetition_penalty` alone. The mechanism above belongs to the penalty, and a 1-2 sentence Hebrew
summary rarely repeats a trigram legitimately, so the penalty is very likely carrying the effect —
but nobody should later cite `repetition_penalty` alone as the isolated cause. Separating them
would cost another $0.38 and change no decision, so it was not run.

**Length is not driving the arm differences** (checked before A4 reported, so the number could not
be chosen to fit): within-arm correlation of prediction word count against judge faithfulness is
r = **−0.09** for base-nopen (median 44 words) and **+0.02** for A1-nopen (median 11 words); the
Gemini teacher's is −0.29. Longer is, if anything, *slightly* worse inside an arm. The caveat that
remains: each arm's own length spread is narrow, so this rules out a length artifact *within* the
observed ranges, not across the 11-vs-44-word gap between arms.

**Mechanism confirmed WITHOUT the judge (added 2026-07-29, $0).** Everything above rests on one
Gemini judge that also supplies the teacher and the target filter, so the headline finding deserves
corroboration that does not involve it. The stated mechanism makes a purely lexical prediction: at
1.2 the model is pushed off the *article's own* vocabulary. Measured over the same 120 articles,
Hebrew-normalized, base arm only:

| | mean output words | fraction of output words present in the article | article-absent words per summary |
|---|---|---|---|
| penalty 1.2 | 31.6 | **0.204** | **25.1** |
| penalty 1.0 | 42.4 | **0.685** | **12.9** |

Paired Δ in article-coverage = **+0.481 [+0.456, +0.506]**, n=120. At 1.2 barely a fifth of the
model's words came from the article it was summarizing; at 1.0 it is over two thirds. And despite
writing **34% more words**, the un-penalized model emits **half as many** strings that appear
nowhere in the source. That is the predicted signature, measured by token counting, with zero
dependence on any LLM judge — the judge says *how much it cost*, and the token statistics say
*why*.

**Fix applied, not just recorded:** the 1.2/3 pair was hardcoded in three other generation paths
(`evaluation/infer.py`, `evaluation/predict_base_hf_job.py` ×2, `evaluation/prompt_sweep_hf_job.py`)
and is now off in all of them, with `train_hf_job.py`'s defaults flipped to 1.0/0. Otherwise the
next observation-notebook or base-baseline run would silently reproduce the bug. Note for anyone
re-reading the prompt arena: **rounds 1-3 were swept with the penalties on**, so their absolute
compliance/judge numbers carry this cost too.

**Why the fine-tune still trails a good base**: A1's targets were 15-word/1-sentence, so it learned
to compress the identifying specifics away (mean 10.6 words vs base 30.9). That is precisely the
over-compression failure mode of entry #5, and it is a *target* choice, not a training defect.

**A4 submitted** (job `6a6905e715e81eca66a8d556`): same recipe, but the teacher and student share a
**35-word / 2-sentence** instruction (`data/distill.py --target-words`), giving the student room to
carry who/what/where. Dataset `avreymi/amlk-distill-data-w35` (1841 train). Generated at
`repetition_penalty=1.0` — the new default for every arm from here.

## Entry #8 — A4 RESULT: the fine-tune reaches parity with a correctly-decoded base (2026-07-29)

Job `6a6905e715e81eca66a8d556`, COMPLETED ~1h30m (~$1.50). Same recipe as A1; the only change is
that teacher and student share a **35-word / 2-sentence** instruction that explicitly asks for
who/what/where, instead of the arena's 15-word/1-sentence prompt. Generated at penalty 1.0.

| system (n=120, penalty 1.0) | faithfulness | fluency | mean words |
|---|---|---|---|
| Gemini teacher (ceiling) | 4.908 | 4.992 | 54 |
| **A4 — distilled, 35-word targets** | **4.517** | **4.975** | **31.0** |
| base, correctly decoded | 4.475 | 4.975 | 41.3 |
| A1 — distilled, 15-word targets | 3.950 | 4.950 | 10.4 |

| comparison (paired) | faithfulness Δ [95% CI] | fluency Δ [95% CI] |
|---|---|---|
| **A4 vs A1** (target length, see caveat) | **+0.567 [+0.31, +0.82]** | +0.025 [−0.02, 0.07] |
| A4 vs correctly-decoded base | +0.042 [−0.16, 0.25] | 0.000 [−0.05, 0.05] |

**Verdict: parity, not a win — and the parity is the informative part.** A4 does not beat the base
(CI includes 0), but it *matches* it at 4.52/4.98 while writing **31 words instead of 41**, in a
consistent 2-sentence register, and it closes 87% of the base→teacher gap. Its score distribution
is 78×5 and 31×4 out of 120, with a single 1.

The A4-vs-A1 delta is the cleanly attributable training result: **+0.567 faithfulness from changing
one number in the instruction the targets were written under**. Strictly, target length is not the
*only* difference — A4's predictions were generated bf16 inside its training job, A1's `-nopen`
predictions 4-bit inside the E1 job (same precision mismatch flagged in entry #6). A quantization
difference cannot plausibly be worth 0.567 judge points, so the attribution stands; it was not worth
$0.45 to separate. Entry #5's diagnosis — that A1's
losses were over-compression, not bad grounding — is confirmed: given room for the identifying
facts, the same pipeline produces summaries that carry the price, the channel, the person's name.

Cumulatively, from the project's starting point (2.242/3.458) to A4 (4.517/4.975), the combination
of *fixing the decode* and *changing the targets* is worth **+2.28 faithfulness and +1.52 fluency**.

**A6 submitted** (job `6a691d7da9f4e0ab00b2bef8`, ~$1.50): 60-word / 3-sentence targets
(`avreymi/amlk-distill-data-w60`, 1847 examples). Rationale: the teacher scores 4.908 at a median
of 54 words, and A4's own gain came from moving toward that length. If the trend holds it is the
best remaining shot at *beating* the base rather than matching it. A larger 35-word set
(`amlk-distill-data-w35-full`, 4323 examples) stays unfunded until A6 reports.

## Entry #9 — A6: the length lever is saturated (2026-07-29)

Job `6a691d7da9f4e0ab00b2bef8`, COMPLETED 1h44m (~$1.75). Identical to A4 except the shared
teacher/student instruction is **60 words / 3 sentences** instead of 35/2.

| arm | faithfulness | fluency |
|---|---|---|
| A4 (35-word targets) | 4.517 | 4.975 |
| **A6 (60-word targets)** | **4.508** | **4.975** |
| base, correct decode | 4.475 | 4.975 |

A6 vs base: +0.033 [−0.15, 0.22] — parity, same as A4. A6 vs A4: a 0.009 difference, i.e. nothing.

**Verdict: the target-length lever is exhausted.** 15 → 35 words bought +0.567; 35 → 60 buys 0.000.
The hypothesis that drove A6 (get closer to the teacher's natural 54-word length and inherit more of
its 4.908) is **refuted**: length was never the teacher's advantage past ~35 words. Do not spend
further budget on target length.

Where the remaining gap actually lives: A4's score distribution is 78×5, 31×4, 7×3, 3×2, 1×1. The
mean is held down by ~11 examples, not by a broad deficit. Beating the base therefore requires
fixing the *tail*, which is a target-**quality** question, not a target-**length** one — motivating
A7 below rather than a bigger dose of the same data.

## Entry #10 — A7: judge-verified targets, and more of them (2026-07-29, submitted)

Two things change at once, deliberately, because the budget no longer supports separating them and
they push the same direction:
1. **Quality:** every teacher target in the full 35-word set is scored by the *same* judge that
   scores the student, and only targets rated **faithfulness = 5 and fluency ≥ 4** survive. The
   format filter used until now checks shape (length, sentences, script); it cannot see a wrong
   number. Entry #9 localized the student's remaining gap to a ~10% tail, and a target the judge
   itself calls unfaithful is the most plausible way to teach that tail.
2. **Quantity:** the source is the full curated train split distilled (4323) rather than the 1841
   used by A4.

**Verification outcome (a finding in itself):** **3356/4323 (77.6%)** of teacher targets passed;
**22% did not** — the judge rated them below a perfect 5 on faithfulness against their own article.
Every arm so far trained on that 22% unfiltered. Val: 133/176. Dataset
`avreymi/amlk-distill-data-w35-verified`. Job `6a6937eb15e81eca66a8d763` (3356 examples ≈ 210
steps, ~$2.40).

Built with `data/distill.py --verify-existing` (API only, no HF budget). This is a standard
quality-filtered distillation / rejection-sampling-style recipe, with the unusual property that the
filter and the metric are the *same* judge — which is a legitimate advantage for the target metric
and simultaneously a caveat to state in the paper (it optimizes toward that judge's preferences,
so an independent judge should confirm any win before it is claimed as general).

## Entry #11 — A7 killed mid-run at step 100/210; scoring what survived (2026-07-29)

Job `6a6937eb15e81eca66a8d763` ERRORed after 1h08m, immediately following the step-100 eval and
Hub push (loss 0.509, eval_loss 0.546) — not a timeout (declared 4h), the same infra-kill pattern
as B1. `hub_strategy=every_save` did its job: the **step-100 adapter is on the Hub**, though as
root files rather than a `checkpoint-100/` folder, so a true Trainer resume is not possible — the
optimizer state was not preserved.

Rather than pay ~$2.40 to retrain from scratch, submitted an inference-only job on the surviving
adapter (`6a694859a9f4e0ab00b2c23b`, suffix `-step100`, ~$0.50). At step 100 of 210 the model has
seen ~1600 examples — comparable *exposure* to A4's full 1841-example epoch — so it is a fair, if
imperfect, read on whether judge-verified targets move the plateau. Caveat to carry into the
comparison: the cosine LR schedule never completed, so this adapter stopped at a relatively high
learning rate rather than annealing to zero, which typically costs a little quality.

## Entry #12 — A7 result and the SFT plateau (2026-07-29)

Inference job `6a694859a9f4e0ab00b2c23b` on the surviving step-100 adapter (~$0.45).

| arm (all penalty 1.0, n=120) | faithfulness | fluency | Δ vs base [95% CI] |
|---|---|---|---|
| A4 — 1841 distilled, 35-word | 4.517 | 4.975 | +0.042 [−0.16, 0.25] |
| A6 — 1847 distilled, 60-word | 4.508 | 4.975 | +0.033 [−0.15, 0.22] |
| **A7 — 3356 judge-verified, 35-word (step 100/210)** | **4.367** | **4.933** | −0.108 [−0.29, 0.07] |
| base, correct decode | 4.475 | 4.975 | — |
| Gemini teacher | 4.908 | 4.992 | +0.433 |

**The SFT plateau is the result.** Three target recipes — different lengths, different sizes, with
and without judge verification — all land in a 0.15 band around the base, every CI straddling zero.
Filtering out the 22% of imperfect targets did **not** help (with the caveat that this checkpoint
stopped mid-schedule at a high learning rate, which costs some quality; a completed A7 might recover
the ~0.15 back to A4's level, but nothing in this series suggests it would clear the base).

Read together with entries #8–#9, the shape is a classic **distillation ceiling**: the student
readily imitates the teacher's *format* — length, register, sentence count, no digests — and
reaches the base's factual quality, but it does not inherit the teacher's factual *precision*
(4.91). More data, cleaner data, and longer targets all fail to move it, which points at student
capacity rather than at the training signal. That is a real, publishable finding for the paper, and
it is the honest answer to "can SFT beat a correctly-decoded base here": **not on this scale of
data, no.**

What has not been tried is the class of method that does not work by imitation: **preference
optimization** (DPO), which trains on *contrasts* — this is right, that is wrong — instead of on
one correct answer. It is the only remaining technique with a mechanism aimed at the failure tail
rather than at the average. Funded next, with the remaining ~$7.

## Entry #13 — DPO: preference pairs built from the teacher's own misses (2026-07-29)

**Hypothesis.** Every SFT arm converged to the base's quality because imitation learning has no way
to represent "not that". DPO optimizes a *contrast*, so it can push probability mass away from the
specific failure mode (a fluent summary with one wrong detail) instead of only pulling it toward
correct answers.

**The pair-construction problem, and how it was solved for $0 of GPU.** DPO needs (prompt, chosen,
rejected). The textbook recipe samples `rejected` from the policy itself, which here would mean
~1000 GPU generations (~$2.30) before training even starts. Instead the pairs come from data that
already existed: the **22% of teacher targets that failed judge verification** become `rejected`,
and the teacher is **resampled at temperature 1.0** on those same articles — a temperature-0
resample would return the identical failed text — with a resample that *passes* the judge becoming
`chosen`. Both sides are therefore the same teacher, same article, same format, differing only in
judged faithfulness, which is exactly the axis being optimized.

Yield: 967 failed → 948 well-formed and different resamples → **356 pairs** where the new sample
scores a perfect 5 and the old one did not. `avreymi/amlk-distill-pairs-w35`.

356 pairs is small (≈44 optimizer steps at effective batch 8) — enough for DPO to shift a decoding
preference, not enough to teach new knowledge. That matches the intent: the model already writes
the right *kind* of summary, and the target is its error tail.

**Implementation** — `training/dpo_hf_job.py`, a new self-contained HF Jobs script (submitter and
job in one file, like `predict_base_hf_job.py`). It starts from the **A4 adapter** (the best SFT
arm), loads it 4-bit with `is_trainable=True` so PEFT's frozen base serves as DPO's implicit
reference model (no second 7B copy), and generates the judged subset with the identical decode
config. Smoke first: job `6a69529315e81eca66a8d868` (32 pairs, 10 steps, 4 predictions, ~$0.30) —
a brand-new script path, and the three ERRORed jobs in this session all cost more than a smoke does.

**The smoke earned its keep twice.** First run ERRORed on `DPOConfig.__init__() got an unexpected
keyword argument 'max_prompt_length'` — the field does not exist in the trl build the container
resolved. A self-contained job script cannot pin the version it will be given, so the fix is
structural rather than a one-line edit: the config is now assembled as a dict and filtered against
`inspect.signature(DPOConfig.__init__)`, printing whatever it had to drop. A future trl rename
degrades to a logged omission instead of a crash after the 12-minute install. Second smoke
(`6a69537ca9f4e0ab00b2c315`) COMPLETED: trained, pushed, generated. Total smoke cost ~$0.30, versus
~$1.00 for discovering either failure in the real run.

**Real run: job `6a6956c7a9f4e0ab00b2c344`** (356 pairs, 1 epoch, β=0.1, lr 5e-6, judged subset,
~$0.95). **Training completed all 45 steps and the DPO objective clearly learned the preference:**

| final training metric | value | meaning |
|---|---|---|
| `rewards/accuracies` | **0.833** | the policy ranks `chosen` above `rejected` on 83% of pairs |
| `rewards/margins` | 0.0997 (rising from ~0) | the gap it puts between them is growing |
| `rewards/chosen` | −0.003 | chosen barely moved from the reference — good, no collapse |
| `rewards/rejected` | −0.103 | the loss is achieved by pushing the bad summary *down* |

That last pair is the healthy DPO signature: it suppressed the unfaithful continuation without
dragging the good one with it. The job then died **during generation, at ~20-30/120** — the same
infra-kill that hit B1 and A7, now three times in one session on this account. The adapter was
pushed before it died, so predictions are being regenerated by an inference-only job
(`6a69647815e81eca66a8d923`, ~$0.50) instead of retraining.

## Entry #14 — DPO RESULT, and the final picture (2026-07-29)

Inference job `6a69647815e81eca66a8d923` on the DPO adapter (~$0.45).

| comparison (n=120, paired) | faithfulness Δ [95% CI] | fluency Δ [95% CI] |
|---|---|---|
| DPO vs A4 (its own SFT starting point) | +0.033 [−0.09, 0.15] | −0.008 [−0.04, 0.02] |
| DPO vs correctly-decoded base | +0.075 [−0.12, 0.27] | −0.008 [−0.05, 0.04] |

DPO = **4.550 / 4.967**, 31.5 words, distribution 79×5, 33×4. It is the highest faithfulness of any
trained arm, and the only arm whose point estimate sits above the base on faithfulness — but the CI
still includes zero, so by this notebook's own rule **it is not a win**, and its gain over its SFT
starting point is well inside noise.

Note the tension worth recording: training-time `rewards/accuracies` reached 0.833, i.e. DPO
genuinely learned to rank the faithful summary above the unfaithful one — yet that ranking ability
translated into ~0.03 judge points at generation time. Learning to *score* a contrast is not the
same as changing what greedy decoding *emits*; with 356 pairs and β=0.1 the policy barely moved off
its reference. A larger pair set is the obvious follow-up, but the observed slope does not suggest
it would clear the 0.22-point bar.

### Final results table — everything measured, one instrument, same 120 articles

| system | decode | faithfulness | fluency | words |
|---|---|---|---|---|
| Gemini 2.5 Flash-Lite (teacher / ceiling) | — | 4.908 | 4.992 | 54 |
| **DPO on top of A4** | 1.0 | **4.550** | 4.967 | 31.5 |
| **A4 — SFT, 35-word distilled targets** | 1.0 | **4.517** | 4.975 | 31.0 |
| A6 — SFT, 60-word distilled targets | 1.0 | 4.508 | 4.975 | — |
| **zero-shot base, correct decode** | 1.0 | **4.475** | 4.975 | 41.3 |
| A7 — SFT, judge-verified targets (step 100/210) | 1.0 | 4.367 | 4.933 | — |
| HeSum gold references (the labels themselves) | — | 4.233 | 4.983 | 19 |
| A1 — SFT, 15-word distilled targets | 1.0 | 3.950 | 4.950 | 10.4 |
| zero-shot base | 1.2 | 2.933 | 4.200 | 30.9 |
| A1 — SFT, 15-word targets | 1.2 | 2.558 | 4.175 | 10.6 |
| **project's original fine-tune (HeSum, step 200)** | 1.2 | **2.242** | 3.458 | 42.2 |
| B1 — SFT on HeSum references, full epoch | 1.2 | 2.167 | 3.558 | — |

### The answer to the question this notebook was opened for

*"Training fails to improve the model — make it really improve."*

1. **The single biggest lever was not training at all.** The decode penalty pair cost ~1.5
   faithfulness and ~0.8 fluency on every system ever measured here (#7). Free to fix, no retraining.
2. **Within training, the target text is what matters, and it matters a lot.** At identical step
   count and identical decode, distilled targets beat HeSum references by **+0.392 faithfulness /
   +0.617 fluency** (#6), and moving the target length from 15 to 35 words is worth another
   **+0.567 faithfulness** (#8). Both CIs exclude zero. Those are real training improvements.
3. **Against a *correctly decoded* base, no configuration achieved a statistically significant
   win.** Best is DPO at +0.075 [−0.12, 0.27]. Six arms — three target lengths, two data sizes,
   quality filtering, and preference optimization — all land in a 0.2 band around 4.5 while the
   teacher sits at 4.91. That is a distillation ceiling, and it is a legitimate finding, not a
   failure to try: the student inherits the teacher's *format* completely and its *precision* not
   at all.
4. **What the fine-tune does buy, and it is not nothing:** the same quality in **31 words instead
   of 41**, in a stable 2-sentence register, with no digests and no format drift — i.e. a
   controllable summarizer rather than a verbose one. For the paper, that is the honest claim.

**End to end, from the project's starting point (2.242 / 3.458) to the best arm (4.550 / 4.967):
+2.31 faithfulness and +1.51 fluency.**

### Recommended next steps (not funded here)

- Re-measure the D1 battery with the corrected decode before anything goes in the paper.
- Confirm the decode finding with an independent judge (the same Gemini family judges *and*
  supplies the teacher and the target filter — self-preference is a live caveat throughout).
- If beating the base is the goal, the evidence points at student capacity, so: a larger base
  model, or many more preference pairs (thousands, sampled from the policy itself), not more SFT.

## Arm queue (written before results, so hindsight can't reorder it)

| arm | change | why it might work | est. cost |
|---|---|---|---|
| **A1** | LoRA on 1941 Gemini-distilled targets, 1 epoch | targets score 4.9/5.0 on the judged axis; the student only has to imitate style + grounding | ~$1.5 |
| **B1** | same recipe, same 1941 *articles*, HeSum headline targets | the only clean control: separates "wrong targets" from "under-trained adapter" (step-200 arm was a SIGTERM'd run) | ~$1.5 |
| **A2** | A1 + more data / 2 epochs, or lower LR | if A1 helps, find how much of the teacher gap a bigger dose closes | ~$2 |
| **E1** | inference-only re-decode of A1 at `repetition_penalty=1.0`, both arms | entry #4: the penalty is applied over the prompt, so it penalizes copying the article's own entities | ~$0.6 |
| **C** | DPO on (teacher = chosen, this model's own output = rejected) | pairs already exist for free on the same articles; optimizes the *contrast* the judge measures rather than likelihood | ~$2 |
| **D** | rejection sampling / self-distillation (sample K from the base, keep the best-scoring) | keeps the student's own distribution, so less style shock than an external teacher | ~$3 |

Nothing below A1 gets funded until A1 has a number.

Datasets prepared while A1/B1 ran (API only, $0 of HF budget), so a follow-up arm can launch the
moment a result lands: `avreymi/amlk-distill-data` (1941 train) and `avreymi/amlk-distill-data-full`
(**4537** train / 189 val, the whole curated train split distilled; 4683 requested, 146 lost to
teacher refusals and the format filter). Both are new private repos — `amlk-training-data` is
untouched.

## Budget tally

| entry | job(s) | cost |
|---|---|---|
| #0, #2, #3 | — (local API only) | $0.00 |
| #1 smoke | `6a68e74b` (a10g-small, 9m54s, COMPLETED) | ~$0.17 |
| #5 A1 | `6a68ea3915e81eca66a8d3f2` (1h29m, COMPLETED) | ~$1.50 |
| #6 B1 | `6a68eae1a9f4e0ab00b2ba5d` (1h35m, ERROR at gen 70/120) | ~$1.60 |
| #6 B1 regen | `6a69018815e81eca66a8d526` (inference-only, ~45m est.) | ~$0.75 est. |
| #7 E1 | `6a68ffc4a9f4e0ab00b2bc85` (22m41s, COMPLETED) | ~$0.38 |
| #8 A4 | `6a6905e715e81eca66a8d556` (~1h30m, COMPLETED) | ~$1.50 |
| #9 A6 | `6a691d7da9f4e0ab00b2bef8` (1h44m, COMPLETED) | ~$1.75 |
| #11 A7 | `6a6937eb15e81eca66a8d763` (1h08m, ERROR at step 100) | ~$1.15 |
| #12 A7 inference | `6a694859a9f4e0ab00b2c23b` (COMPLETED) | ~$0.45 |
| #13 DPO smokes | `6a695293` (ERROR, API), `6a69537c` (COMPLETED) | ~$0.30 |
| #13 DPO run | `6a6956c7a9f4e0ab00b2c344` (55m, trained then ERROR in gen) | ~$0.95 |
| #14 DPO inference | `6a69647815e81eca66a8d923` (COMPLETED) | ~$0.45 |
| **total spent** | | **≈$11.1 / $16** |

All Gemini API work — 4 distilled datasets, judge verification of 4323 targets, preference-pair
construction, and every judging pass — cost $0 of the HF budget.

A1 and B1 run in parallel — same recipe, same 1941-example step count (`--max-train 1941` on the
untouched `avreymi/amlk-training-data`), same judged subset, same skipped base arm. The **only**
difference is the target text, which is what makes the pair interpretable.
