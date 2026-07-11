# AMLK Training: Diagnosis and Improvement Plan

---

# CODE AUDIT (2026-07-11) — verified against the code, the tokenizer, and the HF job logs

Scope: **code defects only**. Dataset and training regime are being left as-is for now, so
P0 (grounding filter) and the training-hyperparameter items below are **not** actioned.
Everything in this section was empirically verified, not inferred.

**Coverage boundary.** Deep-read + empirically exercised: `training/{config,train,train_hf_job}.py`,
`data/{prompts,preprocess,clean}.py`, `evaluation/{evaluate,infer,base_predict,predict_base_hf_job,
hebrew_constraint,error_analysis}.py`, the last 4 HF job logs, and the smoke's pushed predictions.
Grepped only, not reviewed line-by-line: `data/download.py`, `evaluation/{build_report_tables,
predict,eval_hf_job,topic_clustering,style_labels,stratify_by_topic,viewer}`. The side-analysis
modules are off the critical path for the next run.

## Bugs, ranked by impact on the next real run

**C0 — The finetuned model is trained and served without the instruct model's chat template.**
Verified: `"[INST]" in train["prompt"][0]` → `False`. `dictalm2.0-instruct` ships a chat template
(`{{bos_token}}[INST] {content} [/INST]`) and its card requires it, but `data/preprocess.py` bakes
the raw completion-style prompt into the `prompt` column, and `build_input_text` returns that
prompt verbatim for `label != "base"`. So the *only* arm that is format-correct is the zero-shot
base one; the finetuned arm — the one whose hallucination prompted this whole review — is fed a
format the model was never instruction-tuned on. This is the same defect class as C1 (wrong prompt
format reaching the model), just on the finetuned arm instead of the base arm. Fixing it requires
touching preprocess + inference, which is why it sits under "leave training as-is" — but it is a
**bug**, not a design preference, and it is free to fix.

**C1 — Double BOS in the zero-shot base prompt.** `apply_chat_template(tokenize=False)` renders a
string that already begins with `<s>`; the following `tokenizer(prompts, ...)` call then prepends
a *second* `<s>` (dictalm2 sets `add_bos_token=True`). Verified token ids:
`['<s>', '<s>', '[', 'INST', ']', 'Sum']`. Hits the **base** arm only — the finetuned arm uses the
raw prompt and gets a clean single BOS. Prospective, not retrospective: DictaLM-3-Base had no chat
template, so this path never fired before.
*Observed impact is modest*: the smoke's `predictions-base.jsonl` is coherent, on-topic Hebrew
(over-long, but not garbled). Treat this as a correctness defect to fix, not as an explanation of
bad numbers. Fix: `add_special_tokens=False` when tokenizing already-templated text.
**Four call sites** — `training/train_hf_job.py`, `evaluation/base_predict.py`,
`evaluation/infer.py`, `evaluation/predict_base_hf_job.py`.

**C2 — `/no_think` is injected as literal text into a Mistral prompt, and the four copies have
drifted.** `dictalm2.0-instruct` is Mistral-based with no thinking mode, and its `[INST]` template
has no notion of `/no_think`. The wrapped string ends `'...think [/INST]\n'` — those literal
characters are now part of the user turn. Leftover from the Qwen3/DictaLM-3 era, as is
`strip_think()` (dead code for this base).
Worse, the four hand-maintained copies of this function have **already diverged**:
`train_hf_job.py` and `base_predict.py` append `/no_think`; `predict_base_hf_job.py` does not. So
the in-training base arm and the multi-model base baseline are fed **different prompts** and are
not strictly comparable. The "keep the twins in sync by hand" convention has failed in practice —
worth collapsing to one shared source rather than re-syncing four copies.

**C3 — The observation notebook will OOM on a T4.** `evaluation/infer.py:load_finetuned_model`
defaults to `quantize=False`, and `notebooks/evaluation_observation.ipynb:279` calls it as
`load_finetuned_model(MODEL_REPO)`. That loads 7B in bf16 (~14.7 GB) on a 16 GB Colab T4, plus
adapter and KV cache at batch 8 / seq 2048. T4 (Turing) also has no native bf16. The default was
correct for the old 1.7B base; the 7B swap silently broke it.

**C4 — `train_hf_job.py` hardcodes hyperparameters and ignores `METHOD_PRESETS`.**
`per_device_train_batch_size=2`, `gradient_accumulation_steps=8`, `learning_rate=2e-4` are
literals. They *happen* to equal the `qlora` preset, so the default path is correct — this is
**latent, not active**. But `--submit-hf --method full` silently trains at lr=2e-4 instead of the
preset's 5e-5, and neither `lora` nor `full` is viable for a 7B on a 24 GB A10G anyway (full FT
needs ~100 GB of master-weight + optimizer state). The CLI advertises three regimes; one works.

**C5 — The 6h default timeout is razor-thin for the full run.** Measured from the smoke
(job `6a524384effc02a91cbd98c6`): ~38 s/optimizer-step. 6073 train examples / (2×8) = 380 steps
→ ~4.0 h training, + ~8 min of evals, + generation of 760×2 = 1520 summaries at batch 8
(~190 batches × ~30 s) ≈ 1.6 h. **Total ≈ 5.8 h against a 6h timeout.** Every observed prediction
ran to the full 256-token cap without emitting EOS, so generation is at its worst case, not a
typical one. Use `--timeout 8h` (a10g-small is $1/h). Timeout enforcement is unreliable in either
direction — job `6a4fbd8c` ran 12h20m under a 6h declaration and still completed.

## Verified working — do not "fix" these

- **ROUGE on Hebrew is correct.** `_UnicodeTokenizer` overrides rouge_score's ASCII-only default
  tokenizer. Checked: identical Hebrew → 1.0, disjoint Hebrew → 0.0.
- **The judge scores prediction-vs-ARTICLE**, not vs-reference (`evaluate.py:JUDGE_PROMPT`,
  `error_analysis.py:LABEL_PROMPT`). P3's premise is already satisfied in code.
- **The judge's `text[:6000]` clip affects 0/760 test articles** (median 3,916 chars). Non-issue.
- **`/data` is per-job scoped** — `hf jobs inspect` shows `path: 20260711T132209-9b0590`. No
  cross-job checkpoint contamination is possible.
- **The checkpoint-resume path HAS fired on a real infra restart — on the 1.7B run.** Job
  `6a4fbd8c` logs show one container start plus `Found existing checkpoint(s) in /data/output —
  resuming training`, and the job completed. CLAUDE.md and P4 below claim the mechanism has *never*
  been exercised; that is wrong. Caveat in the other direction: it has fired only on the 1.7B
  config, never on 7B/QLoRA, so "works on this base" is still unproven.
- **Tests: 67 pass**, 3 fail only on optional `plotly` not being installed.

## Corrections to the plan below (P0–P5)

- **P4's "add `seed=42` (currently never set anywhere)" is wrong.** HF `TrainingArguments`
  already defaults `seed=42`; runs are deterministic today.
- **P4's "QLoRA becomes the correct default" is already done** — `train.py:279` defaults to qlora.
- **P4's "the resume path has never been tested" is wrong** — see above.
- **Missing `paged_adamw_8bit` is a memory margin, not a failure.** The smoke fit comfortably at
  batch 2 / seq 2048. Do not expect an OOM without it.
- **P0's retention figures are unreliable.** Recomputed on the full 10k: median coverage **0.739**
  (plan says 0.67); threshold 0.6 retains **80.5%**, 0.7 retains **59.6%** (plan predicts ~46%).
  The direction of P0 holds; the magnitudes do not. Moot while data is left as-is.

## Judgment calls, not bugs (flagged, not actioned — data/training left as-is)

- The Hebrew decode constraint bans every Latin-script token, but **4.7% of references contain
  Latin** ("CNN", "ynet"). The model is trained toward targets it is structurally forbidden to
  emit at decode time. A real inconsistency between the training target and the decode config,
  but small and deliberate-looking — calling it a judgment call, not a bug.
- `load_best_model_at_end=True` with `metric_for_best_model="eval_loss"` selects the checkpoint
  that best reproduces the references — including the ungrounded ones (see P0).
- Every prediction observed in the smoke ran to the full 256-token cap without emitting EOS
  (base 378–474 chars, finetuned 618–662, vs. a 157-char median reference). Consistent with P2's
  decode-config hypothesis, but 10 training steps is not enough to conclude anything.

---

## Context

You're unsatisfied with training results (hallucination + low ROUGE). Two research passes found **six separate problems**, only one of which is a hyperparameter issue. They stack in this order of impact:

| # | Problem | One-line fix |
|---|---|---|
| P0 | Training targets are largely unsupported by their articles | Filter training data by article-support coverage |
| P1 | Fine-tuning an *instruct* model without its instruction format | Wrap prompts in the model's chat template |
| P2 | Decode config (`no_repeat_ngram_size`, `repetition_penalty`) already proven to crush scores once | Re-test with plain greedy decode |
| P3 | "Low ROUGE" is a miscalibrated expectation for this dataset | Re-anchor success metric to faithfulness, not ROUGE |
| P4 | 7B model swap invalidated all hardware/preset assumptions | Re-validate regime (QLoRA, paged optimizer, timeout) on 7B |
| P5 | English instruction to a Hebrew-instruct-tuned model | Cheap A/B, adopt only if visibly better |

Do them **in this order** — P1/P2/P3/P4 are prerequisites for a clean read on P0's data experiment, and several are free or near-free to check before spending on a real training run.

---

## P0 — Training targets are largely unsupported by their articles

**Diagnosis.** SFT with `completion_only_loss=True` maximizes `log P(reference | article)`. Gradient descent has no notion of whether the reference is *derivable* from the article — it just pushes up the probability of whatever text follows. Measured on the test set with generous Hebrew stemming (niqqud/final-form normalization + clitic-prefix strip + 4-char stem, biased to overcount matches):

- **Median reference has only ~67% of its content words present in its source article.**
- One reference name-drops "זורנל", "המגזין", and the *Israel Hayom* deputy editor — none appear in the article.
- One "summary" is a scraped related-articles widget: `אולי יעניין אותך גם:` ("you might also be interested in").
- The repo's existing always-on roundup drop (`is_roundup_digest`, 3+ pipe segments) only partially helps: **among references that survive it, 12.7% are still <50% grounded.**

This is corroborated independently by the HeSum paper itself (ACL Findings 2024): summaries are journalist-written **"extended subheadings"** (teasers, not summaries), 42% novel unigrams, 73% novel bigrams — among the highest abstractiveness of any summarization dataset, by design.

For thousands of examples, the only way to reduce loss is to raise the model's general prior toward emitting plausible-but-unsourced Hebrew media-column content after *any* article — a mechanical instruction to hallucinate. This explains two results that otherwise look like bugs: judge faithfulness *fell below the untuned base* (2.98 → 2.64) after fine-tuning, and ROUGE got *worse* with more epochs (11.4 @ 1 epoch → 5.1 @ 3 epochs — more passes, more of that gradient).

**The real risk of fixing this is not data starvation — it's extraction.** A coverage filter selects *for* extractive references. If ROUGE rises because the model became a lead-copier, that undermines the project's own lead-bias thesis (TODO F). Treat `lead_copying` rate as a hard gate, not a footnote.

**Fix — Phase 0: prove/calibrate (local, CPU, ~1h, $0). Gate.**
- New module `data/grounding.py` (stdlib only, no `datasets`/`transformers` — same convention as `data/clean.py`): `normalize()`, `stem()` (clitic-strip + 4-char truncate), `content_stems()`, `coverage(article, summary)`, `coverage_bucket()`, `is_grounded()`.
- CLI `--report` on the **full 10k** `combined.jsonl`: coverage deciles, retention at each candidate threshold, how much it removes *beyond* the existing roundup drop.
- Add two model-free oracles, stratified by coverage bucket, on the existing prediction files:
  - **Gemini on ~300 examples** (`evaluation/predict.py`) — a faithful oracle. Its ROUGE-by-bucket curve is the *learnable floor*.
  - **Lead-3 extractive baseline** (first 3 sentences) — 100% grounded by construction; this is TODO B'.3, owed anyway. Its curve tells you if a threshold is just selecting *extractable* references.
  - Read both curves together: Gemini's answers "can any faithful system score here?"; Lead-3's answers "am I about to turn this into a lead-copying task?"
- **Coverage must be computed against the *truncated* article** (post `truncate_to_tokens`, `MAX_LENGTH-256`=1792 tokens) — the text the model actually trains on — at both Phase 0 and preprocess, or the calibrated threshold won't transfer.
- **Gate:** ROUGE should rise monotonically with coverage bucket for every system. If flat, the hypothesis is wrong — stop before spending on GPU.

**Fix — Phase 1: wire the filter as a train-time knob, not a preprocess-time drop.**
- `data/preprocess.py` — always emit `coverage` + `coverage_bucket` columns on all splits (train/val/test). No filtering here.
- `training/train_hf_job.py` — new `MIN_COVERAGE` env var filters `train_ds` and `val_ds` (never `test_ds`). Must run **before** the smoke/mini `select(range(50))` slices, or a smoke job can go near-empty and pass vacuously.
- `training/config.py` — `model_repo(hf_user, variant, min_coverage=0.0)` appends `-g70` when filtering; `dataset_repo` stays unsuffixed so **one** dataset repo serves both A/B arms (`--skip-data-upload` already exists).
- **Never filter test.** Report it twice: overall (headline, honest) and stratified by coverage bucket (where the real story is). Threshold: pick from the Phase 0 oracle curves, not a priori; prior estimate 0.6–0.7 (~46% retention, ~4.6k rows — still ~290 optimizer steps at effective batch 16, not starved).

---

## P1 — Fine-tuning an instruct model without its instruction format

**Diagnosis.** `dicta-il/dictalm2.0-instruct` is a 7B Mistral-based model, instruction-tuned via the Zephyr recipe. Its model card is explicit: prompts must be wrapped `[INST] ... [/INST]`. The pipeline currently trains and infers the **fine-tuned** system on the raw completion-style prompt (`"Summarize the following Hebrew text...\n\nSummary:\n"`) with no chat wrapping — only the **zero-shot base** system gets `apply_chat_template()` (`train_hf_job.py:build_input_text`, `label == "base"` branch only).

Consequences: the entire reason to pick the instruct variant — its instruction-following — is bypassed; the fine-tune starts from raw text-continuation mode and must relearn instruction behavior from ~8k examples; the A/B is skewed (base answers in its native format, finetuned in a foreign one). SFT literature is consistent here: when a chat template exists, training should reuse it, or performance measurably degrades.

This defect is new: the previous base (DictaLM-3-Base) had no chat template, so raw prompts were *correct* for it. The swap to an instruct model silently made the raw-prompt path wrong.

**Fix.** Build the training prompt via `tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)` in `data/preprocess.py` (or at train time), giving `<s>[INST] {instruction+article} [/INST]`; completion stays the reference (TRL already appends EOS). Make `build_input_text` use the *same* wrapped format for the finetuned system at inference so train and inference match. Watch the **double-BOS trap**: the template emits `<s>`, and the tokenizer may add BOS again at encode time — strip one. Keep the raw-prompt fallback when `tokenizer.chat_template` is falsy, to preserve the DictaLM-3-Base probe path.

Highest leverage of all six fixes, and free.

---

## P2 — The decode config already proved it crushes scores, once

**Diagnosis.** The project's own recorded results are direct causal evidence: **v1 → v2 used the identical adapter and changed only decoding — ROUGE-1 fell 11.4 → 4.7.** Yet the current shared decode config (`train_hf_job.py` / `evaluation/infer.py`) still carries the suspects:
- `no_repeat_ngram_size=3` — blocks any repeated 3-gram, but Hebrew news summaries legitimately repeat multi-token entity names (ministries, outlet names like "ידיעות אחרונות"); blocking forces the model off a faithful continuation mid-entity. This operates on token identity, not meaning — a known limitation in the literature.
- `repetition_penalty=1.2` — a strong additional push away from high-probability (faithful) tokens.
- Symptom: v2/v3 median prediction ≈ 548 chars vs ~150-char references — over-generation that directly destroys ROUGE precision.

**Fix.** Cheap and self-verifying (~$1, no retraining): the old DictaLM-3 adapter (`avreymi/amlk-dictalm3-1.7b-sft-clean`) is still on the Hub, and `--inference-only --pred-suffix` exists precisely for this. Submit one inference-only job with plain greedy decode (explicit EOS, `max_new_tokens=256`, **no** `no_repeat_ngram_size`, **no** `repetition_penalty`), score both files locally (`evaluation.evaluate --limit 200 --skip-llm` is enough for length/ROUGE). If plain decode recovers length/ROUGE (expected, given v1=11.4 was under the old decode), remove both knobs from the shared config; re-add the mildest only if visible looping returns.

---

## P3 — "Low ROUGE" is a miscalibrated expectation for this dataset

**Diagnosis.** From the HeSum paper's own results table: **fine-tuned mLongT5 = 17.5 ROUGE-1; GPT-4 = 13.6; GPT-3.5 = 13.7.** Your v1's 11.4 was already within GPT-4's range. More importantly, the paper reports **ROUGE correlates *negatively* with human judgment on HeSum** (Pearson r ≈ −0.16, p < 0.00005) — a direct consequence of Hebrew's morphology and flexible word order plus the dataset's teaser-style abstractiveness. Optimizing or selecting checkpoints on ROUGE here is actively counterproductive.

**Fix.** No code change, but a firm policy change: primary success metric everywhere (this plan, report tables, checkpoint selection) is **judge faithfulness + error-analysis rates**, scored prediction-vs-*article*. Report ROUGE with the HeSum context line (GPT-4=13.6; negative human correlation) so it's read correctly in the paper. Never select thresholds, checkpoints, or "which arm won" purely on ROUGE.

---

## P4 — The 7B swap invalidated the repo's hardware/regime assumptions

**Diagnosis.** All existing docs, presets, and cost/time estimates were written for a ~1.7–2B model ("prefer `--method lora` over qlora for a model this size"). `dictalm2.0-instruct` is **7B**:
- LoRA (bf16 base, no quantization) is now marginal on a 24GB A10G — ~14GB weights alone plus activations at seq 2048/batch 4 risks OOM. **QLoRA becomes the correct default**, and the missing `optim=paged_adamw_8bit` (QLoRA quantizes the base to 4-bit but still runs a full fp32 Adam optimizer state — the classic QLoRA misconfiguration) matters more than before.
- Training time is roughly 3–4× the old 1.7B estimates; the 6h timeout that a 1.7B run already brushed against is a real risk for more than 1 epoch. (`/data/output` + `resume_from_checkpoint` mitigate this, but the resume path has never been tested against a real infra restart.)
- **Nothing has been validated end-to-end on this base** — the "pipeline validated" smoke test was on the old 1.7B model.
- Separately, still-live wiring bug: `train_hf_job.py` hardcodes `batch=2, accum=8, lr=2e-4` and ignores `METHOD_PRESETS` entirely, so `--submit-hf --method full` silently trains at `lr=2e-4` instead of the preset's `5e-5`.

**Fix.**
- Default `--method qlora`; add `optim="paged_adamw_8bit"` to the qlora path.
- Re-check the `lora` preset's `batch=4` for 7B — likely reduce, or gate with an explicit OOM warning.
- Run `--smoke-test` (~$0.05) **on the 7B** before anything else in this plan touches real training. Read step-time in the logs and extrapolate full-run duration against the 6h timeout; raise `--timeout` if the projection exceeds ~4.5h.
- Fix the preset-wiring bug while in this code: serialize the resolved config from `train.py` to env as JSON (`TRAIN_CONFIG`, `LORA_CONFIG`) and have `train_hf_job.py` consume it instead of hardcoding — makes it structurally impossible for the two entry points to diverge again.
- Add `seed=42` (currently never set anywhere but the data split) — non-negotiable once you're running an A/B, or the arms differ by RNG alone.
- **Do not pre-commit the learning rate.** A `2e-4 → 1e-4` change has real downside (undertraining ~4.6k filtered examples at 1 epoch) and weak supporting evidence (the "1 epoch beat 3" finding is about epoch count, already fixed by `DEFAULT_EPOCHS=1`). Instead, use the existing `--mini-test` (~$0.10, ~25 real optimizer steps) as a **config-sanity gate**: run it at both 2e-4 and 1e-4, require the loss curve to look sane, and pick the LR from that — before spending on two full runs whose combined hyperparameter changes (P1 format fix + P4 optimizer/LR/seed) have never been validated together.

---

## P5 — English instruction to a Hebrew-instruction-tuned model (optional, cheap)

**Diagnosis.** The prompt instruction is in English ("Summarize the following Hebrew text..."), but DictaLM 2.0-instruct was tuned with an *extended Hebrew instruct dataset*. A Hebrew-language instruction is plausibly closer to its tuning distribution. Low confidence — flagging it because it's nearly free to check.

**Fix.** Generate ~50 zero-shot summaries with a Hebrew-phrased instruction vs. the current English one (Gemini path, or a tiny inference-only job) and eyeball both in the existing predictions viewer. Adopt Hebrew only if visibly better; otherwise keep English — it's also what the Gemini baseline uses, which keeps systems comparable.

---

## Execution order

1. **P1** — chat-template the training/inference prompt for the finetuned system. Free, no GPU.
2. **P2** — decode re-test on the existing Hub adapter. ~$1, self-verifying, no retraining.
3. **P3** — re-anchor documentation/eval reporting to faithfulness-first. No cost.
4. **P4** — regime fixes (QLoRA default, paged optimizer, seed, preset-wiring fix) + smoke test on the 7B (~$0.05) + mini-test LR gate (~$0.10).
5. **P5** — optional Hebrew-instruction probe. Near-free.
6. **P0** — Phase 0 grounding report + oracle calibration (local, $0, gate) → Phase 1 filter wiring → the real A/B:
   - **Arm U** (control, `MIN_COVERAGE=0.0`) vs **Arm F** (treatment, `MIN_COVERAGE≈0.6–0.7`), identical seed and hyperparameters (all P1–P4 fixes applied to *both* arms), ~$4–6 total.
   - No free control arm exists — `avreymi/amlk-dictalm2-instruct-sft` isn't on the Hub yet, and all prior recorded numbers are from a different base model, so every comparison must be within this run.

## Success gates (fix now, before seeing numbers)
1. **Faithfulness** (judge, prediction-vs-article): Arm F > Arm U, and Arm F > its own zero-shot base. Primary endpoint.
2. **Hallucination rate** (error_analysis): Arm F < Arm U.
3. **`lead_copying` must not rise** in Arm F. Hard gate — a rise means hallucination was traded for extraction, undermining the lead-bias thesis. Report training-set abstractiveness (novel n-grams) of filtered vs. unfiltered alongside it (TODO B'.4).
4. **ROUGE-1 on high-coverage test buckets**: Arm F > Arm U, reported as a secondary/stratified number only, never as the headline.

## Files touched (when executing)
- `data/grounding.py` *(new)* — coverage metric, `--report`/`--label` CLI
- `data/preprocess.py`, `data/prompts.py` — coverage columns; chat-template-wrapped prompt (+BOS handling)
- `training/config.py` — `model_repo(..., min_coverage)`; qlora default + `paged_adamw_8bit`; seed; preset fix plumbing
- `training/train.py` — serialize `TRAIN_CONFIG`/`LORA_CONFIG`/`MIN_COVERAGE` to env
- `training/train_hf_job.py`, `evaluation/infer.py` — consume serialized config; matched chat-template inference; simplified decode; coverage filter ordered before smoke/mini slices; full filtered val instead of first-200
- `evaluation/stratify_by_topic.py` — add `coverage_bucket` to `--label-field`
- `evaluation/build_report_tables.py` — HeSum-context caveat line, faithfulness-first ordering
- `tests/test_grounding.py` *(new)*, `tests/test_preprocess.py` (coverage column + chat-template prompt)

Every touched file needs its header updated per `AGENTS.md` (role / code flow / execution environment). No new doc files in the repo.

## Sources
- [DictaLM 2.0 Instruct model card](https://huggingface.co/dicta-il/dictalm2.0-instruct) — 7B, `[INST]` format requirement
- [HeSum: a Novel Dataset for Abstractive Text Summarization in Hebrew (arXiv 2406.03897)](https://arxiv.org/abs/2406.03897) — results table, abstractiveness stats, ROUGE/human negative correlation
- [HeSum full text](https://arxiv.org/html/2406.03897v2)
- [Adapting LLMs to Hebrew: DictaLM 2.0 (arXiv 2407.07080)](https://arxiv.org/html/2407.07080v1)
- [TRL chat-template handling, issue #3140](https://github.com/huggingface/trl/issues/3140)
- [SFT dataset formatting guide (Red Hat Developer)](https://developers.redhat.com/articles/2025/08/18/introduction-supervised-fine-tuning-dataset-formats)
- [Repetition penalties in LM generation](https://mbrenndoerfer.com/writing/repetition-penalties-language-model-generation)
- [Improving Faithfulness in Abstractive Summarization (arXiv 2104.09061)](https://arxiv.org/pdf/2104.09061)
- [Entity Coverage Control for faithful summarization (arXiv 2207.02263)](https://arxiv.org/pdf/2207.02263)


update-the-prompt