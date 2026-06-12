# Post-mortem: QLoRA training job `avreymi/6a2bc974822d86c524179991` (2026-06-12)

Engineering post-mortem of the first full QLoRA fine-tuning run of Qwen3-2B for AMLK
(Hebrew news summarization). Covers what happened, why the job got stuck, whether the
hardware/method choices were right, a hyperparameter audit of the training itself, what
it cost, and what to change before the three truncation-probe runs due 30.06. The appendix contains paper-ready compute and
methodology notes for the experimental-setup and limitations sections.

---

## 1. TL;DR

- **Training itself succeeded.** 500 steps / 1 epoch over 8,000 examples in 3h54m;
  train_loss 1.705, eval_loss 1.777. The LoRA adapter was pushed to
  `avreymi/amlk-qwen3-2b-sft` and is safe.
- **The job then got stuck for ~2 hours** in an unbatched, log-less prediction loop that
  needed ~4h for 2,000 generations but had less than 2h of budget left. It was killed at
  the 6h timeout (status CANCELED) having pushed **zero** predictions, because the script
  only uploads results after *all* generation finishes.
- **Cost:** ~$9.00 for this job (6h × $1.50/h on a10g-large), of which ~$3 bought nothing.
  Total spend across all of today's jobs ≈ $10.
- **Four takeaways:** (1) a10g-large and a10g-small have the *same 24 GB GPU* — we paid
  1.5× for vCPUs/RAM a GPU-bound job doesn't use; (2) QLoRA solves a memory problem a 2B
  model on 24 GB doesn't have — plain bf16 LoRA would be faster; (3) any long loop in a
  cloud job needs batching, progress logging, and incremental result pushes; (4) the LoRA
  adapter only attached to **6 of the model's 24 layers** — Qwen3-2B is a hybrid-attention
  architecture and our `q/k/v/o_proj` target list doesn't exist in its 18 linear-attention
  layers (see §5, the biggest quality lever found in this audit).

## 2. Timeline

All times UTC, reconstructed from `hf jobs inspect/logs` and the wandb run.

| Phase | Time | Duration | Evidence |
|---|---|---|---|
| Job created / container started | 08:55:16 / 08:55:25 | — | `hf jobs inspect` |
| Environment setup: uv installs 88 packages (torch 507 MB, CUDA libs >1.5 GB) + model download (~950 MB) + dataset download | 08:55 → ~09:05 | ~10 min | install log |
| Training: 500 optimizer steps, evals at steps 200/400/500 | ~09:05 → ~13:00 | 3h54m (`train_runtime` 14,040s) | trainer log |
| Adapter + tokenizer pushed to Hub | ~13:00 | ~1 min | upload log |
| **Stuck:** "Generating fine-tuned predictions..." then silence — unbatched greedy generation, one example at a time, no progress output | ~13:00 → ~14:55 | ~2h | log ends here |
| Killed at the 6h timeout | ~14:55 | — | status CANCELED |

Training health (from the step logs): loss fell 2.13 → ~1.65 within the first half-epoch
and plateaued; eval_loss went 1.800 (step 200) → 1.778 (step 400) → 1.777 (step 500);
eval token accuracy ~0.599. Healthy curve, no overfitting, one epoch was enough — in
fact the model was essentially converged by epoch ~0.5.

## 3. Root cause of the stuck phase

The original `generate_predictions` loop had three compounding problems:

1. **No batching.** One example per `model.generate()` call, ~7–8s each. Two systems
   (fine-tuned + zero-shot base via `disable_adapter()`) × 1,000 test examples = 2,000
   generations ≈ **4 hours**. Only ~2h of the 6h budget remained — it could never finish.
2. **No progress logging.** The loop printed nothing, so from the outside the job was
   indistinguishable from a hang. We burned wall-clock time diagnosing it.
3. **All-or-nothing upload.** Predictions were written and pushed only after *both*
   loops completed. When the timeout hit, ~2h of completed generations were discarded.
   (A secondary factor: `use_cache` had been disabled for gradient checkpointing and was
   never re-enabled, slowing every decode step.)

**The fix (already deployed):** `train_hf_job.py` now batches generation (batch 8,
left-padding), re-enables the KV cache, logs progress, and gained an `INFERENCE_ONLY=1`
mode that loads the pushed adapter and regenerates predictions on a cheap a10g-small
(`python -m training.train --submit-hf --inference-only`). A rerun
(`6a2c2088871c005b5352b4ac`) was in flight at the time of writing.

Two residual flaws remain in the deployed fix and should be patched before the probe runs:

- **Files are still pushed only at the very end.** Push each `predictions-*.jsonl`
  immediately after its generation loop (and ideally every ~200 rows) so a timeout never
  destroys finished work.
- **The progress print fires only every 200 examples**, not 50 as intended: with
  batch 8, `end` is always a multiple of 8, and `end % 50 == 0` is only true at
  multiples of 200. Use `if (i // batch_size) % 10 == 0` instead.
- **The inference-only job inherited a 30-minute timeout.** Setup is ~4 min and 2,000
  batched generations take ~25–35 min, so this is borderline. Use `1h` — on a10g-small
  the worst case is $1.

## 4. Were the choices right?

### Hardware: a10g-large — wrong flavor, right GPU class

Per the [HF Jobs pricing table](https://huggingface.co/docs/hub/jobs-pricing),
**a10g-small and a10g-large have the identical GPU (A10G, 24 GB)**. The large differs
only in vCPUs (12 vs 4) and RAM (46 vs 15 GB) and costs $1.50/h vs $1.00/h. This
workload is GPU-bound (the tokenize/collate work for batch 2 × seq 2048 is light), so
the same training run on a10g-small would cost ~$3.90 instead of ~$5.85 for the
training phase — a 33% cut for likely zero slowdown. The 15 GB of system RAM is ample
for a 2B model (~4 GB of weights).

Alternatives considered: 1×L4 ($0.80/h, 24 GB) is cheaper but a slower chip — probably a
wash on cost and worse on wall-clock. A100-large ($2.50/h, 80 GB) is ~2–2.5× faster, so
roughly cost-neutral while halving turnaround — worth it when iterating against the
14.06 deadline, not needed for unattended overnight runs.

### Method: QLoRA — solved a problem we don't have

QLoRA's purpose is squeezing a big model into a small GPU. Qwen3-2B in bf16 is ~4 GB of
weights; with LoRA (adapter params only in the optimizer), batch 2 × 2048 and gradient
checkpointing, plain **bf16 LoRA fits in 24 GB with lots of headroom**. The nf4
quantization instead *costs* us: bitsandbytes dequantizes weights on every forward, a
typical 20–40% per-step slowdown, and 4-bit weights are a (mild) quality risk the paper
would have to caveat. Switching `--method lora` (already supported by `train.py`)
should cut training to roughly 2.5–3h with equal-or-better quality. QLoRA remains the
right default only if we later scale to a 7–8B base model.

### Missing fast-path dependencies

The job logs contain:

> `[transformers] The fast path is not available because one of the required library is
> not installed. Falling back to torch implementation.` (pointing at
> `flash-linear-attention` and `causal-conv1d`)

Qwen3-2B's hybrid attention layers have an optimized kernel path that we silently did
not use, in either training or generation. Adding `flash-linear-attention` and
`causal-conv1d` to the PEP 723 dependency block in `train_hf_job.py` is a one-line
change with a potentially large speedup on both phases. **Action: benchmark this in the
next mini-test before the probe runs.**

### Training length

eval_loss improved only 0.023 after epoch 0.4 (1.800 → 1.777). Capping at ~300 steps
would save ~1.5h (~$1.5–2.3) per run at negligible quality cost. For the paper, one
clean full epoch is easier to report; for the three probe variants, consider
`--max-steps 300` if budget gets tight. **Caveat:** §5.2 argues this plateau reflects
the adapter's tiny capacity — re-check the curve after extending LoRA coverage (§5.1)
before locking in a shorter schedule.

### Base model choice (research-level)

Qwen3-2B remains a defensible choice: solid multilingual coverage including Hebrew, a
size that trains in hours on a single 24 GB card, and an open license. The healthy loss
curve (token accuracy ~60% on free-form Hebrew summaries) confirms it learns the task.
Hebrew-specialized alternatives (e.g. DictaLM-family) are larger (7B), which would force
actual QLoRA and multiply cost; mT5-class seq2seq models are older and weaker. No change
recommended — but the paper should note the model was not Hebrew-specialized and cite
this as a possible ceiling.

## 5. The training itself: hyperparameter audit

This section audits the training configuration against what the run's own artifacts
(wandb curves, the pushed adapter, the base model's config) reveal, plus standard
fine-tuning literature. Configuration used:

| Hyperparameter | Value | Verdict |
|---|---|---|
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj` | **Wrong for this architecture — see 5.1** |
| LoRA rank / α / dropout | 16 / 32 / 0.05 | Fine (α/r = 2 is standard); dropout pointless in 1 epoch |
| Learning rate / schedule | 2e-4, cosine → 0, 5% warmup | Standard LoRA recipe, fine |
| Effective batch | 16 (per-device 2 × accum 8) | Fine; bf16 LoRA preset already moves to 4 × 4 |
| Epochs / steps | 1 epoch = 500 steps | Enough *at current adapter capacity* — see 5.2 |
| Max sequence length | 2,048 (articles cut to 1,792) | Acceptable; blurs the whole-vs-lead probe contrast — see 5.4 |
| Generation cap | `max_new_tokens=128`, greedy | Truncates ~9% of summaries — see 5.3 |

### 5.1 The headline finding: LoRA covered only 6 of 24 layers

Qwen3-2B is **not a standard transformer**. Its config
(`model_type: qwen3_5`) shows a hybrid-attention text model: **24 layers = 18
linear-attention (Gated DeltaNet) + 6 full-attention** (`full_attention_interval: 4`),
plus a vision tower we don't use. The safetensors weight map confirms the module names:

- Only the **6 full-attention layers** have `self_attn.{q,k,v,o}_proj` — the modules our
  LoRA config targets.
- The **18 linear-attention layers** use `linear_attn.{in_proj_qkv, in_proj_z, in_proj_a,
  in_proj_b, out_proj}` plus a conv1d — **none matched, zero adapters attached**.
- All 24 layers have `mlp.{gate,up,down}_proj` — also not targeted.

The smoking gun is the pushed adapter itself: `adapter_model.safetensors` is **2.96 MB ≈
1.48 M trainable parameters, 0.07% of the model** (typical LoRA setups train 0.5–2%).
We fine-tuned a quarter of the network's depth and none of its MLPs. The QLoRA paper
(Dettmers et al., 2023) found that *which* layers LoRA covers matters far more than rank —
attaching to all linear layers was necessary to match full fine-tuning, while rank had
little effect.

**Fix:** extend `target_modules` in `training/config.py` and `train_hf_job.py` to the
hybrid architecture's actual modules:

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",          # 6 full-attention layers
    "in_proj_qkv", "in_proj_z", "out_proj",          # 18 linear-attention layers
    "gate_proj", "up_proj", "down_proj",             # MLP, all 24 layers
]
```

That is ≈ 15.6 M trainable params (0.8% of the model, ~31 MB adapter) — a 10× capacity
increase for negligible compute/memory cost. Avoid PEFT's `"all-linear"` shortcut here:
on this multimodal checkpoint it could also attach adapters inside the vision tower and
the multi-token-prediction (`mtp`) head. (Note `in_proj_a`/`in_proj_b` are tiny
decay/gate projections; including them is harmless but optional.)

### 5.2 The loss plateau is probably a capacity ceiling, not a data ceiling

From the wandb history (run `0detapav`): train loss drops 2.13 → ~1.70 within the first
~100 steps, then oscillates in a flat 1.55–1.77 band for the remaining 80% of the epoch;
eval_loss moves only 1.800 → 1.777 from step 200 to 500; grad norms stay healthy
(0.8–1.0, no spikes). With 1.48 M trainable parameters this is exactly what a saturated
adapter looks like: the model learned the *format* of the task quickly and then had no
capacity left to learn more. Conclusions:

- **Don't add epochs or data at the current configuration** — the curve says they buy
  nothing.
- **Re-measure after the 5.1 fix.** With 10× the capacity the plateau may sit lower and
  arrive later; training length should be re-decided from that curve, not this one.
- LR 2e-4 can stay as-is for the wider adapter; if the new curve is unstable in the
  first 50 steps, drop to 1e-4 — but there is no evidence of instability today.

### 5.3 Generation length cap clips ~9% of summaries

Tokenizing the 1,000 test references with the Qwen3-2B tokenizer: median 73 tokens,
p90 = 126, p95 = 151, p99 = 187 — **9.1% of reference summaries are longer than the
`max_new_tokens=128` cap** used in `train_hf_job.py`'s generation loop. A model that
learned to match reference length gets its longest summaries cut mid-sentence, which
depresses ROUGE/BERTScore recall and will surface as fake "omission" errors in the error
analysis. **Fix: `max_new_tokens=256`** — covers p99 with margin and adds only ~1–2 min
to the batched inference pass.

### 5.4 Smaller dials, and what *not* to bother with

- **Article truncation to 1,792 tokens** (median HeSum article ~2,500) means the trained
  "whole" variant actually sees "the first ~70% of the article". This is fine for the
  main result but **blurs the whole-vs-lead contrast in the truncation probe** — the
  paper must state the effective input window, and the probe's lead-only variant should
  be defined well under 1,792 tokens so the contrast is real.
- **LoRA dropout 0.05 → 0.** In a single epoch every example is seen once; overfitting
  is impossible and dropout only adds gradient noise. Marginal, but free.
- **Rank:** leave at 16. Raising rank without fixing module coverage (5.1) is spending
  on the wrong axis; after 5.1, capacity is no longer the binding constraint.
- **NEFTune** (`neftune_noise_alpha=5` in `SFTConfig`) is a one-line TRL option with
  reported instruction-tuning gains; worth one cheap A/B at mini-test scale, not a
  blocker.
- **Not worth it now:** rsLoRA/DoRA variants (second-order gains at best), packing
  (interacts awkwardly with `completion_only_loss`), bigger effective batch (throughput,
  not quality), label smoothing.

### 5.5 How to validate cheaply

The mini-test ladder makes hyperparameter changes nearly free (~$0.30 / 17 min each on
a10g-small). Before the probe runs, A/B at mini scale: **(a)** current vs extended
target modules (expect a visibly lower eval_loss for the same steps), **(b)** optionally
LoRA dropout 0 and NEFTune. Promote the winner to one full run. Caveat: eval_loss is a
proxy — confirm the winner with ROUGE on its generated predictions before committing all
three probe variants to it.

## 6. Cost analysis

Actual spend on 2026-06-12 (billed per minute, only while Starting/Running):

| Job | Purpose | Flavor | Runtime | Rate | Cost |
|---|---|---|---|---|---|
| `6a2bc5af`, `6a2bc623` | failed first submissions (env bugs) | a10g-small | 50s + 3m38s | $1.00/h | ~$0.08 |
| `6a2bc755` | smoke test (10 steps) | a10g-small | 7m31s | $1.00/h | ~$0.13 |
| `6a2bcd88` | mini test (80 ex / 5 epochs) | a10g-small | 17m21s | $1.00/h | ~$0.29 |
| **`6a2bc974`** | **full QLoRA run** | **a10g-large** | **~6h (timeout)** | **$1.50/h** | **~$9.00** |
| `6a2c2088` | inference-only rerun | a10g-small | ~30m (est.) | $1.00/h | ~$0.50 |
| | | | | **Total** | **~$10.00** |

Of the $9.00 main job: ~$5.85 training (useful), ~$0.25 setup (unavoidable), ~$2.90
stuck inference (wasted). The smoke/mini-test discipline worked — it caught four
classes of bug for ~$0.50 before the expensive run.

Counterfactual optimized run (same outputs):

| Decision | Cost effect |
|---|---|
| a10g-small instead of a10g-large | training $5.85 → $3.90 |
| bf16 LoRA instead of QLoRA (~30% faster steps) | $3.90 → ~$2.70 |
| fast-path kernels installed | further reduction, unbenchmarked |
| batched inference in-job (~30 min) | +$0.50, instead of $2.90 wasted + $0.50 rerun |
| **Optimized total** | **≈ $3.50–4.50 vs ≈ $9.50 actually spent on full-run + rerun** |

Projected for the remaining work: 3 probe variants (whole is done; lead + body remain,
plus any rerun) at the optimized configuration ≈ **$3–5 each** instead of ~$9.5.

## 7. Recommendations for the truncation-probe runs (due 30.06)

Checklist, in priority order:

1. **Extend LoRA `target_modules` to the hybrid architecture** (§5.1) — the single
   biggest expected quality improvement, at zero extra cost. Validate with a mini-test A/B.
2. **Switch full runs to a10g-small** (`flavor` in `submit_hf_job`); keep a10g-large
   only if a mini-test shows data loading bottlenecking the GPU.
3. **Use `--method lora` (bf16) instead of qlora** for the 2B model. Verify with one
   mini-test that loss curves match.
4. **Add `flash-linear-attention` + `causal-conv1d`** to `train_hf_job.py` dependencies;
   benchmark the speedup in the same mini-test.
5. **Raise `max_new_tokens` to 256** in the generation loop (§5.3).
6. **Push predictions incrementally** — upload each `predictions-*.jsonl` right after
   its loop (or every ~200 rows), never only at script end.
7. **Fix the progress-print condition** (`(i // batch_size) % 10 == 0`).
8. **Set timeouts with margin:** training jobs `4h` (expected ~2.5–3h), inference-only
   jobs `1h`. Tighter timeouts than 6h also bound the cost of any future hang.
9. **Keep the smoke → mini → full ladder.** It cost $0.50 today and saved a failed $9 run.
10. Optional: if wall-clock matters more than elegance near a deadline, run the three
    probe variants in parallel on three a10g-smalls — same total cost, one-third the time.

## Appendix: paper-ready compute & methodology notes

**Model & adaptation.** Qwen/Qwen3-2B (hybrid-attention architecture: 24 text layers =
18 linear-attention Gated DeltaNet + 6 full-attention; multimodal checkpoint, text-only
use), 4-bit NF4 quantization with double quantization (bitsandbytes), bf16 compute.
LoRA: r=16, α=32, dropout 0.05, target modules q/k/v/o — which exist only in the 6
full-attention layers, so 1.48 M trainable parameters (0.07% of the model, 2.96 MB
adapter); the 18 linear-attention layers and all MLPs were not adapted.

**Training.** TRL `SFTTrainer`, completion-only loss (loss on summary tokens only),
max sequence length 2,048 (articles truncated to 1,792 tokens so the reference summary
always survives). Effective batch 16 (per-device 2 × grad-accum 8), LR 2e-4, cosine
schedule, 5% warmup, 1 epoch = 500 optimizer steps over 8,000 examples
(13.67 M completion tokens seen).

**Results of the run.** Final train_loss 1.705; eval_loss 1.800 → 1.778 → 1.777 at
steps 200/400/500 (200-example validation slice); eval token accuracy 0.599;
eval entropy 1.795.

**Compute.** Single NVIDIA A10G (24 GB), HuggingFace Jobs. Training: 3.9 GPU-hours
(0.57 samples/s). Total including environment setup, evaluation passes, and test-set
inference: ≈ 6.5 A10G GPU-hours ≈ $10 at on-demand rates. Test-set inference (1,000
greedy generations × 2 systems, 128 max new tokens, batch 8) ≈ 0.5 GPU-hours.

**Limitations to note.** (1) The base model is multilingual, not Hebrew-specialized;
(2) 4-bit quantized fine-tuning may slightly degrade quality vs bf16 LoRA (unmeasured
here); (3) the zero-shot baseline shares the (quantized) base weights with the
fine-tuned system; (4) the LLM judge (Gemini) is the same model family as the advanced
baseline — self-preference bias possible; (5) the adapter covered only the 6
full-attention layers (0.07% trainable parameters) — reported scores are a lower bound
on what this base model can reach with full-coverage LoRA; (6) inference capped
generation at 128 new tokens while 9.1% of reference summaries are longer (p95 = 151
tokens), depressing recall-oriented scores for the longest summaries.

---

*Sources: `hf jobs inspect/logs` for jobs listed above; wandb project
`amlk-hebrew-summarization` (run `qlora-whole-hfjob`, id `0detapav`);
[HF Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing); `Qwen/Qwen3-2B`
`config.json` + safetensors weight map (architecture / module names);
`avreymi/amlk-qwen3-2b-sft` `adapter_config.json` + adapter size (coverage evidence);
Qwen3-2B tokenizer over the 1,000 test references (length stats).*
