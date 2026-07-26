# Training Handoff Contract

#status/planned

Training is owned externally. This note is the interface: what we hand over, what constraints travel with it, what we require back, and what happens if some of it does not arrive. It exists because the experiment's validity is decided almost entirely at the handoff boundary, not during training.

The single most important line: **we produce the split, not the training owner.** Everything else is recoverable; a split created independently per arm is not.

## What we hand over

### The frozen split

One split, produced on our side, used unchanged by every arm.

```text
data_curation/artifacts/splits/frozen_split_v1.json
```

```json
{
  "split_version": "v1",
  "created": "2026-07-26",
  "source_artifact_sha256": "...",
  "test":  ["hesum_id", "..."],
  "val":   ["hesum_id", "..."],
  "train": ["hesum_id", "..."]
}
```

Ids only, referencing `hesum_id` in `final_clean_hesum.json`, so the split is small, diffable, and cannot drift from the data it points at. The `source_artifact_sha256` pins which build of the curated dataset the ids were drawn from, mirroring the guard `build_curated_dataset.py` already applies to the model-curation input.

Rules:

- **Test split of 800 to 1,000 rows, carved before any training run.** Not sampled afterwards, not re-sampled per arm.
- **Every test row carries both references** — its original HeSum headline and its curated headline — so a single test set serves every evaluation view in [[Reference Quality Experiment]].
- **No test row appears in any arm's training or validation set.** This is the leakage guard, and it is checkable from the ids alone.
- Test rows must be excluded from the rubric anchor examples in [[Reference Quality Rubric]], or the judge has seen them.

Why ours: if each arm gets its own split, differences between arms are confounded with differences between test sets and no comparison is possible. This is not a preference about tidiness, it is the thing that makes the experiment an experiment.

### The arm definitions

| Arm | Training data | n | Purpose |
|-----|---------------|---|---------|
| **A** | Raw HeSum, subsampled | = size of B | control: same quantity, uncurated |
| **B** | Curated | 5,854 minus held-out rows | treatment |
| C (optional) | Raw HeSum, full | 10,000 minus held-out rows | the naive baseline |

**Size-matching A to B is non-negotiable.** The curated set is smaller than the raw one by construction, so an unmatched comparison confounds data quality with data quantity and answers nothing. Arm A must be a random subsample of raw HeSum at exactly B's training-set size, drawn with a recorded seed.

Arm C is worth having if budget allows, because it represents what someone would actually do with HeSum off the shelf, but it cannot substitute for A.

### Constraints that travel with the handoff

- Identical hyperparameters across arms — learning rate, epochs, LoRA rank and targets, batch size, schedule.
- Identical base model across arms.
- Identical decode configuration at generation time.
- The same frozen split for every arm.
- A recorded random seed for the arm-A subsample.

Any deviation must be reported rather than absorbed silently. A deviation we know about is a caveat; one we discover later invalidates the comparison.

## What we require back

**Preferred: prediction files.** One file per arm over the frozen test split.

```json
{"hesum_id": "1", "prediction": "...", "arm": "A"}
```

Predictions are strongly preferred over checkpoints because they remove any need for us to load a model. This machine has 8 GB and freezes on a Qwen-class load, and `evaluation/infer.py` is documented remote-GPU-only for exactly that reason. If predictions arrive, our entire side of E4 is API and CPU work.

**Acceptable: checkpoints.** Adapter or merged weights plus the base model id. We then run inference ourselves on HF Jobs or the Databricks GPU cluster, never locally.

**Required either way: a manifest.**

```json
{
  "arm": "A",
  "base_model": "...",
  "split_version": "v1",
  "train_n": 0,
  "subsample_seed": 0,
  "hyperparameters": { "epochs": 0, "learning_rate": 0.0, "lora_r": 0 },
  "decode": { "max_new_tokens": 0, "temperature": 0.0 },
  "deviations": []
}
```

The `deviations` field is the important one and should be explicitly empty rather than absent. It is the honest channel for "we had to change something", and an empty list is a positive assertion that the contract held.

## What we run

Everything after the checkpoints or predictions arrive is ours:

- The four-dimension rubric judge from [[Reference Quality Rubric]], the same instrument used on dataset references, so model outputs and references land on one axis.
- AlephBERT BERTScore against both reference versions, plus appendix ROUGE, via `evaluation/evaluate.py`.
- Blind pairwise between arm outputs, with the position-bias check and null calibration described in [[Reference Quality Experiment]].
- Per-stratum readout using the row labels from [[Dataset Defect Taxonomy]], so we can say which defect class curation actually fixed downstream.
- Lead-overlap distribution per arm against its training data's, which is the [[Lead Bias Probe]] question.

**Stated in advance:** arm A scoring higher against original references, and arm B scoring higher against curated references, are both tautological — each arm matches the reference style it was trained on. Neither is evidence of anything. The claim rests entirely on the reference-free blind pairwise and on the rubric scores, which never compare against a reference at all.

## Graceful degradation

Recorded now, before results exist, so the conclusion cannot quietly stretch to fit whatever arrives.

**A and B both delivered.** The full causal claim about curation is available. This is the minimum for the paper's central result.

**B only.** Nothing about curation can be concluded. The deliverable reduces to a descriptive quality report on one model, and the paper's contribution falls back entirely on the dataset audit in E1 to E3 — which does stand alone, since it never depended on training.

**B and C only, no A.** The comparison is confounded by training-set size, because C has 10,000 rows and B has 5,854. Must be reported as a confounded comparison, not as a quality effect. A curated-data win here is uninterpretable: it would be surprising enough to mention, since more data usually helps, but it cannot be attributed to curation.

**Arms trained on different splits.** Not recoverable. The comparison would have to be abandoned and re-run.

## Coverage gap

Nothing in this design tests whether **dropping** rows was correct. Arms A and B differ in both the rows they contain and the headlines on the rows they share, so a B-over-A win cannot be attributed to filtering rather than to headline repair.

Isolating it would need a fourth arm trained on curated headlines *plus* the dropped rows, which separates the repair effect from the filter effect. Deferred unless the training owner has budget. Until then, E1's stratum analysis is the only evidence bearing on the filtering decision, and it is indirect.

Related: [[Reference Quality Experiment]], [[Data Curation Pipeline]], [[Reference Quality Rubric]], [[Paper Figures]], [[Lead Bias Probe]]
