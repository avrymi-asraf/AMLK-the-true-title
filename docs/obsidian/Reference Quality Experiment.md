# Reference Quality Experiment

#status/planned

Four experiments establishing that the curation in [[Data Curation Pipeline]] was justified and that it
helped. Full pre-registered detail, including every test and threshold, is in
`docs/superpowers/specs/2026-07-26-dataset-review-experimental-design.md` — this note is the readable map.

**The organizing idea:** each experiment audits a different *action* the pipeline took. That framing is
also the paper's structure.

| Experiment | Question | Curation action audited |
|------------|----------|-------------------------|
| **E1** | Are the flagged rows genuinely worse? | the diagnosis |
| **E2** | Are the rewritten headlines better, and by how much? | the repair, graded |
| **E3** | Are they preferred head to head? | the repair, forced choice |
| **E4** | Does a model trained on curated data do better? | the whole intervention |

E1 to E3 are ours. E4 straddles the boundary in [[Training Handoff Contract]].

## The instrument

Everything rests on the rubric judge in [[Reference Quality Rubric]]: it reads the **article** and scores a
headline on four ordinal dimensions — faithfulness, single-focus, informativeness, cleanliness.

Why not a metric: string similarity needs two headlines, and we have one plus the article. An earlier
design generated a probe headline and inferred reference quality from the agreement gap, which added a
model's worth of confound to every number. A judge that reads the source measures the thing directly.

Because the same instrument also scores model outputs in E4, dataset quality and model quality land on one
axis — the reason a single figure can carry both.

## E1 — Was the diagnosis right?

Judge scores the **original** headline of all 10,000 rows. Compare across the strata in
[[Dataset Defect Taxonomy#Analysis strata]].

**H0:** each sub-score's distribution is independent of stratum.

The logic: one model, one prompt, one rubric across every stratum. Only the reference changes. A
systematic drop in a stratum implicates the reference rather than the instrument.

Tests: Mann-Whitney U against S0 with Holm correction, **Cliff's delta as the conclusion-carrying
statistic**, bootstrap CIs on medians, and ordinal logistic regression as the primary analysis. At n in the
thousands every p-value will be tiny, so effect size decides — the regression is what separates
overlapping strata and holds article length fixed.

No Kruskal-Wallis omnibus: it assumes independent groups and our strata overlap.

**Predicted signatures** are written down in advance in [[Reference Quality Rubric#Why the dimensions are
split this way]]. A defect damaging an unpredicted dimension is a finding; every defect damaging everything
equally means the rubric is not discriminating.

**Length and lead bias** ride along here — sub-scores and headline-to-lead overlap against
`log(article_tokens)` across all 10,000 rows, including those the filters removed. See [[Lead Bias Probe]].

## E2 — Was the repair right, and by how much?

The **3,069 rewritten rows** only; on the 2,785 kept rows both headlines are the same string.

Judge scores the curated headline too, then Wilcoxon signed-rank on the paired per-row differences, per
dimension, broken out by edit sub-type.

Gives magnitude and mechanism: *which* dimension curation actually improved. Also report the share of rows
where a score went **down** — curation degrading a reference is worth catching rather than averaging away.

**Known weakness:** ceiling compression. If both headlines score 4, a real preference is invisible. That is
precisely what E3 exists to catch.

## E3 — Was the repair right, as a forced choice?

Same rows, sampled to ~1,000. Judge sees the article and both headlines in randomized order and picks one
or calls a tie, never told which is original.

Forced choice catches consistent small preferences that graded scores compress, and randomized order makes
it the more adversarial test of the confound that our curated references were LLM-written.

Four bias controls, which are what make the number credible rather than asserted:

- **Position bias** — subsample run in both orders, flip rate reported
- **Null calibration** — the judge's bias floor, measured by asking it to choose between effectively
  identical options
- **Placebo population** — the 2,785 kept rows should land near 50%; deviation is instrument bias and gets
  subtracted
- **Family separation** — judge from neither `gpt-5.6-luna`'s family nor the training base model's

Plus **human validation**: ~150 rows, both team members, independent and blind, same rubric. Cohen's kappa
for human-human and judge-human. One round validates E1's instrument and E3's judge together.

**If E2 and E3 disagree** — rubric says better, head-to-head says chance — report it as a finding. Do not
pick the more favourable result.

## E4 — Did the intervention pay off?

Training external. See [[Training Handoff Contract]] for the split, arms, size-matching requirement, and
what we need back. We run the rubric judge, BERTScore, blind pairwise between arms, per-stratum readout,
and the lead-overlap comparison.

**Pre-registered non-claims:** arm A winning against original references and arm B winning against curated
references are both tautological — each matches the reference style it trained on. The claim rests only on
the reference-free rubric scores and the blind pairwise.

## Sequencing

Steps 1 to 4 need nothing from the training owner and should not wait:

1. Build the unified row-label artifact — pure joins over existing files, no API cost
2. Draw F1 and F2 ([[Paper Figures]]) — no API cost
3. Draft and review the Hebrew rubric anchors
4. **Pilot the rubric on a few hundred rows** — the last cheap moment to revise it
5. Full E1 pass → F3, F4, F5
6. E2 and E3 → F6, F7
7. Human validation, parallel with 5 → F9
8. E4 when arms arrive → F8

## What could undermine this

The full list is in the spec's threats-to-validity section. The three that most deserve attention:

- **Distribution shift, not just quality.** Curation changed what the corpus is *about* — shorter articles,
  fewer roundups — not only how clean it is. An E4 win may partly reflect an easier task distribution.
- **The drops are untested.** Nothing isolates whether removing 632 unusable and 3,514 filtered rows was
  right. It would need a fourth training arm.
- **Human validation is small and not independent.** 150 rows annotated by the two people who designed the
  rubric. Report the limitation rather than the reassurance.

Related: [[Reference Quality Rubric]], [[Dataset Defect Taxonomy]], [[Training Handoff Contract]], [[Paper Figures]], [[Data Curation Pipeline]], [[Lead Bias Probe]]
