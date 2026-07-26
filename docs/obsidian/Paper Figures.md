# Paper Figures

#status/planned

The figure set for the paper. One figure carries one claim, and every figure records which experiment produced it, so the paper and the code cannot drift apart.

F1 and F2 need no experiment — they come from artifacts already on disk and should be drawn first, before any judge budget is spent.

## Manifest

| Id | Claim | Source | Chart |
|----|-------|--------|-------|
| **F1** | This is what the curation pipeline did | artifacts only | Sankey funnel |
| **F2** | HeSum contains this much of each defect | artifacts only | horizontal bars |
| **F3** | Defective rows have measurably worse references | E1 | stacked ordinal bars, faceted |
| **F4** | The differences are large enough to matter | E1 | Cliff's delta forest plot |
| **F5** | Reference quality declines with article length | E1 | two-panel line, shared x |
| **F6** | Curation improved specific dimensions, by this much | E2 | dumbbell + transition heatmap |
| **F7** | Curated headlines are preferred head to head | E3 | stacked win bar + placebo |
| **F8** | The model trained on curated data is better | E4 | rubric bars + pairwise |
| **F9** | The judge is a reliable instrument | validation | agreement heatmap |
| **Q1** | What the defects look like | qualitative | Hebrew example cards |

## F1 — Curation funnel

Sankey from 10,000 raw rows through each filter to the 5,854 final ones, terminating in the kept-versus-rewritten headline fork. Counts from [[Data Curation Pipeline]].

The orientation figure. A reader who sees only this should understand the pipeline's shape and that a third of the corpus was removed before any model looked at it. Plotly `go.Sankey`. Buildable today.

Show the two deterministic filters as separate flows converging, not as one combined "filtering" node — the fact that they are independent and only intersected later is part of the story.

## F2 — Defect prevalence

Horizontal bars of stratum size with corpus share annotated, including the small strata so the long tail stays visible. This is the descriptive contribution: a claim about HeSum as a resource, independent of anything we trained.

Keep `other_unusable` broken out into its three constituent labels here even though the analysis collapses them, since prevalence is exactly what those 63 rows can support.

## F3 — Rubric distributions by stratum

The core result. Stacked horizontal bars of the ordinal 1-to-5 distribution, one bar per stratum, faceted into four panels for the four dimensions.

**Stacked ordinal bars rather than box plots or bar-of-means.** The scores are ordinal ([[Reference Quality Rubric#Scores are ordinal]]), so a mean invents a scale that does not exist and a box plot implies quantiles on five discrete levels. The stacked distribution also exposes ceiling effects directly, which is the specific weakness E2 has to worry about — if S0 is 80% fives, E2's paired comparison has very little room to move.

Sequential colorblind-safe palette across the five levels; the scale is ordered and the colours must show it.

## F4 — Effect sizes

Cliff's delta forest plot: one row per stratum-by-dimension, 95% CI whiskers, a vertical rule at zero, and a shaded band across the negligible range at |delta| < 0.147.

This figure is the antidote to p-value theatre. At n in the thousands every comparison in E1 will be statistically significant, so significance carries no information and the forest plot is what tells the reader which differences are real in the sense that matters. The shaded negligible band does the interpretive work visually, without the reader needing to remember the thresholds.

## F5 — Length and lead bias

Two vertically stacked panels sharing an x-axis of log article tokens. Above: the four rubric sub-scores as binned medians with CI ribbons. Below: headline-to-lead overlap on the same axis. A vertical rule marks the 4,000-token filter threshold.

Does double duty. It justifies demoting `over_token_budget` from a stratum to a covariate ([[Dataset Defect Taxonomy#Why article length is a covariate, not a stratum]]) by showing the effect is continuous with no discontinuity at the arbitrary cut, and it is the [[Lead Bias Probe]] result. Plot the full 10,000 rows including the ones the filter removed — those rows are the informative part of this figure.

## F6 — Paired change from curation

Dumbbell plot of median original against median curated per dimension, over the 3,069 rewritten rows, with the Wilcoxon result annotated. Beside it, a transition heatmap of original score by curated score for whichever dimension moves most.

The dumbbell gives magnitude and direction per dimension. The heatmap answers the question the dumbbell cannot: which rows moved, and did anything get **worse**? Off-diagonal mass below the diagonal would mean curation degraded some references, which is a finding we would want to catch rather than average away.

## F7 — Head-to-head win rate

Stacked horizontal bar of curated-wins / tie / original-wins with bootstrap CI, and the **null-calibration placebo bar drawn immediately beneath in the same figure**.

Putting the placebo in the same frame is what makes the headline number credible rather than merely asserted. A reader sees the real comparison and the instrument's bias floor at once and can judge the gap themselves. Break out by headline edit sub-type if the sub-typing lands.

## F8 — Before and after

Arm A against arm B on the **identical rubric axis as F3**, so dataset quality and model quality are visually comparable — the payoff for using one instrument throughout. Beside it, the arm-versus-arm blind pairwise under the same placebo treatment as F7.

Do not put the against-original-references and against-curated-references scores in this figure without labelling them as tautological. Per [[Training Handoff Contract]], each arm matches the reference style it trained on, so those bars look like a result and are not one.

## F9 — Judge validation, appendix

Agreement heatmap of judge against human on the 150-row validation set, plus judge test-retest, annotated with quadratically weighted kappa per dimension. Include human-human agreement: if two annotators disagree on a dimension, that dimension is underspecified and the judge's apparent confidence on it is spurious.

Appendix placement, but not optional. Without it every other figure is discounted.

## Q1 — Qualitative exhibit

Not a plot. A handful of Hebrew example cards showing article snippet, original headline, curated headline, and arm outputs, chosen to illustrate the main defect classes. `evaluation/viewer/app.py` already renders these with RTL support; export a curated selection.

Pick examples that are representative rather than extreme. One pipe digest, one roundup, one media stub, one case where the original headline was fine and curation correctly left it alone — that last one matters for showing the pipeline is not indiscriminate.

## Conventions

**Plotly**, matching the house style already set by `plot_topic_sizes` and `plot_clusters` in `evaluation/topic_clustering.py`.

**Sequential colorblind-safe palette** for ordinal 1-to-5 bands. A categorical rainbow on ordered data misleads.

**Hebrew stays out of axis labels, legends, and tick marks.** Bidirectional text in static vector export is a reliable source of silently reversed strings, and a reversed Hebrew label is the kind of error that survives review because nobody reading the English draft notices. Keep all chart furniture in English and confine Hebrew to Q1's example cards, where the viewer's RTL rendering already works. This sidesteps the problem rather than fighting it with `python-bidi` and a reshaper.

**Export text-preserving vector** — PDF or SVG with real text nodes. `AGENTS.md` already records that the existing presentation SVG is flattened paths and therefore uneditable; do not repeat that.

**Figure functions live in the repo; notebooks stay thin.** Same pattern as `notebooks/cluster_topics_databricks.py`, which clones the repo so it calls tested functions rather than duplicating logic. Each figure gets a stable id, a named generating function, and its claim recorded here.

## Compute placement

Most of this does not want a GPU. Being deliberate about it, rather than defaulting to the cluster because it is available:

**API and CPU only** — the rubric judge for E1 to E3, every statistical test, and all figure rendering. This is the bulk of the work and it runs locally.

**Worth the Databricks GPU** — batch AlephBERT BERTScore, which at roughly 20,000 scorings (10,000 rows against two reference versions) is tolerable on CPU and minutes on a GPU. Also any embedding work if a topic-stratified figure is wanted, which is the existing `evaluation/topic_clustering.py` use case.

**Never local** — loading a base or fine-tuned model. `AGENTS.md` records that this 8 GB machine freezes, and `evaluation/infer.py` is marked remote-GPU-only. If checkpoints arrive instead of predictions, inference goes to HF Jobs or the Databricks GPU cluster.

## Sequencing

The judge dominates both cost and wall-clock, so the order matters:

1. **F1 and F2** from existing artifacts. No API spend, and they are immediately useful for talking about the project.
2. **Pilot the rubric** on a few hundred rows. Check test-retest agreement and that score distributions are not degenerate before committing to the full pass. Revise the rubric here if needed — this is the last cheap moment to do so.
3. **Full E1 judge pass**, then F3 to F5.
4. **E2 and E3** on the 3,069 rewritten rows, then F6 and F7.
5. **F9** from the validation set, which can run in parallel with step 3 since the human annotation is independent.
6. **F8** once arms arrive from the training owner.

Related: [[Reference Quality Experiment]], [[Reference Quality Rubric]], [[Dataset Defect Taxonomy]], [[Training Handoff Contract]], [[Data Curation Pipeline]]
