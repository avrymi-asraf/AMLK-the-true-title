# Dataset Defect Taxonomy

#status/done

What is wrong with HeSum rows, how it was labeled, and which groups can carry a statistical claim. The labeling itself is described in [[Data Curation Pipeline]]; this note is about the categories and their analytical use.

Two axes exist in principle. Only the first is labeled.

- **Source axis** — is the article usable as a training source? Six labels, applied by `gpt-5.6-luna` to 6,486 rows.
- **Headline axis** — is the headline usable as a training target? Decided for 5,854 rows, but recorded only as keep-or-replace with no reason.

## Source labels, verbatim

The definitions below are the exact prompt text from `data_curation/model_curation/source_filter/filter_prompt_schema.py`, reproduced because the label boundaries *are* the measurement instrument and paraphrasing them would change what the counts mean.

**usable** — 5,854 rows (90.3%)
> the supplied text is coherent, self-contained, primarily textual, has one clearly dominant central focus, and contains enough substantive information to serve as a useful training source. It may include supporting examples, reactions, background, consequences, comparisons, minor subtopics, links, credits, calls to action, media references, formatting noise, or boilerplate when the essential content is present and the peripheral material can be safely ignored.

**unusable_multiple_independent_items** — 569 rows (8.8%)
> the supplied text contains two or more independently developed items rather than one clearly dominant subject. Use this when substantial parts of the text could reasonably stand as separate articles, including press reviews, roundups, or digests, even when the items share a broad person, category, publication, or theme.
> Do not use this when the text has one central focus and the additional material only supports, explains, illustrates, or develops that focus.

**unusable_substantive_content_not_in_text** — 27 rows (0.4%)
> the supplied text mainly points to, introduces, embeds, or surrounds substantive content that is not included in the supplied text. Use this when understanding the central content requires an unavailable video, audio recording, image, gallery, PDF, external page, embedded post, interactive element, or linked resource.
> Do not use this merely because the text contains a URL, media link, podcast link, download link, embed marker, credit, or call to action. If the supplied text itself contains a substantive article body, transcript, interview, report, or developed explanation, evaluate that text normally.

**unusable_damaged_or_fragmentary_text** — 4 rows (0.1%)
> the supplied text is severely corrupted, unintelligible, fragmentary, truncated, or assembled from broken or mismatched fragments in a way that prevents reliable understanding of its main content.
> Do not use this merely because the writing is imperfect, formatting is noisy, some sentences are awkward, minor details are missing, or the text contains ellipses.

**unusable_insufficient_substantive_content** — 32 rows (0.5%)
> the supplied text is coherent, self-contained, primarily textual, and focused on one topic, but it does not contain enough meaningful information, development, explanation, evidence, reporting, or argument to serve as a useful training source. Use this when the supplied text appears to be the full available text, but is too thin to support an informative headline.
> Do not use this merely because the text is short, introductory, or written as a notice. A short or introductory text is usable when it clearly identifies a meaningful subject, event, claim, investigation, argument, or development and provides enough context to support an informative headline.

**unusable_other** — 0 rows (0.0%)
> the supplied text is clearly unsuitable as a training source for a distinct reason not covered by the other labels.

Percentages are of the 6,486 rows that reached the model, not of the full 10,000.

## What the taxonomy gets right

Three properties make these labels trustworthy enough to build on.

The categories are **mutually exclusive with explicit negative guards**. Every unusable label carries a "Do not use this when..." clause that fences it off from its nearest neighbour. This is the single biggest driver of consistency in LLM labeling, because the failure mode of a bare label list is boundary drift, and the guards target exactly that.

**`unusable_other` came back empty.** A catch-all that attracts zero rows across 6,486 attempts is empirical evidence that the five substantive categories are exhaustive over this corpus. Worth reporting as a validation result rather than a footnote — it is the closest thing we have to a completeness check on the taxonomy.

The labels are about the **article**, judged only on the supplied text, with an explicit instruction not to infer missing content from URLs or outside knowledge. That keeps the judgement reproducible.

## What it gets wrong

**It is a usability taxonomy, not a defect taxonomy.** Every label answers "should this row be dropped?" For a dataset review the more valuable question is "what is wrong with this row?", because prevalence of defects is a claim about the resource whereas a drop count is a claim about our pipeline. The distinction shows up concretely: `unusable_multiple_independent_items` conflates a press review, a topical roundup, and a two-story article, which are different phenomena with different implications for a model trained on them.

**The headline axis is unlabeled.** Stage 7 knows why each of the 3,069 headlines was unfit — the reasons are enumerated at length in `refine_prompt_schema.py`, covering boilerplate, site labels, pipe fragments, duplication, vagueness, teasers, wrong focus, and truncation — but the artifact records only `null` or a replacement string. The largest single repair in the pipeline is therefore unexplained per row. Recoverable from the `(original, replacement)` diff at low cost, since it needs two short strings rather than the article.

**Extreme imbalance makes four of six labels untestable.** At 569, 32, 27, 4, and 0 rows, only `multiple_independent_items` can support a distributional test. The rest are prevalence findings. Collapse them to a single `other_unusable` group of 63 rows for reporting and exclude them from statistics.

**The axis most relevant to the original research question is absent.** Nothing labels headline-to-lead overlap, which is the direct measure of lead bias and computable per row for free. It should be a covariate on every row, not a category. See [[Lead Bias Probe]].

**No reliability estimate.** The labels are a single pass from one model with no re-label, no second model, and no human check. For a paper whose contribution *is* the labels, that is the gap a reviewer will press on. The 150-row human validation set described in [[Reference Quality Rubric]] partly covers this, but it validates the quality rubric rather than these six categories.

## Analysis strata

The groups used for testing in [[Reference Quality Experiment]]. Strata **overlap** by design — a row can be both a pipe digest and a roundup — which is why the analysis uses separate binary indicators in one regression rather than one categorical variable.

| Stratum | n | Source | Role |
|---------|---|--------|------|
| S0 `clean` | 2,785 | passed every filter, headline kept | reference group |
| S2 `multi_pipe_headline` | 2,412 | deterministic, 2+ pipes | testable |
| S3 `multiple_independent_items` | 569 | LLM label | testable |
| S4 `headline_rewritten` | 3,069 | derived from stage 7 | testable, sub-typed by edit |
| S5 `other_unusable` | 63 | LLM labels, collapsed | prevalence only |

Every stratum is recoverable by joining artifacts that already exist. No new labeling pass is needed. The numbering skips S1 deliberately, for the reason below.

## Why article length is a covariate, not a stratum

The obvious fifth stratum — `over_token_budget`, 2,659 rows — is **not usable as a stratum**, and this is the most important methodological point in the note.

The filter is a hard cut at 4,000 tokens. Every row in S0 lies below the threshold and every over-budget row lies above it. The treatment and the confounder are the same variable, so the two groups share **no common support** in article length.

The consequences are concrete. Caliper matching is impossible, because no clean row is length-comparable to an over-budget one. A regression can only identify the effect by extrapolating the length curve across the threshold, which means the estimate rests entirely on an assumed functional form rather than on data — the weakest identification available. And any raw difference between the groups is fully explained by length alone, since longer articles score lower against a short headline no matter how good that headline is.

The honest reading is that `over_token_budget` is an **arbitrary cut on a continuous variable**, not a defect class. Nothing is wrong with a 5,000-token article as such.

So model length continuously over all 10,000 rows and ask the better question: does reference quality decline as articles get longer, and does headline-to-lead overlap rise? That is the lead-bias question, recovered in a stronger form than the original training-variant probe, and it uses the rows the filter discarded rather than throwing away the informative tail.

By contrast **S2 survives as a genuine stratum**, because pipe count is not an arbitrary cut in the same way: a three-segment digest is a qualitatively different object from a single headline, not the same object further along a scale.

## Headline edit sub-types

Proposed sub-typing of the 3,069 rewrites, derived from the diff rather than from a new labeling pass. Assign from the `(original, replacement)` pair:

- `pipes_removed` — original contained pipes, replacement does not
- `boilerplate_stripped` — replacement is close to a substring of the original after removing a site label, credit, or category tag
- `truncation_repaired` — original ends mid-clause or is visibly cut
- `light_edit` — high token overlap, small local change
- `full_rewrite` — low token overlap with the original

The distinction that matters for [[Reference Quality Experiment]] is `full_rewrite` against everything else, since a full rewrite means the original headline carried little usable signal while the rest mean it was salvageable. If curation's benefit concentrates in the full-rewrite group, the defect is severe but rare; if it spreads evenly, the corpus is uniformly noisy. Those are different papers.

Related: [[Data Curation Pipeline]], [[Reference Quality Experiment]], [[Reference Quality Rubric]], [[Lead Bias Probe]], [[HeSum Paper Insights]]
