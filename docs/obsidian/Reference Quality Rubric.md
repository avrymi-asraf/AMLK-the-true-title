# Reference Quality Rubric

#status/planned

The measurement instrument for the whole dataset review. Every result in [[Reference Quality Experiment]] and every figure in [[Paper Figures]] is only as good as this rubric, so it is specified here in full and validated before it is used at scale.

## Why a rubric judge instead of a metric

The question is "is this headline a good training target for this article?" A string-similarity metric cannot answer it, because it needs two headlines to compare and we only have one plus the article. The earlier design worked around that by generating a probe headline and measuring agreement, then inferring reference quality from the gap — an indirection that adds a whole model's worth of confound to every number.

A judge that reads the article removes the indirection. It scores the reference **directly** against its source, which is the construct we actually care about.

The cost is that the judge is an LLM, and our curated references were also written by an LLM (`gpt-5.6-luna`). That is a real threat, addressed by family separation and null calibration below, not waved away.

## The four dimensions

One aggregate quality score would hide the finding. The interesting claim is not "defective rows score lower" but *which* aspect of quality each defect damages, so the rubric returns four independent ordinal sub-scores.

### faithfulness — 1 to 5

Is every claim in the headline supported by the article?

- **5** Every element is directly supported. Attribution and uncertainty preserved where the article hedges.
- **4** Fully supported, with one minor framing choice the article does not quite make explicit.
- **3** Mostly supported, but one detail is an inference beyond what the text states.
- **2** Contains a claim the article does not support, or states a hedged claim as fact.
- **1** Contradicts the article, or centres on something absent from it.

### single-focus — 1 to 5

Does the headline name one dominant subject, or bundle several?

- **5** One clear subject.
- **4** One subject with a subordinate clause that supports it.
- **3** Two related aspects given roughly equal weight, still recognisably one story.
- **2** Two or more separable stories joined together.
- **1** A list of unrelated items with no dominant subject.

### informativeness — 1 to 5

Could a reader tell what happened?

- **5** Names the who and what, and enough of the specifics to be useful alone.
- **4** Clear subject and event, some specifics missing.
- **3** Identifies the topic but not the development — what it is *about*, not what *happened*.
- **2** Gestures at a topic without content. Teaser-like.
- **1** Pure hook, empty question, or promotional line carrying no information.

### cleanliness — 1 to 5

Is it free of scraping artifacts?

- **5** Clean headline text, nothing extraneous.
- **4** One minor punctuation or spacing oddity.
- **3** Contains a removable fragment such as a site label, category tag, or credit.
- **2** Multiple artifacts, or separator characters used as structure.
- **1** Largely boilerplate, metadata, or broken fragments rather than a headline.

## Why the dimensions are split this way

The split is not cosmetic; it defuses a specific confound.

The judge can **see** pipe characters. A single quality score would let a pipe digest be penalised for looking messy, and we could never tell whether the low score meant "this bundles three stories" (a real defect that would harm a trained model) or "this has odd punctuation" (cosmetic). Separating cleanliness from single-focus routes the formatting penalty to cleanliness and leaves single-focus measuring the thing we care about.

The same logic applies across the taxonomy in [[Dataset Defect Taxonomy]]. Predicted signatures, stated in advance so the results can contradict them:

| Defect | Expected damage |
|--------|-----------------|
| `multi_pipe_headline` | single-focus and cleanliness down; faithfulness roughly intact |
| `multiple_independent_items` | single-focus down; the others near normal |
| `substantive_content_not_in_text` | informativeness down; faithfulness ambiguous |
| `insufficient_substantive_content` | informativeness down alone |
| Long articles | informativeness declining with length as the subheading covers less of the article |

If a defect turns out to damage a dimension we did not predict, that is a result. If every defect damages every dimension equally, the rubric is not discriminating and needs revision before use.

## Scores are ordinal

A 1-to-5 rubric produces ordered categories, not measurements. The gap between 4 and 5 is not the gap between 1 and 2, so a mean over these scores describes a scale that does not exist. Consequences, carried through into [[Reference Quality Experiment]] and [[Paper Figures]]:

- Report **full distributions**, not means. Stacked ordinal bars, not bar-of-means.
- Use **rank-based tests** — Mann-Whitney, Wilcoxon, Cliff's delta — which are valid on ordinal data as they are.
- Use **ordinal logistic regression**, not OLS, for the confound model.

## Judge protocol

**Family separation.** The judge must come from a different model family than the curator (`gpt-5.6-luna`) and than the base model the training owner fine-tunes. Without this, a judge could prefer curated references because they were written in its own idiom, and we would have no way to distinguish that from genuine quality. Gemini is available in the repo via `evaluation/gemini_client.py`, and `evaluation/hf_client.py` provides a HF-hosted alternative — the same self-preference concern already recorded in [[Evaluation Metrics]].

**Hebrew anchor examples.** Every dimension needs a worked Hebrew example at score levels 1, 3, and 5, drawn from real HeSum rows, included in the prompt. Unanchored 1-to-5 scales drift over a batch of thousands, and drift correlated with position would be indistinguishable from a real effect. Draw the anchors from rows that are **excluded** from the analysis so they cannot leak.

**Structured output.** Four integers plus a one-sentence justification per dimension, via a strict JSON schema, matching the pattern already used in `data_curation/model_curation/`. The justifications are not scored but make error inspection possible and let the human validators see whether the judge is reasoning or pattern-matching.

**Blinding.** The judge sees the article and one headline. Never the stratum, the filter outcome, or whether the headline is original or curated. Randomise row order across the batch.

## Validation, before any full run

Three checks, all cheap, all mandatory. Nothing downstream is interpretable without them.

**Test-retest.** Run the judge twice at nonzero temperature over a 300-row subsample and report quadratically weighted kappa per dimension. An instrument that cannot reproduce its own scores cannot detect a real difference. This is also the first thing a reviewer asks about an LLM judge.

**Human validation.** About 150 rows, annotated independently and blind by both team members with this exact rubric. Report Cohen's kappa for human-human agreement and judge-human agreement per dimension. Human-human agreement matters as much as judge-human: if two humans cannot agree on single-focus, the dimension is underspecified and the judge's confidence on it is spurious. Use `evaluation/viewer/app.py` as the annotation interface — it already renders RTL Hebrew.

**Pilot before scale.** Run the full rubric on a few hundred rows first, check that score distributions are not degenerate (all 4s and 5s would mean the rubric cannot discriminate) and that the predicted defect signatures above are directionally visible. Revise the rubric then, not after paying for 10,000 rows. Sequencing is in [[Paper Figures#Compute placement]].

## Reuse across the project

The same instrument scores dataset references in E1 and E2 and **model outputs** in E4. That is deliberate: it puts reference quality and model quality on one axis, so a single figure can show whether a model trained on curated data produces headlines that score like curated references. Two separate metric families could not support that comparison.

Related: [[Reference Quality Experiment]], [[Dataset Defect Taxonomy]], [[Evaluation Metrics]], [[Paper Figures]]
