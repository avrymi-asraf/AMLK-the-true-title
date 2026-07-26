# Data Curation Pipeline

#status/done

Turns the raw HeSum download into a curated headline-generation dataset. Lives in `data_curation/`, and the authoritative stage-by-stage reference with every artifact path and shape is `data_curation/CURATION_ROADMAP.md` — this note records the numbers, the decisions behind them, and where the design has consequences for [[Reference Quality Experiment]].

Rebuild:

```bash
python -m data_curation.build_curated_dataset
```

Only the two model-result artifacts need to ship with the repo (`source_filter_results.json`, `headline_target_curation_results.json`). Everything else regenerates deterministically, and the rebuild refuses to proceed unless the regenerated `source_filter_input.json` matches the sha256 the model actually saw. That guard is worth appreciating: it makes the LLM-labeled results verifiably attached to a specific input, which is what lets the curation be reproduced rather than merely trusted.

## Verified counts

Read off the artifacts on 2026-07-26, not from memory:

| Stage | Input | Output | Effect |
|-------|-------|--------|--------|
| 1. Download | `biunlp/HeSum` | 10,000 | baseline |
| 2. Tail boilerplate trim | 10,000 | 10,000 | 722 rows had a scraped tail removed |
| 3. Token budget | 10,000 | keep map | 7,341 keep / **2,659 remove** (26.6%) |
| 4. Multi-pipe headline | 10,000 | keep map | 7,588 keep / **2,412 remove** (24.1%) |
| 5. Intersection | both maps | 6,486 | **35.1% removed** before any model saw a row |
| 6. Source filter | 6,486 | 6,486 labels | 5,854 usable / **632 unusable** |
| 7. Headline curation | 5,854 | 5,854 decisions | 2,785 kept / **3,069 rewritten** (52.4%) |
| 8. Final assembly | — | **5,854** | 41.5% net reduction from raw |

Curation model for stages 6 and 7: **`gpt-5.6-luna`** via the OpenAI Batch API, structured JSON output with a strict schema, prompt caching on.

## The two deterministic filters

Both are thresholds, and both are load-bearing for the analysis, so their exact definitions matter.

**Token budget** (`filter_over_token_budget.py`). Tokenizes with `dicta-il/dictalm2.0-instruct` and keeps rows where `tokens(text) + tokens(headline) <= 4000`. Two details worth knowing: the budget covers the *pair*, not the article alone, and the tokenizer is DictaLM's, which is a leftover from [[Project Pivot|Era 2]] and the right choice only as long as the trained model stays in that family. If the training owner uses a different base model, this filter's boundary moves.

**Multi-pipe headline** (`filter_multi_pipe_headlines.py`). Keeps rows where `headline.count("|") <= 1`, so it removes headlines with **two or more** pipes, meaning three or more bundled segments. It removes rows; it does not trim pipes out of headlines. Pipe *repair* happens later, inside stage 7, where the model rewrites a noisy headline into one coherent target.

The two filters are independent and only combined at stage 5. Both write **complete 10,000-id boolean maps** rather than filtered row lists, so membership is fully recoverable — which is what makes the strata in [[Dataset Defect Taxonomy]] available without re-labeling anything.

One practical note: those two files (`token_budget_upto_4000.json`, `headline_pipes_upto_1.json`) are **not committed**. Only the two model-result artifacts ship with the repo; the keep-maps regenerate when you run `build_curated_dataset` (or `run_pre_model_cleanup` alone). So the strata are recoverable, but recovering them starts with a pipeline run, not with a file that is already there.

## Design consequences

Three properties of this pipeline shape what the analysis can and cannot claim.

**Filtering happens before labeling.** Stages 3 and 4 remove 3,514 rows before `gpt-5.6-luna` ever reads one, so those rows carry no LLM label. Recovering them as analysis strata is a join on the keep-maps, not a re-run — but it does mean the deterministic and model-based defect classes are not measured by the same instrument.

**The headline axis has no reason codes.** Stage 7 records `replacement_headline: null | string` and nothing else. We know 3,069 headlines were unfit as training targets; we do not know from the artifact *why* any individual one was. The reasons are enumerated in the stage-7 prompt (`refine_prompt_schema.py`) but were never emitted per row. Sub-typing them post hoc from the `(original, replacement)` diff is cheap, because it needs only the two strings.

**The token filter and the lead-bias question conflict.** Dropping every row over 4,000 tokens removes the long-article tail, which is exactly the population where lead bias would be most visible. Convenient for training, destructive for the original probe. Recorded as a tension in [[Lead Bias Probe]]; the resolution is to treat length as a continuous covariate over the full 10,000 rows rather than as a filter boundary. See [[Dataset Defect Taxonomy]] for why the over-budget group cannot serve as an analysis stratum at all.

## Final artifact

```text
data_curation/artifacts/final_clean_hesum.json    5,854 records
```

```json
{ "hesum_id": "1", "text": "...", "headline": "..." }
```

`hesum_id` preserves the original HeSum string id, so every curated row can be joined back to its raw counterpart and to both keep-maps. `headline` is the original when stage 7 returned null, otherwise the curated replacement. This is the file the training owner receives, split per [[Training Handoff Contract]].

## Relationship to the older `--clean` profile

`data/clean.py` and the `--clean` flag in `data/preprocess.py` were an earlier attempt at the same problem: rewrite pipe digests into prose with regex, optionally drop the worst roundups. It was crude — string surgery with no notion of whether the underlying article was usable — and `data_curation/` supersedes it. The old profile stays in the tree because the raw pipeline is still byte-for-byte reproducible through it, but new work should not use it.

Related: [[Project Pivot]], [[Dataset Defect Taxonomy]], [[Reference Quality Experiment]], [[Training Handoff Contract]]
