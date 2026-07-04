# Topic Clustering + Eval Stratification — Design

**Date:** 2026-07-04
**Project:** AMLK — Hebrew Text Summarization
**Scope:** New side-pipeline that discovers topic clusters in the dataset and uses them to
stratify existing evaluation reports (ROUGE/BERTScore/failure rates) by topic.

---

## 1. Motivation

The dataset (`outputs/data/raw/combined.jsonl`) currently carries no topic/genre label, only
`{text, summary, source}` where `source` is `hesum`/`iahlt`. The goal is to discover the topics
present in the corpus (politics, sports, economy, ...) and use them to check whether the model's
quality/failure rates vary by topic — e.g. "does the model hallucinate more on economy articles
than on sports articles?" This is a diagnostic for the paper's error-analysis section, not a
change to training or the main metric battery.

---

## 2. Architecture

Two independent artifacts, produced by two separate components:

```
combined.jsonl ──[Databricks GPU notebook]──► topics.jsonl (+ topics-summary.json)
                                                      │
predictions-*.jsonl ──[local, evaluation/stratify_by_topic.py]──► per-topic report
(+ *.errors.json, optional)                                       (ROUGE/BERTScore/failure-rates × topic)
```

`topics.jsonl` is a one-time, gitignored artifact (like `combined.jsonl` / the processed Arrow
splits) — regenerated only if the corpus or clustering approach changes, not on every eval run.

---

## 3. Component 1 — Topic discovery notebook (Databricks, GPU)

**File:** `notebooks/cluster_topics_databricks.py` (new). Databricks source format
(`# Databricks notebook source` / `# COMMAND ----------` cell markers) so it's git-diffable and
importable directly into a Databricks workspace. Run manually, occasionally, on a GPU cluster —
this is an optional-for-speed step, not a hard GPU requirement (see §6).

### Method

1. **Embed** each article's `summary` (short, clean, present for every record, and — critically —
   the exact string that survives untouched into every predictions file's `reference` field, so
   it doubles as a free join key back to eval data) using
   **`dicta-il/neodictabert-bilingual-embed`** — a Hebrew-native sentence-embedding model
   explicitly fine-tuned for clustering/semantic search (top-10 on the Hebrew Semantic Retrieval
   National Challenge), not a raw BERT encoder. Raw `onlplab/alephbert-base` (used for BERTScore)
   is deliberately *not* reused here: it was never fine-tuned for whole-sentence similarity, so
   its embeddings suffer the classic BERT "anisotropy" problem and cluster poorly by cosine
   distance — that's exactly the gap Sentence-BERT-style fine-tuning (and this Dicta model) fixes.
2. **Cluster** with **BERTopic** (UMAP → HDBSCAN → class-based TF-IDF), fixed random seed for
   reproducibility. Chosen over plain KMeans because: (a) HDBSCAN infers the number of topics
   from density instead of requiring a pre-chosen `k`; (b) it explicitly flags genuine outliers
   (one-off articles — obituaries, corrections) as noise instead of force-fitting every article
   into some cluster; (c) c-TF-IDF gives each cluster a keyword signature for free, without an
   LLM call per article.
3. **Name each cluster**: one Gemini call per real topic (top c-TF-IDF keywords + ~8 summaries
   nearest the centroid → one short Hebrew label). ~10–20 calls total for the whole corpus, not
   10,000. The noise cluster (`cluster_id = -1`) gets a fixed label `"לא מסווג"` with no LLM call
   — it's expected to be too heterogeneous for one label.
4. **Write outputs** to DBFS FileStore:
   - `topics.jsonl` — one row per article: `{summary, source, cluster_id, topic_label, keywords}`
   - `topics-summary.json` — cluster sizes + labels + keywords, for a quick sanity check of the
     discovered taxonomy before trusting it
   A markdown cell documents how to download both back into the repo
   (`outputs/data/raw/topics.jsonl`, `outputs/results/topics-summary.json`) via the Databricks
   `/files/...` FileStore URL.

### Notebook cell outline

1. Markdown header (role, execution environment, how to run) — satisfies the project's mandatory
   file-header rule for a notebook that can't carry a Python docstring.
2. GPU check (`torch.cuda.is_available()` + device info) + `%pip install bertopic
   sentence-transformers google-generativeai` + `dbutils.library.restartPython()`.
3. Config: `dbutils.widgets` for the input path (uploaded `combined.jsonl`) and the Gemini key
   (widget text; a comment documents the `dbutils.secrets` alternative for a configured scope).
4. Load `combined.jsonl`.
5. Embed summaries on GPU.
6. Fit BERTopic.
7. Name clusters via Gemini.
8. Write + display download instructions for both output files.

---

## 4. Component 2 — Stratified eval report (local, CPU)

**File:** `evaluation/stratify_by_topic.py` (new). Runs locally like the rest of `evaluation/`;
no GPU, no Databricks dependency — only needs `topics.jsonl` downloaded from the notebook run.

1. Load a predictions file (e.g. `predictions-finetuned.jsonl`) + `topics.jsonl`.
2. Join on exact text match: `prediction["reference"] == topic["summary"]`. Count and print
   unmatched rows rather than silently dropping them, so a broken join is visible immediately.
3. Group matched predictions by `topic_label`. Topics with fewer than a minimum count (default
   10) are reported as `"skipped: n too small"` instead of a noisy per-topic score. The noise
   bucket (`"לא מסווג"`) is always reported separately, never merged into a real topic.
4. Reuse `evaluation/evaluate.py`'s existing `compute_rouge` / `compute_bertscore` per topic
   group — no changes to `evaluate.py` itself.
5. If a matching `*.errors.json` (from `evaluation/error_analysis.py`) exists for the same
   predictions file, also break down its per-example failure-type rates by topic, via the same
   join key.
6. Write one JSON report: `{topic_label: {n, rouge, bertscore, failure_rates?}}`.

---

## 5. Data flow / join key

No changes to `data/preprocess.py`, the Arrow splits, or anything already pushed to the Hub.
The join relies on an existing invariant: `evaluation/infer.py` / `training/train_hf_job.py`
copy `batch["summary"][j]` verbatim into each prediction row's `reference` field, so the summary
text is a stable, already-existing key between `combined.jsonl` (whole corpus, where topics are
computed) and any predictions file (test-set subset, where topics are consumed).

---

## 6. Execution environment

- **Notebook (`cluster_topics_databricks.py`)**: manual, occasional run on a Databricks GPU
  cluster, imported/run by the user (no MCP access to Databricks from the agent side today).
  The GPU is for speed, not a hard requirement — `dicta-il/neodictabert-bilingual-embed` is a
  0.4B-parameter encoder-only model (~1.6 GB fp32 weights, no autoregressive generation/KV-cache
  growth), the same class of job as the AlephBERT-base BERTScore step this project already runs
  locally on CPU. This is a deliberate departure from AMLK's default local/HF-Jobs/Colab stack,
  scoped to this one side-analysis; it should not be read as a general policy change.
- **`stratify_by_topic.py`**: local CPU, same environment as `evaluate.py` / `error_analysis.py`.

---

## 7. New dependency

`bertopic` (pulls in `umap-learn`, `hdbscan`, `scikit-learn`; `sentence-transformers` is also
needed to load the embedding model with `trust_remote_code=True` per its model card). Added to
`requirements.txt`. Not needed at all for `stratify_by_topic.py`, which only depends on what
`evaluate.py` already imports.

---

## 8. Edge cases

- **HDBSCAN noise** (`cluster_id = -1`): kept as its own explicit bucket end-to-end (topic
  discovery → stratified report), never merged into a real topic, never sent to Gemini for
  naming.
- **Small topics**: real topics can legitimately be small in the *corpus* (a niche cluster still
  gets a Gemini label); what's suppressed is a per-topic *metric score* in the stratified report
  when too few *matched test-set predictions* fall into that topic to be meaningful.
- **Join misses**: logged as a count, not silently dropped — surfaces if the summary-text join
  key ever stops matching exactly (e.g. future whitespace/normalization changes upstream).
- **Determinism**: UMAP/HDBSCAN use a fixed random seed, so re-running against the same corpus
  reproduces the same clusters. Gemini's cluster *names* are not guaranteed byte-identical across
  reruns (LLM output), but are deterministic in scope (same cluster → one naming call).

---

## 9. Testing

Per the project's existing convention (fast tests + a small number of gated live tests, e.g.
`tests/test_evaluation.py`):
- **Fast, no-model/no-API test** for `stratify_by_topic.py`'s pure logic (join, grouping,
  small-topic skipping, noise-bucket separation) against a tiny synthetic fixture.
- **One gated live test** exercising the real BERTopic fit + one Gemini naming call against a
  handful of synthetic Hebrew summaries, skipped without `GEMINI_API_KEY` — mirrors the existing
  gated live Gemini test.
- The Databricks notebook itself is not unit-tested (no local GPU to run it against); its cells
  are validated by a manual run, the same way `evaluation_observation.ipynb` was validated by
  cell-by-cell execution rather than pytest.

---

## 10. What is NOT in this design

- No changes to training, `data/preprocess.py`, or the Arrow dataset schema.
- No changes to `evaluate.py`'s existing (non-stratified) report format.
- No automatic re-clustering pipeline — this is a one-off, manually-triggered artifact.
- No direct agent-driven deployment to Databricks (MCP connection unavailable) — the notebook is
  handed off for the user to import and run.
