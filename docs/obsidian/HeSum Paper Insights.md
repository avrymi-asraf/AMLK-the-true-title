# HeSum Paper Insights

Paper: **HeSum: a Novel Dataset for Abstractive Text Summarization in Hebrew**  
[arXiv:2406.03897](https://arxiv.org/pdf/2406.03897) · [GitHub](https://github.com/OnlpLab/HeSum)

## Dataset facts (relevant to AMLK)

- 10,000 article–summary pairs from Hebrew independent journalism sites
- Summaries = professional **extended subheadings** (not body extracts)
- High **abstractiveness**: ~42% novel unigrams, ~73% novel bigrams (Table 2)
- Long articles: avg ~1,400 words (Table 1); 90th percentile ~5,276 **tokens** (Appendix)
- Morphologically rich Hebrew → n-gram metrics undercount valid paraphrases

AMLK uses `biunlp/HeSum` via `data/download.py` → `outputs/data/raw/combined.jsonl`.
The curation pipeline downloads it separately via `data_curation/data_download/download_hesum.py`.

## How HeSum was built, and what that implies (2026-07-26)

Re-read as dataset criticism rather than as a benchmark to beat. This section is the literature basis
for the [[Project Pivot]].

The paper is honest about its construction, and the construction is the source of the defects we found:

- **Scraped, not authored.** Articles and summaries were collected from Hebrew news sites. Nobody wrote
  a summary for the summarization task; the "summary" field is whatever the site published as an
  extended subheading.
- **Subheadings serve the publisher, not the reader of a benchmark.** A subheading exists to draw a
  click. That makes teasers, hooks, and promotional phrasing structurally likely rather than accidental
  — which is why `informativeness` is a separate dimension in [[Reference Quality Rubric]].
- **No filtering for one-story-per-row.** Press reviews, roundups, and digests were collected like any
  other article, so a single "summary" can bundle several unrelated headlines. We measured 8.8% of
  labeled rows as multi-item, plus 24.1% of the corpus carrying two or more pipes in the headline.
- **Scraping artifacts survive into the fields.** Site labels, credits, category tags, and boilerplate
  tails appear inside summaries. 722 rows had a repeated tail removable by pure string matching alone.

**What this changes about the SOTA table below.** Those ROUGE numbers are computed against these
references. A model that scores 17.5 ROUGE-1 is 17.5-similar to a subheading that may be a three-headline
digest. High agreement with a defective reference is not evidence of quality, and this is the same logic
as the paper's own finding that ROUGE correlates about -0.16 with human judgement here. Our conclusion
goes one step further than the paper's: the problem is not only that ROUGE is a poor metric for Hebrew,
it is that a meaningful share of the targets are not summaries.

**What the paper already told us and we initially under-weighted.** The 90th-percentile article is about
5,276 tokens. Any 4,000-token budget therefore excludes a large, non-random slice of the corpus — see
the tension recorded in [[Lead Bias Probe]]. And "professional extended subheadings" was always a
description of provenance, not a quality guarantee.

Full defect counts and label definitions: [[Dataset Defect Taxonomy]].

## HeSum SOTA (Table 3)

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore (AlephBERT) |
|-------|---------|---------|---------|----------------------|
| GPT-4 | 13.6 | 3.7 | 10.4 | 77.3 |
| GPT-3.5 | 13.7 | 3.8 | 10.6 | 77.0 |
| mLongT5 (fine-tuned) | **17.5** | **7.6** | **14.7** | 57.6 |
| Human eval (coherence / completeness) | — | — | — | GPT > mLongT5 despite lower ROUGE |

**Takeaway:** High ROUGE can mean “copies surface form” (mLongT5); semantic metrics + humans favor GPT. Our finetuned ROUGE 11.4 is in the same ballpark as GPT-4, not “catastrophic.”

## AlephBERT for BERTScore

HeSum uses **AlephBERT** (`onlplab/alephbert-base`) as BERTScore backbone — Hebrew monolingual, better for MRL than generic multilingual models.

AMLK currently uses `xlm-roberta-large` in `evaluation/evaluate.py` → switch planned in [[Fix Plan#Phase 0]].

## Prompt language (Table 8)

They tested prefix / input / output in Hebrew (H) vs English (E). Best GPT-3.5 config: **E-H-H** (English instruction, Hebrew article, Hebrew output) → ROUGE-1 **17.1**.

AMLK `data/prompts.py` is already E-H-H:

```
Summarize the following Hebrew text. Write the summary in Hebrew:
```

Paper body mentions E-E-H but their own table shows E-H-H wins. **Do not translate full articles to English for training.**

Borrow instead: **length constraint** (“up to 3 sentences”) from their GPT prompt (Figure 2).

## ROUGE vs human eval

Pearson correlation between ROUGE and human scores ≈ **−0.16** (p < 2.39×10⁻⁵). Higher ROUGE ≠ better summary for Hebrew.

→ Lead with [[Evaluation Metrics]]; ROUGE for comparability only.

## mLongT5 training (Table 6)

- Early stopping on **ROUGE-1**
- Base: 18 epochs; Large: 12 epochs
- Long-sequence encoder-decoder (designed for ~2.7k token articles)

AMLK uses Qwen3-2B causal LM + QLoRA, 1 epoch — different regime but the **early-stopping-on-generation-metric** lesson still applies at the principle level.

## Error types (Table 4)

Fine-tuned mLongT5: repetition, **copy from article** (13%), low abstractiveness.  
GPT models: Hebrew morphology errors (gender, smixut, definiteness), hallucinations.

Maps to our [[Prediction Failure Modes]] and `evaluation/error_analysis.py` labels.

## Tokenizer / MRL lesson

- **Generator tokenizer:** keep Qwen BPE — cannot cheaply swap without retraining embeddings.
- **Evaluation tokenizer:** morpheme-aware analysis (Table 5) — apply on ROUGE side, not training.

Related: [[Evaluation Metrics]], [[Lead Bias Probe]], [[Fix Plan]]
