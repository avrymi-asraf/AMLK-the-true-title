"""
Topic clustering: discovers topic clusters over the whole Hebrew news corpus so evaluation
results can later be broken down by topic (e.g. "does the model hallucinate more on economy
articles than sports?"). Embeds each article's summary with a Hebrew-native, clustering-tuned
sentence-embedding model, clusters with BERTopic (UMAP + HDBSCAN + a Hebrew-only c-TF-IDF
vectorizer, plus outlier-reduction/topic-merging passes to keep noise and near-duplicate topics
in check — see fit_topics()'s docstring), then names each cluster with one Gemini call. This
module holds the real, testable clustering logic; the
Databricks notebook (notebooks/cluster_topics_databricks.py) is a thin driver that clones the
repo, supplies the data + GPU, and calls cluster_dataset()/write_topics() from here — the same
"importable twin" pattern as evaluation/infer.py for train_hf_job.py. Its output (topics.jsonl)
is consumed locally, with no GPU needed, by evaluation/stratify_by_topic.py. See
docs/superpowers/specs/2026-07-04-topic-clustering-design.md for the full design.

Execution environment: the embedding step benefits from a GPU (faster) but doesn't require one
— dicta-il/neodictabert-bilingual-embed is a 0.4B-parameter encoder-only model, the same class
of job as the AlephBERT-base BERTScore step this project already runs locally on CPU (see
evaluation/evaluate.py). BERTopic/HDBSCAN clustering and the Gemini naming calls are always
CPU/API-only regardless of where embedding happened.
"""
import json
import os
from pathlib import Path

from evaluation.evaluate import gemini_json
from evaluation.gemini_client import GEMINI_MODEL

# A raw BERT encoder (e.g. onlplab/alephbert-base, used for BERTScore) is deliberately not
# reused here: without Sentence-BERT-style fine-tuning, whole-sentence BERT embeddings are
# anisotropic (poor cosine-similarity geometry), which makes them cluster badly. This model was
# fine-tuned specifically for clustering/semantic search in Hebrew.
EMBEDDING_MODEL = "dicta-il/neodictabert-bilingual-embed"
NOISE_TOPIC_ID = -1
NOISE_LABEL = "לא מסווג"

# BERTopic's default CountVectorizer (token pattern \w+, English stop_words) lets years, numeric
# IDs, and Latin media-brand tokens (ynet, nrg, bbc...) dominate c-TF-IDF keyword lists instead of
# actual Hebrew topic words — this range-class pattern matches Hebrew-letter sequences only
# (covers final forms ך/ם/ן/ף/ץ, all within U+05D0-U+05EA), so digits/Latin tokens never become
# keywords. A curated, non-exhaustive list of the most common Hebrew function words, since
# scikit-learn ships no built-in Hebrew stopword list.
HEBREW_TOKEN_PATTERN = r"(?u)[א-ת]{2,}"
HEBREW_STOPWORDS = frozenset("""
את של על עם אל מן כי אם גם רק אבל אולם אך או כן לא אין יש היה היתה היו יהיה תהיה יהיו
זה זאת זו אלה אלו הוא היא הם הן אני אתה אנחנו אתם אתן מי מה איך איפה מתי למה מדוע האם
כמה איזה אילו כל כמו עוד כבר תמיד שוב בכלל בעצם למעשה דבר דברים אחד אחת שני שתי שלושה שלוש
זהו זוהי כזה כזאת כאלה לפני אחרי תחת מעל ליד אצל נגד בעד לגבי בין עד כדי בשביל למען בעקבות
בשל למרות אף מאוד מאד ביותר יותר פחות הרבה מעט לעולם פעם שם כאן פה הנה הרי נו וכן כלומר
היינו וגם ולא שלא כשהוא כשהיא בו בה בהם בהן לו לה להם להן אותו אותה אותם אותן עליו עליה
עליהם עליהן ממנו ממנה מהם מהן עצמו עצמה עצמם עצמן כולם כולן כול שהוא שהיא שיש שאין
""".split())


def _build_vectorizer(ngram_range: tuple[int, int] = (1, 2)):
    """CountVectorizer for BERTopic's c-TF-IDF step, restricted to Hebrew words (see
    HEBREW_TOKEN_PATTERN/HEBREW_STOPWORDS above)."""
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(token_pattern=HEBREW_TOKEN_PATTERN, stop_words=list(HEBREW_STOPWORDS),
                            ngram_range=ngram_range)


NAMING_PROMPT = """You name topic clusters of Hebrew news articles.
Given these representative keywords and example summaries from one cluster, reply with a single
short Hebrew topic name (2-4 words, e.g. "פוליטיקה וממשלה", "ספורט", "כלכלה ועסקים").
Reply with ONLY a JSON object: {{"label": "<short Hebrew topic name>"}}

KEYWORDS: {keywords}

EXAMPLE SUMMARIES:
{examples}
"""


def embed_summaries(summaries: list[str]):
    """Encode summaries with the Hebrew-native, clustering-tuned embedding model."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {EMBEDDING_MODEL} on {device}...", flush=True)
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device=device)
    print(f"Embedding {len(summaries)} summaries...", flush=True)
    embeddings = model.encode(summaries, show_progress_bar=True, batch_size=64)
    print(f"Embedding done ({embeddings.shape}).", flush=True)
    return embeddings


def fit_topics(summaries: list[str], embeddings, min_cluster_size: int = 100,
               min_samples: int | None = 10, seed: int = 42, reduce_outliers: bool = True,
               nr_topics: int | str | None = "auto"):
    """Fit BERTopic (UMAP + HDBSCAN + c-TF-IDF) over precomputed embeddings.

    HDBSCAN infers the number of topics from density instead of requiring a pre-chosen k. Three
    knobs address the two failure modes seen on the full ~10k-doc corpus (~50% noise; near-
    duplicate topic names like "תקשורת ומדיה" vs "תקשורת וטלוויזיה"):
    - `_build_vectorizer()` (module-level) makes c-TF-IDF keywords Hebrew words instead of years/
      IDs/Latin site names, which is what made those topics look like duplicates in the first
      place (their real keywords are genuinely different once numbers/brand tokens are dropped).
    - `min_samples` decoupled from `min_cluster_size`: HDBSCAN ties min_samples to
      min_cluster_size unless told otherwise, which is unusually conservative about what counts
      as a core point; per BERTopic's FAQ, a smaller fixed min_samples produces less raw noise
      without having to loosen min_cluster_size (which controls topic granularity instead).
    - `language="multilingual"`: BERTopic defaults to `language="english"`, whose `_preprocess_text`
      strips every non-[A-Za-z0-9] character — i.e. all Hebrew — before c-TF-IDF, yielding an
      empty vocabulary on this corpus. Any non-"english" language skips that strip; "multilingual"
      is BERTopic's documented choice for non-English text.
    - `reduce_outliers`/`nr_topics`: BERTopic's own outlier-reduction (reassigns -1 docs to their
      nearest topic by embedding cosine similarity) and "auto" topic-merging (HDBSCAN over the
      topics' own c-TF-IDF vectors, so only genuinely near-duplicate topics merge — dissimilar
      ones stay separate) passes. Both are opt-out (set to False/None) if you'd rather keep
      HDBSCAN's raw, more conservative "explicit outlier" behavior.

    Returns (topic_model, cluster_ids) — cluster_ids aligns 1:1 with summaries.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    print(f"Fitting BERTopic on {len(summaries)} docs (UMAP → HDBSCAN)...", flush=True)
    umap_model = UMAP(random_state=seed, n_neighbors=15, n_components=5, metric="cosine")
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                             metric="euclidean", cluster_selection_method="eom",
                             prediction_data=True)
    topic_model = BERTopic(language="multilingual", umap_model=umap_model, hdbscan_model=hdbscan_model,
                            vectorizer_model=_build_vectorizer(), calculate_probabilities=False,
                            verbose=True)
    raw_cluster_ids, _ = topic_model.fit_transform(summaries, embeddings)
    cluster_ids = [int(c) for c in raw_cluster_ids]
    n_noise = sum(c == NOISE_TOPIC_ID for c in cluster_ids)
    n_topics = len(set(cluster_ids) - {NOISE_TOPIC_ID})
    print(f"Raw HDBSCAN: {n_topics} topics, {n_noise}/{len(cluster_ids)} noise "
          f"({n_noise / len(cluster_ids):.1%}).", flush=True)

    topics_reassigned = False
    if reduce_outliers and n_noise:
        print("Reassigning noise docs to their nearest topic by embedding similarity...", flush=True)
        cluster_ids = topic_model.reduce_outliers(summaries, cluster_ids, strategy="embeddings",
                                                    embeddings=embeddings)
        topic_model.topics_ = [int(c) for c in cluster_ids]
        topics_reassigned = True
        n_noise = sum(c == NOISE_TOPIC_ID for c in cluster_ids)
        print(f"After outlier reduction: {n_noise}/{len(cluster_ids)} noise "
              f"({n_noise / len(cluster_ids):.1%}).", flush=True)

    if nr_topics:
        n_before = len(set(cluster_ids) - {NOISE_TOPIC_ID})
        print(f"Merging near-duplicate topics (nr_topics={nr_topics!r})...", flush=True)
        # BERTopic >=0.17: reduce_topics(docs, nr_topics=...) reads/writes topic_model.topics_
        # internally. The pre-0.17 API reduce_topics(docs, topics, nr_topics=...) is gone —
        # passing cluster_ids positionally collides with nr_topics and raises TypeError.
        topic_model.topics_ = [int(c) for c in cluster_ids]
        topic_model.reduce_topics(summaries, nr_topics=nr_topics)
        cluster_ids = [int(c) for c in topic_model.topics_]
        topics_reassigned = True
        n_after = len(set(cluster_ids) - {NOISE_TOPIC_ID})
        print(f"Topics merged: {n_before} -> {n_after}.", flush=True)

    if topics_reassigned:
        topic_model.update_topics(summaries, vectorizer_model=_build_vectorizer())

    n_topics = len(set(cluster_ids) - {NOISE_TOPIC_ID})
    n_noise = sum(c == NOISE_TOPIC_ID for c in cluster_ids)
    print(f"Clustering done: {n_topics} topics, {n_noise} noise docs.", flush=True)
    return topic_model, cluster_ids


def name_topic(gemini_model, topic_model, topic_id: int, n_examples: int = 8) -> str:
    """One Gemini call: turn a cluster's c-TF-IDF keywords + representative examples into a
    short Hebrew label. Never called for the noise cluster (see NOISE_LABEL) — it's expected to
    be too heterogeneous for one label."""
    keywords = [word for word, _ in topic_model.get_topic(topic_id)][:10]
    examples = topic_model.get_representative_docs(topic_id)[:n_examples]
    prompt = NAMING_PROMPT.format(
        keywords=", ".join(keywords),
        examples="\n".join(f"- {e}" for e in examples),
    )
    result = gemini_json(gemini_model, prompt)
    return result.get("label", "").strip() or f"cluster_{topic_id}"


def cluster_dataset(records: list[dict], gemini_model=None, min_cluster_size: int = 100,
                     min_samples: int | None = 10, seed: int = 42, reduce_outliers: bool = True,
                     nr_topics: int | str | None = "auto"):
    """Full pipeline: embed -> cluster -> name. Each record needs 'summary' (and 'source').
    See fit_topics() for what min_samples/reduce_outliers/nr_topics do.

    Returns (rows, topic_model, embeddings): rows align 1:1 with records —
    {summary, source, cluster_id, topic_label, keywords}. embeddings is returned (not just
    discarded) so plot_clusters() can reuse them without a second, expensive embedding pass.
    """
    summaries = [r["summary"] for r in records]
    print(f"cluster_dataset: {len(records)} records", flush=True)
    embeddings = embed_summaries(summaries)
    topic_model, cluster_ids = fit_topics(summaries, embeddings, min_cluster_size, min_samples,
                                           seed, reduce_outliers, nr_topics)

    if gemini_model is None:
        import google.generativeai as genai

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)

    topic_ids = sorted(set(int(c) for c in cluster_ids))
    print(f"Naming {sum(t != NOISE_TOPIC_ID for t in topic_ids)} clusters via Gemini...", flush=True)
    labels_by_topic: dict[int, str] = {}
    keywords_by_topic: dict[int, list[str]] = {}
    for i, topic_id in enumerate(topic_ids):
        if topic_id == NOISE_TOPIC_ID:
            labels_by_topic[topic_id] = NOISE_LABEL
            keywords_by_topic[topic_id] = []
            continue
        print(f"  naming cluster {topic_id} ({i + 1}/{len(topic_ids)})...", flush=True)
        keywords_by_topic[topic_id] = [w for w, _ in topic_model.get_topic(topic_id)][:10]
        labels_by_topic[topic_id] = name_topic(gemini_model, topic_model, topic_id)

    rows = [
        {"summary": r["summary"], "source": r.get("source"), "cluster_id": int(cid),
         "topic_label": labels_by_topic[int(cid)], "keywords": keywords_by_topic[int(cid)]}
        for r, cid in zip(records, cluster_ids)
    ]
    return rows, topic_model, embeddings


def plot_clusters(topic_model, summaries: list[str], embeddings):
    """2D scatter of the discovered clusters, for a visual sanity check alongside
    topic_summary()'s numeric table. Uses BERTopic's built-in visualize_documents, which runs
    its own fresh 2D UMAP projection for plotting — separate from the 5D one fit_topics() used
    for HDBSCAN clustering — and returns a Plotly figure (hover text shows each summary).
    """
    return topic_model.visualize_documents(summaries, embeddings=embeddings, hide_annotations=True)


def topic_summary(rows: list[dict]) -> list[dict]:
    """Cluster sizes + labels + keywords, sorted largest-first, for a quick sanity check of the
    discovered taxonomy before trusting it."""
    by_topic: dict[int, dict] = {}
    for row in rows:
        cid = row["cluster_id"]
        if cid not in by_topic:
            by_topic[cid] = {"cluster_id": cid, "topic_label": row["topic_label"],
                              "keywords": row["keywords"], "count": 0}
        by_topic[cid]["count"] += 1
    return sorted(by_topic.values(), key=lambda t: -t["count"])


def write_topics(rows: list[dict], topics_path: Path, summary_path: Path) -> None:
    """Write topics.jsonl (one row per article) + topics-summary.json (per-cluster rollup)."""
    topics_path = Path(topics_path)
    summary_path = Path(summary_path)
    topics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(topics_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(topic_summary(rows), ensure_ascii=False, indent=2))
