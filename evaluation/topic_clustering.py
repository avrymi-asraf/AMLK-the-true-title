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


# Outlet / boilerplate tokens that dominate summary c-TF-IDF when clustering on headlines
# instead of article bodies — harmless when embed_field='text', belt-and-suspenders otherwise.
MEDIA_STOPWORDS = frozenset("""
ידיעות אחרונות ישראל היום הארץ מעריב מקורבת וואלה nrg ynet וואלה חדשות העיתונים
התקשורת העיתונות בעיתונות מהעיתונות בתקשורת
""".split())


def _truncate_text(text: str, max_chars: int = 4000) -> str:
    """First N chars of article body — enough topical signal; the embedding model truncates further."""
    return text[:max_chars].strip()


def _build_vectorizer(ngram_range: tuple[int, int] = (1, 2)):
    """CountVectorizer for BERTopic's c-TF-IDF step, restricted to Hebrew words (see
    HEBREW_TOKEN_PATTERN/HEBREW_STOPWORDS above)."""
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(token_pattern=HEBREW_TOKEN_PATTERN,
                            stop_words=list(HEBREW_STOPWORDS | MEDIA_STOPWORDS),
                            ngram_range=ngram_range, lowercase=False)


NAMING_PROMPT = """You name topic clusters of Hebrew news articles.
Given these representative keywords and example summaries from one cluster, reply with a single
short Hebrew topic name (2-4 words) for the news *domain* or subject area — e.g. "פוליטיקה
וממשלה", "ספורט", "כלכלה ועסקים", "משפט וצדק", "ביטחון".
Do NOT name the media format, outlet, or generic journalism meta-topic (avoid labels like
"חדשות ותקשורת", "תקשורת ועיתונות", "כותרות").
Reply with ONLY a JSON object: {{"label": "<short Hebrew topic name>"}}

KEYWORDS: {keywords}

EXAMPLE SUMMARIES:
{examples}
"""


def embed_texts(texts: list[str]):
    """Encode texts with the Hebrew-native, clustering-tuned embedding model."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {EMBEDDING_MODEL} on {device}...", flush=True)
    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device=device)
    print(f"Embedding {len(texts)} texts...", flush=True)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    print(f"Embedding done ({embeddings.shape}).", flush=True)
    return embeddings


# Backward-compatible alias — callers may still say embed_summaries.
embed_summaries = embed_texts


def fit_topics(cluster_docs: list[str], embeddings, min_cluster_size: int = 25,
               min_samples: int | None = 5, seed: int = 42, reduce_outliers: bool = True,
               outlier_threshold: float = 0.35, nr_topics: int | str | None = None):
    """Fit BERTopic (UMAP + HDBSCAN + c-TF-IDF) over precomputed embeddings.

    `cluster_docs` is what BERTopic tokenizes for c-TF-IDF keywords — pass truncated article
    bodies (not headlines) when possible; see cluster_dataset(embed_field='text').

    Tuning notes from full-corpus runs:
    - Summaries alone collapse into one "חדשות ותקשורת" mega-topic (~99% of docs) because
      headlines share outlet names and format. Embed/cluster on article `text` instead.
    - `reduce_outliers` with threshold=0 (BERTopic default) force-assigns every noise doc to its
      nearest topic, flooding the largest cluster. Use outlier_threshold (~0.35 cosine sim) so
      uncertain docs stay -1.
    - `nr_topics='auto'` over-merges distinct domains into a few media-meta topics — off by default.
    - Lower min_cluster_size (25) + min_samples (5) yields more granular HDBSCAN topics than 40/10.
    - `language='multilingual'` is required — English mode strips all Hebrew before c-TF-IDF.

    Returns (topic_model, cluster_ids) — cluster_ids aligns 1:1 with cluster_docs.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    print(f"Fitting BERTopic on {len(cluster_docs)} docs (UMAP → HDBSCAN)...", flush=True)
    umap_model = UMAP(random_state=seed, n_neighbors=10, n_components=5, min_dist=0.0, metric="cosine")
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                             metric="euclidean", cluster_selection_method="eom",
                             prediction_data=True)
    topic_model = BERTopic(language="multilingual", umap_model=umap_model, hdbscan_model=hdbscan_model,
                            vectorizer_model=_build_vectorizer(), calculate_probabilities=False,
                            verbose=True)
    raw_cluster_ids, _ = topic_model.fit_transform(cluster_docs, embeddings)
    cluster_ids = [int(c) for c in raw_cluster_ids]
    n_noise = sum(c == NOISE_TOPIC_ID for c in cluster_ids)
    n_topics = len(set(cluster_ids) - {NOISE_TOPIC_ID})
    print(f"Raw HDBSCAN: {n_topics} topics, {n_noise}/{len(cluster_ids)} noise "
          f"({n_noise / len(cluster_ids):.1%}).", flush=True)

    topics_reassigned = False
    if reduce_outliers and n_noise:
        print(f"Reassigning noise docs with embedding similarity >= {outlier_threshold}...", flush=True)
        cluster_ids = topic_model.reduce_outliers(cluster_docs, cluster_ids, strategy="embeddings",
                                                    embeddings=embeddings, threshold=outlier_threshold)
        topic_model.topics_ = [int(c) for c in cluster_ids]
        topics_reassigned = True
        n_noise = sum(c == NOISE_TOPIC_ID for c in cluster_ids)
        print(f"After outlier reduction: {n_noise}/{len(cluster_ids)} noise "
              f"({n_noise / len(cluster_ids):.1%}).", flush=True)

    if nr_topics:
        n_before = len(set(cluster_ids) - {NOISE_TOPIC_ID})
        print(f"Merging near-duplicate topics (nr_topics={nr_topics!r})...", flush=True)
        topic_model.topics_ = [int(c) for c in cluster_ids]
        topic_model.reduce_topics(cluster_docs, nr_topics=nr_topics)
        cluster_ids = [int(c) for c in topic_model.topics_]
        topics_reassigned = True
        n_after = len(set(cluster_ids) - {NOISE_TOPIC_ID})
        print(f"Topics merged: {n_before} -> {n_after}.", flush=True)

    if topics_reassigned:
        topic_model.update_topics(cluster_docs, vectorizer_model=_build_vectorizer())

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


def cluster_dataset(records: list[dict], gemini_model=None, min_cluster_size: int = 25,
                     min_samples: int | None = 5, seed: int = 42, reduce_outliers: bool = True,
                     outlier_threshold: float = 0.35, nr_topics: int | str | None = None,
                     embed_field: str = "text", max_embed_chars: int = 4000):
    """Full pipeline: embed -> cluster -> name. Each record needs `summary` (join key) and, when
    embed_field='text', `text` (article body). Cluster geometry + c-TF-IDF keywords come from
    truncated article bodies by default — summaries alone collapse into one media-meta mega-topic.

    Returns (rows, topic_model, embeddings): rows align 1:1 with records —
    {summary, source, cluster_id, topic_label, keywords}. embeddings is returned (not just
    discarded) so plot_clusters() can reuse them without a second, expensive embedding pass.
    """
    summaries = [r["summary"] for r in records]
    if embed_field == "summary":
        cluster_docs = summaries
    elif embed_field == "text":
        cluster_docs = [_truncate_text(r["text"], max_embed_chars) for r in records]
    else:
        raise ValueError(f"embed_field must be 'text' or 'summary', got {embed_field!r}")

    print(f"cluster_dataset: {len(records)} records (embed_field={embed_field!r})", flush=True)
    embeddings = embed_texts(cluster_docs)
    topic_model, cluster_ids = fit_topics(cluster_docs, embeddings, min_cluster_size, min_samples,
                                           seed, reduce_outliers, outlier_threshold, nr_topics)

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


def plot_clusters(topic_model, cluster_docs: list[str], embeddings):
    """2D scatter of the discovered clusters, for a visual sanity check alongside
    topic_summary()'s numeric table. Uses BERTopic's built-in visualize_documents, which runs
    its own fresh 2D UMAP projection for plotting — separate from the 5D one fit_topics() used
    for HDBSCAN clustering — and returns a Plotly figure (hover text shows each cluster doc).
    """
    return topic_model.visualize_documents(cluster_docs, embeddings=embeddings, hide_annotations=True)


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
