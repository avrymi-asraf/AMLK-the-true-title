"""
Topic clustering: discovers topic clusters over the whole Hebrew news corpus so evaluation
results can later be broken down by topic (e.g. "does the model hallucinate more on economy
articles than sports?"). Embeds each article's summary with a Hebrew-native, clustering-tuned
sentence-embedding model, clusters with BERTopic (UMAP + HDBSCAN + a Hebrew-only c-TF-IDF
vectorizer, plus outlier-reduction/topic-merging passes to keep noise and near-duplicate topics
in check — see fit_topics()'s docstring), then names each cluster with one Gemini call, then
collapses any clusters Gemini still named identically (merge_duplicate_labels()) so the final
report has one row per distinct real-world topic. This
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

# Layout/journalism-meta words ("front page headline", "as reported this morning") that show up
# as top keywords across many unrelated topics — they describe the *article format*, not its
# subject, and were the reason several full-corpus clusters got a fake "חדשות בישראל"-style label
# (2026-07-04 run) instead of their real domain. Dropping them lets c-TF-IDF surface the actual
# distinguishing subject words, which also reduces the number of near-duplicate topic names.
BOILERPLATE_STOPWORDS = frozenset("""
כותרת הכותרת הראשית בכותרת נכתב כותב עיתון העיתון בעיתון מהעיתון הבוקר שער גיליון מוסף
כתבה ידיעה דיווח מדווח לינק קישור אתמול אמש השבוע
""".split())


def _truncate_text(text: str, max_chars: int = 4000) -> str:
    """First N chars of article body — enough topical signal; the embedding model truncates further."""
    return text[:max_chars].strip()


def _build_vectorizer(ngram_range: tuple[int, int] = (1, 2)):
    """CountVectorizer for BERTopic's c-TF-IDF step, restricted to Hebrew words (see
    HEBREW_TOKEN_PATTERN/HEBREW_STOPWORDS above)."""
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(token_pattern=HEBREW_TOKEN_PATTERN,
                            stop_words=list(HEBREW_STOPWORDS | MEDIA_STOPWORDS | BOILERPLATE_STOPWORDS),
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


def _cuda_free_gib() -> float | None:
    """Free GPU memory in GiB, or None when CUDA is unavailable."""
    import torch

    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return free / (1024 ** 3)


def _resolve_embed_device(prefer: str = "auto", min_free_gib: float = 2.0) -> str:
    """Pick cuda vs cpu. `auto` uses cuda only when enough memory is free — on a shared
    Databricks cluster another notebook can leave <100 MB free and a blind cuda pick OOMs."""
    import torch

    prefer = prefer.lower()
    if prefer not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"embed_device must be 'auto', 'cpu', or 'cuda', got {prefer!r}")
    if prefer == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if prefer == "cuda":
            raise RuntimeError("embed_device='cuda' requested but no CUDA device is available")
        print("CUDA not available — embedding on CPU.", flush=True)
        return "cpu"
    free_gib = _cuda_free_gib()
    if prefer == "cuda":
        print(f"Using CUDA ({free_gib:.2f} GiB free).", flush=True)
        return "cuda"
    # auto
    if free_gib is not None and free_gib < min_free_gib:
        print(f"CUDA has only {free_gib:.2f} GiB free (< {min_free_gib} GiB) — likely another "
              "process on this cluster GPU; embedding on CPU instead.", flush=True)
        return "cpu"
    print(f"Using CUDA ({free_gib:.2f} GiB free).", flush=True)
    return "cuda"


def _release_cuda() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def embed_texts(texts: list[str], *, device: str = "auto", batch_size: int = 8,
                min_free_gib: float = 2.0):
    """Encode texts with the Hebrew-native, clustering-tuned embedding model.

    Defaults to batch_size=8 (not 64) because embed_field='text' feeds ~4k-char article
    snippets — large batches OOM even on a clean 22 GB GPU. When device='auto', cuda is
    skipped if less than min_free_gib is free (common on shared Databricks clusters); on
    cuda OOM the call retries on CPU automatically.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    resolved = _resolve_embed_device(device, min_free_gib)
    _release_cuda()

    def _encode(on_device: str, bs: int):
        print(f"Loading {EMBEDDING_MODEL} on {on_device} (batch_size={bs})...", flush=True)
        model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device=on_device)
        print(f"Embedding {len(texts)} texts...", flush=True)
        try:
            return model.encode(texts, show_progress_bar=True, batch_size=bs)
        finally:
            del model
            _release_cuda()

    try:
        embeddings = _encode(resolved, batch_size)
    except torch.cuda.OutOfMemoryError:
        if resolved != "cuda":
            raise
        print("CUDA OOM during encode — retrying on CPU...", flush=True)
        _release_cuda()
        embeddings = _encode("cpu", max(4, batch_size // 2))

    print(f"Embedding done ({embeddings.shape}).", flush=True)
    return embeddings


# Backward-compatible alias — callers may still say embed_summaries.
embed_summaries = embed_texts


def fit_topics(cluster_docs: list[str], embeddings, min_cluster_size: int = 60,
               min_samples: int | None = 15, seed: int = 42, reduce_outliers: bool = True,
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
    - `min_cluster_size=60`/`min_samples=15` (raised from an initial 25/5 pass that produced ~100
      topics, many of them near-duplicate Gemini labels for what was really the same domain seen
      through slightly different HDBSCAN sub-clusters) — coarser HDBSCAN granularity up front means
      fewer, larger, more distinct topics before naming even runs. See also
      `cluster_dataset(merge_duplicate_labels=True)`, which additionally collapses any topics that
      still end up sharing an identical Gemini label.
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


def merge_duplicate_labels(rows: list[dict]) -> list[dict]:
    """Collapse clusters that ended up with the same Gemini topic_label into one logical topic.

    Even with a coarser HDBSCAN (see fit_topics) and boilerplate-stripped keywords, a few raw
    clusters can still be near-duplicate slices of the same real-world domain (e.g. two clusters
    both named "ביטחון וצבא") and Gemini has no visibility across clusters to avoid repeating a
    label. This is a cheap, local, no-extra-API-call fix: pick the smallest cluster_id per label
    as the canonical id and union the keyword lists, so topic_summary()/write_topics() report one
    row per distinct label instead of several. Does not touch topic_model — only the row-level
    `cluster_id`/`keywords` used for reporting and stratification.
    """
    if not rows:
        return rows
    canonical_id_by_label: dict[str, int] = {}
    keywords_by_label: dict[str, list[str]] = {}
    for row in rows:
        label = row["topic_label"]
        canonical_id_by_label[label] = min(canonical_id_by_label.get(label, row["cluster_id"]), row["cluster_id"])
        bucket = keywords_by_label.setdefault(label, [])
        for kw in row["keywords"]:
            if kw not in bucket:
                bucket.append(kw)

    return [
        {**row, "cluster_id": canonical_id_by_label[row["topic_label"]],
         "keywords": keywords_by_label[row["topic_label"]][:10]}
        for row in rows
    ]


def cluster_dataset(records: list[dict], gemini_model=None, min_cluster_size: int = 60,
                     min_samples: int | None = 15, seed: int = 42, reduce_outliers: bool = True,
                     outlier_threshold: float = 0.35, nr_topics: int | str | None = None,
                     embed_field: str = "text", max_embed_chars: int = 4000,
                     merge_duplicates: bool = True, embed_device: str = "auto",
                     embed_batch_size: int = 8):
    """Full pipeline: embed -> cluster -> name. Each record needs `summary` (join key) and, when
    embed_field='text', `text` (article body). Cluster geometry + c-TF-IDF keywords come from
    truncated article bodies by default — summaries alone collapse into one media-meta mega-topic.
    `merge_duplicates` runs merge_duplicate_labels() on the result (see its docstring) so clusters
    Gemini happened to name identically are reported as one topic.

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
    embeddings = embed_texts(cluster_docs, device=embed_device, batch_size=embed_batch_size)
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

    # So plot_clusters() can show Hebrew topic names in the legend (not "0_keyword_keyword").
    topic_model.set_topic_labels({**labels_by_topic, NOISE_TOPIC_ID: NOISE_LABEL})

    rows = [
        {"summary": r["summary"], "source": r.get("source"), "cluster_id": int(cid),
         "topic_label": labels_by_topic[int(cid)], "keywords": keywords_by_topic[int(cid)]}
        for r, cid in zip(records, cluster_ids)
    ]
    if merge_duplicates:
        n_before = len(set(r["cluster_id"] for r in rows))
        rows = merge_duplicate_labels(rows)
        n_after = len(set(r["cluster_id"] for r in rows))
        print(f"Merged duplicate-labeled clusters: {n_before} -> {n_after}.", flush=True)
    return rows, topic_model, embeddings


def plot_clusters(topic_model, cluster_docs: list[str], embeddings, *,
                  hover_texts: list[str] | None = None, sample: float | None = 0.15):
    """2D scatter of the discovered clusters, for a visual sanity check alongside
    topic_summary()'s numeric table. Uses BERTopic's built-in visualize_documents, which runs
    its own fresh 2D UMAP projection for plotting — separate from the 5D one fit_topics() used
    for HDBSCAN clustering — and returns a Plotly figure.

    Pass short `hover_texts` (e.g. summaries) when `cluster_docs` are long article bodies — embedding
    10k × 4k-char hovers blows past Databricks' ~20 MB command-result cap. `sample` keeps at most
    that fraction of docs per topic (BERTopic built-in); None plots every point (fine for smoke runs).
    """
    if hover_texts is None:
        hover_texts = cluster_docs
    if len(hover_texts) != len(embeddings):
        raise ValueError(f"hover_texts length {len(hover_texts)} != embeddings length {len(embeddings)}")
    return topic_model.visualize_documents(hover_texts, embeddings=embeddings, hide_annotations=True,
                                            custom_labels=True, sample=sample)


def plot_topic_sizes(summary_rows: list[dict], top_n: int = 30):
    """Horizontal bar chart of cluster sizes (topic_summary() output), largest first — a quick
    visual complement to the numeric table for spotting fragmentation/imbalance (e.g. one
    mega-topic dwarfing the rest) at a glance. Small (<=top_n bars), so unlike plot_clusters() it
    never risks the Databricks cell-output cap and can be shown inline with displayHTML.
    """
    import plotly.express as px

    top = summary_rows[:top_n]
    labels = [f"{t['topic_label']} ({t['cluster_id']})" for t in top]
    counts = [t["count"] for t in top]
    fig = px.bar(x=counts[::-1], y=labels[::-1], orientation="h",
                 labels={"x": "articles", "y": "topic"},
                 title=f"Top {len(top)} topic cluster sizes")
    return fig


def write_plot_html(fig, path: Path) -> Path:
    """Write a Plotly figure to disk (e.g. DBFS FileStore) for viewing outside the notebook cell."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


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
