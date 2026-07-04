# Databricks notebook source
# MAGIC %md
# MAGIC # Topic clustering — Hebrew news corpus
# MAGIC
# MAGIC Discovers topic clusters over the whole AMLK dataset (`outputs/data/raw/combined.jsonl`,
# MAGIC ~10,000 Hebrew news articles) so evaluation results can later be broken down by topic —
# MAGIC e.g. "does the model hallucinate more on economy articles than sports?" See
# MAGIC `docs/superpowers/specs/2026-07-04-topic-clustering-design.md` for the full design.
# MAGIC
# MAGIC Embeds each article's summary with a Hebrew-native, clustering-tuned embedding model
# MAGIC (`dicta-il/neodictabert-bilingual-embed`), clusters with BERTopic (UMAP + HDBSCAN +
# MAGIC c-TF-IDF), then names each cluster with one Gemini call. Writes `topics.jsonl` +
# MAGIC `topics-summary.json` to DBFS FileStore for download back into the repo
# MAGIC (`outputs/data/raw/topics.jsonl`, `outputs/results/topics-summary.json`), which
# MAGIC `evaluation/stratify_by_topic.py` then consumes locally — no GPU needed for that step.
# MAGIC
# MAGIC **Role in the project / execution environment:** this is a manual, occasional side-
# MAGIC analysis, not part of the main training/evaluation pipeline. It runs on a Databricks GPU
# MAGIC cluster for speed, but the GPU is not a hard requirement — the embedding model is a
# MAGIC 0.4B-parameter encoder (no autoregressive generation), the same class of job as the
# MAGIC AlephBERT-base BERTScore step AMLK already runs locally on CPU. This is a deliberate,
# MAGIC scoped-to-this-notebook departure from AMLK's default local/HF-Jobs/Colab stack — see
# MAGIC `AGENTS.md` for the project's default execution model.
# MAGIC
# MAGIC **Before running:**
# MAGIC 1. Locally: `python -m data.download` (if not already done) to produce
# MAGIC    `outputs/data/raw/combined.jsonl`.
# MAGIC 2. Upload `combined.jsonl` to your workspace (e.g.
# MAGIC    `/Workspace/Users/<you>@similarweb.com/amlk/combined.jsonl`) or DBFS
# MAGIC    (`dbfs:/FileStore/amlk/combined.jsonl`). The `combined_jsonl_path` widget defaults to
# MAGIC    the workspace path below.
# MAGIC 3. Set the widgets below (repo URL/branch). For the Gemini API key, prefer uploading your
# MAGIC    local `.env` next to the data file in workspace, or to DBFS — **do not** type the raw
# MAGIC    key into the `gemini_api_key` widget in a shared/committed workspace.
# MAGIC 4. Attach this notebook to a GPU cluster (e.g. a single-node "Databricks Runtime ML GPU"
# MAGIC    instance) and run all cells top to bottom.

# COMMAND ----------

import torch

print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU detected — this will still work on CPU, just slower (see notebook header).")

# COMMAND ----------

# MAGIC %pip install bertopic sentence-transformers google-generativeai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bootstrap: pull in the real repo code
# MAGIC
# MAGIC Clones the AMLK repo so this notebook calls the exact same `evaluation.topic_clustering`
# MAGIC functions covered by the local tests, instead of duplicating clustering logic inline —
# MAGIC the same "single source of truth" reasoning as `evaluation/infer.py` for the
# MAGIC evaluation-observation notebook. Point `repo_url`/`repo_branch` at a fork/branch if
# MAGIC you're testing uncommitted changes.
# MAGIC
# MAGIC **Never hardcode the API key in this file** — it's tracked in a public git repo, and
# MAGIC anything committed stays in git history even after being removed. Secrets are resolved
# MAGIC in this order (mirrors the `.env`-upload pattern already used for the Colab
# MAGIC evaluation-observation notebook, see `.claude/skills/colab-cli/SKILL.md`):
# MAGIC 1. A Databricks secret scope, if you've set one up (`dbutils.secrets.get`).
# MAGIC 2. An `.env` file uploaded to DBFS (e.g. via the "Data" UI or `dbutils.fs.cp` from your
# MAGIC    local `.env`) — read at runtime, never stored in the notebook.
# MAGIC 3. The `gemini_api_key` widget, as a last resort — note that widget values can be saved
# MAGIC    with the notebook, so avoid this in a shared/committed workspace.

# COMMAND ----------

dbutils.widgets.text("repo_url", "https://github.com/avrymi-asraf/AMLK-the-true-title.git", "Repo URL")
dbutils.widgets.text("repo_branch", "main", "Branch")
dbutils.widgets.text(
    "code_root",
    "/Workspace/Users/amit.benbenishti@similarweb.com/amlk",
    "Local repo root (Workspace) — used if evaluation/topic_clustering.py is here",
)
dbutils.widgets.text(
    "combined_jsonl_path",
    "/Workspace/Users/amit.benbenishti@similarweb.com/amlk/combined.jsonl",
    "combined.jsonl path (Workspace or /dbfs/...)",
)
dbutils.widgets.text(
    "env_file_path",
    "/Workspace/Users/amit.benbenishti@similarweb.com/amlk/.env",
    "Uploaded .env path (Workspace or /dbfs/..., optional)",
)
dbutils.widgets.text("secret_scope", "", "Databricks secret scope (optional)")
dbutils.widgets.text("gemini_api_key", "", "GEMINI_API_KEY (last resort — prefer .env or a secret scope)")
dbutils.widgets.text("min_cluster_size", "40", "HDBSCAN min_cluster_size (topic granularity)")
dbutils.widgets.text("min_samples", "10", "HDBSCAN min_samples (lower = less raw noise; blank = tie to min_cluster_size)")
dbutils.widgets.dropdown("reduce_outliers", "True", ["True", "False"], "Reassign noise docs to nearest topic (embedding similarity)")
dbutils.widgets.text("nr_topics", "auto", "Merge near-duplicate topics: 'auto', an int, or blank to skip")
dbutils.widgets.text("record_limit", "0", "Max records (0 = all; try 500 for a smoke test)")

# COMMAND ----------

import os
import shutil
import subprocess
import sys
from pathlib import Path

_TOPIC_CLUSTERING = Path("evaluation/topic_clustering.py")


def _resolve_repo_dir() -> str:
    """Prefer a Workspace copy of the repo (uncommitted code); else shallow-clone from GitHub."""
    workspace = dbutils.widgets.get("code_root")
    if (Path(workspace) / _TOPIC_CLUSTERING).is_file():
        print(f"Using workspace repo at {workspace}")
        return workspace

    repo_dir = "/tmp/amlk-repo"
    marker = Path(repo_dir) / _TOPIC_CLUSTERING
    if not marker.is_file():
        if Path(repo_dir).exists():
            shutil.rmtree(repo_dir)
        subprocess.run(
            ["git", "clone", "--branch", dbutils.widgets.get("repo_branch"),
             "--depth", "1", dbutils.widgets.get("repo_url"), repo_dir],
            check=True,
        )
    if not marker.is_file():
        raise ModuleNotFoundError(
            f"{_TOPIC_CLUSTERING} not found in workspace ({workspace}) or after git clone "
            f"({repo_dir}). Either push the topic-clustering code to GitHub and re-run, or "
            "upload these files under your workspace amlk/ folder:\n"
            "  evaluation/__init__.py\n"
            "  evaluation/topic_clustering.py\n"
            "  evaluation/evaluate.py\n"
            "  evaluation/gemini_client.py"
        )
    print(f"Using git clone at {repo_dir}")
    return repo_dir


repo_dir = _resolve_repo_dir()
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)


def _load_gemini_api_key() -> str:
    """Secret scope > uploaded .env > widget, in that order. Never hardcode a key in this file
    — it's tracked in a public git repo and committed secrets stay in history permanently."""
    scope = dbutils.widgets.get("secret_scope")
    if scope:
        return dbutils.secrets.get(scope=scope, key="gemini_api_key")

    env_path = dbutils.widgets.get("env_file_path")
    if os.path.exists(env_path):
        for line in Path(env_path).read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    return dbutils.widgets.get("gemini_api_key")


os.environ["GEMINI_API_KEY"] = _load_gemini_api_key()
assert os.environ["GEMINI_API_KEY"], (
    "No GEMINI_API_KEY found. Upload your local .env to the env_file_path widget's DBFS "
    "location (dbutils.fs.cp), set up a secret scope, or fill the gemini_api_key widget."
)

# COMMAND ----------

import json
from pathlib import Path

combined_path = dbutils.widgets.get("combined_jsonl_path")
if not Path(combined_path).is_file():
    raise FileNotFoundError(
        f"{combined_path} not found. Upload combined.jsonl to your workspace folder "
        f"(e.g. /Workspace/Users/amit.benbenishti@similarweb.com/amlk/combined.jsonl) "
        f"or set the combined_jsonl_path widget."
    )

with open(combined_path, encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]
print(f"Loaded {len(records)} records from {combined_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Tuning note (2026-07-04):** the first full run put 51% of docs in noise (cluster -1) and
# MAGIC produced near-duplicate topic names ("תקשורת ומדיה" vs "תקשורת וטלוויזיה") whose keywords
# MAGIC were mostly years/IDs/Latin site names (ynet, nrg, bbc...) rather than real Hebrew topic
# MAGIC words — the vectorizer's default English-tuned token pattern let them dominate. Fixed in
# MAGIC `evaluation/topic_clustering.py`: a Hebrew-only c-TF-IDF vectorizer, `min_samples` decoupled
# MAGIC from `min_cluster_size` (less raw noise), and BERTopic's own outlier-reduction +
# MAGIC "auto" topic-merging passes (mitigate remaining noise / near-duplicates). See that module's
# MAGIC `fit_topics()` docstring for details. Re-run this notebook end to end to pick up the fix.

# COMMAND ----------

from evaluation.topic_clustering import cluster_dataset, topic_summary, write_topics

# Smoke-test first: set record_limit widget to e.g. 500 before a full 10k run.
record_limit = int(dbutils.widgets.get("record_limit") or "0")
if record_limit > 0:
    print(f"record_limit={record_limit} — using a subset for this run", flush=True)
    records = records[:record_limit]

_min_samples_raw = dbutils.widgets.get("min_samples").strip()
_nr_topics_raw = dbutils.widgets.get("nr_topics").strip()
_nr_topics = _nr_topics_raw if not _nr_topics_raw or _nr_topics_raw == "auto" else int(_nr_topics_raw)

print(f"Starting cluster_dataset on {len(records)} records...", flush=True)
rows, topic_model, embeddings = cluster_dataset(
    records,
    min_cluster_size=int(dbutils.widgets.get("min_cluster_size")),
    min_samples=int(_min_samples_raw) if _min_samples_raw else None,
    reduce_outliers=dbutils.widgets.get("reduce_outliers") == "True",
    nr_topics=_nr_topics or None,
)
n_clusters = len(set(r["cluster_id"] for r in rows))
print(f"Discovered {n_clusters} clusters (including noise, cluster_id=-1)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sanity-check the discovered taxonomy before trusting it

# COMMAND ----------

summary_rows = topic_summary(rows)
display(spark.createDataFrame(
    [(t["cluster_id"], t["topic_label"], t["count"], ", ".join(t["keywords"])) for t in summary_rows],
    ["cluster_id", "topic_label", "count", "keywords"],
))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize the clusters
# MAGIC
# MAGIC A 2D projection of the embeddings (a fresh UMAP run for plotting, separate from the 5D
# MAGIC one used for HDBSCAN clustering above), colored by topic. Hover a point to see its
# MAGIC summary text. Cluster -1 (noise) is included so you can see how much of the corpus didn't
# MAGIC fit a real topic.

# COMMAND ----------

from evaluation.topic_clustering import plot_clusters

summaries = [r["summary"] for r in records]
fig = plot_clusters(topic_model, summaries, embeddings)
displayHTML(fig.to_html(include_plotlyjs="cdn"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Structural style labels (rule-based, local — no GPU/API)
# MAGIC
# MAGIC A second, independent dimension over the same summaries: not *what topic* an article is
# MAGIC about, but *what format* its summary takes — a single sentence, several sentences, a
# MAGIC "headline | headline | headline" pipe-separated digest, or a question-style headline.
# MAGIC `evaluation/style_labels.py` is pure regex and runs instantly on CPU; it's included here
# MAGIC (rather than only as a standalone local script) so it merges into the same `topics.jsonl`
# MAGIC artifact and can be cross-tabbed against the topic clusters below.

# COMMAND ----------

from evaluation.style_labels import label_dataset as label_style
from evaluation.style_labels import style_summary

style_rows = label_style(records)  # aligned 1:1 with records/rows — same order, no join needed
for row, style_row in zip(rows, style_rows):
    row["style_label"] = style_row["style_label"]

print("Style label distribution:")
print(json.dumps(style_summary(style_rows), indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Topic x style crosstab
# MAGIC
# MAGIC E.g. do certain topics skew toward multi-headline digests more than others?

# COMMAND ----------

import pandas as pd

df = pd.DataFrame(rows)
crosstab = pd.crosstab(df["topic_label"], df["style_label"]).reset_index()
display(spark.createDataFrame(crosstab))

# COMMAND ----------

from pathlib import Path

topics_path = Path("/dbfs/FileStore/amlk/topics.jsonl")
summary_path = Path("/dbfs/FileStore/amlk/topics-summary.json")
write_topics(rows, topics_path, summary_path)  # rows now carry both topic_label and style_label
print(f"Wrote {topics_path} and {summary_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download the outputs back into the repo
# MAGIC
# MAGIC Grab both files from the Databricks "Data" > "DBFS" browser (or the workspace's
# MAGIC `/files/amlk/topics.jsonl` FileStore URL) and save them locally as:
# MAGIC - `outputs/data/raw/topics.jsonl` (now carries both `topic_label` and `style_label`)
# MAGIC - `outputs/results/topics-summary.json`
# MAGIC
# MAGIC Then, locally (no GPU/Databricks needed for this step), stratify any predictions file by
# MAGIC either dimension:
# MAGIC ```bash
# MAGIC python -m evaluation.stratify_by_topic \
# MAGIC   --predictions outputs/results/predictions-finetuned.jsonl \
# MAGIC   --labels outputs/data/raw/topics.jsonl --label-field topic_label \
# MAGIC   --errors outputs/results/finetuned-v3.errors.json \
# MAGIC   --output outputs/results/finetuned-by-topic.json
# MAGIC
# MAGIC python -m evaluation.stratify_by_topic \
# MAGIC   --predictions outputs/results/predictions-finetuned.jsonl \
# MAGIC   --labels outputs/data/raw/topics.jsonl --label-field style_label \
# MAGIC   --errors outputs/results/finetuned-v3.errors.json \
# MAGIC   --output outputs/results/finetuned-by-style.json
# MAGIC ```
