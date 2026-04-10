"""
topic_modelling.py — BERTopic-based topic modelling for r/jobs posts
=====================================================================
Pipeline:
  1. Load all post bodies from the cleaned SQLite DB
  2. Embed using sentence-transformers (all-MiniLM-L6-v2)
  3. Reduce dimensions with UMAP, cluster with HDBSCAN
  4. Extract topics via BERTopic's c-TF-IDF representation
  5. Merge small/noisy topics to reach 10-15 meaningful clusters
  6. Persist results back to the DB (topics + post_topics tables)
  7. Save embeddings and model artefacts for the Streamlit app

Design choices:
  - BERTopic over LDA: captures semantic meaning via transformer
    embeddings rather than relying on bag-of-words co-occurrence.
  - all-MiniLM-L6-v2: best speed/quality trade-off (~80MB, 384-d).
  - HDBSCAN: density-based clustering that auto-discovers cluster count
    and gracefully handles noise (outlier) points.
  - UMAP: non-linear dimensionality reduction preserving local+global
    structure better than PCA/t-SNE for clustering.
  - c-TF-IDF: BERTopic's class-based TF-IDF extracts keywords that
    are distinctive to each topic, not just frequent.
"""

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np

from nlp_logic.topic_insights import build_topic_display_label

logger = logging.getLogger(__name__)

DEFAULT_DB = Path(__file__).resolve().parent.parent / "scraping" / "data" / "jobs_posts.db"
ARTEFACT_DIR = Path(__file__).resolve().parent.parent / "scraping" / "data"

# target topic range
MIN_TOPICS = 8
MAX_TOPICS = 18


def load_posts(db_path: str | Path | None = None) -> tuple[list[str], list[str]]:
    """Load post IDs and bodies from DB. Returns (ids, bodies)."""
    db = str(db_path or DEFAULT_DB)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute("""
        SELECT id, COALESCE(title, '') || ' ' || COALESCE(body, '') AS text
        FROM posts
        WHERE LENGTH(COALESCE(body, '')) > 30
        ORDER BY id
    """).fetchall()
    conn.close()
    ids = [r[0] for r in rows]
    bodies = [r[1] for r in rows]
    logger.info(f"Loaded {len(ids)} posts for topic modelling")
    return ids, bodies


def build_topic_model(
    bodies: list[str],
    min_topic_size: int = 150,
    embedding_model: str = "all-MiniLM-L6-v2",
):
    """
    Build and fit a BERTopic model.

    Returns (model, topics, probs, embeddings).
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    logger.info(f"Embedding {len(bodies)} documents with {embedding_model}...")
    embedder = SentenceTransformer(embedding_model)
    embeddings = embedder.encode(bodies, show_progress_bar=True, batch_size=64)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    # domain-specific stop words for r/jobs — these terms are ubiquitous
    # across all topics and drown out distinguishing keywords
    DOMAIN_STOP_WORDS = [
        "job", "jobs", "work", "working", "worked", "worker", "workers",
        "company", "companies", "employer", "employers", "employee", "employees",
        "position", "role", "career", "hired", "hiring", "hire",
        "just", "like", "don", "ve", "got", "get", "getting", "know",
        "want", "going", "went", "would", "could", "really", "thing",
        "things", "think", "said", "say", "told", "people", "person",
        "one", "time", "day", "year", "years", "new", "good", "bad",
        "make", "made", "feel", "right", "way", "need", "ll", "didn",
        "doesn", "wasn", "isn", "aren", "won", "hasn", "haven",
        "re", "ve", "let",
    ]

    # merge sklearn's English stop words with domain-specific ones
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    combined_stop_words = list(ENGLISH_STOP_WORDS | set(DOMAIN_STOP_WORDS))

    # CountVectorizer with stop words to get meaningful topic keywords
    vectorizer_model = CountVectorizer(
        stop_words=combined_stop_words,
        ngram_range=(1, 2),
        min_df=5,
    )

    model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        verbose=True,
        calculate_probabilities=True,
    )

    logger.info("Fitting BERTopic model...")
    topics, probs = model.fit_transform(bodies, embeddings)

    # reducing outliers =
    n_outliers = sum(1 for t in topics if t == -1)
    logger.info(f"Outliers before reduction: {n_outliers} ({100*n_outliers/len(topics):.1f}%)")

    if n_outliers > 0:
        # assigning outliers to their nearest topic based on embeddings
        new_topics = model.reduce_outliers(bodies, topics, strategy="embeddings",
                                           embeddings=embeddings, threshold=0.5)
        model.update_topics(bodies, topics=new_topics, vectorizer_model=vectorizer_model)
        topics = new_topics

        n_outliers = sum(1 for t in topics if t == -1)
        logger.info(f"Outliers after reduction: {n_outliers} ({100*n_outliers/len(topics):.1f}%)")

    # merge to target range
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    logger.info(f"Topic count before merge: {n_topics}")

    if n_topics > MAX_TOPICS:
        logger.info(f"Reducing topics from {n_topics} to {MAX_TOPICS}...")
        model.reduce_topics(bodies, nr_topics=MAX_TOPICS)
        topics = model.topics_

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    logger.info(f"Final topic count: {n_topics}")

    return model, topics, probs, embeddings


def extract_topic_info(model, topics: list[int], post_ids: list[str]) -> dict:
    """
    Extract structured topic metadata from the fitted model.

    Returns a dict with:
      - "topics": list of topic dicts (id, label, keywords, count, share_pct)
      - "post_assignments": list of (post_id, topic_id) tuples
      - "representative_docs": dict mapping topic_id → list of doc indices
    """
    topic_info = model.get_topic_info()
    total_docs = len(post_ids)

    topic_list = []
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            label = "Outlier / Uncategorised"
        else:
            kw = model.get_topic(tid)
            keywords = [w for w, _ in kw[:10]] if kw else []
            label = build_topic_display_label(keywords, fallback=f"Topic {tid}")

        topic_list.append({
            "id": int(tid),
            "label": label,
            "keywords": [w for w, _ in (model.get_topic(tid) or [])][:10],
            "count": int(row["Count"]),
            "share_pct": round(100.0 * row["Count"] / total_docs, 2),
        })

    # Post to topic mapping
    post_assignments = list(zip(post_ids, [int(t) for t in topics]))

    #representative docs per topic
    rep_docs = {}
    try:
        rd = model.get_representative_docs()
        if rd:
            rep_docs = {int(k): v for k, v in rd.items()}
    except Exception:
        pass

    return {
        "topics": topic_list,
        "post_assignments": post_assignments,
        "representative_docs": rep_docs,
    }


def persist_results(
    db_path: str | Path,
    topic_info: dict,
    embeddings: np.ndarray,
    model,
):
    """
    Save results to the database and filesystem.

    Creates/replaces:
      - DB table `topics`
      - DB table `post_topics`
      - File `data/topic_embeddings.npy`
    """
    db = str(db_path or DEFAULT_DB)
    conn = sqlite3.connect(db)

    # create tables
    conn.execute("DROP TABLE IF EXISTS post_topics")
    conn.execute("DROP TABLE IF EXISTS topics")

    conn.execute("""
        CREATE TABLE topics (
            id          INTEGER PRIMARY KEY,
            label       TEXT,
            keywords    TEXT,       -- JSON array
            post_count  INTEGER,
            share_pct   REAL
        )
    """)

    conn.execute("""
        CREATE TABLE post_topics (
            post_id     TEXT PRIMARY KEY REFERENCES posts(id),
            topic_id    INTEGER REFERENCES topics(id)
        )
    """)

    # insert topic metadata
    for t in topic_info["topics"]:
        conn.execute(
            "INSERT INTO topics (id, label, keywords, post_count, share_pct) VALUES (?,?,?,?,?)",
            (t["id"], t["label"], json.dumps(t["keywords"]), t["count"], t["share_pct"]),
        )

    # post-topic assignments
    conn.executemany(
        "INSERT OR REPLACE INTO post_topics (post_id, topic_id) VALUES (?, ?)",
        topic_info["post_assignments"],
    )

    conn.commit()
    conn.close()
    logger.info(f"Persisted {len(topic_info['topics'])} topics and "
                f"{len(topic_info['post_assignments'])} post assignments to DB")

    #save embeddings
    emb_path = ARTEFACT_DIR / "topic_embeddings.npy"
    np.save(emb_path, embeddings)
    logger.info(f"Saved embeddings → {emb_path}")

    # docs as json
    rep_path = ARTEFACT_DIR / "representative_docs.json"
    with open(rep_path, "w") as f:
        json.dump(topic_info.get("representative_docs", {}), f, indent=2)
    logger.info(f"Saved representative docs → {rep_path}")


def run_pipeline(db_path: str | Path | None = None, min_topic_size: int = 150):
    """
    Full topic modelling pipeline: load → embed → cluster → persist.
    """
    db = db_path or DEFAULT_DB
    logger.info("=" * 60)
    logger.info("  Topic Modelling Pipeline — r/jobs")
    logger.info("=" * 60)

    post_ids, bodies = load_posts(db)
    model, topics, probs, embeddings = build_topic_model(bodies, min_topic_size)
    info = extract_topic_info(model, topics, post_ids)

    persist_results(db, info, embeddings, model)

    #print summary 
    print("\n" + "=" * 60)
    print("  Topic Modelling Results")
    print("=" * 60)
    for t in sorted(info["topics"], key=lambda x: x["count"], reverse=True):
        tag = "OUTLIER" if t["id"] == -1 else f"Topic {t['id']}"
        print(f"  {tag:>12s}  {t['count']:>5,} posts ({t['share_pct']:5.1f}%)  "
              f"{t['label'][:60]}")
    print("=" * 60)

    return info


#cli
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run_pipeline()
