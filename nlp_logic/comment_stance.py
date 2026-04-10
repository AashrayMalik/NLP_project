"""
comment_stance.py — Stance detection & argument summarisation
==============================================================
For each major topic (top N by post count):
  1. Build a broader discussion frame from the most upvoted posts
  2. Classify each comment's stance (for/opposing/neutral) using
     zero-shot classification with facebook/bart-large-mnli
  3. Group users by their dominant stance
  4. Extract top-scored comments per side as representative arguments

Design choices:
  - Zero-shot classification avoids the need for labelled training data
  - BART-large-MNLI is a strong baseline for NLI-based stance detection
  - We sample comments per topic (max 500) to keep runtime reasonable
  - Extractive summarisation (top quotes) is fully offline and
    reproducible, unlike generative approaches requiring API keys
"""

import json
import logging
import sqlite3
from pathlib import Path

from nlp_logic.topic_insights import build_topic_display_label

logger = logging.getLogger(__name__)

DEFAULT_DB = Path(__file__).resolve().parent.parent / "scraping" / "data" / "jobs_posts.db"
CACHE_DIR = Path(__file__).resolve().parent.parent / "scraping" / "data"

# How many topics to analyse (None = all non-outlier topics)
TOP_N_TOPICS = None
# Max comments to classify per topic (keep highest-scoring comments only)
MAX_COMMENTS_PER_TOPIC = 100
# Min comment length to consider for stance
MIN_COMMENT_LENGTH = 30
# Confidence guardrails so weak predictions fall into a neutral bucket
MIN_STANCE_CONFIDENCE = 0.45
MIN_STANCE_MARGIN = 0.03
# Runtime tuning for zero-shot inference
PARENT_TITLE_CHARS = 96
COMMENT_SNIPPET_CHARS = 180
MODEL_MAX_LENGTH = 256
BATCH_SIZE = 16


def load_topic_data(db_path: str | Path | None = None) -> dict:
    """
    Load topic metadata and identify top N topics.

    Returns dict with topic info keyed by topic_id.
    """
    db = str(db_path or DEFAULT_DB)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT id, label, keywords, post_count
        FROM topics
        WHERE id != -1
        ORDER BY post_count DESC
    """
    params: tuple = ()
    if TOP_N_TOPICS is not None:
        query += "\nLIMIT ?"
        params = (TOP_N_TOPICS,)

    rows = conn.execute(query, params).fetchall()

    topics = {}
    for r in rows:
        topics[r["id"]] = {
            "id": r["id"],
            "label": r["label"],
            "keywords": json.loads(r["keywords"]) if r["keywords"] else [],
            "post_count": r["post_count"],
        }

    conn.close()
    return topics


def load_reference_posts(db_path: str | Path, topic_id: int, limit: int = 5) -> list[dict]:
    """Load the top-scoring posts used to build the topic discussion frame."""
    db = str(db_path or DEFAULT_DB)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT p.title, p.body, p.score, COALESCE(NULLIF(TRIM(p.flair), ''), 'No Flair') AS flair
        FROM posts p
        JOIN post_topics pt ON p.id = pt.post_id
        WHERE pt.topic_id = ?
        ORDER BY p.score DESC
        LIMIT ?
    """, (topic_id, limit)).fetchall()
    conn.close()

    return [
        {
            "title": row["title"] or "",
            "body": row["body"] or "",
            "score": row["score"],
            "flair": row["flair"],
        }
        for row in rows
    ]


def build_discussion_frame(topic_info: dict, reference_posts: list[dict]) -> dict:
    """
    Build a broader reference frame for stance classification.

    The old implementation anchored on a single post title, which made the
    labels brittle. Here we blend the cleaned topic label with several
    representative post titles so comments are classified against a wider
    framing of the discussion.
    """
    display_label = build_topic_display_label(topic_info.get("keywords", []), fallback=topic_info["label"])
    titles = [post["title"].strip() for post in reference_posts if post.get("title")]
    title_samples = titles[:3]

    if title_samples:
        reference_frame = (
            f"Topic focus: {display_label}. Representative posts: "
            + " | ".join(title[:140] for title in title_samples)
        )
        anchor_title = title_samples[0]
    else:
        reference_frame = f"Topic focus: {display_label}"
        anchor_title = display_label

    return {
        "display_label": display_label,
        "anchor_title": anchor_title,
        "reference_frame": reference_frame[:500],
        "reference_posts": [
            {"title": post["title"][:200], "score": post["score"], "flair": post["flair"]}
            for post in reference_posts[:3]
        ],
    }


def load_comments_for_topic(db_path: str | Path, topic_id: int) -> list[dict]:
    """Load comments for posts in a given topic, sorted by score."""
    db = str(db_path or DEFAULT_DB)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

    rows = conn.execute("""
        SELECT c.id, c.body, c.score, c.author, COALESCE(p.title, '') AS parent_title
        FROM comments c
        JOIN posts p ON c.post_id = p.id
        JOIN post_topics pt ON c.post_id = pt.post_id
        WHERE pt.topic_id = ?
          AND LENGTH(c.body) >= ?
          AND c.author != '[deleted]'
        ORDER BY c.score DESC
        LIMIT ?
    """, (topic_id, MIN_COMMENT_LENGTH, MAX_COMMENTS_PER_TOPIC)).fetchall()

    conn.close()
    return [
        {
            "id": row[0],
            "body": row[1],
            "score": row[2],
            "author": row[3],
            "parent_title": row[4],
        }
        for row in rows
    ]


def classify_stances(comments: list[dict], classifier) -> list[dict]:
    """
    Classify each comment's stance relative to its own parent post framing.

    We still aggregate results at the topic level, but the inference happens
    locally so a comment is judged against the post it is actually replying to.
    """
    if not comments:
        return []

    sequences = [
        (
            f"Post: {comment['parent_title'][:PARENT_TITLE_CHARS]} "
            f"Comment: {comment['body'][:COMMENT_SNIPPET_CHARS]}"
        )
        for comment in comments
    ]
    hypothesis = "This comment is {}."

    results = []

    for i in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[i:i + BATCH_SIZE]
        batch_comments = comments[i:i + BATCH_SIZE]

        try:
            outputs = classifier(
                batch,
                candidate_labels=[
                    "supporting the parent post",
                    "opposing the parent post",
                    "neutral or unclear",
                ],
                hypothesis_template=hypothesis,
                multi_label=False,
                batch_size=BATCH_SIZE,
                truncation=True,
                max_length=MODEL_MAX_LENGTH,
            )

            # Handle single result vs list
            if isinstance(outputs, dict):
                outputs = [outputs]

            for comment, output in zip(batch_comments, outputs):
                labels = output["labels"]
                scores = output["scores"]
                top_label = labels[0]
                top_score = float(scores[0])
                margin = top_score - float(scores[1]) if len(scores) > 1 else top_score

                if (
                    top_label == "neutral or unclear"
                    or top_score < MIN_STANCE_CONFIDENCE
                    or margin < MIN_STANCE_MARGIN
                ):
                    stance = "neutral"
                elif top_label == "supporting the parent post":
                    stance = "for"
                else:
                    stance = "opposing"

                results.append({
                    "comment_id": comment["id"],
                    "body": comment["body"],
                    "score": comment["score"],
                    "author": comment["author"],
                    "stance": stance,
                    "confidence": round(top_score, 3),
                    "margin": round(margin, 3),
                })
        except Exception as e:
            logger.warning(f"Batch classification error: {e}")
            # Fall back to neutral for failed batch
            for c in batch_comments:
                results.append({
                    "comment_id": c["id"],
                    "body": c["body"],
                    "score": c["score"],
                    "author": c["author"],
                    "stance": "neutral",
                    "confidence": 0.5,
                    "margin": 0.0,
                })

    return results


def summarise_arguments(stance_results: list[dict]) -> dict:
    """
    Extract top-scored comments per side as representative arguments.
    """
    total = max(len(stance_results), 1)
    grouped = {
        "for": [r for r in stance_results if r["stance"] == "for"],
        "opposing": [r for r in stance_results if r["stance"] == "opposing"],
        "neutral": [r for r in stance_results if r["stance"] == "neutral"],
    }

    summary = {}
    for bucket, rows in grouped.items():
        rows.sort(key=lambda item: item["score"], reverse=True)
        users = set(row["author"] for row in rows)
        summary[bucket] = {
            "count": len(rows),
            "pct": round(100 * len(rows) / total, 1),
            "user_count": len(users),
            "top_arguments": [
                {
                    "body": row["body"][:500],
                    "score": row["score"],
                    "author": row["author"],
                    "confidence": row["confidence"],
                }
                for row in rows[:5]
            ],
        }

    # Backward-compatible aliases for any older UI code.
    summary["support"] = summary["for"]
    summary["oppose"] = summary["opposing"]
    return summary


def persist_results(db_path: str | Path, all_stances: dict):
    """Save stance analysis results as JSON cache."""
    cache_path = Path(str(db_path or DEFAULT_DB)).parent / "stance_cache.json"
    with open(cache_path, "w") as f:
        json.dump(all_stances, f, indent=2, default=str)
    logger.info(f"Stance analysis cached → {cache_path}")


def run_pipeline(db_path: str | Path | None = None):
    """Full stance detection pipeline for top topics."""
    from transformers import pipeline

    db = db_path or DEFAULT_DB

    logger.info("=" * 60)
    logger.info("  Stance Detection Pipeline — r/jobs")
    logger.info("=" * 60)

    # Load zero-shot classifier
    logger.info("Loading facebook/bart-large-mnli classifier...")
    logger.info("Using CPU inference for predictable offline runtime.")
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1,
    )
    logger.info("Classifier loaded.")

    # Get top topics
    topics = load_topic_data(db)
    logger.info(f"Analysing stance for {len(topics)} topics")

    all_stances = {}

    for topic_id, topic_info in topics.items():
        logger.info(f"  Topic {topic_id}: {topic_info['label'][:50]}")

        reference_posts = load_reference_posts(db, topic_id)
        frame = build_discussion_frame(topic_info, reference_posts)
        logger.info(f"    Reference frame: {frame['anchor_title'][:80]}")

        # Load comments
        comments = load_comments_for_topic(db, topic_id)
        logger.info(f"    Comments loaded: {len(comments)}")

        if not comments:
            continue

        # Classify stances
        stance_results = classify_stances(comments, classifier)

        # Summarise
        summary = summarise_arguments(stance_results)
        summary["reference_frame"] = frame["reference_frame"]
        summary["dominant_position"] = frame["anchor_title"]
        summary["reference_posts"] = frame["reference_posts"]
        summary["topic_id"] = topic_id
        summary["topic_label"] = frame["display_label"]
        summary["total_comments_analysed"] = len(comments)
        summary["stance_method"] = {
            "labels": ["for", "opposing", "neutral"],
            "min_confidence": MIN_STANCE_CONFIDENCE,
            "min_margin": MIN_STANCE_MARGIN,
        }

        all_stances[str(topic_id)] = summary

        logger.info(f"    For: {summary['for']['pct']:.0f}% "
                     f"({summary['for']['count']}) | "
                     f"Opposing: {summary['opposing']['pct']:.0f}% "
                     f"({summary['opposing']['count']}) | "
                     f"Neutral: {summary['neutral']['pct']:.0f}% "
                     f"({summary['neutral']['count']})")

    # Persist
    persist_results(db, all_stances)

    # Print summary
    print("\n" + "=" * 60)
    print("  Stance Detection Results")
    print("=" * 60)
    for tid, s in all_stances.items():
        print(f"  Topic {tid:>2s}: For {s['for']['pct']:5.1f}% | "
              f"Opposing {s['opposing']['pct']:5.1f}% | "
              f"Neutral {s['neutral']['pct']:5.1f}%  "
              f"({s['total_comments_analysed']} comments)")
    print("=" * 60)

    return all_stances


# cli
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run_pipeline()
