"""
agg.py — Aggregate statistics for r/jobs subreddit database
=============================================================
Queries the cleaned SQLite DB and returns structured statistics:
  - Total posts, comments, unique authors
  - Posts/comments per month
  - Average post/comment length
  - Top flairs by count
  - Score distributions
  - Most active authors
  - Engagement tier breakdown
"""

import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "scraping" / "data" / "jobs_posts.db"


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Return a read-only connection to the database."""
    db = str(db_path or DEFAULT_DB)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def compute_overview(conn: sqlite3.Connection) -> dict:
    """High-level KPI numbers."""
    r = conn.execute("""
        SELECT
            COUNT(*)                                                    AS total_posts,
            COUNT(DISTINCT author)                                      AS unique_post_authors,
            AVG(LENGTH(body))                                           AS avg_post_length,
            AVG(score)                                                  AS avg_post_score,
            MIN(created_utc)                                            AS earliest_post,
            MAX(created_utc)                                            AS latest_post
        FROM posts
        WHERE author != '[deleted]'
    """).fetchone()

    c = conn.execute("""
        SELECT
            COUNT(*)                   AS total_comments,
            COUNT(DISTINCT author)     AS unique_comment_authors,
            AVG(LENGTH(body))          AS avg_comment_length,
            AVG(score)                 AS avg_comment_score
        FROM comments
        WHERE author != '[deleted]'
    """).fetchone()

    return {
        "total_posts":            r["total_posts"],
        "total_comments":         c["total_comments"],
        "unique_post_authors":    r["unique_post_authors"],
        "unique_comment_authors": c["unique_comment_authors"],
        "avg_post_length":        round(r["avg_post_length"] or 0, 1),
        "avg_comment_length":     round(c["avg_comment_length"] or 0, 1),
        "avg_post_score":         round(r["avg_post_score"] or 0, 1),
        "avg_comment_score":      round(c["avg_comment_score"] or 0, 1),
        "earliest_post":          r["earliest_post"],
        "latest_post":            r["latest_post"],
    }


def posts_per_month(conn: sqlite3.Connection) -> list[dict]:
    """Monthly post counts."""
    rows = conn.execute("""
        SELECT month, COUNT(*) AS cnt
        FROM posts
        GROUP BY month
        ORDER BY month
    """).fetchall()
    return [{"month": r["month"], "count": r["cnt"]} for r in rows]


def comments_per_month(conn: sqlite3.Connection) -> list[dict]:
    """Monthly comment counts (via post month)."""
    rows = conn.execute("""
        SELECT p.month, COUNT(*) AS cnt
        FROM comments c
        JOIN posts p ON c.post_id = p.id
        GROUP BY p.month
        ORDER BY p.month
    """).fetchall()
    return [{"month": r["month"], "count": r["cnt"]} for r in rows]


def flair_distribution(conn: sqlite3.Connection) -> list[dict]:
    """Post counts per flair, sorted descending."""
    rows = conn.execute("""
        SELECT COALESCE(flair, 'No Flair') AS flair, COUNT(*) AS cnt
        FROM posts
        GROUP BY flair
        ORDER BY cnt DESC
    """).fetchall()
    return [{"flair": r["flair"], "count": r["cnt"]} for r in rows]


def tier_breakdown(conn: sqlite3.Connection) -> dict:
    """High vs Low engagement tier counts."""
    rows = conn.execute("""
        SELECT tier, COUNT(*) AS cnt
        FROM posts
        GROUP BY tier
    """).fetchall()
    return {r["tier"]: r["cnt"] for r in rows}


def score_distribution_posts(conn: sqlite3.Connection) -> list[dict]:
    """Post score histogram buckets."""
    rows = conn.execute("""
        SELECT
            CASE
                WHEN score < 0   THEN 'Negative'
                WHEN score = 0   THEN '0'
                WHEN score <= 5  THEN '1-5'
                WHEN score <= 10 THEN '6-10'
                WHEN score <= 25 THEN '11-25'
                WHEN score <= 50 THEN '26-50'
                WHEN score <= 100 THEN '51-100'
                ELSE '100+'
            END AS bucket,
            COUNT(*) AS cnt
        FROM posts
        GROUP BY bucket
        ORDER BY MIN(score)
    """).fetchall()
    return [{"bucket": r["bucket"], "count": r["cnt"]} for r in rows]


def score_distribution_comments(conn: sqlite3.Connection) -> list[dict]:
    """Comment score histogram buckets."""
    rows = conn.execute("""
        SELECT
            CASE
                WHEN score < 0   THEN 'Negative'
                WHEN score = 0   THEN '0'
                WHEN score <= 5  THEN '1-5'
                WHEN score <= 10 THEN '6-10'
                WHEN score <= 25 THEN '11-25'
                WHEN score <= 50 THEN '26-50'
                WHEN score <= 100 THEN '51-100'
                ELSE '100+'
            END AS bucket,
            COUNT(*) AS cnt
        FROM comments
        GROUP BY bucket
        ORDER BY MIN(score)
    """).fetchall()
    return [{"bucket": r["bucket"], "count": r["cnt"]} for r in rows]


def top_authors_by_posts(conn: sqlite3.Connection, limit: int = 15) -> list[dict]:
    """Most active authors by post count."""
    rows = conn.execute("""
        SELECT author, COUNT(*) AS cnt, AVG(score) AS avg_score
        FROM posts
        WHERE author != '[deleted]'
        GROUP BY author
        ORDER BY cnt DESC
        LIMIT ?
    """, (limit,)).fetchall()
    return [{"author": r["author"], "post_count": r["cnt"],
             "avg_score": round(r["avg_score"], 1)} for r in rows]


def top_authors_by_comments(conn: sqlite3.Connection, limit: int = 15) -> list[dict]:
    """Most active authors by comment count."""
    rows = conn.execute("""
        SELECT author, COUNT(*) AS cnt, AVG(score) AS avg_score
        FROM comments
        WHERE author != '[deleted]'
        GROUP BY author
        ORDER BY cnt DESC
        LIMIT ?
    """, (limit,)).fetchall()
    return [{"author": r["author"], "comment_count": r["cnt"],
             "avg_score": round(r["avg_score"], 1)} for r in rows]


def avg_length_per_month(conn: sqlite3.Connection) -> list[dict]:
    """Average post and comment body length per month."""
    rows = conn.execute("""
        SELECT
            p.month,
            AVG(LENGTH(p.body)) AS avg_post_len,
            AVG(LENGTH(c.body)) AS avg_comment_len
        FROM posts p
        LEFT JOIN comments c ON c.post_id = p.id
        GROUP BY p.month
        ORDER BY p.month
    """).fetchall()
    return [{"month": r["month"],
             "avg_post_length": round(r["avg_post_len"] or 0, 1),
             "avg_comment_length": round(r["avg_comment_len"] or 0, 1)}
            for r in rows]


def compute_all(db_path: str | Path | None = None) -> dict:
    """Run every aggregate query and return a single dict."""
    conn = get_connection(db_path)
    try:
        return {
            "overview":               compute_overview(conn),
            "posts_per_month":        posts_per_month(conn),
            "comments_per_month":     comments_per_month(conn),
            "flair_distribution":     flair_distribution(conn),
            "tier_breakdown":         tier_breakdown(conn),
            "score_dist_posts":       score_distribution_posts(conn),
            "score_dist_comments":    score_distribution_comments(conn),
            "top_authors_posts":      top_authors_by_posts(conn),
            "top_authors_comments":   top_authors_by_comments(conn),
            "avg_length_per_month":   avg_length_per_month(conn),
        }
    finally:
        conn.close()


#test
if __name__ == "__main__":
    import json
    stats = compute_all()
    print(json.dumps(stats, indent=2, default=str))
