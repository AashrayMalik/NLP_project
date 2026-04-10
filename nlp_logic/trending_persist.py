"""
trending_persist.py — Trending vs Persistent topic classification
==================================================================
For each topic discovered by BERTopic, compute monthly post counts
and classify as either 'persistent' or 'trending' based on:
  - Presence: fraction of months the topic appears in
  - Coefficient of Variation (CV): std/mean of monthly counts
  - Spike detection: whether any single month exceeds 2x the mean

Design choices:
  - Persistent: appears in >= 70% of months AND CV < 0.5
  - Trending:   has a spike (some month > 2x mean) OR appears in < 50% of months
  - Seasonal:   everything else (appears regularly but with some variation)
  - Monthly granularity chosen because the dataset spans 12 months,
    giving enough data points for variance analysis without being
    too fine-grained (weekly would be too noisy).
"""

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DB = Path(__file__).resolve().parent.parent / "scraping" / "data" / "jobs_posts.db"
CACHE_DIR = Path(__file__).resolve().parent.parent / "scraping" / "data"


def compute_monthly_topic_counts(db_path: str | Path | None = None) -> dict:
    """
    For each topic, count how many posts appear per month.

    Returns dict: { topic_id: { month: count, ... }, ... }
    """
    db = str(db_path or DEFAULT_DB)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)

    rows = conn.execute("""
        SELECT pt.topic_id, p.month, COUNT(*) AS cnt
        FROM post_topics pt
        JOIN posts p ON pt.post_id = p.id
        WHERE pt.topic_id != -1
        GROUP BY pt.topic_id, p.month
        ORDER BY pt.topic_id, p.month
    """).fetchall()

    # Get all months in dataset
    all_months = [r[0] for r in conn.execute(
        "SELECT DISTINCT month FROM posts ORDER BY month"
    ).fetchall()]

    conn.close()

    # Build per-topic monthly counts (fill missing months with 0)
    topic_months = {}
    for topic_id, month, cnt in rows:
        if topic_id not in topic_months:
            topic_months[topic_id] = {m: 0 for m in all_months}
        topic_months[topic_id][month] = cnt

    return topic_months, all_months


def classify_topics(topic_months: dict, all_months: list[str]) -> list[dict]:
    """
    Classify each topic as persistent, trending, or seasonal.

    Uses relative thresholds based on the distribution of CV and trend
    slopes across topics, since in this dataset all topics appear every month.
    """
    n_months = len(all_months)
    results = []
    all_cvs = []
    all_slopes = []

    for topic_id, month_counts in topic_months.items():
        counts = np.array([month_counts.get(m, 0) for m in all_months])
        total = int(counts.sum())
        mean_count = float(counts.mean())
        std_count = float(counts.std())

        # Presence: fraction of months with > 0 posts
        active_months = int(np.sum(counts > 0))
        presence = active_months / n_months

        # Coefficient of variation
        cv = std_count / mean_count if mean_count > 0 else 0.0

        # Spike detection: any month > 2x the mean?
        has_spike = bool(np.any(counts > 2 * mean_count))

        # Linear trend (slope via least-squares regression)
        x = np.arange(n_months)
        if n_months > 1 and np.std(counts) > 0:
            slope = float(np.polyfit(x, counts, 1)[0])
        else:
            slope = 0.0
        # Normalise slope relative to mean
        norm_slope = slope / mean_count if mean_count > 0 else 0.0

        # Peak month
        peak_idx = int(np.argmax(counts))
        peak_month = all_months[peak_idx]
        peak_count = int(counts[peak_idx])

        all_cvs.append(cv)
        all_slopes.append(abs(norm_slope))

        results.append({
            "topic_id": int(topic_id),
            "trend_type": "",  # filled below
            "presence": round(presence, 3),
            "cv": round(cv, 3),
            "has_spike": has_spike,
            "slope": round(slope, 2),
            "norm_slope": round(norm_slope, 3),
            "mean_monthly": round(mean_count, 1),
            "total_posts": total,
            "peak_month": peak_month,
            "peak_count": peak_count,
            "monthly_counts": {m: int(c) for m, c in zip(all_months, counts)},
        })

    # relative classification using percentiles
    # Since all topics in r/jobs appear every month (100% presence),
    # we use relative CV and slope to separate them:
    #   - Bottom third CV + low slope → persistent (stable volume)
    #   - Top third CV or high slope → trending (growing/spiking)
    #   - Middle → seasonal (moderate variation)
    if all_cvs:
        cv_p33 = float(np.percentile(all_cvs, 33))
        cv_p66 = float(np.percentile(all_cvs, 66))
        slope_p66 = float(np.percentile(all_slopes, 66))

        for r in results:
            if r["presence"] < 0.5:
                r["trend_type"] = "trending"
            elif r["has_spike"]:
                r["trend_type"] = "trending"
            elif r["cv"] <= cv_p33 and abs(r["norm_slope"]) < slope_p66:
                r["trend_type"] = "persistent"
            elif r["cv"] >= cv_p66 or abs(r["norm_slope"]) >= slope_p66:
                r["trend_type"] = "trending"
            else:
                r["trend_type"] = "seasonal"

    return results


def persist_results(db_path: str | Path, classifications: list[dict]):
    """Save trend classification to the topics table and cache file."""
    db = str(db_path or DEFAULT_DB)
    conn = sqlite3.connect(db)

    # Add trend_type column if not exists
    try:
        conn.execute("ALTER TABLE topics ADD COLUMN trend_type TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    for c in classifications:
        conn.execute(
            "UPDATE topics SET trend_type = ? WHERE id = ?",
            (c["trend_type"], c["topic_id"]),
        )

    conn.commit()
    conn.close()

    # Save full classification data as JSON for the app
    cache_path = Path(str(db_path or DEFAULT_DB)).parent / "trending_cache.json"
    with open(cache_path, "w") as f:
        json.dump(classifications, f, indent=2)

    logger.info(f"Trending/persistent classification cached → {cache_path}")


def run_pipeline(db_path: str | Path | None = None):
    """Full trending vs persistent pipeline."""
    db = db_path or DEFAULT_DB

    logger.info("=" * 60)
    logger.info("  Trending vs Persistent Pipeline — r/jobs")
    logger.info("=" * 60)

    topic_months, all_months = compute_monthly_topic_counts(db)
    classifications = classify_topics(topic_months, all_months)

    persist_results(db, classifications)

    # Print summary
    persistent = [c for c in classifications if c["trend_type"] == "persistent"]
    trending = [c for c in classifications if c["trend_type"] == "trending"]
    seasonal = [c for c in classifications if c["trend_type"] == "seasonal"]

    print(f"\n  Persistent topics: {len(persistent)}")
    for c in persistent:
        print(f"    Topic {c['topic_id']:>2d}  presence={c['presence']:.0%}  "
              f"CV={c['cv']:.2f}  mean={c['mean_monthly']:.0f}/mo")

    print(f"\n  Trending topics: {len(trending)}")
    for c in trending:
        print(f"    Topic {c['topic_id']:>2d}  presence={c['presence']:.0%}  "
              f"CV={c['cv']:.2f}  spike={'yes' if c['has_spike'] else 'no'}  "
              f"peak={c['peak_month']}")

    if seasonal:
        print(f"\n  Seasonal topics: {len(seasonal)}")
        for c in seasonal:
            print(f"    Topic {c['topic_id']:>2d}  presence={c['presence']:.0%}  "
                  f"CV={c['cv']:.2f}")

    return classifications


# cli
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run_pipeline()
