"""
precompute.py — Run NLP pipelines and cache results

Executes each pipeline sequentially and persists results to the
SQLite DB and filesystem so the Streamlit app loads instantly.

Usage:
    python precompute.py                        # run all pipelines
    python precompute.py --only agg             # run only aggregate stats
    python precompute.py --only topics          # run only topic modelling
    python precompute.py --only trending        # run only trending/persistent
    python precompute.py --only stance          # run only stance detection
    python precompute.py --db path/to/jobs.db   # custom DB path
"""

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_DB = Path(__file__).resolve().parent / "scraping" / "data" / "jobs_posts.db"
CACHE_DIR = Path(__file__).resolve().parent / "scraping" / "data"


def run_aggregate(db_path: Path):
    """Run aggregate statistics and cache to JSON."""
    from nlp_logic.agg import compute_all

    logger.info("Running aggregate statistics...")
    t0 = time.time()
    stats = compute_all(db_path)
    cache_path = CACHE_DIR / "agg_cache.json"
    with open(cache_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Aggregate stats cached → {cache_path}  ({time.time() - t0:.1f}s)")
    return stats


def run_topics(db_path: Path, min_topic_size: int = 50):
    """Run topic modelling pipeline."""
    from nlp_logic.topic_modelling import run_pipeline

    logger.info("Running topic modelling pipeline...")
    t0 = time.time()
    info = run_pipeline(db_path, min_topic_size)
    logger.info(f"Topic modelling complete  ({time.time() - t0:.1f}s)")
    return info


def run_trending(db_path: Path):
    """Run trending vs persistent classification."""
    from nlp_logic.trending_persist import run_pipeline

    logger.info("Running trending vs persistent classification...")
    t0 = time.time()
    result = run_pipeline(db_path)
    logger.info(f"Trending classification complete  ({time.time() - t0:.1f}s)")
    return result


def run_stance(db_path: Path):
    """Run stance detection pipeline."""
    from nlp_logic.comment_stance import run_pipeline

    logger.info("Running stance detection pipeline...")
    t0 = time.time()
    result = run_pipeline(db_path)
    logger.info(f"Stance detection complete  ({time.time() - t0:.1f}s)")
    return result


def main():
    parser = argparse.ArgumentParser(description="Precompute NLP pipelines")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to SQLite DB")
    parser.add_argument(
        "--only",
        choices=["agg", "topics", "trending", "stance"],
        help="Run only a specific pipeline",
    )
    parser.add_argument("--min-topic-size", type=int, default=50,
                        help="Min cluster size for BERTopic")
    args = parser.parse_args()

    total_t0 = time.time()
    logger.info("=" * 60)
    logger.info("  Precompute Pipeline — r/jobs")
    logger.info(f"  DB: {args.db}")
    logger.info("=" * 60)

    if args.only is None or args.only == "agg":
        run_aggregate(args.db)

    if args.only is None or args.only == "topics":
        run_topics(args.db, args.min_topic_size)

    if args.only is None or args.only == "trending":
        run_trending(args.db)

    if args.only is None or args.only == "stance":
        run_stance(args.db)

    logger.info(f"\nAll pipelines complete  (total: {time.time() - total_t0:.1f}s)")


if __name__ == "__main__":
    main()
