"""CLI for building the r/jobs semantic KG hybrid RAG artifacts."""

import argparse
import json
from pathlib import Path

from rag.config import DB_PATH
from rag.index import build_rag_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB vector index and NetworkX graph for RAG.")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Path to jobs_posts.db")
    parser.add_argument("--limit", type=int, default=None, help="Optional chunk limit for smoke tests")
    parser.add_argument("--no-reset", action="store_true", help="Append to existing Chroma collection instead of resetting")
    args = parser.parse_args()

    summary = build_rag_index(db_path=args.db, reset=not args.no_reset, limit=args.limit)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
