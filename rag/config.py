"""Shared configuration for the r/jobs RAG pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "scraping" / "data"
DB_PATH = DATA_DIR / "jobs_posts.db"
CACHE_DIR = DATA_DIR

CHROMA_DIR = DATA_DIR / "rag_chroma"
GRAPH_PATH = DATA_DIR / "rag_graph.json"
MANIFEST_PATH = DATA_DIR / "rag_manifest.json"
COLLECTION_NAME = "reddit_jobs_rag"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

POST_MAX_TOKENS = 450
POST_CHUNK_TARGET_TOKENS = 380
POST_OVERLAP_TOKENS = 60
COMMENT_MAX_TOKENS = 420
PARENT_SNIPPET_TOKENS = 80

DEFAULT_RETRIEVAL_K = 28
DEFAULT_FINAL_K = 10
DEFAULT_GRAPH_HOPS = 2
