"""ChromaDB indexing and query helpers."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

from rag.chunking import RagChunk
from rag.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:batch_size": 5000,
    "hnsw:sync_threshold": 20000,
}


def require_chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("ChromaDB is not installed. Run `uv sync` before building the RAG index.") from exc
    return chromadb


def require_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is not installed. Run `uv sync` before building the RAG index.") from exc
    return SentenceTransformer


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def clear_accelerator_cache() -> None:
    """Release cached accelerator memory between large embedding batches."""
    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class RagVectorStore:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(
        self,
        persist_dir: str | Path = CHROMA_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        chromadb = require_chromadb()
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata=COLLECTION_METADATA,
        )
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            SentenceTransformer = require_sentence_transformers()
            device = os.getenv("RAG_EMBEDDING_DEVICE", "cpu")
            self._embedder = SentenceTransformer(
                self.embedding_model_name,
                device=device,
                local_files_only=True,
            )
        return self._embedder

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=COLLECTION_METADATA,
        )

    def count(self) -> int:
        return int(self.collection.count())

    def add_chunks(self, chunks: list[RagChunk], batch_size: int = 1024) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings = self.embedder.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            self.collection.add(
                ids=[chunk.id for chunk in batch],
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=[sanitize_metadata(chunk.metadata) for chunk in batch],
            )
            clear_accelerator_cache()
            print(f"Indexed {min(start + len(batch), len(chunks)):,}/{len(chunks):,} chunks", flush=True)

    def query(self, question: str, k: int = 20) -> list[dict[str, Any]]:
        # BGE models benefit from a query prefix for retrieval tasks
        prefixed = f"Represent this sentence: {question}"
        embedding = self.embedder.encode([prefixed], normalize_embeddings=True)[0].tolist()
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        rows: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for chunk_id, doc, metadata, distance in zip(ids, docs, metadatas, distances):
            rows.append({
                "id": chunk_id,
                "text": doc,
                "metadata": metadata or {},
                "distance": float(distance),
                "similarity": max(0.0, 1.0 - float(distance)),
            })
        return rows

    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        result = self.collection.get(ids=ids, include=["documents", "metadatas"])
        rows = []
        for chunk_id, doc, metadata in zip(result.get("ids", []), result.get("documents", []), result.get("metadatas", [])):
            rows.append({
                "id": chunk_id,
                "text": doc,
                "metadata": metadata or {},
                "distance": 1.0,
                "similarity": 0.0,
            })
        return rows

    def get_where(self, where: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
        result = self.collection.get(where=sanitize_metadata(where), limit=limit, include=["documents", "metadatas"])
        rows = []
        for chunk_id, doc, metadata in zip(result.get("ids", []), result.get("documents", []), result.get("metadatas", [])):
            rows.append({
                "id": chunk_id,
                "text": doc,
                "metadata": metadata or {},
                "distance": 1.0,
                "similarity": 0.0,
            })
        return rows
