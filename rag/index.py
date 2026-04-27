"""Build the vector index and semantic graph artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from rag.chunking import build_chunks
from rag.config import CHROMA_DIR, DB_PATH, GRAPH_PATH, MANIFEST_PATH
from rag.graph import build_graph, save_graph
from rag.vector_store import RagVectorStore


def build_rag_index(
    db_path: str | Path = DB_PATH,
    reset: bool = True,
    limit: int | None = None,
) -> dict[str, int | str]:
    chunks = build_chunks(db_path=db_path, limit=limit)

    store = RagVectorStore(persist_dir=CHROMA_DIR)
    if reset:
        store.reset()
    store.add_chunks(chunks)

    graph = build_graph(db_path=db_path)
    save_graph(graph, GRAPH_PATH)

    manifest = {
        "chunks": len(chunks),
        "vectors": store.count(),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "db_path": str(db_path),
        "chroma_dir": str(CHROMA_DIR),
        "graph_path": str(GRAPH_PATH),
        "limited": limit is not None,
        "limit": limit,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest
