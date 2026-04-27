"""Build and query the lightweight semantic graph for graph-RAG expansion."""

from __future__ import annotations

import json
import sqlite3
from collections import deque
from pathlib import Path
from typing import Any

import networkx as nx

from rag.config import CACHE_DIR, DB_PATH, GRAPH_PATH


def _node(kind: str, value: Any) -> str:
    return f"{kind}:{value}"


def _safe(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def build_graph(db_path: str | Path = DB_PATH) -> nx.Graph:
    """Create an object-level graph over posts, comments, topics, flairs, months, trends, and stances."""
    graph = nx.Graph()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    try:
        posts = conn.execute("""
            SELECT p.id, p.title, p.flair, p.month, p.score, COALESCE(pt.topic_id, -999) AS topic_id
            FROM posts p
            LEFT JOIN post_topics pt ON pt.post_id = p.id
        """).fetchall()

        for row in posts:
            post_node = _node("post", row["id"])
            graph.add_node(post_node, kind="post", post_id=row["id"], title=_safe(row["title"]), score=int(row["score"] or 0))

            flair = _safe(row["flair"], "No Flair")
            flair_node = _node("flair", flair)
            graph.add_node(flair_node, kind="flair", label=flair)
            graph.add_edge(post_node, flair_node, relation="has_flair")

            if row["month"]:
                month_node = _node("month", row["month"])
                graph.add_node(month_node, kind="month", label=row["month"])
                graph.add_edge(post_node, month_node, relation="posted_in")

            if row["topic_id"] is not None and int(row["topic_id"]) != -999:
                topic_node = _node("topic", row["topic_id"])
                graph.add_node(topic_node, kind="topic", topic_id=int(row["topic_id"]))
                graph.add_edge(post_node, topic_node, relation="belongs_to_topic")

        comments = conn.execute("""
            SELECT id, post_id, score
            FROM comments
        """).fetchall()

        for row in comments:
            comment_node = _node("comment", row["id"])
            post_node = _node("post", row["post_id"])
            graph.add_node(comment_node, kind="comment", comment_id=row["id"], post_id=row["post_id"], score=int(row["score"] or 0))
            if graph.has_node(post_node):
                graph.add_edge(comment_node, post_node, relation="comments_on")

        topics = conn.execute("""
            SELECT id, label, trend_type
            FROM topics
            WHERE id != -1
        """).fetchall()

        for row in topics:
            topic_node = _node("topic", row["id"])
            graph.add_node(topic_node, kind="topic", topic_id=int(row["id"]), label=_safe(row["label"]))

            if row["trend_type"]:
                trend_node = _node("trend", row["trend_type"])
                graph.add_node(trend_node, kind="trend", label=row["trend_type"])
                graph.add_edge(topic_node, trend_node, relation="has_trend_type")
    finally:
        conn.close()

    stance_cache = _load_json(CACHE_DIR / "stance_cache.json", {})
    for topic_id, stance in stance_cache.items():
        topic_node = _node("topic", topic_id)
        if not graph.has_node(topic_node):
            continue
        for bucket in ("for", "opposing", "neutral"):
            data = stance.get(bucket)
            if not data:
                continue
            stance_node = _node("stance", f"{topic_id}:{bucket}")
            graph.add_node(
                stance_node,
                kind="stance",
                topic_id=int(topic_id),
                bucket=bucket,
                pct=float(data.get("pct", 0.0)),
                count=int(data.get("count", 0)),
            )
            graph.add_edge(topic_node, stance_node, relation="has_stance_bucket")

    return graph


def save_graph(graph: nx.Graph, path: str | Path = GRAPH_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(graph)
    with open(path, "w") as f:
        json.dump(data, f)


def load_graph(path: str | Path = GRAPH_PATH) -> nx.Graph:
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return nx.node_link_graph(data)


def chunk_seed_nodes(metadata: dict[str, Any]) -> list[str]:
    seeds: list[str] = []
    source_type = metadata.get("source_type")
    post_id = metadata.get("post_id")
    comment_id = metadata.get("comment_id")
    topic_id = metadata.get("topic_id")
    flair = metadata.get("flair")
    month = metadata.get("month")

    if source_type == "comment" and comment_id:
        seeds.append(_node("comment", comment_id))
    if post_id:
        seeds.append(_node("post", post_id))
    if topic_id not in (None, "", -999, "-999"):
        seeds.append(_node("topic", topic_id))
    if flair:
        seeds.append(_node("flair", flair))
    if month:
        seeds.append(_node("month", month))

    return seeds


def expand_nodes(graph: nx.Graph, seeds: list[str], hops: int = 2, max_nodes: int = 80) -> dict[str, int]:
    """Return reachable nodes and their shortest graph distance from the seed set."""
    distances: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque()

    for seed in seeds:
        if graph.has_node(seed):
            distances[seed] = 0
            queue.append((seed, 0))

    while queue and len(distances) < max_nodes:
        node, distance = queue.popleft()
        if distance >= hops:
            continue
        for neighbor in graph.neighbors(node):
            if neighbor in distances:
                continue
            distances[neighbor] = distance + 1
            queue.append((neighbor, distance + 1))
            if len(distances) >= max_nodes:
                break

    return distances


def metadata_matches_nodes(metadata: dict[str, Any], nodes: set[str]) -> bool:
    for seed in chunk_seed_nodes(metadata):
        if seed in nodes:
            return True
    return False
