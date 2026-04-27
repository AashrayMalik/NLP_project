"""Hybrid vector + graph retrieval for the r/jobs QA system."""

from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path
from typing import Any

from rag.config import (
    DB_PATH,
    DEFAULT_FINAL_K,
    DEFAULT_GRAPH_HOPS,
    DEFAULT_RETRIEVAL_K,
    GRAPH_PATH,
)
from rag.graph import chunk_seed_nodes, expand_nodes, load_graph
from rag.vector_store import RagVectorStore


def classify_query(question: str) -> str:
    q = question.lower()
    factual_terms = (
        "how many",
        "how large",
        "count",
        "size",
        "most common",
        "distribution",
        "average",
        "total",
        "dataset cover",
        "time period",
        "date range",
        "span",
        "top comments",
        "comments collected",
        "were collected",
        "kind of comments",
        "कितने",
        "सबसे आम",
        "डेटासेट",
        "समय अवधि",
        "तारीख",
        "कमेंट",
    )
    if any(term in q for term in factual_terms):
        return "factual"
    if any(term in q for term in ("trend", "trending", "persistent", "seasonal", "month", "over time", "रुझान", "लगातार")):
        return "trend"
    if any(term in q for term in ("think", "feel", "opinion", "view", "advice", "recommend", "attitude", "सोचते", "सलाह", "राय")):
        return "opinion"
    return "general"


def _shorten(text: str, max_chars: int = 1200) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sql_item(item_id: str, text: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "text": text,
        "metadata": {"source_type": "sql_fact"},
        "similarity": 1.0,
        "distance": 0.0,
        "rerank_score": 1.0,
        "retrieval_origins": ["sql"],
    }


def _sql_evidence(question: str, db_path: str | Path = DB_PATH) -> list[dict[str, Any]]:
    """Add deterministic corpus facts for obvious factual questions."""
    q = question.lower()
    evidence: list[dict[str, Any]] = []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        wants_counts = any(term in q for term in ("total", "how many", "how large", "count", "posts", "comments", "size"))
        wants_dates = any(term in q for term in ("cover", "time period", "date range", "span", "month", "year"))
        wants_comment_policy = any(term in q for term in ("top comments", "comments collected", "were collected", "kind of comments"))
        wants_trending_topics = "trending" in q and ("topic" in q or "theme" in q)

        if wants_counts:
            row = conn.execute("SELECT COUNT(*) AS posts FROM posts").fetchone()
            comments = conn.execute("SELECT COUNT(*) AS comments FROM comments").fetchone()
            evidence.append(_sql_item(
                "sql:corpus_counts",
                f"Corpus counts: {row['posts']:,} posts and {comments['comments']:,} collected top comments.",
            ))

        if wants_dates:
            row = conn.execute("""
                SELECT
                    MIN(date(created_utc, 'unixepoch')) AS start_date,
                    MAX(date(created_utc, 'unixepoch')) AS end_date,
                    COUNT(DISTINCT month) AS months
                FROM posts
            """).fetchone()
            evidence.append(_sql_item(
                "sql:date_range",
                f"Corpus date range: posts span {row['start_date']} through {row['end_date']} across {row['months']} monthly buckets.",
            ))

        if wants_comment_policy:
            row = conn.execute("""
                SELECT MAX(n) AS max_comments, ROUND(AVG(n), 2) AS avg_comments
                FROM (
                    SELECT COUNT(*) AS n
                    FROM comments
                    GROUP BY post_id
                )
            """).fetchone()
            evidence.append(_sql_item(
                "sql:comment_collection_policy",
                "Comment collection policy: the scraper fetched up to the top 5 comments by score for each qualifying post. "
                f"In the cleaned database, posts have at most {row['max_comments']} collected comments and average {row['avg_comments']} comments per post.",
            ))

        if "flair" in q or "category" in q:
            rows = conn.execute("""
                SELECT COALESCE(NULLIF(TRIM(flair), ''), 'No Flair') AS flair, COUNT(*) AS n
                FROM posts
                GROUP BY flair
                ORDER BY n DESC
                LIMIT 12
            """).fetchall()
            text = "Most common flairs: " + "; ".join(f"{r['flair']} ({r['n']:,})" for r in rows)
            evidence.append(_sql_item("sql:top_flairs", text))

        if "topic" in q or "theme" in q:
            if wants_trending_topics:
                rows = conn.execute("""
                    SELECT label, post_count, ROUND(share_pct, 2) AS share_pct
                    FROM topics
                    WHERE id != -1 AND COALESCE(trend_type, '') = 'trending'
                    ORDER BY post_count DESC
                """).fetchall()
                text = "Topics marked as trending: " + "; ".join(
                    f"{r['label']} ({r['post_count']:,} posts, {r['share_pct']}%)"
                    for r in rows
                )
                evidence.append(_sql_item("sql:trending_topics", text))
            else:
                rows = conn.execute("""
                    SELECT label, post_count, ROUND(share_pct, 2) AS share_pct, COALESCE(trend_type, 'unknown') AS trend_type
                    FROM topics
                    WHERE id != -1
                    ORDER BY post_count DESC
                    LIMIT 8
                """).fetchall()
                text = "Largest discovered topics: " + "; ".join(
                    f"{r['label']} ({r['post_count']:,} posts, {r['share_pct']}%, {r['trend_type']})"
                    for r in rows
                )
                evidence.append(_sql_item("sql:top_topics", text))
    finally:
        conn.close()

    return evidence


class HybridRetriever:
    """Retrieve compact evidence using Chroma similarity and graph expansion."""

    def __init__(
        self,
        store: RagVectorStore | None = None,
        graph_path: str | Path = GRAPH_PATH,
        db_path: str | Path = DB_PATH,
    ) -> None:
        self.store = store or RagVectorStore()
        self.db_path = db_path
        self.graph = load_graph(graph_path) if Path(graph_path).exists() else None

    def _graph_candidates(self, initial: list[dict[str, Any]], hops: int) -> list[dict[str, Any]]:
        if self.graph is None:
            return []

        seeds: list[str] = []
        for item in initial[:10]:
            seeds.extend(chunk_seed_nodes(item.get("metadata", {})))

        distances = expand_nodes(self.graph, seeds, hops=hops)
        nodes = set(distances)
        candidates: dict[str, dict[str, Any]] = {}

        for node, distance in sorted(distances.items(), key=lambda item: item[1]):
            kind, _, value = node.partition(":")
            rows: list[dict[str, Any]] = []

            if kind == "comment":
                rows = self.store.get_by_ids([f"comment:{value}"])
            elif kind == "post":
                rows = self.store.get_where({"post_id": value}, limit=4)
            elif kind == "topic":
                rows = self.store.get_by_ids([f"topic:{value}"])
                rows += self.store.get_where({"topic_id": _safe_int(value, -999)}, limit=4)
            elif kind == "flair":
                rows = self.store.get_where({"flair": value}, limit=4)
            elif kind == "month":
                rows = self.store.get_where({"month": value}, limit=4)

            for row in rows:
                if row["id"] not in candidates:
                    row["graph_distance"] = distance
                    row["graph_nodes"] = list(nodes)[:40]
                    candidates[row["id"]] = row

            if len(candidates) >= 80:
                break

        return list(candidates.values())

    def _score(self, item: dict[str, Any], query_type: str) -> float:
        metadata = item.get("metadata", {})
        source_type = metadata.get("source_type", "")
        similarity = float(item.get("similarity", 0.0))
        reddit_score = max(0, _safe_int(metadata.get("score"), 0))
        score_boost = min(math.log1p(reddit_score) / 12.0, 0.35)
        graph_distance = item.get("graph_distance")
        graph_boost = 0.0 if graph_distance is None else max(0.0, 0.25 - 0.07 * float(graph_distance))

        source_boosts = {
            "factual": {"sql_fact": 0.45, "topic_summary": 0.18, "post": 0.08, "comment": 0.02},
            "trend": {"sql_fact": 0.45, "topic_summary": 0.32, "post": 0.08, "comment": 0.04},
            "opinion": {"comment": 0.24, "topic_summary": 0.18, "post": 0.10},
            "general": {"post": 0.16, "comment": 0.14, "topic_summary": 0.12},
        }
        source_boost = source_boosts.get(query_type, source_boosts["general"]).get(source_type, 0.0)
        return similarity * 0.68 + score_boost + graph_boost + source_boost

    def retrieve(
        self,
        question: str,
        retrieval_k: int = DEFAULT_RETRIEVAL_K,
        final_k: int = DEFAULT_FINAL_K,
        graph_hops: int = DEFAULT_GRAPH_HOPS,
    ) -> dict[str, Any]:
        query_type = classify_query(question)
        initial = self.store.query(question, k=retrieval_k)
        graph_candidates = self._graph_candidates(initial, hops=graph_hops)
        sql_items = _sql_evidence(question, self.db_path) if query_type in {"factual", "trend"} else []
        initial_ids = {item["id"] for item in initial}
        graph_ids = {item["id"] for item in graph_candidates}
        sql_ids = {item["id"] for item in sql_items}

        merged: dict[str, dict[str, Any]] = {}
        for item in initial + graph_candidates + sql_items:
            existing = merged.get(item["id"])
            if existing is None or item.get("similarity", 0.0) > existing.get("similarity", 0.0):
                merged[item["id"]] = item

        for item in merged.values():
            origins = []
            if item["id"] in initial_ids:
                origins.append("vector")
            if item["id"] in graph_ids:
                origins.append("graph")
            if item["id"] in sql_ids:
                origins.append("sql")
            item["retrieval_origins"] = origins
            item["rerank_score"] = self._score(item, query_type)

        evidence = sorted(merged.values(), key=lambda item: item["rerank_score"], reverse=True)[:final_k]
        source_type_counts: dict[str, int] = {}
        origin_counts: dict[str, int] = {}
        for item in evidence:
            source_type = str(item.get("metadata", {}).get("source_type", "unknown"))
            source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
            for origin in item.get("retrieval_origins", []):
                origin_counts[origin] = origin_counts.get(origin, 0) + 1
        return {
            "question": question,
            "query_type": query_type,
            "evidence": evidence,
            "context": format_context(evidence),
            "debug": {
                "initial_vector_ids": [item["id"] for item in initial],
                "graph_candidate_ids": [item["id"] for item in graph_candidates],
                "sql_evidence_ids": [item["id"] for item in sql_items],
                "source_type_counts": source_type_counts,
                "origin_counts": origin_counts,
            },
        }


def format_context(evidence: list[dict[str, Any]]) -> str:
    blocks = []
    for index, item in enumerate(evidence, 1):
        metadata = item.get("metadata", {})
        blocks.append("\n".join([
            f"[Evidence {index}]",
            f"ID: {item['id']}",
            f"Type: {metadata.get('source_type', 'unknown')}",
            f"Topic: {metadata.get('topic_label', '')}",
            f"Flair: {metadata.get('flair', '')}",
            f"Month: {metadata.get('month', '')}",
            f"Score: {metadata.get('score', '')}",
            f"Text: {_shorten(item.get('text', ''), 1400)}",
        ]))
    return "\n\n".join(blocks)


def retrieval_only_answer(
    question: str,
    evidence: list[dict[str, Any]],
    *,
    target_language: str = "english",
) -> str:
    if not evidence:
        if target_language.lower() in {"hi", "hindi"}:
            return "कॉर्पस में इस प्रश्न का उत्तर देने के लिए पर्याप्त प्राप्त साक्ष्य नहीं हैं।"
        return "The corpus does not contain enough retrieved evidence to answer this question."

    citations = ", ".join(item["id"] for item in evidence[:4])
    source_types = {item.get("metadata", {}).get("source_type", "unknown") for item in evidence[:6]}
    if target_language.lower() in {"hi", "hindi"}:
        return (
            "रिट्रीवल-ओनली मोड: सिस्टम ने प्रासंगिक कॉर्पस साक्ष्य ढूंढे, लेकिन किसी LLM endpoint को कॉल नहीं किया गया। "
            f"सबसे मजबूत साक्ष्य {', '.join(sorted(source_types))} से आते हैं। "
            f"अंतिम उत्तर लिखने से पहले इन citations को देखें: {citations}."
        )
    return (
        "Retrieval-only mode: the system found relevant corpus evidence but no LLM endpoint was called. "
        f"The strongest evidence comes from {', '.join(sorted(source_types))}. "
        f"Review these citations before writing the final answer: {citations}."
    )
