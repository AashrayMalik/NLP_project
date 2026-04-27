"""Chunk the cleaned r/jobs SQLite corpus for semantic retrieval."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nlp_logic.reddit_jargon import normalize_jargon
from nlp_logic.topic_insights import build_topic_display_label
from rag.config import (
    CACHE_DIR,
    COMMENT_MAX_TOKENS,
    DB_PATH,
    PARENT_SNIPPET_TOKENS,
    POST_CHUNK_TARGET_TOKENS,
    POST_MAX_TOKENS,
    POST_OVERLAP_TOKENS,
)


@dataclass(frozen=True)
class RagChunk:
    """A stable retrieval unit stored in ChromaDB."""

    id: str
    text: str
    metadata: dict[str, Any]


_TOKEN_RE = re.compile(r"\S+")


def approx_token_count(text: str) -> int:
    """Cheap token approximation sufficient for paragraph-aware chunking."""
    return len(_TOKEN_RE.findall(text or ""))


def first_tokens(text: str, max_tokens: int) -> str:
    tokens = _TOKEN_RE.findall(text or "")
    if len(tokens) <= max_tokens:
        return " ".join(tokens)
    return " ".join(tokens[:max_tokens]) + " ..."


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text or "") if p.strip()]
    if paragraphs:
        return paragraphs
    return [text.strip()] if text and text.strip() else []


def _tail_tokens(text: str, max_tokens: int) -> str:
    tokens = _TOKEN_RE.findall(text or "")
    return " ".join(tokens[-max_tokens:])


def pack_paragraphs(
    paragraphs: list[str],
    max_tokens: int = POST_CHUNK_TARGET_TOKENS,
    overlap_tokens: int = POST_OVERLAP_TOKENS,
) -> list[str]:
    """Pack paragraphs into body chunks while keeping paragraph boundaries."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = approx_token_count(paragraph)

        if paragraph_tokens > max_tokens:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            words = _TOKEN_RE.findall(paragraph)
            start = 0
            while start < len(words):
                end = min(start + max_tokens, len(words))
                chunks.append(" ".join(words[start:end]))
                if end == len(words):
                    break
                start = max(0, end - overlap_tokens)
            continue

        if current and current_tokens + paragraph_tokens > max_tokens:
            chunks.append("\n\n".join(current))
            overlap = _tail_tokens(chunks[-1], overlap_tokens)
            current = [overlap, paragraph] if overlap else [paragraph]
            current_tokens = approx_token_count("\n\n".join(current))
        else:
            current.append(paragraph)
            current_tokens += paragraph_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _safe_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _metadata(row: sqlite3.Row, source_type: str, **extra: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_type": source_type,
        "post_id": _safe_text(row["post_id"] if "post_id" in row.keys() else row["id"]),
        "comment_id": _safe_text(row["comment_id"], "") if "comment_id" in row.keys() else "",
        "topic_id": int(row["topic_id"]) if row["topic_id"] is not None else -999,
        "topic_label": _safe_text(row["topic_label"], "Unknown"),
        "flair": _safe_text(row["flair"], "No Flair"),
        "month": _safe_text(row["month"], ""),
        "score": int(row["score"] or 0),
        "created_utc": int(row["created_utc"] or 0) if "created_utc" in row.keys() else 0,
    }
    metadata.update(extra)
    return metadata


def _post_header(row: sqlite3.Row, chunk_index: int, chunk_count: int) -> str:
    return "\n".join([
        f"Title: {_safe_text(row['title'], 'Untitled')}",
        f"Flair: {_safe_text(row['flair'], 'No Flair')}",
        f"Topic: {_safe_text(row['topic_label'], 'Unknown')}",
        f"Month: {_safe_text(row['month'])}",
        f"Post score: {int(row['score'] or 0)}",
        f"Chunk: {chunk_index + 1} of {chunk_count}",
    ])


def make_post_chunks(row: sqlite3.Row) -> list[RagChunk]:
    """Create body-first post chunks with the title used as context only."""
    title = _safe_text(row["title"], "Untitled")
    body = _safe_text(row["body"])
    header_probe = _post_header(row, 0, 1)
    full_text = f"{header_probe}\n\nPost body:\n{body}".strip()

    if approx_token_count(full_text) <= POST_MAX_TOKENS:
        metadata = _metadata(row, "post", chunk_index=0, chunk_count=1)
        return [RagChunk(id=f"post:{row['id']}:chunk:0", text=normalize_jargon(full_text), metadata=metadata)]

    paragraphs = split_paragraphs(body)
    body_chunks = pack_paragraphs(paragraphs)
    chunks: list[RagChunk] = []

    for index, body_chunk in enumerate(body_chunks):
        header = _post_header(row, index, len(body_chunks))
        text = f"{header}\n\nPost excerpt from '{title}':\n{body_chunk}".strip()
        text = normalize_jargon(text)
        metadata = _metadata(row, "post", chunk_index=index, chunk_count=len(body_chunks))
        chunks.append(RagChunk(id=f"post:{row['id']}:chunk:{index}", text=text, metadata=metadata))

    return chunks


def make_comment_chunk(row: sqlite3.Row) -> RagChunk | None:
    """Create an atomic comment chunk with parent title and body snippet."""
    comment = _safe_text(row["comment_body"])
    if not comment:
        return None

    parent_context = first_tokens(_safe_text(row["parent_body"]), PARENT_SNIPPET_TOKENS)
    text = "\n".join([
        f"Parent post title: {_safe_text(row['parent_title'], 'Untitled')}",
        f"Parent post context: {parent_context}",
        f"Flair: {_safe_text(row['flair'], 'No Flair')}",
        f"Topic: {_safe_text(row['topic_label'], 'Unknown')}",
        f"Month: {_safe_text(row['month'])}",
        f"Comment score: {int(row['score'] or 0)}",
        "",
        "Comment:",
        first_tokens(comment, COMMENT_MAX_TOKENS),
    ]).strip()

    metadata = _metadata(row, "comment")
    return RagChunk(id=f"comment:{row['comment_id']}", text=normalize_jargon(text), metadata=metadata)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def _topic_summary_text(row: sqlite3.Row, trend: dict[str, Any], stance: dict[str, Any], reps: list[str]) -> str:
    keywords = json.loads(row["keywords"]) if row["keywords"] else []
    display_label = build_topic_display_label(keywords, fallback=row["label"])
    trend_type = row["trend_type"] or trend.get("trend_type", "unknown")

    stance_lines = []
    for bucket in ("for", "opposing", "neutral"):
        data = stance.get(bucket, {})
        if data:
            stance_lines.append(f"{bucket.title()}: {data.get('pct', 0)}% ({data.get('count', 0)} comments)")

    argument_lines = []
    for bucket in ("for", "opposing", "neutral"):
        for arg in stance.get(bucket, {}).get("top_arguments", [])[:2]:
            body = first_tokens(arg.get("body", ""), 60)
            if body:
                argument_lines.append(f"- {bucket}: {body}")

    representative_posts = [f"- {first_tokens(doc, 80)}" for doc in reps[:3]]

    return "\n".join([
        f"Topic: {display_label}",
        f"Raw label: {row['label']}",
        f"Keywords: {', '.join(keywords)}",
        f"Post count: {int(row['post_count'] or 0)}",
        f"Share of corpus: {float(row['share_pct'] or 0):.2f}%",
        f"Trend type: {trend_type}",
        "",
        "Stance summary:",
        "\n".join(stance_lines) if stance_lines else "No stance summary available.",
        "",
        "Representative arguments:",
        "\n".join(argument_lines) if argument_lines else "No representative arguments available.",
        "",
        "Representative posts:",
        "\n".join(representative_posts) if representative_posts else "No representative posts available.",
    ]).strip()


def make_topic_chunk(row: sqlite3.Row, trend: dict[str, Any], stance: dict[str, Any], reps: list[str]) -> RagChunk:
    keywords = json.loads(row["keywords"]) if row["keywords"] else []
    display_label = build_topic_display_label(keywords, fallback=row["label"])
    text = _topic_summary_text(row, trend, stance, reps)
    metadata = {
        "source_type": "topic_summary",
        "post_id": "",
        "comment_id": "",
        "topic_id": int(row["id"]),
        "topic_label": display_label,
        "flair": "",
        "month": "",
        "score": int(row["post_count"] or 0),
        "created_utc": 0,
        "chunk_index": 0,
        "chunk_count": 1,
        "trend_type": row["trend_type"] or trend.get("trend_type", "unknown"),
        "post_count": int(row["post_count"] or 0),
        "share_pct": float(row["share_pct"] or 0),
        "keywords": ", ".join(keywords),
    }
    return RagChunk(id=f"topic:{row['id']}", text=text, metadata=metadata)


def iter_post_rows(db_path: str | Path = DB_PATH) -> list[sqlite3.Row]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("""
            SELECT
                p.id,
                p.id AS post_id,
                p.title,
                p.body,
                p.score,
                p.created_utc,
                COALESCE(NULLIF(TRIM(p.flair), ''), 'No Flair') AS flair,
                p.month,
                COALESCE(pt.topic_id, -999) AS topic_id,
                COALESCE(t.label, 'Unknown') AS topic_label
            FROM posts p
            LEFT JOIN post_topics pt ON pt.post_id = p.id
            LEFT JOIN topics t ON t.id = pt.topic_id
            WHERE LENGTH(COALESCE(p.body, '')) > 0
            ORDER BY p.id
        """).fetchall()
    finally:
        conn.close()


def iter_comment_rows(db_path: str | Path = DB_PATH) -> list[sqlite3.Row]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("""
            SELECT
                c.id AS comment_id,
                c.post_id,
                c.body AS comment_body,
                c.score,
                c.author,
                c.created_utc,
                p.title AS parent_title,
                p.body AS parent_body,
                COALESCE(NULLIF(TRIM(p.flair), ''), 'No Flair') AS flair,
                p.month,
                COALESCE(pt.topic_id, -999) AS topic_id,
                COALESCE(t.label, 'Unknown') AS topic_label
            FROM comments c
            JOIN posts p ON p.id = c.post_id
            LEFT JOIN post_topics pt ON pt.post_id = p.id
            LEFT JOIN topics t ON t.id = pt.topic_id
            WHERE LENGTH(COALESCE(c.body, '')) > 0
            ORDER BY c.id
        """).fetchall()
    finally:
        conn.close()


def iter_topic_rows(db_path: str | Path = DB_PATH) -> list[sqlite3.Row]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("""
            SELECT id, label, keywords, post_count, share_pct, trend_type
            FROM topics
            WHERE id != -1
            ORDER BY post_count DESC
        """).fetchall()
    finally:
        conn.close()


def build_chunks(db_path: str | Path = DB_PATH, limit: int | None = None) -> list[RagChunk]:
    """Build all post, comment, and topic-summary chunks."""
    chunks: list[RagChunk] = []

    for row in iter_post_rows(db_path):
        chunks.extend(make_post_chunks(row))
        if limit and len(chunks) >= limit:
            return chunks[:limit]

    for row in iter_comment_rows(db_path):
        chunk = make_comment_chunk(row)
        if chunk:
            chunks.append(chunk)
        if limit and len(chunks) >= limit:
            return chunks[:limit]

    trending = {str(item.get("topic_id")): item for item in _load_json(CACHE_DIR / "trending_cache.json", [])}
    stance = _load_json(CACHE_DIR / "stance_cache.json", {})
    reps = _load_json(CACHE_DIR / "representative_docs.json", {})

    for row in iter_topic_rows(db_path):
        tid = str(row["id"])
        chunks.append(make_topic_chunk(row, trending.get(tid, {}), stance.get(tid, {}), reps.get(tid, [])))
        if limit and len(chunks) >= limit:
            return chunks[:limit]

    return chunks
