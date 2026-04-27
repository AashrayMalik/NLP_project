"""Extended retrieval diagnostics for the RAG pipeline."""

from __future__ import annotations

import csv
import random
import sqlite3
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from rag.config import DB_PATH
from rag.eval_utils import causal_overclaim_flag, citation_metrics, load_jsonl, write_csv, write_jsonl
from rag.llms import Provider
from rag.qa import answer_question
from rag.retriever import HybridRetriever


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def evaluate_diagnostic_queries(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    provider: Provider = "retrieval_only",
    final_k: int = 10,
) -> dict[str, Any]:
    questions = load_jsonl(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retriever = HybridRetriever()
    rows: list[dict[str, Any]] = []

    for item in questions:
        result = answer_question(
            item["question"],
            provider=provider,
            retriever=retriever,
            final_k=final_k,
            retrieval_query=item.get("retrieval_query"),
            target_language=item.get("target_language", "english"),
        )
        evidence = result["evidence"]
        retrieved_ids = [entry["id"] for entry in evidence]
        source_types = [str(entry.get("metadata", {}).get("source_type", "unknown")) for entry in evidence]
        source_type_set = sorted(set(source_types))
        comment_count = sum(1 for source_type in source_types if source_type == "comment")
        must_include = item.get("must_include_source_types", [])
        source_hit = all(required in source_type_set for required in must_include)
        comment_required = bool(item.get("comment_required", False))
        min_comment_evidence = int(item.get("min_comment_evidence", 0))
        comment_requirement_met = (comment_count >= min_comment_evidence) if comment_required else True
        row = {
            "id": item["id"],
            "question": item["question"],
            "purpose": item.get("purpose", ""),
            "expected_query_type": item.get("expected_query_type", ""),
            "query_type": result["query_type"],
            "query_type_match": item.get("expected_query_type", "") == result["query_type"],
            "tags": item.get("tags", []),
            "paraphrase_group": item.get("paraphrase_group", ""),
            "must_include_source_types": must_include,
            "comment_required": comment_required,
            "min_comment_evidence": min_comment_evidence,
            "comment_count": comment_count,
            "retrieved_source_types": source_type_set,
            "source_type_hit": source_hit,
            "comment_requirement_met": comment_requirement_met,
            "retrieved_evidence_ids": retrieved_ids,
            "retrieval_source_type_counts": result["debug"].get("source_type_counts", {}),
            "retrieval_origin_counts": result["debug"].get("origin_counts", {}),
            "generated_answer": result["answer"],
            "causal_claim_detected": causal_overclaim_flag(result["answer"]),
        }
        row.update(citation_metrics(row["generated_answer"], retrieved_ids))
        rows.append(row)

    paraphrase_rows: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["paraphrase_group"]:
            groups.setdefault(str(row["paraphrase_group"]), []).append(row)
    for group, items in sorted(groups.items()):
        overlaps = [
            _jaccard(set(left["retrieved_evidence_ids"]), set(right["retrieved_evidence_ids"]))
            for left, right in combinations(items, 2)
        ]
        paraphrase_rows.append({
            "paraphrase_group": group,
            "n": len(items),
            "mean_evidence_jaccard": mean(overlaps) if overlaps else 1.0,
        })

    summary_rows = [{
        "model": provider,
        "n": len(rows),
        "query_type_accuracy": mean(float(row["query_type_match"]) for row in rows),
        "source_type_hit_rate": mean(float(row["source_type_hit"]) for row in rows),
        "comment_required_satisfaction_rate": mean(
            float(row["comment_requirement_met"]) for row in rows if row["comment_required"]
        ) if any(row["comment_required"] for row in rows) else 1.0,
        "average_comment_evidence_count": mean(row["comment_count"] for row in rows),
        "paraphrase_mean_jaccard": mean(row["mean_evidence_jaccard"] for row in paraphrase_rows)
        if paraphrase_rows else "",
        "citation_presence_rate": mean(float(row["citation_present"]) for row in rows),
        "mean_citation_validity": mean(
            row["citation_validity"] for row in rows if isinstance(row["citation_validity"], float)
        ) if any(isinstance(row["citation_validity"], float) for row in rows) else "",
        "causal_claim_rate": mean(float(row["causal_claim_detected"]) for row in rows),
    }]

    causal_rows = [{
        "model": provider,
        "id": row["id"],
        "question": row["question"],
        "generated_answer": row["generated_answer"],
        "retrieved_evidence_ids": "; ".join(row["retrieved_evidence_ids"]),
        "causal_claim_detected": int(row["causal_claim_detected"]),
        "causal_overclaim_manual": "",
    } for row in rows]

    write_jsonl(output_dir / "diagnostic_results.jsonl", rows)
    write_csv(output_dir / "diagnostic_results.csv", rows)
    write_csv(output_dir / "paraphrase_breakdown.csv", paraphrase_rows)
    write_csv(output_dir / "metrics_summary.csv", summary_rows)
    write_csv(output_dir / "causal_overclaim_review.csv", causal_rows)
    return {"rows": rows, "summary": summary_rows, "paraphrase": paraphrase_rows}


def _score_bucket(score: int, thresholds: tuple[int, int]) -> str:
    low, high = thresholds
    if score <= low:
        return "low"
    if score >= high:
        return "high"
    return "mid"


def _comment_probe_query(row: sqlite3.Row) -> str:
    comment_excerpt = " ".join(str(row["body"]).split()[:32])
    return (
        f"Parent post: {row['title']}. Topic: {row['topic_label']}. "
        f"Comment viewpoint: {comment_excerpt}"
    )


def sample_comment_probes(
    *,
    db_path: str | Path = DB_PATH,
    sample_size: int = 24,
    seed: int = 7,
) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT
                c.id,
                c.post_id,
                c.body,
                c.score,
                p.title,
                COALESCE(NULLIF(TRIM(p.flair), ''), 'No Flair') AS flair,
                p.month,
                COALESCE(t.label, 'Unknown') AS topic_label
            FROM comments c
            JOIN posts p ON p.id = c.post_id
            LEFT JOIN post_topics pt ON pt.post_id = p.id
            LEFT JOIN topics t ON t.id = pt.topic_id
            WHERE LENGTH(COALESCE(c.body, '')) > 30
            ORDER BY c.id
        """).fetchall()
    finally:
        conn.close()

    scores = sorted(int(row["score"] or 0) for row in rows)
    if not scores:
        return []
    low_threshold = scores[len(scores) // 3]
    high_threshold = scores[(2 * len(scores)) // 3]

    grouped: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in rows:
        bucket = _score_bucket(int(row["score"] or 0), (low_threshold, high_threshold))
        grouped.setdefault((str(row["month"]), bucket), []).append(row)

    rng = random.Random(seed)
    group_keys = sorted(grouped)
    for key in group_keys:
        rng.shuffle(grouped[key])

    selected: list[dict[str, Any]] = []
    while len(selected) < sample_size and any(grouped[key] for key in group_keys):
        for key in group_keys:
            if len(selected) >= sample_size:
                break
            if not grouped[key]:
                continue
            row = grouped[key].pop()
            selected.append({
                "comment_id": row["id"],
                "post_id": row["post_id"],
                "month": row["month"],
                "flair": row["flair"],
                "topic_label": row["topic_label"],
                "score": int(row["score"] or 0),
                "score_bucket": key[1],
                "probe_query": _comment_probe_query(row),
            })
    return selected


def run_comment_probes(
    output_dir: str | Path,
    *,
    sample_size: int = 24,
    final_k: int = 10,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probes = sample_comment_probes(sample_size=sample_size)
    retriever = HybridRetriever()

    rows: list[dict[str, Any]] = []
    for probe in probes:
        result = retriever.retrieve(probe["probe_query"], final_k=final_k)
        final_ids = [entry["id"] for entry in result["evidence"]]
        initial_ids = result["debug"].get("initial_vector_ids", [])
        target_id = f"comment:{probe['comment_id']}"
        rows.append({
            **probe,
            "initial_hit": target_id in initial_ids,
            "final_hit": target_id in final_ids,
            "final_rank": final_ids.index(target_id) + 1 if target_id in final_ids else "",
            "retrieved_evidence_ids": final_ids,
            "failure_reason": ""
            if target_id in final_ids else "comment_not_in_top_k",
        })

    summary_rows = [{
        "n": len(rows),
        "initial_hit_rate": mean(float(row["initial_hit"]) for row in rows) if rows else 0.0,
        "final_hit_rate": mean(float(row["final_hit"]) for row in rows) if rows else 0.0,
        "mean_final_rank_when_hit": mean(
            row["final_rank"] for row in rows if isinstance(row["final_rank"], int)
        ) if any(isinstance(row["final_rank"], int) for row in rows) else "",
    }]

    write_jsonl(output_dir / "comment_probe_results.jsonl", rows)
    write_csv(output_dir / "comment_probe_results.csv", rows)
    write_csv(output_dir / "comment_probe_summary.csv", summary_rows)
    return {"rows": rows, "summary": summary_rows}
