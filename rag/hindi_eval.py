"""Evaluate Hindi cross-lingual QA over the English r/jobs corpus."""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean
from typing import Any

from rag.eval_utils import (
    causal_overclaim_flag,
    citation_metrics,
    compute_bertscore,
    compute_chrf,
    flatten_tags,
    is_refusal,
    load_jsonl,
    write_csv,
    write_jsonl,
)
from rag.llms import Provider
from rag.qa import answer_question
from rag.retriever import HybridRetriever


def load_manual_review(output_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    for path in (output_dir / "hindi_manual_review.csv", output_dir / "hindi_manual_review.jsonl"):
        if not path.exists():
            continue
        if path.suffix == ".csv":
            with open(path, newline="") as f:
                iterable = list(csv.DictReader(f))
        else:
            iterable = load_jsonl(path)
        for row in iterable:
            labels[(str(row["model"]), str(row["id"]))] = row
    return labels


def build_manual_review_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    review_rows = []
    for row in rows:
        review_rows.append({
            "model": row["model"],
            "id": row["id"],
            "base_id": row.get("base_id", ""),
            "type": row["type"],
            "tags": flatten_tags(row.get("tags")),
            "answerable": row["answerable"],
            "question_hi": row["question_hi"],
            "gold_answer_hi": row["gold_answer_hi"],
            "generated_answer_hi": row["generated_answer_hi"],
            "retrieved_evidence_ids": "; ".join(row["retrieved_evidence_ids"]),
            "fluency_manual": row.get("fluency_manual", ""),
            "adequacy_manual": row.get("adequacy_manual", ""),
            "causal_overclaim_manual": row.get("causal_overclaim_manual", ""),
        })
    return review_rows


def _manual_numeric(rows: list[dict[str, Any]], field: str) -> list[int]:
    values = []
    for row in rows:
        value = str(row.get(field, "")).strip()
        if value in {"1", "2", "3", "4", "5"}:
            values.append(int(value))
    return values


def _manual_binary(rows: list[dict[str, Any]], field: str) -> list[int]:
    values = []
    for row in rows:
        value = str(row.get(field, "")).strip()
        if value in {"0", "1"}:
            values.append(int(value))
    return values


def build_tag_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tag_rows: list[dict[str, Any]] = []
    for row in rows:
        for tag in row.get("tags", []):
            tag_rows.append({
                "model": row["model"],
                "tag": tag,
                "id": row["id"],
                "chrf": row["chrf"],
                "bertscore_f1": row["bertscore_f1"],
                "citation_present": float(row["citation_present"]),
                "citation_validity": row["citation_validity"] if isinstance(row["citation_validity"], float) else None,
                "adversarial_refusal_correct": float(row["adversarial_refusal_correct"]),
            })

    summary: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in tag_rows:
        groups.setdefault((row["model"], row["tag"]), []).append(row)

    for (model, tag), items in sorted(groups.items()):
        summary.append({
            "model": model,
            "tag": tag,
            "n": len(items),
            "chrf": mean(item["chrf"] for item in items),
            "bertscore_f1": mean(item["bertscore_f1"] for item in items if isinstance(item["bertscore_f1"], float))
            if any(isinstance(item["bertscore_f1"], float) for item in items) else "",
            "citation_presence_rate": mean(item["citation_present"] for item in items),
            "mean_citation_validity": mean(
                item["citation_validity"] for item in items if isinstance(item["citation_validity"], float)
            ) if any(isinstance(item["citation_validity"], float) for item in items) else "",
            "adversarial_refusal_accuracy": mean(item["adversarial_refusal_correct"] for item in items),
        })
    return summary


def evaluate_hindi_models(
    qa_path: str | Path,
    providers: list[Provider],
    output_dir: str | Path,
    *,
    compute_bert: bool = True,
    bert_model: str = "bert-base-multilingual-cased",
    limit: int | None = None,
) -> dict[str, Any]:
    questions = load_jsonl(qa_path)
    if limit is not None:
        questions = questions[:limit]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manual_labels = load_manual_review(output_dir)

    retriever = HybridRetriever()
    all_results: list[dict[str, Any]] = []

    for provider in providers:
        for qi, item in enumerate(questions):
            # Throttle Gemini to stay under 10 RPM free-tier limit
            if provider == "gemini" and qi > 0:
                import time; time.sleep(7)
            result = answer_question(
                item["question_hi"],
                provider=provider,
                retriever=retriever,
                retrieval_query=item.get("retrieval_query_en") or item.get("gold_answer_en") or item["question_hi"],
                target_language="hindi",
            )
            manual = manual_labels.get((str(provider), str(item["id"])), {})
            row = {
                "id": item["id"],
                "base_id": item.get("base_id", ""),
                "type": item["type"],
                "tags": item.get("tags", []),
                "answerable": bool(item["answerable"]),
                "question_hi": item["question_hi"],
                "retrieval_query_en": item.get("retrieval_query_en", ""),
                "gold_answer_hi": item["gold_answer_hi"],
                "gold_answer_en": item["gold_answer_en"],
                "model": provider,
                "generated_answer_hi": result["answer"],
                "query_type": result["query_type"],
                "retrieved_evidence_ids": [e["id"] for e in result["evidence"]],
                "retrieval_source_type_counts": result["debug"].get("source_type_counts", {}),
                "retrieval_origin_counts": result["debug"].get("origin_counts", {}),
                "causal_claim_detected": causal_overclaim_flag(result["answer"]),
                "fluency_manual": manual.get("fluency_manual", ""),
                "adequacy_manual": manual.get("adequacy_manual", ""),
                "causal_overclaim_manual": manual.get("causal_overclaim_manual", ""),
            }
            row.update(citation_metrics(row["generated_answer_hi"], row["retrieved_evidence_ids"]))
            all_results.append(row)

    predictions = [row["generated_answer_hi"] for row in all_results]
    references = [row["gold_answer_hi"] for row in all_results]
    chrf_scores = compute_chrf(predictions, references)
    bert_scores = compute_bertscore(
        predictions,
        references,
        model_type=bert_model,
    ) if compute_bert else [None] * len(all_results)

    for row, chrf, bert in zip(all_results, chrf_scores, bert_scores):
        row["chrf"] = chrf
        row["bertscore_f1"] = bert
        row["adversarial_refusal_correct"] = (not row["answerable"]) and is_refusal(row["generated_answer_hi"])

    write_jsonl(output_dir / "results_all.jsonl", all_results)
    for provider in providers:
        provider_rows = [row for row in all_results if row["model"] == provider]
        write_jsonl(output_dir / f"results_{provider}.jsonl", provider_rows)

    review_rows = build_manual_review_rows(all_results)
    write_jsonl(output_dir / "hindi_manual_review.jsonl", review_rows)
    write_csv(output_dir / "hindi_manual_review.csv", review_rows)

    tag_breakdown = build_tag_breakdown(all_results)
    write_csv(output_dir / "tag_breakdown.csv", tag_breakdown)

    summary_rows = []
    for provider in providers:
        rows = [row for row in all_results if row["model"] == provider]
        fluency_values = _manual_numeric(rows, "fluency_manual")
        adequacy_values = _manual_numeric(rows, "adequacy_manual")
        causal_values = _manual_binary(rows, "causal_overclaim_manual")
        summary_rows.append({
            "model": provider,
            "chrf": mean(row["chrf"] for row in rows),
            "bertscore_f1": mean(row["bertscore_f1"] for row in rows) if compute_bert else "",
            "citation_presence_rate": mean(float(row["citation_present"]) for row in rows),
            "mean_citation_validity": mean(
                row["citation_validity"] for row in rows if isinstance(row["citation_validity"], float)
            ) if any(isinstance(row["citation_validity"], float) for row in rows) else "",
            "adversarial_refusal_accuracy": mean(
                row["adversarial_refusal_correct"] for row in rows if not row["answerable"]
            ) if any(not row["answerable"] for row in rows) else 0.0,
            "fluency_mean_manual": mean(fluency_values) if fluency_values else "",
            "fluency_reviewed": len(fluency_values),
            "adequacy_mean_manual": mean(adequacy_values) if adequacy_values else "",
            "adequacy_reviewed": len(adequacy_values),
            "causal_claim_rate": mean(float(row["causal_claim_detected"]) for row in rows),
            "causal_overclaim_rate_manual": mean(causal_values) if causal_values else "",
            "causal_overclaim_reviewed": len(causal_values),
        })

    write_csv(output_dir / "metrics_summary.csv", summary_rows, list(summary_rows[0].keys()))
    write_report(output_dir / "comparative_report.md", summary_rows)
    return {
        "results": all_results,
        "summary": summary_rows,
        "tag_breakdown": tag_breakdown,
    }


def write_report(path: str | Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Hindi Cross-Lingual QA Comparison",
        "",
        "| Model | chrF | BERTScore F1 | Citation Presence | Citation Validity | Refusal Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        bert = row["bertscore_f1"]
        bert_text = f"{bert:.3f}" if isinstance(bert, float) else ""
        validity = row["mean_citation_validity"]
        validity_text = f"{validity:.1%}" if isinstance(validity, float) else ""
        lines.append(
            f"| {row['model']} | {row['chrf']:.3f} | {bert_text} | {row['citation_presence_rate']:.1%} | "
            f"{validity_text} | {row['adversarial_refusal_accuracy']:.1%} |"
        )

    lines.extend([
        "",
        "## Manual Review",
        "",
        "Fill `fluency_manual` and `adequacy_manual` on a 1-5 scale in `hindi_manual_review.csv`. "
        "Mark `causal_overclaim_manual` as 1 when the answer makes an unsupported causal claim and 0 otherwise.",
        "",
        "## Edge Cases",
        "",
        "Inspect `tag_breakdown.csv` for code-mixed, romanized Hindi, Reddit slang, named-entity, and adversarial subsets.",
    ])
    Path(path).write_text("\n".join(lines))
