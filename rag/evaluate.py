"""Evaluate Groq and Gemini answers over the hand-written QA set."""

from __future__ import annotations

from pathlib import Path
from statistics import mean
from typing import Any

from rag.eval_utils import (
    citation_metrics,
    compute_bertscore,
    compute_rouge_l,
    is_refusal,
    load_jsonl,
    write_csv,
    write_jsonl,
)
from rag.llms import Provider
from rag.qa import answer_question
from rag.retriever import HybridRetriever


def load_manual_faithfulness(output_dir: Path) -> dict[tuple[str, str], int]:
    """Load prior human binary faithfulness labels, if the review file has been marked."""
    labels: dict[tuple[str, str], int] = {}
    for path in (output_dir / "faithfulness_review.csv", output_dir / "faithfulness_review.jsonl"):
        if not path.exists():
            continue
        if path.suffix == ".csv":
            import csv

            with open(path, newline="") as f:
                rows = csv.DictReader(f)
                iterable = list(rows)
        else:
            iterable = load_jsonl(path)
        for row in iterable:
            value = str(row.get("faithful_manual", "")).strip()
            if value in {"0", "1"}:
                labels[(str(row["model"]), str(row["id"]))] = int(value)
    return labels


def build_faithfulness_review_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    review_rows = []
    for row in rows:
        review_rows.append({
            "model": row["model"],
            "id": row["id"],
            "type": row["type"],
            "answerable": row["answerable"],
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "generated_answer": row["generated_answer"],
            "retrieved_evidence_ids": "; ".join(row["retrieved_evidence_ids"]),
            "faithful_manual": row.get("faithful_manual", ""),
        })
    return review_rows


def evaluate_models(
    qa_path: str | Path,
    providers: list[Provider],
    output_dir: str | Path,
    compute_bert: bool = True,
) -> dict[str, Any]:
    questions = load_jsonl(qa_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manual_labels = load_manual_faithfulness(output_dir)

    retriever = HybridRetriever()
    all_results: list[dict[str, Any]] = []

    for provider in providers:
        provider_rows = []
        for item in questions:
            result = answer_question(
                item["question"],
                provider=provider,
                retriever=retriever,
            )
            row = {
                "id": item["id"],
                "type": item["type"],
                "answerable": bool(item["answerable"]),
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "model": provider,
                "generated_answer": result["answer"],
                "query_type": result["query_type"],
                "retrieved_evidence_ids": [e["id"] for e in result["evidence"]],
                "retrieval_source_type_counts": result["debug"].get("source_type_counts", {}),
                "retrieval_origin_counts": result["debug"].get("origin_counts", {}),
                "faithful_manual": manual_labels.get((str(provider), str(item["id"])), ""),
            }
            row.update(citation_metrics(row["generated_answer"], row["retrieved_evidence_ids"]))
            provider_rows.append(row)
            all_results.append(row)

    predictions = [row["generated_answer"] for row in all_results]
    references = [row["gold_answer"] for row in all_results]
    rouge_l = compute_rouge_l(predictions, references)
    bert_f1 = compute_bertscore(predictions, references) if compute_bert else [None] * len(all_results)

    for row, rouge, bert in zip(all_results, rouge_l, bert_f1):
        row["rouge_l"] = rouge
        row["bertscore_f1"] = bert
        row["adversarial_refusal_correct"] = (not row["answerable"]) and is_refusal(row["generated_answer"])

    write_jsonl(output_dir / "results_all.jsonl", all_results)
    for provider in providers:
        provider_rows = [row for row in all_results if row["model"] == provider]
        write_jsonl(output_dir / f"results_{provider}.jsonl", provider_rows)

    review_rows = build_faithfulness_review_rows(all_results)
    write_jsonl(output_dir / "faithfulness_review.jsonl", review_rows)
    write_csv(output_dir / "faithfulness_review.csv", review_rows)

    summary_rows = []
    for provider in providers:
        rows = [row for row in all_results if row["model"] == provider]
        unanswerable = [row for row in rows if not row["answerable"]]
        refusal_acc = mean([row["adversarial_refusal_correct"] for row in unanswerable]) if unanswerable else 0.0
        faithful_values = [int(row["faithful_manual"]) for row in rows if str(row["faithful_manual"]) in {"0", "1"}]
        faithfulness = mean(faithful_values) if faithful_values else None
        summary_rows.append({
            "model": provider,
            "rouge_l": mean(row["rouge_l"] for row in rows),
            "bertscore_f1": mean(row["bertscore_f1"] for row in rows) if compute_bert else "",
            "faithfulness_pct_manual": faithfulness,
            "faithfulness_reviewed": len(faithful_values),
            "adversarial_refusal_accuracy": refusal_acc,
            "citation_presence_rate": mean(float(row["citation_present"]) for row in rows),
            "mean_citation_validity": mean(
                row["citation_validity"] for row in rows if isinstance(row["citation_validity"], float)
            ) if any(isinstance(row["citation_validity"], float) for row in rows) else "",
        })

    write_csv(output_dir / "metrics_summary.csv", summary_rows, list(summary_rows[0].keys()))

    write_report(output_dir / "comparative_report.md", summary_rows)
    return {"results": all_results, "summary": summary_rows}


def write_report(path: str | Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# RAG Model Comparison",
        "",
        "| Model | ROUGE-L | BERTScore F1 | Faithfulness | Adversarial Refusal Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        bert = row["bertscore_f1"]
        bert_text = f"{bert:.3f}" if isinstance(bert, float) else ""
        faithfulness = row["faithfulness_pct_manual"]
        faithfulness_text = f"{faithfulness:.1%}" if isinstance(faithfulness, float) else "pending manual review"
        lines.append(
            f"| {row['model']} | {row['rouge_l']:.3f} | {bert_text} | {faithfulness_text} | "
            f"{row['adversarial_refusal_accuracy']:.1%} |"
        )

    lines.extend([
        "",
        "## Qualitative Analysis",
        "",
        "Review `results_all.jsonl` and `faithfulness_review.csv` after the endpoint run. In this corpus, factual "
        "questions should be grounded by SQL evidence for counts, date range, flairs, and topic summaries, while "
        "opinion questions should cite retrieved comments or topic-summary chunks. Strong answers cite evidence IDs, "
        "describe r/jobs opinions as tendencies, and mention that collected comments are top comments rather than full threads.",
        "",
        "## Manual Faithfulness",
        "",
        "Mark `faithful_manual` as 1 or 0 in `faithfulness_review.csv`, then rerun the same evaluation command. "
        "Existing labels are preserved and aggregated into the faithfulness percentage in the table.",
    ])
    Path(path).write_text("\n".join(lines))
