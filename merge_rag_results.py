"""Merge per-model RAG result files into combined outputs.

Run after individual model evaluations to regenerate results_all.jsonl,
metrics_summary.csv, and comparative_report.md without re-calling any LLM.
"""

from pathlib import Path
from statistics import mean

from rag.eval_utils import (
    compute_bertscore,
    compute_rouge_l,
    is_refusal,
    load_jsonl,
    write_csv,
    write_jsonl,
)
from rag.evaluate import build_faithfulness_review_rows, load_manual_faithfulness, write_report


def merge(output_dir: Path, *, compute_bert: bool = False) -> None:
    all_results: list[dict] = []
    providers: list[str] = []

    for path in sorted(output_dir.glob("results_*.jsonl")):
        if path.name == "results_all.jsonl":
            continue
        rows = load_jsonl(path)
        if not rows:
            continue
        model = rows[0]["model"]
        providers.append(model)
        all_results.extend(rows)
        print(f"  Loaded {len(rows)} rows from {path.name} (model={model})")

    if not all_results:
        print("No per-model result files found.")
        return

    # Recompute metrics if missing
    needs_rouge = any("rouge_l" not in r for r in all_results)
    needs_bert = compute_bert and any(r.get("bertscore_f1") is None for r in all_results)

    if needs_rouge:
        predictions = [r["generated_answer"] for r in all_results]
        references = [r["gold_answer"] for r in all_results]
        rouge_l = compute_rouge_l(predictions, references)
        for row, score in zip(all_results, rouge_l):
            row["rouge_l"] = score

    if needs_bert:
        predictions = [r["generated_answer"] for r in all_results]
        references = [r["gold_answer"] for r in all_results]
        bert_f1 = compute_bertscore(predictions, references)
        for row, score in zip(all_results, bert_f1):
            row["bertscore_f1"] = score

    for row in all_results:
        row.setdefault("adversarial_refusal_correct",
                       (not row["answerable"]) and is_refusal(row["generated_answer"]))

    # Load manual faithfulness labels
    manual_labels = load_manual_faithfulness(output_dir)
    for row in all_results:
        key = (str(row["model"]), str(row["id"]))
        if key in manual_labels:
            row["faithful_manual"] = manual_labels[key]

    write_jsonl(output_dir / "results_all.jsonl", all_results)

    review_rows = build_faithfulness_review_rows(all_results)
    write_jsonl(output_dir / "faithfulness_review.jsonl", review_rows)
    write_csv(output_dir / "faithfulness_review.csv", review_rows)

    # Build summary
    summary_rows = []
    for provider in providers:
        rows = [r for r in all_results if r["model"] == provider]
        unanswerable = [r for r in rows if not r["answerable"]]
        refusal_acc = mean([r["adversarial_refusal_correct"] for r in unanswerable]) if unanswerable else 0.0
        faithful_values = [int(r["faithful_manual"]) for r in rows if str(r.get("faithful_manual", "")) in {"0", "1"}]
        faithfulness = mean(faithful_values) if faithful_values else None
        has_bert = any(isinstance(r.get("bertscore_f1"), float) for r in rows)
        has_cv = any(isinstance(r.get("citation_validity"), float) for r in rows)
        summary_rows.append({
            "model": provider,
            "rouge_l": mean(r["rouge_l"] for r in rows),
            "bertscore_f1": mean(r["bertscore_f1"] for r in rows if isinstance(r.get("bertscore_f1"), float)) if has_bert else "",
            "faithfulness_pct_manual": faithfulness,
            "faithfulness_reviewed": len(faithful_values),
            "adversarial_refusal_accuracy": refusal_acc,
            "citation_presence_rate": mean(float(r["citation_present"]) for r in rows),
            "mean_citation_validity": mean(
                r["citation_validity"] for r in rows if isinstance(r["citation_validity"], float)
            ) if has_cv else "",
        })

    write_csv(output_dir / "metrics_summary.csv", summary_rows, list(summary_rows[0].keys()))
    write_report(output_dir / "comparative_report.md", summary_rows)

    print(f"\nMerged {len(all_results)} results from {len(providers)} models: {providers}")
    for row in summary_rows:
        print(f"  {row}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge per-model RAG results.")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--compute-bertscore", action="store_true")
    args = parser.parse_args()
    merge(args.output_dir, compute_bert=args.compute_bertscore)
