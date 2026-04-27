"""CLI for extended RAG diagnostics and comment probes."""

import argparse
from pathlib import Path

from rag.diagnostics import evaluate_diagnostic_queries, run_comment_probes


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structure, comment, and causal-overclaim diagnostics.")
    parser.add_argument("--qa", type=Path, default=Path("evaluation/rag_diagnostic_set.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/diagnostics"))
    parser.add_argument("--model", choices=["groq", "gemini", "retrieval_only"], default="retrieval_only")
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--comment-sample-size", type=int, default=24)
    args = parser.parse_args()

    diagnostics = evaluate_diagnostic_queries(
        dataset_path=args.qa,
        output_dir=args.output_dir,
        provider=args.model,
        final_k=args.final_k,
    )
    comment_probes = run_comment_probes(
        output_dir=args.output_dir,
        sample_size=args.comment_sample_size,
        final_k=args.final_k,
    )

    for row in diagnostics["summary"]:
        print(row)
    for row in comment_probes["summary"]:
        print(row)


if __name__ == "__main__":
    main()
