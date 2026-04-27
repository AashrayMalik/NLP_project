"""CLI for running the required comparative RAG evaluation."""

import argparse
from pathlib import Path

from rag.evaluate import evaluate_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Groq/Gemini RAG answers against the QA set.")
    parser.add_argument("--qa", type=Path, default=Path("evaluation/qa_set.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--models", nargs="+", choices=["groq", "gemini", "retrieval_only"], default=["groq", "gemini"])
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip BERTScore for faster dry runs")
    args = parser.parse_args()

    summary = evaluate_models(
        qa_path=args.qa,
        providers=args.models,
        output_dir=args.output_dir,
        compute_bert=not args.skip_bertscore,
    )
    for row in summary["summary"]:
        print(row)


if __name__ == "__main__":
    main()
