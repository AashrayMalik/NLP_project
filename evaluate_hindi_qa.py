"""CLI for Hindi cross-lingual QA evaluation."""

import argparse
from pathlib import Path

from rag.hindi_eval import evaluate_hindi_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Hindi cross-lingual QA over the English r/jobs corpus.")
    parser.add_argument("--qa", type=Path, default=Path("evaluation/hindi_qa_set.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/hindi_results"))
    parser.add_argument("--models", nargs="+", choices=["groq", "gemini", "retrieval_only"], default=["groq", "gemini"])
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip multilingual BERTScore for faster runs")
    parser.add_argument("--limit", type=int, default=None, help="Max number of questions to evaluate per model")
    parser.add_argument(
        "--bert-model",
        default="bert-base-multilingual-cased",
        help="Multilingual model name to use for BERTScore when not skipped",
    )
    args = parser.parse_args()

    summary = evaluate_hindi_models(
        qa_path=args.qa,
        providers=args.models,
        output_dir=args.output_dir,
        compute_bert=not args.skip_bertscore,
        bert_model=args.bert_model,
        limit=args.limit,
    )
    for row in summary["summary"]:
        print(row)


if __name__ == "__main__":
    main()
