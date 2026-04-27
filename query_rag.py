"""CLI for asking one question against the RAG system."""

import argparse
import json

from rag.qa import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question against the r/jobs RAG index.")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--model", choices=["groq", "gemini", "retrieval_only"], default="retrieval_only")
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--language", default="english", help="Target answer language, e.g. english or hindi")
    parser.add_argument("--retrieval-query", help="Optional English retrieval bridge query")
    args = parser.parse_args()

    result = answer_question(
        args.question,
        provider=args.model,
        final_k=args.final_k,
        retrieval_query=args.retrieval_query,
        target_language=args.language,
    )
    print(result["answer"])
    print("\nEvidence:")
    print(json.dumps([
        {
            "id": item["id"],
            "type": item.get("metadata", {}).get("source_type"),
            "rerank_score": round(item.get("rerank_score", 0.0), 3),
            "origins": item.get("retrieval_origins", []),
        }
        for item in result["evidence"]
    ], indent=2))


if __name__ == "__main__":
    main()
