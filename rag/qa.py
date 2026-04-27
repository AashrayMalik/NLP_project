"""High-level QA orchestration."""

from __future__ import annotations

from rag.llms import Provider, generate_answer
from rag.retriever import HybridRetriever


def answer_question(
    question: str,
    provider: Provider = "retrieval_only",
    retriever: HybridRetriever | None = None,
    final_k: int = 10,
    retrieval_query: str | None = None,
    target_language: str = "english",
) -> dict:
    retriever = retriever or HybridRetriever()
    retrieval_query = retrieval_query or question
    retrieval = retriever.retrieve(retrieval_query, final_k=final_k)
    answer = generate_answer(
        provider=provider,
        question=question,
        context=retrieval["context"],
        evidence=retrieval["evidence"],
        target_language=target_language,
    )
    return {
        "question": question,
        "retrieval_query": retrieval_query,
        "provider": provider,
        "answer": answer,
        "query_type": retrieval["query_type"],
        "evidence": retrieval["evidence"],
        "context": retrieval["context"],
        "debug": retrieval.get("debug", {}),
    }
