"""LLM endpoint wrappers for grounded QA generation."""

from __future__ import annotations

import logging
import os
import time
from typing import Literal

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

from rag.retriever import retrieval_only_answer

load_dotenv()

Provider = Literal["groq", "gemini", "retrieval_only"]

DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _language_name(target_language: str) -> str:
    lower = target_language.lower()
    if lower in {"hi", "hindi"}:
        return "Hindi"
    return "English"


def build_system_prompt(target_language: str = "english") -> str:
    language_name = _language_name(target_language)
    return f"""You answer questions about the r/jobs subreddit using only the provided corpus evidence.

Rules:
- Use only the evidence in the context.
- If the evidence is weak or absent, say the corpus does not contain enough evidence.
- Do not invent statistics, companies, dates, or claims.
- Summarize community opinions as tendencies, not universal beliefs.
- Remember that comments are collected top comments, not exhaustive full threads.
- Write the answer in {language_name}.
- Cite evidence IDs inline, for example: [comment:abc123].
"""


def build_prompt(question: str, context: str, target_language: str = "english") -> str:
    language_name = _language_name(target_language)
    return f"""Question:
{question}

Corpus evidence:
{context}

Answer with a grounded, concise response in {language_name} and cite the evidence IDs you used."""


def _generate_groq(
    question: str,
    context: str,
    *,
    target_language: str = "english",
    model: str = DEFAULT_GROQ_MODEL,
    max_retries: int = 5,
    initial_delay: float = 30.0,
) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("groq is not installed. Run `uv sync`.") from exc

    client = Groq(api_key=api_key)
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": build_system_prompt(target_language)},
                    {"role": "user", "content": build_prompt(question, context, target_language)},
                ],
                temperature=0.2,
                max_tokens=700,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            err_msg = str(exc).lower()
            is_rate_limit = any(k in err_msg for k in ("429", "rate_limit", "rate limit", "quota"))
            if is_rate_limit and attempt < max_retries:
                logger.warning("Groq rate-limited (attempt %d/%d), retrying in %.0fs…", attempt, max_retries, delay)
                time.sleep(delay)
                delay = min(delay * 2, 600)  # cap at 10 min
            else:
                raise


def _generate_gemini(
    question: str,
    context: str,
    *,
    target_language: str = "english",
    model: str = DEFAULT_GEMINI_MODEL,
    max_retries: int = 5,
    initial_delay: float = 10.0,
) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("google-genai is not installed. Run `uv sync`.") from exc

    client = genai.Client(api_key=api_key)
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=build_prompt(question, context, target_language),
                config=types.GenerateContentConfig(
                    system_instruction=build_system_prompt(target_language),
                    temperature=0.2,
                    max_output_tokens=1000,
                ),
            )
            return response.text or ""
        except Exception as exc:
            err_msg = str(exc).lower()
            is_retryable = any(k in err_msg for k in ("429", "resource_exhausted", "rate", "quota", "503", "unavailable"))
            if is_retryable and attempt < max_retries:
                logger.warning("Gemini retryable error (attempt %d/%d), retrying in %.0fs…", attempt, max_retries, delay)
                time.sleep(delay)
                delay *= 2  # exponential backoff
            else:
                raise


def generate_answer(
    provider: Provider,
    question: str,
    context: str,
    evidence: list[dict] | None = None,
    *,
    target_language: str = "english",
) -> str:
    if provider == "groq":
        return _generate_groq(question, context, target_language=target_language)
    if provider == "gemini":
        return _generate_gemini(question, context, target_language=target_language)
    if provider == "retrieval_only":
        return retrieval_only_answer(question, evidence or [], target_language=target_language)
    raise ValueError(f"Unsupported provider: {provider}")
