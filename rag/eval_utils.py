"""Shared evaluation helpers for English QA, Hindi QA, and retrieval diagnostics."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


UNANSWERABLE_MARKERS = (
    # English
    "not enough evidence",
    "does not contain enough evidence",
    "corpus does not contain",
    "not supported",
    "insufficient evidence",
    # Hindi (Devanagari)
    "पर्याप्त साक्ष्य नहीं",
    "पर्याप्त प्रमाण नहीं",
    "कॉर्पस में पर्याप्त साक्ष्य नहीं",
    "कॉर्पस में पर्याप्त प्रमाण नहीं",
    # Hindi (Romanized)
    "paryaapt saakshya nahin",
    "jaankari uplabdh nahin",
    "koi vishesh jaankari nahin",
    "paryapt evidence nahin",
    "corpus mein paryaapt",
)

_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")
_GENERIC_CITATION_RE = re.compile(r"^Evidence\s", re.IGNORECASE)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_rouge_l(predictions: list[str], references: list[str]) -> list[float]:
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError("rouge-score is not installed. Run `uv sync`.") from exc
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return [scorer.score(ref, pred)["rougeL"].fmeasure for pred, ref in zip(predictions, references)]


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    *,
    lang: str = "en",
    model_type: str | None = None,
) -> list[float]:
    try:
        from bert_score import score
    except ImportError as exc:
        raise RuntimeError("bert-score is not installed. Run `uv sync`.") from exc

    kwargs: dict[str, Any] = {"verbose": False}
    if model_type:
        kwargs["model_type"] = model_type
    else:
        kwargs["lang"] = lang

    _, _, f1 = score(predictions, references, **kwargs)
    return [float(value) for value in f1]


def _char_ngrams(text: str, n: int) -> Counter[str]:
    normalized = " ".join((text or "").split())
    if len(normalized) < n:
        return Counter()
    return Counter(normalized[i:i + n] for i in range(len(normalized) - n + 1))


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_chrf(
    predictions: list[str],
    references: list[str],
    *,
    max_order: int = 6,
    beta: float = 2.0,
) -> list[float]:
    """Compute a lightweight chrF score without adding new runtime dependencies."""
    beta_sq = beta * beta
    scores: list[float] = []
    for pred, ref in zip(predictions, references):
        f_scores = []
        for order in range(1, max_order + 1):
            pred_counts = _char_ngrams(pred, order)
            ref_counts = _char_ngrams(ref, order)
            overlap = sum((pred_counts & ref_counts).values())
            precision = _safe_divide(overlap, sum(pred_counts.values()))
            recall = _safe_divide(overlap, sum(ref_counts.values()))
            denom = beta_sq * precision + recall
            f_scores.append(_safe_divide((1 + beta_sq) * precision * recall, denom))
        scores.append(sum(f_scores) / len(f_scores))
    return scores


def mean_or_blank(values: list[float | None]) -> float | str:
    numeric = [value for value in values if isinstance(value, (int, float))]
    if not numeric:
        return ""
    return sum(float(value) for value in numeric) / len(numeric)


def mean_or_none(values: list[float | int]) -> float | None:
    if not values:
        return None
    return sum(float(value) for value in values) / len(values)


def is_refusal(answer: str) -> bool:
    lower = answer.lower()
    return any(marker in lower for marker in UNANSWERABLE_MARKERS)


def extract_citations(answer: str) -> list[str]:
    raw = [match.strip() for match in _CITATION_RE.findall(answer or "") if match.strip()]
    # Normalize internal whitespace around colons: "comment: abc" → "comment:abc"
    return [re.sub(r'\s*:\s*', ':', c) for c in raw]


def citation_metrics(answer: str, retrieved_ids: list[str]) -> dict[str, Any]:
    citations = extract_citations(answer)
    # Separate real evidence IDs from generic "Evidence N" placeholders
    generic = [c for c in citations if _GENERIC_CITATION_RE.match(c)]
    real_citations = [c for c in citations if not _GENERIC_CITATION_RE.match(c)]
    retrieved = set(retrieved_ids)
    valid = [c for c in real_citations if c in retrieved]
    return {
        "citations": citations,
        "citation_count": len(citations),
        "citation_present": bool(real_citations),
        "real_citations": real_citations,
        "real_citation_count": len(real_citations),
        "generic_citation_count": len(generic),
        "valid_citations": valid,
        "valid_citation_count": len(valid),
        "citation_validity": _safe_divide(len(valid), len(real_citations)) if real_citations else None,
    }


def causal_overclaim_flag(answer: str) -> bool:
    lower = (answer or "").lower()
    patterns = (
        "because",
        "caused by",
        "caused",
        "led to",
        "results in",
        "resulted in",
        "due to",
        "the reason is",
        "की वजह से",
        "के कारण",
        "कारण है",
        "इससे",
    )
    return any(pattern in lower for pattern in patterns)


def flatten_tags(tags: list[str] | None) -> str:
    return "; ".join(tags or [])


def grouped_means(rows: list[dict[str, Any]], key: str, metric_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)

    summary = []
    for group, items in sorted(groups.items()):
        summary_row: dict[str, Any] = {key: group, "n": len(items)}
        for metric_key in metric_keys:
            values = [
                float(item[metric_key])
                for item in items
                if isinstance(item.get(metric_key), (int, float))
            ]
            summary_row[metric_key] = mean_or_none(values)
        summary.append(summary_row)
    return summary


def ensure_parent(path: str | Path) -> Path:
    result = Path(path)
    result.parent.mkdir(parents=True, exist_ok=True)
    return result
