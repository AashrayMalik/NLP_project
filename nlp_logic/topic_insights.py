"""
topic_insights.py — Topic label cleanup and flair coverage analysis
===================================================================
Lightweight heuristics used by the app and preprocessing pipelines to:
  - turn BERTopic keyword lists into cleaner display labels
  - compare discovered topics against existing Reddit post flairs
  - highlight candidate flair gaps where topics are only weakly covered
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "scraping" / "data" / "jobs_posts.db"

GENERIC_TOPIC_TERMS = {
    "actually", "advice", "americans", "ask", "asked", "asking", "current",
    "currently", "data", "day", "days", "did", "doing", "help", "hour",
    "hours", "human", "legit", "life", "line", "looking", "month", "months",
    "notice", "question", "questions", "real", "search", "searching",
    "started", "test", "tests", "tips", "tool", "tools", "use", "using",
    "week", "weeks", "year", "years", "000", "2025",
}

FLAIR_ALIASES = {
    "Applications": {"application", "applications", "apply", "applied", "applying"},
    "Compensation": {"benefit", "benefits", "compensation", "hourly", "negotiate",
                      "negotiation", "pay", "raise", "range", "salary", "wage"},
    "Interviews": {"dress", "interview", "interviews", "question", "questions",
                    "screen", "screening", "tips", "wear"},
    "Job searching": {"job", "jobs", "looking", "market", "remote", "search",
                       "searching", "unemployed"},
    "Office relations": {"boss", "coworker", "coworkers", "hr", "manager", "office",
                          "performance", "pip", "team"},
    "Onboarding": {"employment", "offer", "onboarding", "orientation", "start"},
    "Post-interview": {"callback", "email", "follow", "offer", "postinterview",
                        "recruiter"},
    "References": {"reference", "references"},
    "Recruiters": {"linkedin", "recruiter", "recruiters"},
    "Resumes/CVs": {"cover", "cv", "cvs", "letter", "letters", "resume", "resumes"},
    "Unemployment": {"fired", "layoff", "layoffs", "termination", "unemployment"},
    "Work/Life balance": {"balance", "commute", "home", "hybrid", "office", "remote", "rto"},
}

SPECIAL_TOKEN_FORMATS = {
    "ai": "AI",
    "cv": "CV",
    "cvs": "CVs",
    "hr": "HR",
    "linkedin": "LinkedIn",
    "pip": "PIP",
    "rto": "RTO",
    "ssn": "SSN",
}


def curated_topic_label(keywords: list[str]) -> str | None:
    """Return a cleaner human label for common recurring topic shapes."""
    tokens = topic_tokens_from_keywords(keywords)

    if {"background", "check"} <= tokens:
        return "Background Checks"
    if {"cover", "letter"} <= tokens:
        return "Cover Letters"
    if "drug" in tokens and ("testing" in tokens or "weed" in tokens or "pass" in tokens):
        return "Drug Testing"
    if "ai" in tokens and "tech" in tokens:
        return "AI and Tech Disruption"
    if ("market" in tokens and ("layoff" in tokens or "economy" in tokens)) or (
        "unemployment" in tokens and ("layoff" in tokens or "economy" in tokens)
    ):
        return "Layoffs and the Job Market"
    if (
        ("salary" in tokens or "raise" in tokens or "negotiate" in tokens)
        and ("offer" in tokens or "pay" in tokens or "range" in tokens)
    ):
        return "Salary Negotiation and Raises"
    if ("degree" in tokens and "college" in tokens) or (
        "degree" in tokens and "experience" in tokens
    ) or ("college" in tokens and "experience" in tokens):
        return "Entry-Level Job Search and Career Starts"
    if "reference" in tokens or "references" in tokens:
        return "References and Reference Checks"
    if "pip" in tokens and "performance" in tokens:
        return "PIPs and Performance Reviews"
    if "fired" in tokens and ("boss" in tokens or "manager" in tokens):
        return "Being Fired and Manager Conflict"
    if "quit" in tokens and ("manager" in tokens or "pay" in tokens or "hours" in tokens):
        return "Quitting and Workplace Burnout"
    if "linkedin" in tokens and ("scam" in tokens or "recruiter" in tokens):
        return "LinkedIn Scams and Recruiter Legitimacy"
    if "interview" in tokens and ("dress" in tokens or "nervous" in tokens or "wear" in tokens):
        return "Interview Prep and First-Time Nerves"
    if "interview" in tokens and ("offer" in tokens or "email" in tokens or "recruiter" in tokens):
        return "Interview Follow-Ups and Offer Delays"
    if (
        ("resume" in tokens or "application" in tokens or "apply" in tokens or "applying" in tokens)
        and ("experience" in tokens or "interview" in tokens)
    ):
        return "Applications, Resumes, and Search Strategy"
    if "commute" in tokens and ("remote" in tokens or "benefit" in tokens or "pay" in tokens):
        return "Pay, Commute, and Remote Trade-Offs"
    if ("remote" in tokens and "hybrid" in tokens) or ("remote" in tokens and "office" in tokens):
        return "Remote vs Hybrid Work"

    return None


def canonical_token(token: str) -> str:
    """Normalise a token for matching without being too aggressive."""
    token = token.strip().lower()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Split text into alphanumeric tokens and normalise simple variants."""
    return [canonical_token(t) for t in re.findall(r"[a-zA-Z0-9]+", text.lower())]


def prettify_phrase(text: str) -> str:
    """Title-case a keyword phrase while preserving common acronyms."""
    words = []
    for raw in text.split():
        lower = raw.lower()
        words.append(SPECIAL_TOKEN_FORMATS.get(lower, raw.title()))
    return " ".join(words)


def topic_tokens_from_keywords(keywords: list[str]) -> set[str]:
    """Return the specific non-generic tokens represented by a topic."""
    tokens = set()
    for keyword in keywords:
        for token in tokenize(keyword):
            if token not in GENERIC_TOPIC_TERMS:
                tokens.add(token)
    return tokens


def build_topic_display_label(
    keywords: list[str],
    fallback: str | None = None,
    max_terms: int = 3,
) -> str:
    """Turn noisy BERTopic keywords into a cleaner short label."""
    curated = curated_topic_label(keywords)
    if curated:
        return curated

    cleaned_keywords = []
    seen_phrases = set()

    for keyword in keywords:
        phrase = " ".join(keyword.strip().split())
        if not phrase:
            continue
        lower = phrase.lower()
        if lower in seen_phrases:
            continue
        seen_phrases.add(lower)
        tokens = [t for t in tokenize(lower) if t not in GENERIC_TOPIC_TERMS]
        if not tokens:
            continue
        cleaned_keywords.append((phrase, tokens))

    cleaned_keywords.sort(
        key=lambda item: (
            len(item[1]) > 1,
            len(item[1]),
            sum(len(t) for t in item[1]),
        ),
        reverse=True,
    )

    selected: list[tuple[str, set[str]]] = []
    covered_tokens: set[str] = set()

    for phrase, tokens in cleaned_keywords:
        token_set = set(tokens)
        if token_set <= covered_tokens:
            continue
        selected.append((phrase, token_set))
        covered_tokens |= token_set
        if len(selected) >= max_terms:
            break

    if not selected and fallback:
        return fallback

    label_parts = [prettify_phrase(phrase) for phrase, _ in selected]
    return " / ".join(label_parts) if label_parts else (fallback or "Topic")


def suggest_missing_flair(keywords: list[str], display_label: str) -> str | None:
    """Produce a human-friendly candidate flair name for uncovered topics."""
    tokens = topic_tokens_from_keywords(keywords)

    if "scam" in tokens or ("linkedin" in tokens and "interview" in tokens):
        return "Job scams"
    if {"background", "check"} <= tokens:
        return "Background checks"
    if {"pip", "performance"} & tokens:
        return "Performance plans"
    if ({"remote", "hybrid"} <= tokens) or ({"remote", "office"} <= tokens):
        return "Remote / hybrid work"
    if "drug" in tokens and ("testing" in tokens or "weed" in tokens or "pass" in tokens):
        return "Drug tests"
    if "ai" in tokens:
        return "AI in hiring/work"

    return None


def flair_overlap_score(topic_tokens: set[str], flair: str) -> int:
    """Score how well an existing flair lexically matches a topic."""
    flair_tokens = {t for t in tokenize(flair) if t not in GENERIC_TOPIC_TERMS}
    flair_tokens |= {canonical_token(t) for t in FLAIR_ALIASES.get(flair, set())}
    overlap = topic_tokens & flair_tokens
    return len(overlap)


def compute_topic_flair_analysis(db_path: str | Path | None = None) -> list[dict]:
    """
    Compare topic clusters to post flairs and flag coverage gaps.

    Returns a list of topic-level records ordered by topic size.
    """
    db = str(db_path or DEFAULT_DB)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT
            t.id,
            t.label,
            t.keywords,
            t.post_count,
            t.share_pct,
            COALESCE(NULLIF(TRIM(p.flair), ''), 'No Flair') AS flair,
            COUNT(*) AS cnt
        FROM topics t
        JOIN post_topics pt ON pt.topic_id = t.id
        JOIN posts p ON p.id = pt.post_id
        WHERE t.id != -1
        GROUP BY t.id, t.label, t.keywords, t.post_count, t.share_pct, flair
        ORDER BY t.post_count DESC, cnt DESC
    """).fetchall()

    conn.close()

    grouped: dict[int, dict] = {}
    for row in rows:
        topic_id = row["id"]
        if topic_id not in grouped:
            keywords = json.loads(row["keywords"]) if row["keywords"] else []
            display_label = build_topic_display_label(keywords, fallback=row["label"])
            grouped[topic_id] = {
                "topic_id": topic_id,
                "raw_label": row["label"],
                "display_label": display_label,
                "keywords": keywords,
                "post_count": row["post_count"],
                "share_pct": row["share_pct"],
                "flairs": [],
            }

        grouped[topic_id]["flairs"].append({
            "flair": row["flair"],
            "count": row["cnt"],
            "pct": round(100.0 * row["cnt"] / max(row["post_count"], 1), 1),
        })

    analysis = []
    for topic in grouped.values():
        topic_tokens = topic_tokens_from_keywords(topic["keywords"])
        flairs = sorted(topic["flairs"], key=lambda item: item["count"], reverse=True)
        dominant_flair = flairs[0]["flair"] if flairs else "No Flair"
        dominant_share = flairs[0]["pct"] if flairs else 0.0
        no_flair_share = next((f["pct"] for f in flairs if f["flair"] == "No Flair"), 0.0)

        best_lexical_flair = None
        best_lexical_score = -1
        for flair_info in flairs:
            score = flair_overlap_score(topic_tokens, flair_info["flair"])
            if score > best_lexical_score:
                best_lexical_score = score
                best_lexical_flair = flair_info["flair"]

        if dominant_share >= 45:
            flair_fit = "Strong"
        elif dominant_share >= 30 or best_lexical_score >= 2:
            flair_fit = "Partial"
        else:
            flair_fit = "Gap"

        suggested_missing = suggest_missing_flair(topic["keywords"], topic["display_label"])
        gap_reason = None
        if suggested_missing and topic["post_count"] >= 75 and dominant_share < 45:
            gap_reason = (
                f"No single flair owns this topic (top flair: {dominant_flair}, {dominant_share:.1f}% of posts)."
            )
        else:
            suggested_missing = None

        analysis.append({
            "topic_id": topic["topic_id"],
            "display_label": topic["display_label"],
            "raw_label": topic["raw_label"],
            "keywords": topic["keywords"],
            "post_count": topic["post_count"],
            "share_pct": topic["share_pct"],
            "top_flairs": flairs[:3],
            "dominant_flair": dominant_flair,
            "dominant_flair_pct": dominant_share,
            "no_flair_pct": no_flair_share,
            "best_lexical_flair": best_lexical_flair,
            "best_lexical_score": best_lexical_score,
            "flair_fit": flair_fit,
            "suggested_missing_flair": suggested_missing,
            "gap_reason": gap_reason,
        })

    analysis.sort(key=lambda item: item["post_count"], reverse=True)
    return analysis
