"""
reddit_jargon.py — Reddit and r/jobs slang normalisation
=========================================================
Expands common abbreviations and Reddit-specific informal language
at word boundaries so that embedding models can match semantically
equivalent text.  This is applied at chunk-creation time, not
retroactively to the cleaned database.
"""

from __future__ import annotations

import re

# ── Abbreviation dictionary (r/jobs domain) ────────────────────────────
# Keys are UPPERCASE; matching is case-insensitive at word boundaries.

REDDIT_SLANG: dict[str, str] = {
    # Workplace / HR
    "PIP": "performance improvement plan",
    "HRBP": "HR business partner",
    "OKR": "objectives and key results",
    "KPI": "key performance indicator",
    "IC": "individual contributor",
    "EM": "engineering manager",
    "TL": "team lead",
    "PM": "project manager",
    "TPM": "technical program manager",
    "SWE": "software engineer",
    "DE": "data engineer",
    "DS": "data scientist",
    "QA": "quality assurance",
    "BA": "business analyst",
    "EA": "executive assistant",
    "VP": "vice president",
    "C-SUITE": "C-suite executive leadership",
    "SME": "subject matter expert",
    "FTE": "full-time employee",
    "RIF": "reduction in force",
    "WARN": "WARN Act layoff notice",

    # Compensation
    "TC": "total compensation",
    "RSU": "restricted stock unit",
    "RSUs": "restricted stock units",
    "OTE": "on-target earnings",
    "W2": "W-2 tax employee",
    "1099": "1099 independent contractor",
    "COL": "cost of living",
    "COLA": "cost of living adjustment",
    "ESPP": "employee stock purchase plan",
    "COBRA": "COBRA health insurance continuation",
    "EAP": "employee assistance program",

    # Work arrangements
    "WFH": "work from home",
    "RTO": "return to office",
    "OE": "overemployment",
    "PTO": "paid time off",
    "OOO": "out of office",
    "EOD": "end of day",
    "COB": "close of business",
    "FMLA": "Family and Medical Leave Act",

    # Job search
    "ATS": "applicant tracking system",
    "JD": "job description",
    "NDA": "non-disclosure agreement",
    "NCA": "non-compete agreement",
    "LOE": "level of experience",
    "YOE": "years of experience",
    "LC": "LeetCode",
    "OA": "online assessment",
    "HM": "hiring manager",
    "BGC": "background check",

    # Companies / tiers
    "FAANG": "Facebook Apple Amazon Netflix Google top tech companies",
    "MAANG": "Meta Apple Amazon Netflix Google top tech companies",
    "WITCH": "Wipro Infosys TCS Cognizant HCL outsourcing companies",
    "B4": "Big Four accounting firms",
    "MBB": "McKinsey Bain BCG management consulting",

    # Reddit-specific
    "TLDR": "summary",
    "TL;DR": "summary",
    "IMO": "in my opinion",
    "IMHO": "in my humble opinion",
    "FWIW": "for what it's worth",
    "AFAIK": "as far as I know",
    "YMMV": "your mileage may vary",
    "DAE": "does anyone else",
    "ELI5": "explain like I'm five",
    "IANAL": "I am not a lawyer",
    "IME": "in my experience",
    "FYI": "for your information",
    "IIRC": "if I recall correctly",
    "PSA": "public service announcement",
    "LPT": "life pro tip",
    "ITT": "in this thread",
    "TIL": "today I learned",
    "AITA": "am I the asshole",
    "NTA": "not the asshole",
    "YTA": "you're the asshole",
    "SMH": "shaking my head",
}

# Sarcasm marker — strip "/s" tag that Reddit uses to mark sarcasm
_RE_SARCASM_TAG = re.compile(r"\s*/s\b", re.IGNORECASE)

# Build a single regex that matches any known abbreviation at word boundaries.
# Sort by length (longest first) so "TL;DR" matches before "TL".
_sorted_keys = sorted(REDDIT_SLANG.keys(), key=len, reverse=True)
_escaped = [re.escape(k) for k in _sorted_keys]
_JARGON_RE = re.compile(
    r"\b(" + "|".join(_escaped) + r")\b",
    re.IGNORECASE,
)


def _replace_match(match: re.Match) -> str:
    """Replace the matched abbreviation with its expansion."""
    key = match.group(0).upper()
    # Handle keys with special characters like "TL;DR"
    for k, v in REDDIT_SLANG.items():
        if k == key:
            return f"{match.group(0)} ({v})"
    # Fallback: try without exact case
    expansion = REDDIT_SLANG.get(key)
    if expansion:
        return f"{match.group(0)} ({expansion})"
    return match.group(0)


def normalize_jargon(text: str) -> str:
    """Expand Reddit/workplace abbreviations in-place, keeping the original
    abbreviation for readability and appending the expansion in parentheses.

    Example: "Got PIP'd at work" → "Got PIP (performance improvement plan)'d at work"
    """
    if not text:
        return text

    # Remove sarcasm tags
    text = _RE_SARCASM_TAG.sub("", text)

    # Expand abbreviations
    text = _JARGON_RE.sub(_replace_match, text)

    return text
