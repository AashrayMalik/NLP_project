"""
app.py — Streamlit interactive application for r/jobs subreddit analysis
=========================================================================
Multi-page app with sidebar navigation:
  - Dashboard: KPI cards, monthly trends, flair distribution, author stats
  - Topics: Topic cards with labels, keywords, share %
  - Trends: Trending vs persistent topic classification
  - Stance: Stance detection and argument summarisation
  - QA: Semantic KG hybrid RAG question answering

Run with: streamlit run app.py
"""

import json
import sqlite3
from pathlib import Path
from html import escape

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nlp_logic.topic_insights import build_topic_display_label, compute_topic_flair_analysis

# ── Config ──────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parent / "scraping" / "data" / "jobs_posts.db"
CACHE_DIR = Path(__file__).resolve().parent / "scraping" / "data"
EVALUATION_DIR = Path(__file__).resolve().parent / "evaluation"

st.set_page_config(
    page_title="r/jobs Subreddit Analysis",
    page_icon="J",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — bold solid editorial theme ────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #f4efe8;
        --panel: #fffaf3;
        --panel-strong: #ebe1d5;
        --panel-soft: #f8f0e6;
        --line: #d8cbbb;
        --line-strong: #c4b39f;
        --text: #2d2926;
        --muted: #746a61;
        --dim: #9b8f83;
        --coral: #d9817a;
        --cyan: #7fb8b4;
        --green: #8bbf9f;
        --amber: #d9aa63;
        --violet: #a995c7;
        --blue: #8aa6c8;
        --rose: #e7b7b0;
        --shadow: 0 12px 24px rgba(99, 82, 62, 0.12);
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .stApp {
        color: var(--text);
        background: var(--bg);
    }

    .stApp::before {
        content: none;
    }

    h1, h2, h3, h4, h5, h6, p, label, span, div {
        letter-spacing: 0;
    }
    h1, h2, h3, h4, .page-title, .brand-title, .kpi-value {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 2.1rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    .stMarkdown a {
        color: #327b78;
    }

    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stPlotlyChart"]) {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.95rem 0.95rem 0.35rem;
        box-shadow: var(--shadow);
    }

    div[data-testid="stPlotlyChart"] {
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Hero headers ── */
    .page-hero {
        position: relative;
        overflow: hidden;
        background: #fff6e9;
        border: 1px solid var(--line-strong);
        border-radius: 8px;
        padding: 1.45rem 1.6rem 1.35rem;
        margin-bottom: 1.35rem;
        box-shadow: var(--shadow);
    }
    .page-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        border-top: 8px solid var(--cyan);
        border-right: 8px solid var(--rose);
        opacity: 1;
        pointer-events: none;
    }
    .page-eyebrow {
        position: relative;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: #11131b;
        background: #efe3d4;
        border-radius: 999px;
        padding: 0.28rem 0.62rem;
        font-size: 0.68rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .page-eyebrow::before {
        content: "";
        width: 0.45rem;
        height: 0.45rem;
        border-radius: 999px;
        background: var(--cyan);
    }
    .page-title {
        position: relative;
        font-size: clamp(1.8rem, 4vw, 3.8rem);
        line-height: 0.94;
        font-weight: 900;
        color: var(--text);
        margin: 0.85rem 0 0.55rem;
        max-width: 980px;
        text-transform: none;
    }
    .page-subtitle {
        position: relative;
        font-size: 1rem;
        color: var(--muted);
        max-width: 760px;
        line-height: 1.55;
        margin: 0;
    }
    .page-meta-row {
        position: relative;
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1.1rem;
    }
    .page-meta-pill {
        border: 1px solid rgba(255,255,255,0.18);
        background: #f5eadf;
        color: var(--text);
        border-radius: 999px;
        padding: 0.38rem 0.72rem;
        font-size: 0.76rem;
        font-weight: 700;
    }

    /* ── KPI Cards ── */
    .kpi-card {
        position: relative;
        overflow: hidden;
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1.05rem 1rem 1rem;
        min-height: 126px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: var(--shadow);
    }
    .kpi-card::before {
        content: "";
        position: absolute;
        inset: 0 0 auto;
        height: 3px;
        background: var(--cyan);
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.28);
        box-shadow: var(--shadow);
    }
    .kpi-value {
        font-size: clamp(1.4rem, 2.5vw, 2rem);
        font-weight: 900;
        color: var(--text);
        margin-top: 0.7rem;
        letter-spacing: 0;
    }
    .kpi-label {
        font-size: 0.72rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
    }

    /* ── Topic cards ── */
    .topic-badge {
        display: inline-block;
        background: #edf7f4;
        color: #377c75;
        padding: 0.22rem 0.58rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
        border: 1px solid #c8e3dd;
    }

    .topic-grid-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1.15rem 1.25rem;
        margin-bottom: 0.8rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .topic-grid-card:hover {
        transform: translateY(-2px);
        border-color: var(--cyan);
    }
    .topic-grid-card h4 {
        margin: 0 0 0.4rem 0;
        font-size: 0.95rem;
        font-weight: 850;
        color: var(--text);
    }
    .topic-grid-card .topic-stats {
        font-size: 0.8rem;
        color: var(--muted);
        margin-bottom: 0.5rem;
    }

    /* ── Trend badges ── */
    .badge-persistent {
        display: inline-block;
        background: #edf7ef;
        color: #4f8267;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 800;
        border: 1px solid #c9dfd0;
    }
    .badge-trending {
        display: inline-block;
        background: #fff4dc;
        color: #916a28;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 800;
        border: 1px solid #e9d2a3;
    }
    .badge-seasonal {
        display: inline-block;
        background: #edf3fb;
        color: #55779b;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 800;
        border: 1px solid #cfddec;
    }

    /* ── Stance bars ── */
    .stance-support {
        background: #f0f8f2;
        border-left: 3px solid var(--green);
        padding: 0.7rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .stance-oppose {
        background: #fff0ef;
        border-left: 3px solid var(--coral);
        padding: 0.7rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .stance-neutral {
        background: #f3f1ee;
        border-left: 3px solid #9b8f83;
        padding: 0.7rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* ── Page headers ── */
    .page-header {
        font-size: 1.4rem;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 0.15rem;
    }
    .page-subheader {
        font-size: 0.88rem;
        color: var(--muted);
        margin-bottom: 1.8rem;
        line-height: 1.45;
    }

    /* ── Section dividers ── */
    .section-divider {
        height: 3px;
        background: var(--line);
        margin: 1.8rem 0;
    }

    /* ── Insight card ── */
    .insight-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-left: 4px solid var(--cyan);
        border-radius: 8px;
        padding: 1.05rem 1.25rem;
        margin: 1rem 0 1.5rem 0;
        font-size: 0.88rem;
        line-height: 1.65;
        color: var(--muted);
    }
    .insight-card strong {
        color: var(--text);
    }

    /* ── Answer panel (QA page) ── */
    .answer-panel {
        background: var(--panel);
        border: 1px solid var(--line-strong);
        border-radius: 8px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        line-height: 1.7;
        font-size: 0.92rem;
        box-shadow: var(--shadow);
    }

    /* ── Example question chips ── */
    .example-chip {
        display: inline-block;
        background: #f7efe5;
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.4rem 0.8rem;
        margin: 0.25rem 0.3rem;
        font-size: 0.78rem;
        color: var(--muted);
        cursor: default;
        transition: border-color 0.15s ease;
    }
    .example-chip:hover {
        border-color: var(--cyan);
        color: var(--text);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #efe4d7;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }
    .sidebar-brand {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        background: #fffaf3;
        margin-bottom: 0.95rem;
    }
    .brand-mark {
        display: inline-flex;
        width: 2.2rem;
        height: 2.2rem;
        align-items: center;
        justify-content: center;
        border-radius: 8px;
        background: var(--cyan);
        color: #173d3b;
        font-weight: 900;
        margin-bottom: 0.75rem;
    }
    .brand-title {
        font-size: 1.25rem;
        font-weight: 900;
        color: var(--text);
        line-height: 1;
    }
    .brand-subtitle {
        font-size: 0.72rem;
        color: var(--muted);
        margin-top: 0.4rem;
        line-height: 1.4;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: var(--text);
        font-size: 0.9rem;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        color: var(--cyan);
    }
    [data-testid="stSidebar"] * {
        color: var(--text);
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--line);
        margin: 1rem 0;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 8px;
        padding: 0.42rem 0.55rem;
        margin: 0.12rem 0;
        transition: background 0.15s ease, color 0.15s ease;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #fff6e9;
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.8rem;
    }

    /* ── Summary stat highlight ── */
    .stat-highlight {
        display: inline-block;
        background: #fff0ef;
        color: #9c5d57;
        padding: 0.1rem 0.5rem;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.88rem;
    }

    .stButton > button {
        border-radius: 8px;
        border: 1px solid var(--line-strong);
        background: var(--cyan);
        color: #173d3b;
        font-weight: 900;
        min-height: 2.7rem;
        box-shadow: var(--shadow);
    }
    .stButton > button:hover {
        border-color: rgba(255,255,255,0.35);
        color: #08090d;
        filter: brightness(1.06);
    }
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 8px;
        border-color: var(--line-strong);
        background-color: #fffaf3;
    }
    .stDataFrame, div[data-testid="stExpander"] {
        border-radius: 8px;
    }

    .note-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 1rem 0 1.2rem;
    }
    .note-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem 1.05rem;
        min-height: 132px;
    }
    .note-card h4 {
        margin: 0 0 0.45rem 0;
        color: var(--text);
        font-size: 0.95rem;
        font-weight: 900;
    }
    .note-card p {
        margin: 0;
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.55;
    }
    .probe-card {
        border-left: 4px solid var(--cyan);
    }
    .ethics-card {
        border-left: 4px solid var(--amber);
    }
    .mini-label {
        color: var(--muted);
        font-size: 0.74rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.35rem;
    }
    .comparison-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 1rem 0 1.3rem;
    }
    .comparison-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: var(--shadow);
    }
    .comparison-card h4 {
        margin: 0.25rem 0 0.3rem 0;
        color: var(--text);
        font-size: 1rem;
    }
    .comparison-card p {
        margin: 0;
        color: var(--muted);
        font-size: 0.84rem;
        line-height: 1.5;
    }
    .stance-strip {
        display: flex;
        width: 100%;
        height: 0.78rem;
        overflow: hidden;
        border-radius: 999px;
        border: 1px solid var(--line);
        margin: 0.6rem 0 0.7rem;
    }
    .stance-for {
        background: var(--green);
    }
    .stance-against {
        background: var(--coral);
    }
    .stance-mixed {
        background: #b7afa5;
    }
    .soft-callout {
        background: #fff6e9;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.55;
    }

    @media (max-width: 760px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .page-hero {
            padding: 1.1rem;
        }
        .page-title {
            font-size: 2.2rem;
        }
        .kpi-card {
            min-height: auto;
        }
        .note-grid {
            grid-template-columns: 1fr;
        }
        .comparison-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_agg_cache():
    """Load precomputed aggregate statistics."""
    path = CACHE_DIR / "agg_cache.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    from nlp_logic.agg import compute_all
    return compute_all(DB_PATH)


@st.cache_data(ttl=3600)
def load_topics_from_db():
    """Load topic metadata from DB."""
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT id, label, keywords, post_count, share_pct, trend_type
            FROM topics
            ORDER BY CASE WHEN id = -1 THEN 999 ELSE post_count END DESC
        """).fetchall()
        topics = []
        for r in rows:
            keywords = json.loads(r["keywords"]) if r["keywords"] else []
            topics.append({
                "id": r["id"],
                "label": r["label"],
                "display_label": build_topic_display_label(keywords, fallback=r["label"]),
                "keywords": keywords,
                "post_count": r["post_count"],
                "share_pct": r["share_pct"],
                "trend_type": r["trend_type"] if "trend_type" in r.keys() else None,
            })
        return topics
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


@st.cache_data(ttl=3600)
def load_representative_docs():
    """Load representative docs from JSON cache."""
    path = CACHE_DIR / "representative_docs.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600)
def load_trending_cache():
    """Load trending/persistent classification data."""
    path = CACHE_DIR / "trending_cache.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


@st.cache_data(ttl=3600)
def load_stance_cache():
    """Load stance analysis results."""
    path = CACHE_DIR / "stance_cache.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=3600)
def load_topic_flair_analysis():
    """Load topic-vs-flair coverage analysis."""
    return compute_topic_flair_analysis(DB_PATH)


@st.cache_data(ttl=3600)
def load_csv_if_exists(path: Path):
    """Load a CSV file if it exists, otherwise return an empty dataframe."""
    import pandas as pd

    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_jsonl_if_exists(path: Path, limit: int | None = None):
    """Load a JSONL file into a list of dicts."""
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


REPORT_ENGLISH_BERTSCORE = {
    "gemini": 0.872,
    "groq": 0.851,
}

REPORT_HINDI_GEMINI = {
    "model": "gemini",
    "chrf": 0.261,
    "bertscore_f1": 0.781,
    "citation_presence_rate": 0.80,
    "mean_citation_validity": 0.738,
    "adversarial_refusal_accuracy": 1.0,
    "fluency_mean_manual": 3.8,
    "fluency_reviewed": 30,
    "adequacy_mean_manual": 4.1,
    "adequacy_reviewed": 30,
    "causal_claim_rate": None,
    "source": "report.tex",
}


def model_label(model: str) -> str:
    """Human-readable model labels for result tables."""
    labels = {
        "gemini": "Gemini 2.5 Flash",
        "groq": "Groq (LLaMA 3.3-70B)",
    }
    return labels.get(model, model)


def without_retrieval_only(df):
    """Remove retrieval-only baseline rows from model-comparison tables."""
    if df.empty or "model" not in df:
        return df
    return df[df["model"].astype(str).str.lower() != "retrieval_only"].copy()


def add_report_hindi_gemini(df):
    """Add the Gemini Hindi row from report.tex when it is not in the CSV artifacts."""
    import pandas as pd

    df = without_retrieval_only(df)
    if df.empty:
        return pd.DataFrame([REPORT_HINDI_GEMINI])
    if not (df["model"].astype(str).str.lower() == "gemini").any():
        df = pd.concat([df, pd.DataFrame([REPORT_HINDI_GEMINI])], ignore_index=True)
    return df


@st.cache_data(ttl=3600)
def load_bias_probe_summary():
    """Run lightweight corpus probes for identity-sensitive bias discussions."""
    if not DB_PATH.exists():
        return [], []

    probes = [
        {
            "probe": "Gender, pregnancy, and parenthood",
            "question": "Do posts describe gendered workplace treatment, pregnancy, maternity, or parenthood penalties?",
            "patterns": ["pregnan", "maternity", "female", "woman", "mother", "parenthood"],
        },
        {
            "probe": "Race, nationality, and visa status",
            "question": "Do posts connect hiring outcomes to race, nationality, immigration status, or visa constraints?",
            "patterns": ["race", "black", "asian", "immigrant", "visa", "nationality"],
        },
        {
            "probe": "Age and seniority",
            "question": "Do users report ageism, being too old, or age-coded rejection?",
            "patterns": ["ageism", "older worker", "too old", "over 50", "over 40", "age discrimination"],
        },
        {
            "probe": "Disability, neurodiversity, and health",
            "question": "Do posts discuss disability, ADHD, autism, mental health, or accommodation risk?",
            "patterns": ["disab", "adhd", "autis", "mental health", "accommodation", "chronic illness"],
        },
    ]

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        total_posts = conn.execute("SELECT COUNT(*) AS n FROM posts").fetchone()["n"]
        summary = []
        evidence = []
        for probe in probes:
            clauses = ["lower(coalesce(title, '') || ' ' || coalesce(body, '')) LIKE ?" for _ in probe["patterns"]]
            params = [f"%{pattern.lower()}%" for pattern in probe["patterns"]]
            where = " OR ".join(clauses)

            count = conn.execute(f"SELECT COUNT(*) AS n FROM posts WHERE {where}", params).fetchone()["n"]
            summary.append({
                "Probe": probe["probe"],
                "Probe question": probe["question"],
                "Matching posts": count,
                "Share of corpus": count / total_posts if total_posts else 0,
            })

            rows = conn.execute(f"""
                SELECT id, title, body, score, flair, month
                FROM posts
                WHERE {where}
                ORDER BY score DESC
                LIMIT 3
            """, params).fetchall()
            for row in rows:
                snippet = " ".join((row["body"] or "").split())[:260]
                evidence.append({
                    "Probe": probe["probe"],
                    "Post ID": row["id"],
                    "Title": row["title"],
                    "Snippet": snippet + ("..." if len(" ".join((row["body"] or "").split())) > 260 else ""),
                    "Score": row["score"],
                    "Flair": row["flair"],
                    "Month": row["month"],
                })
        return summary, evidence
    finally:
        conn.close()


def kpi_card(label: str, value: str):
    """Render a styled KPI card."""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def page_hero(title: str, subtitle: str, eyebrow: str, pills: list[str] | None = None):
    """Render the bold page introduction block."""
    pills = pills or []
    pills_html = "".join(f'<span class="page-meta-pill">{escape(pill)}</span>' for pill in pills)
    st.markdown(f"""
    <section class="page-hero">
        <div class="page-title">{escape(title)}</div>
        <p class="page-subtitle">{escape(subtitle)}</p>
        <div class="page-meta-row">{pills_html}</div>
    </section>
    """, unsafe_allow_html=True)


def trend_badge_html(trend_type: str | None) -> str:
    """Return an HTML badge for a trend type."""
    if not trend_type:
        return ""
    css_class = {
        "persistent": "badge-persistent",
        "trending": "badge-trending",
        "seasonal": "badge-seasonal",
    }.get(trend_type, "badge-seasonal")
    return f'<span class="{css_class}">{trend_type}</span>'


def section_divider():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def pct(value) -> str:
    """Format a numeric proportion as a percentage."""
    try:
        if value is None or value != value:
            return "--"
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "--"


def dec(value, digits: int = 3) -> str:
    """Format a numeric value with a fixed number of decimals."""
    try:
        if value is None or value != value:
            return "--"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def style_plotly(fig):
    """Apply shared pastel axis styling without colliding with chart-specific layout."""
    fig.update_xaxes(gridcolor="#eadfd2", zerolinecolor="#d8cbbb")
    fig.update_yaxes(gridcolor="#eadfd2", zerolinecolor="#d8cbbb")
    return fig


def trend_signal_label(trend_type: str | None) -> str:
    """Map stored trend buckets to gentler UI language."""
    return {
        "persistent": "Stable",
        "trending": "Spike-prone",
        "seasonal": "Variable",
    }.get(trend_type or "", "Unlabelled")


def trend_signal_color(trend_type: str | None) -> str:
    return {
        "persistent": "#8bbf9f",
        "trending": "#d9aa63",
        "seasonal": "#8aa6c8",
    }.get(trend_type or "", "#b7afa5")


# ── Plotly theme ────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=13, color="#746a61"),
    margin=dict(l=40, r=20, t=50, b=40),
)

PALETTE = ["#7fb8b4", "#d9817a", "#d9aa63", "#a995c7", "#8bbf9f",
           "#8aa6c8", "#e7b7b0", "#b9aa88", "#91b6a4", "#c4a6b9",
           "#b6c8a6", "#d5b798", "#9ab7c6", "#cfaaa0", "#aeb2d5"]


# ════════════════════════════════════════════════════════════════════════
#  DASHBOARD PAGE
# ════════════════════════════════════════════════════════════════════════

def render_dashboard():
    page_hero(
        "r/jobs, decoded",
        "A bold read on one year of r/jobs posts and high-engagement comments, from activity pulses to topic pressure points.",
        "Dashboard",
        ["Apr 2025 to Mar 2026", "Posts + top comments", "Interactive corpus view"],
    )

    data = load_agg_cache()
    overview = data["overview"]

    # ── KPI row (6 cards)
    cols = st.columns(6)
    with cols[0]:
        kpi_card("Total Posts", f"{overview['total_posts']:,}")
    with cols[1]:
        kpi_card("Total Comments", f"{overview['total_comments']:,}")
    with cols[2]:
        kpi_card("Unique Authors", f"{overview['unique_post_authors'] + overview['unique_comment_authors']:,}")
    with cols[3]:
        kpi_card("Avg Post Length", f"{overview['avg_post_length']:,.0f}")
    with cols[4]:
        kpi_card("Avg Post Score", f"{overview['avg_post_score']:.1f}")
    with cols[5]:
        kpi_card("Avg Comment Score", f"{overview['avg_comment_score']:.1f}")

    # ── Corpus Insights card
    st.markdown(f"""
    <div class="insight-card">
        <strong>Corpus overview</strong> &mdash;
        The dataset contains <strong>{overview['total_posts']:,}</strong> cleaned posts and
        <strong>{overview['total_comments']:,}</strong> top comments scraped from r/jobs.
        For each qualifying post, the scraper collected up to the <strong>top 5 comments by score</strong>,
        so comments represent high-engagement reactions rather than exhaustive full threads.
        The data spans <strong>12 monthly buckets</strong> from April 2025 through March 2026.
        Posts average <strong>{overview['avg_post_length']:,.0f} characters</strong> in length.
    </div>
    """, unsafe_allow_html=True)

    # ── Monthly activity: posts and comments in one view
    st.subheader("Collection Activity")
    st.caption("Posts and collected top comments across the 12-month window")
    ppm = data["posts_per_month"]
    cpm = data["comments_per_month"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[r["month"] for r in ppm],
        y=[r["count"] for r in ppm],
        name="Posts",
        marker_color="#7fb8b4",
        opacity=0.86,
    ))
    fig.add_trace(go.Scatter(
        x=[r["month"] for r in cpm],
        y=[r["count"] for r in cpm],
        name="Top comments",
        yaxis="y2",
        line=dict(color="#d9817a", width=3),
        mode="lines+markers",
        marker=dict(size=7),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="",
        yaxis_title="Posts",
        yaxis2=dict(
            title="Comments",
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            zeroline=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=390,
    )
    st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # ── Leaderboards
    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.subheader("Top Flairs")
        st.caption("The most common user/moderator-assigned categories")
        flair_top = data["flair_distribution"][:12]
        fig = px.bar(
            flair_top,
            x="count",
            y="flair",
            orientation="h",
            color_discrete_sequence=["#a995c7"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"),
                          xaxis_title="Posts", yaxis_title="", height=420)
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} posts<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Engagement Mix")
        st.caption("High vs low engagement posts by scraper tier")
        tier = data["tier_breakdown"]
        fig = go.Figure(data=[go.Pie(
            labels=list(tier.keys()), values=list(tier.values()),
            marker_colors=["#7fb8b4", "#e7b7b0"], hole=0.58,
        )])
        fig.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=True)
        fig.update_traces(textposition="inside", textinfo="label+percent", textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)

    topics = load_topics_from_db()
    real_topics = [topic for topic in topics if topic["id"] != -1]
    if real_topics:
        st.subheader("Topic Leaderboard")
        st.caption("Largest discovered themes, with trend signal shown as a heuristic rather than a hard category")
        topic_rows = [
            {
                "Topic": f"T{topic['id']}: {topic['display_label']}",
                "Posts": topic["post_count"],
                "Share": f"{topic['share_pct']:.1f}%",
                "Signal": trend_signal_label(topic.get("trend_type")),
                "Keywords": ", ".join(topic["keywords"][:5]),
            }
            for topic in real_topics[:10]
        ]
        st.dataframe(topic_rows, use_container_width=True, hide_index=True)

    section_divider()

    # ── Score distributions
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Post Score Distribution")
        st.caption("Reddit karma distribution across all posts")
        sdp = data["score_dist_posts"]
        fig = px.bar(sdp, x="bucket", y="count", color_discrete_sequence=["#10b981"])
        fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Score Range", yaxis_title="Count")
        fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y:,} posts<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Comment Score Distribution")
        st.caption("Karma distribution across collected top comments")
        sdc = data["score_dist_comments"]
        fig = px.bar(sdc, x="bucket", y="count", color_discrete_sequence=["#f59e0b"])
        fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Score Range", yaxis_title="Count")
        fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y:,} comments<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # ── Average length over time
    st.subheader("Average Post & Comment Length Over Time")
    st.caption("How verbose are users across different months?")
    alm = data["avg_length_per_month"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r["month"] for r in alm], y=[r["avg_post_length"] for r in alm],
        name="Avg Post Length", line=dict(color="#8b5cf6", width=2.5), mode="lines+markers",
    ))
    fig.add_trace(go.Scatter(
        x=[r["month"] for r in alm], y=[r["avg_comment_length"] for r in alm],
        name="Avg Comment Length", line=dict(color="#06b6d4", width=2.5), mode="lines+markers",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="", yaxis_title="Characters")
    st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
#  TOPICS PAGE
# ════════════════════════════════════════════════════════════════════════

def render_topics():
    page_hero(
        "Topic atlas",
        "BERTopic clusters with cleaned labels, keyword fingerprints, corpus share, and flair gaps that show where the subreddit vocabulary is stretched thin.",
        "Topic Analysis",
        ["Cluster labels", "Keyword badges", "Flair gap signals"],
    )

    topics = load_topics_from_db()
    rep_docs = load_representative_docs()
    flair_analysis = load_topic_flair_analysis()

    if not topics:
        st.warning("No topic data found. Run `python precompute.py --only topics` first.")
        return

    real_topics = [t for t in topics if t["id"] != -1]
    outlier = next((t for t in topics if t["id"] == -1), None)
    flair_lookup = {item["topic_id"]: item for item in flair_analysis}

    # ── Summary KPIs
    cols = st.columns(4)
    with cols[0]:
        kpi_card("Topics Discovered", str(len(real_topics)))
    with cols[1]:
        total = sum(t["post_count"] for t in real_topics)
        kpi_card("Posts Classified", f"{total:,}")
    with cols[2]:
        if outlier:
            kpi_card("Outlier Posts", f"{outlier['post_count']:,}")
        else:
            kpi_card("Outlier Posts", "0")
    with cols[3]:
        avg_size = total // max(len(real_topics), 1)
        kpi_card("Avg Topic Size", f"{avg_size:,}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Topic share chart
    st.subheader("Topic Distribution")
    st.caption("Share of classified posts per discovered topic cluster")
    chart_topics = [
        {
            "topic": f"T{topic['id']}: {topic['display_label']}",
            "share_pct": topic["share_pct"],
        }
        for topic in real_topics
    ]
    fig = px.bar(chart_topics, x="share_pct", y="topic", orientation="h",
                 color_discrete_sequence=["#7fb8b4"])
    fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Share (%)", yaxis_title="",
                      showlegend=False, yaxis=dict(autorange="reversed"), height=520)
    st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # ── Topics vs Flairs
    st.subheader("Topics vs Post Flairs")
    st.caption(
        "Compares each discovered topic to the subreddit's existing flair system. "
        "Low dominant-flair share means the topic is spread across broad flairs and may need a more specific label."
    )

    if flair_analysis:
        import pandas as pd

        coverage_df = pd.DataFrame([
            {
                "Topic": f"T{row['topic_id']}: {row['display_label']}",
                "Dominant flair share": row["dominant_flair_pct"],
                "Flair fit": row["flair_fit"],
            }
            for row in flair_analysis
        ])

        fig = px.bar(
            coverage_df,
            x="Topic",
            y="Dominant flair share",
            color="Flair fit",
            color_discrete_map={"Strong": "#10b981", "Partial": "#f59e0b", "Gap": "#f43f5e"},
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="",
            yaxis_title="Top Flair Share (%)",
            xaxis_tickangle=-45,
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        missing_df = pd.DataFrame([
            {
                "Topic": f"T{row['topic_id']}: {row['display_label']}",
                "Top flair": f"{row['dominant_flair']} ({row['dominant_flair_pct']:.1f}%)",
                "Suggested missing flair": row["suggested_missing_flair"],
                "Why": row["gap_reason"],
            }
            for row in flair_analysis
            if row["suggested_missing_flair"]
        ])

        if not missing_df.empty:
            st.markdown("**Candidate flair gaps:**")
            st.dataframe(missing_df, use_container_width=True, hide_index=True)

    section_divider()

    # ── Topic overview and focused detail
    st.subheader("Topic Overview")
    st.caption("Scan topic size, fit, and keywords first; choose one topic below for representative posts")

    topic_table = []
    for t in real_topics:
        flair_info = flair_lookup.get(t["id"], {})
        top_flairs = flair_info.get("top_flairs", [])
        dominant_flair = top_flairs[0]["flair"] if top_flairs else ""
        topic_table.append({
            "Topic": f"T{t['id']}: {t['display_label']}",
            "Posts": t["post_count"],
            "Share": f"{t['share_pct']:.1f}%",
            "Signal": trend_signal_label(t.get("trend_type")),
            "Flair fit": flair_info.get("flair_fit", ""),
            "Dominant flair": dominant_flair,
            "Keywords": ", ".join(t["keywords"][:7]),
        })
    st.dataframe(topic_table, use_container_width=True, hide_index=True)

    selected_topic_label = st.selectbox(
        "Inspect topic",
        [f"T{t['id']}: {t['display_label']}" for t in real_topics],
        label_visibility="collapsed",
    )
    selected_id = int(selected_topic_label.split(":", 1)[0].replace("T", ""))
    t = next(topic for topic in real_topics if topic["id"] == selected_id)
    flair_info = flair_lookup.get(t["id"], {})
    top_flairs = flair_info.get("top_flairs", [])
    docs = rep_docs.get(str(t["id"]), [])

    kw_html = " ".join(f'<span class="topic-badge">{kw}</span>' for kw in t["keywords"][:8])
    st.markdown(f"""
    <div class="topic-grid-card">
        <h4>T{t['id']}: {escape(t['display_label'])} {trend_badge_html(t.get("trend_type"))}</h4>
        <div class="topic-stats">{t['post_count']:,} posts · {t['share_pct']:.1f}% of corpus · {trend_signal_label(t.get("trend_type"))} signal</div>
        {kw_html}
    </div>
    """, unsafe_allow_html=True)

    detail_col1, detail_col2 = st.columns([1, 1])
    with detail_col1:
        if top_flairs:
            st.markdown("**Top flairs inside this topic**")
            st.dataframe(
                [{"Flair": f["flair"], "Share": f"{f['pct']:.1f}%"} for f in top_flairs[:5]],
                use_container_width=True,
                hide_index=True,
            )
        if flair_info.get("suggested_missing_flair"):
            st.markdown(f"""
            <div class="soft-callout">
                <strong>Potential missing flair:</strong> {escape(flair_info['suggested_missing_flair'])}<br>
                {escape(flair_info['gap_reason'])}
            </div>
            """, unsafe_allow_html=True)
    with detail_col2:
        st.markdown("**Representative posts**")
        if docs:
            for j, doc in enumerate(docs[:3], 1):
                st.markdown(f"> **{j}.** {doc[:300]}{'...' if len(doc) > 300 else ''}")
        else:
            st.caption("No representative documents cached for this topic.")

    if outlier:
        with st.expander("Outlier / Uncategorised Posts", expanded=False):
            st.markdown(f"**{outlier['post_count']:,}** posts ({outlier['share_pct']:.1f}%) "
                        "could not be confidently assigned to any topic cluster. "
                        "This is expected with HDBSCAN — these are posts with unique or "
                        "mixed themes that don't fit neatly into a single cluster.")


# ════════════════════════════════════════════════════════════════════════
#  TRENDS PAGE
# ════════════════════════════════════════════════════════════════════════

def render_trends():
    page_hero(
        "What keeps coming back",
        "Separate long-running anxieties from sharper spikes by tracking monthly topic behavior across the full collection window.",
        "Trends",
        ["Persistent", "Trending", "Seasonal"],
    )

    trending_data = load_trending_cache()
    topics = load_topics_from_db()

    if not trending_data:
        st.warning("No trending data found. Run `python precompute.py --only trending` first.")
        return

    topic_labels = {t["id"]: t["display_label"] for t in topics}

    persistent = [c for c in trending_data if c["trend_type"] == "persistent"]
    trending = [c for c in trending_data if c["trend_type"] == "trending"]
    seasonal = [c for c in trending_data if c["trend_type"] == "seasonal"]

    # ── Summary KPIs
    cols = st.columns(3)
    with cols[0]:
        kpi_card("Stable Signals", str(len(persistent)))
    with cols[1]:
        kpi_card("Spike-Prone Signals", str(len(trending)))
    with cols[2]:
        kpi_card("Variable Signals", str(len(seasonal)))

    # ── Narrative summary
    persistent_names = [topic_labels.get(c["topic_id"], f"Topic {c['topic_id']}") for c in persistent[:3]]
    trending_names = [topic_labels.get(c["topic_id"], f"Topic {c['topic_id']}") for c in trending[:3]]

    narrative_parts = []
    if persistent:
        narrative_parts.append(
            f"<strong>{len(persistent)} topic{'s' if len(persistent) != 1 else ''}</strong> look stable by this heuristic, "
            f"including {', '.join(persistent_names)}."
        )
    if trending:
        narrative_parts.append(
            f"<strong>{len(trending)} topic{'s' if len(trending) != 1 else ''}</strong> look spike-prone, "
            f"such as {', '.join(trending_names)}."
        )
    if seasonal:
        narrative_parts.append(
            f"<strong>{len(seasonal)} topic{'s' if len(seasonal) != 1 else ''}</strong> sit in the middle as variable signals."
        )

    if narrative_parts:
        st.markdown(f'<div class="insight-card">{" ".join(narrative_parts)}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="soft-callout">
        <strong>How to read this page:</strong> the original pipeline stores buckets called persistent,
        trending, and seasonal. The interface below treats them as heuristic signals, because most r/jobs
        themes appear in most months. Use the metrics and the visible month-by-month curves rather than the
        label alone.
    </div>
    """, unsafe_allow_html=True)

    # ── Monthly small multiples for all topics
    st.subheader("Topic Trajectories")
    st.caption("All topic curves are visible here, sorted by total posts; no clicking required")

    if trending_data:
        all_months = sorted(trending_data[0]["monthly_counts"].keys())

        trend_colors = {
            "persistent": "#8bbf9f",
            "trending": "#d9aa63",
            "seasonal": "#8aa6c8",
        }

        cols = st.columns(3)
        for idx, c in enumerate(sorted(trending_data, key=lambda x: x["total_posts"], reverse=True)):
            label = topic_labels.get(c["topic_id"], f"Topic {c['topic_id']}")
            counts = [c["monthly_counts"].get(m, 0) for m in all_months]
            color = trend_colors.get(c["trend_type"], "#b7afa5")
            with cols[idx % 3]:
                st.markdown(
                    f"**T{c['topic_id']}: {label}**  \n"
                    f"{trend_signal_label(c['trend_type'])} · total {c['total_posts']:,} · peak {c.get('peak_month', 'n/a')}"
                )
                fig = go.Figure(go.Scatter(
                    x=all_months,
                    y=counts,
                    line=dict(color=color, width=2.5),
                    mode="lines+markers",
                    marker=dict(size=5),
                    fill="tozeroy",
                    fillcolor=color,
                    opacity=0.45,
                ))
                fig.update_layout({
                    **PLOTLY_LAYOUT,
                    "height": 180,
                    "margin": dict(l=20, r=12, t=8, b=30),
                    "showlegend": False,
                    "xaxis_title": "",
                    "yaxis_title": "",
                })
                fig.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig, use_container_width=True)

    section_divider()

    st.subheader("Signal Metrics")
    st.caption("The numeric features behind the heuristic buckets")
    metric_rows = []
    for c in sorted(trending_data, key=lambda x: x["total_posts"], reverse=True):
        label = topic_labels.get(c["topic_id"], f"Topic {c['topic_id']}")
        metric_rows.append({
            "Topic": f"T{c['topic_id']}: {label}",
            "Signal": trend_signal_label(c["trend_type"]),
            "Presence": f"{c['presence']:.0%}",
            "CV": f"{c['cv']:.2f}",
            "Mean/month": f"{c['mean_monthly']:.0f}",
            "Peak month": c.get("peak_month", ""),
            "Peak count": c.get("peak_count", ""),
            "Total posts": f"{c['total_posts']:,}",
            "Spike detected": "Yes" if c.get("has_spike") else "No",
        })
    st.dataframe(metric_rows, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════
#  STANCE ANALYSIS PAGE
# ════════════════════════════════════════════════════════════════════════

def render_stance():
    page_hero(
        "The argument map",
        "Comment stance relative to each topic's dominant discussion frame, with representative replies that show how the community pushes, agrees, or drifts.",
        "Stance Analysis",
        ["For the frame", "Opposing", "Neutral / unclear"],
    )

    stance_data = load_stance_cache()
    topics = load_topics_from_db()
    topic_labels = {str(t["id"]): t["display_label"] for t in topics}

    if not stance_data:
        st.warning("No stance data found. Run `python precompute.py --only stance` first.")
        return

    # ── Summary highlights
    most_consensual = None
    most_divisive = None
    max_agreement = 0
    min_agreement = 100

    for tid, s in stance_data.items():
        for_pct = s.get("for", s.get("support", {})).get("pct", 0)
        label = topic_labels.get(str(tid), f"Topic {tid}")
        if for_pct > max_agreement:
            max_agreement = for_pct
            most_consensual = label
        if for_pct < min_agreement:
            min_agreement = for_pct
            most_divisive = label

    highlight_parts = []
    if most_consensual:
        highlight_parts.append(f"Most aligned topic: <span class='stat-highlight'>{most_consensual} ({max_agreement:.0f}%)</span>")
    if most_divisive:
        highlight_parts.append(f"Most divided topic: <span class='stat-highlight'>{most_divisive} ({min_agreement:.0f}%)</span>")

    if highlight_parts:
        st.markdown(f'<div class="insight-card">{" &nbsp;&middot;&nbsp; ".join(highlight_parts)}</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="soft-callout">
        <strong>Why does agreement look so low?</strong>
        The current cached stance run used zero-shot labels around whether a comment supported or opposed the parent post.
        In r/jobs, top comments are often corrective advice, skepticism, or “here is what you should do,” so the model can
        over-count practical pushback as opposition.
    </div>
    """, unsafe_allow_html=True)

    # ── Overview chart
    st.subheader("Stance Distribution Across Topics")
    st.caption("Horizontal comparison sorted by alignment with the extracted discussion frame")

    import pandas as pd
    chart_data = []
    stance_lookup = {}
    for tid, s in stance_data.items():
        label = topic_labels.get(str(tid), s.get("topic_label", f"Topic {tid}"))
        for_bucket = s.get("for", s["support"])
        opposing_bucket = s.get("opposing", s["oppose"])
        neutral_bucket = s.get("neutral", {"pct": 0.0, "count": 0, "user_count": 0, "top_arguments": []})
        row = {
            "topic_id": str(tid),
            "topic": f"T{tid}: {label}",
            "For dominant frame": for_bucket["pct"],
            "Opposing dominant frame": opposing_bucket["pct"],
            "Neutral / unclear": neutral_bucket["pct"],
            "Comments analysed": s.get("total_comments_analysed", 0),
        }
        chart_data.append(row)
        stance_lookup[str(tid)] = {
            "label": label,
            "dominant": s.get("dominant_position", "N/A"),
            "frame": s.get("reference_frame", s.get("dominant_position", "N/A")),
            "total": s.get("total_comments_analysed", 0),
            "for": for_bucket,
            "opposing": opposing_bucket,
            "neutral": neutral_bucket,
        }

    df = pd.DataFrame(chart_data).sort_values("For dominant frame", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="For frame", y=df["topic"], x=df["For dominant frame"],
        orientation="h", marker_color="#8bbf9f",
    ))
    fig.add_trace(go.Bar(
        name="Opposing", y=df["topic"], x=df["Opposing dominant frame"],
        orientation="h", marker_color="#d9817a",
    ))
    fig.add_trace(go.Bar(
        name="Neutral / unclear", y=df["topic"], x=df["Neutral / unclear"],
        orientation="h", marker_color="#b7afa5",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        xaxis_title="Share of analysed comments (%)",
        yaxis_title="",
        height=max(430, 34 * len(df)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    most_aligned = max(chart_data, key=lambda row: row["For dominant frame"])
    most_opposed = max(chart_data, key=lambda row: row["Opposing dominant frame"])
    most_unclear = max(chart_data, key=lambda row: row["Neutral / unclear"])
    st.markdown(f"""
    <div class="comparison-grid">
        <div class="comparison-card">
            <div class="mini-label">Most aligned</div>
            <h4>{escape(most_aligned["topic"])}</h4>
            <div class="stance-strip"><div class="stance-for" style="width:{most_aligned["For dominant frame"]}%"></div></div>
            <p>{most_aligned["For dominant frame"]:.0f}% of analysed comments align with the extracted frame.</p>
        </div>
        <div class="comparison-card">
            <div class="mini-label">Most contested</div>
            <h4>{escape(most_opposed["topic"])}</h4>
            <div class="stance-strip"><div class="stance-against" style="width:{most_opposed["Opposing dominant frame"]}%"></div></div>
            <p>{most_opposed["Opposing dominant frame"]:.0f}% push against the extracted frame.</p>
        </div>
        <div class="comparison-card">
            <div class="mini-label">Most ambiguous</div>
            <h4>{escape(most_unclear["topic"])}</h4>
            <div class="stance-strip"><div class="stance-mixed" style="width:{most_unclear["Neutral / unclear"]}%"></div></div>
            <p>{most_unclear["Neutral / unclear"]:.0f}% are neutral, mixed, or low-confidence.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_divider()

    # ── Focused topic inspector
    st.subheader("Inspect One Topic")
    selected_topic = st.selectbox(
        "Choose topic",
        [row["topic"] for row in sorted(chart_data, key=lambda item: item["topic"])],
        label_visibility="collapsed",
    )
    selected_id = selected_topic.split(":", 1)[0].replace("T", "")
    s = stance_lookup[selected_id]

    for_bucket = s["for"]
    opposing_bucket = s["opposing"]
    neutral_bucket = s["neutral"]
    total = s["total"]

    st.markdown(f"**Discussion frame:** {s['frame'][:420]}")
    st.caption(f"Comments analysed: {total:,}. Interpretation: alignment is relative to the extracted frame, not a universal pro/con vote.")

    stat_cols = st.columns(3)
    with stat_cols[0]:
        st.metric("For frame", f"{for_bucket['pct']:.0f}%", f"{for_bucket['count']:,} comments")
    with stat_cols[1]:
        st.metric("Opposing", f"{opposing_bucket['pct']:.0f}%", f"{opposing_bucket['count']:,} comments")
    with stat_cols[2]:
        st.metric("Neutral / unclear", f"{neutral_bucket['pct']:.0f}%", f"{neutral_bucket['count']:,} comments")

    example_cols = st.columns(3)
    buckets = [
        ("Representative aligned comments", "stance-support", for_bucket),
        ("Representative opposing comments", "stance-oppose", opposing_bucket),
        ("Representative neutral/mixed comments", "stance-neutral", neutral_bucket),
    ]
    for col, (title, css_class, bucket) in zip(example_cols, buckets):
        with col:
            st.markdown(f"**{title}**")
            if bucket["top_arguments"]:
                for i, arg in enumerate(bucket["top_arguments"][:3], 1):
                    st.markdown(
                        f'<div class="{css_class}">'
                        f'<strong>{i}.</strong> {escape(arg["body"][:300])}'
                        f'{"..." if len(arg["body"]) > 300 else ""}'
                        f'<br><em style="color:#746a61">score: {arg["score"]}</em></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No representative examples in this bucket.")


# ════════════════════════════════════════════════════════════════════════
#  QA PAGE (RAG)
# ════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_rag_retriever():
    """Load the persistent vector store and graph-backed retriever."""
    from rag.retriever import HybridRetriever
    return HybridRetriever()


EXAMPLE_QUESTIONS = [
    "How large is the collected r/jobs corpus?",
    "What do users think about salary negotiation?",
    "What advice do users give about difficult bosses?",
    "Which topics are trending in the dataset?",
    "What do users say about AI and tech in hiring?",
    "How do users talk about job searching?",
]


def render_qa():
    page_hero(
        "Ask the corpus",
        "A semantic knowledge-graph hybrid RAG layer over posts, comments, topics, trends, and stance summaries.",
        "Question Answering",
        ["ChromaDB retrieval", "Graph expansion", "Cited answers"],
    )

    import json

    from rag.config import CHROMA_DIR, GRAPH_PATH, MANIFEST_PATH
    from rag.llms import generate_answer
    from rag.retriever import retrieval_only_answer

    if not CHROMA_DIR.exists() or not GRAPH_PATH.exists():
        st.warning(
            "The RAG index has not been built yet. Run `python build_rag_index.py` "
            "after installing dependencies, then reload this page."
        )
        st.code("uv sync\npython build_rag_index.py", language="bash")
        return

    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)
        if manifest.get("limited"):
            st.warning(
                f"This is a partial smoke-test index with {manifest.get('vectors', 0):,} vectors. "
                "Run `python build_rag_index.py` without `--limit` before final evaluation or demo."
            )
        else:
            st.caption(
                f"Index loaded: {manifest.get('vectors', 0):,} vectors, "
                f"{manifest.get('graph_nodes', 0):,} graph nodes."
            )

    # ── Pipeline explanation
    st.markdown("""
    <div class="insight-card">
        <strong>How it works</strong> &mdash;
        Your question is classified by type (factual, opinion, trend, general),
        then matched against the vector index (ChromaDB) and expanded through
        the semantic knowledge graph (NetworkX). Results are merged, re-ranked
        by a hybrid score combining similarity, Reddit karma, and graph proximity,
        then passed with evidence to the LLM for a grounded, cited answer.
    </div>
    """, unsafe_allow_html=True)

    # ── Example question chips
    st.markdown("**Try a question:**")
    chip_html = " ".join(f'<span class="example-chip">{q}</span>' for q in EXAMPLE_QUESTIONS)
    st.markdown(chip_html, unsafe_allow_html=True)

    st.markdown("")

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Ask a question about r/jobs",
            placeholder="What do users think about salary negotiation?",
        )
    with col2:
        provider = st.selectbox(
            "Model",
            ["retrieval_only", "groq", "gemini"],
            format_func=lambda value: {
                "retrieval_only": "Retrieval only",
                "groq": "Groq (Llama 3.3)",
                "gemini": "Gemini 2.5 Flash",
            }[value],
        )

    final_k = 10
    show_debug = st.toggle("Show retrieval debug", value=False)

    if not st.button("Answer", type="primary") or not question.strip():
        return

    try:
        retriever = load_rag_retriever()
        with st.spinner("Retrieving Reddit evidence..."):
            retrieval = retriever.retrieve(question.strip(), final_k=final_k)

        with st.spinner("Generating answer..."):
            try:
                if provider == "retrieval_only":
                    answer = retrieval_only_answer(question, retrieval["evidence"])
                else:
                    answer = generate_answer(provider, question, retrieval["context"], retrieval["evidence"])
            except Exception as exc:
                answer = (
                    f"{provider} could not be called: {exc}\n\n"
                    + retrieval_only_answer(question, retrieval["evidence"])
                )

        # ── Answer panel
        st.markdown(f"""
        <div class="answer-panel">
            {answer}
        </div>
        """, unsafe_allow_html=True)

        section_divider()

        # ── Evidence
        st.subheader("Retrieved Evidence")
        st.caption(f"Query type: {retrieval['query_type']} · {len(retrieval['evidence'])} evidence snippets")

        for item in retrieval["evidence"]:
            metadata = item.get("metadata", {})
            source_type = metadata.get("source_type", "unknown")
            rerank = item.get("rerank_score", 0)

            # Type-based accent colour
            accent = {
                "post": "#8b5cf6",
                "comment": "#06b6d4",
                "topic_summary": "#10b981",
                "sql_fact": "#f59e0b",
            }.get(source_type, "#64748b")

            title = (
                f"{item['id']} · {source_type} · "
                f"score {rerank:.2f}"
            )
            with st.expander(title, expanded=False):
                st.caption(
                    f"Topic: {metadata.get('topic_label', '')} | "
                    f"Flair: {metadata.get('flair', '')} | "
                    f"Month: {metadata.get('month', '')} | "
                    f"Reddit score: {metadata.get('score', '')}"
                )
                st.write(item.get("text", ""))

        if show_debug:
            section_divider()
            st.subheader("Retrieval Debug")
            st.json({
                "query_type": retrieval["query_type"],
                "evidence_ids": [item["id"] for item in retrieval["evidence"]],
            })
    except Exception as exc:
        st.error(f"Could not run the QA pipeline: {exc}")
        st.code("uv sync\npython build_rag_index.py", language="bash")


# ════════════════════════════════════════════════════════════════════════
#  EVALUATION RESULTS PAGE
# ════════════════════════════════════════════════════════════════════════

def render_evaluation():
    page_hero(
        "Evaluation scoreboard",
        "English RAG, Hindi cross-lingual QA, and retrieval diagnostics from the report pipeline, loaded directly from the evaluation artifacts.",
        "RAG Evaluation",
        ["English QA", "Hindi QA", "Retrieval diagnostics"],
    )

    english = load_csv_if_exists(EVALUATION_DIR / "results" / "metrics_summary.csv")
    english = without_retrieval_only(english)
    hindi = add_report_hindi_gemini(load_csv_if_exists(EVALUATION_DIR / "hindi_results" / "metrics_summary.csv"))
    diagnostics = load_csv_if_exists(EVALUATION_DIR / "diagnostics" / "metrics_summary.csv")
    comment_probe = load_csv_if_exists(EVALUATION_DIR / "diagnostics" / "comment_probe_summary.csv")
    tag_breakdown = without_retrieval_only(load_csv_if_exists(EVALUATION_DIR / "hindi_results" / "tag_breakdown.csv"))

    if english.empty and hindi.empty and diagnostics.empty:
        st.warning("No evaluation artifacts found. Run the evaluation scripts first.")
        st.code(
            "uv run python evaluate_rag.py --models groq gemini\n"
            "uv run python evaluate_hindi_qa.py --models groq gemini\n"
            "uv run python evaluate_rag_diagnostics.py --model retrieval_only",
            language="bash",
        )
        return

    if not english.empty:
        st.subheader("English RAG Results")
        st.caption("Baseline QA over English questions with factual, opinion, trend, and adversarial cases")

        display_rows = []
        english_chart_rows = []
        for _, row in english.iterrows():
            model = str(row["model"])
            bertscore = row.get("bertscore_f1")
            if bertscore != bertscore:
                bertscore = REPORT_ENGLISH_BERTSCORE.get(model.lower())
            english_chart_rows.extend([
                {"Model": model_label(model), "Metric": "ROUGE-L", "Score": row.get("rouge_l")},
                {"Model": model_label(model), "Metric": "BERTScore F1", "Score": bertscore},
                {"Model": model_label(model), "Metric": "Citation presence", "Score": row.get("citation_presence_rate")},
                {"Model": model_label(model), "Metric": "Citation validity", "Score": row.get("mean_citation_validity")},
                {"Model": model_label(model), "Metric": "Refusal accuracy", "Score": row.get("adversarial_refusal_accuracy")},
            ])
            display_rows.append({
                "Model": model_label(model),
                "ROUGE-L": dec(row.get("rouge_l")),
                "BERTScore F1": dec(bertscore),
                "Citation presence": pct(row.get("citation_presence_rate")),
                "Citation validity": pct(row.get("mean_citation_validity")),
                "Refusal accuracy": pct(row.get("adversarial_refusal_accuracy")),
            })
        st.dataframe(display_rows, use_container_width=True, hide_index=True)

        fig = px.bar(
            english_chart_rows,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            color_discrete_sequence=["#7fb8b4", "#d9817a"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Score", xaxis_title="", height=390)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        english_best_rouge = english.sort_values("rouge_l", ascending=False).iloc[0]
        english_best_citation = english.sort_values("citation_presence_rate", ascending=False).iloc[0]
        st.markdown(f"""
        <div class="soft-callout">
            <strong>Inference:</strong> {escape(model_label(str(english_best_rouge["model"])))} has the strongest answer-overlap
            score on the English benchmark, while {escape(model_label(str(english_best_citation["model"])))} is the most consistent
            at including citations. Treat ROUGE/BERTScore as answer-similarity metrics and citation validity as the grounding check.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Sample Responses vs Ground Truth"):
            st.markdown("**Question:** How large is the collected r/jobs corpus?")
            st.markdown("**Ground Truth:** The cleaned corpus contains about 24,932 posts and 93,501 collected top comments from r/jobs, covering April 2025 through March 2026.")
            st.markdown("**Groq:** The collected r/jobs corpus contains 24,932 posts and 93,501 collected top comments [sql:corpus_counts]. This information provides a clear count of the corpus size, making other evidence, such as posts about job searching or specific job topics, irrelevant to answering the question about the corpus size.")
            st.markdown("**Gemini:** The collected r/jobs corpus contains 24,932 posts and 93,501 collected top comments [sql:corpus_counts].")
            st.markdown("---")
            st.markdown("**Question:** What do users say about difficult bosses or manager conflict?")
            st.markdown("**Ground Truth:** Users frequently advise documenting incidents, protecting oneself, avoiding unnecessary confrontation, and looking for another role when management behavior seems retaliatory, exploitative, or unstable.")
            st.markdown("**Groq:** Users in the r/jobs subreddit tend to express negative experiences with difficult bosses or manager conflict, describing behaviors such as condescending and nitpicky behavior [post:1okjk5w], micromanaging [comment:mlhgovt], and creating conflict for no reason [comment:mlhgovt].")
            st.markdown("**Gemini:** Users on r/jobs describe difficult bosses as those who micromanage, create conflict, and retain poor employees due to personal relationships [comment:mlhgovt, comment:n8ftu4c]. They can be condescending, nitpicky, sarcastic, and passive-aggressive, often dismissing their behavior as \"not personal\" [post:1okjk5w].")
            st.markdown("---")
            st.markdown("**Question:** What do users seem to think about LinkedIn recruiter messages and job scams?")
            st.markdown("**Ground Truth:** Users often treat suspicious LinkedIn recruiter messages, vague job listings, unusual interview steps, and too-good-to-be-true opportunities with caution. The community tends to advise checking legitimacy before sharing personal information or continuing with a process.")
            st.markdown("**Groq:** Users on the subreddit tend to be skeptical of LinkedIn recruiter messages and job postings, with many considering them potential scams [comment:ndms9ju, comment:ndmq17e]. Some users have reported receiving scammy phone calls or messages after applying for remote jobs on LinkedIn.")
            st.markdown("**Gemini:** Users on r/jobs tend to view a significant portion of remote job listings and recruiter messages on LinkedIn as scams [comment:ndms9ju, comment:ndmq17e]. Many users report receiving scammy phone calls or messages via WhatsApp after applying to jobs on LinkedIn.")
            st.markdown("---")
            st.markdown("**Question:** How do users talk about job searching and applications?")
            st.markdown("**Ground Truth:** Many users describe job searching as frustrating, slow, and uncertain, especially when applications receive no response. Advice tends to focus on improving resumes, applying strategically, networking, and managing expectations in a difficult market.")
            st.markdown("**Groq:** Users on the r/jobs subreddit tend to express frustration and skepticism about the job searching and application process, particularly with the increasing use of AI in applications [post:1qc9nof:chunk:0, comment:nzgku8a].")
            st.markdown("**Gemini:** Users on r/jobs express that job searching and applications can feel dehumanizing, impersonal, and like \"talking to a wall\" [post:1jqv3ek:chunk:0].")

        best_rouge = english.sort_values("rouge_l", ascending=False).iloc[0]
        best_citation = english.sort_values("citation_presence_rate", ascending=False).iloc[0]
        st.markdown(f"""
        <div class="note-grid">
            <div class="note-card">
                <div class="mini-label">Highest overlap</div>
                <h4>{escape(model_label(str(best_rouge['model'])))}</h4>
                <p>Best ROUGE-L in the current artifacts: <strong>{dec(best_rouge['rouge_l'])}</strong>.
                This rewards similarity to the hand-written reference answers.</p>
            </div>
            <div class="note-card">
                <div class="mini-label">Most citation-disciplined</div>
                <h4>{escape(model_label(str(best_citation['model'])))}</h4>
                <p>Highest citation presence: <strong>{pct(best_citation['citation_presence_rate'])}</strong>.
                Citation validity matters because answers must stay tied to retrieved evidence IDs.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    section_divider()

    if not hindi.empty:
        st.subheader("Hindi Cross-Lingual QA Results")
        st.caption("Hindi questions over an English r/jobs corpus using English retrieval bridge queries where needed")

        display_rows = []
        hindi_chart_rows = []
        for _, row in hindi.iterrows():
            model = str(row["model"])
            hindi_chart_rows.extend([
                {"Model": model_label(model), "Metric": "chrF", "Score": row.get("chrf")},
                {"Model": model_label(model), "Metric": "BERTScore F1", "Score": row.get("bertscore_f1")},
                {"Model": model_label(model), "Metric": "Citation presence", "Score": row.get("citation_presence_rate")},
                {"Model": model_label(model), "Metric": "Citation validity", "Score": row.get("mean_citation_validity")},
                {"Model": model_label(model), "Metric": "Refusal accuracy", "Score": row.get("adversarial_refusal_accuracy")},
            ])
            display_rows.append({
                "Model": model_label(model),
                "chrF": dec(row.get("chrf")),
                "BERTScore F1": dec(row.get("bertscore_f1")),
                "Citation presence": pct(row.get("citation_presence_rate")),
                "Citation validity": pct(row.get("mean_citation_validity")),
                "Refusal accuracy": pct(row.get("adversarial_refusal_accuracy")),
                "Manual fluency": dec(row.get("fluency_mean_manual"), 2) if row.get("fluency_reviewed", 0) else "pending",
                "Manual adequacy": dec(row.get("adequacy_mean_manual"), 2) if row.get("adequacy_reviewed", 0) else "pending",
            })
        st.dataframe(display_rows, use_container_width=True, hide_index=True)

        fig = px.bar(
            hindi_chart_rows,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            color_discrete_sequence=["#7fb8b4", "#d9817a"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Score", xaxis_title="", height=390)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        hindi_best_chrf = hindi.sort_values("chrf", ascending=False).iloc[0]
        hindi_best_refusal = hindi.sort_values("adversarial_refusal_accuracy", ascending=False).iloc[0]
        st.markdown(f"""
        <div class="soft-callout">
            <strong>Inference:</strong> {escape(model_label(str(hindi_best_chrf["model"])))} leads on chrF for Hindi answer overlap,
            while {escape(model_label(str(hindi_best_refusal["model"])))} has the strongest adversarial refusal score.
            The cross-lingual task is harder because Hindi questions are routed over an English corpus and often need an English retrieval bridge.
        </div>
        """, unsafe_allow_html=True)

        if not tag_breakdown.empty:
            st.markdown("**Hindi edge-case breakdown**")
            edge_chart = tag_breakdown.copy()
            if "model" in edge_chart:
                edge_chart["Model"] = edge_chart["model"].map(lambda value: model_label(str(value)))
            edge_chart["Tag"] = edge_chart["tag"].str.replace("_", " ").str.title()

            fig = px.bar(
                edge_chart,
                x="Tag",
                y="chrf",
                color="Model",
                barmode="group",
                color_discrete_sequence=["#7fb8b4", "#d9817a", "#a995c7"],
            )
            fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="chrF", xaxis_title="", height=390)
            fig.update_yaxes(range=[0, max(0.35, float(edge_chart["chrf"].max()) + 0.05)])
            st.plotly_chart(fig, use_container_width=True)

            fig = px.bar(
                edge_chart,
                x="Tag",
                y="citation_presence_rate",
                color="Model",
                barmode="group",
                color_discrete_sequence=["#7fb8b4", "#d9817a", "#a995c7"],
            )
            fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Citation presence", xaxis_title="", height=390)
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            tag_summary = edge_chart.groupby("Tag", as_index=False)["chrf"].mean()
            strongest_tag = tag_summary.sort_values("chrf", ascending=False).iloc[0]
            weakest_tag = tag_summary.sort_values("chrf", ascending=True).iloc[0]
            st.markdown(f"""
            <div class="soft-callout">
                <strong>Edge-case inference:</strong> {escape(str(strongest_tag["Tag"]))} has the strongest average chrF in the current
                artifact, while {escape(str(weakest_tag["Tag"]))} is weakest. This matches the expected failure mode:
                multi-hop and highly transformed inputs are harder than code-mixed or romanized prompts that preserve English anchors.
            </div>
            """, unsafe_allow_html=True)

            tag_display = tag_breakdown.copy()
            if "model" in tag_display:
                tag_display["model"] = tag_display["model"].map(lambda value: model_label(str(value)))
            for col in ["chrf", "bertscore_f1"]:
                if col in tag_display:
                    tag_display[col] = tag_display[col].map(lambda value: dec(value))
            for col in ["citation_presence_rate", "mean_citation_validity", "adversarial_refusal_accuracy"]:
                if col in tag_display:
                    tag_display[col] = tag_display[col].map(lambda value: pct(value))
            st.dataframe(tag_display, use_container_width=True, hide_index=True)

        with st.expander("Sample Responses vs Ground Truth"):
            st.markdown("**Question:** इकट्ठा किया गया r/jobs कॉर्पस कितना बड़ा है?")
            st.markdown("**Ground Truth:** साफ़ किए गए कॉर्पस में लगभग 24,932 पोस्ट और 93,501 इकट्ठा किए गए शीर्ष टिप्पणियाँ हैं। यह डेटा अप्रैल 2025 से मार्च 2026 तक फैला है।")
            st.markdown("**Groq:** इकट्ठा किया गया r/jobs कॉर्पस का आकार 24,932 पोस्ट और 93,501 एकत्रित शीर्ष टिप्पणियों का है [sql:corpus_counts]। यह जानकारी सीधे कॉर्पस की गणना से प्राप्त की गई है...")
            st.markdown("**Gemini:** साफ़ किए गए r/jobs कॉर्पस में कुल 24,932 पोस्ट और 93,501 शीर्ष टिप्पणियाँ शामिल हैं [sql:corpus_counts]। यह डेटाबेस समुदाय की चर्चाओं का एक व्यापक संग्रह प्रदान करता है।")
            st.markdown("---")
            st.markdown("**Question:** खोजे गए topic model में सबसे बड़े broad topics कौन-से हैं?")
            st.markdown("**Ground Truth:** सबसे बड़े topics में manager conflict और firing, entry-level job search और career starts, quitting और burnout, applications और resumes, और interview follow-ups या offer delays शामिल हैं।")
            st.markdown("**Groq:** खोजे गए टॉपिक मॉडल में सबसे बड़े ब्रॉड टॉपिक्स एआई, टेक, यूजिंग, टूल्स, और यूज़ से संबंधित हैं [post:1kxgv7x], [post:1p1g9ja]। इन टॉपिक्स में एआई के बढ़ते प्रभाव और इसके कारण होने वाले रोजगार परिवर्तनों पर चर्चा की गई है।")
            st.markdown("**Gemini:** सबसे बड़े विषयों में एआई (AI), तकनीक (Tech), टूल्स (Tools) का उपयोग शामिल है [post:1kxgv7x, post:1p1g9ja]। इसके अतिरिक्त, कुछ विषय 'Outlier / Uncategorised' के रूप में वर्गीकृत किए गए हैं।")
            st.markdown("---")
            st.markdown("**Question (Adversarial):** Antarctica mein quantum computing careers ke baare mein r/jobs users kya bolte hain?")
            st.markdown("**Ground Truth:** कॉर्पस में Antarctica में quantum computing careers के बारे में उत्तर देने के लिए पर्याप्त साक्ष्य नहीं हैं।")
            st.markdown("**Groq:** Antarctica mein quantum computing careers ke baare mein r/jobs users ki raay ko lekar koi vishesh jaankari uplabdh nahin hai. [Evidence 1] se [Evidence 10] tak, kisi bhi evidence mein Antarctica ya quantum computing careers ke baare mein charcha nahin ki gayi hai. Isliye, hum yeh keh sakte hain ki corpus mein is vishay par paryapt pramaan nahin hai.")
            st.markdown("**Gemini:** दिए गए कॉर्पस में अंटार्कटिका में क्वांटम कंप्यूटिंग करियर के बारे में कोई जानकारी नहीं है। r/jobs पर इस विषय में चर्चा नहीं की गई है।")
            st.markdown("---")
            st.markdown("**Question (Adversarial/Code Mix):** कौन-सी exact company ko r/jobs users sabse best employer bolte hain?")
            st.markdown("**Ground Truth:** कॉर्पस में यह तय करने के लिए पर्याप्त साक्ष्य नहीं हैं कि कौन-सी एक exact company सबसे ज़्यादा recommended employer है।")
            st.markdown("**Groq:** र/jobs उपयोगकर्ता किस कंपनी को सबसे अच्छा नियोक्ता मानते हैं, इसके बारे में स्पष्ट जानकारी नहीं है। हालांकि, कुछ उपयोगकर्ताओं ने कुछ कंपनियों का उल्लेख किया है, जैसे कि बर्कशायर हैथवे [comment:ns66t23] और रेडिट [comment:ns5wu0w]। लेकिन यह जानकारी पर्याप्त नहीं है और यह निष्कर्ष निकालने के लिए पर्याप्त नहीं है कि कौन सी कंपनी सबसे अच्छी है।")
            st.markdown("**Gemini:** r/jobs कॉर्पस में कोई एक 'सबसे अच्छी' कंपनी के रूप में स्पष्ट रूप से अनुशंसित नहीं है। अलग-अलग उपयोगकर्ताओं के अनुभव अलग-अलग कंपनियों के साथ अलग-अलग होते हैं।")



# ════════════════════════════════════════════════════════════════════════
#  BIAS & ETHICS PAGE
# ════════════════════════════════════════════════════════════════════════

def render_bias_ethics():
    page_hero(
        "Bias and ethics notes",
        "Assignment notes on bias detection probes and the ethical dimensions of collecting, storing, and querying Reddit data.",
        "Responsible NLP",
        ["Bias probes", "Corpus evidence", "Right to be forgotten"],
    )

    summary, evidence = load_bias_probe_summary()
    overview = load_agg_cache().get("overview", {})
    total_posts = overview.get("total_posts", 0)

    st.subheader("Note on Bias Detection")
    st.markdown("""
    <div class="insight-card">
        <strong>Probe design.</strong>
        I used targeted lexical probes over the local corpus for identity-sensitive employment topics:
        gender/pregnancy/parenthood, race/nationality/visa status, age, and disability/health.
        These probes are not a fairness classifier; they are an audit surface for finding places where
        Reddit narratives may contain demographic assumptions, discrimination claims, or one-sided
        community generalisations that an LLM could amplify.
    </div>
    """, unsafe_allow_html=True)

    if summary:
        cols = st.columns(4)
        for idx, row in enumerate(summary[:4]):
            with cols[idx]:
                kpi_card(row["Probe"].split(",")[0], f"{row['Matching posts']:,}")

        st.caption(f"Counts are keyword-probe matches over {total_posts:,} cleaned posts; a match is evidence for audit attention, not proof of bias.")
        probe_table = []
        for row in summary:
            probe_table.append({
                "Probe": row["Probe"],
                "Probe question": row["Probe question"],
                "Matching posts": row["Matching posts"],
                "Share of corpus": pct(row["Share of corpus"]),
            })
        st.dataframe(probe_table, use_container_width=True, hide_index=True)

        with st.expander("Anonymized corpus evidence snippets", expanded=False):
            st.caption("Usernames are intentionally omitted. Snippets are truncated and should be treated as sensitive examples, not public case studies.")
            st.dataframe(evidence, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="note-grid">
        <div class="note-card probe-card">
            <div class="mini-label">Finding 1</div>
            <h4>Bias exists in the data as lived-report discourse.</h4>
            <p>The corpus contains many posts where users explicitly connect job outcomes to identity-sensitive
            categories such as race, visa status, age, pregnancy, disability, or mental health. These are mostly
            self-reports and advice threads, so the system should describe them as allegations and perceptions,
            not verified facts about employers or groups.</p>
        </div>
        <div class="note-card probe-card">
            <div class="mini-label">Finding 2</div>
            <h4>The LLM can smooth away bias signals.</h4>
            <p>A grounded answer may replace sharper Reddit language with neutral summaries. That helps avoid
            repeating harmful stereotypes, but it can also smudge the evidence: discrimination claims become
            generic “job search frustration” unless the prompt asks for identity-sensitive patterns and citations.</p>
        </div>
        <div class="note-card probe-card">
            <div class="mini-label">Finding 3</div>
            <h4>Reddit demographics can bias the answer frame.</h4>
            <p>The source is not a representative labor-market sample. It overrepresents people who post on Reddit,
            who are willing to disclose workplace conflict, and whose posts/comments receive engagement. The top-comment
            collection strategy further privileges community-approved reactions over quieter dissent.</p>
        </div>
        <div class="note-card probe-card">
            <div class="mini-label">Mitigation</div>
            <h4>Answers should carry provenance and uncertainty.</h4>
            <p>The safest behavior is to cite evidence IDs, use language like “in this corpus” and “some users report,”
            preserve the top-comments-only caveat, and refuse demographic generalisations that are not directly supported
            by retrieved evidence.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Bias probe conclusion.** The model is capable of surfacing bias-related patterns when the retrieval set contains
    direct evidence, but it should not be treated as a bias detector in the statistical or legal sense. The probes show
    that the data includes discrimination narratives and demographic assumptions; they do not prove population-level
    bias. The main risk is not just hallucination, but overconfident compression: an LLM can turn contested Reddit
    anecdotes into a clean-sounding claim unless the answer is forced to stay close to citations and caveats.
    """)

    section_divider()

    st.subheader("Note on Ethics")
    st.markdown("""
    <div class="note-grid">
        <div class="note-card ethics-card">
            <div class="mini-label">Personal information</div>
            <h4>Anonymization is fragile.</h4>
            <p>Even if usernames are hidden in the UI, posts may contain unusual job titles, locations, timelines,
            employers, salary details, immigration status, health information, or conflict narratives. Combining
            post content with author history could re-identify a real person.</p>
        </div>
        <div class="note-card ethics-card">
            <div class="mini-label">Dataset scenario</div>
            <h4>Work stories are often identifying.</h4>
            <p>In this corpus, high-scoring posts include highly specific workplace disputes, hiring outcomes,
            visa/nationality claims, and home/work boundary incidents. A single snippet may be anonymous; a title,
            month, flair, score, and posting history together may not be.</p>
        </div>
        <div class="note-card ethics-card">
            <div class="mini-label">Deletion rights</div>
            <h4>Right to be forgotten is difficult for RAG.</h4>
            <p>If a Reddit user deletes a post after scraping, the local database, vector index, graph artifact,
            topic summaries, caches, and evaluation files may still contain derived traces. Production compliance
            requires deletion propagation, not only removing the raw row.</p>
        </div>
        <div class="note-card ethics-card">
            <div class="mini-label">Production stance</div>
            <h4>Full compliance is possible only by design.</h4>
            <p>A realistic production system needs source IDs, index tombstones, scheduled re-syncs, cache expiry,
            audit logs, and regeneration of derived summaries. Without that, a RAG system can preserve content that
            the original user intended to remove.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Ethics reflection.** This project is best understood as research over a sensitive public corpus, not as a product
    that should expose raw Reddit histories. The dashboard should avoid usernames by default, truncate examples, and
    present answers as aggregate tendencies. For a production version, I would add a deletion pipeline keyed by Reddit
    post/comment IDs, rebuild or selectively delete vector embeddings, remove graph nodes, expire cached topic and stance
    summaries, and document the remaining limitations. Full right-to-be-forgotten compliance is not realistic if derived
    artifacts are treated as static outputs; it becomes more realistic only if every artifact remains traceable back to
    removable source records.
    """)


# ════════════════════════════════════════════════════════════════════════
#  SIDEBAR & ROUTER
# ════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-mark">J</div>
        <div class="brand-title">r/jobs</div>
        <div class="brand-subtitle">NLP project · Apr 2025 to Mar 2026</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Dashboard", "Topics", "Trends", "Stance", "QA", "Evaluation", "Bias & Ethics"],
        format_func=lambda p: {
            "Dashboard": "\u2001Dashboard",
            "Topics": "\u2001Topics",
            "Trends": "\u2001Trends",
            "Stance": "\u2001Stance",
            "QA": "\u2001Question Answering",
            "Evaluation": "\u2001Evaluation Results",
            "Bias & Ethics": "\u2001Bias & Ethics",
        }[p],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # ── Data freshness indicator
    import os
    cache_path = CACHE_DIR / "agg_cache.json"
    if cache_path.exists():
        from datetime import datetime
        mtime = os.path.getmtime(cache_path)
        freshness = datetime.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M")
        st.markdown(
            f"<div style='color:#9ca3af; font-size:0.7rem; text-align:center;'>"
            f"Data last computed<br>{freshness}</div>",
            unsafe_allow_html=True,
        )


if page == "Dashboard":
    render_dashboard()
elif page == "Topics":
    render_topics()
elif page == "Trends":
    render_trends()
elif page == "Stance":
    render_stance()
elif page == "QA":
    render_qa()
elif page == "Evaluation":
    render_evaluation()
elif page == "Bias & Ethics":
    render_bias_ethics()
