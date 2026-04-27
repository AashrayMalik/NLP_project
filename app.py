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

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nlp_logic.topic_insights import build_topic_display_label, compute_topic_flair_analysis

# ── Config ──────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parent / "scraping" / "data" / "jobs_posts.db"
CACHE_DIR = Path(__file__).resolve().parent / "scraping" / "data"

st.set_page_config(
    page_title="r/jobs Subreddit Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — clean dark theme ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ── KPI Cards ── */
    .kpi-card {
        background: #16161e;
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 1.3rem 1rem;
        text-align: center;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #8b5cf6;
        box-shadow: 0 0 16px rgba(139, 92, 246, 0.08);
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #c4b5fd;
        margin: 0.2rem 0;
        letter-spacing: -0.02em;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 500;
    }

    /* ── Topic cards ── */
    .topic-badge {
        display: inline-block;
        background: rgba(139, 92, 246, 0.12);
        color: #a78bfa;
        padding: 0.18rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
        border: 1px solid rgba(139, 92, 246, 0.15);
    }

    .topic-grid-card {
        background: #16161e;
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s ease;
    }
    .topic-grid-card:hover {
        border-color: #3b3b52;
    }
    .topic-grid-card h4 {
        margin: 0 0 0.4rem 0;
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .topic-grid-card .topic-stats {
        font-size: 0.8rem;
        color: #9ca3af;
        margin-bottom: 0.5rem;
    }

    /* ── Trend badges ── */
    .badge-persistent {
        display: inline-block;
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid rgba(16, 185, 129, 0.15);
    }
    .badge-trending {
        display: inline-block;
        background: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid rgba(245, 158, 11, 0.15);
    }
    .badge-seasonal {
        display: inline-block;
        background: rgba(6, 182, 212, 0.12);
        color: #22d3ee;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
        border: 1px solid rgba(6, 182, 212, 0.15);
    }

    /* ── Stance bars ── */
    .stance-support {
        background: rgba(16, 185, 129, 0.08);
        border-left: 3px solid #34d399;
        padding: 0.7rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .stance-oppose {
        background: rgba(244, 63, 94, 0.08);
        border-left: 3px solid #f43f5e;
        padding: 0.7rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .stance-neutral {
        background: rgba(148, 163, 184, 0.08);
        border-left: 3px solid #64748b;
        padding: 0.7rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* ── Page headers ── */
    .page-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 0.15rem;
        letter-spacing: -0.01em;
    }
    .page-subheader {
        font-size: 0.88rem;
        color: #6b7280;
        margin-bottom: 1.8rem;
        line-height: 1.45;
    }

    /* ── Section dividers ── */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #2a2a3a, transparent);
        margin: 1.5rem 0;
    }

    /* ── Insight card ── */
    .insight-card {
        background: #16161e;
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0 1.5rem 0;
        font-size: 0.88rem;
        line-height: 1.65;
        color: #9ca3af;
    }
    .insight-card strong {
        color: #c4b5fd;
    }

    /* ── Answer panel (QA page) ── */
    .answer-panel {
        background: #16161e;
        border: 1px solid #2a2a3a;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        line-height: 1.7;
        font-size: 0.92rem;
    }

    /* ── Example question chips ── */
    .example-chip {
        display: inline-block;
        background: #1e1e2e;
        border: 1px solid #2a2a3a;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        margin: 0.25rem 0.3rem;
        font-size: 0.78rem;
        color: #9ca3af;
        cursor: default;
        transition: border-color 0.15s ease;
    }
    .example-chip:hover {
        border-color: #8b5cf6;
        color: #c4b5fd;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0e0e14;
        border-right: 1px solid #2a2a3a;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0;
        font-size: 0.9rem;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #c4b5fd;
    }
    [data-testid="stSidebar"] * {
        color: #d1d5db;
    }
    [data-testid="stSidebar"] hr {
        border-color: #2a2a3a;
    }

    div[data-testid="stMetric"] {
        background: #16161e;
        border: 1px solid #2a2a3a;
        border-radius: 10px;
        padding: 0.8rem;
    }

    /* ── Summary stat highlight ── */
    .stat-highlight {
        display: inline-block;
        background: rgba(139, 92, 246, 0.1);
        color: #c4b5fd;
        padding: 0.1rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.88rem;
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


def kpi_card(label: str, value: str):
    """Render a styled KPI card."""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
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


# ── Plotly theme ────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=13, color="#9ca3af"),
    margin=dict(l=40, r=20, t=50, b=40),
)

PALETTE = ["#8b5cf6", "#06b6d4", "#10b981", "#f43f5e", "#f59e0b",
           "#a78bfa", "#22d3ee", "#34d399", "#fb923c", "#f472b6",
           "#818cf8", "#2dd4bf", "#fbbf24", "#e879f9", "#60a5fa"]


# ════════════════════════════════════════════════════════════════════════
#  DASHBOARD PAGE
# ════════════════════════════════════════════════════════════════════════

def render_dashboard():
    st.markdown('<div class="page-header">Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheader">Aggregate statistics for the cleaned r/jobs dataset — 12 months of posts and top comments from Apr 2025 to Mar 2026</div>', unsafe_allow_html=True)

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

    # ── Monthly activity
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Posts")
        st.caption("Post volume across the 12-month collection window")
        ppm = data["posts_per_month"]
        fig = px.bar(ppm, x="month", y="count", color_discrete_sequence=["#8b5cf6"])
        fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="", yaxis_title="Posts")
        fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y:,} posts<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Monthly Comments")
        st.caption("Top-comment volume by month")
        cpm = data["comments_per_month"]
        fig = px.bar(cpm, x="month", y="count", color_discrete_sequence=["#06b6d4"])
        fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="", yaxis_title="Comments")
        fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y:,} comments<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # ── Flair distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Flair Distribution")
        st.caption("Reddit post categories assigned by users or moderators")
        flair = data["flair_distribution"]
        fig = px.pie(flair[:15], values="count", names="flair",
                     color_discrete_sequence=PALETTE, hole=0.45)
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=10)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Engagement Tiers")
        st.caption("Posts split by engagement level (score + comment count)")
        tier = data["tier_breakdown"]
        fig = go.Figure(data=[go.Pie(
            labels=list(tier.keys()), values=list(tier.values()),
            marker_colors=["#8b5cf6", "#f43f5e"], hole=0.5,
        )])
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_traces(textposition="inside", textinfo="label+percent", textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)

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

    # ── Flair by month heatmap
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Flairs by Post Count")
        st.caption("Ranked flair categories in the corpus")
        flair_top = data["flair_distribution"][:10]
        fig = px.bar(flair_top, x="count", y="flair", orientation="h",
                     color_discrete_sequence=["#818cf8"])
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"),
                          xaxis_title="Posts", yaxis_title="")
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,} posts<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Post Length Distribution")
        st.caption("How long are r/jobs posts? (character count buckets)")
        import pandas as pd
        # Build length buckets from overview data
        avg_len = overview['avg_post_length']
        st.markdown(f"""
        <div class="insight-card">
            <strong>Average post length:</strong> {avg_len:,.0f} characters<br>
            <strong>Average comment length:</strong> {overview['avg_comment_length']:,.0f} characters<br>
            <strong>Average post score:</strong> {overview['avg_post_score']:.1f} karma<br>
            <strong>Average comment score:</strong> {overview['avg_comment_score']:.1f} karma<br><br>
            Posts in r/jobs tend to be detailed personal stories or questions.
            Comments are shorter and more direct — advice, reactions, or shared experiences.
        </div>
        """, unsafe_allow_html=True)

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
    st.markdown('<div class="page-header">Topic Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheader">BERTopic clusters with cleaned labels, keyword coverage, and flair gap analysis</div>', unsafe_allow_html=True)

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
    fig = px.bar(chart_topics, x="topic", y="share_pct", color="topic",
                 color_discrete_sequence=PALETTE)
    fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="", yaxis_title="Share (%)",
                      showlegend=False, xaxis_tickangle=-45)
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

    # ── Topic card grid (2-column layout)
    st.subheader("Topic Details")
    st.caption("Click any card to see representative posts and detailed flair info")

    for i in range(0, len(real_topics), 2):
        cols = st.columns(2)
        for col_idx, col in enumerate(cols):
            topic_idx = i + col_idx
            if topic_idx >= len(real_topics):
                break

            t = real_topics[topic_idx]
            flair_info = flair_lookup.get(t["id"], {})
            top_flairs = flair_info.get("top_flairs", [])
            trend = t.get("trend_type")

            with col:
                # Card header as HTML
                trend_html = trend_badge_html(trend)
                kw_html = " ".join(
                    f'<span class="topic-badge">{kw}</span>' for kw in t["keywords"][:6]
                )

                flair_text = ""
                if top_flairs:
                    flair_text = " · ".join(f"{f['flair']} ({f['pct']:.0f}%)" for f in top_flairs[:2])

                st.markdown(f"""
                <div class="topic-grid-card">
                    <h4>T{t['id']}: {t['display_label']} {trend_html}</h4>
                    <div class="topic-stats">{t['post_count']:,} posts · {t['share_pct']:.1f}% of corpus{' · ' + flair_text if flair_text else ''}</div>
                    {kw_html}
                </div>
                """, unsafe_allow_html=True)

                # Expander for detailed view
                docs = rep_docs.get(str(t["id"]), [])
                with st.expander("Details", expanded=False):
                    st.markdown(f"**Model Label:** {t['label']}")

                    if flair_info.get("suggested_missing_flair"):
                        st.markdown(
                            f"**Potential missing flair:** {flair_info['suggested_missing_flair']}  "
                            f"  \n{flair_info['gap_reason']}"
                        )

                    if docs:
                        st.markdown("**Representative Posts:**")
                        for j, doc in enumerate(docs[:3], 1):
                            st.markdown(f"> **{j}.** {doc[:300]}{'...' if len(doc) > 300 else ''}")

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
    st.markdown('<div class="page-header">Trending vs Persistent Topics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheader">Classification of topics by temporal behaviour across the 12-month collection window</div>', unsafe_allow_html=True)

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
        kpi_card("Persistent Topics", str(len(persistent)))
    with cols[1]:
        kpi_card("Trending Topics", str(len(trending)))
    with cols[2]:
        kpi_card("Seasonal Topics", str(len(seasonal)))

    # ── Narrative summary
    persistent_names = [topic_labels.get(c["topic_id"], f"Topic {c['topic_id']}") for c in persistent[:3]]
    trending_names = [topic_labels.get(c["topic_id"], f"Topic {c['topic_id']}") for c in trending[:3]]

    narrative_parts = []
    if persistent:
        narrative_parts.append(
            f"<strong>{len(persistent)} topic{'s' if len(persistent) != 1 else ''}</strong> appear consistently across most months, "
            f"including {', '.join(persistent_names)}."
        )
    if trending:
        narrative_parts.append(
            f"<strong>{len(trending)} topic{'s' if len(trending) != 1 else ''}</strong> show spikes or recent emergence, "
            f"such as {', '.join(trending_names)}."
        )
    if seasonal:
        narrative_parts.append(
            f"<strong>{len(seasonal)} topic{'s' if len(seasonal) != 1 else ''}</strong> vary with moderate seasonality."
        )

    if narrative_parts:
        st.markdown(f'<div class="insight-card">{" ".join(narrative_parts)}</div>', unsafe_allow_html=True)

    # ── Monthly sparklines for all topics
    st.subheader("Monthly Post Volume by Topic")
    st.caption("Each line represents a topic's monthly post count, colour-coded by trend classification")

    if trending_data:
        all_months = sorted(trending_data[0]["monthly_counts"].keys())

        trend_colors = {
            "persistent": "#10b981",
            "trending": "#f59e0b",
            "seasonal": "#06b6d4",
        }

        fig = go.Figure()
        for c in sorted(trending_data, key=lambda x: x["total_posts"], reverse=True):
            label = topic_labels.get(c["topic_id"], f"Topic {c['topic_id']}")
            counts = [c["monthly_counts"].get(m, 0) for m in all_months]
            color = trend_colors.get(c["trend_type"], "#6b7280")
            fig.add_trace(go.Scatter(
                x=all_months, y=counts,
                name=f"T{c['topic_id']}: {label[:25]}",
                line=dict(color=color, width=2),
                mode="lines+markers",
                marker=dict(size=4),
            ))

        fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="",
                          yaxis_title="Posts", legend_title="",
                          height=500)
        st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # ── Persistent topics section
    st.subheader("Persistent Topics")
    st.caption("These topics appear consistently across most months with stable volume")

    if not persistent:
        st.info("No persistent topics identified.")
    else:
        for c in persistent:
            label = topic_labels.get(c["topic_id"], f"Topic {c['topic_id']}")
            with st.expander(f"Topic {c['topic_id']}: {label}", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Presence", f"{c['presence']:.0%}")
                with col2:
                    st.metric("CV", f"{c['cv']:.2f}")
                with col3:
                    st.metric("Mean/Month", f"{c['mean_monthly']:.0f}")
                with col4:
                    st.metric("Total Posts", f"{c['total_posts']:,}")

                months = sorted(c["monthly_counts"].keys())
                counts = [c["monthly_counts"][m] for m in months]
                fig = go.Figure(go.Bar(x=months, y=counts, marker_color="#10b981"))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=30))
                st.plotly_chart(fig, use_container_width=True)

    # ── Trending topics section
    st.subheader("Trending Topics")
    st.caption("Topics showing spikes or appearing in specific months, indicating emerging or fading interest")

    if not trending:
        st.info("No trending topics identified.")
    else:
        for c in trending:
            label = topic_labels.get(c["topic_id"], f"Topic {c['topic_id']}")
            with st.expander(f"Topic {c['topic_id']}: {label}", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Presence", f"{c['presence']:.0%}")
                with col2:
                    st.metric("Has Spike", "Yes" if c["has_spike"] else "No")
                with col3:
                    st.metric("Peak Month", c["peak_month"])
                with col4:
                    st.metric("Peak Count", f"{c['peak_count']:,}")

                months = sorted(c["monthly_counts"].keys())
                counts = [c["monthly_counts"][m] for m in months]
                fig = go.Figure(go.Bar(x=months, y=counts, marker_color="#f59e0b"))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=30))
                st.plotly_chart(fig, use_container_width=True)

    # ── Seasonal topics section
    if seasonal:
        st.subheader("Seasonal Topics")
        st.caption("Topics appearing regularly but with moderate variation across months")

        for c in seasonal:
            label = topic_labels.get(c["topic_id"], f"Topic {c['topic_id']}")
            with st.expander(f"Topic {c['topic_id']}: {label}", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Presence", f"{c['presence']:.0%}")
                with col2:
                    st.metric("CV", f"{c['cv']:.2f}")
                with col3:
                    st.metric("Mean/Month", f"{c['mean_monthly']:.0f}")
                with col4:
                    st.metric("Total Posts", f"{c['total_posts']:,}")

                months = sorted(c["monthly_counts"].keys())
                counts = [c["monthly_counts"][m] for m in months]
                fig = go.Figure(go.Bar(x=months, y=counts, marker_color="#06b6d4"))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=30))
                st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════
#  STANCE ANALYSIS PAGE
# ════════════════════════════════════════════════════════════════════════

def render_stance():
    st.markdown('<div class="page-header">Stance Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheader">Comment stance relative to a broader discussion frame built from each topic\'s top posts</div>', unsafe_allow_html=True)

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

    st.info(
        "These labels are relative, not absolute. "
        "'For' means a comment aligns with the topic's dominant discussion frame, "
        "'Opposing' means it pushes back on that framing, and "
        "'Neutral / unclear' captures advice, side discussion, or low-confidence classifications. "
        "Read this as a split around the extracted frame, not a universal pro/con vote on the whole topic."
    )

    # ── Overview chart
    st.subheader("Stance Distribution Across Topics")
    st.caption("Stacked view of for / opposing / neutral splits per topic")

    chart_data = []
    for tid, s in stance_data.items():
        label = topic_labels.get(str(tid), s.get("topic_label", f"Topic {tid}"))
        chart_data.append({
            "topic": f"T{tid}: {label[:35]}",
            "For dominant frame %": s.get("for", s["support"])["pct"],
            "Opposing dominant frame %": s.get("opposing", s["oppose"])["pct"],
            "Neutral / unclear %": s.get("neutral", {"pct": 0.0})["pct"],
        })

    import pandas as pd
    df = pd.DataFrame(chart_data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="For dominant frame", x=df["topic"], y=df["For dominant frame %"],
        marker_color="#10b981",
    ))
    fig.add_trace(go.Bar(
        name="Opposing dominant frame", x=df["topic"], y=df["Opposing dominant frame %"],
        marker_color="#f43f5e",
    ))
    fig.add_trace(go.Bar(
        name="Neutral / unclear", x=df["topic"], y=df["Neutral / unclear %"],
        marker_color="#64748b",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="stack",
                      xaxis_title="", yaxis_title="Percentage",
                      xaxis_tickangle=-45, height=450)
    st.plotly_chart(fig, use_container_width=True)

    section_divider()

    # ── Per-topic stance details
    st.subheader("Topic-Level Stance Breakdown")

    for tid, s in stance_data.items():
        label = topic_labels.get(str(tid), s.get("topic_label", f"Topic {tid}"))
        dominant = s.get("dominant_position", "N/A")
        frame = s.get("reference_frame", dominant)
        total = s.get("total_comments_analysed", 0)
        for_bucket = s.get("for", s["support"])
        opposing_bucket = s.get("opposing", s["oppose"])
        neutral_bucket = s.get("neutral", {"pct": 0.0, "count": 0, "user_count": 0, "top_arguments": []})

        with st.expander(f"Topic {tid}: {label}", expanded=False):
            st.markdown(f"**Anchor Post:** *{dominant[:200]}*")
            st.markdown(f"**Discussion Frame:** {frame[:320]}")
            st.markdown(f"**Comments Analysed:** {total:,}")
            st.caption(
                "Interpretation: left aligns with the frame, middle pushes against it, and neutral captures low-confidence or non-stance comments."
            )

            st.markdown("---")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"### For ({for_bucket['pct']:.0f}%)")
                st.metric("Comments", f"{for_bucket['count']:,}")
                st.metric("Unique Users", f"{for_bucket['user_count']:,}")

                if for_bucket["top_arguments"]:
                    st.markdown("**Representative comments aligned with the frame:**")
                    for i, arg in enumerate(for_bucket["top_arguments"][:3], 1):
                        st.markdown(
                            f'<div class="stance-support">'
                            f'<strong>{i}.</strong> {arg["body"][:300]}'
                            f'{"..." if len(arg["body"]) > 300 else ""}'
                            f'<br><em style="color:#6b7280">— u/{arg["author"]} '
                            f'(score: {arg["score"]})</em></div>',
                            unsafe_allow_html=True,
                        )

            with col2:
                st.markdown(f"### Opposing ({opposing_bucket['pct']:.0f}%)")
                st.metric("Comments", f"{opposing_bucket['count']:,}")
                st.metric("Unique Users", f"{opposing_bucket['user_count']:,}")

                if opposing_bucket["top_arguments"]:
                    st.markdown("**Representative comments pushing against the frame:**")
                    for i, arg in enumerate(opposing_bucket["top_arguments"][:3], 1):
                        st.markdown(
                            f'<div class="stance-oppose">'
                            f'<strong>{i}.</strong> {arg["body"][:300]}'
                            f'{"..." if len(arg["body"]) > 300 else ""}'
                            f'<br><em style="color:#6b7280">— u/{arg["author"]} '
                            f'(score: {arg["score"]})</em></div>',
                            unsafe_allow_html=True,
                        )

            with col3:
                st.markdown(f"### Neutral / Unclear ({neutral_bucket['pct']:.0f}%)")
                st.metric("Comments", f"{neutral_bucket['count']:,}")
                st.metric("Unique Users", f"{neutral_bucket['user_count']:,}")

                if neutral_bucket["top_arguments"]:
                    st.markdown("**Representative neutral or mixed comments:**")
                    for i, arg in enumerate(neutral_bucket["top_arguments"][:3], 1):
                        st.markdown(
                            f'<div class="stance-neutral">'
                            f'<strong>{i}.</strong> {arg["body"][:300]}'
                            f'{"..." if len(arg["body"]) > 300 else ""}'
                            f'<br><em style="color:#6b7280">— u/{arg["author"]} '
                            f'(score: {arg["score"]})</em></div>',
                            unsafe_allow_html=True,
                        )


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
    st.markdown('<div class="page-header">Question Answering</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subheader">Semantic KG hybrid RAG over the collected r/jobs posts, comments, topics, trends, and stance summaries</div>',
        unsafe_allow_html=True,
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
#  SIDEBAR & ROUTER
# ════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 1.1rem; font-weight: 700; color: #e2e8f0; letter-spacing: -0.01em;">
            r/jobs Analysis
        </div>
        <div style="font-size: 0.72rem; color: #9ca3af; margin-top: 0.15rem;">
            NLP Project · Apr 2025 – Mar 2026
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Dashboard", "Topics", "Trends", "Stance", "QA"],
        format_func=lambda p: {
            "Dashboard": "\u2001Dashboard",
            "Topics": "\u2001Topics",
            "Trends": "\u2001Trends",
            "Stance": "\u2001Stance",
            "QA": "\u2001Question Answering",
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
