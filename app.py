"""
app.py — Streamlit interactive application for r/jobs subreddit analysis
=========================================================================
Multi-page app with sidebar navigation:
  - Dashboard: KPI cards, monthly trends, flair distribution, author stats
  - Topics: Topic cards with labels, keywords, share %
  - Trends: Trending vs persistent topic classification
  - Stance: Stance detection and argument summarisation

Run with: streamlit run app.py
"""

import json
import sqlite3
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nlp_logic.topic_insights import build_topic_display_label, compute_topic_flair_analysis

# config
DB_PATH = Path(__file__).resolve().parent / "scraping" / "data" / "jobs_posts.db"
CACHE_DIR = Path(__file__).resolve().parent / "scraping" / "data"

st.set_page_config(
    page_title="r/jobs Subreddit Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# css
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* KPI Card styling */
    .kpi-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #a78bfa;
        margin: 0.3rem 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Topic card styling */
    .topic-badge {
        display: inline-block;
        background: rgba(167, 139, 250, 0.2);
        color: #a78bfa;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.3rem;
    }

    /* Trend type badges */
    .badge-persistent {
        display: inline-block;
        background: rgba(52, 211, 153, 0.2);
        color: #34d399;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-trending {
        display: inline-block;
        background: rgba(251, 146, 60, 0.2);
        color: #fb923c;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-seasonal {
        display: inline-block;
        background: rgba(56, 189, 248, 0.2);
        color: #38bdf8;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Stance bars */
    .stance-support {
        background: linear-gradient(90deg, rgba(52,211,153,0.3), rgba(52,211,153,0.1));
        border-left: 4px solid #34d399;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }
    .stance-oppose {
        background: linear-gradient(90deg, rgba(248,113,113,0.3), rgba(248,113,113,0.1));
        border-left: 4px solid #f87171;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }

    /* Page header */
    .page-header {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .page-subheader {
        font-size: 0.95rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #e2e8f0;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# helpers 
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


# theme
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=40, r=20, t=50, b=40),
)

PALETTE = ["#a78bfa", "#38bdf8", "#34d399", "#f472b6", "#fb923c",
           "#facc15", "#818cf8", "#22d3ee", "#a3e635", "#f87171",
           "#c084fc", "#2dd4bf", "#fbbf24", "#e879f9", "#60a5fa"]


#dashboard

def render_dashboard():
    st.markdown('<div class="page-header">Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheader">Aggregate statistics for the cleaned r/jobs dataset (12 months, Apr 2025 – Mar 2026)</div>', unsafe_allow_html=True)

    data = load_agg_cache()
    overview = data["overview"]

    # ── KPI row
    cols = st.columns(5)
    with cols[0]:
        kpi_card("Total Posts", f"{overview['total_posts']:,}")
    with cols[1]:
        kpi_card("Total Comments", f"{overview['total_comments']:,}")
    with cols[2]:
        kpi_card("Unique Authors", f"{overview['unique_post_authors'] + overview['unique_comment_authors']:,}")
    with cols[3]:
        kpi_card("Avg Post Length", f"{overview['avg_post_length']:,.0f} chars")
    with cols[4]:
        kpi_card("Avg Post Score", f"{overview['avg_post_score']:.1f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly activity 
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Posts")
        ppm = data["posts_per_month"]
        fig = px.bar(ppm, x="month", y="count", color_discrete_sequence=["#a78bfa"])
        fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Month", yaxis_title="Posts")
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Monthly Comments")
        cpm = data["comments_per_month"]
        fig = px.bar(cpm, x="month", y="count", color_discrete_sequence=["#38bdf8"])
        fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Month", yaxis_title="Comments")
        st.plotly_chart(fig, width="stretch")

    # Flair distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Flair Distribution")
        flair = data["flair_distribution"]
        fig = px.pie(flair[:15], values="count", names="flair",
                     color_discrete_sequence=PALETTE, hole=0.4)
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_traces(textposition="inside", textinfo="percent+label", textfont_size=10)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Engagement Tiers")
        tier = data["tier_breakdown"]
        fig = go.Figure(data=[go.Pie(
            labels=list(tier.keys()), values=list(tier.values()),
            marker_colors=["#a78bfa", "#f472b6"], hole=0.5,
        )])
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_traces(textposition="inside", textinfo="label+percent", textfont_size=13)
        st.plotly_chart(fig, width="stretch")

    # Score distributions
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Post Score Distribution")
        sdp = data["score_dist_posts"]
        fig = px.bar(sdp, x="bucket", y="count", color_discrete_sequence=["#34d399"])
        fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Score Range", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Comment Score Distribution")
        sdc = data["score_dist_comments"]
        fig = px.bar(sdc, x="bucket", y="count", color_discrete_sequence=["#fb923c"])
        fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Score Range", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")

    #Top authors
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Authors by Posts")
        tap = data["top_authors_posts"][:10]
        fig = px.bar(tap, x="post_count", y="author", orientation="h",
                     color_discrete_sequence=["#818cf8"])
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"),
                          xaxis_title="Posts", yaxis_title="")
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Top Authors by Comments")
        tac = data["top_authors_comments"][:10]
        fig = px.bar(tac, x="comment_count", y="author", orientation="h",
                     color_discrete_sequence=["#22d3ee"])
        fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed"),
                          xaxis_title="Comments", yaxis_title="")
        st.plotly_chart(fig, width="stretch")

    #Average length over time
    st.subheader("Average Post & Comment Length Over Time")
    alm = data["avg_length_per_month"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r["month"] for r in alm], y=[r["avg_post_length"] for r in alm],
        name="Avg Post Length", line=dict(color="#a78bfa", width=3), mode="lines+markers",
    ))
    fig.add_trace(go.Scatter(
        x=[r["month"] for r in alm], y=[r["avg_comment_length"] for r in alm],
        name="Avg Comment Length", line=dict(color="#38bdf8", width=3), mode="lines+markers",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Month", yaxis_title="Characters")
    st.plotly_chart(fig, width="stretch")


#topics

def render_topics():
    st.markdown('<div class="page-header">Topic Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheader">Cleaner topic summaries, representative posts, and flair coverage for BERTopic clusters</div>', unsafe_allow_html=True)

    topics = load_topics_from_db()
    rep_docs = load_representative_docs()
    flair_analysis = load_topic_flair_analysis()

    if not topics:
        st.warning("No topic data found. Run `python precompute.py --only topics` first.")
        return

    real_topics = [t for t in topics if t["id"] != -1]
    outlier = next((t for t in topics if t["id"] == -1), None)
    flair_lookup = {item["topic_id"]: item for item in flair_analysis}

    # Summary metrics
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

    #Topic share chart
    st.subheader("Topic Distribution")
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
    st.plotly_chart(fig, width="stretch")

    st.subheader("Topics vs Post Flairs")
    st.markdown(
        "This compares each discovered topic to the subreddit's existing flair system. "
        "Low dominant-flair share usually means the topic is spread across broad flairs and may deserve a more specific label."
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
            color_discrete_map={"Strong": "#34d399", "Partial": "#fbbf24", "Gap": "#f87171"},
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            xaxis_title="",
            yaxis_title="Top Flair Share (%)",
            xaxis_tickangle=-45,
            height=420,
        )
        st.plotly_chart(fig, width="stretch")

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

    # Topic cards
    st.subheader("Topic Details")

    for t in real_topics:
        flair_info = flair_lookup.get(t["id"], {})
        top_flairs = flair_info.get("top_flairs", [])

        with st.expander(f"Topic {t['id']}: {t['display_label']}", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Model Label:** {t['label']}")
                st.markdown("**Keywords:**")
                kw_html = " ".join(
                    f'<span class="topic-badge">{kw}</span>' for kw in t["keywords"][:10]
                )
                st.markdown(kw_html, unsafe_allow_html=True)

                if top_flairs:
                    flair_text = ", ".join(
                        f"{item['flair']} ({item['pct']:.1f}%)" for item in top_flairs
                    )
                    st.markdown(f"**Most common flairs:** {flair_text}")

                if flair_info.get("suggested_missing_flair"):
                    st.markdown(
                        f"**Potential missing flair:** {flair_info['suggested_missing_flair']}  "
                        f"  \n{flair_info['gap_reason']}"
                    )

                docs = rep_docs.get(str(t["id"]), [])
                if docs:
                    st.markdown("**Representative Posts:**")
                    for i, doc in enumerate(docs[:3], 1):
                        st.markdown(f"> **{i}.** {doc[:300]}{'...' if len(doc) > 300 else ''}")

            with col2:
                st.metric("Posts", f"{t['post_count']:,}")
                st.metric("Share", f"{t['share_pct']:.1f}%")
                if flair_info:
                    st.metric("Top Flair", flair_info["dominant_flair"])
                    st.metric("Flair Fit", flair_info["flair_fit"])

    if outlier:
        with st.expander("Outlier / Uncategorised Posts", expanded=False):
            st.markdown(f"**{outlier['post_count']:,}** posts ({outlier['share_pct']:.1f}%) "
                        "could not be confidently assigned to any topic cluster. "
                        "This is expected with HDBSCAN — these are posts with unique or "
                        "mixed themes that don't fit neatly into a single cluster.")


#trending vs persistent

def render_trends():
    st.markdown('<div class="page-header">Trending vs Persistent Topics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheader">Classification of topics by temporal behaviour across 12 months</div>', unsafe_allow_html=True)

    trending_data = load_trending_cache()
    topics = load_topics_from_db()

    if not trending_data:
        st.warning("No trending data found. Run `python precompute.py --only trending` first.")
        return

    # Build topic label lookup
    topic_labels = {t["id"]: t["display_label"] for t in topics}

    persistent = [c for c in trending_data if c["trend_type"] == "persistent"]
    trending = [c for c in trending_data if c["trend_type"] == "trending"]
    seasonal = [c for c in trending_data if c["trend_type"] == "seasonal"]

    #Summary KPIs
    cols = st.columns(3)
    with cols[0]:
        kpi_card("Persistent Topics", str(len(persistent)))
    with cols[1]:
        kpi_card("Trending Topics", str(len(trending)))
    with cols[2]:
        kpi_card("Seasonal Topics", str(len(seasonal)))

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly sparklines for all topics
    st.subheader("Monthly Post Volume by Topic")

    # Get all months
    if trending_data:
        all_months = sorted(trending_data[0]["monthly_counts"].keys())

        traces_data = []
        for c in sorted(trending_data, key=lambda x: x["total_posts"], reverse=True):
            label = topic_labels.get(c["topic_id"], f"Topic {c['topic_id']}")
            for m in all_months:
                traces_data.append({
                    "month": m,
                    "count": c["monthly_counts"].get(m, 0),
                    "topic": f"T{c['topic_id']}: {label[:30]}",
                })

        import pandas as pd
        df = pd.DataFrame(traces_data)
        fig = px.line(df, x="month", y="count", color="topic",
                      color_discrete_sequence=PALETTE)
        fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Month",
                          yaxis_title="Posts", legend_title="",
                          height=500)
        st.plotly_chart(fig, width="stretch")

    #Persistent topics section
    st.subheader("Persistent Topics")
    st.markdown("*These topics appear consistently across most months with stable volume.*")

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

                # Sparkline
                months = sorted(c["monthly_counts"].keys())
                counts = [c["monthly_counts"][m] for m in months]
                fig = go.Figure(go.Bar(x=months, y=counts,
                                       marker_color="#34d399"))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=30))
                st.plotly_chart(fig, width="stretch")

    # Trending topics section
    st.subheader("Trending Topics")
    st.markdown("*These topics show spikes or appear only in specific months, indicating emerging or fading interest.*")

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
                fig = go.Figure(go.Bar(x=months, y=counts,
                                       marker_color="#fb923c"))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=30))
                st.plotly_chart(fig, width="stretch")

    #  Seasonal topics section
    if seasonal:
        st.subheader("Seasonal Topics")
        st.markdown("*These topics appear regularly but with moderate variation across months.*")

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
                fig = go.Figure(go.Bar(x=months, y=counts,
                                       marker_color="#38bdf8"))
                fig.update_layout(**PLOTLY_LAYOUT)
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=30))
                st.plotly_chart(fig, width="stretch")


#Stance analysis

def render_stance():
    st.markdown('<div class="page-header">Stance Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subheader">Comment stance relative to a broader discussion frame built from each topic\'s top posts</div>', unsafe_allow_html=True)

    stance_data = load_stance_cache()
    topics = load_topics_from_db()
    topic_labels = {str(t["id"]): t["display_label"] for t in topics}

    if not stance_data:
        st.warning("No stance data found. Run `python precompute.py --only stance` first.")
        return

    st.info(
        "These labels are relative, not absolute. "
        "'For' means a comment aligns with the topic's dominant discussion frame, "
        "'Opposing' means it pushes back on that framing, and "
        "'Neutral / unclear' captures advice, side discussion, or low-confidence classifications. "
        "Read this as a split around the extracted frame, not a universal pro/con vote on the whole topic."
    )

    #Overview chart
    st.subheader("Stance Distribution Across Topics")

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
        marker_color="#34d399",
    ))
    fig.add_trace(go.Bar(
        name="Opposing dominant frame", x=df["topic"], y=df["Opposing dominant frame %"],
        marker_color="#f87171",
    ))
    fig.add_trace(go.Bar(
        name="Neutral / unclear", x=df["topic"], y=df["Neutral / unclear %"],
        marker_color="#94a3b8",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="stack",
                      xaxis_title="", yaxis_title="Percentage",
                      xaxis_tickangle=-45, height=450)
    st.plotly_chart(fig, width="stretch")

    #Per-topic stance details
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
                            f'<div class="stance-support" style="border-left-color:#94a3b8; '
                            f'background: linear-gradient(90deg, rgba(148,163,184,0.25), rgba(148,163,184,0.08));">'
                            f'<strong>{i}.</strong> {arg["body"][:300]}'
                            f'{"..." if len(arg["body"]) > 300 else ""}'
                            f'<br><em style="color:#6b7280">— u/{arg["author"]} '
                            f'(score: {arg["score"]})</em></div>',
                            unsafe_allow_html=True,
                        )


#sidebar and router 

with st.sidebar:
    st.markdown("## r/jobs Analysis")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Dashboard", "Topics", "Trends", "Stance"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<div style='color:#6b7280; font-size:0.75rem; text-align:center;'>"
        "NLP Project · r/jobs Subreddit<br>Apr 2025 – Mar 2026</div>",
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
