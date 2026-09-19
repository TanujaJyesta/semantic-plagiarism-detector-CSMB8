"""Modern, highly professional Streamlit frontend for Explainable Semantic Plagiarism Detection."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5000").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("FRONTEND_REQUEST_TIMEOUT", "180"))
SAMPLE_FILE = Path(__file__).resolve().parent.parent / "data" / "test.pdf"

# -------------------------------------------------------------
# Page Configuration & Professional CSS Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="Explainable Semantic Plagiarism Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROFESSIONAL_CSS = """
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.06) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 6px;
        background: linear-gradient(90deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        line-height: 1.5;
        margin-bottom: 8px;
    }
    .hero-disclaimer {
        font-size: 0.8rem;
        color: #64748b;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 16px 20px;
        border-radius: 14px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(99, 102, 241, 0.4);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #94a3b8;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
    }

    /* Risk Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-high { background-color: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .badge-likely { background-color: rgba(249, 115, 22, 0.18); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.4); }
    .badge-possible { background-color: rgba(234, 179, 8, 0.18); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.4); }
    .badge-none { background-color: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }

    .badge-type {
        background-color: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.35);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-family: monospace;
        font-weight: 600;
    }

    /* Card Panels */
    .custom-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .passage-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8;
        margin-bottom: 8px;
    }
    .passage-body {
        font-size: 0.94rem;
        line-height: 1.55;
        background: rgba(15, 23, 42, 0.6);
        padding: 14px;
        border-radius: 10px;
        border-left: 4px solid #6366f1;
        color: #e2e8f0;
    }
    .source-body {
        border-left-color: #ec4899;
    }
    .explanation-callout {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(99, 102, 241, 0.05) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        padding: 14px;
        border-radius: 10px;
        font-size: 0.9rem;
        color: #bfdbfe;
        margin-top: 12px;
        line-height: 1.5;
    }

    /* History Table Card */
    .history-card {
        background: rgba(30, 41, 59, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
</style>
"""
st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)


# -------------------------------------------------------------
# API Helper Functions
# -------------------------------------------------------------
def request_backend(method: str, path: str, **kwargs):
    """Make a backend request with comprehensive error handling."""
    try:
        url = f"{BACKEND_URL}{path}"
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
    except requests.RequestException:
        return None, f"⚠️ Unable to connect to backend at `{BACKEND_URL}`. Ensure Flask API is running."

    try:
        payload = response.json()
    except Exception:
        raw_text = response.text.strip() if response.text else f"HTTP {response.status_code}"
        if "<!doctype html>" in raw_text.lower() or "<html" in raw_text.lower():
            raw_text = f"Server returned HTTP {response.status_code}"
        return None, f"Backend error: {raw_text}"

    if not response.ok:
        msg = payload.get("message", f"Request failed with status code {response.status_code}.")
        return None, msg

    return payload, None


def get_health():
    """Fetch backend system health and configuration status."""
    return request_backend("GET", "/api/health")


def format_risk_badge(risk: str) -> str:
    mapping = {
        "highly_suspicious": '<span class="badge badge-high">🔴 Highly Suspicious</span>',
        "likely_plagiarism": '<span class="badge badge-likely">🟠 Likely Plagiarism</span>',
        "possibly_similar": '<span class="badge badge-possible">🟡 Possibly Similar</span>',
        "no_strong_evidence": '<span class="badge badge-none">🟢 No Strong Evidence</span>',
    }
    return mapping.get(risk, f'<span class="badge badge-none">{risk}</span>')


# -------------------------------------------------------------
# Sidebar: System Diagnostics & Control Panel
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ System Diagnostics")
    health, health_err = get_health()

    if health_err:
        st.error("Backend Server Offline")
        st.caption(health_err)
        st.code("PYTHONPATH=. .venv/bin/python backend/app.py", language="bash")
    else:
        st.success(f"Backend Server Online (`{health.get('status', 'healthy')}`)")
        c1, c2 = st.columns(2)
        c1.markdown(f"**Database:** {'✅ Ready' if health.get('database') == 'ok' else '⚠️ Error'}")
        c2.markdown(f"**SBERT:** {'✅ Loaded' if health.get('sbert_available') else '❌ Missing'}")
        
        tavily_ok = health.get("tavily_configured", False)
        firecrawl_ok = health.get("firecrawl_configured", False)
        
        st.markdown(f"**Tavily Web Search:** {'✅ Configured' if tavily_ok else '⚠️ Key Needed'}")
        st.markdown(f"**Firecrawl Retrieval:** {'✅ Configured' if firecrawl_ok else '⚠️ Key Needed'}")

    st.divider()
    st.markdown("### ⚙️ Engine Parameters")
    st.markdown("""
    - **Semantic Weight:** `60% (SBERT)`
    - **Lexical Weight:** `40% (TF-IDF)`
    - **Embedding Model:** `all-MiniLM-L6-v2`
    - **URL Deduplication:** `Canonical Normalization`
    """)

    with st.expander("🔑 API Key Configuration"):
        st.markdown("""
        To enable real-time web source discovery & crawling, create a `.env` file in the project folder:
        ```ini
        TAVILY_API_KEY=tvly-xxxxxxxxxxxx
        FIRECRAWL_API_KEY=fc-xxxxxxxxxxxx
        TAVILY_SEARCH_DEPTH=advanced
        ```
        Then restart the Flask backend server.
        """)

    st.divider()
    st.caption("Explainable Semantic Plagiarism Detection v2.0")


# -------------------------------------------------------------
# Hero Banner
# -------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🔍 Explainable Semantic Plagiarism Detection</div>
    <div class="hero-subtitle">
        Analyze single student documents (<b>PDF</b>, <b>DOCX</b>, <b>TXT</b>) against public web sources via <b>Tavily</b> & <b>Firecrawl</b>, 
        evaluating dual-layer lexical (<b>TF-IDF</b>) and semantic (<b>SBERT</b>) similarity with transparent evidence.
    </div>
    <div class="hero-disclaimer">ℹ️ Automated research risk assessment prototype for scholarly review.</div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Document Upload Section
# -------------------------------------------------------------
col_upload, col_sample = st.columns([3, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a document for plagiarism analysis",
        type=["pdf", "docx", "txt"],
        help="Upload a single document (.pdf, .docx, .txt). Maximum size: 10MB.",
        label_visibility="collapsed",
    )

with col_sample:
    use_sample = st.button("📄 Load Sample PDF (`test.pdf`)", use_container_width=True)

# Prepare document payload
file_name = None
file_bytes = None
file_type = None

if use_sample and SAMPLE_FILE.exists():
    file_name = "test.pdf"
    file_bytes = SAMPLE_FILE.read_bytes()
    file_type = "application/pdf"
    st.info(f"Loaded sample file: **{file_name}** ({len(file_bytes) / 1024:.1f} KB)")
elif uploaded_file:
    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    file_type = uploaded_file.type or "application/octet-stream"
    st.success(f"Selected: **{file_name}** ({len(file_bytes) / 1024:.1f} KB)")


# -------------------------------------------------------------
# Execution & Interactive Progress Bar
# -------------------------------------------------------------
if file_bytes and file_name:
    if st.button("🚀 Analyze Document", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Initializing document analysis…")
        status_text = st.empty()
        
        # Step 1: Text extraction
        progress_bar.progress(15, text="📄 Step 1/5: Extracting and normalizing document text…")
        time.sleep(0.3)
        
        # Step 2: Passage selection
        progress_bar.progress(35, text="🎯 Step 2/5: Selecting high-information passages and generating queries…")
        time.sleep(0.3)
        
        # Step 3: Web search via Tavily
        progress_bar.progress(55, text="🌐 Step 3/5: Discovering candidate web sources via Tavily API…")
        time.sleep(0.3)
        
        # Step 4: Scraping via Firecrawl
        progress_bar.progress(75, text="🕷️ Step 4/5: Retrieving clean page markdown via Firecrawl…")
        time.sleep(0.3)
        
        # Step 5: Similarity computation
        progress_bar.progress(90, text="🧠 Step 5/5: Computing TF-IDF & SBERT similarity matrices…")
        
        start_time = time.time()
        payload, error = request_backend(
            "POST",
            "/api/analyze",
            files={"document": (file_name, file_bytes, file_type)},
        )
        elapsed = time.time() - start_time
        
        if error:
            progress_bar.empty()
            st.error(f"❌ Analysis failed: {error}")
            if "TAVILY_API_KEY" in error:
                st.warning("👉 To run live web search, configure your `TAVILY_API_KEY` in `.env` or use the automated test suite.")
        else:
            progress_bar.progress(100, text=f"✅ Analysis completed in {elapsed:.1f} seconds!")
            time.sleep(0.6)
            progress_bar.empty()
            st.session_state["latest_report"] = payload
            st.session_state.pop("history", None)


# -------------------------------------------------------------
# Results Dashboard Presentation
# -------------------------------------------------------------
if "latest_report" in st.session_state:
    report = st.session_state["latest_report"]
    summary = report.get("summary", {})
    matches = report.get("matches", [])
    all_comparisons = report.get("all_comparisons", [])
    sources = report.get("sources", [])
    warnings = report.get("warnings", [])

    st.markdown("---")
    st.markdown(f"### 📊 Analysis Report: `{report.get('filename')}`")
    st.caption(f"Status: **{report.get('status')}** · Total Sentences: **{summary.get('total_passages', 0)}**")

    # Display warnings if any
    for w in warnings:
        st.warning(f"⚠️ {w}")

    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Passages", summary.get("total_passages", 0))
    kpi2.metric("Sources Found", summary.get("sources_found", 0))
    kpi3.metric("Pages Scraped", summary.get("sources_retrieved", 0))
    kpi4.metric("Suspicious Matches", summary.get("suspicious_matches", 0))
    
    highest_sim = summary.get("highest_similarity", 0.0)
    kpi5.metric("Peak Similarity", f"{highest_sim * 100:.1f}%")

    overall_risk = summary.get("overall_risk", "no_strong_evidence")
    st.markdown(f"**Overall Assessment:** {format_risk_badge(overall_risk)}", unsafe_allow_html=True)

    # Dashboard Tabs
    tab_matches, tab_sources, tab_chart, tab_export = st.tabs([
        f"🎯 Evaluated Matches ({len(matches)})",
        f"🌐 Web Sources ({len(sources)})",
        "📈 Similarity Distribution",
        "📄 Export Report",
    ])

    with tab_matches:
        if not matches:
            st.info(
                f"ℹ️ **No passages exceeded the provisional suspicion threshold (50.0% combined score).** "
                f"The highest similarity detected across all retrieved web sources was **{highest_sim * 100:.1f}%**. "
                "You can inspect all evaluated candidate pairs below."
            )

        show_all = st.checkbox("Show all evaluated passage comparisons (including below-threshold candidates)", value=(len(matches) == 0))
        display_list = all_comparisons if show_all else matches

        if not display_list:
            st.write("No candidate comparisons available for display.")
        else:
            fc1, fc2 = st.columns([2, 2])
            with fc1:
                risk_filter = st.multiselect(
                    "Filter Risk Level",
                    options=["highly_suspicious", "likely_plagiarism", "possibly_similar", "no_strong_evidence"],
                    default=["highly_suspicious", "likely_plagiarism", "possibly_similar", "no_strong_evidence"] if show_all else ["highly_suspicious", "likely_plagiarism", "possibly_similar"],
                )
            with fc2:
                search_kw = st.text_input("Search passages by keyword", placeholder="Filter by text...")

            filtered = [
                m for m in display_list
                if (not risk_filter or m.get("risk") in risk_filter)
                and (not search_kw or search_kw.lower() in m.get("submitted_passage", "").lower() or search_kw.lower() in m.get("matched_passage", "").lower())
            ]

            st.caption(f"Displaying **{len(filtered)}** of {len(display_list)} candidate comparisons")

            for idx, match in enumerate(filtered, start=1):
                risk = match.get("risk", "no_strong_evidence")
                mtype = match.get("match_type", "low_similarity")
                sim = match.get("similarity", {})
                source = match.get("source", {})

                with st.expander(f"Candidate #{idx}: {risk.replace('_', ' ').title()} — {mtype} (Combined: {sim.get('combined', 0)*100:.1f}%)", expanded=(idx <= 2)):
                    # Header badges
                    hb1, hb2 = st.columns([2, 1])
                    with hb1:
                        st.markdown(f"{format_risk_badge(risk)} &nbsp; <span class=\"badge-type\">{mtype}</span>", unsafe_allow_html=True)
                    with hb2:
                        if source.get("url"):
                            st.markdown(f"**Source:** [{source.get('title', 'Web Article')}]({source.get('url')})")
                        else:
                            st.markdown(f"**Source:** {source.get('title', 'Unknown Source')}")

                    # Horizontal Similarity Progress Meters
                    m_s1, m_s2, m_s3 = st.columns(3)
                    with m_s1:
                        st.caption(f"Lexical (TF-IDF): **{sim.get('tfidf', 0)*100:.1f}%**")
                        st.progress(float(min(1.0, max(0.0, sim.get("tfidf", 0.0)))))
                    with m_s2:
                        st.caption(f"Semantic (SBERT): **{sim.get('sbert', 0)*100:.1f}%**")
                        st.progress(float(min(1.0, max(0.0, sim.get("sbert", 0.0)))))
                    with m_s3:
                        st.caption(f"Combined Score: **{sim.get('combined', 0)*100:.1f}%**")
                        st.progress(float(min(1.0, max(0.0, sim.get("combined", 0.0)))))

                    # Side-by-side comparison
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        st.markdown('<div class="passage-title">📝 Submitted Student Passage</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="passage-body">{match.get("submitted_passage")}</div>', unsafe_allow_html=True)
                        if match.get("page"):
                            st.caption(f"Document Page: {match.get('page')}")
                    with pc2:
                        st.markdown('<div class="passage-title">🌐 Matched Web Source Passage</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="passage-body source-body">{match.get("matched_passage")}</div>', unsafe_allow_html=True)

                    # Explainability Reasoning Callout
                    st.markdown(f'<div class="explanation-callout">💡 <b>Explanation:</b> {match.get("explanation")}</div>', unsafe_allow_html=True)

    with tab_sources:
        if not sources:
            st.info("No sources retrieved.")
        else:
            src_list = []
            for s in sources:
                src_list.append({
                    "Source Title": s.get("title"),
                    "URL": s.get("url"),
                    "Scrape Status": s.get("retrieval_status"),
                    "Relevance Score": f"{float(s.get('search_score', 0))*100:.1f}%",
                    "Notes": s.get("error") or "Clean text retrieved",
                })
            st.dataframe(pd.DataFrame(src_list), use_container_width=True)

    with tab_chart:
        target_chart_data = matches if matches else all_comparisons[:15]
        if target_chart_data:
            chart_rows = []
            for i, m in enumerate(target_chart_data, 1):
                sim = m.get("similarity", {})
                chart_rows.append({"Passage": f"P{i}", "Metric": "TF-IDF (Lexical)", "Score (%)": round(sim.get("tfidf", 0) * 100, 1)})
                chart_rows.append({"Passage": f"P{i}", "Metric": "SBERT (Semantic)", "Score (%)": round(sim.get("sbert", 0) * 100, 1)})
                chart_rows.append({"Passage": f"P{i}", "Metric": "Combined Score", "Score (%)": round(sim.get("combined", 0) * 100, 1)})
            df_plot = pd.DataFrame(chart_rows)
            st.bar_chart(df_plot, x="Passage", y="Score (%)", color="Metric", use_container_width=True)
        else:
            st.info("No comparisons available for plotting.")

    with tab_export:
        st.markdown("#### 💾 Export Analysis Artifacts")
        st.download_button(
            "📥 Download Full Evidence Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name=f"plagiarism_evidence_{int(time.time())}.json",
            mime="application/json",
            use_container_width=True,
        )
        st.json(report)


# -------------------------------------------------------------
# Section: Analysis History & Clear History Controls
# -------------------------------------------------------------
st.markdown("---")
h_top_col1, h_top_col2, h_top_col3 = st.columns([3, 1, 1])

with h_top_col1:
    st.markdown("### 📚 Analysis History")
    st.caption("Inspect, reload, or manage past plagiarism analyses persisted in the local SQLite database.")

with h_top_col2:
    if st.button("🔄 Refresh History", use_container_width=True):
        st.session_state.pop("history", None)

with h_top_col3:
    # Clear history button with confirmation popover/modal
    with st.popover("🗑️ Clear All History", use_container_width=True):
        st.markdown("⚠️ **Are you sure you want to clear all history?**")
        st.caption("This will delete all stored document analyses, matched passages, and source records from SQLite.")
        if st.button("Confirm Delete All", type="primary", use_container_width=True):
            del_payload, del_err = request_backend("DELETE", "/api/history")
            if del_err:
                st.error(del_err)
            else:
                st.success("All analysis history has been cleared!")
                st.session_state.pop("history", None)
                st.session_state.pop("latest_report", None)
                st.rerun()

# Load history
if "history" not in st.session_state:
    hist_payload, hist_err = request_backend("GET", "/api/history")
    if hist_err:
        st.caption(hist_err)
    else:
        st.session_state["history"] = hist_payload.get("analyses", [])

history_list = st.session_state.get("history", [])

if not history_list:
    st.info("No saved document analyses in history.")
else:
    for item in history_list:
        with st.container():
            col_id, col_name, col_risk, col_inspect, col_del = st.columns([1, 4, 3, 2, 1])
            col_id.markdown(f"**#{item.get('id')}**")
            col_name.markdown(f"📄 **{item.get('filename') or item.get('submitted_document')}**")
            col_risk.markdown(f"{format_risk_badge(item.get('overall_risk', 'no_strong_evidence'))}", unsafe_allow_html=True)
            
            if col_inspect.button(f"Inspect #{item.get('id')}", key=f"inspect_{item.get('id')}", use_container_width=True):
                with st.spinner("Fetching analysis record…"):
                    detail_res, detail_err = request_backend("GET", f"/api/history/{item.get('id')}")
                    if detail_err:
                        st.error(detail_err)
                    elif detail_res.get("analysis"):
                        st.session_state["latest_report"] = detail_res["analysis"]
                        st.rerun()
                        
            if col_del.button("🗑️", key=f"del_{item.get('id')}", help=f"Delete analysis #{item.get('id')}"):
                _, err = request_backend("DELETE", f"/api/history/{item.get('id')}")
                if err:
                    st.error(err)
                else:
                    st.session_state.pop("history", None)
                    st.rerun()
