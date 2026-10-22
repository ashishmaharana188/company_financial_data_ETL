"""Unified Streamlit entry point for the financial ETL and TrendAnalysis UIs.

Run from the repository root with:
    python -m streamlit run unified_app.py

The existing dashboards remain separate pages. This intentionally avoids a
large refactor while giving both pages a single navigation shell.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Financial Analytics Platform",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

financial_page = st.Page(
    "dashboard.py",
    title="Financial Data Center",
    icon=":material/database:",
    default=True,
)
market_page = st.Page(
    "app/main.py",
    title="Market Research Terminal",
    icon=":material/query_stats:",
)

pages = st.navigation(
    {"Financial Platform": [financial_page, market_page]},
    position="sidebar",
)

with st.sidebar:
    st.markdown("### Financial Analytics")
    st.caption("Fundamentals · Markets · Quantitative Research")
    st.divider()

pages.run()
