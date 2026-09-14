"""Entrypoint for the read-only TFT MSc thesis dashboard."""

from __future__ import annotations

import streamlit as st

from dashboard.components.formatting import apply_global_style, render_sidebar


st.set_page_config(
    page_title="TFT MSc Thesis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("pages/1_Overview.py", title="Thesis overview", icon="📘", default=True),
    st.Page("pages/2_Feature_Families.py", title="Feature families", icon="🧩"),
    st.Page("pages/3_Model_Performance.py", title="Model performance", icon="📈"),
    st.Page("pages/4_Interpretability.py", title="Interpretability", icon="🔎"),
    st.Page("pages/5_Exact_Placement.py", title="Exact placement", icon="1️⃣"),
    st.Page("pages/6_Reproducibility.py", title="Reproducibility", icon="✅"),
    st.Page(
        "pages/7_Top_Four_Composition_Analysis.py",
        title="Top-four composition analysis",
        icon=":material/analytics:",
    ),
    st.Page(
        "pages/8_Composition_Evidence.py",
        title="Composition evidence",
        icon=":material/account_tree:",
    ),
    st.Page(
        "pages/9_Robustness_and_Limitations.py",
        title="Robustness and limitations",
        icon=":material/fact_check:",
    ),
]

navigation = st.navigation(pages, position="sidebar")
apply_global_style()
render_sidebar()
navigation.run()
