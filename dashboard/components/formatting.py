"""Academic visual styling and shared Streamlit framing."""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard.components.loaders import (
    load_analysis_contract,
    load_dataset_summary,
    load_selection_manifest,
)
from dashboard.components.validation import DashboardDataError


def apply_global_style() -> None:
    st.markdown(
        """
        <style>
        :root { --ink: #17243a; --muted: #5d6b7f; --accent: #315b7d; }
        .block-container { max-width: 1320px; padding-top: 2.2rem; padding-bottom: 3rem; }
        h1, h2, h3 { color: var(--ink); letter-spacing: -0.018em; }
        h1 { font-weight: 650; }
        [data-testid="stMetric"] {
          border: 1px solid #dfe5eb; border-radius: 10px; padding: 0.8rem 1rem;
          background: #fbfcfd;
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        .method-note {
          border-left: 4px solid #315b7d; background: #f1f5f8; padding: .8rem 1rem;
          border-radius: 0 8px 8px 0; color: #26384d;
        }
        .secondary-note {
          border-left: 4px solid #8a6d3b; background: #fbf8f1; padding: .8rem 1rem;
          border-radius: 0 8px 8px 0; color: #4c3c20;
        }
        .workflow {
          border: 1px solid #dfe5eb; border-radius: 10px; padding: 1rem;
          background: #fbfcfd; line-height: 2.1;
        }
        .caption { color: var(--muted); font-size: .91rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _summary_value(measures: dict[str, Any], key: str, fallback: str = "Unavailable") -> str:
    value = measures.get(key)
    if value is None:
        return fallback
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def render_sidebar() -> None:
    st.sidebar.markdown("### Study context")
    try:
        summary = load_dataset_summary()
        measures = dict(zip(summary["measure"], summary["value"], strict=False))
        contract = load_analysis_contract()
        selection = load_selection_manifest()
        st.sidebar.markdown(
            f"""
            **Patch:** {contract['dataset']['patch']}  
            **Dataset:** {_summary_value(measures, 'accepted_matches')} matches  
            **Primary task:** Top-four classification  
            **Locked model:** {str(selection['selected_model']).upper()}
            """
        )
    except (DashboardDataError, KeyError, TypeError, ValueError):
        st.sidebar.warning("Study metadata is unavailable from the frozen artifacts.")
    st.sidebar.divider()
    st.sidebar.caption(
        "Read-only post-modelling dashboard. No training, tuning, model selection, "
        "or prediction generation occurs here."
    )


def artifact_warning(error: Exception, label: str) -> None:
    st.warning(f"{label} is unavailable. {error}")


def section_kicker(text: str) -> None:
    st.caption(text.upper())


def method_note(text: str, secondary: bool = False) -> None:
    css_class = "secondary-note" if secondary else "method-note"
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)

