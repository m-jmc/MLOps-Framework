"""Sidebar navigation component."""

import streamlit as st
from datetime import datetime


def render():
    """Render sidebar navigation."""

    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1f77b4/ffffff?text=Community+Hospital",
                 use_container_width=True)

        st.markdown("---")

        # Navigation menu
        st.markdown("### Navigation")
        page = st.radio(
            "Select a page:",
            [
                "Overview",
                "Make Prediction",
                "Model Monitoring",
                "Drift Analysis",
                "Governance & Audit"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Model selector (global state)
        st.markdown("### Active Model")
        model_name = st.selectbox(
            "Select model:",
            [
                "heart_disease",
                "readmission_regression"
            ],
            key="global_model_name"
        )

        # Model alias selector
        model_alias = st.selectbox(
            "Model version:",
            [
                "champion",
                "challenger",
                "latest"
            ],
            key="global_model_alias"
        )

        st.markdown("---")

        # Quick stats
        st.markdown("### Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Models", "2", delta=None)
        with col2:
            st.metric("Active", "2", delta=None)

        st.markdown("---")

        # System info
        st.markdown("### System Info")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.caption("Environment: Production")

        # Help section
        with st.expander("ℹ️ Help"):
            st.markdown("""
            **Quick Guide:**

            - **Overview**: System health and model performance
            - **Make Prediction**: Real-time model inference
            - **Monitoring**: Detailed metrics and trends
            - **Drift Analysis**: Data and prediction drift
            - **Governance**: Audit logs and compliance

            **Support**: `mlops@communityhospital.example.com`
            """)

    return page
