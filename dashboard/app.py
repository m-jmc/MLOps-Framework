"""
Community Hospital MLOps Dashboard

Main Streamlit application for monitoring models, making predictions,
and viewing governance metrics.

Modular design allows easy addition of new models and components.
"""

import sys
from pathlib import Path
import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import dashboard components
from src.dashboard.components import (
    sidebar,
    overview,
    prediction,
    monitoring,
    governance,
    drift_analysis
)

# Page configuration
st.set_page_config(
    page_title="Community Hospital MLOps",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main dashboard application."""

    # Header
    st.markdown('<div class="main-header">🏥 Community Hospital MLOps</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production ML Model Monitoring & Governance</div>', unsafe_allow_html=True)

    # Sidebar navigation
    page = sidebar.render()

    # Route to appropriate page
    if page == "Overview":
        overview.render()
    elif page == "Make Prediction":
        prediction.render()
    elif page == "Model Monitoring":
        monitoring.render()
    elif page == "Drift Analysis":
        drift_analysis.render()
    elif page == "Governance & Audit":
        governance.render()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        Community Hospital MLOps Platform v1.0.0 |
        <a href='https://github.com/yourusername/mlops_demonstration'>Documentation</a> |
        Built with MLflow, FEAST, Evidently, and Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
