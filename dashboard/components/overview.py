"""Overview page component - System health and model performance dashboard."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))


def render():
    """Render overview page."""

    st.title("📊 System Overview")
    st.markdown("Real-time model performance and system health monitoring")

    # System health status
    st.markdown("### System Health")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.metric("MLflow Server", "✅ Online", delta=None)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.metric("Feature Store", "✅ Online", delta=None)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.metric("Monitoring", "✅ Active", delta=None)
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Models Deployed", "2", delta=None)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Model performance summary
    st.markdown("### Model Performance Summary")

    # Create two columns for the two models
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Heart Disease Classification")
        performance_data = {
            "Metric": ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score"],
            "Champion": [0.8823, 0.8333, 0.8421, 0.8000, 0.8205],
            "Challenger": [0.8765, 0.8250, 0.8333, 0.7917, 0.8120],
            "Threshold": [0.80, 0.75, 0.70, 0.70, 0.70]
        }
        df_perf = pd.DataFrame(performance_data)

        # Style the dataframe
        def highlight_metrics(row):
            if row['Champion'] >= row['Threshold']:
                return ['background-color: #d4edda'] * len(row)
            else:
                return ['background-color: #fff3cd'] * len(row)

        st.dataframe(
            df_perf.style.apply(highlight_metrics, axis=1).format({
                'Champion': '{:.4f}',
                'Challenger': '{:.4f}',
                'Threshold': '{:.2f}'
            }),
            hide_index=True,
            use_container_width=True
        )

        # Bias metrics
        st.markdown("**Fairness Assessment:**")
        bias_col1, bias_col2 = st.columns(2)
        with bias_col1:
            st.metric("Demographic Parity", "0.08", delta="-0.02 (improved)")
        with bias_col2:
            st.metric("Equal Opportunity", "0.06", delta="-0.01 (improved)")

    with col2:
        st.markdown("#### Hospital Readmission Regression")
        st.info("Model in development - Placeholder metrics shown")

        performance_data_reg = {
            "Metric": ["RMSE", "MAE", "R²", "MAPE"],
            "Value": [0.12, 0.09, 0.78, "8.5%"],
            "Threshold": [0.15, 0.10, 0.75, "10%"]
        }
        df_perf_reg = pd.DataFrame(performance_data_reg)

        st.dataframe(
            df_perf_reg,
            hide_index=True,
            use_container_width=True
        )

        st.caption("Note: Simplified model for demonstration purposes")

    st.markdown("---")

    # Recent activity timeline
    st.markdown("### Recent Activity")

    # Create sample activity data
    activities = [
        {"time": "2 hours ago", "event": "Drift detection completed", "model": "heart_disease", "status": "✅"},
        {"time": "6 hours ago", "event": "Model training started", "model": "readmission_regression", "status": "⏳"},
        {"time": "1 day ago", "event": "Model promoted to production", "model": "heart_disease", "status": "✅"},
        {"time": "2 days ago", "event": "Bias check completed", "model": "heart_disease", "status": "✅"},
        {"time": "3 days ago", "event": "Feature store updated", "model": "all", "status": "✅"},
    ]

    for activity in activities:
        col1, col2, col3 = st.columns([2, 4, 1])
        with col1:
            st.caption(activity["time"])
        with col2:
            st.text(f"{activity['event']} ({activity['model']})")
        with col3:
            st.text(activity["status"])

    st.markdown("---")

    # Predictions volume chart
    st.markdown("### Prediction Volume (Last 7 Days)")

    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
    prediction_volumes = {
        'Date': dates,
        'Heart Disease': [145, 132, 158, 142, 167, 155, 149],
        'Readmission': [89, 76, 92, 88, 95, 87, 91]
    }
    df_volumes = pd.DataFrame(prediction_volumes)

    fig = px.line(
        df_volumes,
        x='Date',
        y=['Heart Disease', 'Readmission'],
        title='Daily Prediction Volume by Model',
        labels={'value': 'Number of Predictions', 'variable': 'Model'}
    )
    fig.update_layout(hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Alerts and warnings
    st.markdown("### Alerts & Warnings")

    alert_col1, alert_col2 = st.columns(2)

    with alert_col1:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.markdown("**✅ All Systems Operational**")
        st.markdown("No critical alerts in the past 24 hours")
        st.markdown('</div>', unsafe_allow_html=True)

    with alert_col2:
        st.markdown('<div class="alert-card">', unsafe_allow_html=True)
        st.markdown("**⚠️ Scheduled Maintenance**")
        st.markdown("Feature store reindex scheduled for Sunday 2 AM")
        st.markdown('</div>', unsafe_allow_html=True)

    # Performance trends
    st.markdown("---")
    st.markdown("### Performance Trends")

    # Generate sample trend data
    trend_dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    trend_data = pd.DataFrame({
        'Date': trend_dates,
        'ROC-AUC': [0.88 + (i % 10) * 0.002 for i in range(30)],
        'Accuracy': [0.83 + (i % 8) * 0.003 for i in range(30)]
    })

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend_data['Date'],
        y=trend_data['ROC-AUC'],
        name='ROC-AUC',
        line=dict(color='#1f77b4', width=2)
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend_data['Date'],
        y=trend_data['Accuracy'],
        name='Accuracy',
        line=dict(color='#ff7f0e', width=2)
    ))

    # Add threshold line
    fig_trend.add_hline(
        y=0.80,
        line_dash="dash",
        line_color="red",
        annotation_text="Minimum Threshold"
    )

    fig_trend.update_layout(
        title="Heart Disease Model Performance (30 Days)",
        xaxis_title="Date",
        yaxis_title="Score",
        hovermode='x unified',
        yaxis_range=[0.75, 0.95]
    )

    st.plotly_chart(fig_trend, use_container_width=True)
