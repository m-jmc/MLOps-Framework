"""Monitoring page component - Detailed metrics and performance tracking."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def render():
    """Render monitoring page."""

    st.title("📈 Model Monitoring")
    st.markdown("Detailed performance metrics and trend analysis")

    # Get selected model
    model_name = st.session_state.get('global_model_name', 'heart_disease')
    model_alias = st.session_state.get('global_model_alias', 'champion')

    st.info(f"Monitoring: **{model_name}** (version: {model_alias})")

    # Time range selector
    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        time_range = st.selectbox(
            "Time Range",
            ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year"]
        )
    with col2:
        metric_type = st.selectbox(
            "Metric Category",
            ["Performance", "Prediction Volume", "Latency", "Errors"]
        )

    st.markdown("---")

    # Performance metrics over time
    if metric_type == "Performance":
        render_performance_metrics(model_name, time_range)
    elif metric_type == "Prediction Volume":
        render_prediction_volume(model_name, time_range)
    elif metric_type == "Latency":
        render_latency_metrics(model_name, time_range)
    elif metric_type == "Errors":
        render_error_metrics(model_name, time_range)


def render_performance_metrics(model_name, time_range):
    """Render performance metrics charts."""

    st.markdown("### Performance Metrics Over Time")

    # Generate sample performance data
    days = get_days_from_range(time_range)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Simulate metrics with slight variation
    np.random.seed(42)
    metrics_data = pd.DataFrame({
        'Date': dates,
        'ROC-AUC': 0.88 + np.random.normal(0, 0.01, days),
        'Accuracy': 0.83 + np.random.normal(0, 0.015, days),
        'Precision': 0.84 + np.random.normal(0, 0.012, days),
        'Recall': 0.80 + np.random.normal(0, 0.018, days),
        'F1-Score': 0.82 + np.random.normal(0, 0.013, days)
    })

    # Clip values to realistic ranges
    for col in ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']:
        metrics_data[col] = metrics_data[col].clip(0.70, 0.95)

    # Create multi-line chart
    fig = go.Figure()

    colors = {
        'ROC-AUC': '#1f77b4',
        'Accuracy': '#ff7f0e',
        'Precision': '#2ca02c',
        'Recall': '#d62728',
        'F1-Score': '#9467bd'
    }

    for metric in ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']:
        fig.add_trace(go.Scatter(
            x=metrics_data['Date'],
            y=metrics_data[metric],
            name=metric,
            line=dict(color=colors[metric], width=2),
            mode='lines+markers'
        ))

    # Add threshold line
    fig.add_hline(
        y=0.75,
        line_dash="dash",
        line_color="red",
        annotation_text="Minimum Threshold",
        annotation_position="right"
    )

    fig.update_layout(
        title=f"{model_name.replace('_', ' ').title()} Performance Metrics",
        xaxis_title="Date",
        yaxis_title="Score",
        hovermode='x unified',
        yaxis_range=[0.68, 0.95],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Current metrics table
    st.markdown("### Current Metrics Summary")

    current_metrics = {
        'Metric': ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC'],
        'Value': [0.8823, 0.8333, 0.8421, 0.8000, 0.8205, 0.6667],
        '7-Day Avg': [0.8801, 0.8310, 0.8405, 0.7985, 0.8190, 0.6650],
        '30-Day Avg': [0.8790, 0.8295, 0.8390, 0.7970, 0.8175, 0.6635],
        'Trend': ['↑', '↑', '↑', '↑', '↑', '↑']
    }
    df_metrics = pd.DataFrame(current_metrics)

    st.dataframe(
        df_metrics.style.format({
            'Value': '{:.4f}',
            '7-Day Avg': '{:.4f}',
            '30-Day Avg': '{:.4f}'
        }),
        hide_index=True,
        use_container_width=True
    )

    # Confusion matrix heatmap
    st.markdown("### Confusion Matrix (Latest Week)")

    col1, col2 = st.columns(2)

    with col1:
        # Create confusion matrix data
        cm = np.array([[85, 15], [20, 80]])

        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Negative', 'Predicted Positive'],
            y=['Actual Negative', 'Actual Positive'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            showscale=True
        ))

        fig_cm.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class"
        )

        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        # Metrics breakdown
        st.markdown("**Classification Metrics:**")
        st.metric("True Positives", "80")
        st.metric("True Negatives", "85")
        st.metric("False Positives", "15")
        st.metric("False Negatives", "20")
        st.markdown("---")
        st.metric("Total Predictions", "200")
        st.metric("Accuracy", "82.5%")


def render_prediction_volume(model_name, time_range):
    """Render prediction volume charts."""

    st.markdown("### Prediction Volume Analysis")

    days = get_days_from_range(time_range)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Generate volume data with weekly pattern
    np.random.seed(42)
    base_volume = 150
    weekly_pattern = [1.0, 0.9, 0.95, 0.92, 0.98, 0.7, 0.6]  # Lower on weekends
    volumes = [
        int(base_volume * weekly_pattern[i % 7] + np.random.normal(0, 10))
        for i in range(days)
    ]

    volume_data = pd.DataFrame({
        'Date': dates,
        'Predictions': volumes
    })

    # Daily volume chart
    fig = px.bar(
        volume_data,
        x='Date',
        y='Predictions',
        title=f"Daily Prediction Volume - {model_name.replace('_', ' ').title()}"
    )
    fig.update_layout(hovermode='x')
    st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Predictions", f"{sum(volumes):,}")
    with col2:
        st.metric("Daily Average", f"{int(np.mean(volumes)):,}")
    with col3:
        st.metric("Peak Day", f"{max(volumes):,}")
    with col4:
        st.metric("Min Day", f"{min(volumes):,}")

    # Hourly distribution (for recent day)
    st.markdown("### Hourly Distribution (Today)")

    hours = list(range(24))
    hourly_volumes = [
        int(20 + 30 * np.sin((h - 6) * np.pi / 12) + np.random.normal(0, 5))
        if 6 <= h <= 18 else int(5 + np.random.normal(0, 2))
        for h in hours
    ]

    fig_hourly = px.bar(
        x=hours,
        y=hourly_volumes,
        labels={'x': 'Hour of Day', 'y': 'Predictions'},
        title='Prediction Volume by Hour'
    )
    st.plotly_chart(fig_hourly, use_container_width=True)


def render_latency_metrics(model_name, time_range):
    """Render latency metrics charts."""

    st.markdown("### Prediction Latency Analysis")

    days = get_days_from_range(time_range)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Generate latency data
    np.random.seed(42)
    latency_data = pd.DataFrame({
        'Date': dates,
        'P50 (ms)': 45 + np.random.normal(0, 5, days),
        'P95 (ms)': 85 + np.random.normal(0, 10, days),
        'P99 (ms)': 120 + np.random.normal(0, 15, days),
    })

    # Latency trends
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=latency_data['Date'],
        y=latency_data['P50 (ms)'],
        name='P50',
        line=dict(color='#2ca02c', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=latency_data['Date'],
        y=latency_data['P95 (ms)'],
        name='P95',
        line=dict(color='#ff7f0e', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=latency_data['Date'],
        y=latency_data['P99 (ms)'],
        name='P99',
        line=dict(color='#d62728', width=2)
    ))

    # SLA threshold
    fig.add_hline(
        y=150,
        line_dash="dash",
        line_color="red",
        annotation_text="SLA Threshold (150ms)"
    )

    fig.update_layout(
        title="Prediction Latency Percentiles",
        xaxis_title="Date",
        yaxis_title="Latency (ms)",
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Current latency metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("P50 Latency", "45ms", delta="-2ms")
    with col2:
        st.metric("P95 Latency", "85ms", delta="-5ms")
    with col3:
        st.metric("P99 Latency", "120ms", delta="-8ms")
    with col4:
        st.metric("SLA Compliance", "99.8%", delta="+0.1%")


def render_error_metrics(model_name, time_range):
    """Render error metrics charts."""

    st.markdown("### Error Analysis")

    days = get_days_from_range(time_range)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Generate error data (low error rate)
    np.random.seed(42)
    error_data = pd.DataFrame({
        'Date': dates,
        'Errors': np.random.poisson(2, days),  # Low error rate
        'Total Requests': [150 + np.random.randint(-20, 20) for _ in range(days)]
    })

    error_data['Error Rate (%)'] = (error_data['Errors'] / error_data['Total Requests'] * 100)

    # Error rate chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=error_data['Date'],
        y=error_data['Error Rate (%)'],
        name='Error Rate',
        line=dict(color='#d62728', width=2),
        fill='tozeroy',
        fillcolor='rgba(214, 39, 40, 0.2)'
    ))

    fig.update_layout(
        title="Prediction Error Rate Over Time",
        xaxis_title="Date",
        yaxis_title="Error Rate (%)",
        hovermode='x'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Error type breakdown
    st.markdown("### Error Type Distribution (Last 7 Days)")

    error_types = pd.DataFrame({
        'Error Type': ['Timeout', 'Invalid Input', 'Model Load Failure', 'Feature Store Error', 'Other'],
        'Count': [5, 12, 2, 3, 4],
        'Percentage': [19.2, 46.2, 7.7, 11.5, 15.4]
    })

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_pie = px.pie(
            error_types,
            values='Count',
            names='Error Type',
            title='Errors by Type'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.dataframe(error_types, hide_index=True, use_container_width=True)

    # Overall error metrics
    st.markdown("### Error Metrics Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Errors (7d)", "26")
    with col2:
        st.metric("Avg Error Rate", "1.5%")
    with col3:
        st.metric("Max Daily Errors", "5")
    with col4:
        st.metric("Error-Free Days", "3/7")


def get_days_from_range(time_range):
    """Convert time range string to number of days."""
    if time_range == "Last 7 Days":
        return 7
    elif time_range == "Last 30 Days":
        return 30
    elif time_range == "Last 90 Days":
        return 90
    elif time_range == "Last Year":
        return 365
    return 7
