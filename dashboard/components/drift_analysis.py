"""Drift analysis page component - Data and prediction drift monitoring."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np


def render():
    """Render drift analysis page."""

    st.title("📉 Drift Analysis")
    st.markdown("Monitor data drift and prediction distribution changes")

    # Get selected model
    model_name = st.session_state.get('global_model_name', 'heart_disease')

    st.info(f"Analyzing drift for: **{model_name}**")

    # Drift analysis tabs
    tab1, tab2, tab3 = st.tabs(["Dataset Drift", "Prediction Drift", "Feature Drift"])

    with tab1:
        render_dataset_drift(model_name)

    with tab2:
        render_prediction_drift(model_name)

    with tab3:
        render_feature_drift(model_name)


def render_dataset_drift(model_name):
    """Render dataset drift analysis."""

    st.markdown("### Dataset Drift Overview")

    # Overall drift status
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.metric("Dataset Drift", "No", delta=None)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.metric("Drift Score", "0.08", delta="-0.02 (improved)")

    with col3:
        st.metric("Drifted Features", "1 / 13", delta=None)

    with col4:
        st.metric("Last Check", "2 hours ago", delta=None)

    st.markdown("---")

    # Feature-level drift scores
    st.markdown("### Feature-Level Drift Scores")

    # Generate sample drift data
    features = [
        'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
        'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
        'exercise_angina', 'st_depression', 'slope', 'vessels', 'thalassemia'
    ]

    np.random.seed(42)
    drift_scores = [
        0.05, 0.03, 0.12, 0.08, 0.06,
        0.04, 0.05, 0.09, 0.03, 0.11,
        0.04, 0.05, 0.03
    ]

    drift_df = pd.DataFrame({
        'Feature': features,
        'Drift Score': drift_scores,
        'Status': ['⚠️ Drift' if s > 0.1 else '✅ OK' for s in drift_scores],
        'Threshold': [0.1] * len(features)
    })

    # Sort by drift score
    drift_df = drift_df.sort_values('Drift Score', ascending=False)

    # Horizontal bar chart
    fig = go.Figure()

    colors = ['#d62728' if s > 0.1 else '#2ca02c' for s in drift_df['Drift Score']]

    fig.add_trace(go.Bar(
        y=drift_df['Feature'],
        x=drift_df['Drift Score'],
        orientation='h',
        marker=dict(color=colors),
        text=drift_df['Drift Score'].round(3),
        textposition='outside'
    ))

    # Add threshold line
    fig.add_vline(
        x=0.1,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold"
    )

    fig.update_layout(
        title="Feature Drift Scores (Higher = More Drift)",
        xaxis_title="Drift Score (Wasserstein Distance)",
        yaxis_title="Feature",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed drift table
    st.markdown("### Detailed Drift Metrics")
    st.dataframe(
        drift_df.style.format({'Drift Score': '{:.4f}', 'Threshold': '{:.2f}'}),
        hide_index=True,
        use_container_width=True
    )

    # Distribution comparison for drifted feature
    st.markdown("### Distribution Comparison: chest_pain_type")
    st.caption("Comparing reference (training) vs current (production) distributions")

    col1, col2 = st.columns(2)

    with col1:
        # Reference distribution
        ref_dist = pd.DataFrame({
            'Type': ['Type 1', 'Type 2', 'Type 3', 'Type 4'],
            'Count': [120, 180, 150, 110]
        })

        fig_ref = px.bar(
            ref_dist,
            x='Type',
            y='Count',
            title='Reference Distribution (Training Data)',
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig_ref, use_container_width=True)

    with col2:
        # Current distribution (with drift)
        curr_dist = pd.DataFrame({
            'Type': ['Type 1', 'Type 2', 'Type 3', 'Type 4'],
            'Count': [100, 200, 170, 90]  # Shifted distribution
        })

        fig_curr = px.bar(
            curr_dist,
            x='Type',
            y='Count',
            title='Current Distribution (Production Data)',
            color_discrete_sequence=['#ff7f0e']
        )
        st.plotly_chart(fig_curr, use_container_width=True)


def render_prediction_drift(model_name):
    """Render prediction drift analysis."""

    st.markdown("### Prediction Drift Overview")

    # Prediction drift status
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.metric("Prediction Drift", "No", delta=None)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.metric("JS Divergence", "0.06", delta="-0.01")

    with col3:
        st.metric("Mean Shift", "0.02", delta="+0.01")

    st.markdown("---")

    # Prediction distribution over time
    st.markdown("### Prediction Distribution Trends")

    # Generate sample prediction data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    pred_data = []

    for date in dates:
        # Simulate prediction distribution
        np.random.seed(hash(date) % 2**32)
        predictions = np.random.beta(2, 3, 200)  # Skewed toward lower probabilities

        pred_data.append({
            'Date': date,
            'Mean Probability': predictions.mean(),
            'Std Probability': predictions.std(),
            'Positive Rate': (predictions > 0.5).mean()
        })

    pred_df = pd.DataFrame(pred_data)

    # Mean probability trend
    fig_mean = go.Figure()

    fig_mean.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Mean Probability'],
        name='Mean Probability',
        line=dict(color='#1f77b4', width=2),
        fill='tonexty'
    ))

    fig_mean.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Mean Probability'] + pred_df['Std Probability'],
        name='Mean + 1 SD',
        line=dict(color='#1f77b4', width=0.5, dash='dash'),
        showlegend=False
    ))

    fig_mean.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Mean Probability'] - pred_df['Std Probability'],
        name='Mean - 1 SD',
        line=dict(color='#1f77b4', width=0.5, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.2)',
        showlegend=False
    ))

    fig_mean.update_layout(
        title="Mean Prediction Probability Over Time",
        xaxis_title="Date",
        yaxis_title="Probability",
        hovermode='x unified'
    )

    st.plotly_chart(fig_mean, use_container_width=True)

    # Positive prediction rate
    fig_pos = px.line(
        pred_df,
        x='Date',
        y='Positive Rate',
        title='Positive Prediction Rate (Prob > 0.5)',
        labels={'Positive Rate': 'Rate'}
    )

    st.plotly_chart(fig_pos, use_container_width=True)

    # Prediction histogram comparison
    st.markdown("### Prediction Distribution: Reference vs Current")

    col1, col2 = st.columns(2)

    with col1:
        # Reference predictions
        np.random.seed(42)
        ref_preds = np.random.beta(2, 3, 1000)

        fig_ref = px.histogram(
            ref_preds,
            nbins=50,
            title='Reference Predictions (Training)',
            labels={'value': 'Probability', 'count': 'Frequency'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_ref.update_layout(showlegend=False)
        st.plotly_chart(fig_ref, use_container_width=True)

    with col2:
        # Current predictions (slight shift)
        np.random.seed(43)
        curr_preds = np.random.beta(2.2, 3, 1000)  # Slight drift

        fig_curr = px.histogram(
            curr_preds,
            nbins=50,
            title='Current Predictions (Last 7 Days)',
            labels={'value': 'Probability', 'count': 'Frequency'},
            color_discrete_sequence=['#ff7f0e']
        )
        fig_curr.update_layout(showlegend=False)
        st.plotly_chart(fig_curr, use_container_width=True)

    # Statistical tests
    st.markdown("### Statistical Tests")

    test_results = pd.DataFrame({
        'Test': ['Kolmogorov-Smirnov', 'Chi-Square', 'Jensen-Shannon Divergence'],
        'Statistic': [0.042, 3.21, 0.06],
        'P-Value': [0.34, 0.52, None],
        'Result': ['No Drift', 'No Drift', 'No Drift']
    })

    st.dataframe(test_results, hide_index=True, use_container_width=True)


def render_feature_drift(model_name):
    """Render individual feature drift analysis."""

    st.markdown("### Individual Feature Analysis")

    # Feature selector
    feature = st.selectbox(
        "Select feature to analyze:",
        [
            'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
            'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
            'exercise_angina', 'st_depression', 'slope', 'vessels', 'thalassemia'
        ]
    )

    st.markdown(f"### Feature: `{feature}`")

    # Generate sample distributions
    np.random.seed(hash(feature) % 2**32)

    if feature in ['age', 'resting_bp', 'cholesterol', 'max_heart_rate', 'st_depression']:
        # Continuous features
        ref_data = np.random.normal(55, 10, 1000) if feature == 'age' else np.random.normal(120, 15, 1000)
        curr_data = ref_data + np.random.normal(2, 5, 1000)  # Slight shift

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                ref_data,
                nbins=30,
                title=f'Reference Distribution: {feature}',
                color_discrete_sequence=['#1f77b4']
            )
            fig.add_vline(x=ref_data.mean(), line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {ref_data.mean():.1f}")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                curr_data,
                nbins=30,
                title=f'Current Distribution: {feature}',
                color_discrete_sequence=['#ff7f0e']
            )
            fig.add_vline(x=curr_data.mean(), line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {curr_data.mean():.1f}")
            st.plotly_chart(fig, use_container_width=True)

        # Summary statistics
        st.markdown("### Summary Statistics")

        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
            'Reference': [
                ref_data.mean(),
                np.median(ref_data),
                ref_data.std(),
                ref_data.min(),
                ref_data.max()
            ],
            'Current': [
                curr_data.mean(),
                np.median(curr_data),
                curr_data.std(),
                curr_data.min(),
                curr_data.max()
            ],
            'Difference': [
                curr_data.mean() - ref_data.mean(),
                np.median(curr_data) - np.median(ref_data),
                curr_data.std() - ref_data.std(),
                curr_data.min() - ref_data.min(),
                curr_data.max() - ref_data.max()
            ]
        })

        st.dataframe(
            stats_df.style.format({
                'Reference': '{:.2f}',
                'Current': '{:.2f}',
                'Difference': '{:.2f}'
            }),
            hide_index=True,
            use_container_width=True
        )

    else:
        # Categorical features
        categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
        ref_counts = [30, 35, 20, 15]
        curr_counts = [28, 38, 22, 12]  # Slight shift

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                x=categories,
                y=ref_counts,
                title=f'Reference Distribution: {feature}',
                labels={'x': 'Category', 'y': 'Count'},
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                x=categories,
                y=curr_counts,
                title=f'Current Distribution: {feature}',
                labels={'x': 'Category', 'y': 'Count'},
                color_discrete_sequence=['#ff7f0e']
            )
            st.plotly_chart(fig, use_container_width=True)

    # Drift score
    st.markdown("### Drift Metrics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Drift Score", "0.08", delta=None)
    with col2:
        st.metric("Threshold", "0.10", delta=None)
    with col3:
        st.markdown('<div class="success-card">', unsafe_allow_html=True)
        st.metric("Status", "✅ No Drift", delta=None)
        st.markdown('</div>', unsafe_allow_html=True)
