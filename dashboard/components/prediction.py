"""Prediction page component - Real-time model inference with explanations."""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.graph_objects as go

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))


def render():
    """Render prediction page."""

    st.title("🔮 Make Prediction")
    st.markdown("Real-time model inference with SHAP explanations")

    # Get selected model from global state
    model_name = st.session_state.get('global_model_name', 'heart_disease')
    model_alias = st.session_state.get('global_model_alias', 'champion')

    st.info(f"Using model: **{model_name}** (version: {model_alias})")

    # Route to appropriate prediction interface
    if model_name == "heart_disease":
        render_heart_disease_prediction()
    elif model_name == "readmission_regression":
        render_readmission_prediction()


def render_heart_disease_prediction():
    """Render heart disease prediction interface."""

    st.markdown("### Patient Information")

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Demographics")
        age = st.slider("Age (years)", 20, 100, 55)
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])

        st.markdown("#### Clinical Measurements")
        chest_pain_type = st.selectbox(
            "Chest Pain Type",
            options=[
                (1, "Typical Angina"),
                (2, "Atypical Angina"),
                (3, "Non-anginal Pain"),
                (4, "Asymptomatic")
            ],
            format_func=lambda x: x[1]
        )

        resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 200)

        fasting_blood_sugar = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[(0, "No"), (1, "Yes")],
            format_func=lambda x: x[1]
        )

    with col2:
        st.markdown("#### Exercise & ECG")

        resting_ecg = st.selectbox(
            "Resting ECG Results",
            options=[
                (0, "Normal"),
                (1, "ST-T Wave Abnormality"),
                (2, "Left Ventricular Hypertrophy")
            ],
            format_func=lambda x: x[1]
        )

        max_heart_rate = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)

        exercise_angina = st.selectbox(
            "Exercise Induced Angina",
            options=[(0, "No"), (1, "Yes")],
            format_func=lambda x: x[1]
        )

        st_depression = st.slider("ST Depression (induced by exercise)", 0.0, 6.0, 1.0, 0.1)

        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            options=[
                (0, "Upsloping"),
                (1, "Flat"),
                (2, "Downsloping")
            ],
            format_func=lambda x: x[1]
        )

        vessels = st.slider("Number of Major Vessels (0-3)", 0, 3, 0)

        thalassemia = st.selectbox(
            "Thalassemia",
            options=[
                (0, "Normal"),
                (1, "Fixed Defect"),
                (2, "Reversible Defect")
            ],
            format_func=lambda x: x[1]
        )

    # Predict button
    if st.button("🔍 Predict Risk", type="primary", use_container_width=True):
        # Collect input data
        patient_data = {
            'age': age,
            'sex': sex[1],
            'chest_pain_type': chest_pain_type[0],
            'resting_bp': resting_bp,
            'cholesterol': cholesterol,
            'fasting_blood_sugar': fasting_blood_sugar[1],
            'resting_ecg': resting_ecg[0],
            'max_heart_rate': max_heart_rate,
            'exercise_angina': exercise_angina[1],
            'st_depression': st_depression,
            'slope': slope[0],
            'vessels': vessels,
            'thalassemia': thalassemia[0]
        }

        # Make prediction (simulated for demo)
        prediction_result = simulate_heart_disease_prediction(patient_data)

        # Display results
        st.markdown("---")
        st.markdown("### Prediction Results")

        # Risk level card
        risk_level = prediction_result['risk_level']
        probability = prediction_result['probability']

        if risk_level == "High":
            card_class = "alert-card"
            emoji = "🔴"
        elif risk_level == "Medium":
            card_class = "alert-card"
            emoji = "🟡"
        else:
            card_class = "success-card"
            emoji = "🟢"

        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"<div style='font-size: 3rem;'>{emoji}</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**Risk Level:** {risk_level}")
            st.markdown(f"**Probability:** {probability:.1%}")
            st.progress(probability)
        st.markdown('</div>', unsafe_allow_html=True)

        # Explanation
        st.markdown("### Feature Importance Explanation")
        st.markdown("Top contributing features to this prediction:")

        # Display SHAP-like feature contributions
        explanation_df = pd.DataFrame(prediction_result['explanation'])
        explanation_df = explanation_df.sort_values('contribution', key=abs, ascending=False)

        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=explanation_df['contribution'],
            y=explanation_df['feature'],
            orientation='h',
            marker=dict(
                color=explanation_df['contribution'],
                colorscale='RdYlGn_r',
                showscale=True
            )
        ))
        fig.update_layout(
            title="Feature Contributions (SHAP-like values)",
            xaxis_title="Impact on Prediction",
            yaxis_title="Feature",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Clinical recommendation
        st.markdown("### Clinical Recommendation")
        st.info(prediction_result['recommendation'])


def render_readmission_prediction():
    """Render hospital readmission prediction interface."""

    st.markdown("### Patient Admission Information")
    st.warning("Readmission model is in development - Simplified interface shown")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age (years)", 18, 100, 65)
        los = st.slider("Length of Stay (days)", 1, 30, 5)
        num_procedures = st.slider("Number of Procedures", 0, 10, 2)

    with col2:
        num_medications = st.slider("Number of Medications", 0, 50, 10)
        num_diagnoses = st.slider("Number of Diagnoses", 1, 16, 5)
        admission_type = st.selectbox("Admission Type", ["Emergency", "Urgent", "Elective"])

    if st.button("🔍 Predict Readmission Risk", type="primary", use_container_width=True):
        # Simulate prediction
        risk_score = np.random.beta(2, 5)  # Skewed toward lower risk

        st.markdown("---")
        st.markdown("### Prediction Results")

        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"**30-Day Readmission Risk:** {risk_score:.1%}")
        st.progress(risk_score)
        st.markdown('</div>', unsafe_allow_html=True)

        st.info("Note: This is a placeholder prediction from the simplified model")


def simulate_heart_disease_prediction(patient_data):
    """Simulate heart disease prediction with SHAP-like explanation."""

    # Simple risk score calculation for demo
    risk_factors = {
        'age': (patient_data['age'] - 20) / 80,
        'sex': patient_data['sex'] * 0.15,
        'chest_pain_type': (patient_data['chest_pain_type'] - 1) / 3 * 0.2,
        'resting_bp': max(0, (patient_data['resting_bp'] - 120) / 80),
        'cholesterol': max(0, (patient_data['cholesterol'] - 200) / 200),
        'fasting_blood_sugar': patient_data['fasting_blood_sugar'] * 0.1,
        'max_heart_rate': max(0, (180 - patient_data['max_heart_rate']) / 120),
        'exercise_angina': patient_data['exercise_angina'] * 0.2,
        'st_depression': patient_data['st_depression'] / 6,
        'vessels': patient_data['vessels'] / 3 * 0.25,
        'thalassemia': patient_data['thalassemia'] / 2 * 0.15
    }

    # Calculate probability
    base_risk = 0.3
    additional_risk = sum(risk_factors.values()) * 0.5
    probability = min(0.95, base_risk + additional_risk)

    # Determine risk level
    if probability >= 0.7:
        risk_level = "High"
        recommendation = "⚠️ High risk of heart disease detected. Immediate cardiology consultation recommended. Consider additional diagnostic testing including stress test and echocardiogram."
    elif probability >= 0.4:
        risk_level = "Medium"
        recommendation = "⚡ Moderate risk identified. Schedule follow-up appointment within 2 weeks. Monitor blood pressure and cholesterol levels. Lifestyle modifications recommended."
    else:
        risk_level = "Low"
        recommendation = "✅ Low risk profile. Continue routine health monitoring. Maintain healthy lifestyle with regular exercise and balanced diet."

    # Create feature importance explanation
    feature_names = {
        'age': 'Age',
        'sex': 'Gender',
        'chest_pain_type': 'Chest Pain Type',
        'resting_bp': 'Resting Blood Pressure',
        'cholesterol': 'Cholesterol',
        'fasting_blood_sugar': 'Fasting Blood Sugar',
        'max_heart_rate': 'Max Heart Rate',
        'exercise_angina': 'Exercise Angina',
        'st_depression': 'ST Depression',
        'vessels': 'Major Vessels',
        'thalassemia': 'Thalassemia'
    }

    # Convert risk factors to SHAP-like contributions (centered around 0)
    contributions = []
    for key, value in risk_factors.items():
        # Convert to contribution (positive increases risk, negative decreases)
        contribution = (value - 0.1) * 0.2  # Normalize
        contributions.append({
            'feature': feature_names[key],
            'value': patient_data[key],
            'contribution': contribution
        })

    return {
        'probability': probability,
        'risk_level': risk_level,
        'recommendation': recommendation,
        'explanation': contributions
    }
