"""
FEAST Feature Definitions for Heart Disease Classification

This module defines the feature views and feature services for the heart disease model.
It follows FEAST best practices for offline training and online serving.
"""

from feast import (
    Entity,
    Feature,
    FeatureView,
    Field,
    FileSource,
    ValueType
)
from feast.types import Float32, Int64, String
from datetime import timedelta
from pathlib import Path

# Define the patient entity
patient = Entity(
    name="patient_id",
    join_keys=["patient_id"],
    value_type=ValueType.STRING,
    description="Unique patient identifier"
)

# Define data sources with absolute paths
base_data_path = Path(__file__).parent / "data" / "base_data" / "patients.parquet"
historical_data_path = Path(__file__).parent / "data" / "historical_data_sample" / "clinical_history.parquet"

# Convert to absolute paths to avoid issues when running from different directories
base_data_path = base_data_path.resolve()
historical_data_path = historical_data_path.resolve()

# Base patient demographics (static features)
base_patient_source = FileSource(
    name="base_patient_source",
    path=str(base_data_path),
    timestamp_field="event_timestamp"
)

# Historical clinical features (time-series)
historical_clinical_source = FileSource(
    name="historical_clinical_source",
    path=str(historical_data_path),
    timestamp_field="event_timestamp"
)

# Feature View: Patient Demographics
patient_demographics = FeatureView(
    name="patient_demographics",
    entities=[patient],
    ttl=timedelta(days=365),  # Demographics don't change frequently
    schema=[
        Field(name="age", dtype=Int64, description="Patient age in years"),
        Field(name="sex", dtype=Int64, description="Sex (0=female, 1=male)"),
    ],
    source=base_patient_source,
    tags={"team": "data_science", "model": "heart_disease"}
)

# Feature View: Clinical Measurements
clinical_measurements = FeatureView(
    name="clinical_measurements",
    entities=[patient],
    ttl=timedelta(days=90),  # Clinical measurements change over time
    schema=[
        Field(name="chest_pain_type", dtype=Int64, description="Chest pain type (1-4)"),
        Field(name="resting_bp", dtype=Int64, description="Resting blood pressure (mm Hg)"),
        Field(name="cholesterol", dtype=Int64, description="Serum cholesterol (mg/dl)"),
        Field(name="fasting_blood_sugar", dtype=Int64, description="Fasting blood sugar >120 mg/dl (0/1)"),
        Field(name="resting_ecg", dtype=Int64, description="Resting ECG results (0-2)"),
        Field(name="max_heart_rate", dtype=Int64, description="Maximum heart rate achieved"),
        Field(name="exercise_angina", dtype=Int64, description="Exercise induced angina (0/1)"),
        Field(name="st_depression", dtype=Float32, description="ST depression induced by exercise"),
        Field(name="slope", dtype=Int64, description="Slope of peak exercise ST segment (0-2)"),
        Field(name="vessels", dtype=Int64, description="Number of major vessels (0-3)"),
        Field(name="thalassemia", dtype=Int64, description="Thalassemia type (0-2)"),
    ],
    source=historical_clinical_source,
    tags={"team": "data_science", "model": "heart_disease"}
)

# Feature Service: Heart Disease Features
# This combines all feature views needed for the heart disease model
from feast import FeatureService

heart_disease_features = FeatureService(
    name="heart_disease_features",
    features=[
        patient_demographics,
        clinical_measurements
    ],
    tags={"model": "heart_disease", "version": "v1"}
)
