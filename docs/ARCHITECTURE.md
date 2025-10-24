# MLOps Architecture - Educational Guide

This document explains how the MLOps components work together to create a production ML system.

## System Overview

The architecture follows a standard ML lifecycle with six connected stages:

```
Data → Feature Store → Training → Model Registry → Inference → Monitoring
  ↓                                                              ↓
  └──────────────────── Feedback Loop ─────────────────────────┘
```

## Component Flow

### 1. Data Layer
**Purpose:** Generate or ingest raw data for ML

In this educational example, we generate synthetic healthcare data with `data_generator.py`.
In production, data comes from databases, APIs, data lakes, or streaming sources.

**Key Concept:** Separate raw data from features. Raw data is unstable and changes format.
Features are stable, versioned, and reusable across models.

### 2. Feature Store (FEAST)
**Purpose:** Centralize feature definitions for consistency

FEAST maintains two stores:
- **Offline Store**: Historical features for training (Parquet files or data warehouse)
- **Online Store**: Real-time features for inference (Redis, DynamoDB, or in-memory)

**Why It Matters:** Without a feature store, training and inference compute features differently,
causing "training-serving skew" where models perform well in training but poorly in production.

**Example:**
```python
# Feature definition (shared between training and inference)
age_group = Feature(
    name="age_group",
    dtype=Int64,
    description="Patient age bracket (0-4: 0, 5-17: 1, 18-64: 2, 65+: 3)"
)

# Training uses historical features
training_data = feast.get_historical_features(
    entity_df=patients,
    features=["patient_demographics:age_group", "vitals:blood_pressure"]
)

# Inference uses online features (low latency)
online_features = feast.get_online_features(
    entity_rows=[{"patient_id": "PT123"}],
    features=["patient_demographics:age_group", "vitals:blood_pressure"]
)
```

### 3. Training Pipeline
**Purpose:** Train models with hyperparameter optimization and validation

Steps:
1. Load features from FEAST offline store
2. Split into train/test (80/20)
3. Hyperparameter optimization with Hyperopt (tests 50 configurations)
4. Train XGBoost with best parameters
5. Evaluate metrics (ROC-AUC, precision, recall)
6. Check for bias across demographic groups
7. Log everything to MLflow

**Key Outputs:**
- Trained model artifact
- Performance metrics
- Hyperparameters used
- Training dataset hash (for lineage)
- Bias fairness scores

### 4. Model Registry (MLflow)
**Purpose:** Version control for models with deployment management

MLflow tracks:
- **Experiments**: Logical grouping of training runs
- **Runs**: Individual training executions with params/metrics
- **Models**: Registered versions with metadata
- **Aliases**: Deployment stages (champion, challenger, staging)

**Champion/Challenger Workflow:**
```
New Model Trained
     ↓
Register as Challenger
     ↓
Compare to Champion
     ↓
Improvement ≥ 2%? ── No → Archive
     ↓ Yes
Automated or Manual Approval
     ↓
Promote to Champion
     ↓
Old Champion → Archived
```

### 5. Inference Pipeline
**Purpose:** Generate predictions with explainability

Steps:
1. Load champion model from MLflow registry
2. Fetch real-time features from FEAST online store
3. Ensure feature order matches training
4. Generate prediction probability
5. Calculate SHAP values for explanation
6. Categorize risk level (low/medium/high)
7. Log inference for monitoring

**SHAP Explanations:**
SHAP (SHapley Additive exPlanations) shows which features contributed most to the prediction.

Example output:
```
Patient PT123: 73% heart disease risk

Top Contributing Factors:
1. cholesterol=280 → ↑ risk by +0.18
2. age=67 → ↑ risk by +0.12
3. max_heart_rate=95 → ↑ risk by +0.09
4. exercise_angina=1 → ↑ risk by +0.07
5. resting_bp=145 → ↑ risk by +0.04
```

### 6. Monitoring & Drift Detection
**Purpose:** Detect when models need retraining

Three drift types monitored:

**Dataset Drift**: Input features change distribution
- Example: Patient age average shifts from 55 to 62
- Detection: Statistical tests (Kolmogorov-Smirnov, Jensen-Shannon)
- Action: Review data pipeline, consider retraining

**Prediction Drift**: Model outputs change distribution
- Example: Model predicts 60% high-risk instead of historical 40%
- Detection: Compare prediction distributions over time
- Action: Investigate input changes, check model performance

**Target Drift**: Actual outcomes change distribution
- Example: Disease prevalence increases from 35% to 50%
- Detection: Compare label distributions (when available)
- Action: Urgent retraining needed (concept drift)

**Monitoring Schedule:**
- Weekly: Dataset drift check (compare last 7 days to training data)
- Monthly: Prediction drift analysis
- Quarterly: Bias fairness audit

## Scaling to Multiple Models

The framework supports 1 → 100+ models through:

1. **Consistent Structure**: All models in `src/models/{model_name}/`
2. **Shared Utilities**: Common code in `src/utils/`
3. **Configuration-Driven**: Model settings in `config.yaml`
4. **Automated CI/CD**: Workflows apply to all models

Adding a new model:
```bash
# 1. Copy template
cp -r src/models/heart_disease src/models/sepsis_prediction

# 2. Update config.yaml
models:
  sepsis_prediction:
    name: "sepsis_classifier"
    type: "classification"
    features: ["sepsis_features:*"]

# 3. Implement train.py and inference.py
# (All utilities automatically work)

# 4. CI/CD automatically picks up new model
```

## Technology Choices

| Component | Technology | Why? |
|-----------|-----------|------|
| Feature Store | FEAST | Open source, supports offline + online stores |
| Model Registry | MLflow | Industry standard, great UI, easy integration |
| Training | XGBoost | Fast, handles tabular data well, built-in feature importance |
| Optimization | Hyperopt | Bayesian optimization, more efficient than grid search |
| Explainability | SHAP | Theory-grounded, works with any model, visual outputs |
| Drift Detection | Evidently | Pre-built tests, generates HTML reports, low-code |
| Orchestration | GitHub Actions | Free for public repos, integrated with code |

## Production Deployment

Development (local):
- SQLite for MLflow
- Parquet files for FEAST offline store
- Local dict for FEAST online store

Production (cloud):
- PostgreSQL/MySQL for MLflow
- S3/BigQuery for FEAST offline store
- Redis/DynamoDB for FEAST online store
- Kubernetes for serving
- Prometheus/Grafana for monitoring

See [Dockerfile](../Dockerfile) for containerized deployment.

## Key Design Principles

1. **Separation of Concerns**: Feature engineering ≠ model training ≠ inference
2. **Version Everything**: Data, features, models, code all versioned
3. **Automate Testing**: Pre-commit hooks, CI/CD pipelines, drift detection
4. **Monitor Continuously**: Drift detection, bias checks, performance tracking
5. **Document Decisions**: Model cards, experiment tracking, audit logs

## Learning Resources

- Feature Stores: https://www.featurestore.org/
- MLOps Levels: https://ml-ops.org/content/mlops-principles
- ML System Design: "Designing Machine Learning Systems" by Chip Huyen
