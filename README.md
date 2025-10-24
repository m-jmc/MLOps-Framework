# MLOps Framework - Educational Reference

> A simplified, educational MLOps platform demonstrating production ML patterns through clear examples and narrative explanations.

## What This Teaches

This repository demonstrates six core MLOps capabilities in a healthcare context:

1. **Feature Store** - Consistent feature serving between training and inference using FEAST
2. **Model Training** - Hyperparameter optimization with XGBoost and Hyperopt
3. **Model Registry** - Version control and champion/challenger patterns with MLflow
4. **Inference Pipeline** - Real-time predictions with SHAP explanations
5. **Drift Detection** - Monitoring for data and prediction drift with Evidently
6. **Governance** - Bias detection and approval workflows

## Quick Start

```bash
# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Explore the educational code
# All code files are simplified pseudo-code with extensive comments
python src/models/heart_disease/train.py
python src/models/heart_disease/inference.py
```

## Learning Path

The code is structured as a teaching tool. Start here:

1. [README.md](README.md) - Overview (you are here)
2. [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design and data flow
3. [CICD_AND_PRECOMMIT.md](CICD_AND_PRECOMMIT.md) - Testing and automation concepts
4. [train.py](src/models/heart_disease/train.py) - Training pipeline walkthrough
5. [inference.py](src/models/heart_disease/inference.py) - Prediction pipeline
6. [mlflow_utils.py](src/utils/mlflow_utils.py) - Model lifecycle management
7. [drift_detector.py](src/monitoring/drift_detector.py) - Model monitoring

## Repository Structure

```
MLOps-Framework/
├── src/
│   ├── models/
│   │   ├── diabetes_score/          # Primary classification model
│   │   ├── heart_disease/         # Second classification model
│   │   └── readmission_regression/ # Third model (regression)
│   ├── utils/
│   │   ├── mlflow_utils.py        # Model registry operations
│   │   ├── feast_utils.py         # Feature store operations
│   │   └── drift_utils.py         # Monitoring utilities
│   ├── monitoring/
│   │   └── drift_detector.py      # Drift detection logic
│   └── governance/
│       ├── audit_logger.py        # Audit trail
│       └── model_approval.py      # Approval workflows
├── docs/
│   └── ARCHITECTURE.md            # System design
├── config.yaml                    # Central configuration
├── Dockerfile                     # Cloud deployment
└── CICD_AND_PRECOMMIT.md         # Testing & automation
```

## Core Concepts Demonstrated

### Feature Store (FEAST)
Ensures training and serving use identical features. Solves the "training-serving skew" problem.
Training uses historical features, inference uses real-time features, but both come from the same
definitions to guarantee consistency.

### Model Registry (MLflow)
Versions models like code. Uses "champion/challenger" pattern: production always uses the champion
alias, new models are registered as challengers, and promotion happens after validation.

### Champion/Challenger Pattern
- Champion: Current production model
- Challenger: New model that outperforms champion in testing
- Promotion: Automated comparison triggers promotion if improvement > threshold (e.g., 2%)
- Rollback: Champion alias can be moved back to previous version if needed

### Drift Detection (Evidently)
Monitors for three drift types:
- Dataset drift: Input features change (e.g., patient demographics shift)
- Prediction drift: Model outputs change (e.g., predicting more high-risk cases)
- Target drift: Actual outcomes change (e.g., disease prevalence increases)

### Bias Detection
Checks model fairness across demographic groups (sex, age). Calculates metrics like demographic
parity (equal positive prediction rates) and equal opportunity (equal true positive rates).

## Extending to Additional Models

The framework is designed for multi-model deployment:

1. Copy model template: `cp -r src/models/heart_disease src/models/new_model`
2. Update configuration in `config.yaml`
3. Implement model-specific logic in `train.py` and `inference.py`
4. Shared utilities (mlflow_utils, feast_utils) work across all models
5. CI/CD pipelines automatically apply to new models

Example multi-model config:

```yaml
models:
  heart_disease:
    name: "heart_disease_classifier"
    type: "classification"

  diabetes_risk:
    name: "diabetes_score"
    type: "classification"

  readmission:
    name: "readmission_regression"
    type: "regression"
```

## Technology Stack

- **FEAST**: Feature store (offline for training, online for serving)
- **MLflow**: Model registry and experiment tracking
- **XGBoost**: Gradient boosting for classification/regression
- **Hyperopt**: Hyperparameter optimization
- **SHAP**: Model explainability (feature importance)
- **Evidently**: Drift detection and monitoring
- **GitHub Actions**: CI/CD automation

## Key Design Patterns

1. **Configuration-Driven**: All settings in `config.yaml`, no hardcoded values
2. **Shared Utilities**: Common operations (MLflow, FEAST) abstracted to utils
3. **Consistent Structure**: All models follow same directory layout
4. **Pseudo-Code Style**: Code simplified for educational clarity
5. **Extensive Documentation**: Every function explains the "why" not just "what"

## Deployment

See [Dockerfile](Dockerfile) for containerized deployment to cloud platforms (AWS, Azure, GCP).

The Docker container includes:
- Python environment with all dependencies
- MLflow tracking server
- FEAST feature server
- Model serving API
- Monitoring dashboards

## Learning Resources

- MLflow Docs: https://mlflow.org/docs/latest/index.html
- FEAST Docs: https://docs.feast.dev/
- Evidently Docs: https://docs.evidentlyai.com/
- SHAP Docs: https://shap.readthedocs.io/


