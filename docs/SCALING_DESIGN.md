# Scaling MLOps: From 1 to 100+ Models

This document explains the architectural decisions for scaling this MLOps platform from a single model to hundreds of production models while maintaining reliability, governance, and operational efficiency.

## Core Scaling Strategy

The platform scales horizontally through **templatization** and **shared infrastructure**:

```
Single Model:
src/models/heart_disease/ → MLflow → Production

100 Models:
src/models/{model_1..100}/ → Shared MLflow Registry → Multi-Model Serving
                          ↓
                    Shared FEAST Feature Store
                          ↓
                  Shared Monitoring & Governance
```

### Key Principles

1. **Consistent Structure**: All models follow identical directory layout
2. **Shared Utilities**: Common operations (MLflow, FEAST, monitoring) abstracted once
3. **Configuration-Driven**: Model-specific logic in YAML, not code
4. **Automated CI/CD**: Pipelines automatically discover and process new models
5. **Centralized Governance**: Single approval workflow for all models

## Model Versioning & Lifecycle Management

### Versioning Strategy

**MLflow Model Registry** provides version control for all models:

```yaml
Model Registry Structure:
  heart_disease_classifier
    ├── v1 (archived)
    ├── v2 (archived)
    ├── v3 (champion) ← Production traffic
    └── v4 (challenger) ← A/B testing

  diabetes_classifier
    ├── v1 (champion)
    └── v2 (challenger)
```

**Every model version includes:**
- Model artifact (pickled XGBoost model)
- Hyperparameters used
- Training dataset hash (lineage)
- Performance metrics (AUC, precision, recall)
- Training environment (Python version, package versions)
- SHAP explainer artifact

### Promotion Workflow

Models progress through stages using **aliases** :

```
Training → Staging → Challenger → Champion → Archived
```

**Champion/Challenger Pattern:**
```python
# Production always loads champion
model_uri = f"models://{model_name}@champion"
model = mlflow.pyfunc.load_model(model_uri)

# When challenger proves better:
# 1. Automated comparison (ROC-AUC improvement ≥ 2%)
# 2. Bias check (disparity < 10%)
# 3. Manual approval (compliance review)
# 4. Promote: Set challenger → champion, old champion → archived
```

**Retirement Policy:**
- Archive versions after 90 days if not champion/challenger
- Retain champion history for 1 year (regulatory compliance)
- Delete artifacts after 2 years (or export to cold storage)

### Why MLflow Over Alternatives

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **MLflow** | Open source, language agnostic, great UI, local → cloud | Limited RBAC, no built-in approval gates | ✓ **Chosen** - Extensible, industry standard |
| SageMaker Model Registry | Managed service, AWS-native RBAC | AWS lock-in, expensive at scale | ✗ Vendor lock-in risk 

## Training Reproducibility & Experiment Tracking

### Reproducibility Requirements

Every training run must be **100% reproducible** for regulatory compliance and debugging:

**1. Code Version**
```python
# Git commit hash logged to MLflow
mlflow.log_param("git_commit", os.popen('git rev-parse HEAD').read().strip())
mlflow.log_param("code_version", "v2.3.1")
```

**2. Data Version**
```python
# Dataset hash ensures exact data used
dataset_hash = hashlib.sha256(training_df.to_csv().encode()).hexdigest()
mlflow.log_param("dataset_hash", dataset_hash)
mlflow.log_param("feast_feature_version", feast_client.get_registry_version())
```

**3. Environment Version**
```python
# Capture exact package versions
mlflow.log_artifact("requirements.txt")
mlflow.log_param("python_version", sys.version)
mlflow.log_param("xgboost_version", xgboost.__version__)
```

**4. Random Seeds**
```python
# Fix all randomness sources
np.random.seed(333)
random.seed(333)
mlflow.log_param("random_seed", 333)
```

**Reproducing a Run:**
```bash
# Given MLflow run ID: abc123
mlflow models serve -m runs:/abc123/model  # Exact model
git checkout $(mlflow runs describe abc123 | grep git_commit)  # Exact code
feast materialize --feature-view $(mlflow runs describe abc123 | grep feast_version)  # Exact features
```

### Experiment Tracking at Scale

**Hierarchical Organization:**
```
MLflow Tracking Server
  ├── Experiment: Heart Disease Models
  │   ├── Run: baseline_xgboost (AUC: 0.82)
  │   ├── Run: hyperopt_optimized (AUC: 0.85)
  │   └── Run: lgbm_comparison (AUC: 0.84)
  ├── Experiment: Diabetes Risk Models
  │   └── Run: ...
  └── Experiment: Readmission Prediction
      └── Run: ...
```

**Automated Experiment Naming:**
```python
# Convention: {model_name}_{data_date}_{git_branch}
experiment_name = f"{config['model']['name']}_{datetime.now():%Y%m%d}_main"
mlflow.set_experiment(experiment_name)
```

**Comparison Queries:**
```python
# Find best model across all experiments
runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id for exp in all_experiments],
    filter_string="metrics.roc_auc > 0.85 AND tags.status = 'production_ready'",
    order_by=["metrics.roc_auc DESC"],
    max_results=10
)
```

### Why MLflow for Tracking

**Alternatives Considered:**
- **TensorBoard**: Limited to TensorFlow/PyTorch, poor artifact management
- **MLflow**: ✓ Open source, self-hosted, works with any ML library

## Environments & Dependencies

### Challenge: 100 Models = 100 Different Dependencies

Models trained at different times require different package versions:
- Model A (2023): `xgboost==1.5.0`, `scikit-learn==1.0.2`
- Model B (2024): `xgboost==2.0.3`, `scikit-learn==1.3.2`

### Solution: Containerized Environments

**Per-Model Docker Images:**
```dockerfile
# Base image with common dependencies
FROM python:3.9-slim AS base
RUN pip install mlflow feast pandas

# Model-specific image
FROM base AS heart_disease_model
COPY models/heart_disease/requirements.txt .
RUN pip install -r requirements.txt
```

**Kubernetes Deployment Manifest:**
```yaml
# Each model gets its own deployment with pinned image
apiVersion: apps/v1
kind: Deployment
metadata:
  name: heart-disease-v3
spec:
  template:
    spec:
      containers:
      - name: inference
        image: mlops-framework:heart-disease-v3  # Pinned to model version
```

**Benefits:**
- ✓ Isolation: Model A's old packages don't conflict with Model B
- ✓ Reproducibility: Image tagged with model version
- ✓ Rollback: Revert to previous image instantly
- ✓ Testing: Validate new model in staging with exact production environment

### Dependency Management Workflow

```bash
# 1. Training time: Capture environment
pip freeze > models/heart_disease/requirements-v3.txt

# 2. Build time: Create versioned image
docker build -t mlops-framework:heart-disease-v3 \
  --build-arg MODEL=heart_disease \
  --build-arg VERSION=v3 .

# 3. Push to registry
docker push mlops-framework:heart-disease-v3

# 4. MLflow logs the image reference
mlflow.log_param("docker_image", "mlops-framework:heart-disease-v3")
```

### Why Docker Over Alternatives

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| **Docker + K8s** | Standard, portable, reproducible | Image bloat, build time | ✓ **Chosen** |
| Conda Environments | Good for data science | Slow, large, hard to deploy | ✗ Deployment pain |
| Virtual Environments | Lightweight | Not reproducible across OS | ✗ Linux→Windows breaks |
| Lambda Layers | Serverless-native | 250MB limit, cold starts | ✗ Too restrictive |

## Deployment & Scheduling

### Deployment Patterns

The platform supports **both batch and online inference**:

#### Online Inference (Real-Time)

```
User Request → Load Balancer → Inference Pod → FEAST → Model → Response
                                     ↓
                               MLflow (load champion)
```

#### Batch Inference (Scheduled)

```
CronJob → Spark Job → Load Data → FEAST → Model → Write Results → S3
```


### Scheduling Strategy

**Online Models (Synchronous):**
- Deployed as Kubernetes Deployments (always running)
- Auto-scaling based on traffic (HPA: 70% CPU threshold)
- Rolling updates for zero-downtime deployments

**Batch Models (Asynchronous):**
- Deployed as Kubernetes CronJobs (scheduled)
- Resource allocation: High memory, high CPU during run
- Spot instances for cost savings (tolerate interruptions)

### Multi-Model Serving

**Option 1: Model Per Service (Current Approach)**
```yaml
# Pros: Isolation, independent scaling, clear ownership
# Cons: More pods = higher baseline cost

heart-disease-api (5 pods)
diabetes-api (3 pods)
readmission-api (2 pods)
```

**Option 2: Multi-Model Server (Future)**
```python
# Single service loads multiple models
# Pros: Lower cost, shared resources
# Cons: Resource contention, blast radius

@app.post("/predict/{model_name}")
def predict(model_name: str, data: dict):
    model = model_cache.get(model_name)  # heart_disease, diabetes, etc.
    return model.predict(data)
```

**Decision**: Start with Option 1 (isolation), migrate high-traffic models to Option 2 when cost becomes issue.

### Orchestration: Airflow vs Kubeflow vs Github Actions

| Tool | Use Case | Pros | Cons | Our Choice |
|------|----------|------|------|------------|
| **Airflow** | Batch pipelines, scheduling | Mature, great UI, Python-native | Not ML-native, no GPU support | ✓ **Batch training** |
| **Kubeflow** | End-to-end ML workflows | K8s-native, ML-focused, pipelines | Complex, overkill for simple tasks | ✗ Too heavy |
| **GitHub Actions** | CI/CD, simple schedules | Free, integrated with code | Not for long-running jobs | ✓ **CI/CD only** 

**Our Approach:**
- **GitHub Actions**: CI/CD (testing, linting, deployment)
- **Kubernetes CronJobs**: Simple scheduled jobs (training, batch inference)
- **Airflow** (future): Complex DAGs with dependencies (multi-stage pipelines)

## Monitoring & Governance

### Drift Detection

**Three Drift Types Monitored:**

1. **Data Drift**: Input features change distribution
2. **Prediction Drift**: Model outputs change distribution
3. **Concept Drift**: Relationship between features and target changes

**Implementation:**
```python
# Weekly scheduled job
@cronjob("0 0 * * 1")  # Every Monday
def detect_drift():
    for model in all_models:
        # Load last 7 days of inference logs
        current_data = load_inference_logs(model, days=7)
        reference_data = load_training_data(model)

        # Evidently AI drift report
        drift_report = evidently.Report(metrics=[DataDriftPreset()])
        drift_report.run(reference_data, current_data)

        if drift_report.drift_detected():
            create_github_issue(model, drift_report)
            notify_slack(model, severity="high")
```

**Alerting Thresholds:**
- Drift score > 10% → Create GitHub issue (low priority)
- Drift score > 25% → Slack alert + PagerDuty (high priority)
- Drift score > 50% → Auto-retrain trigger (urgent)

### Bias Monitoring

**Fairness Metrics (Quarterly Audits):**
```python
protected_attributes = ['sex', 'age_group', 'race']

for attr in protected_attributes:
    for group in data[attr].unique():
        # Demographic parity: P(ŷ=1 | A=a)
        positive_rate = predictions[data[attr] == group].mean()

        # Equal opportunity: TPR parity
        tpr = true_positives / actual_positives

        # Predictive parity: Precision parity
        precision = true_positives / predicted_positives

        # Flag if disparity > 10%
        if max_disparity > 0.10:
            trigger_bias_review(model, attr, group)
```

**Bias Thresholds:**
- Disparity < 10% → Pass (promote to production)
- Disparity 10-15% → Warning (document in model card)
- Disparity > 15% → Block promotion (retrain with fairness constraints)

### Approval Workflow

**Automated Gates:**
```python
def can_promote_to_production(challenger, champion):
    checks = {
        'performance': challenger.auc >= champion.auc + 0.02,  # 2% improvement
        'stability': all(challenger[m] >= champion[m] * 0.95 for m in metrics),
        'bias': max_bias_disparity < 0.10,
        'drift': not currently_drifting(champion)
    }

    if all(checks.values()):
        # All automated checks pass → manual approval
        create_approval_request(challenger)
    else:
        reject_promotion(challenger, failed_checks=checks)
```

**Manual Approval (Compliance Review):**
1. Data scientist reviews model card
2. Compliance officer verifies bias metrics
3. Business stakeholder confirms business value
4. Approval recorded in audit log (immutable)

### Model Lineage

**Full Traceability:**
```
Prediction ID: abc123
  ↓
Model Version: heart_disease_v3 (champion)
  ↓
Training Run: mlflow_run_xyz789
  ↓
Training Data: feast_snapshot_20240115 (hash: 7f8a...)
  ↓
Raw Data: patients_table_v2 (2024-01-01 to 2024-01-14)
  ↓
Data Pipeline: ETL_job_20240114 (Airflow DAG)
```

## Scaling Roadmap

### Phase 1: 1-10 Models (Current)
- Single Kubernetes cluster
- Shared MLflow + FEAST
- Manual approval process
- Weekly drift detection

### Phase 2: 10-50 Models (Next 6 months)
- Multi-tenant K8s (namespace per team)
- Automated approval workflow
- Real-time drift detection (streaming)
- Feature store

### Phase 3: 50-100+ Models (12-18 months)
- Multi-region deployment (US-East, US-West)
- A/B testing framework (traffic splitting)

## Key Takeaways

1. **Standardization Enables Scale**: Consistent structure > custom solutions
2. **Automate Everything**: Manual processes break at 10 models
3. **Observability is Critical**: Can't manage what you can't measure
4. **Start Simple, Add Complexity**: Don't over-engineer for future scale
5. **Open Source First**: Avoid vendor lock-in, maintain portability

The platform prioritizes **operational simplicity** over bleeding-edge features. Every technology choice favors proven, maintainable solutions that work at scale.
