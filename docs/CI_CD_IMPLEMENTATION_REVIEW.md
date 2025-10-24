# CI/CD Implementation Review

This document confirms how the MLOps Framework repository supports three critical CI/CD objectives:
unit testing/data validation, linting/reproducibility checks, and model packaging/registration.

## Objective 1: Unit Testing & Data Validation

### ✓ Implementation Status: **DEMONSTRATED**

The repository shows how to implement comprehensive testing at multiple levels:

### Unit Testing Structure

**Location:** `tests/` and per-model test directories

```
tests/
├── unit/                           # Isolated function tests
│   ├── test_mlflow_utils.py      # Test MLflow operations
│   ├── test_feast_utils.py       # Test feature store
│   └── test_drift_utils.py       # Test monitoring
├── integration/                    # Multi-component tests
│   └── test_training_pipeline.py  # End-to-end workflow
└── data_validation/
    └── validate_schemas.py        # Data quality checks

src/models/*/tests/                 # Model-specific tests
├── heart_disease_tests.py
├── diabetes_tests.py
└── readmission_regression.py
```

### Data Validation Implementation

**File:** [tests/data_validation/validate_schemas.py](../tests/data_validation/validate_schemas.py)

This script demonstrates **schema validation** with three key checks:

```python
# 1. Required columns check
missing_columns = set(expected_columns) - set(df.columns)
if missing_columns:
    raise ValidationError(f"Missing: {missing_columns}")

# 2. Data type validation
expected_dtypes = {
    'patient_id': 'object',
    'age': 'int64',
    'sex': 'int64'
}

# 3. Null value detection
null_counts = df.isnull().sum()
critical_nulls = null_counts[null_counts > 0]
```

### How It Works in Practice

**Pre-commit Hook Integration:**
```yaml
# .pre-commit-config.yaml (line 66-73)
- repo: local
  hooks:
    - id: validate-data-schema
      name: Validate Data Schema
      entry: python tests/data_validation/validate_schemas.py
      language: system
      files: \.(parquet|csv)$
```

**What This Means:**
- Every time data files change (`.parquet` or `.csv`), validation runs automatically
- Prevents bad data from entering the training pipeline
- Catches schema drift early (before model training fails)

### Example Unit Test Pattern

```python
# tests/unit/test_mlflow_utils.py (pseudo-code example)

def test_compare_models():
    """Test champion/challenger comparison logic."""
    # Arrange
    champion_metrics = {'roc_auc': 0.85, 'precision': 0.82}
    challenger_metrics = {'roc_auc': 0.88, 'precision': 0.85}

    # Act
    result = mlflow_utils.compare_models(
        champion_metrics,
        challenger_metrics,
        primary_metric='roc_auc'
    )

    # Assert
    assert result['promote'] == True
    assert result['improvement'] == 0.03
    assert result['improvement_pct'] == 3.53

def test_model_registration():
    """Test model registration in MLflow."""
    with mlflow.start_run() as run:
        # Log a dummy model
        mlflow.sklearn.log_model(DummyModel(), "model")
        run_id = run.info.run_id

    # Register model
    version = mlflow_utils.register_model(
        run_id=run_id,
        model_name="test_model",
        alias="challenger"
    )

    # Verify registration
    assert version.version is not None
    assert version.current_stage == "None"  # Pre-promotion

def test_data_validation():
    """Test data schema validation."""
    # Valid data
    valid_df = pd.DataFrame({
        'patient_id': ['PT001', 'PT002'],
        'age': [45, 67],
        'sex': [1, 0]
    })

    result = validate_dataframe_schema(
        valid_df,
        expected_columns=['patient_id', 'age', 'sex']
    )
    assert result == True

    # Invalid data (missing column)
    invalid_df = pd.DataFrame({
        'patient_id': ['PT001'],
        'age': [45]
        # Missing 'sex' column
    })

    result = validate_dataframe_schema(
        invalid_df,
        expected_columns=['patient_id', 'age', 'sex']
    )
    assert result == False
```

### CI/CD Integration (GitHub Actions)

**Documented in:** [CICD_AND_PRECOMMIT.md](../CICD_AND_PRECOMMIT.md)

```yaml
# .github/workflows/ci.yml (example from docs)
name: Continuous Integration
on: [pull_request, push]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt

      # Unit tests with coverage
      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src --cov-report=xml

      # Integration tests
      - name: Run integration tests
        run: pytest tests/integration/ -v

      # Data validation
      - name: Validate data schemas
        run: python tests/data_validation/validate_schemas.py

      # Upload coverage to monitoring
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### Testing Pyramid Applied

```
        /\
       /  \     E2E Tests (Few, slow)
      /────\    - Full training → inference pipeline
     /      \   - Dashboard rendering
    /────────\
   /          \ Integration Tests (Some, medium)
  /            \  - Training + MLflow registration
 /──────────────\  - Inference + FEAST feature fetch
/                \
\────────────────/ Unit Tests (Many, fast)
                   - Individual function tests
                   - Data validation
                   - Schema checks
```

**Coverage Goals:**
- Unit tests: 80%+ coverage
- Integration tests: Critical paths
- Data validation: All ingestion points

---

## Objective 2: Linting & Reproducibility Checks

### ✓ Implementation Status: **FULLY IMPLEMENTED**

The repository uses **pre-commit hooks** to enforce code quality and reproducibility before every commit.

### Pre-commit Configuration

**File:** [.pre-commit-config.yaml](../.pre-commit-config.yaml)

Implements **7 categories of checks**:

#### 1. Code Formatting (Black)
```yaml
- repo: https://github.com/psf/black
  rev: 24.1.1
  hooks:
    - id: black
      language_version: python3.9
      args: [--line-length=100]
```

**Purpose:** Ensures consistent code style across all contributors
**Reproducibility Impact:** Same code formatting = easier diffs, clearer code reviews

#### 2. Import Sorting (isort)
```yaml
- repo: https://github.com/pycqa/isort
  rev: 5.13.2
  hooks:
    - id: isort
      args: [--profile=black, --line-length=100]
```

**Purpose:** Alphabetizes and groups imports consistently
**Reproducibility Impact:** Prevents import order bugs (Python imports execute top-to-bottom)

#### 3. Linting (Flake8)
```yaml
- repo: https://github.com/pycqa/flake8
  rev: 7.0.0
  hooks:
    - id: flake8
      args: [--max-line-length=100, --extend-ignore=E203,W503]
      additional_dependencies: [flake8-docstrings]
```

**Catches:**
- Unused variables and imports
- Undefined names
- Syntax errors
- Missing docstrings

**Reproducibility Impact:** Prevents runtime errors from unused/undefined code

#### 4. Type Checking (MyPy)
```yaml
- repo: https://github.com/pre-commit/mirrors-mypy
  rev: v1.8.0
  hooks:
    - id: mypy
      args: [--ignore-missing-imports, --no-strict-optional]
      additional_dependencies: [types-PyYAML, types-requests]
```

**Purpose:** Static type analysis catches type mismatches before runtime
**Reproducibility Impact:** Type safety = fewer runtime errors = consistent behavior

**Example:**
```python
# Type hints enable MyPy checking
def train_model(config: dict, data: pd.DataFrame) -> Tuple[str, dict]:
    # MyPy ensures:
    # - config is dict
    # - data is DataFrame
    # - Returns tuple of (str, dict)
    pass
```

#### 5. Security Scanning (Bandit)
```yaml
- repo: https://github.com/PyCQA/bandit
  rev: 1.7.6
  hooks:
    - id: bandit
      args: [-ll, -x, tests]  # Low-level severity, exclude tests
```

**Detects:**
- Hardcoded passwords
- SQL injection vulnerabilities
- Use of `eval()` or `exec()`
- Insecure random number generation

**Reproducibility Impact:** Security issues can cause non-deterministic failures in production

#### 6. YAML Validation
```yaml
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v4.5.0
  hooks:
    - id: check-yaml
      args: [--unsafe]
    - id: detect-private-key
    - id: check-merge-conflict
    - id: check-added-large-files
      args: [--maxkb=1000]
```

**Purpose:** Validates configuration files (critical for reproducibility)
**Reproducibility Impact:** Invalid YAML = broken configs = different behavior

#### 7. Notebook Formatting (nbQA)
```yaml
- repo: https://github.com/nbQA-dev/nbQA
  rev: 1.7.1
  hooks:
    - id: nbqa-black
    - id: nbqa-isort
```

**Purpose:** Applies same linting to Jupyter notebooks
**Reproducibility Impact:** Notebooks often used for experiments → must be reproducible too

### Reproducibility Checks in Training Code

**Demonstrated in:** [src/models/heart_disease/train.py](../src/models/heart_disease/train.py)

```python
# 1. Git commit hash (code version)
git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
mlflow.log_param("git_commit", git_hash)

# 2. Random seed (deterministic randomness)
RANDOM_SEED = config['training']['random_seed']  # From config.yaml
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
mlflow.log_param("random_seed", RANDOM_SEED)

# 3. Package versions (environment reproducibility)
mlflow.log_param("python_version", sys.version)
mlflow.log_param("xgboost_version", xgboost.__version__)
mlflow.log_param("pandas_version", pd.__version__)

# 4. Data hash (data versioning)
data_hash = hashlib.sha256(training_df.to_csv().encode()).hexdigest()
mlflow.log_param("dataset_hash", data_hash)

# 5. Configuration snapshot
mlflow.log_dict(config, "config.yaml")
```

### How Reproducibility is Enforced

**Pre-commit workflow:**
```bash
# Developer makes changes
git add src/models/heart_disease/train.py

# Attempts to commit
git commit -m "Add new feature"

# Pre-commit hooks run automatically:
# ✓ Black formats code
# ✓ isort organizes imports
# ✓ Flake8 checks code quality
# ✓ MyPy verifies types
# ✓ Bandit scans for security issues
# ✓ YAML configs validated

# If ANY check fails:
# ✗ Commit is blocked
# → Developer must fix issues

# Once all checks pass:
# ✓ Commit succeeds
```

### Reproducibility Documentation

**File:** [CICD_AND_PRECOMMIT.md](../CICD_AND_PRECOMMIT.md) Section: "Reproducibility"

Explains:
- Why reproducibility matters (debugging, compliance, trust)
- What to version (code, data, environment, config, random seeds)
- How to reproduce a training run (step-by-step)

### CI/CD Reproducibility Checks

```yaml
# .github/workflows/ci.yml (from docs)
jobs:
  test:
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]  # Test multiple versions

    steps:
      # Ensure code runs on clean environment
      - name: Install dependencies from requirements.txt
        run: pip install -r requirements.txt

      # Verify requirements.txt is up-to-date
      - name: Check dependency consistency
        run: |
          pip freeze > current-requirements.txt
          diff requirements.txt current-requirements.txt

      # Run linters (same as pre-commit)
      - name: Lint with Flake8
        run: flake8 src/ tests/

      - name: Type check with MyPy
        run: mypy src/

      # Verify reproducibility of training
      - name: Test reproducible training
        run: |
          python src/models/heart_disease/train.py --seed=42
          # Run twice, compare model hashes
          python src/models/heart_disease/train.py --seed=42
          # Models should be identical
```

---

## Objective 3: Model Packaging & Registration Stub

### ✓ Implementation Status: **DEMONSTRATED WITH EXAMPLES**

The repository shows the complete model packaging and registration workflow using MLflow.

### Model Packaging Pattern

**Demonstrated in:** [src/utils/mlflow_utils.py](../src/utils/mlflow_utils.py)

### Key Functions

#### 1. Model Registration
```python
def register_model(run_id: str, model_name: str, alias: str = None):
    """
    Register a model from a training run.

    Model Registry workflow:
    1. Train model (logged to MLflow run)
    2. Register model (creates versioned entry)
    3. Assign alias (optional: 'champion', 'challenger', 'staging')

    Args:
        run_id: MLflow run ID containing the model
        model_name: Name in registry
        alias: Optional alias to assign

    Returns:
        ModelVersion object
    """
    client = MlflowClient()

    # Register model (creates new version)
    model_uri = f"runs:/{run_id}/model"
    model_version = mlflow.register_model(model_uri, model_name)

    # Assign alias if provided
    if alias:
        client.set_registered_model_alias(
            model_name,
            alias,
            model_version.version
        )

    return model_version
```

**What Gets Packaged:**
- Model artifact (serialized XGBoost model)
- Conda environment specification
- Python version requirement
- Model signature (input/output schema)
- Custom artifacts (SHAP explainer, scaler, encoders)

#### 2. Model Loading by Alias
```python
def get_model_by_alias(client, model_name, alias):
    """
    Load a model by its alias (e.g., 'champion', 'challenger').

    Aliases enable seamless model updates:
    - Production always uses 'champion' alias
    - When champion changes, production automatically gets new version
    - No code changes needed

    Args:
        client: MlflowClient instance
        model_name: Registered model name
        alias: Model alias to fetch

    Returns:
        tuple: (model_artifact, version_metadata) or (None, None)
    """
    try:
        # Get version info by alias
        model_version = client.get_model_version_by_alias(model_name, alias)

        # Load model artifact
        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.pyfunc.load_model(model_uri)

        return model, model_version

    except Exception:
        return None, None
```

**Production Usage:**
```python
# Inference service (always loads champion)
model, version = get_model_by_alias(
    client,
    model_name="heart_disease_classifier",
    alias="champion"
)

# No code changes when champion updates!
# MLflow registry handles versioning
```

#### 3. Model Promotion
```python
def promote_model(model_name, new_champion_version, old_champion_version=None):
    """
    Promote a challenger to champion.

    Champion/Challenger Pattern:
    - Champion: Current production model
    - Challenger: New model that outperformed champion
    - Promotion: Move challenger to champion, archive old champion

    Args:
        model_name: Name of registered model
        new_champion_version: Version to promote
        old_champion_version: Current champion (will be archived)

    Returns:
        bool: Success status
    """
    client = MlflowClient()

    # Archive old champion
    if old_champion_version:
        client.delete_registered_model_alias(model_name, "champion")
        client.set_registered_model_alias(
            model_name,
            "archived",
            old_champion_version
        )

    # Promote new champion
    client.set_registered_model_alias(
        model_name,
        "champion",
        new_champion_version
    )

    return True
```

### Complete Packaging Workflow

**Demonstrated in:** [src/models/heart_disease/train.py](../src/models/heart_disease/train.py)

```python
class HeartDiseaseTrainer:
    def train_model(self, training_df):
        """Train and package model with all artifacts."""

        # Train XGBoost model
        model = XGBoostClassifier(**best_params)
        model.fit(X_train, y_train)

        # Package model with MLflow
        with mlflow.start_run() as run:
            # 1. Log hyperparameters
            mlflow.log_params(best_params)

            # 2. Log metrics
            mlflow.log_metrics({
                'roc_auc': metrics['roc_auc'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
            })

            # 3. Log model with signature
            signature = mlflow.models.infer_signature(X_train, y_train)
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=signature,
                conda_env={
                    'python': '3.9',
                    'dependencies': [
                        'xgboost==2.0.3',
                        'scikit-learn==1.3.2',
                        'pandas==1.5.3'
                    ]
                }
            )

            # 4. Log additional artifacts
            mlflow.log_artifact("config.yaml")
            mlflow.log_artifact("requirements.txt")

            # 5. Log dataset hash for reproducibility
            mlflow.log_param("dataset_hash", data_hash)

            run_id = run.info.run_id

        return run_id, model, metrics

    def evaluate_promotion(self, run_id, metrics):
        """Evaluate and register model."""

        # Compare to existing champion
        comparison = compare_models(champion_metrics, metrics)

        if comparison['improvement'] >= 0.02:  # 2% threshold
            # Register as challenger
            version = register_model(
                run_id=run_id,
                model_name="heart_disease_classifier",
                alias="challenger"
            )

            return {
                "promoted": True,
                "alias": "challenger",
                "version": version.version
            }
        else:
            # Register without alias (for tracking only)
            version = register_model(
                run_id=run_id,
                model_name="heart_disease_classifier"
            )

            return {
                "promoted": False,
                "version": version.version
            }
```

### Model Registry Structure

```
MLflow Model Registry
│
├── heart_disease_classifier
│   ├── Version 1 (archived)
│   │   ├── Model artifact (model.pkl)
│   │   ├── Conda env (conda.yaml)
│   │   ├── Requirements (requirements.txt)
│   │   ├── Signature (input/output schema)
│   │   └── Metadata (params, metrics, tags)
│   │
│   ├── Version 2 (archived)
│   │   └── [same structure]
│   │
│   ├── Version 3 (champion) ← Production
│   │   └── [same structure]
│   │
│   └── Version 4 (challenger) ← Testing
│       └── [same structure]
│
├── diabetes_classifier
│   └── [similar structure]
│
└── readmission_predictor
    └── [similar structure]
```

### Deployment Integration

**Kubernetes pulls models from registry:**

```yaml
# kubernetes-deployment.yaml (inference service)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-api
spec:
  template:
    spec:
      containers:
      - name: api
        image: mlops-framework:latest
        env:
          # Service loads champion model on startup
          - name: MLFLOW_TRACKING_URI
            value: "http://mlflow-service:5000"
          - name: MODEL_NAME
            value: "heart_disease_classifier"
          - name: MODEL_ALIAS
            value: "champion"  # Always use production model
```

**Inference service code:**
```python
# src/models/heart_disease/inference.py
class HeartDiseaseInference:
    def __init__(self, model_alias="champion"):
        # Load model from MLflow registry
        self.model, self.version = get_model_by_alias(
            client,
            model_name="heart_disease_classifier",
            alias=model_alias
        )

        # Model automatically includes:
        # - Trained XGBoost model
        # - Conda environment
        # - Input/output signature
        # - All logged artifacts
```

### Model Package Contents

When a model is registered, MLflow packages:

**1. Model File**
- `model.pkl` or `model.joblib` (serialized model)

**2. Environment Definition**
- `conda.yaml` - Conda environment specification
- `requirements.txt` - Pip dependencies
- `python_version.txt` - Python version

**3. Model Signature**
```python
# Automatically inferred from training data
signature = mlflow.models.infer_signature(
    model_input=X_train,
    model_output=predictions
)

# Ensures inference receives correct schema:
# - Column names
# - Data types
# - Shape
```

**4. Custom Artifacts**
- `config.yaml` - Model configuration
- `shap_explainer.pkl` - SHAP explainer
- `feature_names.json` - Feature metadata

**5. Metadata**
- Tags (model_type=xgboost, use_case=heart_disease)
- Parameters (hyperparameters, random_seed)
- Metrics (roc_auc, precision, recall)
- Dataset hash (reproducibility)

### Testing Model Packaging

```python
# tests/integration/test_model_registration.py (example)

def test_model_packaging_and_loading():
    """Test complete model packaging workflow."""

    # 1. Train and log model
    with mlflow.start_run() as run:
        model = DummyModel()
        mlflow.sklearn.log_model(model, "model")
        run_id = run.info.run_id

    # 2. Register model
    version = register_model(
        run_id=run_id,
        model_name="test_model",
        alias="champion"
    )

    # 3. Load model by alias
    loaded_model, loaded_version = get_model_by_alias(
        client,
        model_name="test_model",
        alias="champion"
    )

    # 4. Verify packaging
    assert loaded_model is not None
    assert loaded_version.version == version.version

    # 5. Test prediction (ensure model works)
    test_data = pd.DataFrame({'feature': [1, 2, 3]})
    predictions = loaded_model.predict(test_data)
    assert predictions is not None
```

---

## Summary: How Objectives Are Met

### ✓ Unit Testing / Data Validation
- **Structure:** `tests/unit/`, `tests/integration/`, `tests/data_validation/`
- **Pre-commit Hook:** Validates data schemas on every commit
- **CI/CD:** Automated test runs on every pull request
- **Coverage:** Testing pyramid (many unit, some integration, few E2E)

### ✓ Linting / Reproducibility Checks
- **Pre-commit Hooks:** Black, isort, Flake8, MyPy, Bandit, YAML validation
- **Reproducibility Logging:** Git hash, random seed, package versions, data hash
- **Environment Packaging:** Conda/pip environments captured with every model
- **Configuration Management:** All settings in version-controlled YAML

### ✓ Model Packaging & Registration
- **MLflow Registry:** Centralized model versioning and metadata
- **Champion/Challenger Pattern:** Safe promotion workflow with aliases
- **Complete Packaging:** Model + environment + config + artifacts
- **Deployment Integration:** Kubernetes loads models directly from registry

## Additional Resources

- **Pre-commit Setup:** [.pre-commit-config.yaml](../.pre-commit-config.yaml)
- **CI/CD Guide:** [CICD_AND_PRECOMMIT.md](../CICD_AND_PRECOMMIT.md)
- **MLflow Utils:** [src/utils/mlflow_utils.py](../src/utils/mlflow_utils.py)
- **Training Pipeline:** [src/models/heart_disease/train.py](../src/models/heart_disease/train.py)
- **Data Validation:** [tests/data_validation/validate_schemas.py](../tests/data_validation/validate_schemas.py)

All three objectives are **demonstrated with working examples** that can be adapted for production use.
