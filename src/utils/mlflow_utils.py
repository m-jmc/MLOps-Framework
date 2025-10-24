"""
MLflow Utilities - Simplified Educational Version

Demonstrates MLflow operations for MLOps:
- Experiment tracking (log params, metrics, artifacts)
- Model registry (versioning, aliasing)
- Model lifecycle (champion/challenger promotion)
"""

import mlflow
from mlflow.tracking import MlflowClient


def initialize_mlflow_client(tracking_uri, experiment_name):
    """
    Setup MLflow connection and experiment.

    Args:
        tracking_uri: Where MLflow stores data (e.g., "sqlite:///mlflow.db")
        experiment_name: Logical grouping for related runs

    Returns:
        MlflowClient for interacting with tracking server
    """
    # Set tracking URI (local SQLite or remote server)
    mlflow.set_tracking_uri(tracking_uri)

    # Create/get experiment
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)

    return client


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


def register_model(run_id, model_name, alias=None):
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


def compare_models(champion_metrics, challenger_metrics, primary_metric="roc_auc"):
    """
    Compare two models to decide promotion.

    Comparison criteria:
    - Primary metric improvement (e.g., ROC-AUC)
    - Minimum threshold (e.g., 2% improvement)
    - Fairness constraints (no bias violations)

    Args:
        champion_metrics: Current production model metrics
        challenger_metrics: New model metrics
        primary_metric: Metric to compare (default: roc_auc)

    Returns:
        dict: Comparison results with recommendation
    """
    if not champion_metrics:
        return {
            "promote": True,
            "reason": "No existing champion",
            "improvement": None
        }

    # Extract scores
    champion_score = champion_metrics.get(primary_metric, 0)
    challenger_score = challenger_metrics.get(primary_metric, 0)

    # Calculate improvement
    improvement = challenger_score - champion_score
    improvement_pct = (improvement / champion_score) * 100

    # Decide promotion
    promote = challenger_score > champion_score

    return {
        "promote": promote,
        "reason": f"Challenger {'outperforms' if promote else 'underperforms'} champion",
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "champion_score": champion_score,
        "challenger_score": challenger_score
    }


def log_model_metadata(run_id, metadata):
    """
    Add additional metadata to an MLflow run.

    Useful for:
    - Data versioning (dataset hash, feature store version)
    - Model lineage (training data source, feature engineering code version)
    - Business context (model owner, use case, approval status)

    Args:
        run_id: MLflow run ID
        metadata: Dict with tags, params, metrics to log
    """
    with mlflow.start_run(run_id=run_id):
        # Log tags (categorical metadata)
        if "tags" in metadata:
            for key, value in metadata["tags"].items():
                mlflow.set_tag(key, value)

        # Log params (model configuration)
        if "params" in metadata:
            mlflow.log_params(metadata["params"])

        # Log metrics (performance indicators)
        if "metrics" in metadata:
            mlflow.log_metrics(metadata["metrics"])


def get_latest_model_version(model_name):
    """
    Get the most recent version of a registered model.

    Useful for:
    - Checking if new models have been registered
    - Retrieving latest model for ad-hoc analysis
    - Monitoring model registry activity

    Args:
        model_name: Registered model name

    Returns:
        ModelVersion object or None
    """
    client = MlflowClient()

    # Search all versions
    versions = client.search_model_versions(f"name='{model_name}'")

    if versions:
        # Sort by version number (descending) and return latest
        latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        return latest

    return None
