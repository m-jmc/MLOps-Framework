"""
FEAST Feature Store Utilities for Community Hospital MLOps Platform

This module provides helper functions for interacting with the FEAST feature store:
- Feature store initialization
- Feature materialization
- Online/offline feature retrieval
- Feature version tracking
"""

from feast import FeatureStore
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def get_feature_store(repo_path: str = "src/feature_store/heart_disease_features"):
    """
    Initialize and return a FEAST feature store client.

    Args:
        repo_path: Path to the FEAST repository

    Returns:
        FeatureStore instance
    """
    repo_path = Path(repo_path)

    if not repo_path.exists():
        raise FileNotFoundError(f"Feature store repository not found: {repo_path}")

    store = FeatureStore(repo_path=str(repo_path))
    logger.info(f"Initialized feature store from: {repo_path}")

    return store


def get_historical_features(
    store: FeatureStore,
    entity_df: pd.DataFrame,
    features: list,
    full_feature_names: bool = False
):
    """
    Retrieve historical features for training.

    Args:
        store: FeatureStore instance
        entity_df: DataFrame with entity IDs and timestamps
        features: List of feature references (e.g., ["heart_disease_features:age"])
        full_feature_names: Whether to use full feature names in output

    Returns:
        DataFrame with historical features
    """
    logger.info(f"Retrieving {len(features)} historical features for {len(entity_df)} entities")

    training_df = store.get_historical_features(
        entity_df=entity_df,
        features=features,
        full_feature_names=full_feature_names
    ).to_df()

    logger.info(f"Retrieved historical features: {training_df.shape}")

    return training_df


def get_online_features(
    store: FeatureStore,
    entity_rows: list,
    features: list,
    full_feature_names: bool = False
):
    """
    Retrieve online features for real-time inference.

    Args:
        store: FeatureStore instance
        entity_rows: List of entity dictionaries (e.g., [{"patient_id": "PT000001"}])
        features: List of feature references
        full_feature_names: Whether to use full feature names in output

    Returns:
        Dictionary of feature values
    """
    logger.info(f"Retrieving online features for {len(entity_rows)} entities")

    feature_vector = store.get_online_features(
        features=features,
        entity_rows=entity_rows,
        full_feature_names=full_feature_names
    ).to_dict()

    return feature_vector


def materialize_features(
    store: FeatureStore,
    start_date: datetime = None,
    end_date: datetime = None
):
    """
    Materialize features to the online store.

    Args:
        store: FeatureStore instance
        start_date: Start of materialization window (default: 7 days ago)
        end_date: End of materialization window (default: now)

    Returns:
        None
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=7)

    if end_date is None:
        end_date = datetime.now()

    logger.info(f"Materializing features from {start_date} to {end_date}")

    store.materialize(
        start_date=start_date,
        end_date=end_date
    )

    logger.info("Feature materialization complete")


def materialize_incremental(store: FeatureStore, end_date: datetime = None):
    """
    Materialize features incrementally (since last materialization).

    Args:
        store: FeatureStore instance
        end_date: End of materialization window (default: now)

    Returns:
        None
    """
    if end_date is None:
        end_date = datetime.now()

    logger.info(f"Materializing features incrementally up to {end_date}")

    store.materialize_incremental(end_date=end_date)

    logger.info("Incremental feature materialization complete")


def prepare_training_data(
    repo_path: str,
    entity_data_path: str,
    label_data_path: str,
    feature_service: str = None,
    features: list = None,
    drop_columns: list = None
):
    """
    Prepare complete training dataset from FEAST feature store.

    Args:
        repo_path: Path to FEAST repository
        entity_data_path: Path to entity data (CSV or Parquet)
        label_data_path: Path to label data (CSV or Parquet)
        feature_service: Name of feature service to use
        features: List of specific features to retrieve (if not using feature service)
        drop_columns: Columns to drop from final dataset

    Returns:
        DataFrame ready for training
    """
    logger.info("Preparing training data from FEAST feature store")

    # Initialize feature store
    store = get_feature_store(repo_path)

    # Load entity data
    entity_data_path = Path(entity_data_path)
    if entity_data_path.suffix == '.parquet':
        entity_df = pd.read_parquet(entity_data_path)
    else:
        entity_df = pd.read_csv(entity_data_path)

    logger.info(f"Loaded entity data: {entity_df.shape}")

    # Load label data
    label_data_path = Path(label_data_path)
    if label_data_path.suffix == '.parquet':
        label_df = pd.read_parquet(label_data_path)
    else:
        label_df = pd.read_csv(label_data_path)

    logger.info(f"Loaded label data: {label_df.shape}")

    # Determine features to retrieve
    if feature_service:
        features_to_retrieve = [feature_service]
        logger.info(f"Using feature service: {feature_service}")
    elif features:
        features_to_retrieve = features
        logger.info(f"Using {len(features)} specified features")
    else:
        raise ValueError("Must specify either feature_service or features list")

    # Add timestamp if not present (required for historical features)
    if 'event_timestamp' not in entity_df.columns:
        entity_df['event_timestamp'] = datetime.now()

    # Retrieve historical features
    training_df = get_historical_features(
        store=store,
        entity_df=entity_df,
        features=features_to_retrieve,
        full_feature_names=False
    )

    # Merge with labels
    merge_keys = ['patient_id', 'month_key'] if 'month_key' in label_df.columns else ['patient_id']
    training_df = training_df.merge(label_df, on=merge_keys, how='left')

    logger.info(f"Merged training data: {training_df.shape}")

    # Drop specified columns
    if drop_columns:
        columns_to_drop = [col for col in drop_columns if col in training_df.columns]
        training_df = training_df.drop(columns=columns_to_drop)
        logger.info(f"Dropped {len(columns_to_drop)} columns")

    # Drop rows with missing target
    target_col = [col for col in training_df.columns if 'disease' in col or 'outcome' in col]
    if target_col:
        training_df = training_df.dropna(subset=target_col)

    logger.info(f"Final training data: {training_df.shape}")

    return training_df


def get_feature_statistics(store: FeatureStore):
    """
    Get statistics about features in the store.

    Args:
        store: FeatureStore instance

    Returns:
        Dict with feature store statistics
    """
    feature_views = store.list_feature_views()
    entities = store.list_entities()

    stats = {
        "num_feature_views": len(feature_views),
        "num_entities": len(entities),
        "feature_views": [fv.name for fv in feature_views],
        "entities": [e.name for e in entities]
    }

    logger.info(f"Feature store stats: {stats}")

    return stats


def validate_feature_schema(store: FeatureStore, expected_features: list):
    """
    Validate that expected features exist in the feature store.

    Args:
        store: FeatureStore instance
        expected_features: List of expected feature names

    Returns:
        dict: Validation results
    """
    feature_views = store.list_feature_views()
    available_features = []

    for fv in feature_views:
        for feature in fv.features:
            available_features.append(f"{fv.name}:{feature.name}")

    missing_features = [f for f in expected_features if f not in available_features]
    extra_features = [f for f in available_features if f not in expected_features]

    validation = {
        "valid": len(missing_features) == 0,
        "missing_features": missing_features,
        "extra_features": extra_features,
        "total_expected": len(expected_features),
        "total_available": len(available_features)
    }

    if validation["valid"]:
        logger.info("✓ Feature schema validation passed")
    else:
        logger.warning(f"✗ Feature schema validation failed: {len(missing_features)} missing features")

    return validation
