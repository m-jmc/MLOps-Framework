"""
XGBoost Classification Utilities for Community Hospital MLOps Platform

This module provides functions for training and evaluating XGBoost classification models
with hyperparameter optimization, cross-validation, and MLflow logging.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def xgb_classification(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = False,
    hyperopt_max_evals: int = 50
):
    """
    Train XGBoost classification model with hyperparameter optimization and MLflow logging.

    Args:
        df: Input DataFrame with features and target
        target_col: Name of the target column
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        use_smote: Whether to apply SMOTE for imbalanced datasets (default: False)
        hyperopt_max_evals: Max evaluations for hyperparameter optimization (default: 50)

    Returns:
        tuple: (run_id, best_model, metrics, X_train, X_test, y_train, y_test)
    """
    logger.info("Starting XGBoost classification training")

    # Prepare data
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Check if binary or multiclass
    num_classes = len(np.unique(y))
    is_binary = num_classes == 2

    logger.info(f"Classification type: {'Binary' if is_binary else f'Multiclass ({num_classes} classes)'}")

    # Define hyperparameter search space
    space = {
        'max_depth': hp.quniform('max_depth', 3, 10, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1)
    }

    def objective(params):
        """Hyperopt objective function."""
        params_dict = {
            'max_depth': int(params['max_depth']),
            'learning_rate': params['learning_rate'],
            'n_estimators': int(params['n_estimators']),
            'min_child_weight': int(params['min_child_weight']),
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'objective': 'binary:logistic' if is_binary else 'multi:softprob',
            'random_state': random_state
        }

        if not is_binary:
            params_dict['num_class'] = num_classes

        model = xgb.XGBClassifier(**params_dict)

        # Cross-validation
        cv_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_train_fold = X_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_train_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[val_idx]

            # Apply SMOTE if requested
            if use_smote:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=random_state)
                X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)

            model.fit(X_train_fold, y_train_fold)
            y_pred_proba = model.predict_proba(X_val_fold)

            if is_binary:
                score = roc_auc_score(y_val_fold, y_pred_proba[:, 1])
            else:
                score = roc_auc_score(y_val_fold, y_pred_proba, multi_class='ovr', average='weighted')

            cv_scores.append(score)

        return {'loss': -np.mean(cv_scores), 'status': STATUS_OK}

    # Run hyperparameter optimization
    logger.info(f"Running hyperparameter optimization ({hyperopt_max_evals} evaluations)...")
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=hyperopt_max_evals,
        trials=trials,
        rstate=np.random.default_rng(random_state),
        verbose=False
    )

    # Get best parameters
    best_params = {
        'max_depth': int(best['max_depth']),
        'learning_rate': best['learning_rate'],
        'n_estimators': int(best['n_estimators']),
        'min_child_weight': int(best['min_child_weight']),
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
        'objective': 'binary:logistic' if is_binary else 'multi:softprob',
        'random_state': random_state
    }

    if not is_binary:
        best_params['num_class'] = num_classes

    logger.info(f"Best hyperparameters: {best_params}")

    # Train final model with best parameters and log to MLflow
    with mlflow.start_run(log_system_metrics=True) as run:
        run_id = run.info.run_id

        # Log hyperparameters
        mlflow.log_params(best_params)
        mlflow.log_param("use_smote", use_smote)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("cv_folds", 5)

        # Apply SMOTE to full training set if requested
        if use_smote:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            logger.info(f"Applied SMOTE: {X_train.shape} -> {X_train_resampled.shape}")
        else:
            X_train_resampled, y_train_resampled = X_train, y_train

        # Train final model
        best_model = xgb.XGBClassifier(**best_params)
        best_model.fit(X_train_resampled, y_train_resampled)

        # Make predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)

        # Calculate metrics
        metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba, is_binary)

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.xgboost.log_model(best_model, "model")

        # Create and log visualizations
        create_classification_plots(y_test, y_pred, y_pred_proba, best_model, X_train, is_binary)

        logger.info(f"✓ Training complete. Run ID: {run_id}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")

    return run_id, best_model, metrics, X_train, X_test, y_train, y_test


def calculate_classification_metrics(y_true, y_pred, y_pred_proba, is_binary=True):
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        is_binary: Whether it's binary classification

    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary' if is_binary else 'weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary' if is_binary else 'weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary' if is_binary else 'weighted', zero_division=0),
        'matthews_corrcoef': matthews_corrcoef(y_true, y_pred)
    }

    # ROC-AUC calculation
    if is_binary:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
    else:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')

    return metrics


def create_classification_plots(y_true, y_pred, y_pred_proba, model, X_train, is_binary=True):
    """
    Create and log classification visualization plots to MLflow.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        model: Trained XGBoost model
        X_train: Training features
        is_binary: Whether it's binary classification
    """
    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    # Feature Importance
    fig, ax = plt.subplots(figsize=(10, 8))
    importance_df = pd.DataFrame({
        'feature': model.get_booster().feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)

    sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
    ax.set_title('Top 20 Feature Importances')
    ax.set_xlabel('Importance')
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)

    # SHAP Summary Plot (if dataset is not too large)
    if X_train.shape[0] <= 1000:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train[:500])  # Use subset for speed

            fig, ax = plt.subplots(figsize=(10, 8))
            if is_binary:
                shap.summary_plot(shap_values, X_train[:500], show=False)
            else:
                shap.summary_plot(shap_values[1], X_train[:500], show=False)
            mlflow.log_figure(fig, "shap_summary.png")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not create SHAP plot: {e}")


def describe_classification_metrics(metrics: dict):
    """
    Print a formatted description of classification metrics.

    Args:
        metrics: Dictionary of metrics from calculate_classification_metrics
    """
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)
    print(f"  ROC-AUC:              {metrics['roc_auc']:.4f}")
    print(f"  Accuracy:             {metrics['accuracy']:.4f}")
    print(f"  Precision:            {metrics['precision']:.4f}")
    print(f"  Recall:               {metrics['recall']:.4f}")
    print(f"  F1 Score:             {metrics['f1']:.4f}")
    print(f"  Matthews Correlation: {metrics['matthews_corrcoef']:.4f}")
    print("="*60 + "\n")
