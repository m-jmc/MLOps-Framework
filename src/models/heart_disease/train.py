"""
Heart Disease Model Training - Simplified Educational Version

This demonstrates the core MLOps training pipeline:
- Load features from feature store
- Train model with hyperparameter optimization
- Check for bias across demographic groups
- Evaluate and promote models (champion/challenger pattern)
"""

from pathlib import Path
import yaml
import pandas as pd


class HeartDiseaseTrainer:
    """Demonstrates end-to-end model training workflow."""

    def __init__(self, config_path="config.yaml"):
        """Initialize with configuration."""
        # Load configuration from YAML
        self.config = yaml.safe_load(open(config_path))

        # Extract key sections
        self.model_config = self.config['model']
        self.mlflow_config = self.config['mlflow']

        # Initialize MLflow for experiment tracking
        self.mlflow_client = self._initialize_mlflow()

    def _initialize_mlflow(self):
        """Setup MLflow tracking and experiment."""
        # Connect to MLflow tracking server
        # Create experiment if it doesn't exist
        # Return client for logging metrics/models
        pass

    def load_training_data(self):
        """
        Load features from FEAST feature store.

        In production: FEAST retrieves point-in-time correct features
        For demo: Load from parquet files (base_data, entity_data, labels)
        """
        # Load demographic features (patient_id, age, sex, etc.)
        base_df = pd.read_parquet("path/to/base_data/patients.parquet")

        # Load clinical measurements (monthly vitals, labs)
        entity_df = pd.read_parquet("path/to/entity_data/monthly_data.parquet")

        # Load target labels (heart_disease: 0 or 1)
        label_df = pd.read_parquet("path/to/label_data/outcomes.parquet")

        # Join all sources on patient_id and month_key
        training_df = entity_df.merge(base_df, on='patient_id')
        training_df = training_df.merge(label_df, on=['patient_id', 'month_key'])

        return training_df

    def train_model(self, training_df):
        """
        Train XGBoost classifier with hyperparameter optimization.

        Steps:
        1. Split data into train/test (80/20)
        2. Handle class imbalance with SMOTE if needed
        3. Use Hyperopt for hyperparameter search (50 iterations)
        4. Train XGBoost with best parameters
        5. Evaluate on test set (ROC-AUC, precision, recall)
        6. Log everything to MLflow
        """
        # Separate features and target
        X = training_df.drop(columns=['patient_id', 'month_key', 'heart_disease'])
        y = training_df['heart_disease']

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Define hyperparameter search space
        param_space = {
            'max_depth': range(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'n_estimators': range(100, 500)
        }

        # Optimize hyperparameters (Hyperopt finds best combo)
        best_params = hyperopt_search(X_train, y_train, param_space)

        # Train final model with best parameters
        model = XGBoostClassifier(**best_params)
        model.fit(X_train, y_train)

        # Evaluate performance
        y_pred = model.predict(X_test)
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }

        # Log to MLflow (params, metrics, model artifact)
        run_id = self._log_to_mlflow(model, metrics, best_params)

        return run_id, model, metrics, X_test, y_test

    def check_bias(self, model, X_test, y_test, training_df):
        """
        Detect bias across protected demographic groups.

        For each protected attribute (sex, age_group):
        - Calculate per-group metrics (accuracy, precision, recall)
        - Check demographic parity (positive prediction rates)
        - Flag disparities > 10% threshold
        - Log bias metrics to MLflow
        """
        protected_attributes = ['sex', 'age_group']
        bias_results = {}

        for attribute in protected_attributes:
            for group in training_df[attribute].unique():
                # Get predictions for this demographic group
                group_mask = training_df[attribute] == group
                group_metrics = calculate_group_metrics(
                    y_test[group_mask],
                    model.predict(X_test[group_mask])
                )

                # Store results for comparison
                bias_results[f"{attribute}_{group}"] = group_metrics

        # Check for disparities between groups
        self._evaluate_fairness(bias_results)

        return bias_results

    def evaluate_promotion(self, run_id, metrics):
        """
        Decide if new model should be promoted to production.

        Champion/Challenger Pattern:
        - Compare new model (challenger) to current production (champion)
        - Promotion criteria: ≥2% improvement in ROC-AUC + no bias violations
        - If no champion exists: auto-promote as first champion
        - If challenger wins: promote to 'challenger' alias (requires approval)
        - If champion wins: register new model but don't promote
        """
        # Try to load current champion from MLflow registry
        champion_model, champion_version = self._get_champion()

        # No champion? Auto-promote
        if champion_model is None:
            self._register_model(run_id, alias="champion")
            return {"promoted": True, "reason": "First champion"}

        # Compare metrics
        champion_metrics = self._get_champion_metrics(champion_version)
        improvement = metrics['roc_auc'] - champion_metrics['roc_auc']

        # Check promotion criteria
        if improvement >= 0.02:  # 2% improvement threshold
            self._register_model(run_id, alias="challenger")
            return {"promoted": True, "reason": f"Improved by {improvement:.1%}"}
        else:
            self._register_model(run_id, alias=None)
            return {"promoted": False, "reason": "Insufficient improvement"}

    def run_training_pipeline(self):
        """Execute complete training workflow."""
        print("=" * 60)
        print("STARTING TRAINING PIPELINE")
        print("=" * 60)

        # Step 1: Load data from feature store
        training_df = self.load_training_data()

        # Step 2: Train model with hyperparameter optimization
        run_id, model, metrics, X_test, y_test = self.train_model(training_df)

        # Step 3: Check for bias
        bias_results = self.check_bias(model, X_test, y_test, training_df)

        # Step 4: Evaluate promotion
        promotion_result = self.evaluate_promotion(run_id, metrics)

        print("=" * 60)
        print(f"TRAINING COMPLETE")
        print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        print(f"Promoted: {promotion_result['promoted']}")
        print("=" * 60)

        return {
            "run_id": run_id,
            "metrics": metrics,
            "bias_results": bias_results,
            "promotion": promotion_result
        }


def main():
    """Main entry point for training."""
    trainer = HeartDiseaseTrainer()
    summary = trainer.run_training_pipeline()
    return summary


if __name__ == '__main__':
    main()
