"""
Generic Classification Model - Training Pipeline

This module implements a reusable training pipeline for classification models,
including data loading from feature store, model training with hyperparameter optimization,
evaluation, bias detection, and model promotion logic.

"""

import yaml
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Generic training pipeline for classification models.
    
    Workflow:
    1. Load data from feature store
    2. Train model with hyperparameter optimization
    3. Evaluate performance and bias
    4. Promote model based on business rules
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration.
        
        Parameters:
        - config_path: Path to model configuration file
        
        Configuration should include:
        - model: name, version, type
        - mlflow: tracking_uri, experiment_name, model_name
        - feature_store: repo_path, feature_service
        - data: target_column, drop_columns
        - training: test_size, random_seed, hyperparameters
        - evaluation: primary_metric, promotion criteria
        - bias: protected_attributes, thresholds
        """
        logger.info("Initializing Model Trainer")
        
        # Load configuration
        config_path = Path(config_path)
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.mlflow_config = self.config['mlflow']
        self.feature_store_config = self.config['feature_store']
        self.data_config = self.config['data']
        self.training_config = self.config['training']
        self.evaluation_config = self.config['evaluation']
        self.bias_config = self.config.get('bias', {})
        
        logger.info(f"Model: {self.model_config['name']}")
        logger.info(f"Version: {self.model_config['version']}")
        
        # Initialize experiment tracking client
        self.tracking_client = initialize_experiment_tracking(
            tracking_uri=self.mlflow_config['tracking_uri'],
            experiment_name=self.mlflow_config['experiment_name']
        )
    
    
    def load_training_data(self) -> pd.DataFrame:
        """
        Load training data from feature store.
        
        Workflow:
        1. Connect to feature store
        2. Retrieve historical features with point-in-time correctness
        3. Join with target labels
        4. Clean and prepare data
        
        In production, uses feature store's historical API.
        In development, may load from data files.
        
        Similar to prepare_training_data() from inference_utils.
        
        Returns:
        - DataFrame with features and target variable
        """
        logger.info("Loading training data from feature store...")
        
        # Connect to feature store
        feature_store = connect_to_feature_store(
            repo_path=self.feature_store_config['repo_path']
        )
        
        # Load entity dataframe (what to get features for)
        entity_data = load_entity_data(
            data_source=self.feature_store_config['entity_data_path']
        )
        
        # Retrieve historical features with timestamps
        feature_data = feature_store.get_historical_features(
            entity_df=entity_data,
            features=self.feature_store_config['feature_service']
        ).to_dataframe()
        
        # Load target labels
        target_data = load_target_data(
            data_source=self.data_config['label_data_path'],
            target_column=self.data_config['target_column']
        )
        
        # Join features with labels
        training_data = merge_features_and_targets(
            feature_data=feature_data,
            target_data=target_data,
            join_keys=self.data_config['entity_keys']
        )
        
        # Data cleaning
        training_data = remove_columns(
            data=training_data,
            columns_to_drop=self.data_config['drop_columns']
        )
        training_data = remove_missing_targets(
            data=training_data,
            target_column=self.data_config['target_column']
        )
        
        logger.info(f"Training data shape: {training_data.shape}")
        logger.info(f"Target distribution:\n{training_data[self.data_config['target_column']].value_counts()}")
        
        return training_data
    
    
    def train_model(self, training_data: pd.DataFrame) -> Tuple[str, object, Dict, pd.DataFrame, pd.Series]:
        """
        Train classification model with hyperparameter optimization.
        
        Workflow:
        1. Split data into train/test sets
        2. Handle class imbalance if configured
        3. Perform hyperparameter search
        4. Train final model on best parameters
        5. Evaluate on test set
        6. Log to experiment tracker
        
        Parameters:
        - training_data: DataFrame with features and target
        
        Returns:
        - Tuple of (experiment_run_id, trained_model, metrics, test_features, test_labels)
        """
        logger.info("="*60)
        logger.info("STARTING MODEL TRAINING")
        logger.info("="*60)
        
        # Train model using configured algorithm
        run_id, model, metrics, X_test, y_test = train_classifier(
            data=training_data,
            target_column=self.data_config['target_column'],
            algorithm=self.model_config['algorithm'],
            test_size=self.training_config['test_size'],
            random_state=self.training_config['random_seed'],
            handle_imbalance=self.training_config.get('use_smote', False),
            hyperparameter_search=self.training_config['hyperopt']
        )
        
        logger.info("="*60)
        logger.info("MODEL TRAINING COMPLETE")
        logger.info("="*60)
        log_classification_metrics(metrics)
        
        return run_id, model, metrics, X_test, y_test
    
    
    def detect_bias(
        self,
        model: object,
        test_features: pd.DataFrame,
        test_labels: pd.Series,
        original_data: pd.DataFrame
    ) -> Dict:
        """
        Detect bias in model predictions across protected attributes.
        
        Analyzes fairness metrics across demographic groups:
        - Demographic parity: Equal positive prediction rates
        - Equalized odds: Equal TPR and FPR across groups
        - Performance disparity: Equal accuracy/AUC across groups
        
        Similar to evaluate_target_drift() for comparing distributions,
        but focused on fairness rather than temporal drift.
        
        Parameters:
        - model: Trained model
        - test_features: Test set features
        - test_labels: Test set labels
        - original_data: Full dataset (contains protected attributes)
        
        Returns:
        - Dictionary of bias metrics by group
        """
        if not self.bias_config.get('enabled', False):
            logger.info("Bias detection disabled in configuration")
            return {}
        
        logger.info("Analyzing model fairness across protected attributes...")
        
        # Get protected attributes for test set
        test_indices = test_features.index
        test_with_protected = original_data.loc[test_indices].copy()
        
        # Generate predictions
        predictions = model.predict(test_features)
        prediction_probabilities = model.predict_proba(test_features)[:, 1]
        
        bias_results = {}
        
        # Analyze each protected attribute
        for attribute in self.bias_config.get('protected_attributes', []):
            if attribute not in test_with_protected.columns:
                logger.warning(f"Protected attribute '{attribute}' not found")
                continue
            
            logger.info(f"\nAnalyzing fairness for: {attribute}")
            
            # Calculate metrics for each demographic group
            for group_value in test_with_protected[attribute].unique():
                group_mask = test_with_protected[attribute] == group_value
                group_size = group_mask.sum()
                
                if group_size == 0:
                    continue
                
                # Calculate group-specific performance metrics
                group_metrics = calculate_classification_metrics(
                    true_labels=test_labels[group_mask],
                    predicted_labels=predictions[group_mask],
                    prediction_probabilities=prediction_probabilities[group_mask]
                )
                
                # Calculate fairness metrics
                positive_prediction_rate = predictions[group_mask].mean()
                
                bias_key = f"{attribute}_{group_value}"
                bias_results[bias_key] = {
                    'group_size': int(group_size),
                    'accuracy': group_metrics['accuracy'],
                    'precision': group_metrics['precision'],
                    'recall': group_metrics['recall'],
                    'roc_auc': group_metrics['roc_auc'],
                    'positive_prediction_rate': float(positive_prediction_rate)
                }
                
                logger.info(f"  {attribute}={group_value} (n={group_size}):")
                logger.info(f"    Performance: {group_metrics['roc_auc']:.4f}")
                logger.info(f"    Positive Rate: {positive_prediction_rate:.4f}")
        
        # Log bias metrics to experiment tracker
        log_bias_metrics(bias_results)
        
        logger.info("✓ Bias analysis complete")
        
        return bias_results
    
    
    def evaluate_promotion(self, run_id: str, metrics: Dict) -> Dict:
        """
        Evaluate whether model should be promoted to production.
        
        Promotion logic:
        1. Check if champion model exists
        2. If no champion, promote immediately
        3. If champion exists, compare performance
        4. Promote if improvement exceeds threshold
        
        Promotion tiers:
        - Champion: Current production model
        - Challenger: Candidate for production (outperforms champion)
        - Archived: Historical models for reference
        
        Similar to evaluate_dataset_drift() for threshold-based decisions.
        
        Parameters:
        - run_id: Experiment run identifier
        - metrics: Performance metrics of current model
        
        Returns:
        - Dictionary with promotion decision and metadata
        """
        logger.info("Evaluating model promotion criteria...")
        
        model_name = self.mlflow_config['model_name']
        
        # Attempt to load current champion
        champion_model, champion_metadata = load_model_by_alias(
            tracking_client=self.tracking_client,
            model_name=model_name,
            alias="champion"
        )
        
        # No champion exists - promote immediately
        if champion_model is None:
            logger.info("No existing champion found - promoting immediately")
            
            model_version = register_model(
                run_id=run_id,
                model_name=model_name,
                alias="champion"
            )
            
            return {
                "promoted": True,
                "alias": "champion",
                "reason": "First model - no existing champion",
                "version": model_version.version
            }
        
        # Compare with existing champion
        champion_run = get_experiment_run(
            tracking_client=self.tracking_client,
            run_id=champion_metadata.run_id
        )
        champion_metrics = extract_metrics_from_run(champion_run)
        
        # Calculate improvement
        comparison = compare_model_performance(
            champion_metrics=champion_metrics,
            challenger_metrics=metrics,
            primary_metric=self.evaluation_config['primary_metric']
        )
        
        logger.info(f"Champion {self.evaluation_config['primary_metric']}: {comparison['champion_score']:.4f}")
        logger.info(f"Challenger {self.evaluation_config['primary_metric']}: {comparison['challenger_score']:.4f}")
        logger.info(f"Improvement: {comparison['improvement_percentage']:.2f}%")
        
        # Check promotion threshold
        min_improvement = self.evaluation_config['promotion']['min_improvement']
        meets_criteria = comparison['improvement'] >= min_improvement
        
        if meets_criteria:
            logger.info(f"✓ Meets promotion criteria (improvement >= {min_improvement})")
            
            # Register as challenger for staged rollout
            model_version = register_model(
                run_id=run_id,
                model_name=model_name,
                alias="challenger"
            )
            
            return {
                "promoted": True,
                "alias": "challenger",
                "reason": f"Outperforms champion by {comparison['improvement_percentage']:.2f}%",
                "version": model_version.version,
                "comparison": comparison
            }
        else:
            logger.info(f"✗ Does not meet criteria (improvement < {min_improvement})")
            
            # Register for tracking but don't promote
            model_version = register_model(
                run_id=run_id,
                model_name=model_name
            )
            
            return {
                "promoted": False,
                "alias": None,
                "reason": f"Insufficient improvement ({comparison['improvement_percentage']:.2f}%)",
                "version": model_version.version,
                "comparison": comparison
            }
    
    
    def execute_training_pipeline(self) -> Dict:
        """
        Orchestrates the complete training pipeline.
        
        Pipeline stages:
        1. Data Loading: Retrieve features and labels from feature store
        2. Model Training: Train with hyperparameter optimization
        3. Bias Detection: Analyze fairness across protected groups
        4. Model Promotion: Evaluate promotion to production
        5. Reporting: Generate training summary
        
        Returns:
        - Dictionary with training summary and results
        """
        logger.info("\n" + "="*80)
        logger.info("CLASSIFICATION MODEL TRAINING PIPELINE")
        logger.info("="*80)
        logger.info(f"Model: {self.model_config['name']}")
        logger.info(f"Version: {self.model_config['version']}")
        logger.info(f"Experiment: {self.mlflow_config['experiment_name']}")
        logger.info("="*80 + "\n")
        
        # Stage 1: Load training data
        training_data = self.load_training_data()
        
        # Stage 2: Train model
        run_id, model, metrics, test_features, test_labels = self.train_model(training_data)
        
        # Stage 3: Bias detection
        bias_results = self.detect_bias(model, test_features, test_labels, training_data)
        
        # Stage 4: Promotion evaluation
        promotion_result = self.evaluate_promotion(run_id, metrics)
        
        # Generate summary
        summary = {
            "run_id": run_id,
            "model_name": self.model_config['name'],
            "model_version": self.model_config['version'],
            "metrics": metrics,
            "bias_analysis": bias_results,
            "promotion_decision": promotion_result,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("="*80)
        logger.info(f"Experiment Run ID: {run_id}")
        logger.info(f"Model Performance ({self.evaluation_config['primary_metric']}): {metrics[self.evaluation_config['primary_metric']]:.4f}")
        logger.info(f"Promotion Status: {'✓ Promoted' if promotion_result['promoted'] else '✗ Not Promoted'}")
        if promotion_result['promoted']:
            logger.info(f"Promoted As: {promotion_result['alias']}")
            logger.info(f"Model Version: {promotion_result['version']}")
        logger.info("="*80 + "\n")
        
        return summary


def train_classification_model(config_path: str) -> Dict:
    """
    Convenience function to execute training pipeline.
    
    Parameters:
    - config_path: Path to model configuration file
    
    Returns:
    - Training summary dictionary
    """
    trainer = ModelTrainer(config_path)
    summary = trainer.execute_training_pipeline()
    return summary


# Example usage patterns
def main():
    """Main entry point for training."""
    try:
        # For diabetes risk score model
        # summary = train_classification_model("config/diabetes_model_config.yaml")
        
        # For heart disease model
        # summary = train_classification_model("config/heart_disease_config.yaml")
        
        # Generic usage
        summary = train_classification_model("config/model_config.yaml")
        
        # Print summary
        print("\nTraining Summary:")
        print(f"  Run ID: {summary['run_id']}")
        print(f"  Primary Metric: {summary['metrics'][summary['model_name']]:.4f}")
        print(f"  Promoted: {summary['promotion_decision']['promoted']}")
        
        return 0
        
    except Exception as error:
        logger.error(f"Training failed: {error}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())