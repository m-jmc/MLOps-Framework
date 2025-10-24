"""
Generic Classification Model - Inference Pipeline

Implements real-time inference with explainability and logging for monitoring.

Usage:
    from inference import ModelInference
    
    inferencer = ModelInference(config_path, model_alias="champion")
    result = inferencer.predict(input_data)
"""

import yaml
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelInference:
    """
    Real-time inference pipeline with explainability.
    
    Provides:
    - Model loading from registry
    - Single and batch predictions
    - Explainable predictions (SHAP values)
    - Risk level categorization
    - Inference logging for monitoring
    
    References perform_inference() from inference_utils for similar workflow.
    """
    
    def __init__(self, config_path: str, model_alias: str = "champion"):
        """
        Initialize inference pipeline.
        
        Parameters:
        - config_path: Path to model configuration file
        - model_alias: Model version to load ('champion', 'challenger', or 'latest')
        
        Workflow:
        1. Load configuration
        2. Connect to model registry
        3. Load specified model version
        4. Initialize explainability framework
        """
        logger.info("Initializing Inference Pipeline")
        
        # Load configuration
        self.config = load_config(config_path)
        self.model_name = self.config['model']['name']
        self.model_alias = model_alias
        
        # Connect to model registry
        # Similar to diabetes_model_production_champion() from inference_utils
        self.registry_client = connect_to_model_registry(
            registry_uri=self.config['registry']['tracking_uri']
        )
        
        # Load model
        self.model, self.model_metadata = self._load_model()
        
        # Initialize explainability
        self.explainer = None
        if self.model:
            self._initialize_explainer()
        
        logger.info(f"✓ Pipeline ready (model: {self.model_alias} v{self.model_metadata.version})")
    
    
    def _load_model(self):
        """
        Load model from registry by alias.
        
        Similar to load_model_by_alias() from drift_detector and inference_utils.
        
        Returns:
        - Tuple of (model_object, model_metadata)
        """
        logger.info(f"Loading model '{self.model_name}' with alias '{self.model_alias}'")
        
        model, metadata = load_model_by_alias(
            registry_client=self.registry_client,
            model_name=self.model_name,
            alias=self.model_alias
        )
        
        if model is None:
            logger.warning(f"No model found with alias '{self.model_alias}'")
            return None, None
        
        return model, metadata
    
    
    def _initialize_explainer(self):
        """
        Initialize explainability framework for model interpretations.
        
        Uses SHAP (SHapley Additive exPlanations) for:
        - Feature importance
        - Individual prediction explanations
        - Model debugging
        """
        try:
            # Extract underlying model for explainer
            underlying_model = extract_underlying_model(self.model)
            
            # Initialize tree explainer (for tree-based models)
            self.explainer = create_tree_explainer(underlying_model)
            
            logger.info("✓ Explainability framework initialized")
            
        except Exception as error:
            logger.warning(f"Could not initialize explainer: {error}")
            self.explainer = None
    
    
    def predict(self, input_data: Dict, include_explanation: bool = True) -> Dict:
        """
        Generate prediction for single entity with optional explanation.
        
        Workflow:
        1. Validate and prepare input data
        2. Generate prediction probability
        3. Classify into risk categories
        4. Generate explanation (if requested)
        5. Log inference for monitoring
        
        Similar to perform_inference() from inference_utils.
        
        Parameters:
        - input_data: Dictionary with entity features
        - include_explanation: Whether to generate SHAP explanation
        
        Returns:
        - Dictionary with prediction, probability, risk level, and explanation
        """
        if self.model is None:
            raise ValueError(f"No model loaded with alias '{self.model_alias}'")
        
        # Convert to DataFrame format
        input_df = convert_to_dataframe(input_data)
        
        # Ensure feature order matches model training
        input_df = align_features_with_model(input_df, self.model)
        
        # Generate prediction
        prediction_probability = self.model.predict_proba(input_df)[0, 1]
        prediction_label = int(prediction_probability >= 0.5)
        
        # Build result
        result = {
            "entity_id": input_data.get("entity_id", "unknown"),
            "prediction": prediction_label,
            "probability": float(prediction_probability),
            "risk_level": categorize_risk(prediction_probability),
            "model_version": self.model_metadata.version,
            "model_alias": self.model_alias,
            "timestamp": get_current_timestamp()
        }
        
        # Generate explanation if requested
        if include_explanation and self.explainer:
            explanation = self._generate_explanation(input_df)
            result["explanation"] = explanation
        
        # Log inference for monitoring
        # Similar to log_inference_to_db() from inference_utils
        log_inference_to_database(result, input_data)
        
        return result
    
    
    def predict_batch(self, input_dataframe: pd.DataFrame, include_explanation: bool = False) -> pd.DataFrame:
        """
        Generate predictions for multiple entities.
        
        Parameters:
        - input_dataframe: DataFrame with entity features
        - include_explanation: Whether to generate explanations (slower for large batches)
        
        Returns:
        - DataFrame with predictions, probabilities, and optional explanations
        """
        if self.model is None:
            raise ValueError(f"No model loaded with alias '{self.model_alias}'")
        
        logger.info(f"Batch prediction for {len(input_dataframe)} entities")
        
        # Align features
        aligned_data = align_features_with_model(input_dataframe, self.model)
        
        # Generate predictions
        prediction_probabilities = self.model.predict_proba(aligned_data)[:, 1]
        prediction_labels = (prediction_probabilities >= 0.5).astype(int)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            "prediction": prediction_labels,
            "probability": prediction_probabilities,
            "risk_level": [categorize_risk(p) for p in prediction_probabilities],
            "model_version": self.model_metadata.version,
            "timestamp": get_current_timestamp()
        })
        
        # Add entity IDs if available
        if "entity_id" in input_dataframe.columns:
            results_df["entity_id"] = input_dataframe["entity_id"].values
        
        # Generate explanations if requested
        if include_explanation and self.explainer:
            logger.info("Generating explanations for batch...")
            results_df["top_features"] = generate_batch_explanations(
                explainer=self.explainer,
                data=aligned_data
            )
        
        logger.info("✓ Batch prediction complete")
        
        return results_df
    
    
    def _generate_explanation(self, input_df: pd.DataFrame) -> Dict:
        """
        Generate feature contribution explanation using SHAP.
        
        Explains which features drove the prediction and by how much.
        Critical for clinical trust and model debugging.
        
        Parameters:
        - input_df: Single-row DataFrame with features
        
        Returns:
        - Dictionary with feature contributions and rankings
        """
        try:
            # Calculate SHAP values
            shap_values = self.explainer.compute_shap_values(input_df)
            
            # Extract feature contributions
            feature_contributions = {}
            for feature_name, shap_value in zip(input_df.columns, shap_values[0]):
                feature_contributions[feature_name] = {
                    "input_value": float(input_df[feature_name].values[0]),
                    "contribution": float(shap_value),
                    "direction": "increases_risk" if shap_value > 0 else "decreases_risk"
                }
            
            # Rank by absolute contribution
            ranked_features = rank_features_by_importance(feature_contributions)
            
            explanation = {
                "top_contributors": ranked_features[:5],  # Top 5 features
                "all_features": feature_contributions,
                "baseline_value": float(self.explainer.expected_value)
            }
            
            return explanation
            
        except Exception as error:
            logger.warning(f"Could not generate explanation: {error}")
            return None
    
    
    def explain_prediction(self, input_data: Dict, save_visualization: bool = False) -> Dict:
        """
        Generate detailed human-readable explanation for prediction.
        
        Provides:
        - Risk assessment summary
        - Top contributing factors
        - Feature-level analysis
        - Optional visualization export
        
        Parameters:
        - input_data: Dictionary with entity features
        - save_visualization: Whether to save SHAP plot
        
        Returns:
        - Prediction result with formatted explanation
        """
        # Generate prediction with explanation
        result = self.predict(input_data, include_explanation=True)
        
        if not result.get("explanation"):
            logger.warning("Explanation not available")
            return result
        
        # Format human-readable output
        print("\n" + "="*60)
        print("RISK ASSESSMENT")
        print("="*60)
        print(f"Entity ID: {result['entity_id']}")
        print(f"Risk Level: {result['risk_level'].upper()} ({result['probability']:.1%})")
        print(f"Classification: {'High Risk' if result['prediction'] == 1 else 'Low Risk'}")
        print(f"\nModel: {result['model_alias']} (version {result['model_version']})")
        print("="*60)
        
        # Display top contributors
        explanation = result["explanation"]
        print("\nKey Risk Factors:")
        print("-"*60)
        
        for i, (feature, contrib) in enumerate(explanation["top_contributors"].items(), 1):
            direction_symbol = "↑" if contrib["direction"] == "increases_risk" else "↓"
            print(f"{i}. {feature}: {contrib['input_value']} "
                  f"({direction_symbol} risk by {abs(contrib['contribution']):.3f})")
        
        print("="*60 + "\n")
        
        # Generate visualization if requested
        if save_visualization and self.explainer:
            save_explanation_plot(
                explainer=self.explainer,
                input_data=input_data,
                entity_id=result['entity_id']
            )
        
        return result
    
    
    def log_inference_to_database(self, result: Dict, input_data: Dict):
        """
        Persist inference to database for monitoring and audit.
        
        Logging enables:
        - Performance monitoring over time
        - Drift detection (prediction drift)
        - Audit trail for regulatory compliance
        - Debugging production issues
        
        Similar to log_inference_to_db() from inference_utils.
        
        Parameters:
        - result: Prediction result dictionary
        - input_data: Original input features
        """
        try:
            # Connect to inference database
            inference_db = connect_to_inference_database(
                self.config['monitoring']['database_path']
            )
            
            # Create table if not exists
            create_inference_log_table(inference_db)
            
            # Insert inference record
            insert_inference_record(
                database=inference_db,
                entity_id=result.get("entity_id"),
                prediction=result.get("prediction"),
                probability=result.get("probability"),
                risk_level=result.get("risk_level"),
                model_version=result.get("model_version"),
                model_alias=result.get("model_alias"),
                timestamp=result.get("timestamp"),
                input_features=serialize_to_json(input_data)
            )
            
            logger.debug("Inference logged to database")
            
        except Exception as error:
            logger.warning(f"Could not log inference: {error}")


# Helper functions

def categorize_risk(probability: float) -> str:
    """
    Categorize prediction probability into risk levels.
    
    Thresholds should be calibrated based on:
    - Clinical cost-benefit analysis
    - Institutional risk tolerance
    - Regulatory requirements
    
    Parameters:
    - probability: Prediction probability (0-1)
    
    Returns:
    - Risk level string: 'low', 'medium', or 'high'
    """
    if probability < 0.3:
        return "low"
    elif probability < 0.7:
        return "medium"
    else:
        return "high"


def create_inference_log_table(database_connection):
    """
    Creates schema for inference logging.
    
    Table structure captures:
    - Prediction metadata (timestamp, model version)
    - Prediction output (label, probability, risk level)
    - Input features (for drift detection)
    - Entity identifier (for joining with ground truth)
    """
    schema = """
        CREATE TABLE IF NOT EXISTS inference_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id TEXT,
            prediction INTEGER,
            probability REAL,
            risk_level TEXT,
            model_version TEXT,
            model_alias TEXT,
            timestamp TEXT,
            input_features TEXT
        )
    """
    execute_sql(database_connection, schema)


def generate_batch_explanations(explainer, data: pd.DataFrame) -> List[str]:
    """
    Generate simplified explanations for batch predictions.
    
    Returns top contributing features for each prediction.
    """
    shap_values = explainer.compute_shap_values(data)
    
    top_features_list = []
    for shap_row in shap_values:
        # Get top 3 features by absolute contribution
        top_indices = np.argsort(np.abs(shap_row))[-3:][::-1]
        top_features = [data.columns[i] for i in top_indices]
        top_features_list.append(", ".join(top_features))
    
    return top_features_list


# Example usage
def main():
    """Demo inference pipeline for diabetes risk model."""
    logger.info("Diabetes Risk Inference Pipeline - Demo")
    
    # Sample entity data
    sample_input = {
        "entity_id": "PATIENT_12345",
        "age": 55,
        "bmi": 31.2,
        "fasting_glucose": 126,
        "hba1c": 6.2,
        "systolic_bp": 140,
        "diastolic_bp": 88,
        "family_history": 1,
        "physical_activity": 2
    }
    
    try:
        # Initialize inference pipeline
        inferencer = ModelInference(
            config_path="config/model_config.yaml",
            model_alias="champion"
        )
        
        # Generate prediction with explanation
        result = inferencer.explain_prediction(sample_input, save_visualization=False)
        
        print("\nInference Result:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Probability: {result['probability']:.3f}")
        print(f"  Risk Level: {result['risk_level']}")
        
    except Exception as error:
        logger.error(f"Inference failed: {error}", exc_info=True)
        print("\nNote: Ensure model is trained and registered first.")


if __name__ == '__main__':
    main()