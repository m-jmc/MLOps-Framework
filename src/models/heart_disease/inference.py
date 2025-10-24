"""
Heart Disease Model Inference - Simplified Educational Version

This demonstrates real-time ML inference with:
- Loading production models from MLflow registry
- Making predictions with probability scores
- Generating SHAP explanations for interpretability
- Logging inferences for monitoring
"""

import pandas as pd
import yaml
from datetime import datetime


class HeartDiseaseInference:
    """Demonstrates real-time inference with explainability."""

    def __init__(self, config_path="config.yaml", model_alias="champion"):
        """
        Initialize inference pipeline.

        Args:
            config_path: Configuration file path
            model_alias: Which model to use ('champion' or 'challenger')
        """
        self.config = yaml.safe_load(open(config_path))
        self.model_alias = model_alias

        # Load model from MLflow registry
        self.model, self.model_version = self._load_model()

        # Initialize SHAP explainer for model interpretability
        self.explainer = self._initialize_explainer()

    def _load_model(self):
        """
        Load model from MLflow Model Registry.

        In production:
        - Connect to MLflow tracking server
        - Fetch model by alias ('champion', 'challenger')
        - Return model artifact and version metadata
        """
        # Connect to MLflow registry
        mlflow_client = connect_to_mlflow(tracking_uri="...")

        # Load model by alias (e.g., 'champion')
        model_uri = f"models://{self.model_name}@{self.model_alias}"
        model = mlflow.pyfunc.load_model(model_uri)
        version = mlflow_client.get_model_version_by_alias(
            self.model_name,
            self.model_alias
        )

        return model, version

    def _initialize_explainer(self):
        """
        Setup SHAP explainer for model interpretability.

        SHAP (SHapley Additive exPlanations):
        - Explains each prediction by showing feature contributions
        - TreeExplainer optimized for XGBoost/tree-based models
        """
        # Extract underlying XGBoost model
        xgb_model = self.model._model_impl

        # Create SHAP explainer
        explainer = shap.TreeExplainer(xgb_model)

        return explainer

    def predict(self, patient_data, explain=True):
        """
        Make prediction for a single patient.

        Args:
            patient_data: Dict with patient features (age, sex, vitals, etc.)
            explain: Whether to generate SHAP explanation

        Returns:
            dict: Prediction with probability and explanation
        """
        # Convert input to DataFrame
        input_df = pd.DataFrame([patient_data])

        # Reorder columns to match model's training feature order
        input_df = input_df[self.model.feature_names_in_]

        # Generate prediction
        prediction_proba = self.model.predict(input_df)[0, 1]  # P(heart_disease=1)
        prediction_label = int(prediction_proba >= 0.5)

        result = {
            "patient_id": patient_data.get("patient_id"),
            "prediction": prediction_label,
            "probability": float(prediction_proba),
            "risk_level": self._categorize_risk(prediction_proba),
            "model_version": self.model_version.version,
            "timestamp": datetime.now().isoformat()
        }

        # Add SHAP explanation if requested
        if explain and self.explainer:
            explanation = self._generate_explanation(input_df)
            result["explanation"] = explanation

        # Log for monitoring/audit trail
        self._log_inference(result, patient_data)

        return result

    def predict_batch(self, patients_df, explain=False):
        """
        Make predictions for multiple patients (batch inference).

        More efficient than calling predict() in a loop.
        Explanations optional (slower with large batches).
        """
        # Reorder columns
        patients_df = patients_df[self.model.feature_names_in_]

        # Batch prediction
        predictions_proba = self.model.predict(patients_df)[:, 1]
        predictions_label = (predictions_proba >= 0.5).astype(int)

        # Build results DataFrame
        results = pd.DataFrame({
            "prediction": predictions_label,
            "probability": predictions_proba,
            "risk_level": [self._categorize_risk(p) for p in predictions_proba],
            "model_version": self.model_version.version,
            "timestamp": datetime.now().isoformat()
        })

        # Add explanations if requested (compute-intensive)
        if explain and self.explainer:
            shap_values = self.explainer.shap_values(patients_df)
            results["top_contributors"] = self._get_top_features(
                shap_values,
                patients_df.columns
            )

        return results

    def _generate_explanation(self, input_df):
        """
        Generate SHAP explanation showing feature contributions.

        SHAP values indicate how much each feature increases/decreases
        the prediction from the baseline (expected value).

        Example:
            Base risk: 30%
            cholesterol=240 adds +15% (SHAP = +0.15)
            max_heart_rate=180 adds -8% (SHAP = -0.08)
            Final prediction: 37%
        """
        # Calculate SHAP values for this input
        shap_values = self.explainer.shap_values(input_df)

        # Build feature contributions dict
        feature_contributions = {}
        for feature, shap_value in zip(input_df.columns, shap_values[0]):
            feature_contributions[feature] = {
                "value": float(input_df[feature].values[0]),
                "shap_value": float(shap_value),
                "impact": "increases" if shap_value > 0 else "decreases"
            }

        # Sort by absolute impact (most influential features first)
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]["shap_value"]),
            reverse=True
        )

        return {
            "top_contributors": dict(sorted_features[:5]),  # Top 5 features
            "all_features": feature_contributions,
            "base_value": float(self.explainer.expected_value)
        }

    def _categorize_risk(self, probability):
        """
        Convert probability to risk category.

        Risk levels help clinicians prioritize patients:
        - Low (<30%): Routine monitoring
        - Medium (30-70%): Additional testing
        - High (>70%): Immediate intervention
        """
        if probability < 0.3:
            return "low"
        elif probability < 0.7:
            return "medium"
        else:
            return "high"

    def _log_inference(self, result, patient_data):
        """
        Log inference to database for monitoring.

        Captures:
        - Input features and prediction
        - Model version used
        - Timestamp for tracking

        Used for:
        - Drift detection (compare current vs. training distributions)
        - Performance monitoring (when true labels arrive)
        - Audit trail (regulatory compliance)
        """
        # Store in SQLite inference log
        inference_record = {
            "patient_id": result["patient_id"],
            "prediction": result["prediction"],
            "probability": result["probability"],
            "model_version": result["model_version"],
            "timestamp": result["timestamp"],
            "input_features": json.dumps(patient_data)
        }

        # Insert into database
        db.insert("inference_log", inference_record)

    def explain_prediction(self, patient_data, plot=False):
        """
        Generate human-readable explanation for a prediction.

        Prints:
        - Risk level and probability
        - Top contributing factors (with SHAP values)
        - Clinical interpretation

        Optionally generates SHAP force plot visualization.
        """
        result = self.predict(patient_data, explain=True)

        print("=" * 60)
        print("HEART DISEASE RISK ASSESSMENT")
        print("=" * 60)
        print(f"Patient: {result['patient_id']}")
        print(f"Risk: {result['risk_level'].upper()} ({result['probability']:.1%})")
        print(f"\nTop Contributing Factors:")

        explanation = result["explanation"]
        for i, (feature, contrib) in enumerate(explanation["top_contributors"].items(), 1):
            direction = "↑" if contrib["impact"] == "increases" else "↓"
            print(f"{i}. {feature}={contrib['value']} "
                  f"({direction} risk by {abs(contrib['shap_value']):.3f})")

        # Generate SHAP force plot if requested
        if plot:
            self._save_shap_plot(patient_data, result['patient_id'])

        return result


def main():
    """Demo inference pipeline."""
    # Sample patient
    sample_patient = {
        "patient_id": "PT000001",
        "age": 63,
        "sex": 1,  # Male
        "chest_pain_type": 3,
        "resting_bp": 145,
        "cholesterol": 233,
        "max_heart_rate": 150,
        "exercise_angina": 0,
        "st_depression": 2.3
    }

    # Initialize inference pipeline
    inferencer = HeartDiseaseInference(model_alias="champion")

    # Make prediction with explanation
    result = inferencer.explain_prediction(sample_patient, plot=False)

    print(f"\nPrediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.3f}")


if __name__ == '__main__':
    main()
