"""
Drift Detection - Simplified Educational Version

Demonstrates monitoring for ML model drift:
- Dataset drift: Changes in input feature distributions
- Prediction drift: Changes in model output distributions
- Target drift: Changes in actual outcome distributions

Why monitor drift?
- Models degrade when data patterns change over time
- Early detection prevents poor predictions in production
- Triggers retraining when needed
"""

import pandas as pd
import numpy as np
from datetime import datetime


class DriftDetector:
    """Monitors for drift in ML models."""

    def __init__(self, model_name, drift_threshold=0.1):
        """
        Initialize drift detector.

        Args:
            model_name: Name of model to monitor
            drift_threshold: Threshold for significant drift (0-1)
        """
        self.model_name = model_name
        self.drift_threshold = drift_threshold

    def detect_dataset_drift(self, reference_data, current_data):
        """
        Detect drift in input features.

        Compares current feature distributions to reference (training) data.
        Uses Evidently AI library for statistical tests.

        Args:
            reference_data: Baseline dataset (e.g., training data)
            current_data: Current production data

        Returns:
            dict: Drift detection results
        """
        # Create Evidently report for drift analysis
        report = evidently.Report(metrics=[
            evidently.DataDriftPreset()
        ])

        # Run drift detection
        report.run(
            reference_data=reference_data,
            current_data=current_data
        )

        # Extract results
        result_dict = report.as_dict()
        drift_metrics = self._extract_drift_metrics(result_dict)

        # Save HTML report for visualization
        report.save_html(f"drift_report_{self.model_name}.html")

        # Determine if drift is significant
        drift_score = drift_metrics['share_of_drifted_columns']
        drift_detected = drift_score > self.drift_threshold

        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'drifted_columns': drift_metrics['drifted_columns'],
            'num_drifted': len(drift_metrics['drifted_columns']),
            'timestamp': datetime.now().isoformat()
        }

    def detect_prediction_drift(self, reference_predictions, current_predictions):
        """
        Detect drift in model predictions.

        Uses Jensen-Shannon divergence to compare prediction distributions.

        JS divergence measures:
        - 0 = identical distributions
        - 1 = completely different distributions

        Args:
            reference_predictions: Historical predictions (probabilities)
            current_predictions: Recent predictions (probabilities)

        Returns:
            dict: Prediction drift results
        """
        # Create histograms of predictions
        bins = np.linspace(0, 1, 20)  # 20 bins for probability range
        ref_hist, _ = np.histogram(reference_predictions, bins=bins, density=True)
        curr_hist, _ = np.histogram(current_predictions, bins=bins, density=True)

        # Normalize to probability distributions
        ref_hist = ref_hist / ref_hist.sum()
        curr_hist = curr_hist / curr_hist.sum()

        # Calculate Jensen-Shannon divergence
        js_divergence = scipy.spatial.distance.jensenshannon(ref_hist, curr_hist)

        # Check if divergence exceeds threshold
        drift_detected = js_divergence > self.drift_threshold

        return {
            'drift_detected': drift_detected,
            'js_divergence': float(js_divergence),
            'reference_mean': float(np.mean(reference_predictions)),
            'current_mean': float(np.mean(current_predictions)),
            'timestamp': datetime.now().isoformat()
        }

    def detect_target_drift(self, reference_data, current_data, target_column):
        """
        Detect drift in target variable.

        Important for detecting concept drift (relationship changes).

        Example:
        - Training data: 40% positive class
        - Current data: 70% positive class
        - → Target drift detected, model may need retraining

        Args:
            reference_data: Reference dataset with target
            current_data: Current dataset with target
            target_column: Name of target column

        Returns:
            dict: Target drift results
        """
        # Use Evidently for target drift analysis
        report = evidently.Report(metrics=[
            evidently.TargetDriftPreset()
        ])

        column_mapping = {'target': target_column}

        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )

        # Extract drift score
        result_dict = report.as_dict()
        drift_score = self._extract_target_drift_score(result_dict)

        drift_detected = drift_score > self.drift_threshold

        return {
            'drift_detected': drift_detected,
            'drift_score': float(drift_score),
            'timestamp': datetime.now().isoformat()
        }

    def _extract_drift_metrics(self, result_dict):
        """
        Extract drift metrics from Evidently report.

        Evidently provides:
        - Statistical test results per feature
        - Overall drift percentage
        - List of drifted features
        """
        # Navigate Evidently's nested structure
        metrics = result_dict['metrics'][0]['result']

        return {
            'share_of_drifted_columns': metrics['share_of_drifted_columns'],
            'drifted_columns': [
                col for col, info in metrics['drift_by_columns'].items()
                if info['drift_detected']
            ]
        }

    def generate_drift_alert(self, results, drift_type):
        """
        Generate alert message for drift detection.

        Used to:
        - Notify data scientists of model degradation
        - Create GitHub issues for retraining
        - Send alerts to monitoring dashboards

        Args:
            results: Drift detection results
            drift_type: Type of drift ('dataset', 'prediction', 'target')

        Returns:
            dict: Alert message and metadata
        """
        if not results['drift_detected']:
            return {
                'alert': False,
                'message': f"No {drift_type} drift detected"
            }

        # Generate alert message
        message = f"""
        ⚠️ DRIFT ALERT: {drift_type.upper()} DRIFT DETECTED

        Model: {self.model_name}
        Drift Score: {results.get('drift_score', results.get('js_divergence', 0)):.2%}
        Threshold: {self.drift_threshold:.2%}

        Action Required:
        1. Review drift report
        2. Analyze affected features
        3. Consider model retraining
        4. Update data pipelines if needed
        """

        return {
            'alert': True,
            'message': message,
            'severity': 'high' if results['drift_score'] > (self.drift_threshold * 2) else 'medium',
            'recommended_actions': self._get_recommended_actions(drift_type)
        }

    def _get_recommended_actions(self, drift_type):
        """
        Get recommended actions based on drift type.

        Different drift types require different responses.
        """
        actions = {
            'dataset': [
                "Review feature distributions in dashboard",
                "Check data pipeline for issues",
                "Consider model retraining"
            ],
            'prediction': [
                "Analyze prediction distribution changes",
                "Check for input data quality issues",
                "Consider model recalibration"
            ],
            'target': [
                "Investigate target distribution changes",
                "Verify data labeling process",
                "Schedule urgent model retraining"
            ]
        }

        return actions.get(drift_type, ["Review model performance"])


def check_for_drift(model_name, reference_data, current_data, target_column=None):
    """
    Convenience function to check all drift types.

    Typical workflow:
    1. Run weekly (via GitHub Actions)
    2. Compare last 7 days of production data to training data
    3. Generate reports and alerts
    4. Create GitHub issue if drift detected

    Args:
        model_name: Name of model
        reference_data: Training/baseline data
        current_data: Recent production data
        target_column: Target column name (optional)

    Returns:
        dict: Combined drift results
    """
    detector = DriftDetector(model_name)

    results = {}

    # Check dataset drift (features)
    results['dataset_drift'] = detector.detect_dataset_drift(
        reference_data.drop(columns=[target_column] if target_column else []),
        current_data.drop(columns=[target_column] if target_column else [])
    )

    # Check target drift (if labels available)
    if target_column:
        results['target_drift'] = detector.detect_target_drift(
            reference_data,
            current_data,
            target_column
        )

    # Overall assessment
    any_drift = any(
        drift_result.get('drift_detected', False)
        for drift_result in results.values()
    )

    results['overall_drift_detected'] = any_drift

    return results


def main():
    """Demo drift detection."""
    # Create synthetic data with drift
    np.random.seed(42)
    n = 500

    # Reference data (baseline)
    reference_data = pd.DataFrame({
        'age': np.random.normal(55, 10, n),
        'cholesterol': np.random.normal(200, 40, n),
        'heart_disease': np.random.binomial(1, 0.4, n)
    })

    # Current data (with drift - cholesterol distribution changed)
    current_data = pd.DataFrame({
        'age': np.random.normal(55, 10, n),
        'cholesterol': np.random.normal(220, 50, n),  # Mean and variance shifted
        'heart_disease': np.random.binomial(1, 0.4, n)
    })

    # Detect drift
    results = check_for_drift(
        model_name="heart_disease",
        reference_data=reference_data,
        current_data=current_data,
        target_column="heart_disease"
    )

    print("Drift Detection Results:")
    print(f"  Overall Drift: {results['overall_drift_detected']}")
    print(f"  Dataset Drift: {results['dataset_drift']['drift_detected']}")
    print(f"  Drifted Features: {results['dataset_drift']['drifted_columns']}")


if __name__ == '__main__':
    main()
