"""
Hospital Readmission Regression Model - Simplified Working Version

This is a fully functional but simplified implementation of the readmission risk model.
It demonstrates regression model training while keeping complexity minimal for educational purposes.

Use this as:
1. A working reference for the stub version (train.py)
2. A quick demonstration of regression ML in the platform
3. A template for adding new regression models

Simplifications:
- Uses synthetic mock data (no FEAST integration)
- Basic XGBoost regression (no hyperparameter tuning)
- Essential metrics only (RMSE, MAE, R²)
- Minimal bias checking
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleReadmissionModel:
    """Simplified hospital readmission risk model for demonstration."""

    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.model = None

    def generate_mock_data(self, n_samples=1000):
        """
        Generate synthetic readmission data for demonstration.

        Features:
        - age: Patient age (18-95)
        - sex: Gender (0=female, 1=male)
        - length_of_stay: Days in hospital (1-30)
        - num_prior_admissions: Previous hospitalizations (0-10)
        - num_medications: Number of medications (0-20)
        - num_procedures: Procedures during stay (0-10)
        - chronic_conditions: Count of chronic diseases (0-5)
        - emergency_admission: Admitted via ER (0/1)

        Target:
        - readmission_risk: 30-day readmission probability (0-1)
        """
        logger.info(f"Generating {n_samples} synthetic patient records...")

        data = pd.DataFrame({
            'age': np.random.randint(18, 96, n_samples),
            'sex': np.random.binomial(1, 0.5, n_samples),
            'length_of_stay': np.clip(np.random.gamma(2, 2, n_samples), 1, 30).astype(int),
            'num_prior_admissions': np.random.poisson(2, n_samples),
            'num_medications': np.random.poisson(5, n_samples),
            'num_procedures': np.random.poisson(2, n_samples),
            'chronic_conditions': np.random.binomial(5, 0.3, n_samples),
            'emergency_admission': np.random.binomial(1, 0.4, n_samples),
        })

        # Create age groups for bias analysis
        data['age_group'] = pd.cut(data['age'], bins=[0, 45, 65, 100], labels=['young', 'middle', 'senior'])

        # Calculate readmission risk based on realistic factors
        risk = self._calculate_readmission_risk(data)
        data['readmission_risk'] = risk

        logger.info(f"  Mean risk: {risk.mean():.3f}")
        logger.info(f"  Risk range: [{risk.min():.3f}, {risk.max():.3f}]")

        return data

    def _calculate_readmission_risk(self, data):
        """Calculate synthetic readmission risk based on medical factors."""
        # Start with baseline risk
        risk = np.ones(len(data)) * 0.15

        # Age (higher risk for elderly)
        risk += (data['age'] - 65) / 100 * (data['age'] > 65)

        # Length of stay (longer = higher risk)
        risk += data['length_of_stay'] / 100

        # Prior admissions (strong predictor)
        risk += data['num_prior_admissions'] * 0.05

        # Medications (polypharmacy increases risk)
        risk += (data['num_medications'] > 10) * 0.10

        # Chronic conditions
        risk += data['chronic_conditions'] * 0.05

        # Emergency admission
        risk += data['emergency_admission'] * 0.08

        # Add some noise
        risk += np.random.normal(0, 0.05, len(data))

        # Clip to valid probability range
        risk = np.clip(risk, 0, 1)

        return risk

    def prepare_data(self, data):
        """Split data into features and target."""
        # Drop non-feature columns
        X = data.drop(['readmission_risk', 'age_group'], axis=1)
        y = data['readmission_risk']

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed
        )

        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Train XGBoost regression model."""
        logger.info("Training XGBoost regression model...")

        self.model = xgb.XGBRegressor(
            objective='reg:logistic',  # Bounded [0,1] output for probability
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_seed,
            eval_metric='rmse'
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )

        logger.info("✓ Training complete")

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        logger.info("Evaluating model...")

        y_pred = self.model.predict(X_test)

        metrics = {
            'rmse': mean_squared_error(y_test, y_pred, squared=False),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        logger.info("="*50)
        logger.info("MODEL PERFORMANCE")
        logger.info("="*50)
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE:  {metrics['mae']:.4f}")
        logger.info(f"  R²:   {metrics['r2']:.4f}")
        logger.info("="*50)

        return metrics, y_pred

    def check_bias(self, data, y_test, y_pred):
        """Check for bias across demographic groups."""
        logger.info("Checking for bias...")

        X_test_with_groups = data.iloc[y_test.index]

        # Check bias by gender
        logger.info("\nGender Fairness:")
        for sex in [0, 1]:
            mask = X_test_with_groups['sex'] == sex
            if mask.sum() > 0:
                group_rmse = mean_squared_error(
                    y_test[mask],
                    y_pred[mask],
                    squared=False
                )
                gender = "Female" if sex == 0 else "Male"
                logger.info(f"  {gender} RMSE: {group_rmse:.4f} (n={mask.sum()})")

        # Check bias by age group
        logger.info("\nAge Group Fairness:")
        for age_group in ['young', 'middle', 'senior']:
            mask = X_test_with_groups['age_group'] == age_group
            if mask.sum() > 0:
                group_rmse = mean_squared_error(
                    y_test[mask],
                    y_pred[mask],
                    squared=False
                )
                logger.info(f"  {age_group.capitalize()} RMSE: {group_rmse:.4f} (n={mask.sum()})")

    def get_feature_importance(self):
        """Get and display feature importance."""
        importance = pd.DataFrame({
            'feature': self.model.get_booster().feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nFeature Importance:")
        for _, row in importance.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return importance


def main():
    """Main execution flow."""
    logger.info("="*60)
    logger.info("HOSPITAL READMISSION RISK MODEL - SIMPLIFIED VERSION")
    logger.info("="*60)
    logger.info("This demonstrates a working regression model in the MLOps platform")
    logger.info("")

    # Initialize model
    model = SimpleReadmissionModel(random_seed=42)

    # Generate synthetic data
    data = model.generate_mock_data(n_samples=1000)

    # Prepare data
    X_train, X_test, y_train, y_test = model.prepare_data(data)

    # Train
    model.train(X_train, y_train)

    # Evaluate
    metrics, y_pred = model.evaluate(X_test, y_test)

    # Check bias
    model.check_bias(data, y_test, y_pred)

    # Feature importance
    importance = model.get_feature_importance()

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Model Type: XGBoost Regression")
    logger.info(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Test RMSE: {metrics['rmse']:.4f}")
    logger.info(f"Test R²: {metrics['r2']:.4f}")
    logger.info("")
    logger.info("Next Steps:")
    logger.info("  1. Integrate with MLflow for experiment tracking")
    logger.info("  2. Connect to FEAST feature store")
    logger.info("  3. Add hyperparameter optimization")
    logger.info("  4. Implement model promotion workflow")
    logger.info("  5. Generate model card documentation")
    logger.info("="*60)


if __name__ == '__main__':
    main()
