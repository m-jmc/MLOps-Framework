"""
Data Schema Validation for Pre-commit Hook

This script validates that data files conform to expected schemas.
It's called automatically by pre-commit hooks when data files are modified.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


EXPECTED_SCHEMAS = {
    'heart_disease': {
        'required_columns': [
            'patient_id', 'age', 'sex', 'chest_pain_type', 'resting_bp',
            'cholesterol', 'fasting_blood_sugar', 'resting_ecg',
            'max_heart_rate', 'exercise_angina', 'st_depression',
            'slope', 'vessels', 'thalassemia'
        ],
        'dtypes': {
            'patient_id': 'object',
            'age': 'int64',
            'sex': 'int64',
            'chest_pain_type': 'int64'
        }
    }
}


def validate_dataframe_schema(df: pd.DataFrame, expected_columns: list, name: str = "dataset") -> bool:
    """
    Validate DataFrame schema.

    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names
        name: Dataset name for error messages

    Returns:
        bool: True if valid, False otherwise
    """
    valid = True

    # Check for required columns
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"{name}: Missing columns: {missing_columns}")
        valid = False

    # Check for unexpected columns
    extra_columns = set(df.columns) - set(expected_columns)
    if extra_columns:
        logger.warning(f"{name}: Extra columns found: {extra_columns}")

    # Check for null values in critical columns
    null_counts = df[expected_columns].isnull().sum()
    critical_nulls = null_counts[null_counts > 0]
    if not critical_nulls.empty:
        logger.warning(f"{name}: Columns with null values:\n{critical_nulls}")

    return valid


def validate_heart_disease_data():
    """Validate heart disease feature store data."""
    base_path = Path("src/feature_store/heart_disease_features/data")

    if not base_path.exists():
        logger.info("Heart disease data directory not found, skipping validation")
        return True

    valid = True
    schema = EXPECTED_SCHEMAS['heart_disease']

    # Check base data
    base_file = base_path / "base_data" / "patients.parquet"
    if base_file.exists():
        try:
            df = pd.read_parquet(base_file)
            if not validate_dataframe_schema(df, ['patient_id', 'age', 'sex', 'age_group'], "base_data"):
                valid = False
        except Exception as e:
            logger.error(f"Error reading base_data: {e}")
            valid = False

    return valid


def main():
    """Main validation entry point."""
    logger.info("Running data schema validation...")

    all_valid = True

    # Validate heart disease data
    if not validate_heart_disease_data():
        all_valid = False

    if all_valid:
        logger.info("✓ All data schemas are valid")
        return 0
    else:
        logger.error("✗ Data schema validation failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
