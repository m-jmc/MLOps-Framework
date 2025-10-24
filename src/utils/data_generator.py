"""
Heart Disease Dataset Generator

This module generates synthetic heart disease patient data based on the UCI Heart Disease Dataset.
It creates 1,000 patient records with 12 monthly snapshots for drift simulation.

Features:
- Age: Patient age in years (29-77)
- Sex: Gender (0 = female, 1 = male)
- Chest Pain Type: 4 types (1-4)
- Resting BP: Resting blood pressure (mm Hg)
- Cholesterol: Serum cholesterol (mg/dl)
- Fasting Blood Sugar: >120 mg/dl (0 = false, 1 = true)
- Resting ECG: Results (0-2)
- Max Heart Rate: Maximum heart rate achieved
- Exercise Angina: Exercise induced angina (0 = no, 1 = yes)
- ST Depression: ST depression induced by exercise
- Slope: Slope of peak exercise ST segment (0-2)
- Vessels: Number of major vessels colored by fluoroscopy (0-3)
- Thalassemia: 0 = normal, 1 = fixed defect, 2 = reversable defect

Target:
- Heart Disease: Presence of heart disease (0 = no, 1 = yes)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HeartDiseaseDataGenerator:
    """Generates synthetic heart disease patient data with temporal evolution."""

    def __init__(self, n_patients=1000, n_months=12, random_seed=42):
        """
        Initialize the data generator.

        Args:
            n_patients: Number of patients to generate
            n_months: Number of monthly snapshots
            random_seed: Random seed for reproducibility
        """
        self.n_patients = n_patients
        self.n_months = n_months
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Feature distributions based on UCI Heart Disease Dataset statistics
        self.feature_distributions = {
            'age': {'mean': 54, 'std': 9, 'min': 29, 'max': 77},
            'sex': {'prob_male': 0.68},  # 68% male in UCI dataset
            'chest_pain_type': {'probs': [0.47, 0.17, 0.29, 0.07]},  # 1-4
            'resting_bp': {'mean': 131, 'std': 17, 'min': 94, 'max': 200},
            'cholesterol': {'mean': 246, 'std': 51, 'min': 126, 'max': 564},
            'fasting_blood_sugar': {'prob_high': 0.15},  # 15% have fbs > 120
            'resting_ecg': {'probs': [0.48, 0.48, 0.04]},  # 0-2
            'max_heart_rate': {'mean': 150, 'std': 23, 'min': 71, 'max': 202},
            'exercise_angina': {'prob_yes': 0.33},
            'st_depression': {'mean': 1.04, 'std': 1.16, 'min': 0, 'max': 6.2},
            'slope': {'probs': [0.21, 0.50, 0.29]},  # 0-2
            'vessels': {'probs': [0.54, 0.21, 0.13, 0.12]},  # 0-3
            'thalassemia': {'probs': [0.18, 0.18, 0.64]}  # 0-2
        }

    def generate_base_patient_data(self):
        """Generate base patient demographics and static characteristics."""
        logger.info(f"Generating base data for {self.n_patients} patients...")

        patients = pd.DataFrame({
            'patient_id': [f'PT{i:06d}' for i in range(1, self.n_patients + 1)],
            'age': np.clip(
                np.random.normal(
                    self.feature_distributions['age']['mean'],
                    self.feature_distributions['age']['std'],
                    self.n_patients
                ),
                self.feature_distributions['age']['min'],
                self.feature_distributions['age']['max']
            ).astype(int),
            'sex': np.random.binomial(1, self.feature_distributions['sex']['prob_male'], self.n_patients),
        })

        # Add age groups for bias analysis
        patients['age_group'] = pd.cut(
            patients['age'],
            bins=[0, 45, 65, 100],
            labels=['young', 'middle', 'senior']
        )

        logger.info(f"Generated {len(patients)} patient records")
        return patients

    def generate_clinical_features(self, base_patients, month_offset=0):
        """
        Generate clinical features that may change over time.

        Args:
            base_patients: DataFrame with base patient data
            month_offset: Month number (0-11) for temporal drift simulation
        """
        n = len(base_patients)

        # Add slight drift over time (simulate population changes)
        drift_factor = 1 + (month_offset * 0.01)  # 1% drift per month

        clinical_data = pd.DataFrame({
            'patient_id': base_patients['patient_id'],
            'chest_pain_type': np.random.choice(
                [1, 2, 3, 4],
                size=n,
                p=self.feature_distributions['chest_pain_type']['probs']
            ),
            'resting_bp': np.clip(
                np.random.normal(
                    self.feature_distributions['resting_bp']['mean'] * drift_factor,
                    self.feature_distributions['resting_bp']['std'],
                    n
                ),
                self.feature_distributions['resting_bp']['min'],
                self.feature_distributions['resting_bp']['max']
            ).astype(int),
            'cholesterol': np.clip(
                np.random.normal(
                    self.feature_distributions['cholesterol']['mean'] * drift_factor,
                    self.feature_distributions['cholesterol']['std'],
                    n
                ),
                self.feature_distributions['cholesterol']['min'],
                self.feature_distributions['cholesterol']['max']
            ).astype(int),
            'fasting_blood_sugar': np.random.binomial(
                1,
                self.feature_distributions['fasting_blood_sugar']['prob_high'],
                n
            ),
            'resting_ecg': np.random.choice(
                [0, 1, 2],
                size=n,
                p=self.feature_distributions['resting_ecg']['probs']
            ),
            'max_heart_rate': np.clip(
                np.random.normal(
                    self.feature_distributions['max_heart_rate']['mean'],
                    self.feature_distributions['max_heart_rate']['std'],
                    n
                ),
                self.feature_distributions['max_heart_rate']['min'],
                self.feature_distributions['max_heart_rate']['max']
            ).astype(int),
            'exercise_angina': np.random.binomial(
                1,
                self.feature_distributions['exercise_angina']['prob_yes'],
                n
            ),
            'st_depression': np.clip(
                np.random.normal(
                    self.feature_distributions['st_depression']['mean'],
                    self.feature_distributions['st_depression']['std'],
                    n
                ),
                self.feature_distributions['st_depression']['min'],
                self.feature_distributions['st_depression']['max']
            ).round(1),
            'slope': np.random.choice(
                [0, 1, 2],
                size=n,
                p=self.feature_distributions['slope']['probs']
            ),
            'vessels': np.random.choice(
                [0, 1, 2, 3],
                size=n,
                p=self.feature_distributions['vessels']['probs']
            ),
            'thalassemia': np.random.choice(
                [0, 1, 2],
                size=n,
                p=self.feature_distributions['thalassemia']['probs']
            )
        })

        return clinical_data

    def calculate_heart_disease_probability(self, patient_data):
        """
        Calculate probability of heart disease based on risk factors.

        This is a simplified risk model based on known medical correlations.
        """
        # Start with baseline risk
        prob = np.ones(len(patient_data)) * 0.3

        # Age factor (risk increases with age)
        prob += (patient_data['age'] - 40) * 0.01

        # Gender factor (males have higher risk)
        prob += patient_data['sex'] * 0.15

        # Chest pain type (type 4 = asymptomatic, often more severe)
        prob += (patient_data['chest_pain_type'] == 4) * 0.2

        # High blood pressure
        prob += (patient_data['resting_bp'] > 140) * 0.15

        # High cholesterol
        prob += (patient_data['cholesterol'] > 240) * 0.15

        # Exercise angina
        prob += patient_data['exercise_angina'] * 0.20

        # ST depression (significant if > 2.0)
        prob += (patient_data['st_depression'] > 2.0) * 0.20

        # Number of vessels
        prob += patient_data['vessels'] * 0.10

        # Thalassemia (type 2 = reversible defect, higher risk)
        prob += (patient_data['thalassemia'] == 2) * 0.15

        # Clip to valid probability range
        prob = np.clip(prob, 0, 1)

        return prob

    def generate_labels(self, patient_data):
        """Generate heart disease labels based on risk probabilities."""
        probs = self.calculate_heart_disease_probability(patient_data)
        labels = np.random.binomial(1, probs)
        return labels

    def generate_temporal_data(self):
        """Generate all temporal snapshots (monthly data)."""
        logger.info(f"Generating {self.n_months} months of temporal data...")

        # Generate base patient data (static)
        base_patients = self.generate_base_patient_data()

        # Calculate base date (12 months ago from now)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        all_data = []

        for month in range(self.n_months):
            # Calculate month key
            month_date = start_date + timedelta(days=30 * month)
            month_key = month_date.strftime('%Y%m')

            logger.info(f"  Generating data for month {month_key} ({month + 1}/{self.n_months})...")

            # Generate clinical features for this month
            clinical_data = self.generate_clinical_features(base_patients, month_offset=month)

            # Merge with base patient data
            month_data = base_patients.merge(clinical_data, on='patient_id')

            # Generate labels
            month_data['heart_disease'] = self.generate_labels(month_data)

            # Add temporal metadata
            month_data['month_key'] = month_key
            month_data['event_timestamp'] = month_date

            all_data.append(month_data)

        # Combine all months
        full_data = pd.concat(all_data, ignore_index=True)

        logger.info(f"Generated {len(full_data)} total records ({self.n_patients} patients × {self.n_months} months)")

        return full_data

    def split_for_feast(self, full_data):
        """
        Split data into FEAST-compatible formats:
        - Base data: Patient demographics
        - Historical data: Time-series clinical features (for offline training)
        - Entity data: Monthly snapshots (for online serving)
        - Label data: Outcomes
        """
        logger.info("Splitting data for FEAST feature store...")

        # Base data: Static patient demographics (use most recent timestamp per patient)
        # Include event_timestamp for FEAST compatibility
        base_data = (
            full_data
            .sort_values('event_timestamp')
            .groupby('patient_id')
            .agg({
                'age': 'last',
                'sex': 'last',
                'age_group': 'last',
                'event_timestamp': 'last'
            })
            .reset_index()
        )

        # Historical data: All clinical features with timestamps (for training)
        historical_data = full_data[[
            'patient_id', 'event_timestamp', 'chest_pain_type', 'resting_bp',
            'cholesterol', 'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
            'exercise_angina', 'st_depression', 'slope', 'vessels', 'thalassemia'
        ]]

        # Entity data: Monthly snapshots for online feature retrieval
        # (Most recent value per patient per month)
        entity_data = full_data[[
            'patient_id', 'month_key', 'chest_pain_type', 'resting_bp',
            'cholesterol', 'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate',
            'exercise_angina', 'st_depression', 'slope', 'vessels', 'thalassemia'
        ]]

        # Label data: Outcomes for supervised learning
        label_data = full_data[['patient_id', 'month_key', 'heart_disease', 'event_timestamp']]

        logger.info(f"  Base data: {len(base_data)} patients")
        logger.info(f"  Historical data: {len(historical_data)} records")
        logger.info(f"  Entity data: {len(entity_data)} monthly snapshots")
        logger.info(f"  Label data: {len(label_data)} outcomes")

        return base_data, historical_data, entity_data, label_data

    def save_datasets(self, output_dir='src/feature_store/heart_disease_features/data'):
        """Generate and save all datasets."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (output_path / 'base_data').mkdir(parents=True, exist_ok=True)
        (output_path / 'historical_data_sample').mkdir(parents=True, exist_ok=True)
        (output_path / 'entity_data_sample').mkdir(parents=True, exist_ok=True)
        (output_path / 'label_data').mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving datasets to {output_path}...")

        # Generate full temporal data
        full_data = self.generate_temporal_data()

        # Split for FEAST
        base_data, historical_data, entity_data, label_data = self.split_for_feast(full_data)

        # Save as Parquet files (efficient for FEAST)
        base_data.to_parquet(output_path / 'base_data' / 'patients.parquet', index=False)
        historical_data.to_parquet(output_path / 'historical_data_sample' / 'clinical_history.parquet', index=False)
        entity_data.to_parquet(output_path / 'entity_data_sample' / 'monthly_data.parquet', index=False)
        label_data.to_parquet(output_path / 'label_data' / 'outcomes.parquet', index=False)

        # Also save CSV versions for easy inspection
        base_data.to_csv(output_path / 'base_data' / 'patients.csv', index=False)
        label_data.to_csv(output_path / 'label_data' / 'outcomes.csv', index=False)

        logger.info("✓ All datasets saved successfully!")

        # Print summary statistics
        self._print_summary_statistics(full_data)

        return full_data

    def _print_summary_statistics(self, data):
        """Print summary statistics of generated data."""
        logger.info("\n" + "="*60)
        logger.info("DATASET SUMMARY")
        logger.info("="*60)
        logger.info(f"Total records: {len(data)}")
        logger.info(f"Unique patients: {data['patient_id'].nunique()}")
        logger.info(f"Time range: {data['month_key'].min()} to {data['month_key'].max()}")
        logger.info(f"\nHeart Disease Prevalence: {data['heart_disease'].mean():.1%}")
        logger.info(f"  - No disease: {(data['heart_disease'] == 0).sum()} ({(data['heart_disease'] == 0).mean():.1%})")
        logger.info(f"  - Has disease: {(data['heart_disease'] == 1).sum()} ({(data['heart_disease'] == 1).mean():.1%})")
        logger.info(f"\nGender Distribution:")
        logger.info(f"  - Female: {(data['sex'] == 0).sum()} ({(data['sex'] == 0).mean():.1%})")
        logger.info(f"  - Male: {(data['sex'] == 1).sum()} ({(data['sex'] == 1).mean():.1%})")
        logger.info(f"\nAge Statistics:")
        logger.info(f"  - Mean: {data['age'].mean():.1f} years")
        logger.info(f"  - Std: {data['age'].std():.1f} years")
        logger.info(f"  - Range: {data['age'].min()}-{data['age'].max()} years")
        logger.info("="*60 + "\n")


def main():
    """Main function to generate heart disease dataset."""
    # Create data generator
    generator = HeartDiseaseDataGenerator(
        n_patients=1000,
        n_months=12,
        random_seed=42
    )

    # Generate and save datasets
    data = generator.save_datasets()

    logger.info("Heart disease dataset generation complete!")

    return data


if __name__ == '__main__':
    main()
