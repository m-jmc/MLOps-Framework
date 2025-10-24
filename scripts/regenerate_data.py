"""Force regenerate data with event_timestamp in base_data"""
import sys
import importlib
from pathlib import Path

# Force reload the module
if 'src.utils.data_generator' in sys.modules:
    del sys.modules['src.utils.data_generator']

# Import fresh
from src.utils.data_generator import HeartDiseaseDataGenerator

print("Generating data with event_timestamp in base_data...")
generator = HeartDiseaseDataGenerator(n_patients=1000, random_seed=42)
full_data = generator.save_datasets()

print("\n" + "="*60)
print("Verifying base_data structure...")
print("="*60)

import pandas as pd
base_df = pd.read_parquet("src/feature_store/heart_disease_features/data/base_data/patients.parquet")
print(f"Base data shape: {base_df.shape}")
print(f"Columns: {base_df.columns.tolist()}")
print(f"\nFirst 2 rows:\n{base_df.head(2)}")

if 'event_timestamp' in base_df.columns:
    print("\n✓ SUCCESS: event_timestamp is in base_data!")
else:
    print("\n✗ ERROR: event_timestamp is MISSING from base_data!")
    print("Check src/utils/data_generator.py line ~290-302")