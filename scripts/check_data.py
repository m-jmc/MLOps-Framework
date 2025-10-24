"""Quick script to check generated data structure"""
import pandas as pd
from pathlib import Path

base_path = Path("src/feature_store/heart_disease_features/data")

print("=== Base Data ===")
base_df = pd.read_parquet(base_path / "base_data/patients.parquet")
print(f"Shape: {base_df.shape}")
print(f"Columns: {base_df.columns.tolist()}")
print(f"Sample:\n{base_df.head(2)}\n")

print("=== Historical Data ===")
hist_df = pd.read_parquet(base_path / "historical_data_sample/clinical_history.parquet")
print(f"Shape: {hist_df.shape}")
print(f"Columns: {hist_df.columns.tolist()}")
print(f"Sample:\n{hist_df.head(2)}\n")

print("=== Entity Data ===")
entity_df = pd.read_parquet(base_path / "entity_data_sample/monthly_data.parquet")
print(f"Shape: {entity_df.shape}")
print(f"Columns: {entity_df.columns.tolist()}")
print(f"Sample:\n{entity_df.head(2)}\n")

print("=== Label Data ===")
label_df = pd.read_parquet(base_path / "label_data/outcomes.parquet")
print(f"Shape: {label_df.shape}")
print(f"Columns: {label_df.columns.tolist()}")
print(f"Sample:\n{label_df.head(2)}\n")