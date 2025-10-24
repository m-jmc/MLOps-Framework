# Feature Store - ML Model Platform

This directory contains feature store definitions for all models in the ML platform.

## Overview

A feature store provides:
- **Consistent feature definitions** across training and inference
- **Online serving** for real-time predictions  
- **Offline serving** for batch training
- **Point-in-time correctness** to prevent data leakage
- **Feature versioning** and lineage tracking

## Directory Structure
```
feature_store/
├── model_a_features/           # Features for model A
│   ├── data/                   # Feature data storage
│   │   ├── base_data/          # Static entity attributes
│   │   ├── temporal_data/      # Time-series features
│   │   ├── entity_snapshots/   # Periodic snapshots
│   │   └── labels/             # Target variables
│   ├── feature_definitions.py  # Feature view definitions
│   └── feature_store.yaml      # Store configuration
│
├── model_b_features/           # Features for model B
│   └── ...
│
└── README.md                   # This file
```

## Quick Start

### 1. Initialize Feature Store

Apply feature definitions to the registry:
```bash
# Navigate to specific feature store
cd feature_store/model_features

# Register features
feast apply

# Verify registration
feast feature-views list
feast entities list
```

### 2. Materialize Features

Load features into online store for serving:
```bash
# Materialize all features up to current time
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

### 3. Retrieve Features
```python
from feast import FeatureStore

# Initialize
store = FeatureStore(repo_path="path/to/feature_store")

# Get online features for inference
features = store.get_online_features(
    features=["feature_view_name:feature_1", "feature_view_name:feature_2"],
    entity_rows=[{"entity_id": "entity_123"}]
).to_dict()
```

## Core Concepts

### Entities
Unique identifiers for feature lookup (e.g., `user_id`, `transaction_id`)

### Feature Views
Logical groupings of related features with:
- **Data source**: Where features are stored
- **Features**: List of columns to expose
- **TTL**: How long features remain valid

### Feature Services
Named collections of feature views for specific models

## Usage Patterns

### Offline Features (Training)
```python
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="path/to/feature_store")

# Entity dataframe with timestamps
entity_df = pd.DataFrame({
    "entity_id": ["id_001", "id_002", "id_003"],
    "event_timestamp": [timestamp_1, timestamp_2, timestamp_3]
})

# Retrieve historical features with point-in-time correctness
training_data = store.get_historical_features(
    entity_df=entity_df,
    features=["feature_service_name"]
).to_df()
```

### Online Features (Inference)
```python
from feast import FeatureStore

store = FeatureStore(repo_path="path/to/feature_store")

# Real-time feature retrieval
features = store.get_online_features(
    features=["feature_service_name"],
    entity_rows=[{"entity_id": "id_001"}]
).to_dict()

# Use for prediction
prediction = model.predict(features)
```

## Best Practices

### Feature Engineering
- Keep transformations simple in feature definitions
- Group related features into logical views
- Set appropriate TTLs based on feature staleness tolerance
- Document all features with clear descriptions

### Data Quality
- Validate schema before materializing
- Handle missing values appropriately
- Monitor feature freshness
- Track data lineage and versions

### Performance
- Materialize incrementally (only changed features)
- Partition large datasets by time period
- Monitor retrieval latency
- Cache frequently accessed features

### Governance
- Version features alongside model versions
- Implement access controls as needed
- Audit feature usage across models
- Define feature deprecation policies

## Adding New Features

1. **Create feature directory structure**:
```bash
   mkdir -p feature_store/new_model_features/data
```

2. **Define features** in `feature_definitions.py`:
   - Entities (unique identifiers)
   - Data sources (where data lives)
   - Feature views (logical groupings)
   - Feature services (model-specific collections)

3. **Configure store** in `feature_store.yaml`:
```yaml
   project: new_model_features
   provider: local
   registry: data/registry.db
   online_store:
     type: sqlite
     path: data/online_store.db
```

4. **Register and materialize**:
```bash
   feast apply
   feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

## Production Deployment

For production environments:

1. **Use production-grade stores**:
   - Registry: PostgreSQL, MySQL
   - Online store: Redis, DynamoDB, Datastore
   - Offline store: BigQuery, Snowflake, Redshift

2. **Example production config**:
```yaml
   project: model_features
   provider: cloud_provider
   registry: postgresql://user:pass@host:5432/registry
   online_store:
     type: redis
     connection_string: redis://host:6379
   offline_store:
     type: data_warehouse
```

3. **Automate materialization** with schedulers (Airflow, Kubernetes CronJobs)

4. **Monitor** feature freshness, latency, and data quality

## Resources

- [Feature Store Documentation](https://docs.feast.dev/)
- [Feature Store Concepts](https://docs.feast.dev/getting-started/concepts)
- [Production Deployment Guide](https://docs.feast.dev/how-to-guides/running-feast-in-production)