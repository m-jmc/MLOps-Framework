from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    RequestSource,
    ValueType
)

from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64, String, UnixTimestamp

from datetime import datetime, timedelta
import pandas as pd
import os
from pathlib import Path



directory = Path.cwd() / "historical_data_sample"
files = os.listdir(directory)

most_recent_date = None
most_recent_file = None

for file in files:
    try:
        date_part = file.split('_')[-1].split('.')[0]  
        file_date = datetime.strptime(date_part, "%Y-%m-%d")
        if most_recent_date is None or file_date > most_recent_date:
            most_recent_date = file_date
            most_recent_file = file
    except (ValueError, IndexError):
        continue

# Output the most recent file found
if most_recent_file:
    print(f"The most recent file is: {most_recent_file}")
else:
    print("No valid date-containing files found.")

local_data = os.path.join(Path.cwd() / "historical_data_sample", f"{most_recent_file}")


member = Entity(name="member_id", 
                  join_keys=["member_id"], 
                  value_type=ValueType.INT64, 
                  description="Index for diabetes dataset") 

""" 
# There is no SQLite source in Feast 0.12.0 but otherwise something like this?
db_data = os.path.join(Path.cwd() / "base_data" / "diabetes_data.db")
diabetes_global_source = SqliteSource(
    name="diabetes_global_source_sqlite_source",
    path=db_data,  
    query= f"SELECT * FROM 'diabetes_historical_{mr_db}'",  # SQL query to fetch the features
    timestamp_field="query_date",
    created_timestamp_column="created_date"
)
"""

# Without direct sqlite support, we're using the parquet files generated, so just pretend.
diabetes_global_source = FileSource(
    name="diabetes_global_source_file_source",
    path=local_data,
    timestamp_field="query_date",
    created_timestamp_column="created_date",
)

global_diabetes_view = FeatureView(
    # The unique name of this feature view. Two feature views in a single
    # project cannot have the same name
    name="global_diabetes_view",
    entities=[member],
    ttl=timedelta(days=30),
    # The list of features defined below act as a schema to both define features
    # for both materialization of features into a store, and are used as references
    # during retrieval for building a training dataset or serving features
    schema = [
            Field(name="smoker", dtype=Int64),
            Field(name="physical_activity", dtype=Float32),
            Field(name="alcohol_consumption", dtype=Float32),
            Field(name="height_meters", dtype=Float32),
            Field(name="weight_kg", dtype=Float32),
            Field(name="salary", dtype=Int64),
            Field(name="asthma", dtype=Int64),
            Field(name="cancer", dtype=Int64),
            Field(name="heart_disease", dtype=Int64),
            Field(name="stroke", dtype=Int64),
            Field(name="high_blood_pressure", dtype=Int64),
            Field(name="high_cholesterol", dtype=Int64),
            Field(name="obesity", dtype=Int64),
            Field(name="retinopathy", dtype=Int64),
            Field(name="neuropathy", dtype=Int64),
            Field(name="pregnancies", dtype=Int64),
            Field(name="skinthickness", dtype=Int64),
            Field(name="bmi", dtype=Float32),
            Field(name="diabetespedigreefunction", dtype=Float32),
            Field(name="age", dtype=Int64),
            Field(name="a1c", dtype=Float32),
            Field(name="vitamin_d", dtype=Float32),
            Field(name="member_id", dtype=Int64),
            Field(name="created_date", dtype=UnixTimestamp),
            Field(name="query_date", dtype=UnixTimestamp),
        ],
    online=True,
    source=diabetes_global_source,
    # Tags are user defined key/value pairs that are attached to each
    # feature view
    tags={"team": "data_science"},
)

# Defines a way to push data (to be available offline, online or both) into Feast.
# This would work for an hourly batch that we want to have updated within the feature store at inference
"""
diabetes_push_source = PushSource(
    name="diabetes_push_source",
    batch_source=diabetes_source,
)
"""

# Define a request data source which encodes features / information only
# available at request time (e.g. part of the user initiated HTTP request)
diabetes_input_request = RequestSource(
    name="vals_to_calculate_diabetes_risk",
    schema = [
        Field(name="systolic_bp", dtype=Int64),
        Field(name="diastolic_bp", dtype=Int64),
        Field(name="cholesterol", dtype=Int64),
        Field(name="blood_oxygen", dtype=Float32),
        Field(name="heart_rate", dtype=Int64),
        Field(name="glucose", dtype=Int64),
        Field(name="insulin", dtype=Int64),
    ],
)

@on_demand_feature_view(
    sources=[global_diabetes_view, diabetes_input_request],
    schema=[
        Field(name="diabetes_risk_score", dtype=Float64),
    ],
)
def diabetes_risk_score(inputs: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a diabetes risk score from historical and on-demand/real-time features.

    Parameters:
        inputs (pd.DataFrame): DataFrame containing all necessary fields from diabetes_fv and diabetes_input_request.

    Returns:
        pd.DataFrame: DataFrame with an additional column 'diabetes_risk_score'.
    """
    df = pd.DataFrame()

    required_columns = [
        'a1c', 'glucose', 'bmi', 'physical_activity', 'smoker',
        'heart_disease', 'high_blood_pressure', 'age'
    ]
    
    missing_columns = [col for col in required_columns if col not in inputs.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if inputs[required_columns].isnull().any().any():
        raise ValueError("Input contains NA or NaN values in required columns")

    # Calculate weights based on hypothetical clinical relevance
    weights = {
        'a1c': 0.25,
        'glucose': 0.20,
        'bmi': 0.15,
        'physical_activity': 0.10,
        'smoker': 0.05,
        'heart_disease': 0.10,
        'high_blood_pressure': 0.05,
        'age': 0.10
    }

    df['diabetes_risk_score'] = (
          inputs['a1c'] * weights['a1c']
        + inputs['glucose'] * weights['glucose']
        + inputs['bmi'] * weights['bmi']
        + inputs['physical_activity'] * weights['physical_activity']
        + inputs['smoker'] * weights['smoker']
        + inputs['heart_disease'] * weights['heart_disease']
        + inputs['high_blood_pressure'] * weights['high_blood_pressure']
        + inputs['age'] * weights['age']
    )

    return df


diabetes_v1 = FeatureService(
    name="diabetes_features_v1",
    features=[global_diabetes_view, diabetes_risk_score],
)
