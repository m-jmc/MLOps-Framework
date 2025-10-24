import pandas as pd
import numpy as np
from statistical_distance_library import calculate_distribution_distance


def evaluate_dataset_drift(config):
    """
    Evaluates drift between model training data and current production data.
    
    Workflow:
    1. Load production model metadata
    2. Retrieve training dataset from model timestamp
    3. Retrieve most recent production dataset
    4. Compare distributions using drift detection library
    5. Generate report and alert if drift detected
    
    Returns:
    - drift_detected: boolean indicating if drift was found
    - Saves HTML report to output directory
    """
    
    # Database connections
    database_connection = connect_to_database(config['database_path'])
    
    # Model registry connections
    model, model_metadata, registry_client = load_production_model(
        config['model_registry_path']
    )
    
    # Extract training timestamp from model metadata
    training_period = extract_training_period(model_metadata.creation_timestamp)
    
    # Retrieve datasets for comparison
    training_dataset = get_data_for_period(database_connection, training_period)
    current_period = get_most_recent_period(database_connection)
    
    if current_period == training_period:
        print("Model trained on most recent data - no comparison needed")
        return False
    
    current_dataset = get_data_for_period(database_connection, current_period)
    
    # Generate drift analysis report
    drift_report = create_drift_report(
        reference_data=training_dataset,
        current_data=current_dataset
    )
    
    # Extract drift detection result
    drift_detected = extract_metric_from_report(
        report=drift_report,
        metric_path=['metrics', 0, 'result', 'dataset_drift']
    )
    
    if drift_detected:
        print('Data drift detected - recommend model retraining')
        trigger_alert('Data drift detected in production')
    else:
        print('No data drift detected')
    
    # Save report for review
    save_report_to_file(drift_report, config['output_directory'])
    
    return drift_detected


def calculate_distribution_divergence(reference_distribution, 
                                      current_distribution, 
                                      threshold=0.1):
    """
    Calculates statistical distance between two probability distributions.
    
    Used to detect drift in target variable or feature distributions.
    Distance of 0 = identical distributions
    Distance of 1 = maximally different distributions
    
    Parameters:
    - reference_distribution: baseline distribution (e.g., training data)
    - current_distribution: comparison distribution (e.g., production data)
    - threshold: drift detection threshold
    
    Returns:
    - Dictionary with drift analysis results
    """
    
    # Convert to probability distributions
    reference_probability = calculate_mean(reference_distribution)
    current_probability = calculate_mean(current_distribution)
    
    # Create binary probability vectors
    reference_vector = create_probability_vector(reference_probability)
    current_vector = create_probability_vector(current_probability)
    
    # Calculate statistical distance
    divergence_score = compute_statistical_distance(
        reference_vector, 
        current_vector
    )
    
    drift_detected = divergence_score > threshold
    
    return {
        'is_drift': drift_detected,
        'divergence_score': divergence_score,
        'threshold': threshold,
        'reference_probability': reference_probability,
        'current_probability': current_probability
    }


def evaluate_target_drift(config):
    """
    Evaluates drift in model predictions compared to actual outcomes.
    
    Workflow:
    1. Load ground truth labels from recent period
    2. Load model predictions from inference logs
    3. Compare distributions
    4. Report drift metrics
    
    This helps detect:
    - Model performance degradation
    - Changes in outcome distribution
    - Prediction bias shifts
    """
    
    # Connect to data sources
    database_connection = connect_to_database(config['database_path'])
    inference_connection = connect_to_database(config['inference_database_path'])
    
    # Get most recent time period
    recent_period = get_most_recent_period(database_connection)
    
    # Load actual outcomes
    actual_outcomes_query = build_query(
        table='ground_truth_outcomes',
        filters={'time_period': recent_period}
    )
    actual_outcomes = execute_query(database_connection, actual_outcomes_query)
    
    # Load model predictions
    predictions_query = build_query(
        table='inference_logs',
        columns=['prediction']
    )
    model_predictions = execute_query(inference_connection, predictions_query)
    
    # Compare distributions
    drift_analysis = calculate_distribution_divergence(
        reference_distribution=actual_outcomes['outcome'].values,
        current_distribution=model_predictions['prediction'].values
    )
    
    # Report results
    print(f"Drift detected: {drift_analysis['is_drift']}")
    print(f"Divergence score: {drift_analysis['divergence_score']:.4f}")
    print(f"Reference probability: {drift_analysis['reference_probability']:.4f}")
    print(f"Current probability: {drift_analysis['current_probability']:.4f}")
    
    return drift_analysis


# Helper functions
def extract_training_period(timestamp):
    """Extracts time period identifier from model training timestamp"""
    date = convert_timestamp_to_date(timestamp)
    return format_as_period(date)


def extract_metric_from_report(report, metric_path):
    """Safely extracts nested metric from report structure"""
    try:
        value = report
        for key in metric_path:
            value = value[key]
        return value
    except (KeyError, IndexError, TypeError):
        return None


def get_most_recent_period(database_connection):
    """Retrieves the most recent time period from database"""
    query = """
        SELECT MAX(time_period) AS period 
        FROM historical_data
    """
    result = execute_query(database_connection, query)
    return result['period'].values[0]


def get_data_for_period(database_connection, period):
    """Retrieves all data for a specific time period"""
    query = f"""
        SELECT * 
        FROM historical_data
        WHERE time_period = '{period}'
    """
    return execute_query(database_connection, query)