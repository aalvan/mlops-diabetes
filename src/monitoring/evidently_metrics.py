import os
import datetime
import random
import logging
import pytz
import pandas as pd
import psycopg

import mlflow
from mlflow import MlflowClient
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric, ColumnQuantileMetric, ColumnDistributionMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
DROP TABLE IF EXISTS metrics;
CREATE TABLE metrics(
    timestamp TIMESTAMP,
    prediction_drift FLOAT,
    num_drifted_columns INTEGER,
    share_missing_values FLOAT,
    quantile_value FLOAT
);
"""

os.environ["AWS_PROFILE"] = 'mlflow-profile'

EXPERIMENT = 'diabetes-uci'
TRACKING_SERVER_HOST = 'ec2-18-221-177-184.us-east-2.compute.amazonaws.com'
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
mlflow.set_experiment(EXPERIMENT)
MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()

model_name = "diabetes-predictor"
model_stage = "Production"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

dataset_path = 'data/processed/diabetes.csv'
diabetes_df = pd.read_csv(dataset_path)

reference_data = diabetes_df.drop('Outcome', axis=1)
reference_data['prediction'] = model.predict(reference_data)

numerical_features = [*reference_data.columns]

column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=numerical_features,
    target=None
)

report = Report(metrics=[
    ColumnDriftMetric(column_name='Glucose'),
    ColumnDistributionMetric(column_name='BMI'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric(),
    ColumnQuantileMetric(column_name='Glucose', quantile=0.5)]
)

def prep_db():
    with psycopg.connect("host=localhost port=5432 user=postgres password=example", autocommit=True) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example") as conn:
            conn.execute(create_table_statement)

def generate_synthetic_data(base_df, variation_factor=0.1):
    synthetic_df = base_df.copy()
    for col in synthetic_df.columns:
        if synthetic_df[col].dtype in [int, float]:
            synthetic_df[col] += synthetic_df[col] * variation_factor * (2 * random.random() - 1)
    return synthetic_df

def calculate_metrics_postgresql(curr, current_data, timestamp):
    # Ensure the 'prediction' column is not in current_data
    if 'prediction' in current_data.columns:
        current_data = current_data.drop(columns=['prediction'])
    
    # Generate predictions
    current_data['prediction'] = model.predict(current_data)

    # Run the Evidently report
    report.run(reference_data=reference_data, current_data=current_data, column_mapping=column_mapping)

    result = report.as_dict()

    # Extract the relevant metrics
    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][2]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][3]['result']['current']['share_of_missing_values']
    quantile_value = result['metrics'][4]['result']['current']['value']
    
	# Add random hours, minutes, and seconds to the timestamp
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    random_seconds = random.randint(0, 59)
    fractional_seconds = random.random()  # Generates a float between 0 and 1
    timestamp = timestamp + datetime.timedelta(
        hours=random_hours, 
        minutes=random_minutes, 
        seconds=random_seconds + fractional_seconds
    )
    # Insert the metrics into the database
    curr.execute(
        "insert into metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values, quantile_value) values (%s, %s, %s, %s, %s)",
        (timestamp, prediction_drift, num_drifted_columns, share_missing_values, quantile_value)
    )


def batch_monitoring_backfill():
    prep_db()
    with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example", autocommit=True) as conn:
        with conn.cursor() as curr:
            start_date = datetime.datetime(2024, 1, 1, 0, 0)
            for month in range(5,11):
                timestamp = start_date + datetime.timedelta(days=30 * month)
                synthetic_data = generate_synthetic_data(reference_data, variation_factor=0.1 * (month + 1))
                calculate_metrics_postgresql(curr, synthetic_data, timestamp)
                logging.info(f"Data for {timestamp.strftime('%B %Y')} sent")

if __name__ == '__main__':
    batch_monitoring_backfill()
