import os

from joblib import load
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
import mlflow
from mlflow import MlflowClient



os.environ["AWS_PROFILE"] = 'mlflow-profile'

EXPERIMENT = 'diabetes-uci'
TRACKING_SERVER_HOST = 'ec2-18-222-215-179.us-east-2.compute.amazonaws.com'
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
mlflow.set_experiment(EXPERIMENT)
MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()

model_name = "diabetes-predictor"
model_stage = "Production"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

latest_versions = client.get_latest_versions(name=model_name, stages=[model_stage])
latest_version = latest_versions[0]
RUN_ID = latest_version.run_id

def predict(features):

    input_data = pd.DataFrame([features])

    preds = model.predict(input_data)

    return float(preds[0])

app = Flask('diabetes-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    features = request.get_json()
    print(f"Received features: {features}")
    pred = predict(features)
    print(f"Prediction: {pred}")

    result = {
        'diagnostic': pred,
        'model_version': RUN_ID
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)