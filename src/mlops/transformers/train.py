from typing import Tuple

import mlflow
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

os.environ["AWS_PROFILE"] = "mlflow-profile"
TRACKING_SERVER_HOST = "ec2-18-222-74-4.us-east-2.compute.amazonaws.com"
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
mlflow.set_experiment("diabetes-uci")

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def transform(training_set: 
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    , *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    X_train_scaled, X_test_scaled, y_train, y_test = training_set

    data_path = '../data/raw/diabetes.csv'

    with mlflow.start_run():
        mlflow.set_tag("developer", "alexis")
        mlflow.set_tag("model", "random_forest")
        mlflow.log_param('train_data_path', f'{data_path}')

        artifact_path = '../models/random_forest.joblib'

        max_depth = 10
        max_features = 'log2'
        n_estimators =  50

        rf_model = RandomForestClassifier(
            max_depth=max_depth, max_features=max_features, n_estimators=n_estimators
        )
        rf_model.fit(X_train_scaled, y_train)
        
        y_pred = rf_model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)

        mlmetrics = {'accuracy': acc}
        print(f'accuracy: {acc}')
        mlflow.log_metrics(mlmetrics)
        
        
        mlparams = {
        'max_depth': max_depth,
        'max_features': max_features,
        'n_estimators': n_estimators
        }
        mlflow.log_params(mlparams)

        mlflow.sklearn.log_model(
            rf_model,
            artifact_path=artifact_path,
        )
        artifact_uri = mlflow.get_artifact_uri(artifact_path=artifact_path)
    print(f'artifact_uri: {artifact_uri}')
    return artifact_uri


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'