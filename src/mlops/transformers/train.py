import os
from typing import Tuple, Dict

import mlflow
from mlflow.tracking import MlflowClient

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def transform(training_set: 
    Tuple[Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
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
    os.environ["AWS_PROFILE"] = kwargs.get('profile')

    
    experiment = kwargs.get('experiment')
    TRACKING_SERVER_HOST = kwargs.get('TRACKING_SERVER_HOST')
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_experiment(experiment)
    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()

    hyperparameters, X_train, X_test, y_train, y_test = training_set

    data_path = '../data/raw/diabetes.csv'

    with mlflow.start_run() as run:
        mlflow.log_params(hyperparameters)
        mlflow.set_tag("developer", "alexis")
        mlflow.set_tag("model", "random_forest")
        mlflow.log_param('train_data_path', f'{data_path}')

        artifact_path = 'models/random_forest.joblib'
        print(f'Best params: {hyperparameters}')

        rf_model = RandomForestClassifier(**hyperparameters)
        rf_model.fit(X_train, y_train)
        
        y_pred = rf_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        mlmetrics = {'accuracy': acc}
        print(f'accuracy: {acc}')
        mlflow.log_metrics(mlmetrics)
        
        mlflow.sklearn.log_model(
            rf_model,
            artifact_path=artifact_path,
        )
        artifact_uri = mlflow.get_artifact_uri(artifact_path=artifact_path)
        run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/model"
    model_name = "diabetes-predictor"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    model_version = result.version

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Production",
        archive_existing_versions=False
    )

    return f'Model {model_name} version {model_version} transitioned to stage "Production"'

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'