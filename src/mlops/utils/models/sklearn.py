import os

from typing import Dict

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import mlflow

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope

def tune_hyperparameters(X_train: pd.DataFrame, y_train:pd.Series, X_test: pd.DataFrame, y_test:pd.Series, 
    max_evals: int, profile: str, TRACKING_SERVER_HOST: str, experiment: str)->Dict:
    os.environ["AWS_PROFILE"] = profile
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_experiment(experiment)
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("developer", "alexis")
            mlflow.set_tag("model", "random_forest")
                
            rf_model = RandomForestClassifier(**params)
            rf_model.fit(X_train, y_train)
            
            y_pred = rf_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            mlmetrics = {'accuracy': acc}
            print(f'accuracy: {acc}')
            mlflow.log_metrics(mlmetrics)

        return {'loss': -acc, 'status': STATUS_OK, 'params': params}

    search_space = {
    'max_depth': scope.int(hp.choice('max_depth', [10, 20, 30])),
    'max_features': hp.choice('max_features', ['sqrt','log2']),
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 50)),
    }
    trials = Trials()
    best = fmin(fn=objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    best_params = trials.best_trial['result']['params']

    return best_params

