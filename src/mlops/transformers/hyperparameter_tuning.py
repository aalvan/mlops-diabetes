from typing import Tuple, Dict
import pandas as pd

from src.mlops.utils.models.sklearn import tune_hyperparameters

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(training_data: Tuple[pd.DataFrame, pd.DataFrame, 
pd.DataFrame, pd.DataFrame], *args, **kwargs)-> Dict:
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

    profile = kwargs.get('profile')
    TRACKING_SERVER_HOST = kwargs.get('TRACKING_SERVER_HOST')
    experiment = kwargs.get('experiment')
    max_evals = 5
    X_train, X_test, y_train, y_test = training_data
   
    hyperparameters = tune_hyperparameters(X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test, max_evals=max_evals,profile=profile,
    TRACKING_SERVER_HOST=TRACKING_SERVER_HOST, experiment=experiment)

    return hyperparameters, X_train, X_test, y_train, y_test


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
 