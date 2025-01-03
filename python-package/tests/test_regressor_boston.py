import microgbtpy
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import root_mean_squared_error

from sklearn.datasets import fetch_california_housing


RANDOM_SEED = 123
TEST_SPLIT = 0.1
params = {
    "gamma": 0.1,
    "lambda": 1.0,
    "max_depth": 4.0,
    "shrinkage_rate": 1.0,
    "min_split_gain": 0.1,
    "learning_rate": 0.1,
    "min_tree_size": 3,
    "num_boosting_rounds": 1000.0,
    "metric": 1.0,
}


def _load_california_housing():
    """Load California housing regression dataset"""

    data, target = fetch_california_housing(return_X_y=True)

    print("Input dataset dimensions {}".format(data.shape))
    print("Target dims: {}".format(target.shape))

    ######################
    # Train / test split #
    ######################
    X_train, X_valid, y_train, y_valid = train_test_split(
        data, target, test_size=TEST_SPLIT, random_state=RANDOM_SEED, shuffle=True
    )

    return X_train, X_valid, y_train, y_valid


def test_microgbt_housing_rmse():
    num_iters = 100
    early_stopping_rounds = 10

    X_train, X_valid, y_train, y_valid = _load_california_housing()

    # Train
    gbt = microgbtpy.GBT(params)
    gbt.train(X_train, y_train, X_valid, y_valid, num_iters, early_stopping_rounds)

    # Predict
    y_valid_preds = []
    for x in X_valid:
        y_valid_preds.append(gbt.predict(x, gbt.best_iteration()))

    assert root_mean_squared_error(y_valid, y_valid_preds) < 4
