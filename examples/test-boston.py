#!/usr/bin/env python
import microgbtpy
from math import sqrt
import logging.config
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston


logging.config.fileConfig("logging.ini", disable_existing_loggers=False)
logger = logging.getLogger(__name__)

TEST_SPLIT = 0.1
RANDOM_SEED = 123

def regression_metrics(gbt, X, y_true, label):
    y_preds = []
    for row in range(X.shape[0]):
        y_preds.append(gbt.predict(X[row, :], gbt.best_iteration()))

    # print(y_true)
    # print(y_preds)
    logger.info("*************[{}]*************".format(label))
    logger.info("******************************".format(label))
    logger.info("[{}]RMSE={:3f}".format(label, sqrt(mean_squared_error(y_true, y_preds))))
    logger.info("[{}]R^2-Score={:3f}".format(label, r2_score(y_true, y_preds)))
    logger.info("******************************".format(label))


data, target = load_boston(return_X_y=True)

print("Input dataset dimensions {}".format(data.shape))
print("Target dims: {}".format(target.shape))

######################
# Train / test split #
######################
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=TEST_SPLIT,
    random_state=RANDOM_SEED, shuffle=True
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=TEST_SPLIT,
    random_state=RANDOM_SEED, shuffle=True
)



params = {
    "gamma": 0.1,
    "lambda": 1.0,
    "max_depth": 4.0,
    "shrinkage_rate": 1.0,
    "min_split_gain": 0.1,
    "learning_rate": 0.1,
    "min_tree_size": 3,
    "num_boosting_rounds": 1000.0,
    "metric": 1.0
}

gbt = microgbtpy.GBT(params)
print(gbt)

num_iters = 100
early_stopping_rounds = 10


gbt.train(X_train, y_train, X_valid, y_valid, num_iters, early_stopping_rounds)

logger.info("Best iteration {}".format(gbt.best_iteration()))
regression_metrics(gbt, X_train, y_train, "Training")
regression_metrics(gbt, X_valid, y_valid, "Validation")
regression_metrics(gbt, X_test, y_test, "Testing")
