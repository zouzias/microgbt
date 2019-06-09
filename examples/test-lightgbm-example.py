#!/usr/bin/env python3
import gbtpy
import pandas as pd
from math import sqrt
import logging.config
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


logging.config.fileConfig("logging.ini", disable_existing_loggers=False)
logger = logging.getLogger(__name__)

def regression_metrics(gbt, X_, y_, label):
    y_preds = []
    for row in range(X_.shape[0]):
        y_preds.append(gbt.predict(X_[row, :], gbt.best_iteration()))

    logger.info("*************[{}]*************".format(label))
    logger.info("******************************".format(label))
    logger.info("* [{}] RMSE={:3f}".format(label, sqrt(mean_squared_error(y_, y_preds))))
    logger.info("* [{}] R^2-Score={:3f}".format(label, r2_score(y_, y_preds)))
    logger.info("******************************".format(label))



df_train = pd.read_csv('../data/lightgbm-regression.train', header=None, sep='\t')
df_test = pd.read_csv('../data/lightgbm-regression.test', header=None, sep='\t')

y = df_train[0].values
y_test = df_test[0].values
X = df_train.drop(0, axis=1).values
X_test= df_test.drop(0, axis=1).values


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.15,
    random_state=42, shuffle=True
)

print("Input train dataset dimensions {}".format(X_train.shape))
print("Target train dims: {}".format(y_train.shape))
print("Input test valid dimensions {}".format(X_valid.shape))
print("Target valid dims: {}".format(y_valid.shape))
print("Input test dataset dimensions {}".format(X_test.shape))
print("Target test dims: {}".format(y_test.shape))


# Copied from https://github.com/microsoft/LightGBM/blob/master/examples/regression/train.conf
params = {
    "gamma": 0.1,
    "lambda": 1.0,
    "max_depth": 4.0,
    "shrinkage_rate": 1.0,
    "min_split_gain": 0.1,
    "learning_rate": 0.05,
    "min_tree_size": 100,
    "metric": 1.0
}

# Define the GBT
gbt = gbtpy.GBT(params)
print(gbt)

# Training related parameters
num_iters = 5
early_stopping_rounds = 5
gbt.train(X_train, y_train, X_valid, y_valid, num_iters, early_stopping_rounds)


logger.info("Best iteration {}".format(gbt.best_iteration()))
regression_metrics(gbt, X_train, y_train, "Training")
regression_metrics(gbt, X_valid, y_valid, "Validation")
regression_metrics(gbt, X_test, y_test, "Testing")
