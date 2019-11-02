#!/usr/bin/env python
import microgbtpy
import logging.config
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.config.fileConfig("logging.ini", disable_existing_loggers=False)
logger = logging.getLogger(__name__)


def binclass_metrics(gbt, X, y_true, label, show_report = False):
    y_preds = []
    for x in X:
        y_preds.append(gbt.predict(x, gbt.best_iteration()))

    # logger.info("Residuals:")
    # logger.info(y_preds - y_true)

    logger.info("*************[{}]*************".format(label))
    logger.info("* [{}]LogLoss={:3f}".format(label, log_loss(y_true, y_preds)))
    logger.info("* [{}]ROC-AUC={:3f}".format(label, roc_auc_score(y_true, y_preds)))
    if show_report:
        logger.info("\n{}".format(classification_report(y_true, [y > 0.5 for y in y_preds])))


def load_flight_delay():
    logger.info("Loading flights data...")
    dataset = "flight"
    URL = "../data/flight_delay.csv.gz"
    logger.debug("Using dataset {} from {}".format(dataset, URL))
    features = pd.read_csv(URL)

    logger.debug("Successfully downloaded data {}...".format(dataset))
    target_col = u"dep_delayed_15min"
    X, y = features, features[target_col]
    cols_to_rm = [target_col]
    X.drop(columns=cols_to_rm, axis=1, inplace=True)

    X["Origin_Dest"] = X.Origin.fillna("N/A") + "_" + X.Dest.fillna("N/A")
    X["Origin_Dest_UniqueCarrier"] = X.Origin.fillna("N/A") + "_" + X.Dest.fillna("N/A")

    cats = ['Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'UniqueCarrier',
            'Origin', 'Dest', 'Origin_Dest', 'Origin_Dest_UniqueCarrier']
    for cat in cats:
        X[cat] = LabelEncoder().fit_transform(X[cat])

    cats = ['Origin', 'Dest', 'UniqueCarrier', 'DayofMonth', 'Origin_Dest_UniqueCarrier']
    cat_indices = [list(X.columns).index(c) for c in cats]

    logger.info(X.columns)
    logger.info("Categorical indices: {}".format(",".join([str(c) for c in cat_indices])))
    return SimpleImputer().fit_transform(X), y.values, cat_indices


data, target, cat_indices = load_flight_delay()

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.15,
    random_state=42, shuffle=True
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.15,
    random_state=42, shuffle=True
)

params = {
    "gamma": 0.1,
    "lambda": 1.0,
    "max_depth": 4.0,
    "shrinkage_rate": 1.0,
    "min_split_gain": 0.1,
    "min_tree_size": 3,
    "max_bin" : 255,
    "learning_rate": 0.1,
    "num_boosting_rounds": 100.0,
    "metric": 0
}

# Define the GBT
gbt = microgbtpy.GBT(params)
print(gbt)


num_iters = 100
early_stopping_rounds = 10
y_train = y_train.astype(np.double)
gbt.train(X_train, y_train, X_valid, y_valid, num_iters, early_stopping_rounds)

logger.info("Best iteration {}".format(gbt.best_iteration()))
binclass_metrics(gbt, X_train, y_train, "Training")
binclass_metrics(gbt, X_valid, y_valid, "Validation")
binclass_metrics(gbt, X_test, y_test, "Testing", show_report=True)

