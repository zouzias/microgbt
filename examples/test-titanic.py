#!/usr/bin/env python3
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



df = pd.read_csv("../data/titanic.csv")

target = df["Survived"].astype(int).values
df.drop(columns=["Survived", "PassengerId"], inplace=True)

df["Embarked_C"] = df["Embarked"].isin(["C"]).astype(int)
df["Embarked_S"] = df["Embarked"].isin(["S"]).astype(int)
df["is_male"] = df["Sex"].isin(["male"]).astype(int)

# df = df.select_dtypes(include=[int, float, bool])

df = df[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare',
         'Cabin', 'Embarked']]
print("Selected column: {}".format(df.columns))
print(df.head())
cats = ['Sex', 'Parch', 'Cabin', 'Embarked']
for cat in cats:
    df[cat] = LabelEncoder().fit_transform(df[cat].astype(str))
data = SimpleImputer().fit_transform(df)

print("Input dataset dimensions {}".format(data.shape))
print("Target dims: {}".format(target.shape))

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
    "max_bin" : 16,
    "shrinkage_rate": 1.0,
    "min_split_gain": 0.1,
    "learning_rate": 0.1,
    "min_tree_size": 3,
    "num_boosting_rounds": 10000.0,
    "metric": 0.0
}

gbt = microgbtpy.GBT(params)

print(gbt)

print("Max depth = {}".format(gbt.max_depth()))
print("Max lambda = {}".format(gbt.get_lambda()))


num_iters = 100
early_stopping_rounds = 10


y_train = y_train.astype(np.double)

gbt.train(X_train, y_train, X_valid, y_valid, num_iters, early_stopping_rounds)

logger.info("Best iteration {}".format(gbt.best_iteration()))
binclass_metrics(gbt, X_train, y_train, "Training")
binclass_metrics(gbt, X_valid, y_valid, "Validation")
binclass_metrics(gbt, X_test, y_test, "Testing", show_report=True)
