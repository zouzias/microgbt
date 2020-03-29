import microgbtpy
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score


RANDOM_SEED = 123
TEST_SIZE = 0.3
params = {
    "gamma": 0.1,
    "lambda": 1.0,
    "max_depth": 4.0,
    "shrinkage_rate": 1.0,
    "min_split_gain": 0.1,
    "learning_rate": 0.1,
    "min_tree_size": 3,
    "num_boosting_rounds": 10000.0,
    "metric": 0.0
}

def load_titanic():
    """ Load Titanic dataset """
    df = pd.read_csv("../data/titanic.csv")

    target = df["Survived"].astype(int).values.astype(np.double)

    df.drop(columns=["Survived", "PassengerId"], inplace=True)

    df["Embarked_C"] = df["Embarked"].isin(["C"]).astype(int)
    df["Embarked_S"] = df["Embarked"].isin(["S"]).astype(int)
    df["is_male"] = df["Sex"].isin(["male"]).astype(int)

    df = df[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]

    cats = ['Sex', 'Parch', 'Cabin', 'Embarked']
    for cat in cats:
        df[cat] = LabelEncoder().fit_transform(df[cat].astype(str))

    data = SimpleImputer().fit_transform(df)

    X_train, X_valid, y_train, y_valid = train_test_split(data, target,
                                                          test_size=TEST_SIZE,
                                                          random_state=RANDOM_SEED,
                                                          shuffle=True)
    y_train = y_train.astype(np.double)
    y_valid = y_valid.astype(np.double)
    return X_train, X_valid, y_train, y_valid




def test_microgbt_input_params():
    gbt = microgbtpy.GBT(params)

    assert gbt.max_depth() == params["max_depth"]
    assert gbt.min_split_gain() == params["min_split_gain"]
    assert gbt.learning_rate() == params["min_split_gain"]
    assert gbt.gamma() == params["gamma"]
    assert gbt.get_lambda() == params["lambda"]
    assert gbt.learning_rate() == params["learning_rate"]
    assert gbt.best_iteration() == 0


def test_microgbt_train_predict():
    num_iters = 100
    early_stopping_rounds = 10

    X_train, X_valid, y_train, y_valid = load_titanic()

    # Train
    gbt = microgbtpy.GBT(params)
    gbt.train(X_train, y_train,
              X_valid, y_valid,
              num_iters,
              early_stopping_rounds)

    # Predict
    for x in X_valid:
        pred = gbt.predict(x, gbt.best_iteration())
        assert 0 <= pred <= 1


def test_microgbt_titanic_roc():
    num_iters = 100
    early_stopping_rounds = 10

    X_train, X_valid, y_train, y_valid = load_titanic()

    # Train
    gbt = microgbtpy.GBT(params)
    gbt.train(X_train, y_train,
              X_valid, y_valid,
              num_iters,
              early_stopping_rounds)

    # Predict
    y_valid_preds = []
    for x in X_valid:
        y_valid_preds.append(gbt.predict(x, gbt.best_iteration()))

    roc = roc_auc_score(y_valid, y_valid_preds)

    assert roc > 0.7, "Area under the curve must be more than 0.7"
