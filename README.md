[![Build Status](https://travis-ci.org/zouzias/microgbt.svg?branch=master)](https://travis-ci.org/zouzias/microgbt/builds)
[![Coverage Status](https://coveralls.io/repos/github/zouzias/microgbt/badge.svg?branch=master)](https://coveralls.io/github/zouzias/microgbt?branch=master)
[![License](https://img.shields.io/badge/license-Apache-blue.svg)](LICENSE)


# microGBT

microGBT is a minimalistic ([650 LOC](NOTES.md)) Gradient Boosting Trees implementation in C++11 following [xgboost's paper](https://arxiv.org/abs/1603.02754), i.e., the tree building process is based on the gradient and Hessian vectors (Newton-Raphson method).

A minimalist Python API is available using [pybind11](https://github.com/pybind/pybind11). To use it,

```python
import microgbtpy

params = {
    "gamma": 0.1,
    "lambda": 1.0,
    "max_depth": 4.0,
    "shrinkage_rate": 1.0,
    "min_split_gain": 0.1,
    "learning_rate": 0.1,
    "min_tree_size": 3,
    "num_boosting_rounds": 100.0,
    "metric": 0.0
}

gbt = microgbtpy.GBT(params)

# Training
gbt.train(X_train, y_train, X_valid, y_valid, num_iters, early_stopping_rounds)

# Predict
y_pred = gbt.predict(x, gbt.best_iteration())
```
## Goals

The main goal of the project is to be educational and provide a minimalistic codebase that allows experimentation with Gradient Boosting Trees.

## Features

Currently, the following loss functions are supported:
* Logistic loss for binary classification, `logloss.h`
* Root Mean Squared Error (RMSE) for regression, `rmse.h`

Set the parameter `metric` to 0.0 and 1.0 for logistic regression and RMSE, respectively.


## Development

```bash
git clone https://github.com/zouzias/microgbt.git
cd microgbt
./runBuild

```

### Binary Classification (Titanic)
A binary classification example using the [Titanic dataset](https://www.kaggle.com/naresh31/titanic-machine-learning-from-disaster). Run

```bash
cd examples/
./test-titanic.py
```
the output should include

````
              precision    recall  f1-score   support

           0       0.75      0.96      0.84        78
           1       0.91      0.55      0.69        56

   micro avg       0.79      0.79      0.79       134
   macro avg       0.83      0.76      0.77       134
weighted avg       0.82      0.79      0.78       134
`
````

### Regression Example (Lightgbm)

To run the LightGBM regression [example](https://github.com/microsoft/LightGBM/tree/master/examples/regression), type

````bash
cd examples/
./test-lightgbm-example.py
````

the output should end with

```
2019-05-19 22:54:04,825 - __main__ - INFO - *************[Testing]*************
2019-05-19 22:54:04,825 - __main__ - INFO - ******************************
2019-05-19 22:54:04,825 - __main__ - INFO - * [Testing]RMSE=0.447120
2019-05-19 22:54:04,826 - __main__ - INFO - * [Testing]R^2-Score=0.194094
2019-05-19 22:54:04,826 - __main__ - INFO - ******************************

```
