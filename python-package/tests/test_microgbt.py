import microgbtpy
import pytest

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
    "metric": 0.0,
}


@pytest.fixture
def gbt():
    return microgbtpy.GBT(params)


def test_microgbt_max_depth(gbt):
    assert gbt.max_depth() == params["max_depth"]


def test_microgbt_min_split_gain(gbt):
    assert gbt.min_split_gain() == params["min_split_gain"]


def test_microgbt_learning_rate(gbt):
    assert gbt.learning_rate() == params["learning_rate"]


def test_microgbt_gamma(gbt):
    assert gbt.gamma() == params["gamma"]


def test_microgbt_lambda(gbt):
    assert gbt.get_lambda() == params["lambda"]


def test_microgbt_best_iteration(gbt):
    assert gbt.best_iteration() == 0


def test_microgbt_repl(gbt):
    assert len(str(gbt)) > 0
