"""
tests/test_stream_regression.py
Regression tests for numerical weak spots found in the Assignment 1 feedback.

The A1 marker found that a distance computed from squared norms lost the
difference between nearby points far from the origin (1e9 against 1e9 + 1 came
out as 0), and that a selection algorithm became slow on all-equal data. The
streaming package has the same two risks in other places, so each one gets a
test here:

- values that differ by a tiny amount at a huge scale must still be separated by
  the tree, and the scaler and running statistics must keep their precision
  when every value sits near 1e9;
- all-equal and duplicate-heavy data must not make the tree slow or wrong.

Run with: python -m pytest tests/test_stream_regression.py -v   (from the repo root)
"""

import time
import warnings

import numpy as np
import pytest

from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stats import RunningStats
from numcompute_stream.tree import DecisionTreeClassifier


@pytest.fixture(autouse=True)
def _no_stray_warnings():
    """Fail any test that leaks a RuntimeWarning (e.g. 'invalid value in divide')."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


# ----------------------------------------------------------------------
# Tiny gaps at a huge scale
# ----------------------------------------------------------------------
def test_tree_separates_values_one_apart_at_scale_1e9():
    X = np.array([[1e9]] * 50 + [[1e9 + 1]] * 50)
    y = np.array([0] * 50 + [1] * 50)
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert tree.score(X, y) == 1.0
    assert 1e9 < tree.threshold_[0] < 1e9 + 1


def test_tree_separates_adjacent_floating_point_numbers():
    low = 1.0
    high = np.nextafter(1.0, 2.0)               # the very next number a float can hold
    X = np.array([[low]] * 20 + [[high]] * 20)
    y = np.array([0] * 20 + [1] * 20)
    tree = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert tree.score(X, y) == 1.0
    assert tree.predict([[low]])[0] == 0 and tree.predict([[high]])[0] == 1


def test_buffered_tree_keeps_tiny_gaps_across_chunks():
    tree = DecisionTreeClassifier(max_depth=3, buffer_size=100)
    X = np.array([[1e9]] * 50 + [[1e9 + 1]] * 50)
    y = np.array([0] * 50 + [1] * 50)
    order = np.random.default_rng(0).permutation(100)
    X, y = X[order], y[order]
    for start in range(0, 100, 25):
        tree.partial_fit(X[start:start + 25], y[start:start + 25])
    assert tree.n_retrains_ == 1
    assert tree.score(X, y) == 1.0


def test_scaler_keeps_precision_when_every_value_is_near_1e9():
    data = 1e9 + np.random.default_rng(0).normal(size=(5000, 2))
    scaler = StandardScaler()
    for chunk in np.array_split(data, 10):
        scaler.partial_fit(chunk)
    expected = (data - 1e9).std(axis=0)         # the same spread, computed near zero
    np.testing.assert_allclose(scaler.scale_, expected, rtol=1e-5)


def test_running_stats_keep_precision_when_every_value_is_near_1e9():
    data = 1e9 + np.random.default_rng(1).normal(size=(5000, 2))
    stats = RunningStats()
    for chunk in np.array_split(data, 10):
        stats.update_stats(chunk)
    np.testing.assert_allclose(stats.variance, (data - 1e9).var(axis=0), rtol=1e-5)


# ----------------------------------------------------------------------
# All-equal and duplicate-heavy data
# ----------------------------------------------------------------------
def _fit_seconds(X, y):
    start = time.perf_counter()
    tree = DecisionTreeClassifier(max_depth=5).fit(X, y)
    return tree, time.perf_counter() - start


def test_all_equal_features_give_one_leaf_and_are_not_slower_than_random_data():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=50000)
    equal_tree, equal_seconds = _fit_seconds(np.ones((50000, 5)), y)
    _, random_seconds = _fit_seconds(rng.normal(size=(50000, 5)), y)
    assert equal_tree.n_nodes_ == 1               # equal values can never be separated
    assert equal_seconds <= 2 * random_seconds    # a ratio, so it holds on slow machines too


def test_duplicate_heavy_data_is_learned_and_gives_the_same_tree_every_time():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 3, size=(50000, 5)).astype(float)   # only three distinct values per column
    y = (X[:, 2] >= 1).astype(int)
    first, _ = _fit_seconds(X, y)
    second, _ = _fit_seconds(X, y)
    assert first.score(X, y) == 1.0
    np.testing.assert_array_equal(first.predict(X), second.predict(X))
    assert first.get_depth() == second.get_depth()