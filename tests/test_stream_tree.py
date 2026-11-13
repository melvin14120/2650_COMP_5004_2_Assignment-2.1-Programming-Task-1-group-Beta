"""
tests/test_stream_tree.py
Unit tests for numcompute_stream.tree

Covers: batch fit/predict, depth and min_samples_split limits, ties and
duplicates, NaN handling, the vectorised split search (checked against a
reference loop), buffered streaming (buffer, retrain, flush, empty chunks)
and input validation.

Run with: python -m pytest tests/test_stream_tree.py -v   (from the repo root)
"""

import warnings

import numpy as np
import pytest

from numcompute.preprocessing import NotFittedError
from numcompute_stream import tree as tr
from numcompute_stream.tree import DecisionTreeClassifier


@pytest.fixture(autouse=True)
def _no_stray_warnings():
    """Fail any test that leaks a RuntimeWarning (e.g. 'invalid value in divide')."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


def _xor(n=400, seed=0):
    """Noise-free XOR-style data: the label depends on features 0 and 1 only."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    return X, ((X[:, 0] > 0.2) ^ (X[:, 1] > -0.3)).astype(int)


def _brute_best_gain(X, y, criterion):
    """Reference loop: best impurity decrease over every feature and every midpoint."""
    def imp(v):
        if v.size == 0:
            return 0.0
        p = np.bincount(v, minlength=2) / v.size
        return 1 - (p ** 2).sum() if criterion == "gini" else -sum(q * np.log2(q) for q in p if q > 0)

    parent, best = imp(y), 0.0
    for j in range(X.shape[1]):
        values = np.unique(X[:, j])
        for a, b in zip(values[:-1], values[1:]):
            mask = X[:, j] <= (a + b) / 2
            best = max(best, parent - (mask.sum() * imp(y[mask]) + (~mask).sum() * imp(y[~mask])) / y.size)
    return best


# ---------- batch behaviour ----------
def test_fit_separable_data_is_perfect():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert model.predict([[0.4], [2.6]]).tolist() == [0, 1]
    assert model.score(X, y) == 1.0


def test_fit_returns_self():
    model = DecisionTreeClassifier()
    assert model.fit([[0.0], [1.0]], [0, 1]) is model


def test_threshold_is_midpoint_between_distinct_values():
    model = DecisionTreeClassifier(max_depth=1).fit([[1.0], [3.0]], [0, 1])
    assert model.threshold_[0] == 2.0


def test_xor_needs_depth_two():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 5, dtype=float)
    y = np.array([0, 1, 1, 0] * 5)
    assert DecisionTreeClassifier(max_depth=1).fit(X, y).score(X, y) < 1.0
    assert DecisionTreeClassifier(max_depth=2).fit(X, y).score(X, y) == 1.0


def test_max_depth_zero_gives_single_majority_leaf():
    model = DecisionTreeClassifier(max_depth=0).fit([[0.0], [1.0], [2.0]], [1, 1, 0])
    assert model.n_nodes_ == 1 and model.get_depth() == 0
    assert model.predict([[5.0]]).tolist() == [1]


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_max_depth_is_respected(depth):
    X, y = _xor()
    assert DecisionTreeClassifier(max_depth=depth).fit(X, y).get_depth() <= depth


def test_min_samples_split_stops_growth():
    X, y = _xor(50)
    assert DecisionTreeClassifier(max_depth=None, min_samples_split=51).fit(X, y).n_nodes_ == 1


def test_single_class_is_one_leaf():
    X = np.random.default_rng(0).normal(size=(10, 2))
    model = DecisionTreeClassifier().fit(X, np.zeros(10, dtype=int))
    assert model.n_nodes_ == 1
    assert model.predict(np.zeros((3, 2))).tolist() == [0, 0, 0]
    assert model.classes_.tolist() == [0]


def test_all_equal_features_cannot_split():
    assert DecisionTreeClassifier().fit(np.ones((8, 3)), [0, 1] * 4).n_nodes_ == 1


def test_duplicate_values_are_never_separated():
    X = np.array([[1.0], [1.0], [1.0], [2.0]])
    y = np.array([0, 1, 0, 1])
    model = DecisionTreeClassifier().fit(X, y)
    # Identical feature values must land in the same leaf, whatever their labels.
    leaves = model._leaf_values(X)
    assert np.array_equal(leaves[0], leaves[1]) and np.array_equal(leaves[1], leaves[2])


def test_single_sample_fit():
    assert DecisionTreeClassifier().fit([[1.0, 2.0]], ["a"]).predict([[9.0, 9.0]]).tolist() == ["a"]


def test_string_labels_and_predict_proba():
    X = np.array([[0.0], [1.0], [5.0], [6.0]])
    y = np.array(["cat", "cat", "dog", "dog"])
    model = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert model.predict([[0.5], [5.5]]).tolist() == ["cat", "dog"]
    assert model.classes_.tolist() == ["cat", "dog"]
    np.testing.assert_allclose(model.predict_proba([[0.5], [5.5]]).sum(axis=1), 1.0)


def test_feature_importances_identify_the_informative_feature():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    importances = DecisionTreeClassifier(max_depth=2).fit(X, (X[:, 2] > 0).astype(int)).feature_importances_
    assert importances.argmax() == 2
    assert importances.sum() == pytest.approx(1.0)


def test_non_contiguous_input_matches_contiguous():
    X, y = _xor()
    view = X[:, ::2]
    a = DecisionTreeClassifier(max_depth=3).fit(view, y)
    b = DecisionTreeClassifier(max_depth=3).fit(np.ascontiguousarray(view), y)
    assert np.array_equal(a.predict(view), b.predict(np.ascontiguousarray(view)))


def test_input_is_not_modified():
    X, y = _xor(60)
    X_before, y_before = X.copy(), y.copy()
    DecisionTreeClassifier().fit(X, y).predict(X)
    assert np.array_equal(X, X_before) and np.array_equal(y, y_before)


# ---------- impurity and the vectorised split search ----------
def test_impurity_values():
    assert tr._impurity(np.array([5.0, 5.0]), "gini") == pytest.approx(0.5)
    assert tr._impurity(np.array([5.0, 5.0]), "entropy") == pytest.approx(1.0)
    assert tr._impurity(np.array([4.0, 0.0]), "entropy") == 0.0
    assert tr._impurity(np.array([0.0, 0.0]), "gini") == 0.0


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_best_split_matches_reference_loop(criterion):
    rng = np.random.default_rng(3)
    for _ in range(5):
        X = rng.integers(0, 6, size=(40, 3)).astype(float)       # many ties on purpose
        y = (X[:, 0] + rng.integers(0, 3, 40) > 4).astype(int)
        found = tr._best_split(X, np.eye(2)[y], np.arange(3), criterion)
        brute = _brute_best_gain(X, y, criterion)
        if found is None:
            assert brute <= 1e-9
        else:
            assert found[3] == pytest.approx(brute, abs=1e-9)


def test_tie_break_prefers_lowest_feature():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    assert DecisionTreeClassifier(max_depth=1).fit(X, y).feature_[0] == 0


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_both_criteria_learn_the_same_easy_problem(criterion):
    X, y = _xor()
    assert DecisionTreeClassifier(criterion=criterion, max_depth=4).fit(X, y).score(X, y) > 0.95


# ---------- nan handling ----------
def test_training_and_predicting_with_nan():
    X, y = _xor()
    X[np.random.default_rng(0).random(X.shape) < 0.1] = np.nan
    model = DecisionTreeClassifier(max_depth=4).fit(X, y)
    assert np.mean(model.predict(X) == y) > 0.7
    assert model.predict(np.full((2, 4), np.nan)).shape == (2,)   # an all-NaN row still gets a label


def test_all_nan_feature_is_ignored():
    X = np.column_stack([np.full(10, np.nan), np.arange(10.0)])
    y = (np.arange(10) > 4).astype(int)
    model = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert model.feature_[0] == 1
    assert model.score(X, y) == 1.0


# ---------- buffered streaming ----------
def test_no_tree_until_the_buffer_fills():
    X, y = _xor(300)
    model = DecisionTreeClassifier(buffer_size=250)
    model.partial_fit(X[:100], y[:100]).partial_fit(X[100:200], y[100:200])
    assert not model.is_fitted and model.n_buffered_ == 200 and model.n_retrains_ == 0
    with pytest.raises(NotFittedError, match="200 of 250"):
        model.predict(X[:5])
    model.partial_fit(X[200:], y[200:])
    assert model.is_fitted and model.n_buffered_ == 0 and model.n_retrains_ == 1


def test_retrain_uses_only_the_buffered_chunks_and_clears_the_buffer():
    model = DecisionTreeClassifier(buffer_size=4, max_depth=1)
    model.partial_fit([[0.0], [1.0]], [0, 0]).partial_fit([[2.0], [3.0]], [0, 0])
    assert model.classes_.tolist() == [0]
    model.partial_fit([[0.0], [1.0]], [5, 5]).partial_fit([[2.0], [3.0]], [7, 7])
    assert model.classes_.tolist() == [5, 7]          # the first buffer was not kept
    assert model.n_retrains_ == 2 and model.n_buffered_ == 0


def test_predict_uses_the_last_tree_between_retrains():
    X, y = _xor(600)
    model = DecisionTreeClassifier(buffer_size=300, max_depth=4)
    model.partial_fit(X[:300], y[:300])
    before = model.predict(X[300:400]).copy()
    model.partial_fit(X[300:400], 1 - y[300:400])     # buffered only, so the tree must not change
    assert np.array_equal(model.predict(X[300:400]), before)
    assert model.n_retrains_ == 1


def test_chunk_larger_than_the_buffer_retrains_once():
    X, y = _xor(500)
    model = DecisionTreeClassifier(buffer_size=100).partial_fit(X, y)
    assert model.n_retrains_ == 1 and model.n_seen_ == 500


def test_flush_trains_on_a_partial_buffer():
    X, y = _xor(50)
    model = DecisionTreeClassifier(buffer_size=1000).partial_fit(X, y).flush()
    assert model.is_fitted and model.n_buffered_ == 0
    assert not DecisionTreeClassifier().flush().is_fitted        # empty buffer: nothing happens


def test_fit_discards_the_buffer():
    X, y = _xor(100)
    model = DecisionTreeClassifier(buffer_size=1000).partial_fit(X[:30], y[:30])
    model.fit(X, y)
    assert model.n_buffered_ == 0 and model.n_retrains_ == 1


def test_empty_chunk_is_ignored():
    model = DecisionTreeClassifier(buffer_size=10)
    model.partial_fit(np.empty((0, 3)), np.empty(0))
    assert model.n_seen_ == 0 and model.n_buffered_ == 0
    model.partial_fit(np.zeros((2, 3)), [0, 1])
    model.partial_fit(np.empty((0, 3)), np.empty(0))
    assert model.n_seen_ == 2
    with pytest.raises(ValueError, match="fitted with 3 features"):
        model.partial_fit(np.empty((0, 4)), np.empty(0))


def test_feature_count_must_stay_constant():
    model = DecisionTreeClassifier().partial_fit(np.zeros((2, 3)), [0, 1])
    with pytest.raises(ValueError, match="fitted with 3 features"):
        model.partial_fit(np.zeros((2, 4)), [0, 1])
    model.flush()
    with pytest.raises(ValueError, match="fitted with 3 features"):
        model.predict(np.zeros((1, 5)))


def test_streaming_reaches_good_accuracy():
    X, y = _xor(2000, seed=5)
    model = DecisionTreeClassifier(max_depth=5, buffer_size=400)
    for start in range(0, 1600, 100):
        model.partial_fit(X[start:start + 100], y[start:start + 100])
    assert model.score(X[1600:], y[1600:]) > 0.9


def test_memory_bytes_grows_with_the_buffer():
    model = DecisionTreeClassifier(buffer_size=10 ** 6)
    before = model.memory_bytes()
    model.partial_fit(np.zeros((100, 5)), np.zeros(100))
    assert model.memory_bytes() > before


# ---------- validation and errors ----------
@pytest.mark.parametrize("kwargs", [
    dict(criterion="mse"), dict(min_samples_split=1), dict(buffer_size=0), dict(max_depth=-1),
    dict(max_features=0), dict(max_features=1.5), dict(max_features="cube"),
])
def test_invalid_hyperparameters_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**kwargs)


@pytest.mark.parametrize("kwargs", [dict(buffer_size=2.5), dict(max_features=True), dict(max_features=[1])])
def test_wrong_hyperparameter_types_raise_type_error(kwargs):
    with pytest.raises(TypeError):
        DecisionTreeClassifier(**kwargs)


def test_max_features_options_and_reproducibility():
    X, y = _xor(300)
    a = DecisionTreeClassifier(max_features="sqrt", random_state=7).fit(X, y)
    b = DecisionTreeClassifier(max_features="sqrt", random_state=7).fit(X, y)
    assert np.array_equal(a.feature_, b.feature_)
    assert DecisionTreeClassifier(max_features=0.5)._n_candidate_features(10) == 5
    assert DecisionTreeClassifier(max_features="log2")._n_candidate_features(8) == 3
    assert DecisionTreeClassifier(max_features=99)._n_candidate_features(4) == 4


def test_bad_inputs_raise_informative_errors():
    model = DecisionTreeClassifier()
    with pytest.raises(ValueError, match="2-D"):
        model.fit([1.0, 2.0, 3.0], [0, 1, 0])
    with pytest.raises(ValueError, match="3 rows but y has 2"):
        model.fit(np.zeros((3, 1)), [0, 1])
    with pytest.raises(ValueError, match="infinity"):
        model.fit([[np.inf], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="NaN labels"):
        model.fit([[0.0], [1.0]], [0.0, np.nan])
    with pytest.raises(ValueError, match="at least one sample"):
        model.fit(np.empty((0, 2)), [])
    with pytest.raises(ValueError, match="y of shape"):
        model.fit(np.zeros((2, 1)), np.zeros((2, 1)))
    with pytest.raises(TypeError, match="numeric input"):
        model.fit([["a"], ["b"]], [0, 1])
    with pytest.raises(TypeError, match="dtype object"):
        model.fit([[0.0], [1.0]], np.array([None, 1], dtype=object))


def test_use_before_training_raises_not_fitted():
    model = DecisionTreeClassifier()
    for call in (lambda: model.predict([[1.0]]), lambda: model.predict_proba([[1.0]]),
                 lambda: model.get_depth(), lambda: model.n_nodes_):
        with pytest.raises(NotFittedError):
            call()
    assert not hasattr(model, "classes_")          # NotFittedError is also an AttributeError


def test_score_on_zero_samples_is_nan():
    model = DecisionTreeClassifier().fit([[0.0], [1.0]], [0, 1])
    assert np.isnan(model.score(np.empty((0, 1)), np.empty(0)))


def test_repr_and_get_params():
    model = DecisionTreeClassifier(max_depth=3, buffer_size=7)
    assert "buffer_size=7" in repr(model)
    assert model.get_params()["max_depth"] == 3

    

# ---------- gaps found by the fault-injection check ----------
def test_nan_follows_the_larger_child_at_prediction():
    # Right child has 4 training samples, left has 3: NaN must go right (class 1).
    X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0], [13.0]])
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    model = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert not model.nan_left_[0]
    assert model.predict([[np.nan]]).tolist() == [1]
    # Mirror image: left child has 4 samples, right has 3, so NaN must go left (class 0).
    X2 = np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    y2 = np.array([0, 0, 0, 0, 1, 1, 1])
    model2 = DecisionTreeClassifier(max_depth=1).fit(X2, y2)
    assert model2.nan_left_[0]
    assert model2.predict([[np.nan]]).tolist() == [0]


def test_a_feature_with_many_nan_is_penalised_in_the_split_choice():
    # Feature 0 separates the classes perfectly but is NaN in half the rows
    # (scaled gain 0.25); feature 1 is complete and separates them less well
    # (gain 0.3). The complete feature must win the root split.
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    f0 = np.array([0.0, np.nan, np.nan, 1.0, 10.0, np.nan, np.nan, 11.0])
    f1 = np.array([0.0, 1.0, 2.0, 10.0, 3.0, 11.0, 12.0, 13.0])
    model = DecisionTreeClassifier(max_depth=1).fit(np.column_stack([f0, f1]), y)
    assert model.feature_[0] == 1


def test_feature_importances_are_weighted_by_node_size_and_match_a_hand_calculation():
    # Root splits on feature 0 (decrease 4/3), then the right node (3 samples)
    # splits on feature 1 (decrease 4/3): both features contribute equally.
    X = np.array([[0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 0])
    model = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert model.feature_[0] == 0
    np.testing.assert_allclose(model.feature_importances_, [0.5, 0.5])


def test_threshold_between_adjacent_floats_stays_below_the_upper_value():
    # lo has an odd last bit, so lo + (hi - lo) / 2 rounds up to hi; the tree
    # must fall back to lo so that hi still goes to the right child.
    lo = np.nextafter(1.0, 2.0)
    hi = np.nextafter(lo, 2.0)
    model = DecisionTreeClassifier(max_depth=1).fit(np.array([[lo], [hi]]), np.array([0, 1]))
    assert model.threshold_[0] == lo
    assert model.predict(np.array([[lo], [hi]])).tolist() == [0, 1]

    

def test_split_search_returns_none_when_there_is_nothing_to_split():
    # one sample, or no candidate features: no cut exists
    assert tr._best_split(np.array([[1.0]]), np.array([[1.0, 0.0]]), np.array([0]), "gini") is None
    assert tr._best_split(np.array([[0.0], [1.0]]), np.eye(2), np.array([], dtype=int), "gini") is None