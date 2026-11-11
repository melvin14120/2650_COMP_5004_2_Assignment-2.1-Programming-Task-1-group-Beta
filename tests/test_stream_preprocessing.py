"""
tests/test_stream_preprocessing.py
Unit tests for numcompute_stream.preprocessing

Covers: StandardScaler, SimpleImputer (mean, median, most_frequent, constant)
and OneHotEncoder with partial_fit. The main check throughout: feeding data in
chunks gives the same fitted values as the A1 class fitted on all of it at
once. Also: NaN, constant and all-NaN columns, empty chunks, categories that
appear late, warnings, state after fit/reset and input validation.

Run with: python -m pytest tests/test_stream_preprocessing.py -v   (from the repo root)
"""

import warnings

import numpy as np
import pytest

from numcompute import preprocessing as a1
from numcompute.preprocessing import NotFittedError
from numcompute_stream.preprocessing import Imputer, OneHotEncoder, SimpleImputer, StandardScaler


@pytest.fixture(autouse=True)
def _no_stray_warnings():
    """Fail any test that leaks a RuntimeWarning (e.g. 'invalid value in divide')."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


def _table(n=300, f=3, seed=0, nan_rate=0.1):
    """Random numeric table with scattered NaN."""
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=5.0, scale=3.0, size=(n, f))
    X[rng.random(X.shape) < nan_rate] = np.nan
    return X


def _codes(n=200, f=2, seed=0):
    """Small integer codes (many ties) with scattered NaN, for most_frequent."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 5, size=(n, f)).astype(float)
    X[rng.random(X.shape) < 0.1] = np.nan
    return X


def _stream(model, X, parts):
    """Feed X to model.partial_fit in `parts` nearly equal chunks."""
    for chunk in np.array_split(X, parts):
        model.partial_fit(chunk)
    return model


# ---------- StandardScaler ----------
def test_scaler_chunks_equal_a1_fit():
    X = _table()
    sc = _stream(StandardScaler(), X, 7)
    ref = a1.StandardScaler().fit(X)
    np.testing.assert_allclose(sc.mean_, ref.mean_)
    np.testing.assert_allclose(sc.var_, ref.var_)
    np.testing.assert_allclose(sc.scale_, ref.scale_)
    np.testing.assert_allclose(sc.transform(X), ref.transform(X))


def test_scaler_fit_matches_a1_fit_with_constant_and_all_nan_columns():
    X = np.column_stack([_table(30, 1)[:, 0], np.full(30, 0.1), np.full(30, np.nan)])
    sc, ref = StandardScaler().fit(X), a1.StandardScaler().fit(X)
    np.testing.assert_allclose(sc.mean_, ref.mean_)          # NaN equals NaN in assert_allclose
    np.testing.assert_allclose(sc.scale_, ref.scale_)
    np.testing.assert_allclose(sc.var_, ref.var_)
    assert np.isnan(sc.mean_[2]) and sc.scale_[1] == 0.0


def test_constant_column_transforms_to_exactly_zero_across_chunks():
    X = np.full((60, 1), 0.1)
    sc = _stream(StandardScaler(), X, 3)
    assert sc.mean_[0] == 0.1 and sc.scale_[0] == 0.0
    assert (sc.transform(X) == 0.0).all()                    # not 1e-17 rounding noise


def test_constant_first_chunk_then_varying_chunk():
    first, second = np.full((10, 1), 5.0), np.array([[1.0], [9.0]])
    sc = StandardScaler().partial_fit(first)
    assert sc.scale_[0] == 0.0
    sc.partial_fit(second)
    ref = a1.StandardScaler().fit(np.vstack([first, second]))
    np.testing.assert_allclose(sc.mean_, ref.mean_)
    np.testing.assert_allclose(sc.scale_, ref.scale_)


def test_scaler_flags_with_mean_and_with_std():
    X = _table(50, 2)
    sc = _stream(StandardScaler(with_mean=False, with_std=False), X, 2)
    assert sc.mean_.tolist() == [0.0, 0.0] and sc.scale_.tolist() == [1.0, 1.0]
    ref = a1.StandardScaler(with_mean=False).fit(X)
    np.testing.assert_allclose(_stream(StandardScaler(with_mean=False), X, 2).scale_, ref.scale_)


def test_scaler_counts_only_numbers_per_column():
    X = np.array([[1.0, np.nan], [2.0, np.nan], [np.nan, 5.0]])
    sc = StandardScaler().partial_fit(X)
    assert sc.n_samples_seen_.tolist() == [2, 1]
    sc.partial_fit(X)
    assert sc.n_samples_seen_.tolist() == [4, 2]


def test_scaler_empty_chunks_and_unfitted_use():
    sc = StandardScaler()
    sc.partial_fit(np.empty((0, 3)))
    assert not hasattr(sc, "n_features_in_")                 # an empty first chunk decides nothing
    with pytest.raises(NotFittedError):
        sc.transform(np.zeros((1, 3)))
    sc.partial_fit(np.zeros((2, 3))).partial_fit(np.empty((0, 3)))
    assert sc.n_samples_seen_.tolist() == [2, 2, 2]
    with pytest.raises(ValueError, match="fitted with 3 features"):
        sc.partial_fit(np.empty((0, 4)))


def test_scaler_bad_input_raises_informative_errors():
    sc = StandardScaler()
    with pytest.raises(ValueError, match="2-D"):
        sc.partial_fit([1.0, 2.0])
    with pytest.raises(ValueError, match="infinity"):
        sc.partial_fit([[np.inf]])
    with pytest.raises(TypeError, match="numeric"):
        sc.partial_fit([["a"]])
    with pytest.raises(ValueError, match="at least one feature"):
        sc.partial_fit(np.zeros((3, 0)))
    sc.partial_fit(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="fitted with 3 features"):
        sc.partial_fit(np.zeros((2, 5)))
    with pytest.raises(ValueError, match="at least one sample"):
        StandardScaler().fit(np.empty((0, 2)))               # fit keeps the A1 rule for empty input


def test_scaler_fit_forgets_but_partial_fit_continues_after_fit():
    A, B, C = _table(40, 2, 1), _table(50, 2, 2), _table(30, 2, 3)
    sc = StandardScaler().partial_fit(A)
    sc.fit(B)
    np.testing.assert_allclose(sc.mean_, a1.StandardScaler().fit(B).mean_)      # A is forgotten
    sc.partial_fit(C)
    np.testing.assert_allclose(sc.mean_, a1.StandardScaler().fit(np.vstack([B, C])).mean_)


def test_scaler_reset_returns_self_and_allows_a_new_width():
    sc = StandardScaler().partial_fit(np.zeros((2, 3)))
    assert sc.reset() is sc and not hasattr(sc, "mean_")
    sc.partial_fit(np.zeros((2, 5)))
    assert sc.n_features_in_ == 5


def test_scaler_fit_transform_repr_and_chaining():
    X = _table(40, 2)
    sc = StandardScaler()
    assert sc.partial_fit(X) is sc
    np.testing.assert_allclose(StandardScaler().fit_transform(X), a1.StandardScaler().fit_transform(X))
    assert repr(sc) == "StandardScaler(with_mean=True, with_std=True)"


def test_scaler_non_contiguous_input_and_input_not_modified():
    X = _table(60, 6)
    view, before = X[:, ::2], X.copy()
    a = StandardScaler().partial_fit(view)
    b = StandardScaler().partial_fit(np.ascontiguousarray(view))
    np.testing.assert_array_equal(a.mean_, b.mean_)
    np.testing.assert_array_equal(X, before)


# ---------- SimpleImputer ----------
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent", "constant"])
def test_imputer_fit_matches_a1_fit(strategy):
    X = _codes(100, 3)
    ours = SimpleImputer(strategy, fill_value=7).fit(X)
    ref = a1.SimpleImputer(strategy, fill_value=7).fit(X)
    np.testing.assert_allclose(ours.statistics_, ref.statistics_)
    np.testing.assert_allclose(ours.transform(X), ref.transform(X))


def test_mean_chunks_equal_a1():
    X = _table(240, 3)
    imp = _stream(SimpleImputer("mean"), X, 6)
    ref = a1.SimpleImputer("mean").fit(X)
    np.testing.assert_allclose(imp.statistics_, ref.statistics_)
    np.testing.assert_allclose(imp.transform(X), ref.transform(X))


def test_median_is_exact_below_capacity():
    X = _table(240, 3, seed=4)
    imp = _stream(SimpleImputer("median", capacity=1000), X, 6)
    np.testing.assert_allclose(imp.statistics_, a1.SimpleImputer("median").fit(X).statistics_)


def test_median_memory_is_bounded_by_capacity():
    X = np.random.default_rng(5).normal(size=(5000, 1))
    imp = _stream(SimpleImputer("median", capacity=100, random_state=0), X, 50)
    assert imp._quantiles.sample_.shape[0] == 100            # only `capacity` rows are ever kept
    assert abs(imp.statistics_[0] - np.median(X)) < 0.4


def test_most_frequent_chunks_equal_a1_including_ties():
    X = _codes(200, 2, seed=6)
    imp = _stream(SimpleImputer("most_frequent"), X, 8)
    np.testing.assert_array_equal(imp.statistics_, a1.SimpleImputer("most_frequent").fit(X).statistics_)
    tie = SimpleImputer("most_frequent").partial_fit([[2.0], [1.0]])
    assert tie.statistics_[0] == 1.0                         # equal counts: the smallest value wins


def test_most_frequent_counts_across_chunks_not_per_chunk():
    imp = SimpleImputer("most_frequent")
    imp.partial_fit([[1.0], [1.0], [2.0]])                   # on its own this chunk says 1
    assert imp.statistics_[0] == 1.0
    imp.partial_fit([[2.0], [2.0], [3.0], [3.0]])            # overall: 1 x2, 2 x3, 3 x2
    assert imp.statistics_[0] == 2.0


def test_constant_strategy_uses_fill_value_and_ignores_the_data():
    imp = SimpleImputer("constant", fill_value=-1).partial_fit([[5.0, np.nan], [6.0, 1.0]])
    assert imp.statistics_.tolist() == [-1.0, -1.0]
    assert imp.transform([[np.nan, 3.0]]).tolist() == [[-1.0, 3.0]]


def test_statistics_follow_the_stream():
    imp = SimpleImputer("mean")
    imp.partial_fit([[1.0], [1.0]])
    assert imp.statistics_[0] == 1.0
    imp.partial_fit([[4.0], [4.0]])
    assert imp.statistics_[0] == 2.5


def test_all_nan_column_warns_in_fit_but_not_in_partial_fit():
    X = np.array([[1.0, np.nan], [3.0, np.nan]])
    with pytest.warns(UserWarning, match=r"columns \[1\] are entirely NaN"):
        SimpleImputer("mean", fill_value=9).fit(X)
    imp = SimpleImputer("mean", fill_value=9)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        imp.partial_fit(X)
    assert imp.statistics_.tolist() == [2.0, 9.0] and imp.n_samples_seen_.tolist() == [2, 0]
    imp.partial_fit([[5.0, 4.0]])                            # the column gets its first number later
    assert imp.statistics_.tolist() == [3.0, 4.0] and imp.n_samples_seen_.tolist() == [3, 1]


def test_imputer_transform_fills_every_nan_and_copies():
    X = _table(40, 2)
    before = X.copy()
    out = _stream(SimpleImputer("median"), X, 4).transform(X)
    assert not np.isnan(out).any()
    np.testing.assert_array_equal(X, before)


def test_imputer_invalid_parameters_raise_value_error():
    with pytest.raises(ValueError, match="strategy"):
        SimpleImputer("average").partial_fit([[1.0]])
    with pytest.raises(ValueError, match="strategy"):
        SimpleImputer("average").fit([[1.0]])
    with pytest.raises(ValueError, match="fill_value"):
        SimpleImputer(fill_value="x").partial_fit([[1.0]])
    with pytest.raises(ValueError, match="finite"):
        SimpleImputer(fill_value=np.inf).partial_fit([[1.0]])
    with pytest.raises(ValueError):
        SimpleImputer(capacity=0)
    with pytest.raises(TypeError):
        SimpleImputer(capacity=2.5)


def test_imputer_chunk_handling_reset_and_unfitted():
    imp = SimpleImputer()
    imp.partial_fit(np.empty((0, 2)))
    assert not hasattr(imp, "n_features_in_")
    with pytest.raises(NotFittedError):
        imp.transform(np.zeros((1, 2)))
    imp.partial_fit([[1.0, 2.0]])
    with pytest.raises(ValueError, match="fitted with 2 features"):
        imp.partial_fit(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="infinity"):
        imp.partial_fit([[np.inf, 1.0]])
    with pytest.raises(TypeError, match="numeric"):
        imp.partial_fit([["a", "b"]])
    assert imp.reset() is imp and not hasattr(imp, "statistics_")


def test_imputer_alias_and_repr():
    assert Imputer is SimpleImputer
    assert "strategy='median'" in repr(SimpleImputer("median", capacity=5))


# ---------- OneHotEncoder ----------
def _words(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.choice(["red", "green", "blue"], n), rng.choice(["yes", "no"], n)])


def test_encoder_chunks_equal_a1_fit():
    X = _words()
    enc = _stream(OneHotEncoder(), X, 5)
    ref = a1.OneHotEncoder().fit(X)
    for ours, theirs in zip(enc.categories_, ref.categories_):
        np.testing.assert_array_equal(ours, theirs)
    np.testing.assert_array_equal(enc.transform(X), ref.transform(X))
    assert enc.n_features_out_ == 5


def test_a_late_category_grows_the_output_and_warns():
    enc = OneHotEncoder().partial_fit(np.array([["a"], ["b"]]))
    with pytest.warns(UserWarning, match=r"column\(s\) \[0\].*grew from 2 to 3"):
        enc.partial_fit(np.array([["c"], ["a"]]))
    assert enc.categories_[0].tolist() == ["a", "b", "c"]
    assert enc.transform(np.array([["c"]])).tolist() == [[0.0, 0.0, 1.0]]


def test_no_warning_when_the_chunk_brings_nothing_new():
    enc = OneHotEncoder().partial_fit(np.array([["a"], ["b"]]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        enc.partial_fit(np.array([["b"], ["a"], ["a"]]))
    assert enc.n_features_out_ == 2


def test_unknown_category_becomes_known_after_partial_fit():
    enc = OneHotEncoder().partial_fit(np.array([["a"], ["b"]]))
    with pytest.raises(ValueError, match="unknown categories"):
        enc.transform(np.array([["c"]]))
    with pytest.warns(UserWarning):
        enc.partial_fit(np.array([["c"]]))
    assert enc.transform(np.array([["c"]])).sum() == 1.0
    ignoring = OneHotEncoder(handle_unknown="ignore").partial_fit(np.array([["a"], ["b"]]))
    assert ignoring.transform(np.array([["z"]])).tolist() == [[0.0, 0.0]]


def test_numeric_categories_with_nan_across_chunks():
    enc = OneHotEncoder()
    enc.partial_fit(np.array([[1.0], [np.nan]]))
    with pytest.warns(UserWarning):
        enc.partial_fit(np.array([[2.0], [np.nan]]))
    assert enc.n_features_out_ == 3                          # 1.0, 2.0 and ONE NaN category
    out = enc.transform(np.array([[np.nan], [2.0]]))
    assert out.tolist() == [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]


def test_encoder_first_chunk_empty_and_unfitted_use():
    enc = OneHotEncoder()
    enc.partial_fit(np.empty((0, 2), dtype=str))
    assert not hasattr(enc, "categories_")
    with pytest.raises(NotFittedError):
        enc.transform(np.array([["a", "b"]]))
    with pytest.raises(NotFittedError):
        enc.n_features_out_
    enc.partial_fit(np.array([["a", "b"]])).partial_fit(np.empty((0, 2), dtype=str))
    assert enc.n_features_out_ == 2


def test_encoder_bad_input_raises_informative_errors():
    enc = OneHotEncoder().partial_fit(np.array([["a", "b"]]))
    with pytest.raises(ValueError, match="fitted with 2 features"):
        enc.partial_fit(np.array([["a"]]))
    with pytest.raises(TypeError, match="partial_fit got number data"):
        enc.partial_fit(np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="2-D"):
        enc.partial_fit(np.array(["a", "b"]))
    with pytest.raises(ValueError, match="handle_unknown"):
        OneHotEncoder(handle_unknown="skip").partial_fit(np.array([["a"]]))
    mixed = OneHotEncoder().partial_fit(np.array([["a"]], dtype=object))
    with pytest.raises(TypeError, match="incomparable"):
        mixed.partial_fit(np.array([[3]], dtype=object))


def test_encoder_fit_restarts_and_feature_names_follow_the_growth():
    enc = OneHotEncoder().partial_fit(np.array([["a"], ["b"]]))
    with pytest.warns(UserWarning):
        enc.partial_fit(np.array([["c"]]))
    assert enc.get_feature_names_out(["k"]).tolist() == ["k_a", "k_b", "k_c"]
    enc.fit(np.array([["z"]]))                                # the inherited A1 fit forgets everything
    assert enc.categories_[0].tolist() == ["z"]


def test_encoder_dtype_and_returns_self():
    enc = OneHotEncoder(dtype=np.int8)
    assert enc.partial_fit(np.array([["a"], ["b"]])) is enc
    assert enc.transform(np.array([["a"]])).dtype == np.int8