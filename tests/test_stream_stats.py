"""
tests/test_stream_stats.py
Unit tests for numcompute_stream.stats

Covers: RunningStats (chunked results equal the A1 batch results, NaN, empty
and constant chunks, merge, numerical stability), StreamingQuantile (exact
below capacity, window, reservoir, reproducibility) and StreamingHistogram
(equals A1 histogram, range from the first chunk, under/overflow, sliding
window), plus input validation for all three.

Run with: python -m pytest tests/test_stream_stats.py -v   (from the repo root)
"""

import warnings

import numpy as np
import pytest

from numcompute.preprocessing import NotFittedError
from numcompute.stats import StreamingStats, describe, histogram, quantile
from numcompute_stream.stats import RunningStats, StreamingHistogram, StreamingQuantile


@pytest.fixture(autouse=True)
def _no_stray_warnings():
    """Fail any test that leaks a RuntimeWarning (e.g. 'invalid value in divide')."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


def _data(n=500, f=4, seed=0, nan_rate=0.1):
    """Random table with scattered NaN."""
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=3.0, scale=2.0, size=(n, f))
    X[rng.random(X.shape) < nan_rate] = np.nan
    return X


def _feed(model, X, sizes):
    """Pass X to model.update_stats in chunks of the given sizes (0 = empty chunk)."""
    start = 0
    for size in sizes:
        model.update_stats(X[start:start + size])
        start += size
    return model


# ---------- RunningStats: results ----------
def test_chunked_results_equal_the_a1_batch_results():
    X = _data()
    rs = _feed(RunningStats(), X, [100, 0, 7, 250, 143])
    expected = describe(X, axis=0)
    np.testing.assert_allclose(rs.mean, expected["mean"])
    np.testing.assert_allclose(rs.std, expected["std"])
    np.testing.assert_allclose(rs.minimum, expected["min"])
    np.testing.assert_allclose(rs.maximum, expected["max"])
    np.testing.assert_array_equal(rs.count, (~np.isnan(X)).sum(axis=0))


def test_one_column_matches_a1_streaming_stats_sample_by_sample():
    x = _data(200, 1, seed=1)[:, 0]
    a1 = StreamingStats()
    for value in x:
        a1.update(value)
    rs = RunningStats().update_stats(x.reshape(-1, 1))
    assert rs.mean[0] == pytest.approx(a1.mean)
    assert rs.variance[0] == pytest.approx(a1.variance)


def test_ddof_one_gives_the_sample_variance():
    X = _data(300, 3, seed=2)
    rs = _feed(RunningStats(ddof=1), X, [120, 180])
    np.testing.assert_allclose(rs.variance, np.nanvar(X, axis=0, ddof=1))


def test_result_does_not_depend_on_how_the_stream_is_chunked():
    X = _data(240, 3, seed=3)
    one = RunningStats().update_stats(X)
    many = _feed(RunningStats(), X, [1] * 240)
    np.testing.assert_allclose(one.mean, many.mean)
    np.testing.assert_allclose(one.variance, many.variance)


def test_large_offset_does_not_lose_precision():
    rng = np.random.default_rng(4)
    X = 1e9 + rng.normal(size=(1000, 1))          # E[x^2] - E[x]^2 would cancel to garbage here
    rs = _feed(RunningStats(), X, [100] * 10)
    assert rs.variance[0] == pytest.approx(np.var(X), rel=1e-6)


def test_all_nan_feature_has_no_statistics():
    X = np.column_stack([np.full(5, np.nan), np.arange(5.0)])
    rs = RunningStats().update_stats(X)
    assert rs.count.tolist() == [0, 5]
    assert np.isnan(rs.mean[0]) and np.isnan(rs.variance[0]) and np.isnan(rs.minimum[0]) and np.isnan(rs.maximum[0])
    assert rs.mean[1] == 2.0


def test_single_value_and_constant_chunks():
    rs = RunningStats().update_stats([[5.0]])
    assert rs.variance[0] == 0.0
    assert np.isnan(RunningStats(ddof=1).update_stats([[5.0]]).variance[0])
    rs = _feed(RunningStats(), np.full((60, 2), 7.0), [20, 20, 20])          # zero-variance chunks
    assert rs.variance.tolist() == [0.0, 0.0] and rs.mean.tolist() == [7.0, 7.0]


def test_non_contiguous_input_matches_contiguous():
    X = _data(100, 6, seed=5)
    view = X[:, ::2]
    a = RunningStats().update_stats(view)
    b = RunningStats().update_stats(np.ascontiguousarray(view))
    np.testing.assert_array_equal(a.mean, b.mean)


def test_input_is_not_modified_and_summary_has_all_keys():
    X = _data(50, 2, seed=6)
    before = X.copy()
    summary = RunningStats().update_stats(X).summary()
    np.testing.assert_array_equal(X, before)
    assert set(summary) == {"count", "mean", "variance", "std", "min", "max"}


# ---------- RunningStats: merge, empty chunks, state ----------
def test_merge_equals_one_accumulator_over_all_the_data():
    X = _data(200, 3, seed=7)
    a = RunningStats().update_stats(X[:80])
    b = RunningStats().update_stats(X[80:])
    a.merge(b)
    whole = RunningStats().update_stats(X)
    np.testing.assert_allclose(a.mean, whole.mean)
    np.testing.assert_allclose(a.variance, whole.variance)
    np.testing.assert_array_equal(a.minimum, whole.minimum)
    assert b.count.sum() == (~np.isnan(X[80:])).sum()                       # other is not modified


def test_merge_with_unfitted_accumulators():
    X = _data(30, 2, seed=8)
    empty = RunningStats()
    assert empty.merge(RunningStats()) is empty and not hasattr(empty, "n_features_in_")
    empty.merge(RunningStats().update_stats(X))
    np.testing.assert_allclose(empty.mean, RunningStats().update_stats(X).mean)
    full = RunningStats().update_stats(X)
    full.merge(RunningStats())
    assert full.count.sum() == (~np.isnan(X)).sum()


def test_merge_rejects_bad_partners():
    rs = RunningStats().update_stats(np.zeros((2, 3)))
    with pytest.raises(TypeError, match="expects a RunningStats"):
        rs.merge("stats")
    with pytest.raises(ValueError, match="feature counts differ"):
        rs.merge(RunningStats().update_stats(np.zeros((2, 4))))


def test_empty_chunk_is_ignored_but_feature_count_is_still_checked():
    rs = RunningStats()
    rs.update_stats(np.empty((0, 3)))
    assert not hasattr(rs, "n_features_in_")                                # an empty first chunk decides nothing
    rs.update_stats(np.zeros((2, 3))).update_stats(np.empty((0, 3)))
    assert rs.count.tolist() == [2, 2, 2]
    with pytest.raises(ValueError, match="fitted with 3 features"):
        rs.update_stats(np.empty((0, 4)))


def test_feature_count_must_stay_constant():
    rs = RunningStats().update_stats(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="X has 4 features, but RunningStats was fitted with 3"):
        rs.update_stats(np.zeros((2, 4)))


def test_results_before_any_data_raise_not_fitted():
    rs = RunningStats()
    for name in ("count", "mean", "variance", "std", "minimum", "maximum"):
        with pytest.raises(NotFittedError):
            getattr(rs, name)
    assert not hasattr(rs, "n_features_in_")


def test_reset_forgets_the_data_and_the_width():
    rs = RunningStats(ddof=1).update_stats(np.zeros((4, 2)))
    assert rs.reset() is rs and rs.ddof == 1
    rs.update_stats(np.ones((3, 5)))                                         # a new width is allowed after reset
    assert rs.count.tolist() == [3] * 5


def test_update_is_an_alias_and_returns_self():
    rs = RunningStats()
    assert rs.update([[1.0]]) is rs and RunningStats.update is RunningStats.update_stats


def test_runningstats_bad_input_raises_informative_errors():
    rs = RunningStats()
    with pytest.raises(ValueError, match="reshape"):
        rs.update_stats([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="infinity"):
        rs.update_stats([[np.inf]])
    with pytest.raises(TypeError, match="numeric"):
        rs.update_stats([["a"]])
    with pytest.raises(ValueError, match="at least one feature"):
        rs.update_stats(np.zeros((3, 0)))
    with pytest.raises(ValueError):
        RunningStats(ddof=-1)
    with pytest.raises(TypeError):
        RunningStats(ddof=1.5)
    assert repr(RunningStats(ddof=1)) == "RunningStats(ddof=1)"


# ---------- StreamingQuantile ----------
def test_quantiles_are_exact_below_capacity_and_match_a1():
    X = _data(300, 3, seed=9)
    sq = _feed(StreamingQuantile(capacity=1000), X, [100, 100, 100])
    np.testing.assert_allclose(sq.quantile(0.3), quantile(X, 0.3, axis=0))
    np.testing.assert_allclose(sq.quantile([0.1, 0.9]), quantile(X, [0.1, 0.9], axis=0))
    assert sq.quantile([0.1, 0.9]).shape == (2, 3)
    np.testing.assert_allclose(sq.median(), np.nanmedian(X, axis=0))


def test_window_mode_keeps_exactly_the_latest_rows():
    sq = StreamingQuantile(capacity=5, mode="window")
    _feed(sq, np.arange(10.0).reshape(-1, 1), [4, 3, 3])
    assert sq.sample_[:, 0].tolist() == [5.0, 6.0, 7.0, 8.0, 9.0]
    assert sq.median()[0] == 7.0 and sq.n_seen_ == 10


def test_reservoir_stays_bounded_and_holds_rows_of_the_stream():
    X = np.arange(1000.0).reshape(-1, 1)
    sq = _feed(StreamingQuantile(capacity=50, random_state=0), X, [100] * 10)
    assert sq.sample_.shape == (50, 1) and sq.n_seen_ == 1000
    assert np.isin(sq.sample_[:, 0], X[:, 0]).all()
    assert len(np.unique(sq.sample_[:, 0])) > 40
    assert sq.sample_.max() > 800                                           # late rows get in: not just the first 50


def test_reservoir_median_is_close_on_a_long_stream():
    X = np.random.default_rng(10).normal(size=(20000, 1))
    sq = _feed(StreamingQuantile(capacity=2000, random_state=1), X, [500] * 40)
    assert abs(sq.median()[0] - np.median(X)) < 0.15
    assert abs(sq.quantile(0.9)[0] - np.quantile(X, 0.9)) < 0.25


def test_reservoir_is_reproducible_with_a_seed():
    X = np.arange(500.0).reshape(-1, 1)
    a = _feed(StreamingQuantile(capacity=20, random_state=3), X, [50] * 10)
    b = _feed(StreamingQuantile(capacity=20, random_state=3), X, [50] * 10)
    np.testing.assert_array_equal(a.sample_, b.sample_)


def test_quantile_ignores_nan_and_gives_nan_for_an_all_nan_feature():
    X = np.array([[1.0, np.nan], [3.0, np.nan], [np.nan, np.nan]])
    out = StreamingQuantile().update_stats(X).median()
    assert out[0] == 2.0 and np.isnan(out[1])


def test_quantile_rejects_bad_q_and_missing_data():
    sq = StreamingQuantile()
    with pytest.raises(NotFittedError):
        sq.quantile(0.5)
    sq.update_stats([[1.0], [2.0]])
    for bad in (-0.1, 1.5, np.nan, [0.2, 2.0]):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            sq.quantile(bad)


def test_quantile_chunk_handling_and_reset():
    sq = StreamingQuantile(capacity=10)
    sq.update_stats(np.empty((0, 2)))
    assert sq.sample_ is None and not hasattr(sq, "n_features_in_")
    sq.update_stats(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="fitted with 2 features"):
        sq.update_stats(np.zeros((3, 5)))
    assert sq.reset() is sq and sq.sample_ is None and sq.n_seen_ == 0


def test_quantile_invalid_parameters_and_input():
    with pytest.raises(ValueError, match="mode"):
        StreamingQuantile(mode="random")
    with pytest.raises(ValueError):
        StreamingQuantile(capacity=0)
    with pytest.raises(TypeError):
        StreamingQuantile(capacity=2.5)
    with pytest.raises(ValueError, match="reshape"):
        StreamingQuantile().update_stats([1.0, 2.0])
    with pytest.raises(ValueError, match="at least one feature"):
        StreamingQuantile().update_stats(np.zeros((2, 0)))


# ---------- StreamingHistogram ----------
def test_chunked_counts_equal_the_a1_histogram():
    x = _data(600, 1, seed=11)[:, 0]
    h = StreamingHistogram(bins=8, range=(-2.0, 8.0))
    for part in np.array_split(x, 7):
        h.update_stats(part)
    expected, edges = histogram(x[(x >= -2.0) & (x <= 8.0)], bins=8, range=(-2.0, 8.0))
    np.testing.assert_array_equal(h.counts_, expected)
    np.testing.assert_allclose(h.edges_, edges)
    assert h.counts_.sum() + h.underflow_ + h.overflow_ == (~np.isnan(x)).sum() == h.n_seen_


def test_range_comes_from_the_first_chunk_and_later_values_over_and_underflow():
    h = StreamingHistogram(bins=4)
    h.update_stats([0.0, 1.0, 2.0, 4.0])
    assert h.edges_.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    h.update_stats([-5.0, 9.0, 10.0, 2.5])
    assert (h.underflow_, h.overflow_) == (1, 2)
    assert h.counts_.sum() == 5 and h.n_seen_ == 8


def test_constant_first_chunk_widens_the_range():
    h = StreamingHistogram(bins=2).update_stats([3.0, 3.0, 3.0])
    assert h.edges_.tolist() == [2.5, 3.0, 3.5]
    assert h.counts_.tolist() == [0, 3]


def test_value_on_the_last_edge_goes_in_the_last_bin():
    h = StreamingHistogram(bins=2, range=(0.0, 2.0)).update_stats([0.0, 1.0, 2.0])
    assert h.counts_.tolist() == [1, 2]


def test_nan_is_ignored_and_infinity_is_counted_outside():
    h = StreamingHistogram(bins=2, range=(0.0, 2.0))
    h.update_stats([0.5, np.nan, np.inf, -np.inf, 1.5])
    assert h.counts_.tolist() == [1, 1] and (h.underflow_, h.overflow_) == (1, 1)
    assert h.n_seen_ == 4


def test_a_chunk_of_only_nan_or_infinity_cannot_set_the_range():
    h = StreamingHistogram(bins=3)
    h.update_stats([np.nan, np.nan]).update_stats([np.inf]).update_stats([])
    assert h.edges_ is None and h.n_seen_ == 0
    with pytest.raises(NotFittedError):
        h.counts_
    h.update_stats([1.0, 2.0, 3.0])
    assert (h.edges_[0], h.edges_[-1]) == (1.0, 3.0)
    assert h.counts_.sum() == 3


def test_sliding_window_counts_only_the_latest_chunks():
    rng = np.random.default_rng(12)
    chunks = [rng.uniform(0.0, 10.0, 50) for _ in range(6)]
    h = StreamingHistogram(bins=5, range=(0.0, 10.0), window=2)
    for part in chunks:
        h.update_stats(part)
    expected, _ = histogram(np.concatenate(chunks[-2:]), bins=5, range=(0.0, 10.0))
    np.testing.assert_array_equal(h.counts_, expected)
    assert h.n_seen_ == 300                                                  # n_seen_ is not windowed


def test_window_also_forgets_under_and_overflow():
    h = StreamingHistogram(bins=2, range=(0.0, 2.0), window=1)
    h.update_stats([-1.0, 5.0, 0.5])
    assert (h.underflow_, h.overflow_) == (1, 1)
    h.update_stats([1.5])
    assert (h.underflow_, h.overflow_) == (0, 0) and h.counts_.tolist() == [0, 1]


def test_density_integrates_to_one_and_is_zero_when_empty():
    h = StreamingHistogram(bins=4, range=(0.0, 4.0))
    assert h.density().tolist() == [0.0] * 4
    h.update_stats(np.random.default_rng(13).uniform(0.0, 4.0, 400))
    assert (h.density() * np.diff(h.edges_)).sum() == pytest.approx(1.0)


def test_multidimensional_input_is_flattened_and_reset_clears():
    h = StreamingHistogram(bins=2, range=(0.0, 2.0)).update_stats(np.array([[0.5, 1.5], [0.6, 1.6]]))
    assert h.counts_.tolist() == [2, 2]
    assert h.reset() is h and h.counts_.tolist() == [0, 0] and h.n_seen_ == 0
    assert StreamingHistogram(bins=2).reset().edges_ is None                 # a data-driven range is forgotten


def test_counts_are_a_copy():
    h = StreamingHistogram(bins=2, range=(0.0, 2.0)).update_stats([0.5])
    h.counts_[0] = 99
    assert h.counts_.tolist() == [1, 0]


def test_histogram_invalid_parameters_and_input():
    for bad_range in ((1.0, 1.0), (2.0, 1.0), (0.0, np.inf), (0.0, 1.0, 2.0)):
        with pytest.raises(ValueError, match="range"):
            StreamingHistogram(range=bad_range)
    with pytest.raises(ValueError):
        StreamingHistogram(bins=0)
    with pytest.raises(TypeError):
        StreamingHistogram(window=1.5)
    with pytest.raises(ValueError):
        StreamingHistogram(window=0)
    with pytest.raises(TypeError, match="numeric"):
        StreamingHistogram().update_stats(["a", "b"])