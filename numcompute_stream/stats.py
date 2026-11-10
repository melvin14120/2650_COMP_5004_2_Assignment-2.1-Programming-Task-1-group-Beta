"""
stats.py
Streaming statistics: moments, quantiles and histograms updated chunk by chunk.

Provides
--------
- RunningStats(ddof=0)
    .update_stats(X_chunk)    merge a chunk into count, mean, variance, min, max
    .merge(other)             combine two accumulators (e.g. from two streams)
    .count / .mean / .variance / .std / .minimum / .maximum / .summary()
- StreamingQuantile(capacity=10000, mode='reservoir'|'window', random_state=None)
    .update_stats(X_chunk)    keep a bounded sample of the stream
    .quantile(q) / .median()
- StreamingHistogram(bins=10, range=None, window=None)
    .update_stats(x_chunk)    count values into fixed bins (optionally a sliding window)
    .counts_ / .edges_ / .density()

Every class also has ``update`` (an alias of ``update_stats``) and ``reset()``.

Design notes
------------
A1's ``StreamingStats`` applies Welford's update one sample at a time. Here a
whole chunk is summarised with array operations (its own count, mean and sum
of squared deviations) and folded into the running totals with the parallel
formula that A1's ``StreamingStats.merge`` already uses::

    delta = mean_b - mean_a
    mean  = mean_a + delta * n_b / n
    M2    = M2_a + M2_b + delta**2 * n_a * n_b / n

so the result equals one pass over all the data but the loop over samples is
gone, and the naive ``E[x^2] - E[x]^2`` (which cancels badly) is never used.

Memory stays bounded: ``RunningStats`` keeps a few numbers per feature,
``StreamingQuantile`` keeps at most ``capacity`` rows and
``StreamingHistogram`` keeps one count per bin (plus one entry per chunk when
a window is used). Quantiles are computed by A1's ``numcompute.stats.quantile``
and bin counts by A1's ``numcompute.stats.histogram``.

Missing values
--------------
NaN is skipped per feature. A feature that has not seen a single number yet
reports ``count == 0`` and NaN for the other statistics.

Axis convention
---------------
Chunks have shape (n_samples, n_features): rows are samples, columns are
features. Results have shape (n_features,). The histogram treats its input as
one stream of numbers and flattens it.
"""

from __future__ import annotations

import warnings
from collections import deque

import numpy as np

from numcompute.preprocessing import NotFittedError, _check_n_features, _check_numeric
from numcompute.stats import histogram as _batch_histogram
from numcompute.stats import quantile as _batch_quantile

from ._checks import _check_int

__all__ = ["RunningStats", "StreamingQuantile", "StreamingHistogram"]


# ----------------------------------------------------------------------
# RunningStats
# ----------------------------------------------------------------------
class RunningStats:
    """Per-feature running count, mean, variance, minimum and maximum.

    Parameters
    ----------
    ddof : int, default 0
        Delta degrees of freedom: the variance divides by ``count - ddof``.
        ``0`` matches A1's ``describe``; ``1`` gives the sample variance.

    Attributes
    ----------
    n_features_in_ : int
        Number of columns, fixed by the first non-empty chunk.
    count, mean, variance, std, minimum, maximum : ndarray, shape (n_features,)
        Read-only results. A feature with no number yet has ``count`` 0 and
        NaN elsewhere; the variance is NaN while ``count <= ddof``.

    Examples
    --------
    >>> rs = RunningStats()
    >>> _ = rs.update_stats([[1.0], [2.0]]); _ = rs.update_stats([[3.0], [4.0]])
    >>> float(rs.mean[0]), float(rs.variance[0])
    (2.5, 1.25)
    """

    def __init__(self, ddof: int = 0):
        self.ddof = _check_int(ddof, "ddof", 0)
        self.reset()

    def reset(self) -> "RunningStats":
        """Forget everything seen so far (``ddof`` is kept)."""
        self.__dict__.pop("n_features_in_", None)
        self._n = self._mean = self._m2 = self._min = self._max = None
        return self

    def __repr__(self) -> str:
        return f"RunningStats(ddof={self.ddof})"

    def _start(self, n_features: int) -> None:
        """Allocate the per-feature accumulators for ``n_features`` columns."""
        if n_features == 0:
            raise ValueError("RunningStats.update_stats requires at least one feature, got 0 columns.")
        self.n_features_in_ = n_features
        self._n = np.zeros(n_features)
        self._mean = np.zeros(n_features)
        self._m2 = np.zeros(n_features)
        self._min = np.full(n_features, np.inf)
        self._max = np.full(n_features, -np.inf)

    def _combine(self, n_b: np.ndarray, mean_b: np.ndarray, m2_b: np.ndarray) -> None:
        """Fold group statistics ``(n_b, mean_b, M2_b)`` into the running ones."""
        n = self._n + n_b
        share = np.divide(n_b, n, out=np.zeros_like(n), where=n > 0)   # n_b / n, 0 for an empty feature
        delta = mean_b - self._mean
        self._mean = self._mean + delta * share
        self._m2 = self._m2 + m2_b + delta ** 2 * self._n * share
        self._n = n

    def update_stats(self, X_chunk) -> "RunningStats":
        """Merge a chunk into the running statistics.

        Parameters
        ----------
        X_chunk : array-like of numbers, shape (n, n_features)
            NaN is skipped per feature. A chunk with ``n == 0`` is ignored.

        Returns
        -------
        self : RunningStats

        Raises
        ------
        TypeError
            If ``X_chunk`` is not numeric.
        ValueError
            If ``X_chunk`` is not 2-D, contains ``±inf``, has zero columns, or
            has a different number of columns than earlier chunks.

        Complexity
        ----------
        Time: O(n * f) for ``n`` rows and ``f`` features, no loop over rows.
        Space: O(n * f) for the temporaries, O(f) kept.
        """
        X = _check_numeric(X_chunk, "RunningStats.update_stats")
        if hasattr(self, "n_features_in_"):
            _check_n_features(self, X, reset=False)
        if X.shape[0] == 0:
            return self
        if not hasattr(self, "n_features_in_"):
            self._start(X.shape[1])

        valid = ~np.isnan(X)
        n_b = valid.sum(axis=0).astype(float)
        total = np.where(valid, X, 0.0).sum(axis=0)
        mean_b = np.divide(total, n_b, out=np.zeros_like(n_b), where=n_b > 0)
        m2_b = np.where(valid, (X - mean_b) ** 2, 0.0).sum(axis=0)
        self._combine(n_b, mean_b, m2_b)
        self._min = np.minimum(self._min, np.where(valid, X, np.inf).min(axis=0))
        self._max = np.maximum(self._max, np.where(valid, X, -np.inf).max(axis=0))
        return self

    update = update_stats

    def merge(self, other: "RunningStats") -> "RunningStats":
        """Fold another accumulator into this one (``other`` is not changed).

        The result equals feeding all of ``other``'s data to ``update_stats``.

        Raises
        ------
        TypeError
            If ``other`` is not a ``RunningStats``.
        ValueError
            If the two accumulators have different numbers of features.

        Complexity
        ----------
        Time and space: O(f).
        """
        if not isinstance(other, RunningStats):
            raise TypeError(f"RunningStats.merge expects a RunningStats, got {type(other).__name__}.")
        if not hasattr(other, "n_features_in_"):
            return self
        if not hasattr(self, "n_features_in_"):
            self._start(other.n_features_in_)
        elif other.n_features_in_ != self.n_features_in_:
            raise ValueError(
                f"RunningStats.merge: feature counts differ ({self.n_features_in_} vs {other.n_features_in_})."
            )
        self._combine(other._n, other._mean, other._m2)
        self._min = np.minimum(self._min, other._min)
        self._max = np.maximum(self._max, other._max)
        return self

    # -- results ---------------------------------------------------------------
    def _need_data(self, what: str) -> None:
        if not hasattr(self, "n_features_in_"):
            raise NotFittedError(f"RunningStats.{what}: no data has been added yet; call update_stats first.")

    @property
    def count(self) -> np.ndarray:
        """Number of non-NaN values seen per feature, shape (n_features,)."""
        self._need_data("count")
        return self._n.astype(int)

    @property
    def mean(self) -> np.ndarray:
        """Mean per feature (NaN where ``count == 0``)."""
        self._need_data("mean")
        return np.where(self._n > 0, self._mean, np.nan)

    @property
    def variance(self) -> np.ndarray:
        """Variance per feature with divisor ``count - ddof`` (NaN if that is <= 0)."""
        self._need_data("variance")
        d = self._n - self.ddof
        return np.divide(self._m2, d, out=np.full(d.shape, np.nan), where=d > 0)

    @property
    def std(self) -> np.ndarray:
        """Standard deviation per feature: the square root of ``variance``."""
        return np.sqrt(self.variance)

    @property
    def minimum(self) -> np.ndarray:
        """Smallest value per feature (NaN where ``count == 0``)."""
        self._need_data("minimum")
        return np.where(self._n > 0, self._min, np.nan)

    @property
    def maximum(self) -> np.ndarray:
        """Largest value per feature (NaN where ``count == 0``)."""
        self._need_data("maximum")
        return np.where(self._n > 0, self._max, np.nan)

    def summary(self) -> dict:
        """All results as a dict with keys 'count', 'mean', 'variance', 'std', 'min', 'max'."""
        return {"count": self.count, "mean": self.mean, "variance": self.variance,
                "std": self.std, "min": self.minimum, "max": self.maximum}


# ----------------------------------------------------------------------
# StreamingQuantile
# ----------------------------------------------------------------------
class StreamingQuantile:
    """Approximate per-feature quantiles from a bounded sample of the stream.

    Parameters
    ----------
    capacity : int, default 10000
        Maximum number of rows kept. While fewer rows have been seen the
        quantiles are exact.
    mode : {'reservoir', 'window'}, default 'reservoir'
        ``'reservoir'`` keeps a uniform random sample of ALL rows seen
        (Vitter's Algorithm R, vectorised per chunk). ``'window'`` keeps the
        latest ``capacity`` rows (a sliding window, so it follows drift).
    random_state : int or None, default None
        Seed for the reservoir.

    Attributes
    ----------
    sample_ : ndarray, shape (<= capacity, n_features) or None
        The rows currently kept (NaN is kept and ignored when querying).
    n_seen_ : int
        Total rows passed to ``update_stats``.
    n_features_in_ : int
        Number of columns, fixed by the first non-empty chunk.

    Examples
    --------
    >>> import numpy as np
    >>> sq = StreamingQuantile(capacity=100)
    >>> _ = sq.update_stats(np.arange(11.0).reshape(-1, 1)); float(sq.median()[0])
    5.0
    """

    def __init__(self, capacity: int = 10000, mode: str = "reservoir", random_state=None):
        self.capacity = _check_int(capacity, "capacity", 1)
        if mode not in ("reservoir", "window"):
            raise ValueError(f"StreamingQuantile: mode must be 'reservoir' or 'window', got {mode!r}.")
        self.mode = mode
        self.random_state = random_state
        self.reset()

    def reset(self) -> "StreamingQuantile":
        """Forget the sample and restart the random generator."""
        self.__dict__.pop("n_features_in_", None)
        self.sample_ = None
        self.n_seen_ = 0
        self._rng = np.random.default_rng(self.random_state)
        return self

    def update_stats(self, X_chunk) -> "StreamingQuantile":
        """Add a chunk to the sample.

        Parameters
        ----------
        X_chunk : array-like of numbers, shape (n, n_features)
            A chunk with ``n == 0`` is ignored.

        Returns
        -------
        self : StreamingQuantile

        Raises
        ------
        TypeError, ValueError
            As for ``RunningStats.update_stats``.

        Complexity
        ----------
        Time: O(n * f) (plus O(capacity * f) to rebuild the window), no loop
        over rows. Space: O(capacity * f).
        """
        X = _check_numeric(X_chunk, "StreamingQuantile.update_stats")
        if hasattr(self, "n_features_in_"):
            _check_n_features(self, X, reset=False)
        n = X.shape[0]
        if n == 0:
            return self
        if self.sample_ is None:
            if X.shape[1] == 0:
                raise ValueError("StreamingQuantile.update_stats requires at least one feature, got 0 columns.")
            self.n_features_in_ = X.shape[1]
            self.sample_ = np.empty((0, X.shape[1]))

        if self.mode == "window":
            combined = np.vstack([self.sample_, X])
            self.sample_ = combined[-self.capacity:].copy() if combined.shape[0] > self.capacity else combined
        else:
            take = min(self.capacity - self.sample_.shape[0], n)   # rows that still fit
            if take > 0:
                self.sample_ = np.vstack([self.sample_, X[:take]])
            rest = X[take:]
            if rest.shape[0] > 0:
                position = self.n_seen_ + take + np.arange(1, rest.shape[0] + 1)   # 1-based place in the stream
                slot = self._rng.integers(0, position)                              # uniform in [0, position)
                keep = slot < self.capacity                                         # replace with prob capacity / position
                self.sample_[slot[keep]] = rest[keep]                               # repeated slot: the last row wins
        self.n_seen_ += n
        return self

    update = update_stats

    def quantile(self, q) -> np.ndarray:
        """Quantile(s) ``q`` of the kept sample, per feature.

        Parameters
        ----------
        q : float or array-like of float
            Each in [0, 1].

        Returns
        -------
        ndarray, shape (n_features,) for a scalar ``q``, else (len(q), n_features)
            NaN for a feature that has no number in the sample.

        Raises
        ------
        NotFittedError
            If no rows have been added.
        ValueError
            If any ``q`` is NaN or outside [0, 1].

        Complexity
        ----------
        Time: O(capacity * f * log capacity) (A1's ``quantile``). Space: O(capacity * f).
        """
        if self.sample_ is None:
            raise NotFittedError("StreamingQuantile.quantile: no data has been added yet; call update_stats first.")
        qa = np.asarray(q, dtype=float)
        if np.isnan(qa).any() or (qa < 0).any() or (qa > 1).any():
            raise ValueError(f"StreamingQuantile.quantile: q must lie in [0, 1], got {q!r}.")
        with warnings.catch_warnings():                      # an all-NaN column gives NaN, not a warning
            warnings.simplefilter("ignore", RuntimeWarning)
            return _batch_quantile(self.sample_, qa, axis=0)

    def median(self) -> np.ndarray:
        """Per-feature median, i.e. ``quantile(0.5)``."""
        return self.quantile(0.5)


# ----------------------------------------------------------------------
# StreamingHistogram
# ----------------------------------------------------------------------
class StreamingHistogram:
    """Fixed-bin histogram of a stream of numbers, optionally over a sliding window.

    Parameters
    ----------
    bins : int, default 10
        Number of equal-width bins.
    range : (float, float) or None, default None
        Bin range ``(low, high)``. If ``None`` it is taken from the first chunk
        that holds a finite number (widened by 0.5 each side if that chunk is
        constant). Values outside the range are not lost: they are counted in
        ``underflow_`` and ``overflow_`` (``±inf`` included).
    window : int or None, default None
        If set, only the latest ``window`` chunks count; older ones are removed.

    Attributes
    ----------
    edges_ : ndarray, shape (bins + 1,) or None
        Bin edges (``None`` until the range is known).
    counts_ : ndarray, shape (bins,)
        Values per bin over the whole stream, or over the window.
    underflow_, overflow_ : int
        Values below ``edges_[0]`` and above ``edges_[-1]``.
    n_seen_ : int
        Total non-NaN values ever added (not reduced by the window).

    Notes
    -----
    NaN is ignored. A chunk with no number in it does not count as a chunk
    for the window.

    Examples
    --------
    >>> h = StreamingHistogram(bins=2, range=(0, 2))
    >>> h.update_stats([0.5, 0.7, 1.5]).counts_.tolist()
    [2, 1]
    """

    def __init__(self, bins: int = 10, range=None, window=None):
        self.bins = _check_int(bins, "bins", 1)
        if range is not None:
            if len(range) != 2:
                raise ValueError(f"StreamingHistogram: range must be (low, high), got {range!r}.")
            lo, hi = float(range[0]), float(range[1])
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError(f"StreamingHistogram: range must be finite with low < high, got {range!r}.")
            range = (lo, hi)
        self.range = range
        self.window = None if window is None else _check_int(window, "window", 1)
        self.reset()

    def reset(self) -> "StreamingHistogram":
        """Forget all counts (and the range, if it was taken from the data)."""
        self.edges_ = None if self.range is None else np.linspace(*self.range, self.bins + 1)
        self._chunks: deque = deque()          # per-chunk (counts, under, over), only used with a window
        self._counts = np.zeros(self.bins, dtype=np.int64)
        self._under = 0
        self._over = 0
        self.n_seen_ = 0
        return self

    def update_stats(self, x_chunk) -> "StreamingHistogram":
        """Count a chunk of numbers into the bins.

        Parameters
        ----------
        x_chunk : array-like of numbers, any shape
            Flattened first. NaN is ignored.

        Returns
        -------
        self : StreamingHistogram

        Raises
        ------
        TypeError
            If ``x_chunk`` is not numeric.

        Complexity
        ----------
        Time: O(n + bins) per chunk. Space: O(bins) (plus O(window * bins) with a window).
        """
        try:
            x = np.asarray(x_chunk, dtype=float).ravel()
        except (TypeError, ValueError) as exc:
            raise TypeError(f"StreamingHistogram.update_stats requires numeric input ({exc}).") from None
        x = x[~np.isnan(x)]
        if x.size == 0:
            return self
        if self.edges_ is None:
            finite = x[np.isfinite(x)]
            if finite.size == 0:                 # only +-inf so far: cannot choose a range yet
                return self
            lo, hi = float(finite.min()), float(finite.max())
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            self.edges_ = np.linspace(lo, hi, self.bins + 1)

        lo, hi = self.edges_[0], self.edges_[-1]
        counts, _ = _batch_histogram(x[(x >= lo) & (x <= hi)], bins=self.edges_)
        under, over = int((x < lo).sum()), int((x > hi).sum())

        if self.window is not None:
            if len(self._chunks) == self.window:                 # drop the oldest chunk's share
                old_counts, old_under, old_over = self._chunks.popleft()
                self._counts -= old_counts
                self._under -= old_under
                self._over -= old_over
            self._chunks.append((counts, under, over))
        self._counts += counts
        self._under += under
        self._over += over
        self.n_seen_ += x.size
        return self

    update = update_stats

    @property
    def counts_(self) -> np.ndarray:
        """Number of values in each bin, shape (bins,)."""
        if self.edges_ is None:
            raise NotFittedError("StreamingHistogram.counts_: no data has been added yet; call update_stats first.")
        return self._counts.copy()

    @property
    def underflow_(self) -> int:
        """Values below the first edge."""
        return int(self._under)

    @property
    def overflow_(self) -> int:
        """Values above the last edge."""
        return int(self._over)

    def density(self) -> np.ndarray:
        """Counts scaled so that the histogram integrates to 1 over the range.

        Returns
        -------
        ndarray, shape (bins,)
            All zeros when no value lies inside the range.
        """
        counts = self.counts_
        total = counts.sum()
        if total == 0:
            return np.zeros(self.bins)
        return counts / (total * np.diff(self.edges_))