"""
preprocessing.py
Streaming versions of the A1 transformers: StandardScaler, SimpleImputer, OneHotEncoder.

Provides
--------
- StandardScaler(with_mean=True, with_std=True)
    .partial_fit(X_chunk)   update the running mean / variance with a chunk
- SimpleImputer(strategy='mean'|'median'|'most_frequent'|'constant', fill_value=0,
                capacity=10000, random_state=None)    (also available as ``Imputer``)
    .partial_fit(X_chunk)   update the fill values with a chunk
- OneHotEncoder(handle_unknown='error'|'ignore', dtype=float64)
    .partial_fit(X_chunk)   add the categories of a chunk (the output can grow)

Design notes
------------
Each class SUBCLASSES the matching A1 class, so ``transform``,
``fit_transform``, ``get_feature_names_out`` and the input checks are the A1
code, unchanged. Only the learning side is new: ``partial_fit`` keeps running
state between chunks, and ``fit`` is "forget everything, then learn from this
one array" (so ``fit`` followed by ``partial_fit`` continues from the data
given to ``fit``, as in scikit-learn).

The running state comes from ``numcompute_stream.stats``:

- StandardScaler: ``RunningStats`` (Welford / Chan merge), exact.
- SimpleImputer 'mean': ``RunningStats``, exact.
- SimpleImputer 'median': ``StreamingQuantile`` (reservoir). Exact while at
  most ``capacity`` rows have been seen, an estimate afterwards.
- SimpleImputer 'most_frequent': per column, the sorted distinct values and
  their counts, merged with ``np.unique``. Exact, but memory grows with the
  number of DISTINCT values, so use it for coded or categorical columns.
- OneHotEncoder: the union of the categories seen so far.

Missing values and edge cases
-----------------------------
- NaN is skipped when learning (scaler, imputer). A column with no number yet
  has NaN statistics in the scaler and ``fill_value`` in the imputer;
  ``n_samples_seen_`` shows how many numbers each column has really seen.
- A constant column keeps ``scale_ == 0`` and transforms to exactly 0 (the A1
  rule); a constant first chunk followed by varying data is handled correctly.
- An empty chunk is ignored. The number of columns is fixed by the first
  non-empty chunk.
- Values above about 1e154 can overflow the variance (A1's ``fit`` has an
  extra rescaling guard that a streaming update cannot share).

Axis convention
---------------
``X`` has shape (n_samples, n_features): rows are samples, columns are
features. The statistics have shape (n_features,).
"""

from __future__ import annotations

import warnings

import numpy as np

from numcompute import preprocessing as _a1
from numcompute.preprocessing import (
    _HANDLE_UNKNOWN,
    _IMPUTE_STRATEGIES,
    _check_2d,
    _check_n_features,
    _check_numeric,
    _kind_group,
)

from ._checks import _check_int
from .stats import RunningStats, StreamingQuantile

__all__ = ["StandardScaler", "SimpleImputer", "Imputer", "OneHotEncoder"]


# ----------------------------------------------------------------------
# StandardScaler
# ----------------------------------------------------------------------
class StandardScaler(_a1.StandardScaler):
    """Standardise features to zero mean and unit variance, learning chunk by chunk.

    Same parameters, attributes and ``transform`` as the A1 ``StandardScaler``.
    ``partial_fit`` folds each chunk into running statistics, so ``mean_``,
    ``var_`` and ``scale_`` always describe all the data seen so far.

    Attributes
    ----------
    mean_, var_, scale_ : ndarray, shape (n_features,)
        As in A1 (population variance; ``scale_ = sqrt(var_)``, 0 for a
        constant column).
    n_samples_seen_ : ndarray of int, shape (n_features,)
        Non-NaN values seen per column.
    n_features_in_ : int
        Number of columns, fixed by the first non-empty chunk.

    Examples
    --------
    >>> import numpy as np
    >>> sc = StandardScaler()
    >>> _ = sc.partial_fit([[1.0], [2.0]]); _ = sc.partial_fit([[3.0], [4.0]])
    >>> float(sc.mean_[0]), round(float(sc.var_[0]), 6)
    (2.5, 1.25)
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        super().__init__(with_mean, with_std)
        self._stats = RunningStats()

    def reset(self) -> "StandardScaler":
        """Forget everything learned so far (the parameters are kept)."""
        for name in ("mean_", "var_", "scale_", "n_features_in_", "n_samples_seen_"):
            self.__dict__.pop(name, None)
        self._stats.reset()
        return self

    def fit(self, X, y=None) -> "StandardScaler":
        """Forget earlier data and learn from ``X`` alone.

        Same checks and results as the A1 ``fit`` (up to rounding), including
        its errors for empty input.

        Raises
        ------
        TypeError, ValueError
            As for the A1 ``StandardScaler.fit``.

        Complexity
        ----------
        O(n * m) time, O(m) stored state.
        """
        self.reset()
        X = _check_numeric(X, "StandardScaler")
        _check_n_features(self, X, reset=True)
        return self._absorb(X)

    def partial_fit(self, X, y=None) -> "StandardScaler":
        """Update the statistics with one chunk.

        Parameters
        ----------
        X : array-like of numbers, shape (n, n_features)
            NaN marks missing values. A chunk with ``n == 0`` is ignored.
        y : ignored

        Returns
        -------
        self : StandardScaler

        Raises
        ------
        TypeError
            If ``X`` is not numeric.
        ValueError
            If ``X`` is not 2-D, contains ``±inf`` or has a different number
            of columns than earlier chunks.

        Complexity
        ----------
        Time: O(n * m) per chunk. Space: O(m) kept.
        """
        X = _check_numeric(X, "StandardScaler.partial_fit")
        if hasattr(self, "n_features_in_"):
            _check_n_features(self, X, reset=False)
        if X.shape[0] == 0:
            return self
        if not hasattr(self, "n_features_in_"):
            _check_n_features(self, X, reset=True)
        return self._absorb(X)

    def _absorb(self, X: np.ndarray) -> "StandardScaler":
        """Merge ``X`` into the running statistics and refresh the public attributes."""
        s = self._stats.update_stats(X)
        constant = s.minimum == s.maximum                  # NaN == NaN is False, so empty columns stay NaN
        mean = np.where(constant, s.minimum, s.mean)       # exact value, so it transforms to exactly 0
        scale = np.where(constant, 0.0, s.std)
        n = X.shape[1]
        with np.errstate(over="ignore"):
            self.var_ = scale ** 2
        self.mean_ = mean if self.with_mean else np.zeros(n)
        self.scale_ = scale if self.with_std else np.ones(n)
        self.n_samples_seen_ = s.count
        return self


# ----------------------------------------------------------------------
# SimpleImputer
# ----------------------------------------------------------------------
def _merge_counts(values_a: np.ndarray, counts_a: np.ndarray, values_b: np.ndarray, counts_b: np.ndarray):
    """Merge two (sorted distinct values, counts) tables into one.

    Complexity: O((a + b) log (a + b)) time and space for tables of size ``a`` and ``b``.
    """
    values = np.concatenate([values_a, values_b])
    counts = np.concatenate([counts_a, counts_b])
    unique, inverse = np.unique(values, return_inverse=True)
    return unique, np.bincount(inverse, weights=counts).astype(np.int64)


class SimpleImputer(_a1.SimpleImputer):
    """Replace NaN with a per-column statistic that is updated chunk by chunk.

    Same ``strategy``, ``fill_value`` and ``transform`` as the A1
    ``SimpleImputer``; ``partial_fit`` keeps ``statistics_`` current.

    Parameters
    ----------
    strategy : {'mean', 'median', 'most_frequent', 'constant'}, default 'mean'
        See the module notes for how each one is kept up to date.
    fill_value : float, default 0
        Value for 'constant' and for columns that have seen no number yet.
    capacity : int, default 10000
        'median' only: rows kept by the reservoir. The median is exact up to
        this many rows.
    random_state : int or None, default None
        'median' only: seed of the reservoir.

    Attributes
    ----------
    statistics_ : ndarray, shape (n_features,)
        Fill value of each column right now.
    n_samples_seen_ : ndarray of int, shape (n_features,)
        Non-NaN values seen per column (0 means ``fill_value`` is used).
    n_features_in_ : int
        Number of columns, fixed by the first non-empty chunk.

    Notes
    -----
    ``fit`` warns (as in A1) when a column is entirely NaN. ``partial_fit``
    does not: an all-NaN column is normal in an early chunk, and
    ``n_samples_seen_`` tells you which columns are still waiting for data.

    Examples
    --------
    >>> import numpy as np
    >>> imp = SimpleImputer()
    >>> _ = imp.partial_fit([[1.0], [np.nan]]); _ = imp.partial_fit([[3.0]])
    >>> imp.transform([[np.nan]]).tolist()
    [[2.0]]
    """

    def __init__(self, strategy: str = "mean", fill_value: float = 0, capacity: int = 10000, random_state=None):
        super().__init__(strategy, fill_value)
        self.capacity = _check_int(capacity, "capacity", 1)
        self.random_state = random_state

    def reset(self) -> "SimpleImputer":
        """Forget everything learned so far (the parameters are kept)."""
        for name in ("statistics_", "n_features_in_", "n_samples_seen_"):
            self.__dict__.pop(name, None)
        self._count = self._stats = self._quantiles = self._values = self._counts = None
        return self

    def _fill(self) -> float:
        """Validate ``strategy`` and ``fill_value`` (same errors as the A1 ``fit``); return the fill value."""
        if self.strategy not in _IMPUTE_STRATEGIES:
            raise ValueError(f"strategy must be one of {_IMPUTE_STRATEGIES}, got {self.strategy!r}")
        try:
            fill = float(self.fill_value)
        except (TypeError, ValueError):
            raise ValueError(f"fill_value must be a number, got {self.fill_value!r}") from None
        if not np.isfinite(fill):
            raise ValueError(f"fill_value must be finite, got {self.fill_value!r}")
        return fill

    def _start(self, n_features: int) -> None:
        """Create the running state that the chosen strategy needs."""
        self._count = np.zeros(n_features)
        if self.strategy == "mean":
            self._stats = RunningStats()
        elif self.strategy == "median":
            self._quantiles = StreamingQuantile(self.capacity, "reservoir", self.random_state)
        elif self.strategy == "most_frequent":
            self._values = [np.empty(0) for _ in range(n_features)]
            self._counts = [np.empty(0, dtype=np.int64) for _ in range(n_features)]

    def fit(self, X, y=None) -> "SimpleImputer":
        """Forget earlier data and learn from ``X`` alone.

        Same checks, results and ``UserWarning`` for all-NaN columns as the A1 ``fit``.

        Raises
        ------
        TypeError, ValueError
            As for the A1 ``SimpleImputer.fit``.

        Complexity
        ----------
        As ``partial_fit`` on one chunk.
        """
        fill = self._fill()
        self.reset()
        X = _check_numeric(X, "SimpleImputer")
        _check_n_features(self, X, reset=True)
        self._start(X.shape[1])
        self._absorb(X, fill)
        empty = np.flatnonzero(self.n_samples_seen_ == 0)
        if empty.size and self.strategy != "constant":
            warnings.warn(
                f"SimpleImputer: columns {empty.tolist()} are entirely NaN; "
                f"filling them with fill_value={fill}",
                UserWarning,
                stacklevel=2,
            )
        return self

    def partial_fit(self, X, y=None) -> "SimpleImputer":
        """Update the fill values with one chunk.

        Parameters
        ----------
        X : array-like of numbers, shape (n, n_features)
            NaN marks missing values. A chunk with ``n == 0`` is ignored.
        y : ignored

        Returns
        -------
        self : SimpleImputer

        Raises
        ------
        TypeError
            If ``X`` is not numeric.
        ValueError
            If ``strategy`` or ``fill_value`` is invalid, or ``X`` is not 2-D,
            contains ``±inf`` or has a different number of columns than
            earlier chunks.

        Complexity
        ----------
        'mean': O(n * m). 'median': O(n * m + capacity * m log capacity).
        'most_frequent': O((d + n) log (d + n)) per column for ``d`` distinct
        values kept. 'constant': O(n * m).
        """
        fill = self._fill()
        X = _check_numeric(X, "SimpleImputer.partial_fit")
        if hasattr(self, "n_features_in_"):
            _check_n_features(self, X, reset=False)
        if X.shape[0] == 0:
            return self
        if not hasattr(self, "n_features_in_"):
            _check_n_features(self, X, reset=True)
            self._start(X.shape[1])
        return self._absorb(X, fill)

    def _absorb(self, X: np.ndarray, fill: float) -> "SimpleImputer":
        """Merge ``X`` into the running state and refresh ``statistics_``."""
        valid = ~np.isnan(X)
        self._count += valid.sum(axis=0)
        n_features = X.shape[1]
        if self.strategy == "constant":
            stats = np.full(n_features, fill)
        elif self.strategy == "mean":
            stats = self._stats.update_stats(X).mean
        elif self.strategy == "median":
            stats = self._quantiles.update_stats(X).median()
        else:
            for j in range(n_features):                    # a loop over columns, not rows
                values, counts = np.unique(X[valid[:, j], j], return_counts=True)
                self._values[j], self._counts[j] = _merge_counts(self._values[j], self._counts[j], values, counts)
            # argmax returns the first maximum and the values are sorted, so ties go to the smallest value
            stats = np.array([v[np.argmax(c)] if c.size else np.nan for v, c in zip(self._values, self._counts)])
        self.statistics_ = np.where(np.isnan(stats), fill, stats)
        self.n_samples_seen_ = self._count.astype(int)
        return self


Imputer = SimpleImputer                                    # the name used in the assignment spec


# ----------------------------------------------------------------------
# OneHotEncoder
# ----------------------------------------------------------------------
class OneHotEncoder(_a1.OneHotEncoder):
    """One-hot encoder whose categories grow as new chunks arrive.

    Same parameters, ``transform`` and ``get_feature_names_out`` as the A1
    ``OneHotEncoder``. ``partial_fit`` adds the categories of each chunk to
    ``categories_`` (sorted, as in A1), so a category first seen in a later
    chunk is no longer "unknown".

    Attributes
    ----------
    categories_ : list of ndarray
        Sorted distinct values of each column seen so far.
    n_features_in_ : int
        Number of input columns, fixed by the first non-empty chunk.
    n_features_out_ : int
        Number of output columns right now (the sum of the category counts).

    Notes
    -----
    When a chunk brings a new category the OUTPUT WIDTH GROWS, and the
    columns after the new one move. A model fitted on the old width cannot
    take the new output, so ``partial_fit`` emits a ``UserWarning`` saying
    which columns grew. Where the width must stay fixed, show the encoder all
    the categories first (or use ``handle_unknown='ignore'`` and stop calling
    ``partial_fit`` on it).

    Examples
    --------
    >>> import numpy as np
    >>> enc = OneHotEncoder()
    >>> _ = enc.partial_fit(np.array([["a"], ["b"]]))
    >>> enc.transform(np.array([["b"]])).tolist()
    [[0.0, 1.0]]
    """

    @property
    def n_features_out_(self) -> int:
        """Number of output columns (the sum of the category counts)."""
        _a1._check_fitted(self, "categories_")
        return int(sum(c.size for c in self.categories_))

    def partial_fit(self, X, y=None) -> "OneHotEncoder":
        """Add the categories found in one chunk.

        Parameters
        ----------
        X : array-like, shape (n, n_features)
            Strings, numbers or objects. A chunk with ``n == 0`` is ignored.
        y : ignored

        Returns
        -------
        self : OneHotEncoder

        Raises
        ------
        TypeError
            If ``X`` holds strings but earlier chunks held numbers (or vice
            versa), or a column mixes incomparable types.
        ValueError
            If ``handle_unknown`` is invalid, ``X`` is not 2-D, or it has a
            different number of columns than earlier chunks.

        Warns
        -----
        UserWarning
            If the chunk adds a category after the first chunk (the output grows).

        Complexity
        ----------
        Time: O((k + n) log (k + n)) per column for ``k`` known categories.
        Space: O(sum of k).
        """
        if self.handle_unknown not in _HANDLE_UNKNOWN:
            raise ValueError(
                f"handle_unknown must be one of {_HANDLE_UNKNOWN}, got {self.handle_unknown!r}"
            )
        X = _check_2d(X, "OneHotEncoder.partial_fit")
        kind = _kind_group(X.dtype)
        fitted = hasattr(self, "categories_")
        if fitted:
            if "object" not in (kind, self._kind) and kind != self._kind:
                raise TypeError(
                    f"OneHotEncoder was fitted on {self._kind} data but partial_fit got "
                    f"{kind} data (dtype {X.dtype})"
                )
            _check_n_features(self, X, reset=False)
        if X.shape[0] == 0:
            return self
        if not fitted:
            return self.fit(X)                             # the first chunk: A1 fit sets everything up

        try:
            merged = [np.unique(np.concatenate([cats, X[:, j]])) for j, cats in enumerate(self.categories_)]
        except TypeError as exc:
            raise TypeError(f"OneHotEncoder: a column mixes incomparable types ({exc})") from None
        grown = [j for j, (new, old) in enumerate(zip(merged, self.categories_)) if new.size != old.size]
        width_before = self.n_features_out_
        self.categories_ = merged
        if grown:
            warnings.warn(
                f"OneHotEncoder.partial_fit: new categories in column(s) {grown}; the output grew from "
                f"{width_before} to {self.n_features_out_} columns. A model fitted on the old width must be retrained.",
                UserWarning,
                stacklevel=2,
            )
        return self