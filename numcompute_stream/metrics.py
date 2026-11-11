"""
metrics.py
Streaming metrics: feed predictions chunk by chunk, read the score at any time.

Provides
--------
- StreamingConfusionMatrix(labels=None, window=None)
- StreamingAccuracy(window=None)
- StreamingPrecision / StreamingRecall / StreamingF1(average='binary', pos_label=1,
                                                      labels=None, window=None)
- StreamingMSE(window=None)
- StreamingAUC(bins=1000, score_range=(0.0, 1.0), window=None)

Every class has the same three methods:
    .update(y_true_chunk, y_pred_chunk)   add one chunk (returns self)
    .result()                             the score over everything counted so far
    .reset()                              forget everything (returns self)
and two counters: ``n_seen_`` (all samples ever given) and ``n_samples_``
(samples the score is currently based on).

Design notes
------------
CUMULATIVE or ROLLING. With ``window=None`` the score covers every sample seen
and only small counts are stored (a confusion matrix, two sums, two histograms).
With ``window=w`` the score covers only the latest ``w`` SAMPLES (not chunks), so
the result does not depend on how the stream is cut into chunks. The window
keeps the last ``w`` samples in memory and recomputes the score from them.

The scores reuse the A1 code: a chunk's confusion matrix comes from
``numcompute.metrics.confusion_matrix``, precision and recall use A1's
averaging, and the AUC is A1's ``auc`` applied to a ROC curve built from
counts. A cumulative score therefore equals the A1 function called on all the
data at once.

NOTHING-TO-MEASURE-YET. ``result()`` returns NaN (it does not raise) while a
score is undefined: no samples yet, or, for AUC, only one class so far; for
binary precision / recall / F1, until ``pos_label`` has appeared. Matplotlib
skips NaN, so early chunks simply leave a gap in a plot. Pass ``labels=[0, 1]``
to get 0.0 instead of NaN for a positive class that has not appeared yet.

Missing values and edge cases
-----------------------------
- NaN in the labels or scores is rejected with ``ValueError`` (as in A1), and a
  rejected chunk leaves the state unchanged.
- An empty chunk is ignored.
- Without ``labels`` the label set grows as new classes appear. Labels of
  numbers and labels of text cannot be mixed (``TypeError``).
- AUC from counts is approximate: scores are put into ``bins`` equal bins and
  a positive and a negative in the same bin count as a tie (half a point). It is
  exact when each bin holds one distinct score, and the error shrinks as
  ``bins`` grows.

Axis convention
---------------
All inputs are 1-D arrays of length n_samples (one entry per sample).
"""

from __future__ import annotations

import numpy as np

from numcompute.metrics import _VALID_AVERAGES, _average_ratio, _resolve_binary_pos_index, auc, confusion_matrix
from numcompute.utils import safe_divide

from ._checks import _check_int

__all__ = [
    "StreamingConfusionMatrix",
    "StreamingAccuracy",
    "StreamingPrecision",
    "StreamingRecall",
    "StreamingF1",
    "StreamingMSE",
    "StreamingAUC",
]


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _check_pair(first, second, where: str, second_name: str):
    """Return two 1-D arrays of the same length, or raise ``ValueError``."""
    first, second = np.asarray(first), np.asarray(second)
    if first.ndim != 1 or second.ndim != 1:
        raise ValueError(
            f"{where}: y_true and {second_name} must be 1-D, got shapes {first.shape} and {second.shape}"
        )
    if first.shape != second.shape:
        raise ValueError(
            f"{where}: y_true has shape {first.shape} but {second_name} has shape {second.shape}"
        )
    return first, second


def _as_float(values, where: str, name: str) -> np.ndarray:
    """Convert to a float array or raise ``TypeError``; reject NaN with ``ValueError``."""
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise TypeError(f"{where}: {name} must be numeric") from None
    if np.isnan(values).any():
        raise ValueError(f"{where}: {name} must not contain NaN")
    return values


def _keep_latest(buffer, new: np.ndarray, window: int) -> np.ndarray:
    """Return the last ``window`` entries of ``buffer`` followed by ``new`` (a copy).

    Complexity: O(window + len(new)) time and space.
    """
    joined = new if buffer is None else np.concatenate([buffer, new])
    return joined[-window:].copy()


def _label_group(labels: np.ndarray) -> str:
    """'number', 'text' or 'object': which labels can be sorted together."""
    if labels.dtype.kind in "biuf":
        return "number"
    return "text" if labels.dtype.kind in "US" else "object"


def _embed(matrix: np.ndarray, labels: np.ndarray, new_labels: np.ndarray) -> np.ndarray:
    """Place ``matrix`` (indexed by ``labels``) inside a bigger matrix indexed by ``new_labels``."""
    at = np.searchsorted(new_labels, labels)
    out = np.zeros((new_labels.size, new_labels.size), dtype=np.int64)
    out[np.ix_(at, at)] = matrix
    return out


class _StreamingMetric:
    """Shared parts: the ``window`` option, the counters, ``reset`` and ``repr``."""

    _params: tuple = ()

    def __init__(self, window=None):
        self.window = None if window is None else _check_int(window, "window", 1)
        self._n_seen = 0

    def reset(self):
        """Forget everything seen so far (the parameters are kept). Returns ``self``."""
        self._n_seen = 0
        return self

    @property
    def n_seen_(self) -> int:
        """Number of samples given to ``update`` since the last ``reset``."""
        return self._n_seen

    @property
    def n_samples_(self) -> int:
        """Number of samples the score is based on (``n_seen_``, or at most ``window``)."""
        return self.n_seen_ if self.window is None else min(self.n_seen_, self.window)

    def __repr__(self) -> str:
        args = ", ".join(f"{name}={getattr(self, name)!r}" for name in self._params)
        return f"{type(self).__name__}({args})"


# ----------------------------------------------------------------------
# Confusion matrix
# ----------------------------------------------------------------------
class StreamingConfusionMatrix(_StreamingMetric):
    """Confusion matrix that accumulates over chunks (or over the latest ``window`` samples).

    Parameters
    ----------
    labels : array-like or None, default None
        Fixed labels that index the matrix. ``None`` means "the labels seen so
        far": the matrix grows when a new class appears.
    window : int or None, default None
        ``None`` counts every sample; ``w`` counts only the latest ``w``
        samples (the labels are then the ones inside the window).

    Examples
    --------
    >>> cm = StreamingConfusionMatrix()
    >>> _ = cm.update([0, 1], [0, 0]); _ = cm.update([1, 1], [1, 1])
    >>> matrix, labels = cm.result()
    >>> matrix.tolist(), labels.tolist()
    ([[1, 0], [1, 2]], [0, 1])
    """

    _params = ("labels", "window")

    def __init__(self, labels=None, window=None):
        super().__init__(window)
        if labels is None:
            self.labels = None
        else:
            self.labels = np.unique(np.asarray(labels))
            if self.labels.ndim != 1 or self.labels.size == 0:
                raise ValueError("StreamingConfusionMatrix: labels must be a non-empty 1-D array")
        self.reset()

    def reset(self) -> "StreamingConfusionMatrix":
        """Forget everything seen so far (the parameters are kept). Returns ``self``."""
        super().reset()
        self._labels = np.empty(0) if self.labels is None else self.labels
        self._matrix = np.zeros((self._labels.size, self._labels.size), dtype=np.int64)
        self._true = self._pred = None
        return self

    def update(self, y_true, y_pred) -> "StreamingConfusionMatrix":
        """Add one chunk of true and predicted labels.

        Parameters
        ----------
        y_true, y_pred : array-like, shape (n,)
            Labels of the same length. An empty chunk is ignored.

        Returns
        -------
        self : StreamingConfusionMatrix

        Raises
        ------
        ValueError
            If the inputs are not 1-D arrays of the same length, contain NaN,
            or contain a label that is not in the fixed ``labels``.
        TypeError
            If the labels mix numbers and text across chunks.

        Complexity
        ----------
        Cumulative: O(n + k^2) time for ``k`` classes. Windowed: O(window + n).
        """
        y_true, y_pred = _check_pair(y_true, y_pred, "StreamingConfusionMatrix.update", "y_pred")
        if y_true.size == 0:
            return self
        chunk, chunk_labels = confusion_matrix(y_true, y_pred, self.labels)    # validates before any change
        if self.window is None:
            self._merge(chunk, chunk_labels)
        else:
            self._true = _keep_latest(self._true, y_true, self.window)
            self._pred = _keep_latest(self._pred, y_pred, self.window)
        self._n_seen += y_true.size
        return self

    def _merge(self, chunk: np.ndarray, chunk_labels: np.ndarray) -> None:
        """Add a chunk's matrix to the running one, growing the label set if needed."""
        if self.labels is not None:
            self._matrix += chunk
            return
        if self._labels.size and _label_group(self._labels) != _label_group(chunk_labels) \
                and "object" not in (_label_group(self._labels), _label_group(chunk_labels)):
            raise TypeError(
                f"StreamingConfusionMatrix.update: labels seen so far are {_label_group(self._labels)}s "
                f"but this chunk has {_label_group(chunk_labels)}s (dtype {chunk_labels.dtype})"
            )
        union = chunk_labels if self._labels.size == 0 else np.union1d(self._labels, chunk_labels)
        self._matrix = _embed(self._matrix, self._labels, union) + _embed(chunk, chunk_labels, union)
        self._labels = union

    def result(self):
        """Return the matrix and its labels, like the A1 ``confusion_matrix``.

        Returns
        -------
        matrix : ndarray of int, shape (k, k)
            ``matrix[i, j]`` counts samples with true label ``labels[i]`` and
            predicted label ``labels[j]``. Shape (0, 0) before any sample when
            ``labels`` is None.
        labels : ndarray, shape (k,)
            The sorted labels indexing the rows and columns.

        Complexity
        ----------
        Cumulative: O(k^2) (a copy). Windowed: O(window + k^2).
        """
        if self.window is not None and self._true is not None:
            return confusion_matrix(self._true, self._pred, self.labels)
        if self.window is not None:                                             # nothing in the window yet
            return np.zeros((self._labels.size,) * 2, dtype=np.int64), self._labels.copy()
        return self._matrix.copy(), self._labels.copy()


# ----------------------------------------------------------------------
# Accuracy
# ----------------------------------------------------------------------
class StreamingAccuracy(_StreamingMetric):
    """Fraction of correct predictions so far (or over the latest ``window`` samples).

    Parameters
    ----------
    window : int or None, default None
        ``None`` for all samples seen; ``w`` for the latest ``w`` samples.

    Examples
    --------
    >>> acc = StreamingAccuracy()
    >>> _ = acc.update([1, 0], [1, 1]); _ = acc.update([1, 1], [1, 1])
    >>> acc.result()
    0.75
    """

    _params = ("window",)

    def __init__(self, window=None):
        super().__init__(window)
        self.reset()

    def reset(self) -> "StreamingAccuracy":
        """Forget everything seen so far (the parameters are kept). Returns ``self``."""
        super().reset()
        self._correct = 0
        self._hits = None
        return self

    def update(self, y_true, y_pred) -> "StreamingAccuracy":
        """Add one chunk of true and predicted labels.

        Parameters
        ----------
        y_true, y_pred : array-like, shape (n,)
            An empty chunk is ignored.

        Returns
        -------
        self : StreamingAccuracy

        Raises
        ------
        ValueError
            If the inputs are not 1-D arrays of the same length.

        Complexity
        ----------
        O(n) time (O(window + n) when windowed); O(1) space cumulative.
        """
        y_true, y_pred = _check_pair(y_true, y_pred, "StreamingAccuracy.update", "y_pred")
        if y_true.size == 0:
            return self
        hits = np.asarray(y_true == y_pred, dtype=bool)
        if self.window is None:
            self._correct += int(hits.sum())
        else:
            self._hits = _keep_latest(self._hits, hits, self.window)
        self._n_seen += y_true.size
        return self

    def result(self) -> float:
        """Accuracy in [0, 1], or NaN before any sample. Complexity: O(1) (O(window) windowed)."""
        if self.n_samples_ == 0:
            return float("nan")
        if self.window is None:
            return self._correct / self._n_seen
        return float(self._hits.mean())


# ----------------------------------------------------------------------
# Precision, recall, F1
# ----------------------------------------------------------------------
class _ClassificationScore(_StreamingMetric):
    """Base for precision, recall and F1: a streaming confusion matrix plus a formula."""

    _params = ("average", "pos_label", "labels", "window")
    _name = ""

    def __init__(self, average: str = "binary", pos_label=1, labels=None, window=None):
        if average not in _VALID_AVERAGES:
            raise ValueError(f"{type(self).__name__}: average must be one of {_VALID_AVERAGES}, got {average!r}")
        super().__init__(window)
        self._cm = StreamingConfusionMatrix(labels, window)
        self.average = average
        self.pos_label = pos_label
        self.labels = self._cm.labels
        if average == "binary" and self.labels is not None:                      # same errors as A1, but at once
            _resolve_binary_pos_index(self.labels, pos_label)

    @property
    def n_seen_(self) -> int:
        """Number of samples given to ``update`` since the last ``reset``."""
        return self._cm.n_seen_

    def reset(self):
        """Forget everything seen so far (the parameters are kept). Returns ``self``."""
        self._cm.reset()
        return self

    def update(self, y_true, y_pred):
        """Add one chunk of true and predicted labels.

        Parameters
        ----------
        y_true, y_pred : array-like, shape (n,)
            An empty chunk is ignored.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the inputs are not 1-D arrays of the same length, contain NaN,
            or contain a label that is not in the fixed ``labels``.
        TypeError
            If the labels mix numbers and text across chunks.

        Complexity
        ----------
        As ``StreamingConfusionMatrix.update``.
        """
        self._cm.update(y_true, y_pred)
        return self

    def result(self) -> float:
        """The score in [0, 1] over everything counted, or NaN while it is undefined.

        NaN means: no samples yet, or (``average='binary'``, no fixed ``labels``)
        ``pos_label`` has not appeared yet.

        Raises
        ------
        ValueError
            If ``average='binary'`` and more than 2 classes have been seen.

        Complexity
        ----------
        O(k^2) for ``k`` classes (plus O(window) when windowed).
        """
        matrix, labels = self._cm.result()
        if matrix.sum() == 0:
            return float("nan")
        if self.average == "binary":
            if labels.size > 2:
                _resolve_binary_pos_index(labels, self.pos_label)               # raises A1's error
            if not np.any(np.asarray(labels == self.pos_label)):
                return float("nan")
        true_positives = np.diag(matrix).astype(float)
        predicted_totals = matrix.sum(axis=0).astype(float)
        actual_totals = matrix.sum(axis=1).astype(float)
        return self._score(true_positives, predicted_totals, actual_totals, labels)

    def _score(self, true_positives, predicted_totals, actual_totals, labels) -> float:
        raise NotImplementedError


class StreamingPrecision(_ClassificationScore):
    """Precision TP / (TP + FP) over the stream, with the A1 averaging rules.

    Parameters
    ----------
    average : {'binary', 'macro', 'micro'}, default 'binary'
        As in A1 ``precision``.
    pos_label : default 1
        Positive class for ``average='binary'``.
    labels : array-like or None, default None
        Fixed labels; with them the score is 0.0 (not NaN) before the positive
        class appears.
    window : int or None, default None
        ``None`` for all samples seen; ``w`` for the latest ``w`` samples.

    Examples
    --------
    >>> p = StreamingPrecision()
    >>> _ = p.update([1, 0, 1], [1, 1, 0]); _ = p.update([1], [1])
    >>> round(p.result(), 4)
    0.6667
    """

    def _score(self, true_positives, predicted_totals, actual_totals, labels) -> float:
        return _average_ratio(true_positives, predicted_totals, self.average, labels, self.pos_label, "precision")


class StreamingRecall(_ClassificationScore):
    """Recall TP / (TP + FN) over the stream, with the A1 averaging rules.

    Parameters are the same as for ``StreamingPrecision``.

    Examples
    --------
    >>> r = StreamingRecall()
    >>> _ = r.update([1, 0, 1], [1, 1, 0]); _ = r.update([1], [1])
    >>> round(r.result(), 4)
    0.6667
    """

    def _score(self, true_positives, predicted_totals, actual_totals, labels) -> float:
        return _average_ratio(true_positives, actual_totals, self.average, labels, self.pos_label, "recall")


class StreamingF1(_ClassificationScore):
    """F1 = 2PR / (P + R) over the stream, with the A1 averaging rules.

    Parameters are the same as for ``StreamingPrecision``. As in A1, 'macro'
    averages the per-class F1 scores.

    Examples
    --------
    >>> f = StreamingF1()
    >>> _ = f.update([1, 0, 1], [1, 1, 0]); _ = f.update([1], [1])
    >>> round(f.result(), 4)
    0.6667
    """

    def _score(self, true_positives, predicted_totals, actual_totals, labels) -> float:
        prec = safe_divide(true_positives, predicted_totals, fill=0.0)
        rec = safe_divide(true_positives, actual_totals, fill=0.0)
        f1_per_class = safe_divide(2 * prec * rec, prec + rec, fill=0.0)
        if self.average == "binary":
            return float(f1_per_class[_resolve_binary_pos_index(labels, self.pos_label)])
        if self.average == "micro":
            p_micro = safe_divide(true_positives.sum(), predicted_totals.sum(), fill=0.0)
            r_micro = safe_divide(true_positives.sum(), actual_totals.sum(), fill=0.0)
            return float(safe_divide(2 * p_micro * r_micro, p_micro + r_micro, fill=0.0))
        return float(f1_per_class.mean())


# ----------------------------------------------------------------------
# MSE
# ----------------------------------------------------------------------
class StreamingMSE(_StreamingMetric):
    """Mean squared error over the stream, for regression.

    Parameters
    ----------
    window : int or None, default None
        ``None`` for all samples seen; ``w`` for the latest ``w`` samples.

    Examples
    --------
    >>> m = StreamingMSE()
    >>> _ = m.update([1.0, 2.0], [1.0, 4.0]); _ = m.update([0.0], [1.0])
    >>> round(m.result(), 4)
    1.6667
    """

    _params = ("window",)

    def __init__(self, window=None):
        super().__init__(window)
        self.reset()

    def reset(self) -> "StreamingMSE":
        """Forget everything seen so far (the parameters are kept). Returns ``self``."""
        super().reset()
        self._sum = 0.0
        self._squares = None
        return self

    def update(self, y_true, y_pred) -> "StreamingMSE":
        """Add one chunk of true and predicted values.

        Parameters
        ----------
        y_true, y_pred : array-like of numbers, shape (n,)
            An empty chunk is ignored.

        Returns
        -------
        self : StreamingMSE

        Raises
        ------
        TypeError
            If the inputs are not numeric.
        ValueError
            If the inputs are not 1-D arrays of the same length or contain NaN
            (one NaN would otherwise ruin every later result).

        Complexity
        ----------
        O(n) time (O(window + n) when windowed); O(1) space cumulative.
        """
        y_true, y_pred = _check_pair(y_true, y_pred, "StreamingMSE.update", "y_pred")
        y_true = _as_float(y_true, "StreamingMSE.update", "y_true")
        y_pred = _as_float(y_pred, "StreamingMSE.update", "y_pred")
        if y_true.size == 0:
            return self
        squares = (y_true - y_pred) ** 2
        if self.window is None:
            self._sum += float(squares.sum())
        else:
            self._squares = _keep_latest(self._squares, squares, self.window)
        self._n_seen += y_true.size
        return self

    def result(self) -> float:
        """Mean squared error, or NaN before any sample. Complexity: O(1) (O(window) windowed)."""
        if self.n_samples_ == 0:
            return float("nan")
        if self.window is None:
            return self._sum / self._n_seen
        return float(self._squares.mean())


# ----------------------------------------------------------------------
# AUC
# ----------------------------------------------------------------------
class StreamingAUC(_StreamingMetric):
    """Area under the ROC curve that accumulates over the stream, from score histograms.

    Scores are counted into ``bins`` equal bins, one count per class per bin, so
    memory stays at O(bins) however long the stream is. The ROC curve is read
    off the counts (highest score first) and its area is the A1 ``auc``.

    Parameters
    ----------
    bins : int, default 1000
        Number of score bins (at least 1). More bins, smaller approximation error.
    score_range : (float, float), default (0.0, 1.0)
        Lowest and highest allowed score (for example probabilities).
    window : int or None, default None
        ``None`` for all samples seen; ``w`` for the latest ``w`` samples.

    Notes
    -----
    A positive and a negative whose scores fall in the same bin count as a tie.
    The result is exact when every bin holds one distinct score value.

    Examples
    --------
    >>> auc_ = StreamingAUC(bins=10)
    >>> _ = auc_.update([0, 1], [0.15, 0.85]); _ = auc_.update([0, 1], [0.55, 0.95])
    >>> auc_.result()
    1.0
    """

    _params = ("bins", "score_range", "window")

    def __init__(self, bins: int = 1000, score_range=(0.0, 1.0), window=None):
        super().__init__(window)
        self.bins = _check_int(bins, "bins", 1)
        try:
            low, high = (float(v) for v in score_range)
        except (TypeError, ValueError):
            raise ValueError(f"score_range must be two numbers (low, high), got {score_range!r}") from None
        if not (np.isfinite(low) and np.isfinite(high) and low < high):
            raise ValueError(f"score_range must be finite with low < high, got {score_range!r}")
        self.score_range = (low, high)
        self.reset()

    def reset(self) -> "StreamingAUC":
        """Forget everything seen so far (the parameters are kept). Returns ``self``."""
        super().reset()
        self._pos = np.zeros(self.bins, dtype=np.int64)
        self._neg = np.zeros(self.bins, dtype=np.int64)
        self._bin = self._label = None
        return self

    def update(self, y_true, y_score) -> "StreamingAUC":
        """Add one chunk of true labels (0 or 1) and scores.

        Parameters
        ----------
        y_true : array-like of 0 / 1, shape (n,)
        y_score : array-like of numbers, shape (n,)
            Higher means "more likely class 1"; must lie inside ``score_range``.
            An empty chunk is ignored.

        Returns
        -------
        self : StreamingAUC

        Raises
        ------
        TypeError
            If the inputs are not numeric.
        ValueError
            If the inputs are not 1-D arrays of the same length, ``y_true`` has
            a value other than 0 or 1, or a score is NaN or outside ``score_range``.

        Complexity
        ----------
        O(n + bins) time (O(window + bins) when windowed); O(bins) space.
        """
        where = "StreamingAUC.update"
        y_true, y_score = _check_pair(y_true, y_score, where, "y_score")
        y_true = _as_float(y_true, where, "y_true")
        y_score = _as_float(y_score, where, "y_score")
        if y_true.size == 0:
            return self
        if not np.isin(y_true, (0, 1)).all():
            raise ValueError(f"{where}: y_true must contain only 0 and 1")
        low, high = self.score_range
        if y_score.min() < low or y_score.max() > high:
            raise ValueError(
                f"{where}: scores must lie in score_range {self.score_range}, "
                f"got values from {y_score.min()} to {y_score.max()}"
            )
        at = np.floor((y_score - low) / (high - low) * self.bins).astype(np.int64)
        at = np.minimum(at, self.bins - 1)                                      # a score equal to `high` goes in the last bin
        positive = y_true == 1
        if self.window is None:
            self._pos += np.bincount(at[positive], minlength=self.bins)
            self._neg += np.bincount(at[~positive], minlength=self.bins)
        else:
            self._bin = _keep_latest(self._bin, at, self.window)
            self._label = _keep_latest(self._label, positive, self.window)
        self._n_seen += y_true.size
        return self

    def _counts(self):
        """Positives and negatives per bin for the samples currently counted."""
        if self.window is None:
            return self._pos, self._neg
        if self._bin is None:
            return np.zeros(self.bins, dtype=np.int64), np.zeros(self.bins, dtype=np.int64)
        return (np.bincount(self._bin[self._label], minlength=self.bins),
                np.bincount(self._bin[~self._label], minlength=self.bins))

    def _curve(self, pos: np.ndarray, neg: np.ndarray):
        """(fpr, tpr) from per-bin counts, highest score first, starting at (0, 0)."""
        tpr = np.concatenate(([0.0], np.cumsum(pos[::-1]) / pos.sum()))
        fpr = np.concatenate(([0.0], np.cumsum(neg[::-1]) / neg.sum()))
        return fpr, tpr

    def roc_curve(self):
        """ROC curve of the samples counted so far, one point per bin.

        Returns
        -------
        fpr, tpr : ndarray, shape (bins + 1,)
            False- and true-positive rates, starting at (0, 0).
        thresholds : ndarray, shape (bins + 1,)
            Lowest score of each bin, highest first; the first entry is +inf
            ("predict nothing as positive"). Same layout as the A1 ``roc_curve``.

        Raises
        ------
        ValueError
            If no 0 or no 1 has been seen yet.

        Complexity
        ----------
        O(bins) time and space (plus O(window) when windowed).
        """
        pos, neg = self._counts()
        if pos.sum() == 0 or neg.sum() == 0:
            raise ValueError("StreamingAUC.roc_curve(): at least one 0 and one 1 are needed so far")
        fpr, tpr = self._curve(pos, neg)
        low, high = self.score_range
        lower_edges = np.linspace(low, high, self.bins + 1)[:-1]
        return fpr, tpr, np.concatenate(([np.inf], lower_edges[::-1]))

    def result(self) -> float:
        """Area under the ROC curve in [0, 1], or NaN until both classes have been seen.

        Complexity
        ----------
        O(bins) (plus O(window) when windowed).
        """
        pos, neg = self._counts()
        if pos.sum() == 0 or neg.sum() == 0:
            return float("nan")
        return auc(*self._curve(pos, neg))