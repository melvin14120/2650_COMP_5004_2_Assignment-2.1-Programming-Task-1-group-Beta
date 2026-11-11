"""
stream.py
StreamTrainer: runs a model or pipeline over a stream of chunks and logs every step.

Provides
--------
- StreamTrainer(model, metrics=None, window=None)
    .score_chunk(X_chunk, y_chunk)   test the CURRENT model on a chunk, update the scores
    .fit_chunk(X_chunk, y_chunk)     train the model on a chunk (calls ``partial_fit``)
    .process_chunk(X_chunk, y_chunk) score_chunk, then fit_chunk, as ONE log row
    .run(stream)                     process_chunk for every (X, y) pair of an iterable
    .history(name)                   one logged column as a NumPy array
    .reset()                         clear the scores and the log (the model is kept)
- iter_chunks(X, y, chunk_size, shuffle=False, random_state=None)
    cut a data set into chunks to simulate a stream

Design notes
------------
The trainer follows the usual "test, then train" (prequential) rule for
streams: a chunk is first used to TEST the model as it is now, and only then to
TRAIN it. Every chunk is therefore an unseen test set for the model, and the
cumulative accuracy is an honest estimate of how the model does on new data.

The log has one row (a plain dict) per chunk, with these columns:

    chunk                 position of the chunk in the stream (0, 1, 2, ...)
    n_samples             rows in the chunk
    scored                False while the model cannot predict yet (no tree trained)
    chunk_accuracy        accuracy on this chunk alone (NaN if not scored)
    cumulative_accuracy   accuracy over all scored chunks so far
    rolling_accuracy      accuracy over the latest ``window`` samples (only if ``window`` is given)
    <metric names>        the running value of each metric passed in ``metrics``
    fitted                True once ``fit_chunk`` has trained on the chunk
    fit_seconds           wall-clock time of ``partial_fit`` (includes a retrain when one happens)
    retrained             True if this chunk made the model retrain
    n_retrains            retrains so far (0 if the model has no ``n_retrains_``)
    n_seen                samples trained on so far
    memory_bytes          approximate bytes held by the model (arrays only)

A metric in ``metrics`` is any object with ``update(y_true, y_pred)``,
``result()`` and ``reset()``, for example the streaming metrics of
``numcompute_stream.metrics``. It receives predicted LABELS, so a score-based
metric such as ``StreamingAUC`` does not belong here. Its logged value is its
running result (cumulative, or rolling if it was built with a ``window``).

``model`` is anything with ``partial_fit(X, y)`` and ``predict(X)``: a
``DecisionTreeClassifier``, or a ``Pipeline`` ending in one. A model that has
no ``n_retrains_`` still works; ``retrained`` is then always False. Before the
first retrain a tree raises ``NotFittedError`` when asked to predict; the
trainer catches this and logs the chunk as ``scored=False``.

Missing values and edge cases
-----------------------------
- An empty chunk (zero rows) is ignored: nothing is scored, trained or logged.
- A row is created by ``score_chunk`` and completed by the next ``fit_chunk``.
  ``fit_chunk`` without a ``score_chunk`` before it makes its own row, and a
  ``score_chunk`` that is never followed by ``fit_chunk`` leaves a row with
  ``fitted=False``.
- If a metric in ``metrics`` raises while updating, the metrics before it keep
  the chunk and no row is logged. The built-in accuracy trackers are checked
  first, so they never end up half updated.

Axis convention
---------------
``X_chunk`` has shape (n_samples, n_features): rows are samples, columns are
features. ``y_chunk`` has shape (n_samples,).
"""

from __future__ import annotations

import math
import time

import numpy as np

from numcompute.preprocessing import NotFittedError

from ._checks import _check_int
from .metrics import StreamingAccuracy

__all__ = ["StreamTrainer", "iter_chunks"]

# Column names the trainer writes itself; a metric may not use them.
_RESERVED = (
    "chunk", "n_samples", "scored", "chunk_accuracy", "cumulative_accuracy",
    "rolling_accuracy", "fitted", "fit_seconds", "retrained", "n_retrains",
    "n_seen", "memory_bytes",
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _as_chunk(X, y, where: str):
    """Return ``(X, y)`` as arrays; ``X`` must be 2-D, ``y`` 1-D with one entry per row."""
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"{where}: X must be 2-D (n_samples, n_features), got shape {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"{where}: y must be 1-D (n_samples,), got shape {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"{where}: X has {X.shape[0]} rows but y has {y.shape[0]} entries.")
    return X, y


def _array_bytes(value) -> int:
    """Bytes held by NumPy arrays inside ``value`` (an array, or lists, tuples and dicts of them)."""
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (list, tuple)):
        return sum(_array_bytes(v) for v in value)
    if isinstance(value, dict):
        return sum(_array_bytes(v) for v in value.values())
    return 0


def _memory_bytes(obj) -> int:
    """Approximate bytes an estimator holds in NumPy arrays.

    Uses the object's own ``memory_bytes()`` when it has one (the tree does),
    adds up the steps of a ``Pipeline`` and the members of a ``FeatureUnion``,
    and otherwise counts the arrays stored in its attributes.
    """
    method = getattr(obj, "memory_bytes", None)
    if callable(method):
        return int(method())
    for parts in (getattr(obj, "steps", None), getattr(obj, "transformers", None)):
        if parts is not None:
            return sum(_memory_bytes(part) for _, part in parts)
    return _array_bytes(vars(obj)) if hasattr(obj, "__dict__") else 0


def _retrain_count(model) -> int:
    """``n_retrains_`` of the model (of the last step for a pipeline); 0 if there is none."""
    steps = getattr(model, "steps", None)
    if steps:
        model = steps[-1][1]
    return int(getattr(model, "n_retrains_", 0))


# ----------------------------------------------------------------------
# StreamTrainer
# ----------------------------------------------------------------------
class StreamTrainer:
    """Manage a model (or pipeline), its scores and a per-chunk log over a stream.

    Parameters
    ----------
    model : object with ``partial_fit(X, y)`` and ``predict(X)``
        Usually a ``Pipeline`` ending in a ``DecisionTreeClassifier``.
    metrics : dict of {str: metric}, optional
        Extra metrics, each with ``update(y_true, y_pred)``, ``result()`` and
        ``reset()``. The name becomes a column of the log.
    window : int or None, default None
        If given, ``rolling_accuracy`` (accuracy over the latest ``window``
        samples) is logged as well as the cumulative accuracy.

    Attributes
    ----------
    log_ : list of dict
        One row per chunk (columns in the module notes).
    accuracy_ : StreamingAccuracy
        Cumulative accuracy over all scored chunks.
    rolling_accuracy_ : StreamingAccuracy or None
        Accuracy over the latest ``window`` samples (``None`` without ``window``).
    n_seen_ : int
        Samples trained on so far.
    last_true_, last_pred_ : ndarray or None
        Labels and predictions of the latest scored chunk, for plotting.

    Raises
    ------
    TypeError
        If ``model`` has no ``partial_fit`` or ``predict``, or a metric lacks
        ``update``, ``result`` or ``reset``, or a metric name is not a string.
    ValueError
        If a metric name is one of the trainer's own columns, or ``window`` < 1.

    Examples
    --------
    >>> import numpy as np
    >>> from numcompute_stream.tree import DecisionTreeClassifier
    >>> trainer = StreamTrainer(DecisionTreeClassifier(max_depth=2, buffer_size=4))
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]]); y = np.array([0, 0, 1, 1])
    >>> row = trainer.process_chunk(X, y)       # no tree yet, so nothing to score
    >>> row["scored"], row["retrained"]
    (False, True)
    >>> row = trainer.process_chunk(X, y)       # tested first, then trained
    >>> row["scored"], row["chunk_accuracy"], row["cumulative_accuracy"]
    (True, 1.0, 1.0)
    """

    def __init__(self, model, metrics=None, window=None):
        for method in ("partial_fit", "predict"):
            if not callable(getattr(model, method, None)):
                raise TypeError(
                    f"StreamTrainer: model must implement {method}; {type(model).__name__} does not."
                )
        metrics = {} if metrics is None else dict(metrics)
        for name, metric in metrics.items():
            if not isinstance(name, str):
                raise TypeError(f"StreamTrainer: metric names must be strings, got {name!r}.")
            if name in _RESERVED:
                raise ValueError(f"StreamTrainer: metric name {name!r} is one of the log's own columns.")
            missing = [m for m in ("update", "result", "reset") if not callable(getattr(metric, m, None))]
            if missing:
                raise TypeError(
                    f"StreamTrainer: metric {name!r} ({type(metric).__name__}) lacks: {', '.join(missing)}."
                )
        self.model = model
        self.metrics = metrics
        self.window = None if window is None else _check_int(window, "window", 1)
        self.reset()

    def __repr__(self) -> str:
        return (f"StreamTrainer(model={type(self.model).__name__}, "
                f"metrics={list(self.metrics)}, window={self.window!r})")

    # -- housekeeping ---------------------------------------------------------
    def reset(self) -> "StreamTrainer":
        """Clear the log and every score. The model keeps what it has learned.

        Returns
        -------
        self : StreamTrainer
        """
        self.log_: list = []
        self.accuracy_ = StreamingAccuracy()
        self.rolling_accuracy_ = None if self.window is None else StreamingAccuracy(window=self.window)
        for metric in self.metrics.values():
            metric.reset()
        self.n_seen_ = 0
        self.last_true_ = None
        self.last_pred_ = None
        self._open = False  # True while the last row is waiting for its fit_chunk
        return self

    def _new_row(self, n_samples: int) -> dict:
        """Append a row holding the current scores and model state; nothing scored or fitted yet."""
        row = {
            "chunk": len(self.log_),
            "n_samples": int(n_samples),
            "scored": False,
            "chunk_accuracy": math.nan,
            "cumulative_accuracy": self.accuracy_.result(),
        }
        if self.rolling_accuracy_ is not None:
            row["rolling_accuracy"] = self.rolling_accuracy_.result()
        for name, metric in self.metrics.items():
            row[name] = metric.result()
        row.update(
            fitted=False,
            fit_seconds=math.nan,
            retrained=False,
            n_retrains=_retrain_count(self.model),
            n_seen=self.n_seen_,
            memory_bytes=_memory_bytes(self.model),
        )
        self.log_.append(row)
        return row

    # -- the two steps of a stream --------------------------------------------
    def score_chunk(self, X_chunk, y_chunk):
        """Test the current model on a chunk and update the scores (no training).

        Parameters
        ----------
        X_chunk : array-like, shape (n_samples, n_features)
        y_chunk : array-like, shape (n_samples,)
            True labels.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,) or None
            The predictions, or ``None`` when nothing was scored: an empty
            chunk, or a model that cannot predict yet (``NotFittedError``).

        Raises
        ------
        ValueError
            If ``X_chunk`` is not 2-D, ``y_chunk`` is not 1-D, or their lengths
            differ; also whatever the model's ``predict`` or a metric raises.

        Complexity
        ----------
        The cost of ``predict`` plus O(n) to update the scores on n samples
        (O(window + n) with a rolling window).
        """
        X, y = _as_chunk(X_chunk, y_chunk, "StreamTrainer.score_chunk")
        if y.shape[0] == 0:
            return None
        try:
            y_pred = np.asarray(self.model.predict(X))
        except NotFittedError:
            self._new_row(y.shape[0])
            self._open = True
            return None
        chunk_accuracy = StreamingAccuracy().update(y, y_pred).result()  # also checks y and y_pred
        self.accuracy_.update(y, y_pred)
        if self.rolling_accuracy_ is not None:
            self.rolling_accuracy_.update(y, y_pred)
        for metric in self.metrics.values():
            metric.update(y, y_pred)
        row = self._new_row(y.shape[0])
        row.update(scored=True, chunk_accuracy=chunk_accuracy)
        self._open = True
        self.last_true_, self.last_pred_ = y, y_pred
        return y_pred

    def fit_chunk(self, X_chunk, y_chunk) -> "StreamTrainer":
        """Train the model on a chunk with ``partial_fit`` and complete its log row.

        Parameters
        ----------
        X_chunk : array-like, shape (n_samples, n_features)
        y_chunk : array-like, shape (n_samples,)

        Returns
        -------
        self : StreamTrainer

        Raises
        ------
        ValueError
            If ``X_chunk`` is not 2-D, ``y_chunk`` is not 1-D, or their lengths
            differ; also whatever the model's ``partial_fit`` raises (then the
            log and ``n_seen_`` are left unchanged).

        Complexity
        ----------
        The cost of the model's ``partial_fit``, which includes a retrain when
        the buffer fills.
        """
        X, y = _as_chunk(X_chunk, y_chunk, "StreamTrainer.fit_chunk")
        if y.shape[0] == 0:
            return self
        before = _retrain_count(self.model)
        start = time.perf_counter()
        self.model.partial_fit(X, y)
        seconds = time.perf_counter() - start
        self.n_seen_ += y.shape[0]
        row = self.log_[-1] if self._open else self._new_row(y.shape[0])
        after = _retrain_count(self.model)
        row.update(
            fitted=True,
            fit_seconds=seconds,
            retrained=after > before,
            n_retrains=after,
            n_seen=self.n_seen_,
            memory_bytes=_memory_bytes(self.model),
        )
        self._open = False
        return self

    def process_chunk(self, X_chunk, y_chunk):
        """Score a chunk, then train on it: one row of the log.

        Parameters
        ----------
        X_chunk : array-like, shape (n_samples, n_features)
        y_chunk : array-like, shape (n_samples,)

        Returns
        -------
        row : dict or None
            A copy of the row that was logged, or ``None`` for an empty chunk.

        Raises
        ------
        ValueError
            As for ``score_chunk`` and ``fit_chunk``.
        """
        X, y = _as_chunk(X_chunk, y_chunk, "StreamTrainer.process_chunk")
        if y.shape[0] == 0:
            return None
        self.score_chunk(X, y)
        self.fit_chunk(X, y)
        return dict(self.log_[-1])

    def run(self, stream) -> "StreamTrainer":
        """Call ``process_chunk`` for every ``(X_chunk, y_chunk)`` pair in ``stream``.

        Parameters
        ----------
        stream : iterable of (X_chunk, y_chunk)
            For example the generator returned by ``iter_chunks``.

        Returns
        -------
        self : StreamTrainer
        """
        for X_chunk, y_chunk in stream:
            self.process_chunk(X_chunk, y_chunk)
        return self

    def history(self, name: str) -> np.ndarray:
        """One column of the log as an array, one entry per chunk.

        Parameters
        ----------
        name : str
            A column such as ``'cumulative_accuracy'`` or the name of a metric.

        Returns
        -------
        values : ndarray, shape (n_chunks,)

        Raises
        ------
        KeyError
            If there is no such column; the message lists the columns.
        """
        if name not in self.columns():
            raise KeyError(f"StreamTrainer.history: no column {name!r}; columns are {self.columns()}.")
        return np.array([row[name] for row in self.log_])

    def columns(self) -> list:
        """Names of the log columns, in order."""
        names = list(_RESERVED)
        if self.rolling_accuracy_ is None:
            names.remove("rolling_accuracy")
        return names + list(self.metrics)


# ----------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------
def iter_chunks(X, y, chunk_size, shuffle=False, random_state=None):
    """Cut a data set into chunks to simulate a stream.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
    chunk_size : int
        Rows per chunk (>= 1). The last chunk may be smaller.
    shuffle : bool, default False
        Shuffle the rows once before cutting (a fixed seed with ``random_state``).
    random_state : int or None
        Seed for the shuffle.

    Returns
    -------
    chunks : generator of (X_chunk, y_chunk)
        Without ``shuffle`` the chunks are views of ``X`` and ``y``, not copies.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D, ``y`` is not 1-D, their lengths differ, or
        ``chunk_size`` < 1. Raised when this function is called, not later.
    TypeError
        If ``chunk_size`` is not an integer.

    Complexity
    ----------
    O(n) in total; each chunk costs O(chunk_size).

    Examples
    --------
    >>> import numpy as np
    >>> chunks = list(iter_chunks(np.arange(10).reshape(5, 2), np.arange(5), 2))
    >>> [len(y) for _, y in chunks]
    [2, 2, 1]
    """
    X, y = _as_chunk(X, y, "iter_chunks")
    chunk_size = _check_int(chunk_size, "chunk_size", 1)
    order = np.random.default_rng(random_state).permutation(y.shape[0]) if shuffle else None
    return _generate(X, y, chunk_size, order)


def _generate(X, y, chunk_size, order):
    """The generator behind ``iter_chunks`` (split off so errors are raised early)."""
    for start in range(0, y.shape[0], chunk_size):
        rows = slice(start, start + chunk_size) if order is None else order[start:start + chunk_size]
        yield X[rows], y[rows]