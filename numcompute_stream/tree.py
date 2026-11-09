"""
tree.py
Decision tree classifier with buffered streaming retraining.

Provides
--------
- DecisionTreeClassifier(criterion='gini'|'entropy', max_depth=5,
  min_samples_split=2, max_features=None, buffer_size=500, random_state=None)
    .fit(X, y)                       train a tree from scratch
    .partial_fit(X_chunk, y_chunk)   buffer a chunk; retrain when the buffer is full
    .flush()                         retrain now on a partly filled buffer
    .predict(X) / .predict_proba(X) / .score(X, y)

Design notes
------------
The tree is grown greedily (CART). At every node the best question
``feature <= threshold`` is the one with the largest drop in impurity
(Gini or entropy). The search is vectorised across ALL thresholds of ALL
features at once: sort each feature once, take cumulative class counts and
evaluate the impurity of every possible cut in one array expression. The
only Python loops run over tree nodes (build) and tree levels (predict),
never over samples.

Buffered retraining: ``partial_fit`` appends each chunk to an internal
buffer. Once the buffer holds at least ``buffer_size`` samples, the tree is
rebuilt from the buffered samples only and the buffer is emptied. Between
retrains ``predict`` keeps using the most recently trained tree::

    tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
    for X_chunk, y_chunk in stream:
        tree.partial_fit(X_chunk, y_chunk)

Missing values and ties
-----------------------
- NaN is ignored when searching thresholds for a feature. At every split the
  NaN samples follow the larger child (stored in the node), so NaN is also
  accepted at predict time.
- A threshold is only placed between two DIFFERENT values, so equal values
  are never separated. Thresholds are midpoints; ``x <= threshold`` goes left.
- Equal-gain splits resolve to the lowest feature index, then the lowest
  threshold. A leaf predicts its most frequent class; ties go to the
  smallest label. Results are therefore deterministic.
- Zero-gain splits are allowed (as in scikit-learn); otherwise XOR-like data
  could never be learned: the first cut gains nothing but enables gain below.

Axis convention
---------------
``X`` has shape (n_samples, n_features): rows are samples, columns are
features. ``y`` is 1-D with one label per row. Input checks and
``NotFittedError`` are reused from ``numcompute.preprocessing`` so the
errors read the same as the A1 transformers.
"""

from __future__ import annotations

import numpy as np

from numcompute.preprocessing import NotFittedError, _check_n_features, _check_numeric

__all__ = ["DecisionTreeClassifier"]

_CRITERIA = ("gini", "entropy")
_EPS = 1e-12  # tolerance when comparing gains for ties

# Attributes created by training; removed again by _reset_state so that
# hasattr(tree, "classes_") is False until a tree exists.
_LEARNED = (
    "feature_", "threshold_", "children_left_", "children_right_",
    "value_", "nan_left_", "classes_", "feature_importances_", "n_features_in_",
)


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------
def _check_labels(y, n_samples: int, owner: str) -> np.ndarray:
    """Return ``y`` as a 1-D array of labels with ``n_samples`` entries.

    Raises
    ------
    TypeError
        If ``y`` has dtype object.
    ValueError
        If ``y`` is not 1-D, holds NaN, or its length differs from ``n_samples``.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(
            f"{owner} expects y of shape (n_samples,), got {y.ndim}-D array with shape {y.shape}."
        )
    if y.dtype.kind == "O":
        raise TypeError(f"{owner} requires y to be a numeric or string array, got dtype object.")
    if y.dtype.kind == "f" and np.isnan(y).any():
        raise ValueError(f"{owner}: y contains NaN labels.")
    if y.shape[0] != n_samples:
        raise ValueError(f"{owner}: X has {n_samples} rows but y has {y.shape[0]} labels.")
    return y


def _check_int(value, name: str, minimum: int) -> int:
    """Return ``value`` as ``int`` after checking it is an integer >= ``minimum``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return int(value)


# ----------------------------------------------------------------------
# Impurity and split search (vectorised)
# ----------------------------------------------------------------------
def _impurity(counts: np.ndarray, criterion: str) -> np.ndarray:
    """Impurity of class-count vectors along the last axis.

    Parameters
    ----------
    counts : ndarray, shape (..., n_classes)
        Class counts of one or many groups.
    criterion : {'gini', 'entropy'}
        ``1 - sum(p^2)`` or ``-sum(p * log2(p))``.

    Returns
    -------
    ndarray, shape (...)
        Impurity of each group; an empty group (all counts 0) has impurity 0.

    Complexity
    ----------
    Time: O(size of counts). Space: O(size of counts).
    """
    total = counts.sum(axis=-1, keepdims=True)
    p = np.divide(counts, total, out=np.zeros(counts.shape, dtype=float), where=total > 0)
    if criterion == "gini":
        return np.where(total[..., 0] > 0, 1.0 - np.sum(p * p, axis=-1), 0.0)
    safe = np.where(p > 0, p, 1.0)  # log2(1) = 0, so a zero probability adds nothing
    return -np.sum(p * np.log2(safe), axis=-1) + 0.0


def _best_split(X: np.ndarray, onehot: np.ndarray, features: np.ndarray, criterion: str):
    """Find the best ``(feature, threshold)`` over ``features`` in one vectorised pass.

    Parameters
    ----------
    X : ndarray, shape (n, n_features)
        Samples that reach the node; NaN allowed.
    onehot : ndarray, shape (n, n_classes)
        One-hot class labels of those samples.
    features : ndarray of int
        Candidate feature indices.
    criterion : {'gini', 'entropy'}

    Returns
    -------
    (feature, threshold, nan_left, score) or None
        ``score`` is the impurity decrease scaled by the fraction of non-NaN
        samples. ``None`` if no valid cut exists (all values equal, or only NaN).

    Complexity
    ----------
    Time: O(f * n log n + f * n * k) for ``f`` features, ``n`` samples and
    ``k`` classes (the sorts plus the class-count cumulative sums).
    Space: O(f * n * k).
    """
    n, _ = onehot.shape
    if n < 2 or features.size == 0:
        return None
    Xs = X[:, features]                                   # (n, f)
    f = Xs.shape[1]
    order = np.argsort(Xs, axis=0, kind="stable")         # NaN sorts last
    vs = np.take_along_axis(Xs, order, axis=0)            # values sorted per feature
    n_valid = n - np.isnan(Xs).sum(axis=0)                # non-NaN rows per feature
    left = np.cumsum(onehot[order], axis=0)               # (n, f, k): class counts at or before each row
    totals = left[np.maximum(n_valid - 1, 0), np.arange(f)]
    totals = np.where((n_valid > 0)[:, None], totals, 0.0)

    left_c = left[:-1]                                    # a cut after sorted row i
    n_left = np.arange(1, n, dtype=float)[:, None]        # (n-1, 1)
    n_right = n_valid[None, :] - n_left                   # (n-1, f)
    right_c = totals[None, :, :] - left_c
    valid = (vs[:-1] < vs[1:]) & (n_right >= 1)           # strict <: equal values are never split

    parent = _impurity(totals, criterion)                 # (f,)
    child = n_left * _impurity(left_c, criterion) + n_right * _impurity(right_c, criterion)
    child = child / np.maximum(n_valid, 1)[None, :]
    gain = (parent[None, :] - child) * (n_valid / n)[None, :]
    gain = np.where(valid, gain, -np.inf)

    gain_t = gain.T                                       # (f, n-1): ties pick lowest feature, then lowest cut
    best = gain_t.max()
    if not np.isfinite(best):
        return None
    flat = int(np.argmax(gain_t.ravel() >= best - _EPS))
    j, i = divmod(flat, n - 1)
    lo, hi = vs[i, j], vs[i + 1, j]
    thr = lo + (hi - lo) / 2.0
    if thr >= hi:                                         # float rounding guard
        thr = lo
    nan_left = bool(i + 1 >= n_valid[j] - (i + 1))        # NaN follows the larger child
    return int(features[j]), float(thr), nan_left, float(best)


# ----------------------------------------------------------------------
# DecisionTreeClassifier
# ----------------------------------------------------------------------
class DecisionTreeClassifier:
    """Depth-limited classification tree (Gini or entropy) with buffered streaming.

    Parameters
    ----------
    criterion : {'gini', 'entropy'}, default 'gini'
        Impurity measure used to score splits.
    max_depth : int or None, default 5
        Maximum depth (the root has depth 0). ``None`` grows until leaves are pure.
    min_samples_split : int, default 2
        Minimum number of samples a node needs before it can be split (>= 2).
    max_features : None, int, float, 'sqrt' or 'log2', default None
        Number of features examined at each split. ``None`` uses all of them,
        a float is a fraction of the features; a random subset is drawn per node.
    buffer_size : int, default 500
        Streaming: retrain once the buffer holds at least this many samples.
    random_state : int or None, default None
        Seed for the feature subsampling (only used when ``max_features`` is set).

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Sorted labels seen in the **most recent training**. The tree is rebuilt
        from the buffer only, so this can change between retrains.
    n_features_in_ : int
        Number of columns, fixed by the first data seen.
    feature_importances_ : ndarray, shape (n_features_in_,)
        Normalised total impurity decrease per feature (zeros for a single leaf).
    feature_, threshold_, children_left_, children_right_, nan_left_ : ndarray
        Flat tree: node ``i`` splits on ``feature_[i]`` (-1 for a leaf) at
        ``threshold_[i]``; children are node ids (-1 for a leaf); ``nan_left_[i]``
        says where NaN goes.
    value_ : ndarray, shape (n_nodes, n_classes)
        Class counts of the training samples that reached each node.
    n_retrains_ : int
        How many times the tree has been trained (``fit`` counts as one).
    n_seen_ : int
        Total samples passed to ``partial_fit``.
    n_buffered_ : int
        Samples currently waiting in the buffer.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]]); y = np.array([0, 0, 1, 1])
    >>> DecisionTreeClassifier(max_depth=1).fit(X, y).predict([[0.5], [2.5]]).tolist()
    [0, 1]
    """

    def __init__(self, criterion="gini", max_depth=5, min_samples_split=2,
                 max_features=None, buffer_size=500, random_state=None):
        if criterion not in _CRITERIA:
            raise ValueError(f"criterion must be one of {_CRITERIA}, got {criterion!r}")
        self.criterion = criterion
        self.max_depth = None if max_depth is None else _check_int(max_depth, "max_depth", 0)
        self.min_samples_split = _check_int(min_samples_split, "min_samples_split", 2)
        self.max_features = max_features
        self._n_candidate_features(1)  # validate max_features now, not in the middle of a stream
        self.buffer_size = _check_int(buffer_size, "buffer_size", 1)
        self.random_state = random_state
        self._reset_state()

    # -- housekeeping ---------------------------------------------------------
    def _reset_state(self) -> None:
        """Forget the tree, the buffer and the counters (not the hyperparameters)."""
        for name in _LEARNED:
            self.__dict__.pop(name, None)
        self.n_retrains_ = 0
        self.n_seen_ = 0
        self.n_buffered_ = 0
        self._buf_X: list = []
        self._buf_y: list = []
        self._rng = np.random.default_rng(self.random_state)

    def get_params(self) -> dict:
        """Constructor arguments as a dict."""
        return dict(criterion=self.criterion, max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split, max_features=self.max_features,
                    buffer_size=self.buffer_size, random_state=self.random_state)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"DecisionTreeClassifier({params})"

    def _n_candidate_features(self, n_features: int) -> int:
        """Number of features to examine per split, from ``max_features``."""
        mf = self.max_features
        if mf is None:
            return n_features
        if isinstance(mf, str):
            if mf == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            if mf == "log2":
                return max(1, int(np.log2(n_features)))
            raise ValueError(f"max_features string must be 'sqrt' or 'log2', got {mf!r}")
        if isinstance(mf, bool):
            raise TypeError("max_features must be None, an int, a float or a string, got bool.")
        if isinstance(mf, (int, np.integer)):
            if mf < 1:
                raise ValueError(f"max_features must be >= 1, got {mf}")
            return min(int(mf), n_features)
        if isinstance(mf, (float, np.floating)):
            if not 0.0 < mf <= 1.0:
                raise ValueError(f"a float max_features must be in (0, 1], got {mf}")
            return max(1, int(mf * n_features))
        raise TypeError("max_features must be None, an int, a float or a string.")

    @property
    def is_fitted(self) -> bool:
        """True once a tree has been trained (by ``fit`` or by a full buffer)."""
        return hasattr(self, "classes_")

    def _require_trained(self, caller: str) -> None:
        if not self.is_fitted:
            raise NotFittedError(
                f"DecisionTreeClassifier.{caller}: no tree has been trained yet "
                f"({self.n_buffered_} of {self.buffer_size} buffered samples). "
                "Call fit(X, y), or partial_fit until the buffer fills, or flush()."
            )

    # -- training -------------------------------------------------------------
    def fit(self, X, y) -> "DecisionTreeClassifier":
        """Train the tree from scratch on ``(X, y)``; any buffered chunks are discarded.

        Parameters
        ----------
        X : array-like of numbers, shape (n_samples, n_features)
            NaN marks missing values.
        y : array-like, shape (n_samples,)
            Class labels (numbers or strings).

        Returns
        -------
        self : DecisionTreeClassifier

        Raises
        ------
        TypeError
            If ``X`` is not numeric or ``y`` has dtype object.
        ValueError
            If ``X`` is not 2-D, empty, or contains ``±inf``; or ``y`` is not
            1-D, holds NaN, or has a different length than ``X``.

        Complexity
        ----------
        Time: O(depth * f * n log n) (see ``_best_split`` for one node).
        Space: O(f * n * k) for the largest node.
        """
        X = _check_numeric(X, "DecisionTreeClassifier.fit")
        y = _check_labels(y, X.shape[0], "DecisionTreeClassifier.fit")
        self._reset_state()
        _check_n_features(self, X, reset=True)
        self._build(X, y)
        return self

    def partial_fit(self, X_chunk, y_chunk) -> "DecisionTreeClassifier":
        """Add a chunk to the buffer; retrain once ``buffer_size`` samples are buffered.

        The retrain uses the buffered samples only, then clears the buffer.
        Until the first retrain, ``predict`` raises ``NotFittedError``.

        Parameters
        ----------
        X_chunk : array-like of numbers, shape (n, n_features)
        y_chunk : array-like, shape (n,)

        Returns
        -------
        self : DecisionTreeClassifier

        Raises
        ------
        TypeError, ValueError
            As for ``fit``; also if the number of features differs from earlier
            chunks. An empty chunk (``n == 0``) is ignored.

        Complexity
        ----------
        Time: O(n) to buffer the chunk; a retrain costs the same as ``fit`` on
        the buffer. Space: O(buffer_size * n_features).
        """
        X = _check_numeric(X_chunk, "DecisionTreeClassifier.partial_fit")
        y = _check_labels(y_chunk, X.shape[0], "DecisionTreeClassifier.partial_fit")
        if X.shape[0] == 0:
            if hasattr(self, "n_features_in_"):
                _check_n_features(self, X, reset=False)
            return self
        _check_n_features(self, X, reset=not hasattr(self, "n_features_in_"))
        self._buf_X.append(X)
        self._buf_y.append(y)
        self.n_buffered_ += X.shape[0]
        self.n_seen_ += X.shape[0]
        if self.n_buffered_ >= self.buffer_size:
            self.flush()
        return self

    def flush(self) -> "DecisionTreeClassifier":
        """Retrain now on whatever is in the buffer (does nothing if it is empty)."""
        if self.n_buffered_ == 0:
            return self
        X = np.concatenate(self._buf_X, axis=0)
        y = np.concatenate(self._buf_y, axis=0)
        self._buf_X, self._buf_y, self.n_buffered_ = [], [], 0
        self._build(X, y)
        return self

    def _build(self, X: np.ndarray, y: np.ndarray) -> None:
        """Grow the tree on ``(X, y)`` with an explicit stack (no recursion limit)."""
        classes, y_idx = np.unique(y, return_inverse=True)
        n, n_feat = X.shape
        onehot = np.zeros((n, classes.size))
        onehot[np.arange(n), y_idx] = 1.0
        n_try = self._n_candidate_features(n_feat)

        feature, thr, left, right, value, nan_left = [], [], [], [], [], []
        importances = np.zeros(n_feat)

        def new_node() -> int:
            feature.append(-1)
            thr.append(np.nan)
            left.append(-1)
            right.append(-1)
            value.append(None)
            nan_left.append(False)
            return len(feature) - 1

        stack = [(new_node(), np.arange(n), 0)]
        while stack:
            node, idx, depth = stack.pop()
            counts = onehot[idx].sum(axis=0)
            value[node] = counts
            imp = float(_impurity(counts, self.criterion))
            if (imp <= _EPS or idx.size < self.min_samples_split
                    or (self.max_depth is not None and depth >= self.max_depth)):
                continue
            if n_try >= n_feat:
                candidates = np.arange(n_feat)
            else:
                candidates = np.sort(self._rng.choice(n_feat, size=n_try, replace=False))
            found = _best_split(X[idx], onehot[idx], candidates, self.criterion)
            if found is None:
                continue
            f, t, nl, _ = found
            col = X[idx, f]
            go_left = np.where(np.isnan(col), nl, col <= t)
            li, ri = idx[go_left], idx[~go_left]
            lc, rc = onehot[li].sum(axis=0), onehot[ri].sum(axis=0)
            importances[f] += (idx.size * imp
                               - li.size * float(_impurity(lc, self.criterion))
                               - ri.size * float(_impurity(rc, self.criterion)))
            feature[node], thr[node], nan_left[node] = f, t, nl
            left[node], right[node] = new_node(), new_node()
            stack.append((right[node], ri, depth + 1))
            stack.append((left[node], li, depth + 1))

        total = importances.sum()
        self.feature_ = np.array(feature, dtype=np.intp)
        self.threshold_ = np.array(thr, dtype=float)
        self.children_left_ = np.array(left, dtype=np.intp)
        self.children_right_ = np.array(right, dtype=np.intp)
        self.nan_left_ = np.array(nan_left, dtype=bool)
        self.value_ = np.vstack(value)
        self.classes_ = classes
        self.feature_importances_ = importances / total if total > 0 else importances
        self.n_retrains_ += 1

    # -- inference ------------------------------------------------------------
    @property
    def n_nodes_(self) -> int:
        """Number of nodes (internal nodes plus leaves) in the current tree."""
        self._require_trained("n_nodes_")
        return int(self.feature_.size)

    def get_depth(self) -> int:
        """Depth of the current tree (a single leaf has depth 0).

        Raises
        ------
        NotFittedError
            If no tree has been trained yet.
        """
        self._require_trained("get_depth")
        nodes, depth = np.array([0]), 0
        while True:                                       # one pass per tree level
            internal = nodes[self.feature_[nodes] >= 0]
            if internal.size == 0:
                return depth
            nodes = np.concatenate([self.children_left_[internal], self.children_right_[internal]])
            depth += 1

    def _leaf_values(self, X: np.ndarray) -> np.ndarray:
        """Class counts of the leaf every row of ``X`` ends in (loops over levels, not rows)."""
        n = X.shape[0]
        node = np.zeros(n, dtype=np.intp)
        rows = np.arange(n)
        while True:
            f = self.feature_[node]
            active = f >= 0
            if not active.any():
                break
            x = X[rows, np.where(active, f, 0)]
            go_left = np.where(np.isnan(x), self.nan_left_[node], x <= self.threshold_[node])
            step = np.where(go_left, self.children_left_[node], self.children_right_[node])
            node = np.where(active, step, node)
        return self.value_[node]

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities from the class counts of the leaf each sample reaches.

        Parameters
        ----------
        X : array-like of numbers, shape (n_samples, n_features_in_)

        Returns
        -------
        proba : ndarray, shape (n_samples, n_classes)
            Columns follow ``classes_``; every row sums to 1.

        Raises
        ------
        NotFittedError
            If no tree has been trained yet.
        TypeError, ValueError
            If ``X`` is not numeric, not 2-D, contains ``±inf`` or has a
            different number of columns than the training data.

        Complexity
        ----------
        Time: O(n * depth), vectorised over samples. Space: O(n).
        """
        self._require_trained("predict_proba")
        X = _check_numeric(X, "DecisionTreeClassifier.predict_proba")
        _check_n_features(self, X, reset=False)
        v = self._leaf_values(X)
        return v / v.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """Predict class labels with the most recently trained tree.

        Parameters
        ----------
        X : array-like of numbers, shape (n_samples, n_features_in_)

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Labels from ``classes_``; equal leaf counts go to the smallest label.

        Raises
        ------
        NotFittedError
            If no tree has been trained yet.
        TypeError, ValueError
            As for ``predict_proba``.

        Complexity
        ----------
        Time: O(n * depth), vectorised over samples. Space: O(n).
        """
        self._require_trained("predict")
        X = _check_numeric(X, "DecisionTreeClassifier.predict")
        _check_n_features(self, X, reset=False)
        return self.classes_[np.argmax(self._leaf_values(X), axis=1)]

    def score(self, X, y) -> float:
        """Mean accuracy of ``predict(X)`` against ``y`` (``nan`` for zero samples)."""
        X = _check_numeric(X, "DecisionTreeClassifier.score")
        y = _check_labels(y, X.shape[0], "DecisionTreeClassifier.score")
        if X.shape[0] == 0:
            return float("nan")
        return float(np.mean(self.predict(X) == y))

    def memory_bytes(self) -> int:
        """Approximate bytes held by the tree arrays and the buffer."""
        total = sum(a.nbytes for a in self._buf_X) + sum(a.nbytes for a in self._buf_y)
        for name in _LEARNED:
            value = getattr(self, name, None)
            if isinstance(value, np.ndarray):
                total += value.nbytes
        return int(total)