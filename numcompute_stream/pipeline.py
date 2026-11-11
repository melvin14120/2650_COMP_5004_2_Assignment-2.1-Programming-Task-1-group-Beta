"""
pipeline.py
Pipeline and FeatureUnion that learn chunk by chunk.

Provides
--------
- Pipeline(steps)
    .partial_fit(X_chunk, y_chunk=None)   learn from one chunk, through every step
- FeatureUnion(transformers)
    .partial_fit(X_chunk, y_chunk=None)   let every member learn from one chunk

Design notes
------------
Both classes SUBCLASS the A1 ones, so construction and its checks,
``named_steps``, indexing, ``fit``, ``transform``, ``fit_transform`` and
``predict`` are the A1 code, unchanged. Only ``partial_fit`` is new.

One ``Pipeline.partial_fit`` call handles one chunk, in the order a stream
needs: each transformer first LEARNS from the chunk (its own ``partial_fit``),
then TRANSFORMS the chunk with its updated statistics and hands the result to
the next step. The last step (usually the model) only learns. A model such as
``DecisionTreeClassifier`` buffers the chunk and retrains when its buffer is
full, so the pipeline needs no extra logic for that.

Before anything is learned, every step is checked for a ``partial_fit``
method, so a missing method raises ``TypeError`` before any step has seen
the chunk.

Missing values and edge cases
-----------------------------
- An empty chunk (zero rows) is ignored by the whole pipeline.
- If a step raises while learning (for example a chunk with the wrong number
  of columns, or a ``y`` of the wrong length), the steps BEFORE it have already
  learned the chunk and keep it. This is the same as ``FeatureUnion.fit`` in A1.
- A step that changes its output width mid-stream (``OneHotEncoder`` meeting a
  new category warns about it) makes the next step see a different number of
  columns, and a model fixed to the old width raises ``ValueError``.
- Slicing a pipeline (``pipe[:1]``) returns a pipeline of this module, so
  ``partial_fit`` stays available.

Axis convention
---------------
``X_chunk`` has shape (n_samples, n_features): rows are samples, columns are
features. ``y_chunk`` has shape (n_samples,) and goes to every step when given.
"""

from __future__ import annotations

import numpy as np

from numcompute import pipeline as _a1

__all__ = ["Pipeline", "FeatureUnion"]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _partial_fit(step, X, y):
    """Call ``step.partial_fit``, passing ``y`` only when it is given."""
    return step.partial_fit(X) if y is None else step.partial_fit(X, y)


def _require_partial_fit(steps, owner: str) -> None:
    """Raise ``TypeError`` naming every ``(name, step)`` that has no callable ``partial_fit``."""
    missing = [
        f"{name!r} ({type(step).__name__})"
        for name, step in steps
        if not callable(getattr(step, "partial_fit", None))
    ]
    if missing:
        raise TypeError(
            f"{owner}.partial_fit needs every step to implement partial_fit; missing on: {', '.join(missing)}"
        )


def _is_empty(X) -> bool:
    """True if ``X`` is an array-like with zero rows."""
    shape = np.shape(X)
    return len(shape) > 0 and shape[0] == 0


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------
class Pipeline(_a1.Pipeline):
    """A1 ``Pipeline`` that can also learn one chunk at a time.

    Parameters
    ----------
    steps : list of (str, object)
        As for the A1 ``Pipeline``. To use ``partial_fit`` every step needs a
        ``partial_fit`` method.

    Examples
    --------
    >>> from numcompute_stream.preprocessing import StandardScaler
    >>> from numcompute_stream.tree import DecisionTreeClassifier
    >>> pipe = Pipeline([("scale", StandardScaler()),
    ...                  ("model", DecisionTreeClassifier(max_depth=2, buffer_size=4))])
    >>> _ = pipe.partial_fit([[0.0], [1.0]], [0, 1])
    >>> _ = pipe.partial_fit([[0.0], [1.0]], [0, 1])
    >>> pipe.predict([[0.0], [1.0]]).tolist()
    [0, 1]
    """

    def __getitem__(self, key):
        """``pipe['name']`` or ``pipe[i]`` returns a step; ``pipe[a:b]`` a sub-Pipeline of this module."""
        if isinstance(key, slice):
            return type(self)(self.steps[key])
        return super().__getitem__(key)

    def partial_fit(self, X, y=None) -> "Pipeline":
        """Learn from one chunk: every transformer learns then transforms, the last step learns.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            One chunk. A chunk with zero rows is ignored.
        y : array-like, shape (n_samples,), optional
            Targets, passed to every step's ``partial_fit`` when not ``None``.

        Returns
        -------
        self : Pipeline

        Raises
        ------
        TypeError
            If any step has no ``partial_fit``; raised before any step learns.
        Exception
            Whatever a step raises while learning (see the module notes: steps
            before it keep what they learned).

        Complexity
        ----------
        Sum of the steps' ``partial_fit`` costs plus the ``transform`` cost of
        every step but the last, on one chunk.
        """
        _require_partial_fit(self.steps, "Pipeline")
        if _is_empty(X):
            return self
        Xt = X
        for _, step in self.steps[:-1]:
            _partial_fit(step, Xt, y)
            Xt = step.transform(Xt)
        _partial_fit(self.steps[-1][1], Xt, y)
        return self


# ----------------------------------------------------------------------
# FeatureUnion
# ----------------------------------------------------------------------
class FeatureUnion(_a1.FeatureUnion):
    """A1 ``FeatureUnion`` whose members can learn one chunk at a time.

    Parameters
    ----------
    transformers : list of (str, object)
        As for the A1 ``FeatureUnion``. To use ``partial_fit`` every member
        needs a ``partial_fit`` method.

    Examples
    --------
    >>> import numpy as np
    >>> from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
    >>> union = FeatureUnion([("z", StandardScaler()), ("fill", SimpleImputer())])
    >>> _ = union.partial_fit([[1.0], [np.nan]]); _ = union.partial_fit([[3.0]])
    >>> union.transform([[2.0], [np.nan]]).tolist()
    [[0.0, 2.0], [nan, 2.0]]
    """

    def partial_fit(self, X, y=None) -> "FeatureUnion":
        """Let every member learn from the same chunk.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            One chunk. A chunk with zero rows is ignored.
        y : array-like, shape (n_samples,), optional
            Passed to every member's ``partial_fit`` when not ``None``.

        Returns
        -------
        self : FeatureUnion

        Raises
        ------
        TypeError
            If any member has no ``partial_fit``; raised before any member learns.
        Exception
            Whatever a member raises while learning (members before it keep
            what they learned).

        Complexity
        ----------
        Sum of the members' ``partial_fit`` costs on one chunk.
        """
        _require_partial_fit(self.transformers, "FeatureUnion")
        if _is_empty(X):
            return self
        for _, member in self.transformers:
            _partial_fit(member, X, y)
        return self