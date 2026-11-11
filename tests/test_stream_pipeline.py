"""
tests/test_stream_pipeline.py
Unit tests for numcompute_stream.pipeline

Covers: Pipeline.partial_fit and FeatureUnion.partial_fit. Checked with small
test doubles (to see exactly what each step receives) and with the real
streaming scaler, imputer, one-hot encoder and decision tree: learn-then-
transform order, y forwarding, steps without partial_fit, empty chunks, errors
in the middle of a chain, slicing, and that everything inherited from the A1
pipeline still works.

Run with: python -m pytest tests/test_stream_pipeline.py -v   (from the repo root)
"""

import warnings

import numpy as np
import pytest

from numcompute import pipeline as a1_pipeline
from numcompute import preprocessing as a1
from numcompute.preprocessing import NotFittedError
from numcompute_stream.pipeline import FeatureUnion, Pipeline
from numcompute_stream.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


@pytest.fixture(autouse=True)
def _no_stray_warnings():
    """Fail any test that leaks a RuntimeWarning (e.g. 'invalid value in divide')."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


# ---------- test doubles (plain classes, no base class) ----------
class Doubler:
    """Transformer that records every chunk it learned from and doubles its input."""

    def __init__(self):
        self.chunks = []
        self.seen_y = []

    def fit(self, X, y=None):                      # the A1 contract needs fit; the tests use partial_fit
        return self

    def partial_fit(self, X, y=None):
        self.chunks.append(np.asarray(X, dtype=float))
        self.seen_y.append(y)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * 2


class Recorder:
    """Model that records every chunk and target it learned from."""

    def __init__(self):
        self.chunks = []
        self.targets = []

    def fit(self, X, y=None):
        return self

    def partial_fit(self, X, y=None):
        self.chunks.append(np.asarray(X, dtype=float))
        self.targets.append(y)
        return self

    def predict(self, X):
        return np.zeros(len(X))


class NoPartial:
    """Transformer with fit/transform only."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)


class OneArgument:
    """Transformer whose fit and partial_fit take only X (no y parameter)."""

    def __init__(self):
        self.calls = 0

    def fit(self, X):
        return self

    def partial_fit(self, X):
        self.calls += 1
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)


def _data(n=120, f=3, seed=0):
    """Features and a label that depends on the first feature."""
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=4.0, scale=2.0, size=(n, f))
    y = (X[:, 0] > 4.0).astype(int)
    return X, y


# ---------- Pipeline: chunk flow ----------
def test_partial_fit_returns_self_and_feeds_each_step_the_output_of_the_previous_one():
    first, last = Doubler(), Recorder()
    pipe = Pipeline([("double", first), ("model", last)])
    chunk = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert pipe.partial_fit(chunk, [0, 1]) is pipe
    np.testing.assert_array_equal(first.chunks[0], chunk)             # the first step sees the raw chunk
    np.testing.assert_array_equal(last.chunks[0], chunk * 2)          # the model sees the doubled chunk
    assert last.targets[0] == [0, 1]


def test_each_chunk_reaches_every_step_once_and_in_order():
    first, last = Doubler(), Recorder()
    pipe = Pipeline([("double", first), ("model", last)])
    chunks = np.array_split(np.arange(12.0).reshape(6, 2), 3)
    for c in chunks:
        pipe.partial_fit(c)
    assert len(first.chunks) == len(last.chunks) == 3
    for seen, original in zip(last.chunks, chunks):
        np.testing.assert_array_equal(seen, original * 2)


def test_y_is_forwarded_to_every_step_only_when_given():
    first, last = Doubler(), Recorder()
    pipe = Pipeline([("double", first), ("model", last)])
    pipe.partial_fit([[1.0]])
    pipe.partial_fit([[2.0]], [1])
    assert first.seen_y == [None, [1]] and last.targets == [None, [1]]


def test_steps_that_take_only_x_work_when_no_y_is_given():
    step = OneArgument()
    pipe = Pipeline([("one", step), ("model", Recorder())])
    pipe.partial_fit([[1.0], [2.0]])                                   # y is not passed to partial_fit(X)
    assert step.calls == 1
    union = FeatureUnion([("one", OneArgument())])
    union.partial_fit([[1.0]])
    assert union.named_transformers["one"].calls == 1


def test_a_transformer_learns_from_the_chunk_before_it_transforms_it():
    X = np.array([[0.0], [10.0]])
    pipe = Pipeline([("scale", StandardScaler()), ("model", Recorder())])
    pipe.partial_fit(X)
    reference = a1.StandardScaler().fit(X).transform(X)                # scaled with THIS chunk's own mean and std
    np.testing.assert_allclose(pipe["model"].chunks[0], reference)


def test_real_scaler_and_tree_match_doing_the_steps_by_hand():
    X, y = _data()
    pipe = Pipeline([("scale", StandardScaler()),
                     ("model", DecisionTreeClassifier(max_depth=3, buffer_size=40))])
    scaler, tree = StandardScaler(), DecisionTreeClassifier(max_depth=3, buffer_size=40)
    for Xc, yc in zip(np.array_split(X, 6), np.array_split(y, 6)):
        pipe.partial_fit(Xc, yc)
        scaler.partial_fit(Xc)
        tree.partial_fit(scaler.transform(Xc), yc)
    probe = X[:25]
    np.testing.assert_array_equal(pipe.predict(probe), tree.predict(scaler.transform(probe)))
    np.testing.assert_allclose(pipe["scale"].mean_, a1.StandardScaler().fit(X).mean_)    # chunked equals batch


def test_predict_follows_the_buffered_retraining_of_the_tree():
    X, y = _data(60)
    pipe = Pipeline([("scale", StandardScaler()),
                     ("model", DecisionTreeClassifier(max_depth=3, buffer_size=50))])
    pipe.partial_fit(X[:20], y[:20])
    with pytest.raises(NotFittedError):
        pipe.predict(X[:5])                                            # the buffer is not full yet
    pipe.partial_fit(X[20:], y[20:])                                   # 60 >= 50: the tree retrains
    assert pipe.predict(X[:5]).shape == (5,)
    assert pipe["model"].n_buffered_ == 0


# ---------- Pipeline: steps without partial_fit, empty chunks, errors ----------
def test_a_step_without_partial_fit_raises_before_any_step_learns():
    first = Doubler()
    pipe = Pipeline([("double", first), ("plain", NoPartial()), ("model", Recorder())])
    with pytest.raises(TypeError, match=r"missing on: 'plain' \(NoPartial\)"):
        pipe.partial_fit([[1.0]], [0])
    assert first.chunks == []                                          # nothing learned the chunk


def test_the_last_step_needs_partial_fit_too_and_all_missing_steps_are_named():
    pipe = Pipeline([("a", NoPartial()), ("b", NoPartial()), ("scale", NoPartial())])
    with pytest.raises(TypeError, match="'a'.*'b'.*'scale'"):
        pipe.partial_fit([[1.0]])


def test_an_empty_chunk_is_ignored_by_every_step():
    first, last = Doubler(), Recorder()
    pipe = Pipeline([("double", first), ("model", last)])
    assert pipe.partial_fit(np.empty((0, 2)), np.empty(0)) is pipe
    assert first.chunks == [] and last.chunks == []
    real = Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier())])
    real.partial_fit(np.empty((0, 3)), np.empty(0))
    assert not hasattr(real["scale"], "n_features_in_")                # an empty first chunk decides nothing


def test_bad_chunk_shapes_raise_informative_errors():
    pipe = Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier(buffer_size=10))])
    with pytest.raises(ValueError, match="2-D"):
        pipe.partial_fit([1.0, 2.0, 3.0], [0, 1, 0])
    pipe.partial_fit(np.zeros((4, 3)), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="fitted with 3 features"):
        pipe.partial_fit(np.zeros((4, 5)), [0, 1, 0, 1])


def test_a_failure_in_a_later_step_leaves_earlier_steps_updated():
    first = Doubler()
    pipe = Pipeline([("double", first), ("model", DecisionTreeClassifier())])
    with pytest.raises(ValueError):
        pipe.partial_fit(np.zeros((4, 2)), [0, 1])                     # 4 rows but only 2 labels
    assert len(first.chunks) == 1                                      # the documented behaviour


def test_the_input_chunk_is_not_modified():
    X, y = _data(30)
    before = X.copy()
    Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier())]).partial_fit(X, y)
    np.testing.assert_array_equal(X, before)


def test_non_contiguous_chunks_give_the_same_result_as_contiguous_ones():
    X, y = _data(80, 6)
    view = X[::2, ::2]

    def run(chunk):
        pipe = Pipeline([("scale", StandardScaler()),
                         ("model", DecisionTreeClassifier(buffer_size=10, max_depth=3))])
        return pipe.partial_fit(chunk, y[::2]).predict(chunk)

    np.testing.assert_array_equal(run(view), run(np.ascontiguousarray(view)))


# ---------- Pipeline: everything inherited from A1 ----------
def test_pipeline_is_an_a1_pipeline_and_keeps_its_api():
    pipe = Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier(max_depth=2))])
    assert isinstance(pipe, a1_pipeline.Pipeline) and len(pipe) == 2
    assert list(pipe.named_steps) == ["scale", "model"]
    assert pipe["scale"] is pipe[0] and isinstance(pipe[1], DecisionTreeClassifier)
    assert repr(pipe).startswith("Pipeline([('scale', StandardScaler(")


def test_fit_and_predict_still_work_for_whole_data():
    X, y = _data(100)
    pipe = Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier(max_depth=3))]).fit(X, y)
    assert (pipe.predict(X) == y).mean() > 0.9


def test_construction_errors_are_the_a1_ones():
    with pytest.raises(ValueError, match="at least one"):
        Pipeline([])
    with pytest.raises(ValueError, match="unique"):
        Pipeline([("a", Doubler()), ("a", Recorder())])
    with pytest.raises(TypeError, match="must implement fit and transform"):
        Pipeline([("model", Recorder()), ("last", Recorder())])


def test_slicing_returns_a_pipeline_that_can_still_partial_fit():
    pipe = Pipeline([("double", Doubler()), ("scale", StandardScaler()), ("model", DecisionTreeClassifier())])
    head = pipe[:2]
    assert type(head) is Pipeline and len(head) == 2
    head.partial_fit(np.array([[1.0], [3.0]]))
    assert pipe["scale"].mean_.tolist() == [4.0]                       # the same step objects, now updated
    with pytest.raises(TypeError, match="str, int or slice"):
        pipe[1.5]


# ---------- Pipeline with imputer and encoder ----------
def test_a_pipeline_of_imputer_scaler_and_tree_handles_missing_values_in_every_chunk():
    X, y = _data(100)
    X[np.random.default_rng(1).random(X.shape) < 0.1] = np.nan
    pipe = Pipeline([("fill", SimpleImputer("mean")), ("scale", StandardScaler()),
                     ("model", DecisionTreeClassifier(max_depth=3, buffer_size=30))])
    for Xc, yc in zip(np.array_split(X, 5), np.array_split(y, 5)):
        pipe.partial_fit(Xc, yc)
    assert not np.isnan(pipe[:2].transform(X)).any()
    assert pipe.predict(X).shape == (100,)


def test_a_pipeline_that_ends_in_an_encoder_grows_with_a_late_category_and_warns():
    pipe = Pipeline([("onehot", OneHotEncoder())])
    pipe.partial_fit(np.array([["a"], ["b"]]))
    with pytest.warns(UserWarning, match="output grew from 2 to 3"):
        pipe.partial_fit(np.array([["c"]]))
    assert pipe.transform(np.array([["c"]])).tolist() == [[0.0, 0.0, 1.0]]


def test_a_model_fixed_to_the_old_width_rejects_a_chunk_after_the_encoder_grows():
    pipe = Pipeline([("onehot", OneHotEncoder()), ("model", DecisionTreeClassifier(buffer_size=100))])
    pipe.partial_fit(np.array([["a"], ["b"]]), [0, 1])
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="fitted with 2 features"):
            pipe.partial_fit(np.array([["c"], ["a"]]), [1, 0])


# ---------- FeatureUnion ----------
def test_feature_union_members_all_learn_every_chunk_and_outputs_are_stacked():
    a, b = Doubler(), Doubler()
    union = FeatureUnion([("a", a), ("b", b)])
    chunk = np.array([[1.0], [2.0]])
    assert union.partial_fit(chunk, [0, 1]) is union
    np.testing.assert_array_equal(a.chunks[0], chunk)
    np.testing.assert_array_equal(b.chunks[0], chunk)
    assert a.seen_y == b.seen_y == [[0, 1]]
    np.testing.assert_array_equal(union.transform(chunk), [[2.0, 2.0], [4.0, 4.0]])


def test_feature_union_chunks_equal_a1_batch_fit_for_the_real_transformers():
    X, _ = _data(90, 2)
    X[np.random.default_rng(2).random(X.shape) < 0.1] = np.nan
    union = FeatureUnion([("z", StandardScaler()), ("fill", SimpleImputer("median"))])
    for chunk in np.array_split(X, 6):
        union.partial_fit(chunk)
    reference = a1_pipeline.FeatureUnion([("z", a1.StandardScaler()), ("fill", a1.SimpleImputer("median"))])
    np.testing.assert_allclose(union.transform(X), reference.fit(X).transform(X), equal_nan=True)


def test_feature_union_member_without_partial_fit_raises_before_any_member_learns():
    first = Doubler()
    union = FeatureUnion([("a", first), ("plain", NoPartial())])
    with pytest.raises(TypeError, match=r"missing on: 'plain' \(NoPartial\)"):
        union.partial_fit([[1.0]])
    assert first.chunks == []


def test_feature_union_ignores_empty_chunks_and_y_is_optional():
    member = Doubler()
    union = FeatureUnion([("a", member)])
    union.partial_fit(np.empty((0, 2)))
    assert member.chunks == []
    union.partial_fit([[1.0]])
    assert member.seen_y == [None]


def test_feature_union_is_an_a1_union_and_keeps_its_checks():
    union = FeatureUnion([("a", Doubler())])
    assert isinstance(union, a1_pipeline.FeatureUnion) and list(union.named_transformers) == ["a"]
    with pytest.raises(ValueError, match="unique"):
        FeatureUnion([("a", Doubler()), ("a", Doubler())])


# ---------- nesting ----------
def test_a_pipeline_with_a_feature_union_step_then_a_tree_learns_chunk_by_chunk():
    X, y = _data(120)
    pipe = Pipeline([
        ("features", FeatureUnion([("z", StandardScaler()), ("fill", SimpleImputer())])),
        ("model", DecisionTreeClassifier(max_depth=3, buffer_size=40)),
    ])
    for Xc, yc in zip(np.array_split(X, 6), np.array_split(y, 6)):
        pipe.partial_fit(Xc, yc)
    assert pipe["model"].n_features_in_ == 6                           # 3 scaled + 3 filled columns
    assert (pipe.predict(X) == y).mean() > 0.8


def test_a_union_of_pipelines_can_partial_fit_when_every_branch_can():
    branch_a = Pipeline([("fill", SimpleImputer()), ("scale", StandardScaler())])
    branch_b = Pipeline([("scale", StandardScaler())])
    union = FeatureUnion([("a", branch_a), ("b", branch_b)])
    X = np.array([[1.0], [3.0]])
    union.partial_fit(X)
    np.testing.assert_allclose(union.transform(X), [[-1.0, -1.0], [1.0, 1.0]])