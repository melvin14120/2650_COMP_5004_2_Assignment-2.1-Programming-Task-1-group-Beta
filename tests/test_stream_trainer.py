"""
tests/test_stream_trainer.py
Unit tests for numcompute_stream.stream

Covers: StreamTrainer (score_chunk, fit_chunk, process_chunk, run, history,
columns, reset) and iter_chunks. Checked against a hand-written "test, then
train" loop with the real tree and scaler: every logged number (chunk,
cumulative and rolling accuracy, extra metrics, retrain flags, memory) must
equal what the loop produces. Also: the model that cannot predict yet, empty
chunks, bad shapes, a model or metric that raises, models without
``n_retrains_``, string labels, non-contiguous input, and chunking.

Run with: python -m pytest tests/test_stream_trainer.py -v   (from the repo root)
"""

import warnings

import numpy as np
import pytest

from numcompute import metrics as a1
from numcompute.preprocessing import NotFittedError
from numcompute_stream.metrics import StreamingAccuracy, StreamingF1
from numcompute_stream.pipeline import FeatureUnion, Pipeline
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.stream import (
    StreamTrainer,
    _array_bytes,
    _memory_bytes,
    _retrain_count,
    iter_chunks,
)
from numcompute_stream.tree import DecisionTreeClassifier


@pytest.fixture(autouse=True)
def _no_stray_warnings():
    """Fail any test that leaks a RuntimeWarning (e.g. 'invalid value in divide')."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


# ---------- test doubles and data ----------
class Echo:
    """Model with no ``n_retrains_``: predicts the first column as an integer."""

    def __init__(self):
        self.chunks = []

    def partial_fit(self, X, y):
        self.chunks.append((np.array(X), np.array(y)))
        return self

    def predict(self, X):
        return np.asarray(X)[:, 0].astype(int)


class Raiser:
    """Metric whose ``update`` always fails."""

    def update(self, y_true, y_pred):
        raise ValueError("this metric cannot take the chunk")

    def result(self):
        return 0.0

    def reset(self):
        return self


def _data(n=300, f=3, seed=0, noise=0.15):
    """Features and a label that mostly follows the first feature."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    y = (X[:, 0] + noise * rng.normal(size=n) > 0).astype(int)
    return X, y


def _tree(buffer_size=60, **kw):
    return DecisionTreeClassifier(max_depth=3, buffer_size=buffer_size, **kw)


def _manual(X, y, chunk, buffer_size=60):
    """The test-then-train loop written by hand: a list of (y_true, y_pred) per scored chunk."""
    tree = _tree(buffer_size)
    scored = []
    for Xc, yc in iter_chunks(X, y, chunk):
        try:
            scored.append((yc, tree.predict(Xc)))
        except NotFittedError:
            scored.append((yc, None))
        tree.partial_fit(Xc, yc)
    return scored, tree


# ---------- construction ----------
def test_a_model_without_partial_fit_or_predict_is_rejected_with_the_missing_name():
    class OnlyPredict:
        def predict(self, X):
            return X

    class OnlyFit:
        def partial_fit(self, X, y):
            return self

    with pytest.raises(TypeError, match="must implement partial_fit; OnlyPredict"):
        StreamTrainer(OnlyPredict())
    with pytest.raises(TypeError, match="must implement predict; OnlyFit"):
        StreamTrainer(OnlyFit())


def test_metric_problems_are_reported_at_construction():
    with pytest.raises(TypeError, match=r"metric 'f1' \(object\) lacks: update, result, reset"):
        StreamTrainer(_tree(), metrics={"f1": object()})
    with pytest.raises(TypeError, match="names must be strings"):
        StreamTrainer(_tree(), metrics={1: StreamingAccuracy()})
    for reserved in ("chunk_accuracy", "memory_bytes", "scored", "rolling_accuracy"):
        with pytest.raises(ValueError, match="one of the log's own columns"):
            StreamTrainer(_tree(), metrics={reserved: StreamingAccuracy()})


def test_window_is_validated_and_creates_the_rolling_tracker():
    with pytest.raises(ValueError, match="window must be >= 1"):
        StreamTrainer(_tree(), window=0)
    with pytest.raises(TypeError, match="window must be an integer"):
        StreamTrainer(_tree(), window=2.5)
    assert StreamTrainer(_tree()).rolling_accuracy_ is None
    assert StreamTrainer(_tree(), window=40).rolling_accuracy_.window == 40


def test_repr_names_the_model_metrics_and_window():
    trainer = StreamTrainer(_tree(), metrics={"f1": StreamingF1()}, window=50)
    assert repr(trainer) == "StreamTrainer(model=DecisionTreeClassifier, metrics=['f1'], window=50)"


# ---------- test, then train ----------
def test_a_model_that_cannot_predict_yet_is_logged_as_not_scored():
    X, y = _data(40)
    trainer = StreamTrainer(_tree(buffer_size=100))
    row = trainer.process_chunk(X, y)
    assert row["scored"] is False and np.isnan(row["chunk_accuracy"]) and np.isnan(row["cumulative_accuracy"])
    assert row["fitted"] is True and row["n_seen"] == 40 and row["retrained"] is False
    assert trainer.last_true_ is None and trainer.last_pred_ is None
    assert trainer.accuracy_.n_samples_ == 0


def test_each_chunk_is_scored_by_the_model_as_it_was_before_that_chunk():
    X, y = _data(300)
    trainer = StreamTrainer(_tree()).run(iter_chunks(X, y, 30))
    scored, _ = _manual(X, y, 30)
    for row, (truth, pred) in zip(trainer.log_, scored):
        if pred is None:
            assert row["scored"] is False
        else:
            assert row["scored"] is True and row["chunk_accuracy"] == pytest.approx(np.mean(truth == pred))


def test_cumulative_accuracy_equals_accuracy_over_all_scored_samples():
    X, y = _data(300)
    trainer = StreamTrainer(_tree()).run(iter_chunks(X, y, 30))
    scored = [(t, p) for t, p in _manual(X, y, 30)[0] if p is not None]
    truth = np.concatenate([t for t, _ in scored])
    pred = np.concatenate([p for _, p in scored])
    assert trainer.log_[-1]["cumulative_accuracy"] == pytest.approx(a1.accuracy(truth, pred))
    assert trainer.accuracy_.n_samples_ == truth.size
    running = [np.mean(np.concatenate([t for t, _ in scored[:k]]) == np.concatenate([p for _, p in scored[:k]]))
               for k in range(1, len(scored) + 1)]
    np.testing.assert_allclose(trainer.history("cumulative_accuracy")[-len(scored):], running)


def test_rolling_accuracy_covers_only_the_latest_window_samples():
    X, y = _data(300)
    trainer = StreamTrainer(_tree(), window=50).run(iter_chunks(X, y, 30))
    scored = [(t, p) for t, p in _manual(X, y, 30)[0] if p is not None]
    truth = np.concatenate([t for t, _ in scored])[-50:]
    pred = np.concatenate([p for _, p in scored])[-50:]
    assert trainer.log_[-1]["rolling_accuracy"] == pytest.approx(a1.accuracy(truth, pred))
    assert trainer.log_[-1]["rolling_accuracy"] != trainer.log_[-1]["cumulative_accuracy"]


def test_extra_metrics_are_updated_with_every_scored_chunk_and_logged_by_name():
    X, y = _data(300)
    trainer = StreamTrainer(_tree(), metrics={"f1": StreamingF1(), "acc": StreamingAccuracy(window=60)})
    trainer.run(iter_chunks(X, y, 30))
    scored = [(t, p) for t, p in _manual(X, y, 30)[0] if p is not None]
    truth = np.concatenate([t for t, _ in scored])
    pred = np.concatenate([p for _, p in scored])
    assert trainer.log_[-1]["f1"] == pytest.approx(a1.f1(truth, pred))
    assert trainer.log_[-1]["acc"] == pytest.approx(a1.accuracy(truth[-60:], pred[-60:]))
    assert trainer.metrics["f1"].n_samples_ == truth.size


def test_the_trainer_matches_a_hand_written_test_then_train_loop_with_a_pipeline():
    X, y = _data(240)
    pipe = Pipeline([("scale", StandardScaler()), ("model", _tree(buffer_size=50))])
    trainer = StreamTrainer(pipe).run(iter_chunks(X, y, 20))
    by_hand = Pipeline([("scale", StandardScaler()), ("model", _tree(buffer_size=50))])
    correct, total = 0, 0
    for Xc, yc in iter_chunks(X, y, 20):
        try:
            pred = by_hand.predict(Xc)
            correct += int((pred == yc).sum())
            total += yc.size
        except NotFittedError:
            pass
        by_hand.partial_fit(Xc, yc)
    assert trainer.accuracy_.n_samples_ == total
    assert trainer.log_[-1]["cumulative_accuracy"] == pytest.approx(correct / total)
    np.testing.assert_array_equal(pipe.predict(X), by_hand.predict(X))


# ---------- score_chunk, fit_chunk, process_chunk, run ----------
def test_return_values_of_the_methods():
    X, y = _data(80)
    trainer = StreamTrainer(_tree(buffer_size=40))
    assert trainer.score_chunk(X[:40], y[:40]) is None                 # no tree yet
    assert trainer.fit_chunk(X[:40], y[:40]) is trainer
    pred = trainer.score_chunk(X[40:], y[40:])
    assert isinstance(pred, np.ndarray) and pred.shape == (40,)
    assert trainer.last_pred_ is pred and np.array_equal(trainer.last_true_, y[40:])
    row = trainer.process_chunk(X[40:], y[40:])
    row["chunk_accuracy"] = -1.0                                       # the returned row is a copy
    assert trainer.log_[-1]["chunk_accuracy"] != -1.0
    assert trainer.run([]) is trainer


def test_score_then_fit_complete_one_row():
    X, y = _data(80)
    trainer = StreamTrainer(_tree(buffer_size=40))
    trainer.process_chunk(X[:40], y[:40])
    trainer.score_chunk(X[40:], y[40:])
    assert len(trainer.log_) == 2 and trainer.log_[1]["fitted"] is False
    assert np.isnan(trainer.log_[1]["fit_seconds"])
    trainer.fit_chunk(X[40:], y[40:])
    assert len(trainer.log_) == 2 and trainer.log_[1]["fitted"] is True
    assert trainer.log_[1]["scored"] is True and trainer.log_[1]["n_seen"] == 80


def test_fit_chunk_alone_makes_its_own_row_and_two_score_chunks_make_two():
    X, y = _data(60)
    trainer = StreamTrainer(_tree(buffer_size=20))
    trainer.fit_chunk(X[:20], y[:20])
    trainer.fit_chunk(X[20:40], y[20:40])
    assert [r["chunk"] for r in trainer.log_] == [0, 1]
    assert all(r["scored"] is False and r["fitted"] for r in trainer.log_)
    trainer.score_chunk(X[40:50], y[40:50])
    trainer.score_chunk(X[50:], y[50:])
    assert [r["chunk"] for r in trainer.log_] == [0, 1, 2, 3]
    assert trainer.log_[2]["fitted"] is False and trainer.log_[3]["fitted"] is False


def test_a_score_on_an_untrained_model_logs_a_row_that_is_neither_scored_nor_fitted():
    X, y = _data(40)
    trainer = StreamTrainer(_tree(buffer_size=50))
    assert trainer.score_chunk(X[:20], y[:20]) is None
    assert len(trainer.log_) == 1
    assert trainer.log_[0]["scored"] is False and trainer.log_[0]["fitted"] is False


def test_a_fit_after_a_finished_chunk_starts_a_new_row_instead_of_overwriting_it():
    X, y = _data(60)
    trainer = StreamTrainer(_tree(buffer_size=50))
    trainer.process_chunk(X[:20], y[:20])
    trainer.fit_chunk(X[20:40], y[20:40])
    assert len(trainer.log_) == 2
    assert [r["n_seen"] for r in trainer.log_] == [20, 40]


def test_n_seen_counts_only_trained_rows_and_matches_the_tree():
    X, y = _data(100)
    trainer = StreamTrainer(_tree(buffer_size=30))
    trainer.score_chunk(X[:30], y[:30])
    assert trainer.n_seen_ == 0
    trainer.run(iter_chunks(X, y, 25))
    assert trainer.n_seen_ == 100 == trainer.model.n_seen_
    assert [r["n_seen"] for r in trainer.log_[-4:]] == [25, 50, 75, 100]


def test_fit_seconds_is_a_non_negative_float_for_trained_rows():
    X, y = _data(120)
    trainer = StreamTrainer(_tree(buffer_size=40)).run(iter_chunks(X, y, 40))
    times = trainer.history("fit_seconds")
    assert times.dtype == float and (times >= 0).all()


# ---------- retrains and memory ----------
def test_retrained_is_true_exactly_when_the_buffer_filled():
    X, y = _data(300)
    trainer = StreamTrainer(_tree(buffer_size=60)).run(iter_chunks(X, y, 25))
    retrained = trainer.history("retrained")
    assert retrained.tolist() == [(k + 1) % 3 == 0 for k in range(12)]   # 25, 50, 75 >= 60: every third chunk
    assert trainer.history("n_retrains")[-1] == trainer.model.n_retrains_
    assert np.all(np.diff(trainer.history("n_retrains")) == retrained[1:])


def test_retrains_of_a_pipeline_are_read_from_its_last_step():
    X, y = _data(200)
    pipe = Pipeline([("scale", StandardScaler()), ("model", _tree(buffer_size=50))])
    trainer = StreamTrainer(pipe).run(iter_chunks(X, y, 25))
    assert _retrain_count(pipe) == pipe["model"].n_retrains_ == 4
    assert trainer.history("retrained").sum() == 4


def test_a_model_without_n_retrains_still_works_and_never_reports_a_retrain():
    X, y = _data(60)
    model = Echo()
    trainer = StreamTrainer(model).run(iter_chunks(X, y, 20))
    assert not trainer.history("retrained").any() and not trainer.history("n_retrains").any()
    assert trainer.history("scored").all() and len(model.chunks) == 3
    assert _retrain_count(model) == 0


def test_memory_of_a_tree_grows_while_the_buffer_fills_and_drops_after_a_retrain():
    X, y = _data(200)
    trainer = StreamTrainer(DecisionTreeClassifier(max_depth=2, buffer_size=100)).run(iter_chunks(X, y, 30))
    memory = trainer.history("memory_bytes")
    assert memory[0] < memory[1] < memory[2]                           # 30, 60, 90 samples buffered
    assert memory[3] < memory[2]                                       # 120 >= 100: retrained, buffer cleared
    assert memory[-1] == trainer.model.memory_bytes()


def test_memory_of_a_pipeline_adds_up_its_steps():
    X, y = _data(100)
    pipe = Pipeline([("scale", StandardScaler()), ("model", _tree(buffer_size=1000))])
    StreamTrainer(pipe).process_chunk(X, y)
    assert _memory_bytes(pipe) == _array_bytes(vars(pipe["scale"])) + pipe["model"].memory_bytes()
    assert _memory_bytes(pipe) > pipe["model"].memory_bytes()


def test_memory_helpers_count_arrays_inside_lists_tuples_and_dicts():
    nested = [np.zeros(4), (np.zeros(2), {"a": np.zeros(1)}), "text", 3.0]
    assert _array_bytes(nested) == 4 * 8 + 2 * 8 + 1 * 8
    assert _array_bytes(5) == 0

    class Holder:
        def __init__(self):
            self.table = np.zeros(10)
            self.parts = [np.zeros(5), 7]
            self.count = 3

        def fit(self, X, y=None):                                      # a FeatureUnion member needs fit and transform
            return self

        def transform(self, X):
            return X

    assert _memory_bytes(Holder()) == 15 * 8
    assert _memory_bytes(3) == 0                                       # no attributes at all
    union = FeatureUnion([("a", StandardScaler().partial_fit(np.ones((2, 2)))), ("b", Holder())])
    assert _memory_bytes(union) == _memory_bytes(union.transformers[0][1]) + 15 * 8


# ---------- empty chunks, bad input, failures ----------
def test_an_empty_chunk_is_ignored_by_every_method():
    X, y = _data(40)
    trainer = StreamTrainer(_tree(buffer_size=20), window=10)
    empty_X, empty_y = np.empty((0, 3)), np.empty(0)
    assert trainer.score_chunk(empty_X, empty_y) is None
    assert trainer.fit_chunk(empty_X, empty_y) is trainer
    assert trainer.process_chunk(empty_X, empty_y) is None
    assert trainer.run([(empty_X, empty_y)]) is trainer
    assert trainer.log_ == [] and trainer.n_seen_ == 0 and trainer.model.n_seen_ == 0
    trainer.process_chunk(X, y)
    assert trainer.process_chunk(empty_X, empty_y) is None             # does not copy the previous row
    assert len(trainer.log_) == 1


def test_bad_shapes_raise_informative_errors_in_every_method_and_change_nothing():
    trainer = StreamTrainer(_tree(buffer_size=20))
    good_X, good_y = np.zeros((4, 3)), np.array([0, 1, 0, 1])
    for method in (trainer.score_chunk, trainer.fit_chunk, trainer.process_chunk):
        with pytest.raises(ValueError, match="X must be 2-D"):
            method(np.zeros(4), good_y)
        with pytest.raises(ValueError, match="y must be 1-D"):
            method(good_X, np.zeros((4, 1)))
        with pytest.raises(ValueError, match="X has 4 rows but y has 3 entries"):
            method(good_X, good_y[:3])
    assert trainer.log_ == [] and trainer.n_seen_ == 0


def test_a_model_that_fails_while_training_leaves_the_log_and_counters_unchanged():
    X, y = _data(40)
    trainer = StreamTrainer(_tree(buffer_size=20))
    trainer.process_chunk(X, y)
    log_before, seen_before = [dict(r) for r in trainer.log_], trainer.n_seen_
    with pytest.raises(ValueError, match="fitted with 3 features"):
        trainer.fit_chunk(np.zeros((5, 4)), np.zeros(5, dtype=int))     # wrong number of columns
    assert trainer.log_ == log_before and trainer.n_seen_ == seen_before


def test_a_model_that_fails_while_predicting_is_not_hidden():
    X, y = _data(40)
    trainer = StreamTrainer(_tree(buffer_size=20))
    trainer.process_chunk(X, y)
    with pytest.raises(ValueError, match="fitted with 3 features"):
        trainer.score_chunk(np.zeros((5, 4)), np.zeros(5, dtype=int))
    assert len(trainer.log_) == 1 and trainer.accuracy_.n_samples_ == 0


def test_a_metric_that_raises_keeps_earlier_metrics_updated_and_logs_no_row():
    X, y = _data(80)
    ok = StreamingAccuracy()
    trainer = StreamTrainer(_tree(buffer_size=40), metrics={"ok": ok, "bad": Raiser()})
    trainer.fit_chunk(X[:40], y[:40])                                  # trains the tree, so the next chunk can be scored
    rows = len(trainer.log_)
    with pytest.raises(ValueError, match="cannot take the chunk"):
        trainer.score_chunk(X[40:], y[40:])
    assert ok.n_samples_ == 40 and len(trainer.log_) == rows            # the documented behaviour


def test_string_labels_work_end_to_end():
    X, y_int = _data(200)
    y = np.where(y_int == 1, "high", "low")
    trainer = StreamTrainer(_tree(buffer_size=50), metrics={"f1": StreamingF1(average="macro")})
    trainer.run(iter_chunks(X, y, 25))
    assert trainer.log_[-1]["cumulative_accuracy"] > 0.8
    assert set(np.unique(trainer.last_pred_)) <= {"high", "low"}


def test_non_contiguous_chunks_give_the_same_log_as_contiguous_ones():
    X, y = _data(240, 6)
    view = X[::2, ::2]

    def run(chunk_source):
        return StreamTrainer(_tree(buffer_size=30)).run(iter_chunks(chunk_source, y[::2], 20)).history("cumulative_accuracy")

    np.testing.assert_array_equal(run(view), run(np.ascontiguousarray(view)))


def test_the_input_chunk_is_not_modified():
    X, y = _data(60)
    X_before, y_before = X.copy(), y.copy()
    StreamTrainer(Pipeline([("scale", StandardScaler()), ("model", _tree(buffer_size=20))])).run(
        iter_chunks(X, y, 20)
    )
    np.testing.assert_array_equal(X, X_before)
    np.testing.assert_array_equal(y, y_before)


def test_a_pipeline_with_an_imputer_handles_missing_values_in_every_chunk():
    X, y = _data(400)
    X[np.random.default_rng(1).random(X.shape) < 0.1] = np.nan
    pipe = Pipeline([("fill", SimpleImputer("mean")), ("scale", StandardScaler()), ("model", _tree(buffer_size=80))])
    trainer = StreamTrainer(pipe, window=100).run(iter_chunks(X, y, 40))
    assert len(trainer.log_) == 10 and trainer.history("scored")[2:].all()
    assert trainer.log_[-1]["cumulative_accuracy"] > 0.75


# ---------- history, columns, reset ----------
def test_history_returns_one_column_and_columns_lists_them_in_order():
    X, y = _data(120)
    trainer = StreamTrainer(_tree(buffer_size=40), metrics={"f1": StreamingF1()}, window=30)
    assert trainer.history("chunk").shape == (0,)                       # an empty log gives empty arrays
    trainer.run(iter_chunks(X, y, 40))
    assert trainer.columns() == [
        "chunk", "n_samples", "scored", "chunk_accuracy", "cumulative_accuracy", "rolling_accuracy",
        "fitted", "fit_seconds", "retrained", "n_retrains", "n_seen", "memory_bytes", "f1",
    ]
    assert trainer.history("chunk").tolist() == [0, 1, 2]
    assert trainer.history("scored").dtype == bool
    assert set(trainer.log_[0]) == set(trainer.columns())
    assert "rolling_accuracy" not in StreamTrainer(_tree()).columns()


def test_history_of_an_unknown_column_lists_the_valid_ones():
    trainer = StreamTrainer(_tree())
    with pytest.raises(KeyError, match="no column 'acc'.*cumulative_accuracy"):
        trainer.history("acc")


def test_reset_clears_the_log_and_scores_but_keeps_the_trained_model():
    X, y = _data(200)
    trainer = StreamTrainer(_tree(buffer_size=40), metrics={"f1": StreamingF1()}, window=50)
    trainer.run(iter_chunks(X, y, 40))
    assert trainer.reset() is trainer
    assert trainer.log_ == [] and trainer.n_seen_ == 0 and trainer.last_pred_ is None
    assert trainer.accuracy_.n_samples_ == 0 and trainer.rolling_accuracy_.n_samples_ == 0
    assert trainer.metrics["f1"].n_samples_ == 0
    row = trainer.process_chunk(X[:40], y[:40])
    assert row["chunk"] == 0 and row["scored"] is True                  # the tree from before still predicts
    assert row["n_seen"] == 40


# ---------- iter_chunks ----------
def test_chunks_cover_the_data_in_order_with_a_smaller_last_chunk():
    X, y = _data(25)
    chunks = list(iter_chunks(X, y, 10))
    assert [len(c[1]) for c in chunks] == [10, 10, 5]
    np.testing.assert_array_equal(np.concatenate([c[0] for c in chunks]), X)
    np.testing.assert_array_equal(np.concatenate([c[1] for c in chunks]), y)
    assert np.shares_memory(chunks[0][0], X)                            # views, not copies


def test_a_chunk_size_larger_than_the_data_gives_one_chunk_and_no_data_gives_none():
    X, y = _data(7)
    assert len(list(iter_chunks(X, y, 100))) == 1
    assert list(iter_chunks(np.empty((0, 3)), np.empty(0), 4)) == []


def test_shuffled_chunks_keep_rows_paired_and_are_reproducible():
    X = np.arange(40.0).reshape(20, 2)
    y = X[:, 0].astype(int)                                             # the label identifies its row
    first = list(iter_chunks(X, y, 6, shuffle=True, random_state=3))
    again = list(iter_chunks(X, y, 6, shuffle=True, random_state=3))
    other = list(iter_chunks(X, y, 6, shuffle=True, random_state=4))
    X_all, y_all = np.concatenate([c[0] for c in first]), np.concatenate([c[1] for c in first])
    np.testing.assert_array_equal(X_all[:, 0].astype(int), y_all)
    assert sorted(y_all.tolist()) == y.tolist() and y_all.tolist() != y.tolist()
    np.testing.assert_array_equal(X_all, np.concatenate([c[0] for c in again]))
    assert not np.array_equal(X_all, np.concatenate([c[0] for c in other]))
    assert not np.shares_memory(first[0][0], X)


def test_chunking_errors_are_raised_when_called_not_when_iterating():
    X, y = _data(10)
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        iter_chunks(X, y, 0)
    with pytest.raises(TypeError, match="chunk_size must be an integer"):
        iter_chunks(X, y, 2.5)
    with pytest.raises(TypeError, match="chunk_size must be an integer"):
        iter_chunks(X, y, True)
    with pytest.raises(ValueError, match="X has 10 rows but y has 9 entries"):
        iter_chunks(X, y[:9], 3)
    with pytest.raises(ValueError, match="X must be 2-D"):
        iter_chunks(X[:, 0], y, 3)