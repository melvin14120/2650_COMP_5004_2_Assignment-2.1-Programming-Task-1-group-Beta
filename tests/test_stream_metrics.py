"""
tests/test_stream_metrics.py
Unit tests for numcompute_stream.metrics

Covers: StreamingConfusionMatrix, StreamingAccuracy, StreamingPrecision,
StreamingRecall, StreamingF1, StreamingMSE and StreamingAUC. The main check
throughout: updating chunk by chunk gives the same score as the A1 function
called on all the data at once (and, with a window, on the last w samples).
Also: labels that grow, scores that are undefined early (NaN), empty chunks,
NaN input, bad shapes, unknown labels, reset and parameter validation.

Run with: python -m pytest tests/test_stream_metrics.py -v   (from the repo root)
"""

import warnings

import numpy as np
import pytest

from numcompute import metrics as a1
from numcompute_stream.metrics import (
    StreamingAccuracy,
    StreamingAUC,
    StreamingConfusionMatrix,
    StreamingF1,
    StreamingMSE,
    StreamingPrecision,
    StreamingRecall,
)


@pytest.fixture(autouse=True)
def _no_stray_warnings():
    """Fail any test that leaks a RuntimeWarning (e.g. 'invalid value in divide')."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


def _labels(n=500, k=3, seed=0, noise=0.3):
    """True labels in 0..k-1 and noisy predictions."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, k, n)
    y_pred = np.where(rng.random(n) < noise, rng.integers(0, k, n), y_true)
    return y_true, y_pred


def _scores(n=800, seed=1):
    """Binary labels and scores in [0, 1] that carry some signal."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return y, np.clip(0.3 * y + 0.7 * rng.random(n), 0.0, 1.0)


def _feed(metric, a, b, parts):
    """Give the metric (a, b) in `parts` nearly equal chunks."""
    for ca, cb in zip(np.array_split(a, parts), np.array_split(b, parts)):
        metric.update(ca, cb)
    return metric


# ---------- StreamingConfusionMatrix ----------
def test_confusion_matrix_chunks_equal_a1():
    y_true, y_pred = _labels()
    matrix, labels = _feed(StreamingConfusionMatrix(), y_true, y_pred, 9).result()
    ref, ref_labels = a1.confusion_matrix(y_true, y_pred)
    np.testing.assert_array_equal(matrix, ref)
    np.testing.assert_array_equal(labels, ref_labels)
    assert labels.dtype == ref_labels.dtype                  # integer labels stay integers


def test_confusion_matrix_grows_when_a_new_class_appears():
    cm = StreamingConfusionMatrix()
    cm.update([0, 2], [0, 2])
    cm.update([1, 1, 3], [1, 0, 3])                          # classes 1 and 3 are new
    matrix, labels = cm.result()
    assert labels.tolist() == [0, 1, 2, 3]
    expected, _ = a1.confusion_matrix([0, 2, 1, 1, 3], [0, 2, 1, 0, 3])
    np.testing.assert_array_equal(matrix, expected)


def test_confusion_matrix_with_text_labels():
    cm = StreamingConfusionMatrix()
    cm.update(np.array(["cat", "dog"]), np.array(["cat", "cat"]))
    cm.update(np.array(["bird", "dog"]), np.array(["dog", "dog"]))
    matrix, labels = cm.result()
    assert labels.tolist() == ["bird", "cat", "dog"]
    assert matrix.tolist() == [[0, 0, 1], [0, 1, 0], [0, 1, 1]]


def test_confusion_matrix_fixed_labels_keep_unseen_rows_and_reject_unknown():
    cm = StreamingConfusionMatrix(labels=[2, 0, 1])
    matrix, labels = cm.result()
    assert labels.tolist() == [0, 1, 2] and matrix.shape == (3, 3) and matrix.sum() == 0
    cm.update([0, 0], [0, 1])
    before = cm.result()[0].copy()
    with pytest.raises(ValueError, match="not present in `labels`"):
        cm.update([0, 5], [0, 0])
    np.testing.assert_array_equal(cm.result()[0], before)    # the rejected chunk changed nothing
    assert cm.n_seen_ == 2


def test_confusion_matrix_fixed_labels_accumulate_over_chunks():
    y_true, y_pred = _labels(200, 3, seed=13)
    matrix, labels = _feed(StreamingConfusionMatrix(labels=[0, 1, 2]), y_true, y_pred, 6).result()
    np.testing.assert_array_equal(matrix, a1.confusion_matrix(y_true, y_pred, labels=[0, 1, 2])[0])


def test_confusion_matrix_window_equals_a1_on_the_last_samples():
    y_true, y_pred = _labels(300, 3, seed=2)
    cm = _feed(StreamingConfusionMatrix(window=64), y_true, y_pred, 11)
    ref, ref_labels = a1.confusion_matrix(y_true[-64:], y_pred[-64:])
    matrix, labels = cm.result()
    np.testing.assert_array_equal(matrix, ref)
    np.testing.assert_array_equal(labels, ref_labels)
    assert cm.n_seen_ == 300 and cm.n_samples_ == 64


def test_confusion_matrix_window_does_not_depend_on_chunk_sizes():
    y_true, y_pred = _labels(200, 3, seed=3)
    one = StreamingConfusionMatrix(window=50).update(y_true, y_pred).result()[0]
    many = _feed(StreamingConfusionMatrix(window=50), y_true, y_pred, 200).result()[0]   # one sample per chunk
    np.testing.assert_array_equal(one, many)


def test_confusion_matrix_window_forgets_old_classes():
    cm = StreamingConfusionMatrix(window=3)
    cm.update([0, 0, 1], [0, 0, 1])
    cm.update([2, 2, 2], [2, 2, 2])
    matrix, labels = cm.result()
    assert labels.tolist() == [2] and matrix.tolist() == [[3]]


def test_confusion_matrix_empty_chunks_and_empty_result():
    cm = StreamingConfusionMatrix()
    matrix, labels = cm.result()
    assert matrix.shape == (0, 0) and labels.size == 0
    assert cm.update([], []) is cm and cm.n_seen_ == 0
    windowed = StreamingConfusionMatrix(window=5)
    assert windowed.result()[0].shape == (0, 0)
    cm.update([0, 1], [0, 1])
    cm.update(np.array([], dtype=str), np.array([], dtype=str))   # an empty chunk of another type decides nothing
    assert cm.n_seen_ == 2 and cm.result()[1].tolist() == [0, 1]


def test_confusion_matrix_bad_input_raises_informative_errors():
    cm = StreamingConfusionMatrix()
    with pytest.raises(ValueError, match="1-D"):
        cm.update([[0, 1]], [[0, 1]])
    with pytest.raises(ValueError, match="shape"):
        cm.update([0, 1], [0])
    with pytest.raises(ValueError, match="NaN"):
        cm.update([0.0, np.nan], [0.0, 1.0])
    assert cm.n_seen_ == 0                                    # nothing was kept
    cm.update([0, 1], [0, 1])
    with pytest.raises(TypeError, match="numbers"):
        cm.update(np.array(["a"]), np.array(["a"]))           # numbers earlier, text now
    with pytest.raises(ValueError, match="window"):
        StreamingConfusionMatrix(window=0)
    with pytest.raises(TypeError, match="window"):
        StreamingConfusionMatrix(window=2.5)
    with pytest.raises(ValueError, match="labels"):
        StreamingConfusionMatrix(labels=[])


def test_confusion_matrix_reset_and_result_is_a_copy():
    cm = StreamingConfusionMatrix().update([0, 1], [0, 1])
    cm.result()[0][0, 0] = 99                                 # changing the returned array
    assert cm.result()[0][0, 0] == 1                          # ... does not change the metric
    assert cm.reset() is cm and cm.n_seen_ == 0 and cm.result()[0].shape == (0, 0)
    cm.update(np.array(["x"]), np.array(["x"]))               # a reset forgets the label type too
    assert cm.result()[1].tolist() == ["x"]


# ---------- StreamingAccuracy ----------
def test_accuracy_chunks_equal_a1():
    y_true, y_pred = _labels()
    acc = _feed(StreamingAccuracy(), y_true, y_pred, 13)
    assert acc.result() == pytest.approx(a1.accuracy(y_true, y_pred))
    assert acc.n_seen_ == acc.n_samples_ == 500


def test_accuracy_window_equals_a1_on_the_last_samples():
    y_true, y_pred = _labels(300, 3, seed=4)
    acc = _feed(StreamingAccuracy(window=40), y_true, y_pred, 17)
    assert acc.result() == pytest.approx(a1.accuracy(y_true[-40:], y_pred[-40:]))
    assert acc.n_seen_ == 300 and acc.n_samples_ == 40


def test_accuracy_nan_before_data_empty_chunks_and_reset():
    acc = StreamingAccuracy()
    assert np.isnan(acc.result())
    assert acc.update([], []) is acc and np.isnan(acc.result())
    acc.update(["a", "b"], ["a", "a"])                        # text labels work
    assert acc.result() == 0.5
    assert acc.reset() is acc and np.isnan(acc.result()) and acc.n_seen_ == 0


def test_accuracy_bad_input_raises_informative_errors():
    acc = StreamingAccuracy()
    with pytest.raises(ValueError, match="shape"):
        acc.update([1, 2], [1])
    with pytest.raises(ValueError, match="1-D"):
        acc.update([[1]], [[1]])
    with pytest.raises(ValueError, match="window"):
        StreamingAccuracy(window=-1)
    assert repr(StreamingAccuracy(window=5)) == "StreamingAccuracy(window=5)"


# ---------- StreamingPrecision / Recall / F1 ----------
_SCORES = [
    (StreamingPrecision, a1.precision),
    (StreamingRecall, a1.recall),
    (StreamingF1, a1.f1),
]


@pytest.mark.parametrize("cls, reference", _SCORES)
@pytest.mark.parametrize("average", ["macro", "micro"])
def test_scores_chunks_equal_a1_multiclass(cls, reference, average):
    y_true, y_pred = _labels(600, 4, seed=5)
    metric = _feed(cls(average=average), y_true, y_pred, 10)
    assert metric.result() == pytest.approx(reference(y_true, y_pred, average=average))


@pytest.mark.parametrize("cls, reference", _SCORES)
def test_scores_chunks_equal_a1_binary(cls, reference):
    y_true, y_pred = _labels(400, 2, seed=6)
    metric = _feed(cls(), y_true, y_pred, 8)
    assert metric.result() == pytest.approx(reference(y_true, y_pred))


@pytest.mark.parametrize("cls, reference", _SCORES)
def test_scores_with_text_labels_and_pos_label(cls, reference):
    rng = np.random.default_rng(7)
    y_true = rng.choice(["no", "yes"], 200)
    y_pred = np.where(rng.random(200) < 0.2, rng.choice(["no", "yes"], 200), y_true)
    metric = _feed(cls(pos_label="yes"), y_true, y_pred, 5)
    assert metric.result() == pytest.approx(reference(y_true, y_pred, pos_label="yes"))


@pytest.mark.parametrize("cls, reference", _SCORES)
def test_scores_window_equals_a1_on_the_last_samples(cls, reference):
    y_true, y_pred = _labels(400, 3, seed=8)
    metric = _feed(cls(average="macro", window=90), y_true, y_pred, 7)
    assert metric.result() == pytest.approx(reference(y_true[-90:], y_pred[-90:], average="macro"))
    assert metric.n_seen_ == 400 and metric.n_samples_ == 90


@pytest.mark.parametrize("cls", [StreamingPrecision, StreamingRecall, StreamingF1])
def test_scores_are_nan_until_there_is_something_to_measure(cls):
    metric = cls()
    assert np.isnan(metric.result())                          # no samples
    metric.update([0, 0], [0, 0])
    assert np.isnan(metric.result())                          # positive class 1 has not appeared
    metric.update([1, 1], [1, 0])
    assert not np.isnan(metric.result())


@pytest.mark.parametrize("cls", [StreamingPrecision, StreamingRecall, StreamingF1])
def test_fixed_labels_give_zero_instead_of_nan_before_the_positive_class(cls):
    metric = cls(labels=[0, 1]).update([0, 0], [0, 0])
    assert metric.result() == 0.0


def test_precision_of_a_class_that_is_never_predicted_is_zero():
    metric = StreamingPrecision().update([1, 1, 0], [0, 0, 0])   # A1 convention: 0/0 -> 0.0
    assert metric.result() == 0.0
    assert StreamingRecall().update([1, 1, 0], [0, 0, 0]).result() == 0.0


def test_binary_with_more_than_two_classes_raises_when_the_result_is_read():
    metric = StreamingF1()
    metric.update([0, 1], [0, 1])
    metric.update([2], [2])                                   # the third class appears later
    with pytest.raises(ValueError, match="at most 2 classes"):
        metric.result()
    assert StreamingF1(average="macro").update([0, 1, 2], [0, 1, 2]).result() == 1.0


def test_scores_invalid_parameters_raise_value_error():
    with pytest.raises(ValueError, match="average"):
        StreamingPrecision(average="weighted")
    with pytest.raises(ValueError, match="pos_label"):
        StreamingRecall(labels=[0, 1], pos_label=2)
    with pytest.raises(ValueError, match="at most 2 classes"):
        StreamingF1(labels=[0, 1, 2])
    with pytest.raises(ValueError, match="window"):
        StreamingF1(window=0)


def test_scores_reset_counters_chaining_and_rejected_chunks():
    metric = StreamingF1()
    assert metric.update([0, 1], [0, 1]) is metric and metric.n_seen_ == 2
    with pytest.raises(ValueError, match="NaN"):
        metric.update([0.0, np.nan], [0.0, 1.0])
    assert metric.n_seen_ == 2 and metric.result() == 1.0
    assert metric.reset() is metric and metric.n_seen_ == 0 and np.isnan(metric.result())
    assert repr(StreamingF1(average="macro")) == (
        "StreamingF1(average='macro', pos_label=1, labels=None, window=None)"
    )


# ---------- StreamingMSE ----------
def test_mse_chunks_equal_a1():
    rng = np.random.default_rng(9)
    y_true, y_pred = rng.normal(size=400), rng.normal(size=400)
    mse = _feed(StreamingMSE(), y_true, y_pred, 12)
    assert mse.result() == pytest.approx(a1.mse(y_true, y_pred))


def test_mse_window_equals_a1_on_the_last_samples():
    rng = np.random.default_rng(10)
    y_true, y_pred = rng.normal(size=300), rng.normal(size=300)
    mse = _feed(StreamingMSE(window=25), y_true, y_pred, 9)
    assert mse.result() == pytest.approx(a1.mse(y_true[-25:], y_pred[-25:]))


def test_mse_nan_before_data_empty_chunks_and_reset():
    mse = StreamingMSE()
    assert np.isnan(mse.result()) and mse.update([], []) is mse
    mse.update([1, 2], [1, 4])                                # integers are fine
    assert mse.result() == 2.0
    assert mse.reset() is mse and np.isnan(mse.result())


def test_mse_rejects_nan_bad_shapes_and_text_without_changing_state():
    mse = StreamingMSE().update([1.0], [3.0])
    with pytest.raises(ValueError, match="NaN"):
        mse.update([np.nan], [1.0])
    with pytest.raises(ValueError, match="shape"):
        mse.update([1.0, 2.0], [1.0])
    with pytest.raises(TypeError, match="numeric"):
        mse.update(["a"], ["b"])
    assert mse.result() == 4.0 and mse.n_seen_ == 1


# ---------- StreamingAUC ----------
def test_auc_is_exact_when_scores_sit_on_bin_values():
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 600)
    score = (rng.integers(0, 10, 600) + 0.5) / 10             # centres of 10 bins: many ties
    auc_ = _feed(StreamingAUC(bins=10), y, score, 7)
    fpr, tpr, _ = a1.roc_curve(y, score)
    assert auc_.result() == pytest.approx(a1.auc(fpr, tpr))


def test_auc_with_many_bins_is_close_to_a1():
    y, score = _scores()
    auc_ = _feed(StreamingAUC(bins=5000), y, score, 9)
    fpr, tpr, _ = a1.roc_curve(y, score)
    assert auc_.result() == pytest.approx(a1.auc(fpr, tpr), abs=2e-3)


def test_auc_does_not_depend_on_chunk_sizes():
    y, score = _scores(300)
    one = StreamingAUC(bins=100).update(y, score).result()
    many = _feed(StreamingAUC(bins=100), y, score, 300).result()
    assert one == pytest.approx(many)


def test_auc_perfect_reversed_and_all_tied():
    y = np.array([0, 0, 1, 1])
    assert StreamingAUC(bins=10).update(y, [0.1, 0.2, 0.8, 0.9]).result() == 1.0
    assert StreamingAUC(bins=10).update(y, [0.9, 0.8, 0.2, 0.1]).result() == 0.0
    assert StreamingAUC(bins=10).update(y, [0.5, 0.5, 0.5, 0.5]).result() == 0.5


def test_auc_is_nan_until_both_classes_have_been_seen():
    auc_ = StreamingAUC(bins=10)
    assert np.isnan(auc_.result())
    auc_.update([1, 1], [0.7, 0.9])
    assert np.isnan(auc_.result())
    with pytest.raises(ValueError, match="at least one 0 and one 1"):
        auc_.roc_curve()
    auc_.update([0], [0.1])
    assert auc_.result() == 1.0


def test_auc_window_equals_a1_on_the_last_samples():
    y, score = _scores(500, seed=3)
    auc_ = _feed(StreamingAUC(bins=4000, window=120), y, score, 6)
    fpr, tpr, _ = a1.roc_curve(y[-120:], score[-120:])
    assert auc_.result() == pytest.approx(a1.auc(fpr, tpr), abs=5e-3)
    assert auc_.n_seen_ == 500 and auc_.n_samples_ == 120


def test_auc_roc_curve_layout_and_area():
    y, score = _scores(200)
    auc_ = StreamingAUC(bins=50).update(y, score)
    fpr, tpr, thresholds = auc_.roc_curve()
    assert fpr.shape == tpr.shape == thresholds.shape == (51,)
    assert (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0.0, 0.0, 1.0, 1.0)
    assert thresholds[0] == np.inf and thresholds[1] == pytest.approx(0.98) and thresholds[-1] == 0.0
    assert (np.diff(fpr) >= 0).all() and (np.diff(tpr) >= 0).all()
    assert a1.auc(fpr, tpr) == auc_.result()


def test_auc_score_range_edges_and_custom_range():
    auc_ = StreamingAUC(bins=4).update([0, 1], [0.0, 1.0])    # 0.0 and 1.0 are both allowed
    assert auc_.result() == 1.0
    wide = StreamingAUC(bins=4, score_range=(-10, 10)).update([0, 1], [-3.0, 8.0])
    assert wide.result() == 1.0


def test_auc_bad_input_raises_informative_errors_and_keeps_state():
    auc_ = StreamingAUC(bins=10).update([0, 1], [0.2, 0.8])
    with pytest.raises(ValueError, match="score_range"):
        auc_.update([0, 1], [0.5, 1.5])
    with pytest.raises(ValueError, match="NaN"):
        auc_.update([0, 1], [0.5, np.nan])
    with pytest.raises(ValueError, match="only 0 and 1"):
        auc_.update([0, 2], [0.5, 0.5])
    with pytest.raises(ValueError, match="shape"):
        auc_.update([0, 1], [0.5])
    with pytest.raises(TypeError, match="numeric"):
        auc_.update(["a", "b"], [0.5, 0.5])
    assert auc_.n_seen_ == 2 and auc_.result() == 1.0
    assert auc_.update([], []) is auc_


def test_auc_invalid_parameters_and_reset():
    with pytest.raises(ValueError, match="bins"):
        StreamingAUC(bins=0)
    with pytest.raises(TypeError, match="bins"):
        StreamingAUC(bins=10.5)
    with pytest.raises(ValueError, match="low < high"):
        StreamingAUC(score_range=(1.0, 1.0))
    with pytest.raises(ValueError, match="two numbers"):
        StreamingAUC(score_range=0.5)
    with pytest.raises(ValueError, match="finite"):
        StreamingAUC(score_range=(0.0, np.inf))
    auc_ = StreamingAUC(bins=10).update([0, 1], [0.2, 0.8])
    assert auc_.reset() is auc_ and np.isnan(auc_.result()) and auc_.n_seen_ == 0
    assert repr(auc_) == "StreamingAUC(bins=10, score_range=(0.0, 1.0), window=None)"


def test_auc_memory_does_not_grow_with_the_stream():
    auc_ = StreamingAUC(bins=100)
    rng = np.random.default_rng(12)
    for _ in range(50):
        auc_.update(rng.integers(0, 2, 1000), rng.random(1000))
    assert auc_._pos.shape == auc_._neg.shape == (100,)
    assert auc_._pos.sum() + auc_._neg.sum() == 50_000