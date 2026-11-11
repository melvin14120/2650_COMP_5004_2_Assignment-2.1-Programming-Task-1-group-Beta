"""
tests/test_stream_visualise.py
Unit tests for numcompute_stream.visualise

Covers: plot_metric_over_time, compare_models and plot_predictions_vs_ground_truth.
The plots are checked through the objects matplotlib builds (lines, scatter
offsets, legend texts, ticks, labels), not by looking at pictures. Also: NaN
gaps, empty and badly shaped input, drawing on a given Axes, saving to a file,
the ``show`` option, and a full run from a StreamTrainer log.

The Agg backend is used, so no window ever opens.

Run with: python -m pytest tests/test_stream_visualise.py -v   (from the repo root)
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from numcompute_stream.stream import StreamTrainer, iter_chunks  # noqa: E402
from numcompute_stream.tree import DecisionTreeClassifier  # noqa: E402
from numcompute_stream.visualise import (  # noqa: E402
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


@pytest.fixture(autouse=True)
def _clean_figures_and_warnings():
    """Fail on a stray RuntimeWarning, and close every figure after each test."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield
    plt.close("all")


def _legend_texts(ax):
    legend = ax.get_legend()
    return [] if legend is None else [text.get_text() for text in legend.get_texts()]


def _scatter(ax, k):
    """Offsets of the k-th scatter collection as an (n, 2) array."""
    return np.asarray(ax.collections[k].get_offsets())


# ---------- plot_metric_over_time ----------
def test_metric_plot_returns_a_figure_with_title_labels_and_grid():
    fig = plot_metric_over_time([0.5, 0.7, 0.9], "Accuracy over time", "accuracy")
    ax = fig.axes[0]
    assert isinstance(fig, Figure) and len(fig.axes) == 1
    assert ax.get_title() == "Accuracy over time"
    assert ax.get_ylabel() == "accuracy" and ax.get_xlabel() == "Chunk"
    assert any(line.get_visible() for line in ax.get_xgridlines())
    assert ax.get_legend() is None                                     # one line needs no legend


def test_metric_plot_draws_the_values_against_the_chunk_numbers():
    values = [0.5, 0.7, 0.9, 0.8]
    ax = plot_metric_over_time(values).axes[0]
    line = ax.lines[0]
    np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2, 3])
    np.testing.assert_array_equal(line.get_ydata(), values)


def test_custom_x_positions_and_xlabel_are_used():
    ax = plot_metric_over_time([1, 2, 3], xlabel="Samples seen", x=[100, 200, 300]).axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [100, 200, 300])
    assert ax.get_xlabel() == "Samples seen"
    with pytest.raises(ValueError, match=r"x has 2 values but the series has 3"):
        plot_metric_over_time([1, 2, 3], x=[1, 2])


def test_x_ticks_are_whole_chunk_numbers_even_for_a_short_series():
    fig = plot_metric_over_time([0.1, 0.2, 0.3])
    fig.canvas.draw()
    ticks = fig.axes[0].get_xticks()
    assert np.array_equal(ticks, np.round(ticks))


def test_nan_values_are_kept_so_matplotlib_draws_a_gap():
    ax = plot_metric_over_time([np.nan, np.nan, 0.5, np.nan, 0.7]).axes[0]
    y = ax.lines[0].get_ydata()
    assert np.isnan(y).tolist() == [True, True, False, True, False]
    plot_metric_over_time([np.nan, np.nan])                             # all missing: nothing drawn, no error


def test_retrain_markers_are_dashed_lines_at_the_given_positions_with_a_legend():
    ax = plot_metric_over_time([0.5, 0.6, 0.7, 0.8, 0.9], retrains=[1, 3]).axes[0]
    segments = ax.collections[0].get_segments()
    assert [seg[0][0] for seg in segments] == [1.0, 3.0]
    assert _legend_texts(ax) == ["retrain"]


def test_no_retrain_markers_for_none_or_an_empty_list():
    for retrains in (None, [], np.empty(0)):
        ax = plot_metric_over_time([0.5, 0.6], retrains=retrains).axes[0]
        assert len(ax.collections) == 0 and ax.get_legend() is None
    with pytest.raises(TypeError, match="retrains must hold numbers"):
        plot_metric_over_time([0.5, 0.6], retrains=["a"])


def test_ylim_is_applied():
    ax = plot_metric_over_time([0.5, 0.6], ylim=(0, 1)).axes[0]
    assert ax.get_ylim() == (0.0, 1.0)


def test_numbers_can_be_given_as_list_tuple_integers_or_booleans():
    for values in ([1, 2, 3], (1.5, 2.5), np.array([1, 2], dtype=np.int32), np.array([True, False])):
        y = plot_metric_over_time(values).axes[0].lines[0].get_ydata()
        assert y.dtype == float


def test_a_single_value_can_be_plotted():
    fig = plot_metric_over_time([0.8])
    assert len(fig.axes[0].lines[0].get_ydata()) == 1


def test_bad_metric_values_raise_informative_errors():
    with pytest.raises(ValueError, match="metric_values is empty"):
        plot_metric_over_time([])
    with pytest.raises(ValueError, match=r"metric_values must be 1-D, got shape \(2, 2\)"):
        plot_metric_over_time(np.ones((2, 2)))
    with pytest.raises(TypeError, match="metric_values must hold numbers, got dtype"):
        plot_metric_over_time(["a", "b"])
    assert plt.get_fignums() == []                                      # a failed call leaves no stray figure


def test_the_input_series_is_not_modified():
    values = np.array([0.5, np.nan, 0.9])
    before = values.copy()
    plot_metric_over_time(values, retrains=np.array([1.0]))
    np.testing.assert_array_equal(values, before)


# ---------- drawing on a given Axes, figure size, save and show ----------
def test_a_given_axes_is_drawn_on_and_no_new_figure_is_made():
    fig, (left, right) = plt.subplots(1, 2)
    assert plot_metric_over_time([1, 2, 3], "left", ax=left) is fig
    assert compare_models([1, 2], [2, 3], ax=right) is fig
    assert plot_predictions_vs_ground_truth([0, 1], [0, 1], ax=right) is fig
    assert plt.get_fignums() == [fig.number] and len(fig.axes) == 2
    assert left.get_title() == "left" and len(left.lines) == 1


def test_a_figure_passed_in_by_the_caller_keeps_its_own_layout():
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.3, right=0.8)
    plot_metric_over_time([1, 2], ax=ax)
    assert (fig.subplotpars.left, fig.subplotpars.right) == (0.3, 0.8)


def test_figsize_sets_the_size_of_a_new_figure():
    fig = plot_metric_over_time([1, 2], figsize=(5, 3))
    assert tuple(fig.get_size_inches()) == (5.0, 3.0)


@pytest.mark.parametrize("name", ["plot.png", "plot.pdf"])
def test_save_path_writes_a_file_of_the_format_given_by_the_extension(tmp_path, name):
    path = tmp_path / name
    plot_metric_over_time([0.5, 0.6], save_path=path)
    start = path.read_bytes()[:4]
    assert start == (b"\x89PNG" if name.endswith("png") else b"%PDF")


def test_all_three_functions_can_save_and_accept_a_string_path(tmp_path):
    plot_metric_over_time([1, 2], save_path=str(tmp_path / "a.png"))
    compare_models([1, 2], [2, 3], save_path=str(tmp_path / "b.png"))
    plot_predictions_vs_ground_truth([0, 1], [0, 0], save_path=str(tmp_path / "c.png"))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.png", "b.png", "c.png"]


def test_nothing_is_saved_by_default_and_a_missing_folder_raises(tmp_path):
    plot_metric_over_time([1, 2])
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(FileNotFoundError):
        plot_metric_over_time([1, 2], save_path=tmp_path / "no_such_folder" / "a.png")


def test_show_is_only_called_when_asked_for(monkeypatch):
    calls = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: calls.append(1))
    plot_metric_over_time([1, 2])
    compare_models([1, 2], [2, 3])
    plot_predictions_vs_ground_truth([0, 1], [0, 1])
    assert calls == []
    plot_metric_over_time([1, 2], show=True)
    compare_models([1, 2], [2, 3], show=True)
    plot_predictions_vs_ground_truth([0, 1], [0, 1], show=True)
    assert len(calls) == 3


# ---------- compare_models ----------
def test_compare_draws_two_lines_with_the_labels_in_a_legend():
    ax = compare_models([0.5, 0.6, 0.7], [0.6, 0.7, 0.8], ("small buffer", "large buffer"),
                        "Buffer size", "accuracy").axes[0]
    assert len(ax.lines) == 2
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0.5, 0.6, 0.7])
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), [0.6, 0.7, 0.8])
    assert _legend_texts(ax) == ["small buffer", "large buffer"]
    assert ax.get_title() == "Buffer size" and ax.get_ylabel() == "accuracy" and ax.get_xlabel() == "Chunk"


def test_the_two_models_differ_by_colour_marker_and_line_style():
    first, second = compare_models([1, 2], [2, 3]).axes[0].lines
    assert first.get_color() != second.get_color()
    assert first.get_marker() != second.get_marker()
    assert first.get_linestyle() != second.get_linestyle()


def test_default_labels_and_list_labels():
    assert _legend_texts(compare_models([1, 2], [2, 3]).axes[0]) == ["Model 1", "Model 2"]
    assert _legend_texts(compare_models([1, 2], [2, 3], ["a", "b"]).axes[0]) == ["a", "b"]


def test_series_of_different_lengths_and_with_gaps_can_be_compared():
    ax = compare_models([np.nan, 0.5, 0.6, 0.7], [0.4, 0.5], ("long", "short")).axes[0]
    assert len(ax.lines[0].get_ydata()) == 4 and len(ax.lines[1].get_ydata()) == 2
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [0, 1])
    ax = compare_models([1, 2, 3], [3, 2, 1], x=[10, 20, 30]).axes[0]
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [10, 20, 30])
    with pytest.raises(ValueError, match="x has 3 values but the series has 2"):
        compare_models([1, 2, 3], [1, 2], x=[1, 2, 3])


def test_labels_must_be_two_strings():
    for bad in ("ab", ("only one",), ("a", "b", "c"), (1, 2), ("a", None)):
        with pytest.raises(ValueError, match="labels must be two strings"):
            compare_models([1, 2], [2, 3], bad)


def test_compare_names_the_series_that_is_bad():
    with pytest.raises(ValueError, match="metric2 is empty"):
        compare_models([1, 2], [])
    with pytest.raises(ValueError, match="metric1 must be 1-D"):
        compare_models(np.ones((2, 2)), [1, 2])
    with pytest.raises(TypeError, match="metric2 must hold numbers"):
        compare_models([1, 2], ["a", "b"])


def test_compare_applies_ylim():
    assert compare_models([1, 2], [2, 3], ylim=(0, 5)).axes[0].get_ylim() == (0.0, 5.0)


# ---------- plot_predictions_vs_ground_truth ----------
def test_predictions_plot_counts_right_and_wrong_in_title_and_legend():
    ax = plot_predictions_vs_ground_truth([0, 1, 1, 0, 1], [0, 1, 0, 0, 0]).axes[0]
    assert ax.get_title() == "Predictions vs ground truth: 3 of 5 correct"
    assert _legend_texts(ax) == ["true class", "correct prediction (3)", "wrong prediction (2)"]
    assert ax.get_xlabel() == "Sample in chunk" and ax.get_ylabel() == "Class"


def test_a_given_title_replaces_the_default_one():
    ax = plot_predictions_vs_ground_truth([0, 1], [0, 1], title="Latest chunk", xlabel="row", ylabel="label").axes[0]
    assert ax.get_title() == "Latest chunk" and ax.get_xlabel() == "row" and ax.get_ylabel() == "label"


def test_true_correct_and_wrong_points_sit_at_the_right_positions():
    y_true = np.array([0, 1, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 2, 1])
    ax = plot_predictions_vs_ground_truth(y_true, y_pred).axes[0]
    np.testing.assert_array_equal(_scatter(ax, 0), np.column_stack([np.arange(5), y_true]))   # all true classes
    np.testing.assert_array_equal(_scatter(ax, 1), [[0, 0], [1, 1], [3, 2]])                   # correct ones
    np.testing.assert_array_equal(_scatter(ax, 2), [[2, 2], [4, 1]])                           # wrong ones: where predicted


def test_the_rows_are_labelled_with_the_sorted_classes_including_ones_only_predicted():
    ax = plot_predictions_vs_ground_truth([5, 5, 9], [5, 7, 9]).axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["5", "7", "9"]
    assert list(ax.get_yticks()) == [0, 1, 2]


def test_string_labels_are_supported():
    y_true = np.array(["cat", "dog", "dog", "cat"])
    y_pred = np.array(["cat", "dog", "cat", "cat"])
    ax = plot_predictions_vs_ground_truth(y_true, y_pred).axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["cat", "dog"]
    assert _scatter(ax, 2).tolist() == [[2.0, 0.0]]                      # predicted 'cat' (row 0) for a dog


def test_all_correct_and_all_wrong_are_handled():
    ax = plot_predictions_vs_ground_truth([0, 1, 0], [0, 1, 0]).axes[0]
    assert _scatter(ax, 2).shape == (0, 2) and "wrong prediction (0)" in _legend_texts(ax)
    ax = plot_predictions_vs_ground_truth([0, 1, 0], [1, 0, 1]).axes[0]
    assert _scatter(ax, 1).shape == (0, 2) and "correct prediction (0)" in _legend_texts(ax)


def test_one_class_and_one_sample_work():
    plot_predictions_vs_ground_truth([3, 3, 3], [3, 3, 3])
    ax = plot_predictions_vs_ground_truth([1], [0]).axes[0]
    assert ax.get_title() == "Predictions vs ground truth: 0 of 1 correct"


def test_bad_labels_raise_informative_errors():
    with pytest.raises(ValueError, match="no samples to plot"):
        plot_predictions_vs_ground_truth([], [])
    with pytest.raises(ValueError, match=r"y_true has shape \(3,\) but y_pred has shape \(2,\)"):
        plot_predictions_vs_ground_truth([0, 1, 0], [0, 1])
    with pytest.raises(ValueError, match="must be 1-D"):
        plot_predictions_vs_ground_truth(np.zeros((2, 2)), np.zeros((2, 2)))
    assert plt.get_fignums() == []


def test_non_contiguous_labels_give_the_same_picture_as_contiguous_ones():
    y_true = np.array([0, 1, 2, 1, 0, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 0, 0, 1, 1, 0])
    a = plot_predictions_vs_ground_truth(y_true[::2], y_pred[::2]).axes[0]
    b = plot_predictions_vs_ground_truth(np.ascontiguousarray(y_true[::2]), np.ascontiguousarray(y_pred[::2])).axes[0]
    for k in range(3):
        np.testing.assert_array_equal(_scatter(a, k), _scatter(b, k))


def test_the_label_arrays_are_not_modified():
    y_true, y_pred = np.array([0, 1, 1]), np.array([0, 0, 1])
    plot_predictions_vs_ground_truth(y_true, y_pred)
    np.testing.assert_array_equal(y_true, [0, 1, 1])
    np.testing.assert_array_equal(y_pred, [0, 0, 1])


# ---------- with a real stream ----------
def test_the_plots_work_on_the_log_of_a_real_stream():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] + 0.2 * rng.normal(size=400) > 0).astype(int)
    small = StreamTrainer(DecisionTreeClassifier(max_depth=3, buffer_size=60)).run(iter_chunks(X, y, 30))
    large = StreamTrainer(DecisionTreeClassifier(max_depth=3, buffer_size=120)).run(iter_chunks(X, y, 30))
    accuracy = small.history("cumulative_accuracy")                      # starts with NaN: not scored yet
    assert np.isnan(accuracy[0])
    ax = plot_metric_over_time(accuracy, "Accuracy", "accuracy",
                               retrains=np.flatnonzero(small.history("retrained"))).axes[0]
    assert len(ax.collections[0].get_segments()) == small.history("retrained").sum()
    compare_models(accuracy, large.history("cumulative_accuracy"), ("60", "120"))
    ax = plot_predictions_vs_ground_truth(small.last_true_, small.last_pred_).axes[0]
    assert ax.get_title().endswith(f"of {len(small.last_true_)} correct")