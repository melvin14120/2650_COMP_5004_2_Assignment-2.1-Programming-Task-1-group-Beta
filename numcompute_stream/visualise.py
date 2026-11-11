"""
visualise.py
Plots for streaming runs, drawn with matplotlib's object-oriented API.

Provides
--------
- plot_metric_over_time(metric_values, title, ylabel, ...)
    one metric (e.g. accuracy) across the chunks of a stream
- compare_models(metric1, metric2, labels, ...)
    the same metric for two models on one set of axes
- plot_predictions_vs_ground_truth(y_true, y_pred, ...)
    predicted against true classes for one chunk (usually the latest)

Design notes
------------
Every function builds its own ``Figure`` and ``Axes`` (or draws on the ``ax``
you pass, to place it in a subplot grid), adds a title, axis labels, a grid and,
where there is more than one thing to tell apart, a legend. It returns the
``Figure``.

Saving and showing are options of every function:

- ``save_path='run.png'`` writes the figure to that file (format from the
  extension); the folder must already exist.
- ``show=True`` calls ``plt.show()``, for scripts. In a Jupyter notebook the
  figure appears below the cell by itself, so ``show`` is not needed there.

Neither is the default, so a function called in a test or a loop does not open
windows. Close figures you no longer need with ``plt.close(fig)``.

Colours come from the Okabe-Ito palette, which stays readable with the common
kinds of colour blindness. Lines also differ by marker and line style, so the
plots still work in black and white.

Missing values
--------------
``NaN`` in a metric series (for example the first chunks of a stream, before
the model can predict) is not drawn: the line starts at the first number and
breaks at any later gap.

Axis convention
---------------
Metric series are 1-D: one value per chunk, in stream order. The x axis is the
chunk number starting at 0 (the ``chunk`` column of ``StreamTrainer.log_``)
unless ``x`` gives other positions, such as the number of samples seen.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

__all__ = ["plot_metric_over_time", "compare_models", "plot_predictions_vs_ground_truth"]

# Okabe-Ito colour-blind-safe palette
_BLUE = "#0072B2"
_VERMILLION = "#D55E00"
_GREY = "#555555"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _as_series(values, name: str, where: str) -> np.ndarray:
    """Return ``values`` as a non-empty 1-D float array, or raise."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{where}: {name} must be 1-D, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{where}: {name} is empty, there is nothing to plot.")
    if arr.dtype.kind not in "biuf":
        raise TypeError(f"{where}: {name} must hold numbers, got dtype {arr.dtype}.")
    return arr.astype(float)


def _x_values(x, n: int, where: str) -> np.ndarray:
    """Chunk numbers 0..n-1, or the given positions (one per value)."""
    if x is None:
        return np.arange(n)
    x = _as_series(x, "x", where)
    if x.shape[0] != n:
        raise ValueError(f"{where}: x has {x.shape[0]} values but the series has {n}.")
    return x


def _positions(values, where: str) -> np.ndarray:
    """Marker positions as a 1-D float array; an empty list is allowed."""
    if values is None:
        return np.empty(0)
    arr = np.asarray(values)
    if arr.size == 0:
        return np.empty(0)
    return _as_series(arr, "retrains", where)


def _axes(ax, figsize):
    """Return ``(fig, ax, created)``: the axes given, or a new figure and axes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, True
    return ax.figure, ax, False


def _decorate(ax, title: str, xlabel: str, ylabel: str, ylim) -> None:
    """Title, axis labels, light grid, whole-number x ticks and optional y limits."""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if ylim is not None:
        ax.set_ylim(ylim)


def _finish(fig, created: bool, save_path, show: bool):
    """Tidy the layout of a figure made here, then save and/or show it."""
    if created:
        fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_metric_over_time(metric_values, title: str = "", ylabel: str = "", xlabel: str = "Chunk",
                          x=None, retrains=None, ylim=None, ax=None, figsize=(8, 4.5),
                          save_path=None, show: bool = False):
    """Plot one metric across the chunks of a stream.

    Parameters
    ----------
    metric_values : array-like of numbers, shape (n_chunks,)
        One value per chunk. ``NaN`` leaves a gap.
    title, ylabel, xlabel : str
        Texts of the plot. ``xlabel`` defaults to ``'Chunk'``.
    x : array-like, shape (n_chunks,), optional
        Positions on the x axis instead of 0, 1, 2, ... (for example the number
        of samples seen after each chunk).
    retrains : array-like, optional
        x positions to mark with a dashed vertical line labelled ``'retrain'``
        (for example the chunks where ``history('retrained')`` is True).
    ylim : (float, float), optional
        Limits of the y axis.
    ax : matplotlib Axes, optional
        Draw here instead of on a new figure.
    figsize : (float, float), default (8, 4.5)
        Size of the new figure in inches.
    save_path : str or path-like, optional
        Write the figure to this file.
    show : bool, default False
        Call ``plt.show()``.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ``metric_values`` is empty or not 1-D, or ``x`` has a different length.
    TypeError
        If ``metric_values`` does not hold numbers.

    Complexity
    ----------
    O(n) for n chunks, plus the cost of drawing.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig = plot_metric_over_time([0.6, 0.7, 0.8], "Accuracy", "accuracy")
    >>> fig.axes[0].get_title(), fig.axes[0].get_ylabel()
    ('Accuracy', 'accuracy')
    >>> plt.close(fig)
    """
    where = "plot_metric_over_time"
    y = _as_series(metric_values, "metric_values", where)
    xs = _x_values(x, y.shape[0], where)
    marks = _positions(retrains, where)
    fig, ax, created = _axes(ax, figsize)
    ax.plot(xs, y, color=_BLUE, marker="o", markersize=4, linewidth=1.8)
    if marks.size:
        ax.vlines(marks, 0, 1, transform=ax.get_xaxis_transform(), colors=_GREY,
                  linestyles="--", linewidth=1.0, label="retrain")
        ax.legend(loc="best")
    _decorate(ax, title, xlabel, ylabel, ylim)
    return _finish(fig, created, save_path, show)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), title: str = "",
                   ylabel: str = "", xlabel: str = "Chunk", x=None, ylim=None, ax=None,
                   figsize=(8, 4.5), save_path=None, show: bool = False):
    """Plot the same metric for two models on one set of axes.

    Parameters
    ----------
    metric1, metric2 : array-like of numbers, shape (n_chunks,)
        One value per chunk for each model. They may have different lengths.
        ``NaN`` leaves a gap.
    labels : sequence of two str, default ('Model 1', 'Model 2')
        Legend names, for example ``('buffer_size=100', 'buffer_size=500')``.
    title, ylabel, xlabel : str
        Texts of the plot.
    x : array-like, optional
        x positions shared by both series (needs the same length as both).
    ylim, ax, figsize, save_path, show
        As for ``plot_metric_over_time``.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If a series is empty or not 1-D, ``labels`` is not two strings, or ``x``
        has a different length.
    TypeError
        If a series does not hold numbers.

    Complexity
    ----------
    O(n1 + n2) for the two series, plus the cost of drawing.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig = compare_models([0.5, 0.6], [0.7, 0.8], ("small buffer", "large buffer"))
    >>> [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    ['small buffer', 'large buffer']
    >>> plt.close(fig)
    """
    where = "compare_models"
    if isinstance(labels, str) or len(labels) != 2 or not all(isinstance(s, str) for s in labels):
        raise ValueError(f"{where}: labels must be two strings, got {labels!r}.")
    first = _as_series(metric1, "metric1", where)
    second = _as_series(metric2, "metric2", where)
    x1 = _x_values(x, first.shape[0], where)
    x2 = _x_values(x, second.shape[0], where)
    fig, ax, created = _axes(ax, figsize)
    ax.plot(x1, first, color=_BLUE, marker="o", markersize=4, linewidth=1.8, linestyle="-", label=labels[0])
    ax.plot(x2, second, color=_VERMILLION, marker="s", markersize=4, linewidth=1.8, linestyle="--", label=labels[1])
    ax.legend(loc="best")
    _decorate(ax, title, xlabel, ylabel, ylim)
    return _finish(fig, created, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, title: str = "", xlabel: str = "Sample in chunk",
                                     ylabel: str = "Class", ax=None, figsize=(8, 4.5),
                                     save_path=None, show: bool = False):
    """Show predicted classes against the true classes for one chunk.

    Every sample is one position on the x axis and every class one row on the y
    axis. A big open circle marks the TRUE class. A small blue dot inside it is a
    correct prediction; an orange-red cross on another row is a wrong one.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n,)
        True and predicted labels (numbers or strings), in the same order.
    title : str, optional
        Defaults to ``'Predictions vs ground truth: k of n correct'``.
    xlabel, ylabel : str
        Axis labels.
    ax, figsize, save_path, show
        As for ``plot_metric_over_time``.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the inputs are empty, not 1-D, or of different lengths.

    Complexity
    ----------
    O(n log n) for n samples (the classes are found with a sort), plus drawing.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig = plot_predictions_vs_ground_truth([0, 1, 1, 0], [0, 1, 0, 0])
    >>> fig.axes[0].get_title()
    'Predictions vs ground truth: 3 of 4 correct'
    >>> plt.close(fig)
    """
    where = "plot_predictions_vs_ground_truth"
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"{where}: y_true and y_pred must be 1-D, got shapes {y_true.shape} and {y_pred.shape}.")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"{where}: y_true has shape {y_true.shape} but y_pred has shape {y_pred.shape}.")
    if y_true.size == 0:
        raise ValueError(f"{where}: there are no samples to plot.")
    n = y_true.shape[0]
    classes, codes = np.unique(np.concatenate([y_true, y_pred]), return_inverse=True)
    codes = codes.ravel()
    true_row, pred_row = codes[:n], codes[n:]
    right = true_row == pred_row
    position = np.arange(n)
    fig, ax, created = _axes(ax, figsize)
    ax.scatter(position, true_row, s=110, facecolors="none", edgecolors=_GREY, linewidths=1.4,
               label="true class")
    ax.scatter(position[right], pred_row[right], s=20, color=_BLUE, label=f"correct prediction ({right.sum()})")
    ax.scatter(position[~right], pred_row[~right], s=55, color=_VERMILLION, marker="x", linewidths=2,
               label=f"wrong prediction ({(~right).sum()})")
    ax.set_yticks(np.arange(classes.shape[0]))
    ax.set_yticklabels([str(c) for c in classes])
    ax.set_ylim(-0.6, classes.shape[0] - 0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)  # below the axes: keeps the points visible
    heading = title or f"Predictions vs ground truth: {int(right.sum())} of {n} correct"
    _decorate(ax, heading, xlabel, ylabel, None)
    return _finish(fig, created, save_path, show)