"""How does buffer_size change accuracy, retraining time and memory in a stream?

A DecisionTreeClassifier with buffered retraining collects chunks in a buffer and
retrains on the buffer when it holds buffer_size samples. A small buffer means
frequent, cheap retrains on little data; a large buffer means rare, costly
retrains on more data. This script measures that trade-off with StreamTrainer
(test on each chunk, then train on it) in three streams:

  stationary  synthetic stream, 20000 samples, the labelling rule never changes
  drift       the same stream, but the rule changes completely halfway
  housing     the real Housing.csv (price above or below the median), shuffled

Every buffer size sees exactly the same chunks. Synthetic streams are repeated with
5 seeds and Housing with 10 shuffles; the tables show the mean (and the standard
deviation of the accuracies). Run from the repository root:

    python -m benchmark.buffer_size_benchmark

It prints the tables and writes to benchmark/results/: buffer_size.md (tables and
environment notes), buffer_size.csv (every run), buffer_size.png (summary plots)
and drift_curves.png (rolling accuracy through the drift for two buffer sizes).

Reading the columns
-------------------
scored %          share of samples scored: nothing can be scored until the first tree exists
cumulative acc    accuracy over every scored sample (the buffer sizes score different samples)
last 25% acc      accuracy over the last quarter of the stream, scored by every buffer size
after drift acc   accuracy on the 2000 samples right after the rule changed
ms per retrain    mean time of one retrain; total s is all retrains of the stream
"""

import csv
import os
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from numcompute.io import read_csv
from numcompute_stream.stream import StreamTrainer, iter_chunks
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.visualise import compare_models

MAX_DEPTH = 5
N_SYNTHETIC = 20000
CHUNK_SYNTHETIC = 100
DRIFT_CHUNK = N_SYNTHETIC // CHUNK_SYNTHETIC // 2   # the rule changes at this chunk
AFTER_DRIFT_CHUNKS = 20
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_PATH = Path(__file__).resolve().parent.parent / "Dataset" / "Housing.csv"
NUMERIC = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
YES_NO = ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]


# ----------------------------------------------------------------------
# Streams
# ----------------------------------------------------------------------
def synthetic_stream(seed, drift):
    """20000 samples, 5 features; the label follows one rule, or another after the drift."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N_SYNTHETIC, 5))
    noise = 0.3 * rng.normal(size=N_SYNTHETIC)
    rule_a = X[:, 0] + 0.5 * X[:, 1] + noise > 0
    if not drift:
        return X, rule_a.astype(int)
    rule_b = X[:, 2] - 0.5 * X[:, 3] + noise > 0
    after = np.arange(N_SYNTHETIC) >= N_SYNTHETIC // 2
    return X, np.where(after, rule_b, rule_a).astype(int)


def load_housing():
    """Housing features and the label 'price at or above the median'.

    Missing numbers stay NaN (the tree sends them to the larger side of a split);
    a yes/no cell that is blank counts as 'no'; furnishing status is one-hot encoded.
    """
    if not DATA_PATH.is_file():
        raise SystemExit(f"Cannot find {DATA_PATH}. Put Housing.csv in the Dataset folder.")
    numeric, _ = read_csv(DATA_PATH, usecols=NUMERIC)
    text, _ = read_csv(DATA_PATH, usecols=YES_NO + ["furnishingstatus"], dtype=str)
    label = (numeric[:, 0] >= np.median(numeric[:, 0])).astype(int)
    flags = (np.char.lower(np.char.strip(text[:, : len(YES_NO)])) == "yes").astype(float)
    furnishing = text[:, -1]
    one_hot = (furnishing[:, None] == np.unique(furnishing)[None, :]).astype(float)
    return np.hstack([numeric[:, 1:], flags, one_hot]), label


def build_scenarios():
    """The three experiments: how to make a stream, which buffer sizes, chunk size, repeats."""
    housing_X, housing_y = load_housing()

    def housing_stream(seed):
        order = np.random.default_rng(seed).permutation(len(housing_y))
        return housing_X[order], housing_y[order]

    return [
        dict(key="stationary", title="Synthetic stream, the rule never changes", make=lambda s: synthetic_stream(s, False),
             buffers=[100, 200, 500, 1000, 2000, 5000], chunk=CHUNK_SYNTHETIC, seeds=range(5), drift=None),
        dict(key="drift", title="Synthetic stream, the rule changes halfway", make=lambda s: synthetic_stream(s, True),
             buffers=[100, 200, 500, 1000, 2000, 5000], chunk=CHUNK_SYNTHETIC, seeds=range(5), drift=DRIFT_CHUNK),
        dict(key="housing", title="Housing.csv (545 houses), shuffled", make=housing_stream,
             buffers=[25, 50, 100, 200, 400], chunk=25, seeds=range(10), drift=None),
    ]


# ----------------------------------------------------------------------
# One run and its measurements
# ----------------------------------------------------------------------
def window_accuracy(trainer, start, stop):
    """Accuracy over chunks start..stop-1 (NaN if the model could not predict in all of them)."""
    scored = trainer.history("scored")[start:stop]
    if not scored.all():
        return float("nan")
    accuracy = trainer.history("chunk_accuracy")[start:stop]
    samples = trainer.history("n_samples")[start:stop]
    return float(np.sum(accuracy * samples) / np.sum(samples))


def run_once(X, y, buffer_size, chunk, drift):
    """Stream (X, y) through a buffered tree and return the measurements as a dict."""
    trainer = StreamTrainer(DecisionTreeClassifier(max_depth=MAX_DEPTH, buffer_size=buffer_size))
    trainer.run(iter_chunks(X, y, chunk))
    n_chunks = len(trainer.log_)
    retrained = trainer.history("retrained")
    retrain_seconds = trainer.history("fit_seconds")[retrained]
    row = {
        "retrains": int(retrained.sum()),
        "scored_pct": 100.0 * trainer.accuracy_.n_samples_ / len(y),
        "cumulative_acc": float(trainer.history("cumulative_accuracy")[-1]),
        "last_quarter_acc": window_accuracy(trainer, (3 * n_chunks) // 4, n_chunks),
        "after_drift_acc": float("nan"),
        "ms_per_retrain": 1000.0 * float(retrain_seconds.mean()) if retrain_seconds.size else float("nan"),
        "total_retrain_s": float(retrain_seconds.sum()),
        "peak_memory_kb": float(trainer.history("memory_bytes").max()) / 1024.0,
    }
    if drift is not None:
        row["after_drift_acc"] = window_accuracy(trainer, drift, drift + AFTER_DRIFT_CHUNKS)
    return row


def run_scenario(scenario):
    """Run every buffer size on every seed; return the list of per-run dicts."""
    runs = []
    for seed in scenario["seeds"]:
        X, y = scenario["make"](seed)
        for buffer_size in scenario["buffers"]:
            row = run_once(X, y, buffer_size, scenario["chunk"], scenario["drift"])
            runs.append({"scenario": scenario["key"], "seed": seed, "buffer_size": buffer_size, **row})
    return runs


# ----------------------------------------------------------------------
# Tables, files and plots
# ----------------------------------------------------------------------
def summarise(runs, buffer_size, name):
    """Mean and standard deviation of one measurement over the seeds for one buffer size."""
    values = np.array([r[name] for r in runs if r["buffer_size"] == buffer_size], dtype=float)
    return float(values.mean()), float(values.std())


def markdown_table(scenario, runs):
    """One Markdown table for a scenario."""
    has_drift = scenario["drift"] is not None
    head = ["buffer_size", "retrains", "scored %", "cumulative acc", "last 25% acc"]
    head += ["after drift acc"] if has_drift else []
    head += ["ms per retrain", "total s", "peak memory (KB)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for b in scenario["buffers"]:
        def mean(name):
            return summarise(runs, b, name)[0]

        def acc(name):
            m, s = summarise(runs, b, name)
            return f"{m:.3f} ± {s:.3f}"

        cells = [str(b), f"{mean('retrains'):.0f}", f"{mean('scored_pct'):.1f}", acc("cumulative_acc"), acc("last_quarter_acc")]
        cells += [acc("after_drift_acc")] if has_drift else []
        cells += [f"{mean('ms_per_retrain'):.2f}", f"{mean('total_retrain_s'):.3f}", f"{mean('peak_memory_kb'):.0f}"]
        lines.append("| " + " | ".join(cells) + " |")
    n_seeds = len(list(scenario["seeds"]))
    return f"### {scenario['title']}\n\nChunk size {scenario['chunk']}, mean of {n_seeds} runs.\n\n" + "\n".join(lines)


def save_csv(runs, path):
    """Write every single run to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)


def plain_log_axis(ax, axis, ticks=None):
    """Log scale with plain numbers (100, 1000, 2.5) instead of powers of ten."""
    plain = FuncFormatter(lambda value, _: f"{value:g}")
    if axis == "x":
        ax.set_xscale("log")
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(plain)
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(NullFormatter())


def plot_summary(scenarios, all_runs, path):
    """Accuracy (top row) and mean retrain time (bottom row) against buffer_size, one column per scenario."""
    fig, axes = plt.subplots(2, len(scenarios), figsize=(4.6 * len(scenarios), 7), squeeze=False)
    for column, (scenario, runs) in enumerate(zip(scenarios, all_runs)):
        buffers = scenario["buffers"]
        top, bottom = axes[0, column], axes[1, column]
        series = [("last_quarter_acc", "last 25% of the stream", "#0072B2", "o", "-"),
                  ("cumulative_acc", "all scored samples", "#009E73", "s", ":")]
        if scenario["drift"] is not None:
            series.append(("after_drift_acc", "just after the drift", "#D55E00", "^", "--"))
        for name, label, colour, marker, style in series:
            top.plot(buffers, [summarise(runs, b, name)[0] for b in buffers], color=colour, marker=marker,
                     linestyle=style, linewidth=1.8, label=label)
        top.set_title(scenario["title"], fontsize=10)
        top.set_ylabel("accuracy")
        plain_log_axis(top, "x", buffers)
        top.grid(True, alpha=0.3)
        top.legend(fontsize=8, loc="lower left" if scenario["drift"] is not None else "lower right")
        bottom.plot(buffers, [summarise(runs, b, "ms_per_retrain")[0] for b in buffers], color="#555555", marker="o", linewidth=1.8)
        bottom.set_xlabel("buffer_size (samples)")
        bottom.set_ylabel("mean time of one retrain (ms)")
        plain_log_axis(bottom, "x", buffers)
        plain_log_axis(bottom, "y")
        bottom.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_drift_curves(path, small=200, large=2000, window=1000):
    """Rolling accuracy through the drift for a small and a large buffer (seed 0)."""
    X, y = synthetic_stream(0, drift=True)
    curves = []
    for buffer_size in (small, large):
        trainer = StreamTrainer(DecisionTreeClassifier(max_depth=MAX_DEPTH, buffer_size=buffer_size), window=window)
        trainer.run(iter_chunks(X, y, CHUNK_SYNTHETIC))
        curves.append(trainer.history("rolling_accuracy"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    compare_models(curves[0], curves[1], (f"buffer_size={small}", f"buffer_size={large}"),
                   f"Rolling accuracy over the last {window} samples; the rule changes at chunk {DRIFT_CHUNK}",
                   "accuracy", ax=ax)
    ax.axvline(DRIFT_CHUNK, color="#555555", linestyle=":", linewidth=1.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    scenarios = build_scenarios()
    run_once(*synthetic_stream(0, False), 500, CHUNK_SYNTHETIC, None)   # warm-up, not recorded
    all_runs = [run_scenario(s) for s in scenarios]

    header = [
        f"Python {platform.python_version()}, NumPy {np.__version__}",
        f"{platform.system()} {platform.release()} ({platform.machine()}), CPU cores: {os.cpu_count()}",
        f"Tree: max_depth={MAX_DEPTH}, gini. Each chunk is tested first, then trained on. Seeded data.",
    ]
    report = "\n".join(header) + "\n\n" + "\n\n".join(markdown_table(s, r) for s, r in zip(scenarios, all_runs)) + "\n"
    print(report)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "buffer_size.md").write_text(report, encoding="utf-8")
    save_csv([row for runs in all_runs for row in runs], RESULTS_DIR / "buffer_size.csv")
    plot_summary(scenarios, all_runs, RESULTS_DIR / "buffer_size.png")
    plot_drift_curves(RESULTS_DIR / "drift_curves.png")
    print(f"Saved buffer_size.md, buffer_size.csv, buffer_size.png and drift_curves.png in {RESULTS_DIR}")


if __name__ == "__main__":
    main()