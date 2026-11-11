"""Compare the vectorised streaming components with plain-Python loop versions.

Four pieces of numcompute_stream are timed against a loop that does the same job:

1. the tree's split search          (_best_split vs loops over samples with running counts)
2. the tree's predict               (all rows level by level vs one row at a time)
3. the streaming confusion matrix   (one chunk update vs counting sample by sample)
4. the running mean and variance    (chunk merge vs Welford's update for every sample)

`compare` (from NumCompute's benchmarking module) checks that both versions give
the same answer before it reports a speedup, so a fast but wrong version cannot
appear in the table. All data are generated with fixed seeds, so every run times
the same work. Run from the repository root:

    python -m benchmark.stream_loop_vs_vectorised

The table is printed and also written to benchmark/results/loop_vs_vectorised.md.
"""

import math
import os
import platform
from pathlib import Path

import numpy as np

from numcompute.benchmarking import compare, format_results_table
from numcompute_stream.metrics import StreamingConfusionMatrix
from numcompute_stream.stats import RunningStats
from numcompute_stream.tree import DecisionTreeClassifier, _best_split

REPEATS = 10
SEED = 0
EPS = 1e-12  # the tree treats gains closer than this as equal
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _gini(counts):
    """Gini impurity of a list of class counts (0 for an empty group)."""
    total = sum(counts)
    if total == 0:
        return 0.0
    return 1.0 - sum((c / total) ** 2 for c in counts)


def build_cases():
    """Return (label, vectorised_fn, loop_fn) triples built on seeded random data."""
    rng = np.random.default_rng(SEED)

    # ---- 1. split search at the root of a tree --------------------------------
    n, n_features, n_classes = 1500, 5, 3
    X = rng.normal(size=(n, n_features))
    y = ((X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=n)) > 0).astype(int) + (X[:, 2] > 1.0)
    onehot = np.zeros((n, n_classes))
    onehot[np.arange(n), y] = 1.0
    features = np.arange(n_features)
    columns = X.T.tolist()
    labels = y.tolist()

    def split_vectorised():
        feature, threshold, _, score = _best_split(X, onehot, features, "gini")
        return np.array([feature, threshold, score])

    def split_loop():
        parent = _gini([labels.count(c) for c in range(n_classes)])
        best = (-math.inf, 0, 0.0)
        for f in range(n_features):
            values = columns[f]
            order = sorted(range(n), key=values.__getitem__)
            left = [0] * n_classes
            right = [labels.count(c) for c in range(n_classes)]
            for position in range(n - 1):
                row = order[position]
                left[labels[row]] += 1
                right[labels[row]] -= 1
                low, high = values[row], values[order[position + 1]]
                if not low < high:
                    continue  # equal values are never separated
                n_left = position + 1
                child = (n_left * _gini(left) + (n - n_left) * _gini(right)) / n
                gain = parent - child
                if gain > best[0] + EPS:  # ties keep the lowest feature, then the lowest cut
                    best = (gain, f, low + (high - low) / 2.0)
        return np.array([best[1], best[2], best[0]])

    # ---- 2. predict with a trained tree ---------------------------------------
    tree = DecisionTreeClassifier(max_depth=6).fit(X, y)
    X_new = rng.normal(size=(5000, n_features))
    rows = X_new.tolist()
    feature_of, threshold_of = tree.feature_.tolist(), tree.threshold_.tolist()
    left_of, right_of = tree.children_left_.tolist(), tree.children_right_.tolist()
    nan_left, counts_of, classes = tree.nan_left_.tolist(), tree.value_.tolist(), tree.classes_.tolist()

    def predict_vectorised():
        return tree.predict(X_new)

    def predict_loop():
        out = []
        for row in rows:
            node = 0
            while feature_of[node] >= 0:
                x = row[feature_of[node]]
                go_left = nan_left[node] if x != x else x <= threshold_of[node]
                node = left_of[node] if go_left else right_of[node]
            counts = counts_of[node]
            out.append(classes[counts.index(max(counts))])
        return np.array(out)

    # ---- 3. one chunk into the streaming confusion matrix ---------------------
    y_true = rng.integers(0, 3, size=20000)
    y_pred = np.where(rng.random(20000) < 0.8, y_true, rng.integers(0, 3, size=20000))
    true_list, pred_list = y_true.tolist(), y_pred.tolist()

    def confusion_vectorised():
        return StreamingConfusionMatrix(labels=[0, 1, 2]).update(y_true, y_pred).result()[0]

    def confusion_loop():
        counts = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        for t, p in zip(true_list, pred_list):
            counts[t][p] += 1
        return np.array(counts)

    # ---- 4. running mean and variance over 20 chunks --------------------------
    data = rng.normal(loc=50.0, scale=10.0, size=(20000, 4))
    chunks = np.array_split(data, 20)
    data_rows = data.tolist()

    def stats_vectorised():
        stats = RunningStats()
        for chunk in chunks:
            stats.update_stats(chunk)
        return np.vstack([stats.mean, stats.variance])

    def stats_loop():
        count, mean, m2 = 0, [0.0] * 4, [0.0] * 4
        for row in data_rows:  # Welford's update, one sample at a time
            count += 1
            for j in range(4):
                delta = row[j] - mean[j]
                mean[j] += delta / count
                m2[j] += delta * (row[j] - mean[j])
        return np.vstack([mean, [value / count for value in m2]])

    return [
        (f"tree split search, {n} samples x {n_features} features", split_vectorised, split_loop),
        (f"tree predict, {len(rows)} samples, depth {tree.get_depth()}", predict_vectorised, predict_loop),
        (f"streaming confusion matrix, {len(true_list)} samples", confusion_vectorised, confusion_loop),
        (f"running mean and variance, {len(data_rows)} samples in 20 chunks", stats_vectorised, stats_loop),
    ]


def main():
    header = [
        f"Python {platform.python_version()}, NumPy {np.__version__}",
        f"{platform.system()} {platform.release()} ({platform.machine()}), CPU cores: {os.cpu_count()}",
        f"Each time is the mean of {REPEATS} runs, in seconds. Seeded data (seed {SEED}).",
    ]
    results = []
    for label, vectorised, loop in build_cases():
        result = compare(vectorised, loop, repeats=REPEATS)
        result["name"] = label
        results.append(result)

    report = "\n".join(header) + "\n\n" + format_results_table(results) + "\n"
    print(report)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "loop_vs_vectorised.md").write_text(report, encoding="utf-8")
    print(f"Saved to {RESULTS_DIR / 'loop_vs_vectorised.md'}")


if __name__ == "__main__":
    main()