# NumCompute

A small scientific-computing toolkit written with **plain Python and NumPy only** - no pandas, scikit-learn or other data or machine-learning libraries.

It covers loading and cleaning data, sorting and ranking, descriptive statistics, evaluation metrics, numerical gradients and a simple pipeline. It also includes benchmarks that compare vectorised NumPy code with ordinary Python loops.

Built for **COMP 5004 - Programming Task 1** by group Beta.

The streaming extension [`numcompute_stream`](#streaming-extension-numcompute_stream) (COMP 5004 Assignment 2, individual work by Melvin) builds on it for data that arrives in chunks.

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Conventions](#conventions)
- [Modules](#modules)
- [Performance](#performance)
- [Demo notebook](#demo-notebook)
- [Streaming extension (numcompute_stream)](#streaming-extension-numcompute_stream)
- [Running the tests](#running-the-tests)
- [Project layout](#project-layout)
- [Team](#team)
- [Troubleshooting](#troubleshooting)

## Installation

You need **Python 3.10 or newer**. NumPy (2.0 or newer) and matplotlib are installed for you.

```bash
# 1. Get the code
git clone -b melvin/a2 https://github.com/melvin14120/2650_COMP_5004_2_Assignment-2.1-Programming-Task-1-group-Beta.git
cd 2650_COMP_5004_2_Assignment-2.1-Programming-Task-1-group-Beta

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
source .venv/bin/activate         # macOS / Linux

# 3. Install the package, plus the tools for running the tests
python -m pip install -e ".[dev]"
```

- `-e` means "editable": changes you make to the code take effect straight away.
- To use the toolkit without the test tools, run `python -m pip install -e .` instead. This installs only NumPy and matplotlib.
- Check that it worked: `python -c "import numcompute; print('ok')"`

## Quickstart

```python
import numpy as np

from numcompute.metrics import accuracy, auc, f1, roc_curve
from numcompute.rank import rank
from numcompute.stats import describe, quantile
from numcompute.utils import euclidean_distance, topk_indices

# 1. Summarise a table that has a missing value (NaN is ignored).
X = np.array([[1.0, 10.0],
              [2.0, np.nan],
              [3.0, 30.0],
              [4.0, 40.0]])
print(np.round(describe(X, axis=0)["mean"], 2))   # one mean per column
print(quantile(X, 0.5, axis=0))                   # one median per column

# 2. Rank values. Tied values share the average of their ranks.
print(rank([10, 20, 20, 30]))

# 3. Nearest neighbours: build a distance matrix, then keep the 2 closest per row.
points = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [0.4, 0.0]])
distances = euclidean_distance(points, points)
print(topk_indices(distances, 2, largest=False))  # each point, then its nearest neighbour

# 4. Score a classifier.
y_true = np.array([0, 1, 1, 0, 1, 0])
scores = np.array([0.2, 0.9, 0.6, 0.6, 0.4, 0.1])
y_pred = (scores >= 0.5).astype(int)
print(round(accuracy(y_true, y_pred), 3), round(f1(y_true, y_pred), 3))
fpr, tpr, _ = roc_curve(y_true, scores)
print(round(auc(fpr, tpr), 3))
```

Output:

```text
[ 2.5  26.67]
[ 2.5 30. ]
[1.  2.5 2.5 4. ]
[[0 3]
 [1 3]
 [2 1]
 [3 0]]
0.667 0.667
0.833
```

For the full workflow - loading a CSV, cleaning it and trying every module - see the [demo notebook](#demo-notebook).

## Conventions

These rules hold across the whole toolkit, so you only need to learn them once.

- **Shapes.** A data table `X` has shape `(n_samples, n_features)`: one row per sample, one column per feature. Labels and scores (`y_true`, `y_pred`, `y_score`) are 1-D arrays with one entry per sample. Distance functions take two tables, `A` of shape `(n, d)` and `B` of shape `(m, d)`.
- **`axis`.** `axis=0` goes down the rows and gives one result per **column** (feature). `axis=1` gives one result per **row** (sample). Negative numbers count from the end, so `-1` is the last axis. Where a function allows `axis=None`, it uses every value. Defaults: `describe` uses `axis=0`; `softmax`, `log_softmax` and `topk_indices` use `axis=-1`; `quantile`, `percentile` and `logsumexp` use `axis=None`.
- **Missing values.** `NaN` means "missing". Statistics functions ignore it, `rank` and `topk_indices` put it last, and functions that cannot work with it raise a `ValueError`. The tables below give the exact rule for each function.
- **Ties.** Equal values are handled by a stated rule: ranks use `average`, `dense` or `ordinal`; `topk_indices` takes the lower index first; ROC curves merge tied scores into one point.
- **Errors.** Wrong shapes, empty arrays and invalid options raise a `ValueError` with a message saying what is wrong.
- **fit / transform.** Preprocessing classes learn from data with `fit(X)`, apply what they learned with `transform(X)`, and `fit_transform(X)` does both in one step.

## Modules

Each module has a short summary and a table of its main functions. The last column gives the NaN and tie rule.

### io

Reads CSV files into NumPy arrays. Blank cells become `NaN` instead of causing an error, the delimiter can be changed (comma, tab and so on), and very large files can be read in chunks so they never have to fit in memory at once.

| Function                                | What it does                                        | NaN and ties                    |
| --------------------------------------- | --------------------------------------------------- | ------------------------------- |
| `read_csv(path, ...)`                   | Load a whole CSV file; returns `(X, column_names)`. | Blank or NA cells become `NaN`. |
| `read_csv_chunks(path, chunksize, ...)` | Read a large CSV one chunk at a time.               | Same as `read_csv`.             |

### preprocessing

Cleans and rescales features before analysis. The imputer fills in missing values, the two scalers put every column on a common scale, and the encoder turns category labels into 0/1 columns. Each class follows the `fit` / `transform` pattern, so what it learns from one dataset can be applied to another.

| Class                | What it does                                           | NaN and ties                       |
| -------------------- | ------------------------------------------------------ | ---------------------------------- |
| `SimpleImputer(...)` | Fill missing values, for example with the column mean. | Replaces `NaN`.                    |
| `StandardScaler()`   | Give each column mean 0 and standard deviation 1.      | Fill `NaN` first with the imputer. |
| `MinMaxScaler(...)`  | Rescale each column to a range such as 0 to 1.         | Fill `NaN` first with the imputer. |
| `OneHotEncoder()`    | Turn each category into its own 0/1 column.            | n/a                                |

### sort_search

Sorting, selecting and searching. `stable_sort` and `multi_key_sort` put values in order (by several columns if needed), `topk` finds the k largest or smallest values without sorting everything, `quickselect` is a hand-written version of the same idea, and `binary_search` finds where a value belongs in sorted data.

| Function              | What it does                                                   | NaN and ties                                      |
| --------------------- | -------------------------------------------------------------- | ------------------------------------------------- |
| `stable_sort(...)`    | Sort values; equal values keep their original order.           | Ties keep their original order.                   |
| `multi_key_sort(...)` | Sort rows by several columns.                                  | A tie in one column is broken by the next column. |
| `topk(...)`           | The k largest or smallest values and their indices.            | See the function's docstring.                     |
| `quickselect(...)`    | The k-th smallest value, written by hand.                      | n/a                                               |
| `binary_search(...)`  | Where a value belongs in sorted data, and whether it is there. | With duplicates, returns the first position.      |

### rank

Ranking with a clear rule for ties, and percentiles. `rank` offers three tie methods, and `percentile` returns the value at a given percentile.

| Function                                                 | What it does                                     | NaN and ties                                                                                     |
| -------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| `rank(data, method="average")`                           | Rank of each value; 1 is the smallest.           | Ties follow `method`: `average`, `dense` or `ordinal`. NaN ranks last (all NaNs share one rank). |
| `percentile(data, q, interpolation="linear", axis=None)` | Value at a percentile; `q` is between 0 and 100. | NaN is ignored.                                                                                  |

### stats

Summary statistics. `describe` gives the mean, median, standard deviation, minimum and maximum along an axis. `StreamingStats` keeps a running mean and variance while values arrive one at a time, so the whole dataset never has to be in memory. `histogram` and `quantile` complete the set.

| Function                                               | What it does                                                                                   | NaN and ties                                                                                             |
| ------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `describe(X, axis=0)`                                  | Mean, median, std, min and max along an axis, returned as a dict.                              | NaN is ignored. A column that is all NaN gives NaN.                                                      |
| `StreamingStats` (also named `WelfordAccumulator`)     | Running mean and variance: `update(x)` adds a value, `merge(other)` combines two accumulators. | NaN values are skipped and not counted. With no values, `mean`, `variance` and `std` raise `ValueError`. |
| `histogram(data, bins=10, range=None)`                 | Number of values in each bin.                                                                  | NaN is dropped before binning.                                                                           |
| `quantile(data, q, interpolation="linear", axis=None)` | Value at a quantile; `q` is between 0 and 1.                                                   | NaN is ignored.                                                                                          |

### metrics

Scores for judging predictions: a confusion matrix, accuracy, precision, recall and F1 for classification, mean squared error for regression, and an ROC curve with its area (AUC) for binary classifiers.

| Function                                                                                             | What it does                                                   | NaN and ties                                                                                                                            |
| ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `confusion_matrix(y_true, y_pred, labels=None)`                                                      | Counts of (true, predicted) pairs; returns `(matrix, labels)`. | NaN labels raise `ValueError`.                                                                                                          |
| `accuracy(y_true, y_pred)`                                                                           | Fraction of predictions that are correct.                      | n/a                                                                                                                                     |
| `precision`, `recall`, `f1` (arguments `y_true, y_pred, average="binary", pos_label=1, labels=None`) | `average` is `binary`, `macro` or `micro`.                     | A class with no predictions or no samples scores `0.0`, never NaN. `binary` needs at most 2 classes, one of them `pos_label`.           |
| `mse(y_true, y_pred)`                                                                                | Mean squared error.                                            | n/a                                                                                                                                     |
| `roc_curve(y_true, y_score)`                                                                         | ROC curve for 0/1 labels; returns `(fpr, tpr, thresholds)`.    | Tied scores are merged into one point, so row order does not matter. NaN scores are not checked (they rank last), so remove them first. |
| `auc(fpr, tpr)`                                                                                      | Area under a curve.                                            | `fpr` must be monotonic with at least 2 points.                                                                                         |

### optim

Numerical derivatives. `grad` estimates the gradient of a function using finite differences (forward or central), and `jacobian` does the same for functions that return a vector.

| Function                               | What it does                                               | NaN and ties |
| -------------------------------------- | ---------------------------------------------------------- | ------------ |
| `grad(f, x, h=1e-5, method="central")` | Gradient of a scalar function at `x`.                      | n/a          |
| `jacobian(F, x, ...)`                  | Jacobian of a vector function at `x`, with shape `(m, n)`. | n/a          |

### pipeline

Chains steps together. A `Pipeline` runs steps one after another (for example impute, then scale). `FeatureUnion` runs steps side by side on the same input and joins the results column by column.

| Class               | What it does                                                                 | NaN and ties |
| ------------------- | ---------------------------------------------------------------------------- | ------------ |
| `Pipeline(steps)`   | Run steps in order with `fit`, `transform`, `fit_transform` and `predict`.   | n/a          |
| `FeatureUnion(...)` | Run steps in parallel on the same input and join their outputs side by side. | n/a          |

### utils

Shared building blocks used by the other modules: pairwise distances, activation functions, numerically stable `softmax` and `logsumexp`, safe division, a top-k helper for nearest neighbours, and a batch iterator.

| Function                                                                        | What it does                                                                                       | NaN and ties                                                                                                                                                                     |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `euclidean_distance(A, B)`, `manhattan_distance(A, B)`, `cosine_distance(A, B)` | Distance from every row of `A` to every row of `B`; result has shape `(n, m)`.                     | A NaN only affects the distances of its own point.                                                                                                                               |
| `sigmoid(x)`, `relu(x)`                                                         | Activation functions.                                                                              | NaN stays NaN.                                                                                                                                                                   |
| `softmax(x, axis=-1)`, `log_softmax(x, axis=-1)`, `logsumexp(x, axis=None)`     | Stable versions that do not overflow on large values.                                              | `-inf` and `+inf` are handled without warnings. A slice that is all `-inf` gives NaN in `softmax` and `log_softmax`, and `-inf` in `logsumexp`. Empty input raises `ValueError`. |
| `safe_divide(num, denom, fill=0.0)`                                             | Division that returns `fill` where the denominator is 0.                                           | NaN stays NaN.                                                                                                                                                                   |
| `topk_indices(X, k, axis=-1, largest=True)`                                     | Indices of the k largest values; use `largest=False` for the smallest, such as nearest neighbours. | Ties: lower index first. NaN ranks last.                                                                                                                                         |
| `batch_iterator(X, batch_size)`                                                 | Yield the data in consecutive batches.                                                             | n/a                                                                                                                                                                              |

### benchmarking

Timing tools. `benchmark` times one function, `compare` times a vectorised and a loop version of the same computation and reports the speedup, and `format_results_table` prints the results as a Markdown table.

| Function                                             | What it does                                                   | NaN and ties                                                                      |
| ---------------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `benchmark(func, *args, repeats=10, warmup=1)`       | Time one function; returns mean, std, min, max and the result. | n/a                                                                               |
| `compare(vectorized_fn, loop_fn, *args, repeats=10)` | Time two versions and report the speedup.                      | Raises if the two results differ in shape or value. Matching NaNs count as equal. |
| `format_results_table(results)`                      | Turn `compare` results into a Markdown table.                  | n/a                                                                               |

## Performance

Vectorised code does the looping inside NumPy instead of in Python. The script `benchmark/run_benchmarks.py` times three NumCompute functions against hand-written plain-Python loops that compute the same result, and reports how much faster the vectorised version is.

### What is measured

| Case                                 | Vectorised version                  | Loop version                                                               | Input data                                                       |
| ------------------------------------ | ----------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| StandardScaler (20,000 x 5)          | `StandardScaler().fit_transform(X)` | Three passes over each column: mean, then standard deviation, then scaling | `X`: 20,000 × 5 random normal values                             |
| euclidean_distance (150 x 150 pairs) | `euclidean_distance(A, B)`          | Three nested loops: row of `A`, row of `B`, feature                        | `A` and `B`: 150 × 3 random normal values each                   |
| rank, average ties (1,500 values)    | `rank(x, method="average")`         | Compares every value with every other value                                | `x`: 1,500 random integers from 0 to 499, so there are many ties |

How the cases are run:

- The data comes from `np.random.default_rng(0)`, so every run uses exactly the same inputs.
- Each function is called once as a warm-up and then timed over 5 runs. The table shows the mean time in seconds.
- Before reporting a speedup, `compare()` checks that both versions return the same result (same shape, and equal under `np.allclose(..., equal_nan=True)`), so a fast but wrong version cannot appear in the table.

### Results

The latest run, saved in `benchmark/results.md`:

| Function                             | Vectorized (s) | Loop (s) | Speedup  |
| ------------------------------------ | -------------- | -------- | -------- |
| StandardScaler (20,000 x 5)          | 0.04245        | 0.8926   | 21.03x   |
| euclidean_distance (150 x 150 pairs) | 0.003185       | 0.562    | 176.48x  |
| rank, average ties (1,500 values)    | 0.002363       | 4.062    | 1719.50x |

**Environment**

- Date: 2026-10-24
- Python: 3.14.6
- NumPy: 2.5.1
- OS: Windows-11-10.0.26300-SP0
- CPU: Intel64 Family 6 Model 158 Stepping 10, GenuineIntel (8 logical cores)
- Method: mean of 5 timed runs after 1 warm-up run; fixed random seed 0

### How to read the results

- **Speedup** is the loop time divided by the vectorised time. `21.03x` means the vectorised version needed about 1/21 of the loop's time.
- The `StandardScaler` and `euclidean_distance` loops do the same amount of work as the vectorised versions (same time complexity), so their speedups come from vectorisation alone: NumPy runs the loops in compiled code instead of the Python interpreter. The vectorised `StandardScaler` even does extra work for NaN handling and overflow safety.
- The `rank` loop also uses a slower algorithm: it compares every pair of values (O(n²)), while `rank` sorts once (O(n log n)). Its very large speedup combines both effects.
- Exact times differ between machines and between runs. The pattern is what matters.

### Reproducing the benchmarks

From the repository root:

```bash
python -m benchmark.run_benchmarks
```

The script prints the table and the environment notes, and overwrites `benchmark/results.md` with them. If you re-run it, copy the new table into this section as well, so the README and `results.md` match. The last section of the [demo notebook](#demo-notebook) runs the same cases (`run_all(seed=0, repeats=5)`) and shows the table inline.

## Demo notebook

`demo/quickstart.ipynb` walks through the toolkit on a small dataset: reading a CSV with missing values, cleaning it, sorting and searching, ranking, statistics, gradients and a benchmark. Its last section runs the same benchmark cases as `benchmark/run_benchmarks.py`.

1. Install the package (see [Installation](#installation)). The notebook must use the same environment.
2. Install Jupyter, which is only needed for the notebook: `python -m pip install notebook`
3. From the repository root, open it: `python -m notebook demo/quickstart.ipynb`
4. In the notebook, choose **Run > Run All Cells**.

In VS Code, open `demo/quickstart.ipynb`, click **Select Kernel** (top right), pick the `.venv` interpreter, and choose **Run All**.

Open the notebook from inside the cloned repository so that any relative paths it uses (for example to the `Dataset/` folder) resolve correctly.

## Streaming extension (numcompute_stream)

`numcompute_stream/` is **Assignment 2** (individual work by Melvin). It extends NumCompute for data that arrives **in chunks**: every component can be updated with `partial_fit` or `update_stats`, and a single decision tree is retrained from a buffer as data comes in. It uses only NumPy and matplotlib. It reuses the group's work: the CSV reader (`numcompute.io`), the A1 transformers and `Pipeline` (extended with `partial_fit`) and the benchmark tools (`numcompute.benchmarking`).

### Streaming quickstart

Run this from the repository root. It streams 1,000 samples (about 5% of the values missing) through an imputer, a scaler and a tree that retrains every 200 samples.

```python
import numpy as np

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler
from numcompute_stream.stream import StreamTrainer, iter_chunks
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.visualise import plot_metric_over_time

rng = np.random.default_rng(0)
X = rng.normal(size=(1000, 3))
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X[rng.random(X.shape) < 0.05] = np.nan          # about 5% of the cells are missing

pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("tree", DecisionTreeClassifier(max_depth=4, buffer_size=200)),
])
trainer = StreamTrainer(pipeline)

for X_chunk, y_chunk in iter_chunks(X, y, chunk_size=100, shuffle=True, random_state=0):
    row = trainer.process_chunk(X_chunk, y_chunk)
    print(row["chunk"], row["scored"], round(row["cumulative_accuracy"], 3), row["n_retrains"])

print("accuracy over all scored samples:", round(trainer.accuracy_.result(), 3))
plot_metric_over_time(trainer.history("cumulative_accuracy"), "Cumulative accuracy", "Accuracy",
                      save_path="accuracy.png")
```

Output (columns: chunk number, was it scored, cumulative accuracy, number of retrains so far):

```text
0 False nan 0
1 False nan 1
2 True 0.9 1
3 True 0.905 2
4 True 0.927 2
5 True 0.917 3
6 True 0.918 3
7 True 0.915 4
8 True 0.906 4
9 True 0.895 5
accuracy over all scored samples: 0.895
```

Chunks 0 and 1 are not scored because the tree does not exist until the buffer holds 200 samples (after chunk 1). After that it retrains every two chunks. The last line saves the accuracy plot as `accuracy.png` in the current folder.

### How it works

- **Test, then train.** `StreamTrainer.process_chunk(X, y)` first scores the chunk with the model as it is now, so every chunk is unseen data when it is scored, and then calls `partial_fit` on it. It adds one row per chunk to `trainer.log_`: `chunk`, `n_samples`, `scored`, `chunk_accuracy`, `cumulative_accuracy`, `rolling_accuracy` (when `window` is set), `fitted`, `fit_seconds`, `retrained`, `n_retrains`, `n_seen`, `memory_bytes`, and one column for each extra metric you pass in. `trainer.history("name")` returns a column as an array.
- **Buffered retraining.** `DecisionTreeClassifier.partial_fit` adds each chunk to a buffer. When the buffer holds at least `buffer_size` samples, the tree is rebuilt from the buffer alone and the buffer is emptied. Between retrains `predict` uses the latest tree. Before the first retrain `predict` raises `NotFittedError`, and the trainer logs those chunks as not scored.
- **Vectorised split search.** The tree sorts each feature once and scores every possible cut of every feature from cumulative class counts in one array expression. Python loops run only over tree nodes (when building) and tree levels (when predicting), never over samples.
- **Missing values and ties.** NaN is ignored when looking for thresholds. At each split NaN rows follow the larger child, so NaN is also accepted when predicting. Equal values are never separated. Splits with equal gain go to the lowest feature, then the lowest threshold, so results are deterministic. Zero-gain splits are allowed, otherwise XOR-like data could never be learned.
- **Streaming safety.** Empty chunks are ignored, and changing the number of features or passing the wrong shape or type raises a `ValueError` that says what is wrong. The tests turn every NumPy `RuntimeWarning` into an error, so the tested paths (empty chunks, all-NaN columns, constant columns, single-class chunks) run without numerical warnings.

### Modules

Data tables have shape `(n_samples, n_features)` and labels are 1-D, as in the rest of NumCompute.

| Module          | Main classes and functions                                                                                                                                                                                                               | Streaming behaviour                                                                                                                                                                                                                                         |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tree`          | `DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=2, max_features=None, buffer_size=500, random_state=None)` with `fit`, `partial_fit`, `flush`, `predict`, `predict_proba`, `score`, `get_depth`, `memory_bytes` | Buffers chunks and retrains when the buffer holds `buffer_size` samples. `flush()` retrains now on a partly filled buffer. `n_retrains_`, `n_seen_` and `n_buffered_` track progress.                                                                       |
| `stream`        | `StreamTrainer(model, metrics=None, window=None)` with `score_chunk`, `fit_chunk`, `process_chunk`, `run`, `history`, `columns`, `reset`; `iter_chunks(X, y, chunk_size, shuffle=False, random_state=None)`                              | Test-then-train logging of accuracy, retrain times and memory. `metrics` is a dict of streaming metric objects.                                                                                                                                             |
| `preprocessing` | `StandardScaler`, `SimpleImputer`, `OneHotEncoder`                                                                                                                                                                                       | `partial_fit` keeps a running mean and variance (scaler), a running mean, median, most frequent value or constant (imputer; the median is exact up to `capacity` rows), and a growing list of categories (encoder, which warns when its output gets wider). |
| `stats`         | `RunningStats(ddof=0)`, `StreamingQuantile(capacity, mode)`, `StreamingHistogram(bins, range, window)`                                                                                                                                   | `update_stats(chunk)` merges a chunk. `RunningStats` gives mean, variance, std, min, max and can `merge` another. Quantiles are exact until `capacity` rows, then approximate (`mode="reservoir"` or a sliding `"window"`).                                 |
| `metrics`       | `StreamingAccuracy`, `StreamingConfusionMatrix`, `StreamingPrecision`, `StreamingRecall`, `StreamingF1`, `StreamingMSE`, `StreamingAUC`                                                                                                  | `update(y_true, y_pred)`, `result()`, `reset()`. An optional `window=` gives a rolling value.                                                                                                                                                               |
| `pipeline`      | `Pipeline`, `FeatureUnion`                                                                                                                                                                                                               | `partial_fit` passes each chunk through the steps (transformers learn, then the final model buffers it).                                                                                                                                                    |
| `visualise`     | `plot_metric_over_time`, `compare_models`, `plot_predictions_vs_ground_truth`                                                                                                                                                            | Return a matplotlib `Figure`. Options: `save_path=`, `show=`, `ax=` (draw in a subplot), `retrains=` (dashed lines at retrains) and `ylim=`.                                                                                                                |

### Streaming demo notebook

`demo/stream_demo.ipynb` reads `Dataset/Housing.csv` with `numcompute.io.read_csv` (some cells are missing), cuts it into chunks of 25 houses and trains an imputer, scaler and tree one chunk at a time. It prints the per-chunk log, final accuracy, precision, recall, F1 and confusion matrix, and draws the accuracy plots, a two-buffer comparison and a predictions-against-truth plot.

1. Install the package (see [Installation](#installation)) and Jupyter: `python -m pip install notebook ipykernel`
2. Open it: `python -m notebook demo/stream_demo.ipynb`, then choose **Run > Run All Cells**. In VS Code, open the file, select your Python interpreter as the kernel and choose **Run All**.

### Streaming benchmarks

Both scripts run from the repository root and use fixed random seeds. Results (tables, a CSV and plots) are written to `benchmark/results/`.

```bash
python -m benchmark.stream_loop_vs_vectorised   # vectorised against plain-Python loops
python -m benchmark.buffer_size_benchmark       # how buffer_size changes accuracy, time and memory
```

**Environment:** Python 3.14.6, NumPy 2.5.1, Windows 11 (AMD64), 8 CPU cores.

**Vectorised against loops** (mean of 10 runs, in seconds; `compare` checks that both versions agree before reporting a speedup):

| Function                                              | Vectorized (s) | Loop (s) | Speedup |
| ----------------------------------------------------- | -------------- | -------- | ------- |
| tree split search, 1500 samples x 5 features          | 0.0204         | 0.3251   | 15.94x  |
| tree predict, 5000 samples, depth 6                   | 0.006154       | 0.08832  | 14.35x  |
| streaming confusion matrix, 20000 samples             | 0.005387       | 0.04915  | 9.12x   |
| running mean and variance, 20000 samples in 20 chunks | 0.03734        | 0.2847   | 7.62x   |

**Effect of `buffer_size`** (test-then-train, tree `max_depth=5`). Every buffer size sees the same chunks. Synthetic streams: 20,000 samples in chunks of 100, mean of 5 seeds. Housing: chunks of 25, mean of 10 shuffles. Accuracies show the mean ± standard deviation.

- `scored %`: share of samples that could be scored, because nothing can be scored before the first tree exists.
- `cumulative acc`: accuracy over all scored samples (different buffer sizes score different samples).
- `last 25% acc`: accuracy over the last quarter of the stream, which every buffer size scores, so it is the fair comparison.
- `after drift acc`: accuracy on the 2,000 samples right after the labelling rule changes.

Stream 1, the rule never changes:

| buffer_size | retrains | scored % | cumulative acc | last 25% acc  | ms per retrain | total s | peak memory (KB) |
| ----------- | -------- | -------- | -------------- | ------------- | -------------- | ------- | ---------------- |
| 100         | 200      | 99.5     | 0.848 ± 0.003  | 0.850 ± 0.003 | 16.26          | 3.252   | 2                |
| 200         | 100      | 99.0     | 0.865 ± 0.003  | 0.865 ± 0.004 | 25.49          | 2.549   | 7                |
| 500         | 40       | 97.5     | 0.881 ± 0.002  | 0.883 ± 0.007 | 44.18          | 1.767   | 21               |
| 1000        | 20       | 95.0     | 0.894 ± 0.002  | 0.889 ± 0.004 | 64.02          | 1.280   | 45               |
| 2000        | 10       | 90.0     | 0.901 ± 0.002  | 0.900 ± 0.004 | 97.49          | 0.975   | 92               |
| 5000        | 4        | 75.0     | 0.906 ± 0.001  | 0.903 ± 0.003 | 198.22         | 0.793   | 233              |

Stream 2, the rule changes halfway:

| buffer_size | retrains | scored % | cumulative acc | last 25% acc  | after drift acc | ms per retrain | total s | peak memory (KB) |
| ----------- | -------- | -------- | -------------- | ------------- | --------------- | -------------- | ------- | ---------------- |
| 100         | 200      | 99.5     | 0.844 ± 0.003  | 0.842 ± 0.004 | 0.827 ± 0.006   | 15.77          | 3.153   | 2                |
| 200         | 100      | 99.0     | 0.860 ± 0.002  | 0.865 ± 0.005 | 0.828 ± 0.013   | 25.81          | 2.581   | 7                |
| 500         | 40       | 97.5     | 0.869 ± 0.001  | 0.878 ± 0.002 | 0.787 ± 0.009   | 42.51          | 1.700   | 21               |
| 1000        | 20       | 95.0     | 0.873 ± 0.002  | 0.889 ± 0.008 | 0.694 ± 0.005   | 63.08          | 1.262   | 45               |
| 2000        | 10       | 90.0     | 0.855 ± 0.002  | 0.898 ± 0.006 | 0.497 ± 0.015   | 95.37          | 0.954   | 92               |
| 5000        | 4        | 75.0     | 0.770 ± 0.003  | 0.905 ± 0.006 | 0.496 ± 0.009   | 188.54         | 0.754   | 233              |

Stream 3, `Dataset/Housing.csv` (545 houses), shuffled:

| buffer_size | retrains | scored % | cumulative acc | last 25% acc  | ms per retrain | total s | peak memory (KB) |
| ----------- | -------- | -------- | -------------- | ------------- | -------------- | ------- | ---------------- |
| 25          | 21       | 95.4     | 0.697 ± 0.015  | 0.680 ± 0.041 | 8.80           | 0.185   | 3                |
| 50          | 10       | 90.8     | 0.711 ± 0.019  | 0.737 ± 0.024 | 15.63          | 0.156   | 6                |
| 100         | 5        | 81.7     | 0.730 ± 0.015  | 0.737 ± 0.030 | 24.24          | 0.121   | 10               |
| 200         | 2        | 63.3     | 0.739 ± 0.027  | 0.739 ± 0.038 | 37.36          | 0.075   | 22               |
| 400         | 1        | 26.6     | 0.767 ± 0.031  | 0.767 ± 0.031 | 56.29          | 0.056   | 44               |

What the results show:

- **A bigger buffer is more accurate on a stable stream.** Last-25% accuracy rises from 0.850 to 0.903. One retrain takes longer (16 ms to 198 ms), but there are far fewer retrains, so the total retraining time falls from 3.3 s to 0.8 s.
- **It costs memory and a late start.** Peak memory grows from 2 KB to 233 KB, and the model cannot score anything until its first tree exists (scored share 99.5% down to 75%).
- **A bigger buffer recovers more slowly from concept drift.** A new tree for the new rule only appears after `buffer_size` new samples. Just after the drift, accuracy is 0.827 for buffer 100 but about 0.5 (chance) for buffers 2000 and 5000. All buffer sizes recover by the end of the stream.
- **Housing shows the same pattern**, but with only 545 rows the differences between buffer sizes 50 and 200 are smaller than the spread between runs.

Plots: `benchmark/results/buffer_size.png` and `benchmark/results/drift_curves.png`. All the numbers are in `benchmark/results/buffer_size.md`, `buffer_size.csv` and `loop_vs_vectorised.md`.

### Streaming tests

There are 286 unit tests (`tests/test_stream_*.py`) and 21 docstring examples. The tests compare the vectorised code with hand-written loops and with the A1 batch versions, cover edge cases (empty chunks, NaN, ties, constant and all-NaN columns, single-class chunks, changing feature counts, shape and type errors) and turn every `RuntimeWarning` into an error.

```bash
python -m pytest tests/                                    # A1 and A2 tests together
python -m pytest tests/test_stream_tree.py tests/test_stream_stats.py tests/test_stream_preprocessing.py tests/test_stream_metrics.py tests/test_stream_pipeline.py tests/test_stream_trainer.py tests/test_stream_visualise.py
python -m pytest --doctest-modules numcompute_stream       # the examples in the docstrings
```

## Running the tests

```bash
python -m pytest tests/                    # run every test
python -m pytest tests/ -v                 # one line per test
python -m pytest tests/test_stats.py       # one module only

# with coverage (needs the [dev] install)
python -m pytest tests/ --cov=numcompute --cov-branch --cov-report=term-missing

# the project's quality gate: fail unless coverage is 100%
python -m pytest tests/ --cov=numcompute --cov-branch --cov-fail-under=100
```

Use `python -m pytest` rather than plain `pytest`: it makes sure the tests run in the same environment you installed the package into. Test files are named `test_<module>.py` and live in `tests/`.

## Project layout

```text
.
├── numcompute/            Assignment 1: the group package, one file per module
│   ├── __init__.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── utils.py
│   └── benchmarking.py
├── numcompute_stream/     Assignment 2: the streaming extension
│   ├── __init__.py
│   ├── tree.py            buffered DecisionTreeClassifier
│   ├── stream.py          StreamTrainer and iter_chunks
│   ├── preprocessing.py   StandardScaler, SimpleImputer, OneHotEncoder with partial_fit
│   ├── stats.py           RunningStats, StreamingQuantile, StreamingHistogram
│   ├── metrics.py         streaming accuracy, confusion matrix, precision, recall, F1, MSE, AUC
│   ├── pipeline.py        Pipeline and FeatureUnion with partial_fit
│   ├── visualise.py       matplotlib plots
│   └── _checks.py         shared input checks
├── tests/                 test_<module>.py (A1) and test_stream_<module>.py (A2)
├── benchmark/
│   ├── __init__.py
│   ├── run_benchmarks.py              A1: vectorised vs loop
│   ├── results.md                     A1: latest results and environment notes
│   ├── stream_loop_vs_vectorised.py   A2: vectorised vs loop
│   ├── buffer_size_benchmark.py       A2: buffer_size against accuracy, time and memory
│   └── results/                       A2: result tables, CSV and plots
├── demo/
│   ├── quickstart.ipynb   A1: end-to-end demo notebook
│   └── stream_demo.ipynb  A2: streaming demo notebook
├── Dataset/               Housing.csv, the sample dataset
├── README.md
├── requirements.txt
└── pyproject.toml         packaging and dependencies
```

## Team

| Team member                                             | Modules                                                                     |
| ------------------------------------------------------- | --------------------------------------------------------------------------- |
| Cyril ([@Cyriloo7](https://github.com/Cyriloo7))        | `stats.py`, `metrics.py`, `utils.py`, `benchmarking.py`                     |
| Melvin ([@Melvin14120](https://github.com/melvin14120)) | `sort_search.py`, `rank.py`, `optim.py`                                     |
| Binitta ([@binittavar](https://github.com/binittavar))  | `io.py`, `preprocessing.py`, `pipeline.py`                                  |
| Melvin (Assignment 2, individual work)                  | `numcompute_stream/` (all modules), its tests, benchmarks and demo notebook |

## Troubleshooting

- **`ModuleNotFoundError: No module named 'numcompute'`** - the package is not installed in the environment you are running. From the repository root, run `python -m pip install -e ".[dev]"`.
- **`ModuleNotFoundError: No module named 'benchmark'`** - run `python -m benchmark.run_benchmarks` from the repository root, the folder that contains `benchmark/`.
- **`pytest: error: unrecognized arguments: --cov`** - the test tools are missing. Run `python -m pip install -e ".[dev]"`.
- **pytest says "no tests ran"** - test files must be named `test_<name>.py` (underscore, not hyphen) and live in `tests/`.
- **`AttributeError: module 'numpy' has no attribute 'trapezoid'`** - your NumPy is older than 2.0. Run `python -m pip install -U "numpy>=2.0"`.
- **PowerShell will not run `Activate.ps1`** - run `Set-ExecutionPolicy -Scope Process RemoteSigned` first, or use Command Prompt and `.venv\Scripts\activate.bat`.
