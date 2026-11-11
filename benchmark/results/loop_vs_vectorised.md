Python 3.14.6, NumPy 2.5.1
Windows 11 (AMD64), CPU cores: 8
Each time is the mean of 10 runs, in seconds. Seeded data (seed 0).

| Function | Vectorized (s) | Loop (s) | Speedup |
|---|---|---|---|
| tree split search, 1500 samples x 5 features | 0.02151 | 0.3757 | 17.47x |
| tree predict, 5000 samples, depth 6 | 0.005642 | 0.06264 | 11.10x |
| streaming confusion matrix, 20000 samples | 0.00369 | 0.01791 | 4.86x |
| running mean and variance, 20000 samples in 20 chunks | 0.03457 | 0.2131 | 6.16x |
