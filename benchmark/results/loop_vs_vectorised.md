Python 3.14.6, NumPy 2.5.1
Windows 11 (AMD64), CPU cores: 8
Each time is the mean of 10 runs, in seconds. Seeded data (seed 0).

| Function | Vectorized (s) | Loop (s) | Speedup |
|---|---|---|---|
| tree split search, 1500 samples x 5 features | 0.0204 | 0.3251 | 15.94x |
| tree predict, 5000 samples, depth 6 | 0.006154 | 0.08832 | 14.35x |
| streaming confusion matrix, 20000 samples | 0.005387 | 0.04915 | 9.12x |
| running mean and variance, 20000 samples in 20 chunks | 0.03734 | 0.2847 | 7.62x |
