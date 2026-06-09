import numpy as np
def bootstrap_ci(data, num_samples=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(data)
    data = np.array(data, dtype=float)
    samples = rng.choice(data, size=(num_samples, n), replace=True)
    means = np.mean(samples, axis=1)
    return [float(np.percentile(means, 100 * (alpha / 2))), float(np.percentile(means, 100 * (1 - alpha / 2)))]

diffs = [1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0] * 5
print(bootstrap_ci(diffs))
