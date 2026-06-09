import numpy as np

def compute_metrics(scores):
    cov_matrix = np.cov(scores, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    lambda_min_sigma = float(np.min(eigenvalues))
    sum_eig = np.sum(eigenvalues)
    sum_eig_sq = np.sum(eigenvalues**2)
    effective_k = float((sum_eig**2) / (sum_eig_sq + 1e-9))
    return lambda_min_sigma, effective_k

n_cases = 1000
np.random.seed(42)
old_scores = np.random.binomial(1, 0.7, size=(n_cases, 6)).astype(float)
old_scores[:, 1] = np.where(np.random.random(n_cases) < 0.9, old_scores[:, 0], old_scores[:, 1])

lambda_before, k_before = compute_metrics(old_scores)

cached_candidates = []
for i in range(n_cases):
    cand = {}
    if np.random.rand() > 0.6:
        cand["trajectory_steps"] = ["step_a", "step_b"]
    else:
        cand["trajectory_steps"] = []
    cached_candidates.append(cand)

axis_scores = []
for cand in cached_candidates:
    if "trajectory_steps" in cand and cand["trajectory_steps"]:
        axis_scores.append(1.0)
    else:
        axis_scores.append(0.0)
axis_scores = np.array(axis_scores, dtype=float)

new_scores = np.column_stack((old_scores, axis_scores))
lambda_after, k_after = compute_metrics(new_scores)

print(f"Before: lambda={lambda_before}, k={k_before}")
print(f"After: lambda={lambda_after}, k={k_after}")
