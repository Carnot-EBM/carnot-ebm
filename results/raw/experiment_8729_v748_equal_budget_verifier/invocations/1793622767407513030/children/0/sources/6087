from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import scipy.stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def compute_p_values(X_cal: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Compute conformal p-values for each verifier score column."""

    n_test, n_verifiers = X_test.shape
    n_cal = X_cal.shape[0]
    p_values = np.zeros((n_test, n_verifiers))
    for i in range(n_verifiers):
        cal_scores = np.sort(X_cal[:, i])
        for j in range(n_test):
            count = np.sum(cal_scores >= X_test[j, i])
            p_values[j, i] = count / (n_cal + 1)
    return p_values


def fisher_combine(p_values: np.ndarray, clip_val: float = 1e-10) -> np.ndarray:
    """Convert verifier p-values into hallucination-risk scores via Fisher."""

    clipped = np.clip(p_values, clip_val, 1.0)
    chi2_stat = -2 * np.sum(np.log(clipped), axis=1)
    p_combined = scipy.stats.chi2.sf(chi2_stat, df=2 * clipped.shape[1])
    return 1.0 - p_combined


def calibrate_group_scores(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """Fit isotonic calibration for one verifier group and score test rows."""

    train_fisher = fisher_combine(compute_p_values(X_train, X_train))
    test_fisher = fisher_combine(compute_p_values(X_train, X_test))

    isotonic_reg = IsotonicRegression(out_of_bounds="clip")
    isotonic_reg.fit(train_fisher, y_train)
    return isotonic_reg.predict(test_fisher)


def run_group_conditional_calibration(
    score_groups: Mapping[str, np.ndarray],
    labels: np.ndarray,
    seeds: Sequence[int],
    group_order: Sequence[str],
    test_size: float = 0.3,
) -> tuple[list[dict[str, float | int]], float, float]:
    """Run group-conditional calibration for each seed and return AUROC stats."""

    seed_results: list[dict[str, float | int]] = []
    test_aurocs: list[float] = []

    for seed in seeds:
        idx = np.arange(len(labels))
        idx_train, idx_test, y_train, y_test = train_test_split(
            idx,
            labels,
            test_size=test_size,
            random_state=seed,
            stratify=labels,
        )

        test_calibrated_by_group: dict[str, np.ndarray] = {}
        for group_name in group_order:
            X_group = score_groups[group_name]
            test_calibrated_by_group[group_name] = calibrate_group_scores(
                X_group[idx_train],
                X_group[idx_test],
                y_train,
            )

        P_matrix = np.column_stack(
            [1.0 - test_calibrated_by_group[group_name] for group_name in group_order]
        )
        test_combined = fisher_combine(P_matrix)
        test_auroc = float(roc_auc_score(y_test, test_combined))
        test_aurocs.append(test_auroc)

        row: dict[str, float | int] = {
            "seed": int(seed),
            "test_auroc_group_cond": test_auroc,
        }
        for group_name in group_order:
            row[f"mean_cal_{group_name}"] = float(test_calibrated_by_group[group_name].mean())
        seed_results.append(row)

    return seed_results, float(np.mean(test_aurocs)), float(np.std(test_aurocs))
