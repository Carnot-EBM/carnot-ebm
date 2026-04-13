"""Isotonic calibration for predictive verification confidence scoring.

**Researcher summary:**
    EBMs produce uncalibrated confidence scores. We apply isotonic regression
    to map raw logistic scores to true violation probabilities.  The calibrator
    learns a monotone mapping during offline training (Exp 263) and produces a
    per-token activation EBM operating threshold that trades off detection rate
    vs. false-positive rate.

**Detailed explanation for engineers:**
    Isotonic calibration takes a raw score (e.g., from logistic regression) and
    maps it to a calibrated probability. We store the mapping as two parallel
    arrays: x_thresholds (bin boundaries) and y_thresholds (output probabilities).

    The calibrator is fit once on a corpus of violations vs. clean cases, and
    the fitted mapping persists in a JSON file.  At inference time, we apply
    the mapping to route cases: if calibrated_prob < threshold → FAST_PATH,
    else → FULL verification.

Spec: REQ-PRED-263-001, REQ-PRED-263-002, REQ-PRED-263-003
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Operating zone constants
# ---------------------------------------------------------------------------

ZONE_BELOW_MARGINAL: str = "BELOW_MARGINAL"
"""Calibration failed to achieve minimum performance targets."""

ZONE_MARGINAL: str = "MARGINAL"
"""Fast-path rate ≥30% (4/δ bound, δ=0.1)."""

ZONE_PRACTICAL: str = "PRACTICAL"
"""Fast-path rate ≥50% (high-speed coverage)."""

ZONE_HIGH_PERFORMANCE: str = "HIGH_PERFORMANCE"
"""Fast-path rate ≥70% (excellent speed-coverage tradeoff)."""


# ---------------------------------------------------------------------------
# IsotonicCalibration dataclass
# ---------------------------------------------------------------------------


@dataclass
class IsotonicCalibration:
    """Isotonic regression output: monotone mapping from raw scores to probabilities.

    Attributes:
        threshold: Operating threshold (0.0 to 1.0). Cases with calibrated
                   probability < threshold route FAST_PATH.
        x_thresholds: Bin boundary scores from isotonic regression (numpy array).
        y_thresholds: Mapped probabilities corresponding to x_thresholds.
        experiment: Experiment ID that produced this calibration (e.g., 263).
        run_date: Date string (YYYYMMDD) for audit trail.
    """

    threshold: float = 0.5
    x_thresholds: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    y_thresholds: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    experiment: int = 263
    run_date: str = "20260413"


# ---------------------------------------------------------------------------
# Fitting: isotonic regression
# ---------------------------------------------------------------------------


def fit_calibration(
    corpus: list[dict[str, Any]],
    *,
    seed: int | None = None,
) -> IsotonicCalibration:
    """Fit isotonic calibration from a corpus of violation/clean cases.

    REQ-PRED-263-001: fit_calibration returns an IsotonicCalibration object
    with a threshold in [0, 1].

    Args:
        corpus: List of corpus rows from Exp 252/262. Each row must have:
                - token_feature_vector: 9-element list of features
                - violation_label: boolean (True = violation, False = clean)
        seed: Optional RNG seed for reproducibility.

    Returns:
        IsotonicCalibration with fitted threshold and isotonic mapping.
    """
    if seed is not None:
        np.random.seed(seed)

    if not corpus:
        # Empty corpus: return default calibration
        return IsotonicCalibration()

    # Extract features and labels
    x_list = []
    y_list = []
    for row in corpus:
        x_list.append(np.array(row["token_feature_vector"], dtype=np.float32))
        y_list.append(float(row["violation_label"]))

    x_array = np.array(x_list, dtype=np.float32)  # shape: (n_samples, 9)
    y_array = np.array(y_list, dtype=np.float32)  # shape: (n_samples,)

    # Compute raw logistic scores using hardcoded default weights
    # (in a real scenario, these would come from PredictiveVerifier training).
    default_w = np.array(
        [0.1, 0.05, 0.3, 0.2, 0.15, 0.1, 0.2, 0.0, 0.1],
        dtype=np.float32,
    )
    raw_scores = np.dot(x_array, default_w) + 0.1  # shape: (n_samples,)

    # Apply sigmoid to convert to [0,1]
    probs = 1.0 / (1.0 + np.exp(-raw_scores))

    # Simple isotonic regression: partition into bins and fit monotone curve
    n_bins = min(10, max(3, len(corpus) // 10))
    sorted_idx = np.argsort(raw_scores)
    bin_size = max(1, len(corpus) // n_bins)

    x_thresholds = []
    y_thresholds = []

    for i in range(n_bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, len(corpus))
        if start >= len(corpus):
            break

        bin_idx = sorted_idx[start:end]
        bin_raw = raw_scores[bin_idx]
        bin_y = y_array[bin_idx]

        x_thresholds.append(float(np.mean(bin_raw)))
        y_thresholds.append(float(np.mean(bin_y)))

    # Ensure monotonicity
    for j in range(1, len(y_thresholds)):
        y_thresholds[j] = max(y_thresholds[j], y_thresholds[j - 1])

    # Find operating threshold: maximize Youden's J = sensitivity + specificity - 1
    # J = TPR - FPR = P(score > t | violation) - P(score > t | clean)
    best_j = -1.0
    best_threshold = 0.5

    for threshold_candidate in np.linspace(0.0, 1.0, 20):
        tp = np.sum((probs >= threshold_candidate) & (y_array == 1))
        fp = np.sum((probs >= threshold_candidate) & (y_array == 0))
        tn = np.sum((probs < threshold_candidate) & (y_array == 0))
        fn = np.sum((probs < threshold_candidate) & (y_array == 1))

        n_pos = tp + fn
        n_neg = tn + fp

        if n_pos > 0 and n_neg > 0:
            tpr = tp / n_pos
            fpr = fp / n_neg
            j = tpr - fpr
            if j > best_j:
                best_j = j
                best_threshold = threshold_candidate

    return IsotonicCalibration(
        threshold=best_threshold,
        x_thresholds=np.array(x_thresholds, dtype=np.float32),
        y_thresholds=np.array(y_thresholds, dtype=np.float32),
        experiment=263,
        run_date="20260413",
    )


# ---------------------------------------------------------------------------
# Inference: apply calibration
# ---------------------------------------------------------------------------


def apply_calibration(
    calibration: IsotonicCalibration,
    verifier: Any,  # PredictiveVerifier instance
    x: np.ndarray,
) -> float:
    """Apply fitted isotonic calibration to a feature vector.

    REQ-PRED-263-001: apply_calibration maps [0,∞) raw score to [0,1] prob.

    Args:
        calibration: IsotonicCalibration from fit_calibration or load_calibration.
        verifier: PredictiveVerifier instance with _w and _b attributes.
        x: Feature vector (shape (9,)) as returned by extract_features.

    Returns:
        Calibrated probability in [0, 1].
    """
    # Compute raw logistic score
    raw = float(np.dot(verifier._w, x) + verifier._b)

    # Apply isotonic mapping via piecewise-linear interpolation
    x_thresh = calibration.x_thresholds
    y_thresh = calibration.y_thresholds

    if raw <= x_thresh[0]:
        return float(y_thresh[0])
    if raw >= x_thresh[-1]:
        return float(y_thresh[-1])

    # Find bracketing bin
    for i in range(len(x_thresh) - 1):
        if x_thresh[i] <= raw <= x_thresh[i + 1]:
            # Linear interpolation
            frac = (raw - x_thresh[i]) / (x_thresh[i + 1] - x_thresh[i])
            return float(y_thresh[i] + frac * (y_thresh[i + 1] - y_thresh[i]))

    # Shouldn't reach here
    return float(y_thresh[-1])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_calibration(calibration: IsotonicCalibration, path: Path) -> None:
    """Persist calibration to JSON.

    REQ-PRED-263-002: save_calibration writes JSON with required keys.

    Args:
        calibration: IsotonicCalibration to save.
        path: Output JSON path.
    """
    data = {
        "threshold": float(calibration.threshold),
        "isotonic_x_thresholds": calibration.x_thresholds.tolist(),
        "isotonic_y_thresholds": calibration.y_thresholds.tolist(),
        "experiment": int(calibration.experiment),
        "run_date": str(calibration.run_date),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_calibration(path: Path) -> IsotonicCalibration:
    """Load calibration from JSON.

    REQ-PRED-263-002: load_calibration reads JSON and reconstructs CalibrationObject.

    Args:
        path: Input JSON path.

    Returns:
        IsotonicCalibration loaded from file.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with open(path) as f:
        data = json.load(f)

    return IsotonicCalibration(
        threshold=float(data["threshold"]),
        x_thresholds=np.array(data["isotonic_x_thresholds"], dtype=np.float32),
        y_thresholds=np.array(data["isotonic_y_thresholds"], dtype=np.float32),
        experiment=int(data.get("experiment", 263)),
        run_date=str(data.get("run_date", "20260413")),
    )


# ---------------------------------------------------------------------------
# Operating zone classification
# ---------------------------------------------------------------------------


def classify_operating_zone(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> str:
    """Classify the operating zone based on performance under a threshold.

    REQ-PRED-263-003: classify_operating_zone returns one of four zones.

    Args:
        probs: Calibrated probabilities (shape (n,)).
        labels: Ground truth violation labels (shape (n,)).
        threshold: Operating threshold for FAST_PATH routing.

    Returns:
        One of: ZONE_BELOW_MARGINAL, ZONE_MARGINAL, ZONE_PRACTICAL,
                ZONE_HIGH_PERFORMANCE.
    """
    # Fast-path rate = fraction of cases routed FAST_PATH (prob < threshold)
    fast_path_rate = np.sum(probs < threshold) / len(probs) if len(probs) > 0 else 0.0

    # Zone boundaries based on speed-coverage tradeoff
    if fast_path_rate >= 0.70:
        return ZONE_HIGH_PERFORMANCE
    elif fast_path_rate >= 0.50:
        return ZONE_PRACTICAL
    elif fast_path_rate >= 0.30:
        return ZONE_MARGINAL
    else:
        return ZONE_BELOW_MARGINAL


# ---------------------------------------------------------------------------
# Operating threshold selection
# ---------------------------------------------------------------------------


def find_operating_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    min_detection_rate: float = 0.6,
    max_fp_rate: float = 0.2,
) -> float:
    """Find operating threshold that meets detection and false-positive targets.

    REQ-PRED-263-003: find_operating_threshold returns a threshold in [0, 1].

    Args:
        probs: Calibrated probabilities (shape (n,)).
        labels: Ground truth violation labels (shape (n,)).
        min_detection_rate: Minimum true positive rate (default 0.6).
        max_fp_rate: Maximum false positive rate (default 0.2).

    Returns:
        Operating threshold (float in [0, 1]).
    """
    best_threshold = 0.5
    best_score = -1.0

    for threshold_candidate in np.linspace(0.0, 1.0, 20):
        # True positives: high-confidence violations (prob >= threshold) that are violations
        tp = np.sum((probs >= threshold_candidate) & (labels == 1))
        # False positives: high-confidence but not violations
        fp = np.sum((probs >= threshold_candidate) & (labels == 0))
        # True negatives and false negatives for rate computation
        tn = np.sum((probs < threshold_candidate) & (labels == 0))
        fn = np.sum((probs < threshold_candidate) & (labels == 1))

        n_pos = tp + fn
        n_neg = tn + fp

        if n_pos == 0 or n_neg == 0:
            continue

        detection_rate = tp / n_pos
        fp_rate = fp / n_neg

        # Prefer thresholds that meet both constraints
        if detection_rate >= min_detection_rate and fp_rate <= max_fp_rate:
            # Score: maximize detection rate, then minimize FP rate
            score = detection_rate - 0.1 * fp_rate
            if score > best_score:
                best_score = score
                best_threshold = threshold_candidate

    return best_threshold
