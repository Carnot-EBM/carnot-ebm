"""EBM Calibration Alignment — EBMCalibrator.

**Researcher summary:**
    Implements Expected Calibration Error (ECE) computation and isotonic
    regression post-hoc calibration for Carnot EBM energy scores.

    Motivation:
    - arXiv 2603.06604 "Know When You're Wrong": SFT calibration gap 15-25pp.
      Energy should predict P(correct), not just flag violations.
    - arXiv 2602.11364 "Energy of Falsehood": diffusion reconstruction energy
      detects hallucinations (AUROC 0.725).  Carnot should match this.

**Detailed explanation for engineers:**
    The core idea: if you sort model outputs by energy (low to high) and split
    them into buckets, the fraction of correct answers in each bucket should
    match the "confidence" implied by that bucket's energy.  If low-energy
    outputs are 90% correct but the model assigns them 60% confidence, that's
    a calibration gap.

    ECE measures this gap across all buckets:
        ECE = sum_b (|accuracy_b - confidence_b| * n_b) / N

    Isotonic regression learns a monotone mapping from raw energy to P(correct)
    so that confidence_b ≈ accuracy_b after calibration.

    We use sigmoid(-energy) as the raw confidence because lower energy should
    mean higher probability of correctness.  Isotonic regression then adjusts
    that mapping to match observed accuracy.

Spec: REQ-CALIB-001, REQ-CALIB-002
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import List

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibrationBin:
    """One energy bin in the calibration curve.

    **What each field means:**
    - energy_low, energy_high: the range of raw energy values in this bin.
    - accuracy: fraction of samples in this bin that had label=1 (correct).
    - n_samples: number of samples in this bin.
    - bin_confidence: mean(sigmoid(-energy)) for samples in this bin.
      This is the raw, uncalibrated confidence implied by the energy.
    """

    energy_low: float
    energy_high: float
    accuracy: float
    n_samples: int
    bin_confidence: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid: sigma(x) = 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class EBMCalibrator:
    """Calibrates raw EBM energy scores to P(correct) via ECE + isotonic regression.

    **Why ECE with equal-frequency bins:**
        Equal-frequency binning ensures each bin has the same number of samples,
        so accuracy estimates are equally reliable across all energy levels.
        Fixed-width bins would have sparse coverage at the extremes.

    **Why isotonic regression:**
        Isotonic regression is the standard non-parametric post-hoc calibration
        method (Zadrozny & Elkan 2002).  It fits a monotone non-decreasing
        function from confidence to accuracy — appropriate here because lower
        energy SHOULD map to higher P(correct).  We fit on (-energy) -> label
        so the direction matches: high (-energy) = low energy = high P(correct).

    Args:
        n_bins: Number of equal-frequency bins for ECE computation. Default 10.
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    def _build_bins(
        self, energies: List[float], labels: List[int]
    ) -> List[CalibrationBin]:
        """Split energies into equal-frequency bins and compute per-bin stats.

        Equal-frequency = each bin has floor(N/n_bins) samples.  The last bin
        absorbs any remainder so we always have exactly n_bins bins.
        """
        n = len(energies)
        arr_e = np.array(energies, dtype=np.float64)
        arr_l = np.array(labels, dtype=np.float64)

        # Sort by energy (low to high)
        order = np.argsort(arr_e)
        arr_e = arr_e[order]
        arr_l = arr_l[order]

        bin_size = n // self.n_bins
        bins: List[CalibrationBin] = []
        for i in range(self.n_bins):
            start = i * bin_size
            # Last bin absorbs remainder
            end = n if i == self.n_bins - 1 else (i + 1) * bin_size
            e_slice = arr_e[start:end]
            l_slice = arr_l[start:end]
            conf_slice = _sigmoid(-e_slice)  # low energy -> high sigmoid(-e)
            bins.append(
                CalibrationBin(
                    energy_low=float(e_slice[0]),
                    energy_high=float(e_slice[-1]),
                    accuracy=float(l_slice.mean()),
                    n_samples=int(end - start),
                    bin_confidence=float(conf_slice.mean()),
                )
            )
        return bins

    def compute_ece(self, energies: List[float], labels: List[int]) -> float:
        """Compute Expected Calibration Error over equal-frequency energy bins.

        ECE = sum_b ( |accuracy_b - confidence_b| * n_b ) / N

        A perfectly calibrated model has ECE=0.0.  An ECE of 0.10 means the
        average gap between predicted confidence and observed accuracy is 10pp.

        Args:
            energies: Raw energy values from the EBM (one per sample).
            labels: Binary correctness labels (1=correct, 0=incorrect).

        Returns:
            ECE as a float in [0, 1].  Lower is better.
        """
        if len(energies) != len(labels):
            raise ValueError(
                f"energies and labels must have the same length, "
                f"got {len(energies)} vs {len(labels)}"
            )
        n = len(energies)
        if n == 0:
            return 0.0

        bins = self._build_bins(energies, labels)
        ece = sum(
            abs(b.accuracy - b.bin_confidence) * b.n_samples for b in bins
        ) / n
        return float(ece)

    def fit_isotonic(
        self, energies: List[float], labels: List[int]
    ) -> IsotonicRegression:
        """Fit isotonic regression on (-energy, label) pairs.

        We fit on NEGATIVE energy because lower energy should correspond to
        higher P(correct).  Isotonic regression requires a non-decreasing
        mapping, so mapping (-energy) -> P(correct) satisfies that constraint.

        Args:
            energies: Raw energy values (one per sample).
            labels: Binary correctness labels (1=correct, 0=incorrect).

        Returns:
            Fitted sklearn IsotonicRegression.  Call .predict(-energy_array)
            to get calibrated P(correct) for new samples.
        """
        arr_e = np.array(energies, dtype=np.float64)
        arr_l = np.array(labels, dtype=np.float64)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(-arr_e, arr_l)
        return iso

    def compute_ece_from_probs(
        self, probs: List[float], labels: List[int]
    ) -> float:
        """Compute ECE from calibrated probabilities (already in [0,1]).

        Used after isotonic regression: convert calibrated probs to ECE by
        treating prob as confidence and comparing to observed accuracy per bin.

        Args:
            probs: Calibrated P(correct) values in [0, 1].
            labels: Binary correctness labels.

        Returns:
            ECE as a float in [0, 1].
        """
        if len(probs) != len(labels):
            raise ValueError(
                f"probs and labels must have same length, "
                f"got {len(probs)} vs {len(labels)}"
            )
        n = len(probs)
        if n == 0:
            return 0.0

        arr_p = np.array(probs, dtype=np.float64)
        arr_l = np.array(labels, dtype=np.float64)

        # Sort by calibrated probability (low to high) for equal-freq bins
        order = np.argsort(arr_p)
        arr_p = arr_p[order]
        arr_l = arr_l[order]

        bin_size = n // self.n_bins
        ece = 0.0
        for i in range(self.n_bins):
            start = i * bin_size
            end = n if i == self.n_bins - 1 else (i + 1) * bin_size
            p_slice = arr_p[start:end]
            l_slice = arr_l[start:end]
            ece += abs(float(l_slice.mean()) - float(p_slice.mean())) * (end - start)
        return float(ece / n)

    def save_curve(self, bins: List[CalibrationBin], path: str) -> None:
        """Save the calibration curve to a JSON file for external plotting.

        The saved JSON contains a list of bin dicts with fields:
        energy_low, energy_high, accuracy, n_samples, bin_confidence.
        This data can be plotted as a reliability diagram.

        Args:
            bins: List of CalibrationBin objects from _build_bins.
            path: Filesystem path to write the JSON file.
        """
        with open(path, "w") as f:
            json.dump([asdict(b) for b in bins], f, indent=2)

    def build_curve(
        self, energies: List[float], labels: List[int]
    ) -> List[CalibrationBin]:
        """Public wrapper: build bins and return them for saving or analysis."""
        return self._build_bins(energies, labels)
