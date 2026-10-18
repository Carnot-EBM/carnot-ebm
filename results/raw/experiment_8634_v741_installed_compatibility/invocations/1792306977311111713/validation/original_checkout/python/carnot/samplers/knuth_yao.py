"""Knuth-Yao categorical sampling simulator for AIA-style hardware studies.

**Researcher summary:**
    A Knuth-Yao sampler draws from a finite categorical distribution by walking
    a discrete-distribution-generating bit matrix with unbiased random bits.
    That makes RNG consumption visible: every emitted category has an auditable
    bit count, which is the software boundary needed before comparing AIA-style
    sampler hardware resource costs.

Spec: REQ-SAMPLE-2043, SCENARIO-SAMPLE-2043
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SPEC_REFS = ["REQ-SAMPLE-2043", "SCENARIO-SAMPLE-2043"]
DEFAULT_PARITY_THRESHOLDS = {
    "max_abs_frequency_delta": 0.03,
    "total_variation_delta": 0.04,
}


def _normalize_probabilities(probabilities: object) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional distribution")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must contain only finite values")
    if np.any(probs < 0.0):
        raise ValueError("probabilities must be non-negative")
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("probabilities must contain positive mass")
    return probs / total


def _dyadic_counts(probabilities: np.ndarray, precision_bits: int) -> np.ndarray:
    if precision_bits < 1 or precision_bits > 52:
        raise ValueError("precision_bits must be in the inclusive range [1, 52]")

    denominator = 1 << int(precision_bits)
    scaled = probabilities * denominator
    counts = np.floor(scaled).astype(np.int64)
    remaining = int(denominator - int(counts.sum()))
    residual_order = np.argsort(-(scaled - counts), kind="mergesort")
    counts[residual_order[:remaining]] += 1
    return counts


def _build_ddg_matrix(counts: np.ndarray, precision_bits: int) -> np.ndarray:
    shifts = np.arange(precision_bits - 1, -1, -1, dtype=np.int64)
    return ((counts[:, None] >> shifts[None, :]) & 1).astype(np.int8)


def _frequency_vector(samples: np.ndarray, n_categories: int) -> np.ndarray:
    counts = np.bincount(samples, minlength=n_categories)
    return counts.astype(np.float64) / float(samples.size)


def _chi_square_statistic(counts: np.ndarray, expected_probabilities: np.ndarray) -> float:
    expected = expected_probabilities * float(counts.sum())
    return float(np.sum((counts - expected) ** 2 / expected))


@dataclass
class KnuthYaoSampler:
    """Sample a categorical distribution with Knuth-Yao DDG bit traversal.

    Parameters
    ----------
    probabilities:
        Finite non-negative category weights. They are normalized internally,
        then quantized onto a denominator of ``2**precision_bits``.
    symbols:
        Optional labels to return from :meth:`sample`. When omitted, category
        indices are returned.
    precision_bits:
        Number of fractional bits used in the DDG matrix. Dyadic inputs whose
        denominator divides ``2**precision_bits`` are represented exactly.
    seed:
        Seed for the unbiased bit generator.

    Spec: REQ-SAMPLE-2043
    """

    probabilities: Sequence[float]
    symbols: Sequence[Any] | None = None
    precision_bits: int = 16
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)
    probabilities_normalized: np.ndarray = field(init=False)
    dyadic_counts: np.ndarray = field(init=False)
    quantized_probabilities: np.ndarray = field(init=False)
    ddg_matrix: np.ndarray = field(init=False)
    bits_consumed: int = field(default=0, init=False)
    samples_drawn: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.probabilities_normalized = _normalize_probabilities(self.probabilities)
        if self.symbols is not None and len(self.symbols) != self.probabilities_normalized.size:
            raise ValueError("symbols must have the same length as probabilities")
        self.dyadic_counts = _dyadic_counts(self.probabilities_normalized, self.precision_bits)
        denominator = float(1 << int(self.precision_bits))
        self.quantized_probabilities = self.dyadic_counts.astype(np.float64) / denominator
        self.ddg_matrix = _build_ddg_matrix(self.dyadic_counts, self.precision_bits)
        self.rng = np.random.default_rng(int(self.seed))

    @property
    def fixed_width_bits_per_sample(self) -> int:
        """Return the baseline bit width for a fixed-precision categorical draw."""
        return int(self.precision_bits)

    def reset_metrics(self) -> None:
        """Clear accumulated RNG-bit and sample counters.

        Spec: REQ-SAMPLE-2043-2
        """
        self.bits_consumed = 0
        self.samples_drawn = 0

    def _draw_bit(self) -> int:
        self.bits_consumed += 1
        return int(self.rng.integers(0, 2))

    def _draw_index(self) -> int:
        depth_index = 0
        for column in range(int(self.precision_bits)):
            depth_index = (2 * depth_index) + self._draw_bit()
            for category, bit in enumerate(self.ddg_matrix[:, column]):
                depth_index -= int(bit)
                if depth_index < 0:
                    self.samples_drawn += 1
                    return int(category)
        raise RuntimeError("DDG traversal exhausted without selecting a category")  # pragma: no cover

    def sample_indices(self, n_samples: int) -> np.ndarray:
        """Return integer category samples from the quantized distribution.

        Spec: REQ-SAMPLE-2043-1
        """
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        return np.fromiter((self._draw_index() for _ in range(int(n_samples))), dtype=np.int64)

    def sample(self, n_samples: int) -> np.ndarray:
        """Return symbol samples when labels were supplied, else category indices.

        Spec: REQ-SAMPLE-2043-1
        """
        indices = self.sample_indices(n_samples)
        if self.symbols is None:
            return indices
        return np.asarray(self.symbols, dtype=object)[indices]

    def bit_metrics(self) -> dict[str, float | int]:
        """Return RNG-bit accounting for samples emitted by this instance.

        Spec: REQ-SAMPLE-2043-2
        """
        average_bits = self.bits_consumed / self.samples_drawn if self.samples_drawn else 0.0
        fixed_width = self.fixed_width_bits_per_sample
        reduction = 1.0 - (average_bits / fixed_width) if fixed_width else 0.0
        return {
            "bits_consumed": int(self.bits_consumed),
            "samples_drawn": int(self.samples_drawn),
            "average_bits_per_sample": float(average_bits),
            "fixed_width_bits_per_sample": int(fixed_width),
            "rng_bit_reduction_vs_fixed_width": float(reduction),
        }


def run_statistical_parity_test(
    *,
    probabilities: Sequence[float],
    n_samples: int,
    precision_bits: int,
    knuth_yao_seed: int,
    standard_rng_seed: int,
    thresholds: Mapping[str, float] = DEFAULT_PARITY_THRESHOLDS,
) -> dict[str, Any]:
    """Compare Knuth-Yao samples with NumPy's standard categorical RNG.

    Spec: REQ-SAMPLE-2043-3, SCENARIO-SAMPLE-2043
    """
    normalized = _normalize_probabilities(probabilities)
    sampler = KnuthYaoSampler(normalized, precision_bits=precision_bits, seed=knuth_yao_seed)
    knuth_yao_samples = sampler.sample_indices(n_samples)

    standard_rng = np.random.default_rng(int(standard_rng_seed))
    standard_samples = standard_rng.choice(normalized.size, size=int(n_samples), p=normalized)

    knuth_yao_counts = np.bincount(knuth_yao_samples, minlength=normalized.size)
    standard_counts = np.bincount(standard_samples, minlength=normalized.size)
    knuth_yao_frequencies = _frequency_vector(knuth_yao_samples, normalized.size)
    standard_frequencies = _frequency_vector(standard_samples, normalized.size)
    frequency_delta = np.abs(knuth_yao_frequencies - standard_frequencies)
    max_abs_delta = float(np.max(frequency_delta))
    total_variation_delta = float(0.5 * np.sum(frequency_delta))
    parity_passed = (
        max_abs_delta <= float(thresholds["max_abs_frequency_delta"])
        and total_variation_delta <= float(thresholds["total_variation_delta"])
    )

    return {
        "spec_refs": SPEC_REFS,
        "n_samples": int(n_samples),
        "precision_bits": int(precision_bits),
        "input_probabilities": normalized.tolist(),
        "quantized_probabilities": sampler.quantized_probabilities.tolist(),
        "dyadic_counts": sampler.dyadic_counts.astype(int).tolist(),
        "knuth_yao_seed": int(knuth_yao_seed),
        "standard_rng_seed": int(standard_rng_seed),
        "knuth_yao_counts": knuth_yao_counts.astype(int).tolist(),
        "standard_rng_counts": standard_counts.astype(int).tolist(),
        "knuth_yao_frequencies": knuth_yao_frequencies.tolist(),
        "standard_rng_frequencies": standard_frequencies.tolist(),
        "max_abs_frequency_delta": max_abs_delta,
        "total_variation_delta": total_variation_delta,
        "knuth_yao_chi_square_vs_expected": _chi_square_statistic(
            knuth_yao_counts,
            sampler.quantized_probabilities,
        ),
        "standard_rng_chi_square_vs_expected": _chi_square_statistic(
            standard_counts,
            normalized,
        ),
        "thresholds": dict(thresholds),
        "parity_passed": bool(parity_passed),
        "bit_metrics": sampler.bit_metrics(),
    }
