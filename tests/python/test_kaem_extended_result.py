"""Tests for KAEMExtendedResult — 100% coverage for the kaem_extended_result module.

Spec: REQ-SAMPLE-020, REQ-SAMPLE-021,
      SCENARIO-SAMPLE-033, SCENARIO-SAMPLE-034
"""

from __future__ import annotations

import pytest

from carnot.models.kaem_extended_result import KAEMExtendedResult


# ---------------------------------------------------------------------------
# SCENARIO-SAMPLE-033: crossover found at n=2000
# ---------------------------------------------------------------------------


def test_crossover_found_at_n2000() -> None:
    """REQ-SAMPLE-020, SCENARIO-SAMPLE-033: crossover found when speedup >= 5x at n=2000."""
    result = KAEMExtendedResult(
        n_vars_tested=[1000, 2000],
        speedups=[2.0, 6.0],
        prior_max_n=1000,
    )
    assert result.crossover_n_vars == 2000
    assert result.kaem_viable_for_cpu is True
    assert result.fpga_path_recommended is False
    assert result.vs_prior == "crossover_found"


def test_crossover_found_early_stop() -> None:
    """REQ-SAMPLE-020: early stop means only the first crossover n_vars is recorded."""
    result = KAEMExtendedResult(
        n_vars_tested=[1000, 2000, 3000],
        speedups=[2.0, 5.5, 8.0],
        prior_max_n=1000,
    )
    assert result.crossover_n_vars == 2000
    assert result.kaem_viable_for_cpu is True
    assert result.fpga_path_recommended is False
    assert result.vs_prior == "crossover_found"


# ---------------------------------------------------------------------------
# SCENARIO-SAMPLE-034: no crossover at n=5000 → FPGA path
# ---------------------------------------------------------------------------


def test_no_crossover_at_n5000_fpga_path() -> None:
    """REQ-SAMPLE-021, SCENARIO-SAMPLE-034: fpga_path_recommended when no crossover at n=5000."""
    result = KAEMExtendedResult(
        n_vars_tested=[1000, 2000, 3000, 5000],
        speedups=[1.5, 2.0, 2.5, 3.0],
        prior_max_n=1000,
    )
    assert result.crossover_n_vars is None
    assert result.kaem_viable_for_cpu is False
    assert result.fpga_path_recommended is True
    assert result.vs_prior == "no_crossover_extended"


# ---------------------------------------------------------------------------
# prior_max_n provenance field
# ---------------------------------------------------------------------------


def test_prior_max_n_stored() -> None:
    """REQ-SAMPLE-020: prior_max_n is stored verbatim for provenance."""
    result = KAEMExtendedResult(
        n_vars_tested=[1000, 2000],
        speedups=[1.0, 1.2],
        prior_max_n=1000,
    )
    assert result.prior_max_n == 1000


def test_prior_max_n_arbitrary_value() -> None:
    """REQ-SAMPLE-020: prior_max_n accepts any positive int."""
    result = KAEMExtendedResult(
        n_vars_tested=[500],
        speedups=[1.0],
        prior_max_n=500,
    )
    assert result.prior_max_n == 500


# ---------------------------------------------------------------------------
# Inherited KAEMCrossoverResult validation (ValueError paths)
# ---------------------------------------------------------------------------


def test_empty_n_vars_raises() -> None:
    """REQ-SAMPLE-020: inherits validation — empty n_vars_tested raises ValueError."""
    with pytest.raises(ValueError, match="non-empty"):
        KAEMExtendedResult([], [], prior_max_n=1000)


def test_mismatched_lengths_raises() -> None:
    """REQ-SAMPLE-020: inherits validation — mismatched list lengths raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        KAEMExtendedResult([1000, 2000], [1.0], prior_max_n=1000)


# ---------------------------------------------------------------------------
# Exact threshold boundary
# ---------------------------------------------------------------------------


def test_exactly_5x_counts_as_crossover() -> None:
    """REQ-SAMPLE-020: speedup exactly equal to VIABILITY_THRESHOLD (5.0) is a crossover."""
    result = KAEMExtendedResult(
        n_vars_tested=[1000, 3000],
        speedups=[2.0, 5.0],
        prior_max_n=1000,
    )
    assert result.crossover_n_vars == 3000
    assert result.kaem_viable_for_cpu is True


def test_just_below_5x_is_not_crossover() -> None:
    """REQ-SAMPLE-020: speedup just below VIABILITY_THRESHOLD is not a crossover."""
    result = KAEMExtendedResult(
        n_vars_tested=[1000, 5000],
        speedups=[2.0, 4.99],
        prior_max_n=1000,
    )
    assert result.crossover_n_vars is None
    assert result.kaem_viable_for_cpu is False
    assert result.fpga_path_recommended is True


# ---------------------------------------------------------------------------
# Inherited helpers still work
# ---------------------------------------------------------------------------


def test_speedup_at_inherited() -> None:
    """REQ-SAMPLE-020: speedup_at() inherited from KAEMCrossoverResult works on extended data."""
    result = KAEMExtendedResult(
        n_vars_tested=[1000, 2000, 3000],
        speedups=[1.2, 3.4, 5.6],
        prior_max_n=1000,
    )
    assert result.speedup_at(2000) == pytest.approx(3.4)
    assert result.max_speedup == pytest.approx(5.6)


def test_speedup_at_missing_raises() -> None:
    """REQ-SAMPLE-020: speedup_at() raises KeyError for an n_vars not in the tested list."""
    result = KAEMExtendedResult(
        n_vars_tested=[1000],
        speedups=[1.5],
        prior_max_n=1000,
    )
    with pytest.raises(KeyError):
        result.speedup_at(9999)
