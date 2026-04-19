"""Tests for KAEMCrossoverResult.

Spec: REQ-SAMPLE-019, SCENARIO-SAMPLE-032
"""

import pytest

from carnot.models.kaem_crossover import KAEMCrossoverResult


# ---------------------------------------------------------------------------
# crossover_n_vars
# ---------------------------------------------------------------------------


def test_crossover_n_vars_found_at_first_eligible():
    # REQ-SAMPLE-019: crossover at first n_vars where speedup >= 5.0
    result = KAEMCrossoverResult([100, 200, 300], [1.5, 3.0, 6.0])
    assert result.crossover_n_vars == 300


def test_crossover_n_vars_first_in_list():
    # Crossover at the very first n_vars in the list
    result = KAEMCrossoverResult([100, 200], [5.0, 8.0])
    assert result.crossover_n_vars == 100


def test_crossover_n_vars_none_when_not_found():
    # REQ-SAMPLE-019: None when no n_vars reaches the threshold
    result = KAEMCrossoverResult([100, 200, 300, 500, 1000], [1.0, 1.5, 2.0, 3.0, 4.9])
    assert result.crossover_n_vars is None


def test_crossover_n_vars_exactly_at_threshold():
    # Exactly 5.0 should count as a crossover
    result = KAEMCrossoverResult([100, 500], [2.0, 5.0])
    assert result.crossover_n_vars == 500


# ---------------------------------------------------------------------------
# kaem_viable_for_production
# ---------------------------------------------------------------------------


def test_kaem_viable_for_production_true_when_crossover_found():
    # SCENARIO-SAMPLE-032: viable when crossover found
    result = KAEMCrossoverResult([100, 200, 300], [1.5, 3.0, 6.0])
    assert result.kaem_viable_for_production is True


def test_kaem_viable_for_production_false_when_no_crossover():
    result = KAEMCrossoverResult([100, 200], [1.0, 2.0])
    assert result.kaem_viable_for_production is False


# ---------------------------------------------------------------------------
# max_speedup
# ---------------------------------------------------------------------------


def test_max_speedup():
    # SCENARIO-SAMPLE-032: max_speedup == 6.0
    result = KAEMCrossoverResult([100, 200, 300], [1.5, 3.0, 6.0])
    assert result.max_speedup == 6.0


def test_max_speedup_single_element():
    result = KAEMCrossoverResult([100], [3.7])
    assert result.max_speedup == 3.7


# ---------------------------------------------------------------------------
# speedup_at
# ---------------------------------------------------------------------------


def test_speedup_at_returns_correct_value():
    # SCENARIO-SAMPLE-032: speedup_at(200) == 3.0
    result = KAEMCrossoverResult([100, 200, 300], [1.5, 3.0, 6.0])
    assert result.speedup_at(200) == 3.0


def test_speedup_at_all_values():
    result = KAEMCrossoverResult([100, 500, 1000], [1.2, 4.5, 7.8])
    assert result.speedup_at(100) == 1.2
    assert result.speedup_at(500) == 4.5
    assert result.speedup_at(1000) == 7.8


def test_speedup_at_missing_key_raises():
    result = KAEMCrossoverResult([100, 200], [1.0, 2.0])
    with pytest.raises(KeyError):
        result.speedup_at(999)


# ---------------------------------------------------------------------------
# constructor validation
# ---------------------------------------------------------------------------


def test_empty_n_vars_raises():
    with pytest.raises(ValueError, match="non-empty"):
        KAEMCrossoverResult([], [])


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        KAEMCrossoverResult([100, 200], [1.0])


# ---------------------------------------------------------------------------
# VIABILITY_THRESHOLD constant
# ---------------------------------------------------------------------------


def test_viability_threshold_is_five():
    assert KAEMCrossoverResult.VIABILITY_THRESHOLD == 5.0
