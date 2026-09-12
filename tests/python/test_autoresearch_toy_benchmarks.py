"""Tests for python/carnot/autoresearch/toy_benchmarks.py -- the real,
independently-computable potential functions closing the self-reported-
fitness gap (adversarial review 2026-09-12, finding 1).

Spec: REQ-AUTO-021
"""

from __future__ import annotations

import math
from unittest.mock import patch

from carnot.autoresearch.toy_benchmarks import (
    BENCHMARK_ENERGY_FUNCTIONS,
    double_well_energy,
    recompute_final_energy,
    rosenbrock_energy,
)


class TestDoubleWellEnergy:
    def test_global_minimum_is_zero(self) -> None:
        assert double_well_energy([1.0, 1.0]) == 0.0
        assert double_well_energy([-1.0, -1.0, 1.0]) == 0.0

    def test_origin_is_not_optimal(self) -> None:
        assert double_well_energy([0.0, 0.0]) == 2.0  # (0-1)^2 * 2 dims

    def test_matches_known_value(self) -> None:
        assert math.isclose(double_well_energy([2.0]), 9.0)  # (4-1)^2


class TestRosenbrockEnergy:
    def test_global_minimum_is_zero(self) -> None:
        assert rosenbrock_energy([1.0, 1.0]) == 0.0
        assert rosenbrock_energy([1.0, 1.0, 1.0]) == 0.0

    def test_origin_is_not_optimal(self) -> None:
        assert rosenbrock_energy([0.0, 0.0]) == 1.0

    def test_requires_at_least_two_dimensions(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            rosenbrock_energy([1.0])


class TestRecomputeFinalEnergy:
    def test_real_state_recomputes_the_true_energy(self) -> None:
        assert recompute_final_energy("double_well", [1.0, 1.0]) == 0.0
        assert recompute_final_energy("rosenbrock", [1.0, 1.0]) == 0.0

    def test_a_fabricated_number_cannot_influence_this_at_all(self) -> None:
        """There is no 'claimed energy' parameter to this function -- the
        only input that matters is the state. This is the structural fix:
        a hypothesis has no channel to report a number directly."""
        e1 = recompute_final_energy("double_well", [0.0, 0.0])
        e2 = recompute_final_energy("double_well", [0.0, 0.0])
        assert e1 == e2 == 2.0

    def test_unknown_benchmark_name_returns_none(self) -> None:
        assert recompute_final_energy("made_up_bench", [1.0, 1.0]) is None

    def test_missing_state_returns_none(self) -> None:
        assert recompute_final_energy("double_well", None) is None
        assert recompute_final_energy("double_well", []) is None

    def test_wrong_type_state_returns_none_not_raise(self) -> None:
        assert recompute_final_energy("double_well", "not a list") is None
        assert recompute_final_energy("double_well", {"x": 1}) is None
        assert recompute_final_energy("double_well", [1.0, "not a number"]) is None

    def test_nan_and_inf_state_values_return_none(self) -> None:
        assert recompute_final_energy("double_well", [float("nan"), 1.0]) is None
        assert recompute_final_energy("double_well", [float("inf"), 1.0]) is None

    def test_too_short_state_for_rosenbrock_returns_none_not_raise(self) -> None:
        assert recompute_final_energy("rosenbrock", [1.0]) is None

    def test_oversized_state_returns_none(self) -> None:
        assert recompute_final_energy("double_well", [1.0] * 1000) is None

    def test_non_finite_computed_energy_returns_none(self) -> None:
        with patch.dict(BENCHMARK_ENERGY_FUNCTIONS, {"non_finite": lambda _state: math.inf}):
            assert recompute_final_energy("non_finite", [1.0]) is None

    def test_registry_covers_the_seeded_benchmarks(self) -> None:
        assert set(BENCHMARK_ENERGY_FUNCTIONS) == {"double_well", "rosenbrock"}
