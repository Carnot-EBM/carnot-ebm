"""Tests for KANHardwareAnalyzer — arXiv 2604.03345 LUT estimation.

Traces to:
  REQ-KAN-020 — KAEMEnergy MUST be analyzable for FPGA LUT budget using
                 arXiv 2604.03345 per-knot estimates.
  SCENARIO-KAN-030 — LUT estimate for N=8 KAEMEnergy on iCE40 HX8K.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from python.carnot.analysis.kan_hw_analysis import KANHardwareAnalyzer


# ---------------------------------------------------------------------------
# REQ-KAN-020 / SCENARIO-KAN-030: lut_estimate_layer formula
# ---------------------------------------------------------------------------


class TestLutEstimateLayer:
    """Validates the per-layer formula: fan_in * fan_out * n_knots * luts_per_segment."""

    def test_basic_formula(self) -> None:
        # REQ-KAN-020: per-knot formula from arXiv 2604.03345
        analyzer = KANHardwareAnalyzer(n_inputs=8, n_hidden=16, n_knots=10, luts_per_segment=10)
        result = analyzer.lut_estimate_layer(fan_in=8, fan_out=16)
        # 8 * 16 * 10 * 10 = 12800
        assert result == 8 * 16 * 10 * 10

    def test_single_connection(self) -> None:
        analyzer = KANHardwareAnalyzer(n_inputs=1, n_hidden=1, n_knots=8, luts_per_segment=12)
        result = analyzer.lut_estimate_layer(fan_in=1, fan_out=1)
        assert result == 1 * 1 * 8 * 12

    def test_output_layer(self) -> None:
        # Layer 2: n_hidden → 1 (scalar energy output)
        analyzer = KANHardwareAnalyzer(n_inputs=8, n_hidden=16, n_knots=10, luts_per_segment=10)
        result = analyzer.lut_estimate_layer(fan_in=16, fan_out=1)
        assert result == 16 * 1 * 10 * 10

    def test_luts_per_segment_scales_linearly(self) -> None:
        a1 = KANHardwareAnalyzer(n_inputs=4, n_hidden=8, n_knots=10, luts_per_segment=8)
        a2 = KANHardwareAnalyzer(n_inputs=4, n_hidden=8, n_knots=10, luts_per_segment=12)
        r1 = a1.lut_estimate_layer(4, 8)
        r2 = a2.lut_estimate_layer(4, 8)
        # midpoint 10 should be between 8 and 12 results
        assert r1 < r2
        assert r2 / r1 == pytest.approx(12 / 8)


# ---------------------------------------------------------------------------
# REQ-KAN-020: total_lut_estimate structure and values
# ---------------------------------------------------------------------------


class TestTotalLutEstimate:
    """Validates total_lut_estimate() output structure and arithmetic."""

    def test_structure(self) -> None:
        analyzer = KANHardwareAnalyzer(n_inputs=8, n_hidden=16, n_knots=10, luts_per_segment=10)
        result = analyzer.total_lut_estimate()
        assert set(result.keys()) == {
            "layer1_luts",
            "layer2_luts",
            "total_luts",
            "ice40_hx8k_budget",
            "within_budget",
        }

    def test_arithmetic_n8(self) -> None:
        # Primary analysis: N=8, n_hidden=16, n_knots=10, luts=10
        analyzer = KANHardwareAnalyzer(n_inputs=8, n_hidden=16, n_knots=10, luts_per_segment=10)
        result = analyzer.total_lut_estimate()
        assert result["layer1_luts"] == 8 * 16 * 10 * 10  # 12800
        assert result["layer2_luts"] == 16 * 1 * 10 * 10  # 1600
        assert result["total_luts"] == 12800 + 1600  # 14400
        assert result["ice40_hx8k_budget"] == 7680
        assert result["within_budget"] is False  # 14400 > 7680

    def test_within_budget_true_for_tiny_model(self) -> None:
        # A 2-knot model with 1 LUT/segment must be within budget
        analyzer = KANHardwareAnalyzer(n_inputs=2, n_hidden=2, n_knots=2, luts_per_segment=1)
        result = analyzer.total_lut_estimate()
        # 2*2*2*1 + 2*1*2*1 = 8 + 4 = 12 << 7680
        assert result["within_budget"] is True
        assert result["total_luts"] == 12

    def test_budget_constant(self) -> None:
        analyzer = KANHardwareAnalyzer(n_inputs=4, n_hidden=4, n_knots=4, luts_per_segment=4)
        result = analyzer.total_lut_estimate()
        assert result["ice40_hx8k_budget"] == 7680

    def test_total_equals_layer_sum(self) -> None:
        analyzer = KANHardwareAnalyzer(n_inputs=4, n_hidden=8, n_knots=6, luts_per_segment=9)
        result = analyzer.total_lut_estimate()
        assert result["total_luts"] == result["layer1_luts"] + result["layer2_luts"]


# ---------------------------------------------------------------------------
# REQ-KAN-020: synthesis_priority logic — all three branches
# ---------------------------------------------------------------------------


class TestSynthesisPriority:
    """Validates all three branches of synthesis_priority()."""

    def test_kan_priority_branch(self) -> None:
        # KAN fits and is cheaper than Ising → KAN_PRIORITY
        # Use tiny model: total_luts = 12 (from test above)
        analyzer = KANHardwareAnalyzer(n_inputs=2, n_hidden=2, n_knots=2, luts_per_segment=1)
        # Ising costs more (but still within budget)
        result = analyzer.synthesis_priority(ising_lut_count=100)
        assert result == "KAN_PRIORITY"

    def test_ising_priority_branch(self) -> None:
        # KAN is over budget or more expensive than Ising → ISING_PRIORITY
        # Use large model: N=8, hidden=16, knots=10 → 14400 LUTs (over 7680)
        analyzer = KANHardwareAnalyzer(n_inputs=8, n_hidden=16, n_knots=10, luts_per_segment=10)
        # Ising is within budget and cheaper
        result = analyzer.synthesis_priority(ising_lut_count=134)
        assert result == "ISING_PRIORITY"

    def test_both_feasible_branch_exact_tie(self) -> None:
        # KAN fits but is NOT cheaper than Ising (equal cost) → BOTH_FEASIBLE
        analyzer = KANHardwareAnalyzer(n_inputs=2, n_hidden=2, n_knots=2, luts_per_segment=1)
        # total_luts = 12, Ising also 12 → not "kan < ising" and not "ising < kan"
        result = analyzer.synthesis_priority(ising_lut_count=12)
        assert result == "BOTH_FEASIBLE"

    def test_both_feasible_when_neither_fits(self) -> None:
        # Both KAN and Ising are over budget — should return BOTH_FEASIBLE
        # (neither exclusive priority condition is satisfied)
        analyzer = KANHardwareAnalyzer(n_inputs=8, n_hidden=16, n_knots=10, luts_per_segment=10)
        # Ising also huge
        result = analyzer.synthesis_priority(ising_lut_count=99999)
        assert result == "BOTH_FEASIBLE"


# ---------------------------------------------------------------------------
# REQ-KAN-020: sensitivity_analysis dict
# ---------------------------------------------------------------------------


class TestSensitivityAnalysis:
    """Validates sensitivity_analysis() across N values."""

    def test_sensitivity_structure(self) -> None:
        analyzer = KANHardwareAnalyzer(n_inputs=8, n_hidden=16, n_knots=10, luts_per_segment=10)
        result = analyzer.sensitivity_analysis([4, 8, 16])
        assert set(result.keys()) == {4, 8, 16}

    def test_sensitivity_scales_with_n(self) -> None:
        # Larger N → more LUTs (layer1 grows linearly with n_inputs)
        analyzer = KANHardwareAnalyzer(n_inputs=4, n_hidden=16, n_knots=10, luts_per_segment=10)
        result = analyzer.sensitivity_analysis([4, 8, 16])
        assert result[4] < result[8] < result[16]

    def test_sensitivity_arithmetic(self) -> None:
        analyzer = KANHardwareAnalyzer(n_inputs=4, n_hidden=16, n_knots=10, luts_per_segment=10)
        result = analyzer.sensitivity_analysis([4])
        expected = 4 * 16 * 10 * 10 + 16 * 1 * 10 * 10  # 6400 + 1600 = 8000
        assert result[4] == expected


# ---------------------------------------------------------------------------
# REQ-KAN-020: constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_invalid_n_inputs(self) -> None:
        with pytest.raises(ValueError, match="n_inputs"):
            KANHardwareAnalyzer(n_inputs=0, n_hidden=16, n_knots=10)

    def test_invalid_n_hidden(self) -> None:
        with pytest.raises(ValueError, match="n_hidden"):
            KANHardwareAnalyzer(n_inputs=8, n_hidden=0, n_knots=10)

    def test_invalid_n_knots(self) -> None:
        with pytest.raises(ValueError, match="n_knots"):
            KANHardwareAnalyzer(n_inputs=8, n_hidden=16, n_knots=1)

    def test_invalid_luts_per_segment(self) -> None:
        with pytest.raises(ValueError, match="luts_per_segment"):
            KANHardwareAnalyzer(n_inputs=8, n_hidden=16, n_knots=10, luts_per_segment=0)
