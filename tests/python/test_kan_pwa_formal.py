"""Tests for GS-KAN PWA formal verification.

Spec: REQ-VERIFY-1372, SCENARIO-VERIFY-1372
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.models.gskan import GSKANEnergy  # noqa: E402
from carnot.verify.kan_pwa_formal import (  # noqa: E402
    build_gskan_pwa_abstraction,
    interval_arithmetic_energy_bound,
    maximize_energy_manual_lp,
    verify_energy_bound,
)
from scripts import experiment_1372_optimal_kan_pwa_formal_verification as exp1372  # noqa: E402


def _toy_gskan() -> GSKANEnergy:
    model = GSKANEnergy(n_vars=4, n_groups=2, n_knots=4, seed=0)
    model.group_ctrl = np.asarray(
        [
            [0.0, 0.2, 0.7, 1.0],
            [0.0, 0.1, 0.4, 0.9],
        ],
        dtype=np.float32,
    )
    model.proj_weights = np.asarray([0.5, 1.0, 1.5, 0.25], dtype=np.float32)
    return model


def test_knot_aligned_pwa_abstraction_is_exact_for_current_gskan() -> None:
    """REQ-VERIFY-1372: native GS-KAN degree-1 splines have exact PWA pieces."""
    model = _toy_gskan()
    abstraction = build_gskan_pwa_abstraction(
        model,
        pwa_segments_per_spline=model.n_knots - 1,
        error_grid_points=41,
    )

    assert abstraction.spline_count == 2
    assert abstraction.pwa_segments_per_spline == 3
    assert abstraction.max_abs_error < 1e-7

    xs = np.linspace(-1.0, 1.0, 31)
    for group_index, spline in enumerate(abstraction.splines):
        original = model._eval_spline_group(group_index, xs)
        approximated = np.asarray([spline.evaluate(float(x)) for x in xs])
        assert np.max(np.abs(original - approximated)) < 1e-7


def test_manual_lp_verifies_energy_bound_over_input_box() -> None:
    """SCENARIO-VERIFY-1372: LP result verifies only when bound is below threshold."""
    model = _toy_gskan()
    abstraction = build_gskan_pwa_abstraction(model)
    input_bounds = ((-0.5, 0.25), (-0.25, 0.5), (-1.0, 0.0), (0.0, 1.0))

    lp_result = maximize_energy_manual_lp(abstraction, model.proj_weights, input_bounds)
    expected = model.energy(np.asarray(lp_result.maximizer, dtype=np.float32))
    assert lp_result.integer_constraints_needed is False
    assert lp_result.certified_upper_bound == pytest.approx(expected, abs=1e-7)

    verified = verify_energy_bound(model, input_bounds, threshold=expected + 1e-5)
    assert verified.result == "verified"
    assert verified.formal_property_verified is True

    counterexample = verify_energy_bound(model, input_bounds, threshold=expected - 1e-5)
    assert counterexample.result == "counterexample"
    assert counterexample.formal_property_verified is False


def test_interval_arithmetic_matches_exact_pwa_for_toy_layer() -> None:
    """REQ-VERIFY-1372: interval baseline is reported beside the PWA/LP bound."""
    model = _toy_gskan()
    abstraction = build_gskan_pwa_abstraction(model)
    input_bounds = ((-1.0, 0.3), (-0.2, 0.8), (-0.4, 0.6), (-0.5, 0.5))

    lp_result = maximize_energy_manual_lp(abstraction, model.proj_weights, input_bounds)
    interval = interval_arithmetic_energy_bound(model, input_bounds)

    assert interval.upper_bound == pytest.approx(lp_result.certified_upper_bound, abs=1e-7)
    assert interval.lower_bound <= interval.upper_bound


def test_experiment_1372_writes_required_artifact_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-1372: experiment runner writes the required formal artifact schema."""
    deliverable = tmp_path / "experiment_1372.json"

    artifact = exp1372.run_experiment(deliverable_path=deliverable, n_epochs=2)
    payload = json.loads(deliverable.read_text())

    assert payload == artifact
    assert exp1372.REQUIRED_ARTIFACT_FIELDS <= set(payload)
    assert payload["status"] == "complete"
    assert payload["milp_verification_result"] == "verified"
    assert payload["formal_property_verified"] is True
    assert payload["kan_formal_claim_allowed"] is True
    assert payload["hardware_correctness_claimed"] is False
    assert "no_hardware_claim" in payload["honest_verdict"]
