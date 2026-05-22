"""REQ-KAN-2871 tests for the tiny KAN PWA/MILP verifier prototype.

Scenario: SCENARIO-KAN-2871.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verify.kan_pwa_milp_tiny import (
    REQUIRED_ARTIFACT_FIELDS,
    build_quadratic_pwa,
    check_tiny_property,
    validate_artifact,
    write_experiment_artifact,
)


def test_req_kan_2871_is_spec_anchored() -> None:
    """REQ-KAN-2871: the tiny PWA/MILP prototype is declared before code."""
    spec = Path("openspec/capabilities/kan/spec.md").read_text(encoding="utf-8")

    assert "REQ-KAN-2871" in spec
    assert "SCENARIO-KAN-2871" in spec


def test_quadratic_pwa_has_exact_local_and_global_error_bounds() -> None:
    """REQ-KAN-2871: PWA chord envelopes expose explicit error bounds."""
    pwa = build_quadratic_pwa()

    assert pwa.n_pieces == 4
    assert pwa.local_error_bound == pytest.approx(0.0625)
    assert pwa.global_error_bound == pytest.approx(0.0625)
    assert pwa.evaluate_center(-0.25) == pytest.approx(0.125)
    assert pwa.evaluate_lower(-0.25) == pytest.approx(0.0625)
    assert pwa.evaluate_upper(-0.25) == pytest.approx(0.125)

    upper_bound, witness_x = pwa.certified_upper_bound(-0.5, 0.5)
    assert upper_bound == pytest.approx(0.25)
    assert witness_x in {-0.5, 0.5}

    shifted_upper, shifted_witness_x = pwa.certified_upper_bound(-0.25, 0.5)
    assert shifted_upper == pytest.approx(0.25)
    assert shifted_witness_x == 0.5

    with pytest.raises(ValueError, match="outside the PWA domain"):
        pwa.evaluate_upper(1.5)


def test_tiny_property_uses_exact_fallback_without_milp_solver() -> None:
    """SCENARIO-KAN-2871: exact PWA vertex enumeration verifies the toy property."""
    pwa = build_quadratic_pwa()

    result = check_tiny_property(pwa)

    assert result.property_verified is True
    assert result.solver_used is None
    assert result.checker_method == "exact_enumerated_pwa_vertices"
    assert result.certified_upper_bound == pytest.approx(0.25)
    assert result.as_serializable()["property_verified"] is True

    failed = check_tiny_property(pwa, threshold=0.249)
    assert failed.property_verified is False


def test_experiment_artifact_schema_and_validation(tmp_path: Path) -> None:
    """REQ-KAN-2871: artifact includes every required solver-boundary field."""
    path = tmp_path / "experiment_2871_kan_pwa_milp_tiny_verifier_v1.json"

    artifact = write_experiment_artifact(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == artifact
    assert REQUIRED_ARTIFACT_FIELDS <= set(payload)
    assert payload["honest_verdict"].startswith("complete_with_exact_enumerated_fallback")
    assert payload["kan_pwa_milp_verifier_ready"] is True
    assert payload["pwa_abstraction_built"] is True
    assert payload["milp_or_exact_property_checked"] is True
    assert payload["solver_used"] is None
    assert payload["property_verified"] is True
    assert payload["local_error_bound"] == pytest.approx(0.0625)
    assert payload["global_error_bound"] == pytest.approx(0.0625)
    assert payload["n_pwa_pieces"] == 4
    assert payload["blocked_reason"] is None
    assert payload["run_date"] == "20260522"
    assert len(payload["reproducibility_checksum"]) == 64
    assert "solver_boundary" in payload["field_principles"]

    assert validate_artifact(payload) == payload
    incomplete = dict(payload)
    incomplete.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_artifact(incomplete)
