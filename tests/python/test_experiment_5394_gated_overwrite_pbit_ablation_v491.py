"""Tests for Exp5394 gated overwrite p-bit action-sequence ablation.

Spec refs: REQ-VERIFY-5394, SCENARIO-VERIFY-5394.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5394_gated_overwrite_pbit_ablation_v491 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5394_gated_overwrite_pbit_ablation_v491.py "
    "-q -o addopts="
)


def test_req_verify_5394_spec_declares_gated_action_sequence_contract() -> None:
    """REQ-VERIFY-5394: OpenSpec anchors the gated action ablation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5394") : spec.index("### REQ-VERIFY-5393")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5394",
        "SCENARIO-VERIFY-5394",
        str(exp.RESULT_RELATIVE_PATH),
        str(exp.GATE_RELATIVE_PATH),
        "overwrite_guidance_corrigendum_clean=true",
        "action-sequence fixtures",
        "trajectory or ordering hints",
        "symbolic solver",
        "monolithic",
        "hint_only",
        "pbit_boundary_hint",
        "fallback_only",
        "pbit_boundary_ablation_ready",
        "hardware_speedup_claim=false",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_verify_5394_gate_reads_exp5393_clean_value() -> None:
    """REQ-VERIFY-5394: Exp5393 must be clean before the ablation runs."""

    gate = exp.load_gate_source(REPO)

    assert gate == {
        "artifact_path": str(exp.GATE_RELATIVE_PATH),
        "gate_field": "overwrite_guidance_corrigendum_clean",
        "gate_value": True,
        "source_status": "complete",
    }


def test_scenario_verify_5394_action_fixtures_cover_solver_hint_actions() -> None:
    """SCENARIO-VERIFY-5394: p-bit hints are accepted or overwritten safely."""

    fixtures = exp.build_action_sequence_fixtures()
    rows = exp.run_ablation(REPO)["mode_results"]
    pbit_rows = [row for row in rows if row["mode"] == "pbit_boundary_hint"]

    assert len(fixtures) == exp.EXPECTED_FIXTURE_COUNT
    assert {fixture.fixture_id for fixture in fixtures} == {
        "act_unlock_deliver",
        "act_assemble_inspect",
        "act_prepare_bake",
        "act_build_deploy",
    }
    assert {fixture.pbit_hint_kind for fixture in fixtures} == {
        "trajectory",
        "ordering",
    }
    assert {row["solver_action"] for row in rows} >= {"accepted", "overwritten", "ignored"}
    assert {row["pbit_hint_class"] for row in pbit_rows} == {
        "helpful",
        "harmful",
        "contradictory",
    }
    assert all(row["solver_authoritative"] is True for row in rows)
    assert all(row["final_valid"] is True for row in rows)
    assert all(row["unsafe_false_accept"] is False for row in rows)
    assert all(row["hardware_speedup_claim"] is False for row in rows)
    assert exp._is_valid_prefix(fixtures[0], ("pickup_key", "pickup_key")) is False
    assert exp._is_valid_prefix(fixtures[0], ("unknown_action",)) is False


def test_scenario_verify_5394_compares_modes_and_preserves_authority() -> None:
    """SCENARIO-VERIFY-5394: all mode metrics are relative to monolithic."""

    diagnostic = exp.run_ablation(REPO)

    assert diagnostic["gate_source"]["gate_value"] is True
    assert diagnostic["fixture_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert diagnostic["compared_modes"] == list(exp.COMPARED_MODES)
    assert diagnostic["validity_rate_by_mode"] == {
        "monolithic": 1.0,
        "hint_only": 1.0,
        "pbit_boundary_hint": 1.0,
        "fallback_only": 1.0,
    }
    assert diagnostic["conflict_delta_by_mode"]["monolithic"] == 0
    assert diagnostic["convergence_delta_by_mode"]["monolithic"] == 0
    assert diagnostic["conflict_delta_by_mode"]["pbit_boundary_hint"] < 0
    assert diagnostic["convergence_delta_by_mode"]["pbit_boundary_hint"] < 0
    assert diagnostic["overwrite_rate"] == pytest.approx(1.0)
    assert diagnostic["fallback_completeness_rate"] == pytest.approx(1.0)
    assert diagnostic["unsafe_false_accepts"] == 0
    assert diagnostic["pbit_proposal_caused_harm"] is False
    assert diagnostic["pbit_boundary_ablation_ready"] is True


def test_req_verify_5394_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5394: artifact exposes the required ablation fields."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=[TEST_COMMAND])

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.07.491"
    assert artifact["gate_source"]["gate_value"] is True
    assert artifact["simulation_only"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["fixture_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert artifact["compared_modes"] == list(exp.COMPARED_MODES)
    assert artifact["pbit_boundary_ablation_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == [{"command": TEST_COMMAND, "outcome": "passed"}]
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES


def test_req_verify_5394_blocked_when_exp5393_gate_is_false(tmp_path: Path) -> None:
    """REQ-VERIFY-5394: false upstream gate emits blocked artifact."""

    missing_gate = exp.load_gate_source(tmp_path)
    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[TEST_COMMAND],
        gate_override={
            "artifact_path": str(exp.GATE_RELATIVE_PATH),
            "gate_field": "overwrite_guidance_corrigendum_clean",
            "gate_value": False,
            "source_status": "flagged",
        },
    )

    assert missing_gate["gate_value"] is False
    assert missing_gate["source_status"] == "missing"
    assert artifact["status"] == "blocked"
    assert artifact["gate_source"]["gate_value"] is False
    assert artifact["pbit_boundary_ablation_ready"] is False
    assert artifact["simulation_only"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert "exp5393_gate_failed" in artifact["readiness_blockers"]
    assert artifact["honest_verdict"].startswith("blocked:")
    exp.validate_artifact(artifact)


def test_req_verify_5394_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5394: checked-in artifact is stable under replay."""

    checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=checked_in["tests_run"])

    assert checked_in == replay
    assert checked_in["status"] == "complete"
    assert checked_in["pbit_boundary_ablation_ready"] is True
    assert checked_in["simulation_only"] is True
    assert checked_in["hardware_speedup_claim"] is False
    exp.validate_artifact(checked_in)


def test_req_verify_5394_validation_rejects_unsafe_or_claim_drift() -> None:
    """REQ-VERIFY-5394: validation fails closed on unsafe schema drift."""

    artifact = exp.build_artifact(root=REPO, tests_run=[TEST_COMMAND])

    bad_status = deepcopy(artifact)
    bad_status["status"] = "done"
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_status)

    bad_milestone = deepcopy(artifact)
    bad_milestone["milestone"] = "2026.07.490"
    with pytest.raises(ValueError, match="milestone"):
        exp.validate_artifact(bad_milestone)

    bad_simulation = deepcopy(artifact)
    bad_simulation["simulation_only"] = False
    with pytest.raises(ValueError, match="simulation_only"):
        exp.validate_artifact(bad_simulation)

    bad_hardware = deepcopy(artifact)
    bad_hardware["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        exp.validate_artifact(bad_hardware)

    bad_modes = deepcopy(artifact)
    bad_modes["compared_modes"] = ["monolithic"]
    with pytest.raises(ValueError, match="compared_modes"):
        exp.validate_artifact(bad_modes)

    bad_unsafe = deepcopy(artifact)
    bad_unsafe["unsafe_false_accepts"] = 1
    bad_unsafe["pbit_boundary_ablation_ready"] = False
    with pytest.raises(ValueError, match="unsafe_false_accepts"):
        exp.validate_artifact(bad_unsafe)

    bad_ready = deepcopy(artifact)
    bad_ready["validity_rate_by_mode"]["pbit_boundary_hint"] = 0.75
    with pytest.raises(ValueError, match="validity_rate_by_mode"):
        exp.validate_artifact(bad_ready)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)
