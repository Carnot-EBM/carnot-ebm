"""Tests for Exp5407 p-bit/QUBO active-constraint stress diagnostic.

Spec refs: REQ-VERIFY-5407, SCENARIO-VERIFY-5407.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_5407_pbit_qubo_active_constraint_stress_v492 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5407_pbit_qubo_active_constraint_stress_v492.py "
    "-q --no-cov"
)


def test_req_verify_5407_spec_declares_pbit_qubo_stress_contract() -> None:
    """REQ-VERIFY-5407: OpenSpec anchors the gated p-bit/QUBO stress diagnostic."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5407") : spec.index("### REQ-VERIFY-5406")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5407",
        "SCENARIO-VERIFY-5407",
        str(exp.RESULT_RELATIVE_PATH),
        str(exp.GATE_RELATIVE_PATH),
        "active_constraint_warmstart_ready=true",
        "sorting-network permutation baseline",
        "QUBO-style precedence energy",
        "exact enumeration",
        "deterministic_solver",
        "pbit_boundary_hint",
        "active_constraint_hint",
        "adversarial_hint",
        "hardware_speedup_claim",
        "pbit_qubo_stress_ready",
        "verifier_ensemble_against_cached_candidates",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_req_verify_5407_gate_reads_exp5406_ready_value() -> None:
    """REQ-VERIFY-5407: Exp5406 readiness is checked before p-bit work."""

    gate = exp.load_active_constraint_gate(REPO)

    assert gate == {
        "artifact_path": str(exp.GATE_RELATIVE_PATH),
        "gate_field": "active_constraint_warmstart_ready",
        "gate_value": True,
        "source_status": "complete",
    }


def test_scenario_verify_5407_sorting_network_qubo_baselines_are_exact() -> None:
    """SCENARIO-VERIFY-5407: every tiny QUBO baseline is exactly enumerated."""

    fixtures = exp.build_stress_fixtures()
    baselines = exp.build_qubo_baselines(fixtures)

    assert len(fixtures) == exp.EXPECTED_FIXTURE_COUNT
    assert len(baselines) == exp.EXPECTED_QUBO_BASELINE_COUNT
    assert {fixture.active_hint_source for fixture in fixtures} == {
        "exp5406_active_constraint_warmstart"
    }
    for fixture, baseline in zip(fixtures, baselines, strict=True):
        assert baseline["fixture_id"] == fixture.fixture_id
        assert baseline["exact_enumerated"] is True
        assert baseline["enumerated_permutation_count"] == math.factorial(len(fixture.actions))
        assert baseline["sorting_network_comparator_count"] > 0
        assert baseline["qubo_variable_count"] == len(fixture.actions) ** 2
        assert baseline["exact_min_energy"] == 0
        assert baseline["deterministic_fallback_energy"] == baseline["exact_min_energy"]
        assert baseline["deterministic_agrees_with_exact"] is True
        assert exp.qubo_precedence_energy(fixture, fixture.expected_sequence) == 0


def test_scenario_verify_5407_modes_preserve_solver_authority() -> None:
    """SCENARIO-VERIFY-5407: p-bit, active, and adversarial rows stay checked."""

    diagnostic = exp.run_diagnostic(REPO)
    rows = diagnostic["row_records"]
    by_mode = {
        mode: [row for row in rows if row["mode"] == mode]
        for mode in exp.COMPARED_MODES
    }

    assert diagnostic["gated_on_active_constraint_ready"] is True
    assert diagnostic["fixture_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert diagnostic["qubo_baseline_count"] == exp.EXPECTED_QUBO_BASELINE_COUNT
    assert diagnostic["compared_modes"] == list(exp.COMPARED_MODES)
    assert all(by_mode.values())
    assert len(rows) == exp.EXPECTED_FIXTURE_COUNT * len(exp.COMPARED_MODES)
    assert all(row["solver_authoritative"] is True for row in rows)
    assert all(row["accepted_without_verification"] is False for row in rows)
    assert all(row["final_valid"] is True for row in rows)
    assert all(row["unsafe_false_accept"] is False for row in rows)
    assert all(row["hardware_speedup_claim"] is False for row in rows)
    assert all(row["fallback_used"] is False for row in by_mode["deterministic_solver"])
    assert all(row["hint_decision"] == "accepted" for row in by_mode["active_constraint_hint"])
    assert all(row["fallback_used"] is True for row in by_mode["adversarial_hint"])
    assert all(
        row["hint_decision"] in {"rejected", "overwritten"}
        for row in by_mode["adversarial_hint"]
    )
    assert all(row["sample_count"] == 0 for row in by_mode["deterministic_solver"])
    assert all(row["sample_count"] > 0 for row in by_mode["pbit_boundary_hint"])
    assert all(row["accepted_sample_count"] > 0 for row in by_mode["pbit_boundary_hint"])
    assert all(row["rejected_sample_count"] > 0 for row in by_mode["pbit_boundary_hint"])
    assert diagnostic["exact_enumeration_agreement_rate"] == pytest.approx(1.0)
    assert 0 < diagnostic["pbit_acceptance_rate"] < 1
    assert diagnostic["solver_conflict_delta"] > 0
    assert diagnostic["solver_iteration_delta"] > 0
    assert diagnostic["fallback_rate"] > 0
    assert diagnostic["unsafe_false_accept_rate"] == pytest.approx(0.0)
    assert diagnostic["pbit_qubo_stress_ready"] is True


def test_scenario_verify_5407_bad_pbit_samples_cannot_bypass_solver() -> None:
    """SCENARIO-VERIFY-5407: invalid p-bit samples are rejected before acceptance."""

    fixture = exp.build_stress_fixtures()[0]
    baseline = exp.exact_enumerate_fixture(fixture)
    row = exp.evaluate_fixture_mode(fixture, "pbit_boundary_hint", baseline)
    adversarial = exp.evaluate_fixture_mode(fixture, "adversarial_hint", baseline)

    assert row["mode"] == "pbit_boundary_hint"
    assert row["sample_count"] == exp.PBIT_SAMPLES_PER_FIXTURE
    assert row["accepted_sample_count"] > 0
    assert row["rejected_sample_count"] > 0
    assert row["accepted_without_verification"] is False
    assert row["solver_authoritative"] is True
    assert row["unsafe_false_accept"] is False
    assert row["exact_enumeration_agrees"] is True
    assert any(not sample["valid"] for sample in row["sample_records"])

    assert adversarial["hint_decision"] == "overwritten"
    assert adversarial["fallback_used"] is True
    assert adversarial["final_valid"] is True
    assert adversarial["unsafe_false_accept"] is False


def test_req_verify_5407_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5407: artifact exposes all required stress-diagnostic fields."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=[TEST_COMMAND])

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.07.492"
    assert artifact["gated_on_active_constraint_ready"] is True
    assert artifact["fixture_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert artifact["qubo_baseline_count"] == exp.EXPECTED_QUBO_BASELINE_COUNT
    assert artifact["exact_enumeration_agreement_rate"] == pytest.approx(1.0)
    assert 0 < artifact["pbit_acceptance_rate"] < 1
    assert artifact["solver_conflict_delta"] > 0
    assert artifact["fallback_rate"] > 0
    assert artifact["unsafe_false_accept_rate"] == pytest.approx(0.0)
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["pbit_qubo_stress_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == [{"command": TEST_COMMAND, "outcome": "passed"}]
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES


def test_req_verify_5407_blocked_when_exp5406_gate_is_false(tmp_path: Path) -> None:
    """REQ-VERIFY-5407: false upstream gate blocks p-bit/QUBO stress claims."""

    missing_gate = exp.load_active_constraint_gate(tmp_path)
    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[TEST_COMMAND],
        gate_override={
            "artifact_path": str(exp.GATE_RELATIVE_PATH),
            "gate_field": "active_constraint_warmstart_ready",
            "gate_value": False,
            "source_status": "blocked",
        },
    )

    assert missing_gate["gate_value"] is False
    assert missing_gate["source_status"] == "missing"
    assert artifact["status"] == "blocked"
    assert artifact["gated_on_active_constraint_ready"] is False
    assert artifact["fixture_count"] == 0
    assert artifact["qubo_baseline_count"] == 0
    assert artifact["pbit_qubo_stress_ready"] is False
    assert artifact["hardware_speedup_claim"] is False
    assert "active_constraint_warmstart_not_ready" in artifact["readiness_blockers"]
    assert artifact["honest_verdict"].startswith("blocked:")
    exp.validate_artifact(artifact)


def test_req_verify_5407_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5407: checked-in JSON is stable under deterministic replay."""

    checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=checked_in["tests_run"])

    assert checked_in == replay
    assert checked_in["pbit_qubo_stress_ready"] is True
    assert checked_in["hardware_speedup_claim"] is False
    exp.validate_artifact(checked_in)


def test_req_verify_5407_validation_rejects_schema_and_claim_drift() -> None:
    """REQ-VERIFY-5407: validation fails closed on unsafe schema drift."""

    artifact = exp.build_artifact(root=REPO, tests_run=[TEST_COMMAND])

    bad_modes = deepcopy(artifact)
    bad_modes["compared_modes"] = ["deterministic_solver"]
    with pytest.raises(ValueError, match="compared_modes"):
        exp.validate_artifact(bad_modes)

    bad_exact = deepcopy(artifact)
    bad_exact["exact_enumeration_agreement_rate"] = 0.5
    with pytest.raises(ValueError, match="exact_enumeration_agreement_rate"):
        exp.validate_artifact(bad_exact)

    bad_hardware = deepcopy(artifact)
    bad_hardware["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        exp.validate_artifact(bad_hardware)

    bad_unsafe = deepcopy(artifact)
    bad_unsafe["unsafe_false_accept_rate"] = 0.1
    bad_unsafe["pbit_qubo_stress_ready"] = False
    with pytest.raises(ValueError, match="unsafe_false_accept_rate"):
        exp.validate_artifact(bad_unsafe)

    bad_ready = deepcopy(artifact)
    bad_ready["solver_conflict_delta"] = 0
    with pytest.raises(ValueError, match="solver_conflict_delta"):
        exp.validate_artifact(bad_ready)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)
