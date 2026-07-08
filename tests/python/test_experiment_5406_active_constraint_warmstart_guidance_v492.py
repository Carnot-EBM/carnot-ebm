"""Tests for Exp5406 active-constraint warm-start guidance.

Spec refs: REQ-VERIFY-5406, SCENARIO-VERIFY-5406.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5406_active_constraint_warmstart_guidance_v492 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5406_active_constraint_warmstart_guidance_v492.py "
    "-q --no-cov"
)


def test_req_verify_5406_spec_declares_active_constraint_contract() -> None:
    """REQ-VERIFY-5406: OpenSpec anchors active-constraint warm-start rows."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5406") : spec.index("### REQ-VERIFY-5394")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5406",
        "SCENARIO-VERIFY-5406",
        str(exp.RESULT_RELATIVE_PATH),
        "active-constraint",
        "conflict-front",
        "no_hint",
        "stale_hint",
        "adversarial_hint",
        "candidate_hint",
        "verifier_ensemble_against_cached_candidates",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_req_verify_5406_fixtures_extend_existing_solver_guidance() -> None:
    """REQ-VERIFY-5406: fixtures include synthetic and Exp5394 carry-forward rows."""

    fixtures = exp.build_constraint_instances()
    synthetic = [item for item in fixtures if item.source_kind == "synthetic"]
    carry_forward = [item for item in fixtures if item.source_kind == "carry_forward_exp5394"]

    assert len(fixtures) == exp.EXPECTED_FIXTURE_COUNT
    assert synthetic
    assert carry_forward
    assert all(item.active_constraint_ids for item in fixtures)
    assert all(item.conflict_front for item in fixtures)
    assert {item.active_set_source for item in synthetic} == {"independent_spec"}
    assert {item.active_set_source for item in carry_forward} == {
        "solver_derived_from_exp5394_fixture"
    }
    assert all(
        item.extended_from
        == "experiment_5394_gated_overwrite_pbit_ablation_v491.ActionSequenceFixture"
        for item in carry_forward
    )
    assert all(
        {"active_constraint_hint", "conflict_front_hint"} <= set(item.hint_fields)
        for item in fixtures
    )


def test_scenario_verify_5406_modes_record_authority_decisions() -> None:
    """SCENARIO-VERIFY-5406: all hint modes are compared under solver authority."""

    diagnostic = exp.run_diagnostic()
    rows = diagnostic["row_records"]
    by_mode = {
        mode: [row for row in rows if row["hint_mode"] == mode]
        for mode in exp.HINT_MODES
    }

    assert diagnostic["fixture_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert diagnostic["hint_modes"] == list(exp.HINT_MODES)
    assert all(by_mode.values())
    assert len(rows) == exp.EXPECTED_FIXTURE_COUNT * len(exp.HINT_MODES)
    assert all(row["solver_authoritative"] is True for row in rows)
    assert all(row["final_valid"] is True for row in rows)
    assert all(row["unsafe_false_accept"] is False for row in rows)
    assert all(row["hint_decision"] == "ignored" for row in by_mode["no_hint"])
    assert all(row["hint_decision"] == "accepted" for row in by_mode["candidate_hint"])
    assert all(row["hint_decision"] == "rejected" for row in by_mode["stale_hint"])
    assert all(row["hint_decision"] == "overwritten" for row in by_mode["adversarial_hint"])
    assert all(row["fallback_used"] is True for row in by_mode["stale_hint"])
    assert all(row["fallback_used"] is True for row in by_mode["adversarial_hint"])
    assert all(row["active_constraint_precision"] == 1.0 for row in by_mode["candidate_hint"])
    assert all(row["active_constraint_recall"] == 1.0 for row in by_mode["candidate_hint"])
    assert diagnostic["solver_conflict_delta"] > 0
    assert diagnostic["solver_iteration_delta"] > 0
    assert diagnostic["solver_overwrite_rate"] > 0
    assert diagnostic["stale_hint_rejection_rate"] == pytest.approx(1.0)
    assert diagnostic["adversarial_hint_rejection_rate"] == pytest.approx(1.0)
    assert diagnostic["unsafe_false_accept_rate"] == pytest.approx(0.0)
    assert diagnostic["active_constraint_warmstart_ready"] is True


def test_scenario_verify_5406_wrong_hint_cannot_bypass_solver_authority() -> None:
    """SCENARIO-VERIFY-5406: adversarial hints are overwritten before acceptance."""

    fixture = exp.build_constraint_instances()[0]
    stale = exp.evaluate_instance_mode(fixture, "stale_hint")
    adversarial = exp.evaluate_instance_mode(fixture, "adversarial_hint")

    for row in (stale, adversarial):
        assert row["hint_matches_active_set"] is False
        assert row["solver_authoritative"] is True
        assert row["final_valid"] is True
        assert row["unsafe_false_accept"] is False
        assert row["accepted_without_verification"] is False
        assert row["final_sequence"] == list(fixture.expected_sequence)

    assert stale["hint_decision"] == "rejected"
    assert stale["fallback_used"] is True
    assert adversarial["hint_decision"] == "overwritten"
    assert adversarial["overwrite_used"] is True


def test_req_verify_5406_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5406: artifact exposes required fields and principles."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(result_path=result_path, tests_run=[TEST_COMMAND])

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["fixture_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert artifact["hint_modes"] == list(exp.HINT_MODES)
    assert artifact["active_constraint_precision"] == pytest.approx(1.0)
    assert artifact["active_constraint_recall"] == pytest.approx(1.0)
    assert artifact["solver_conflict_delta"] > 0
    assert artifact["solver_iteration_delta"] > 0
    assert artifact["solver_overwrite_rate"] > 0
    assert artifact["stale_hint_rejection_rate"] == pytest.approx(1.0)
    assert artifact["adversarial_hint_rejection_rate"] == pytest.approx(1.0)
    assert artifact["unsafe_false_accept_rate"] == pytest.approx(0.0)
    assert artifact["active_constraint_warmstart_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == [{"command": TEST_COMMAND, "outcome": "passed"}]
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES


def test_req_verify_5406_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5406: checked-in JSON is stable under deterministic replay."""

    checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(tests_run=checked_in["tests_run"])

    assert checked_in == replay
    assert checked_in["active_constraint_warmstart_ready"] is True
    assert checked_in["unsafe_false_accept_rate"] == pytest.approx(0.0)
    exp.validate_artifact(checked_in)


def test_req_verify_5406_blocks_when_efficiency_or_safety_controls_fail() -> None:
    """REQ-VERIFY-5406: blocked artifacts keep wrong hints non-authoritative."""

    no_savings = exp.build_artifact(
        tests_run=[TEST_COMMAND],
        row_overrides=lambda rows: _erase_candidate_savings(rows),
    )
    unsafe = exp.build_artifact(
        tests_run=[TEST_COMMAND],
        row_overrides=lambda rows: _make_first_adversarial_row_unsafe(rows),
    )

    assert no_savings["active_constraint_warmstart_ready"] is False
    assert no_savings["honest_verdict"].startswith("blocked:")
    assert "candidate_hints_did_not_reduce_solver_work" in no_savings["readiness_blockers"]
    exp.validate_artifact(no_savings)

    assert unsafe["active_constraint_warmstart_ready"] is False
    assert unsafe["unsafe_false_accept_rate"] > 0
    assert "unsafe_false_accepts_present" in unsafe["readiness_blockers"]
    with pytest.raises(ValueError, match="unsafe_false_accept_rate"):
        exp.validate_artifact(unsafe)


def test_req_verify_5406_validation_rejects_schema_and_control_drift() -> None:
    """REQ-VERIFY-5406: validation fails closed on schema and authority drift."""

    artifact = exp.build_artifact(tests_run=[TEST_COMMAND])

    bad_modes = deepcopy(artifact)
    bad_modes["hint_modes"] = ["candidate_hint"]
    with pytest.raises(ValueError, match="hint_modes"):
        exp.validate_artifact(bad_modes)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_ready = deepcopy(artifact)
    bad_ready["active_constraint_warmstart_ready"] = True
    bad_ready["solver_conflict_delta"] = 0
    with pytest.raises(ValueError, match="solver_conflict_delta"):
        exp.validate_artifact(bad_ready)

    bad_stale = deepcopy(artifact)
    bad_stale["stale_hint_rejection_rate"] = 0.5
    with pytest.raises(ValueError, match="stale_hint_rejection_rate"):
        exp.validate_artifact(bad_stale)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)


def _erase_candidate_savings(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    adjusted = deepcopy(rows)
    for row in adjusted:
        if row["hint_mode"] == "candidate_hint":
            baseline = row["baseline_metrics"]
            assert isinstance(baseline, dict)
            row["solver_conflicts"] = baseline["solver_conflicts"]
            row["solver_iterations"] = baseline["solver_iterations"]
    return adjusted


def _make_first_adversarial_row_unsafe(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    adjusted = deepcopy(rows)
    for row in adjusted:
        if row["hint_mode"] == "adversarial_hint":
            row["unsafe_false_accept"] = True
            row["final_valid"] = False
            break
    return adjusted
