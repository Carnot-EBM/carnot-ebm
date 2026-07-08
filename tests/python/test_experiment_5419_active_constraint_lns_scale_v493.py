"""Tests for Exp5419 active-constraint LNS scale-up diagnostic.

Spec refs: REQ-VERIFY-5419, SCENARIO-VERIFY-5419.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5419_active_constraint_lns_scale_v493 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5419_active_constraint_lns_scale_v493.py "
    "-q --no-cov"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5419_active_constraint_lns_scale_v493.py "
    "-m pytest tests/python/test_experiment_5419_active_constraint_lns_scale_v493.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report "
    "--include=python/carnot/experiment_5419_active_constraint_lns_scale_v493.py "
    "--fail-under=100"
)


def test_req_verify_5419_spec_declares_lns_scale_contract() -> None:
    """REQ-VERIFY-5419: OpenSpec anchors the LNS scale-up artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5419") : spec.index("### REQ-VERIFY-5407")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5419",
        "SCENARIO-VERIFY-5419",
        str(exp.RESULT_RELATIVE_PATH),
        "Exp 5406 active-constraint",
        "LNS-style subproblem selection hints",
        "stale, contradictory, overconfident",
        "constraint violations",
        "dual residual",
        "deterministic_solver_experiment",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_req_verify_5419_fixtures_scale_beyond_exp5406() -> None:
    """REQ-VERIFY-5419: fixtures are larger and expose LNS subproblem hints."""

    fixtures = exp.build_scale_fixtures()

    assert len(fixtures) == exp.EXPECTED_FIXTURE_COUNT
    assert len(fixtures) > exp.EXP5406_FIXTURE_COUNT
    assert min(len(fixture.actions) for fixture in fixtures) >= exp.MIN_ACTION_COUNT
    assert all(fixture.active_constraint_ids for fixture in fixtures)
    assert all(fixture.conflict_front for fixture in fixtures)
    assert all(fixture.lns_subproblem_hint for fixture in fixtures)
    assert len({fixture.fixture_id for fixture in fixtures}) == len(fixtures)


def test_scenario_verify_5419_modes_preserve_authority_and_reduce_work() -> None:
    """SCENARIO-VERIFY-5419: accepted LNS hints reduce work under solver authority."""

    diagnostic = exp.run_diagnostic()
    rows = diagnostic["row_records"]
    by_mode = {mode: [row for row in rows if row["hint_mode"] == mode] for mode in exp.HINT_MODES}

    assert diagnostic["fixture_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert diagnostic["baseline_solver_work"] > diagnostic["guided_solver_work"]
    assert diagnostic["work_delta"] == (
        diagnostic["baseline_solver_work"] - diagnostic["guided_solver_work"]
    )
    assert diagnostic["accepted_hint_count"] == len(by_mode["lns_guided_hint"])
    assert diagnostic["rejected_hint_count"] >= len(by_mode["stale_hint"])
    assert diagnostic["overwritten_hint_count"] >= len(by_mode["contradictory_hint"])
    assert diagnostic["lns_subproblem_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert diagnostic["dual_residual_sanity"] is True
    assert diagnostic["solver_validity_preserved"] is True
    assert diagnostic["aggregate_from_rows_only"] is True
    assert diagnostic["active_constraint_lns_scale_ready"] is True
    assert all(row["solver_authoritative"] is True for row in rows)
    assert all(row["accepted_without_verification"] is False for row in rows)
    assert all(row["final_valid"] is True for row in rows)
    assert all(row["objective_preserved"] is True for row in rows)
    assert all(row["constraint_violation_count"] == 0 for row in rows)
    assert all(row["dual_residual_norm"] == pytest.approx(0.0) for row in rows)
    assert all(row["hint_decision"] == "accepted" for row in by_mode["lns_guided_hint"])
    assert all(row["hint_decision"] == "rejected" for row in by_mode["stale_hint"])
    assert all(row["hint_decision"] == "overwritten" for row in by_mode["contradictory_hint"])
    assert all(row["hint_decision"] == "rejected" for row in by_mode["overconfident_hint"])
    assert all(
        row["guided_solver_work"] < row["baseline_solver_work"]
        for row in by_mode["lns_guided_hint"]
    )


def test_scenario_verify_5419_invalid_hints_cannot_change_final_authority() -> None:
    """SCENARIO-VERIFY-5419: stale, contradictory, and overconfident hints fail closed."""

    fixture = exp.build_scale_fixtures()[0]
    baseline = exp.evaluate_fixture_mode(fixture, "solver_only")
    invalid_rows = [
        exp.evaluate_fixture_mode(fixture, "stale_hint"),
        exp.evaluate_fixture_mode(fixture, "contradictory_hint"),
        exp.evaluate_fixture_mode(fixture, "overconfident_hint"),
    ]

    for row in invalid_rows:
        assert row["hint_matches_solver_view"] is False
        assert row["hint_decision"] in {"rejected", "overwritten"}
        assert row["fallback_used"] is True
        assert row["solver_authoritative"] is True
        assert row["accepted_without_verification"] is False
        assert row["unsafe_false_accept"] is False
        assert row["final_valid"] is True
        assert row["objective_preserved"] is True
        assert row["final_sequence"] == baseline["final_sequence"]
        assert row["guided_solver_work"] == row["baseline_solver_work"]


def test_req_verify_5419_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5419: artifact exposes required fields and principles."""

    tests_run = [
        {"command": TEST_COMMAND, "outcome": "passed"},
        {"command": COVERAGE_COMMAND, "outcome": "passed"},
        {"command": COVERAGE_REPORT_COMMAND, "outcome": "passed"},
    ]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["fixture_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert artifact["baseline_solver_work"] > artifact["guided_solver_work"]
    assert artifact["work_delta"] > 0
    assert artifact["accepted_hint_count"] > 0
    assert artifact["rejected_hint_count"] > 0
    assert artifact["overwritten_hint_count"] > 0
    assert artifact["lns_subproblem_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert artifact["dual_residual_sanity"] is True
    assert artifact["solver_validity_preserved"] is True
    assert artifact["aggregate_from_rows_only"] is True
    assert artifact["active_constraint_lns_scale_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == tests_run


def test_req_verify_5419_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5419: checked-in JSON is stable under deterministic replay."""

    checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(tests_run=checked_in["tests_run"])

    assert checked_in == replay
    assert checked_in["active_constraint_lns_scale_ready"] is True
    assert checked_in["aggregate_from_rows_only"] is True
    exp.validate_artifact(checked_in)


def test_req_verify_5419_validation_rejects_drift_and_unsafe_accepts() -> None:
    """REQ-VERIFY-5419: validation fails closed on schema and authority drift."""

    artifact = exp.build_artifact(tests_run=[{"command": TEST_COMMAND, "outcome": "passed"}])

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_rows = deepcopy(artifact)
    bad_rows["row_records"][0]["baseline_solver_work"] += 1
    with pytest.raises(ValueError, match="aggregate_from_rows_only"):
        exp.validate_artifact(bad_rows)

    bad_validity = deepcopy(artifact)
    bad_validity["row_records"][0]["final_valid"] = False
    with pytest.raises(ValueError, match="solver_validity_preserved"):
        exp.validate_artifact(bad_validity)

    bad_residual = deepcopy(artifact)
    bad_residual["row_records"][0]["dual_residual_norm"] = 1.0
    with pytest.raises(ValueError, match="dual_residual_sanity"):
        exp.validate_artifact(bad_residual)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)


def test_req_verify_5419_blockers_are_row_derived() -> None:
    """REQ-VERIFY-5419: blocked artifacts identify failed row-level gates."""

    artifact = exp.build_artifact(
        tests_run=[{"command": TEST_COMMAND, "outcome": "passed"}],
        row_overrides=lambda rows: _erase_guided_savings(rows),
    )

    assert artifact["active_constraint_lns_scale_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert "guided_work_not_reduced" in artifact["readiness_blockers"]
    exp.validate_artifact(artifact)


def test_req_verify_5419_defensive_branches_keep_schema_fail_closed() -> None:
    """REQ-VERIFY-5419: helper guards cover malformed hints and blocked summaries."""

    fixture = exp.build_scale_fixtures()[0]
    fallback = tuple(fixture.expected_sequence)
    summary = exp.run_diagnostic()

    no_tests = exp.build_artifact()
    assert no_tests["active_constraint_lns_scale_ready"] is False
    assert "tests_not_recorded" in no_tests["readiness_blockers"]
    assert no_tests["tests_run"] == []
    exp.validate_artifact(no_tests)

    string_test = exp.build_artifact(tests_run=[TEST_COMMAND])
    assert string_test["tests_run"] == [{"command": TEST_COMMAND, "outcome": "passed"}]

    assert exp._is_complete_valid_sequence(fixture, fixture.expected_sequence[:-1]) is False
    assert (
        exp._is_complete_valid_sequence(
            fixture,
            tuple(reversed(fixture.expected_sequence)),
        )
        is False
    )
    assert exp._hint_structurally_valid(fixture, ("bad-edge",), (), ()) is False
    assert exp._hint_structurally_valid(fixture, ("missing->deploy",), (), ()) is False
    assert exp._hint_structurally_valid(fixture, ("deploy->deploy",), (), ()) is False
    assert (
        exp._sequence_from_hint_or_baseline(
            fixture,
            fixture.active_constraint_ids,
            fallback=fallback,
        )
        == fallback
    )
    assert exp._sequence_from_hint_or_baseline(fixture, (), fallback=fallback) == fallback
    assert exp._constraint_violation_count(fixture, fallback[:-1]) == len(fixture.precedence) + 1
    assert exp._dual_residual_norm(fixture, fallback[:-1]) == pytest.approx(
        float(len(fixture.precedence) + 1)
    )
    assert exp._objective_cost(fixture, fallback[:-1]) == 10_000

    broken = deepcopy(summary)
    broken["fixture_count"] = 1
    broken["baseline_solver_work"] = 1
    broken["guided_solver_work"] = 2
    broken["accepted_hint_count"] = 0
    broken["rejected_hint_count"] = 0
    broken["overwritten_hint_count"] = 0
    broken["lns_subproblem_count"] = 0
    broken["dual_residual_sanity"] = False
    broken["solver_validity_preserved"] = False
    broken["aggregate_from_rows_only"] = False
    assert exp._readiness_blockers(broken) == [
        "fixture_count_mismatch",
        "fixture_scale_not_larger_than_exp5406",
        "guided_work_not_reduced",
        "no_accepted_hints",
        "no_rejected_hints",
        "no_overwritten_hints",
        "lns_subproblem_count_mismatch",
        "dual_residual_sanity_failed",
        "solver_validity_not_preserved",
        "aggregate_not_row_derived",
    ]


def _erase_guided_savings(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    adjusted = deepcopy(rows)
    for row in adjusted:
        if row["hint_mode"] == "lns_guided_hint":
            row["guided_solver_work"] = row["baseline_solver_work"]
            row["solver_conflicts"] = row["baseline_metrics"]["solver_conflicts"]
            row["solver_iterations"] = row["baseline_metrics"]["solver_iterations"]
    return adjusted
