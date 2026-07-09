"""Tests for Exp5462 active-constraint minimal-core p-bit/p-dit bridge.

Spec refs: REQ-VERIFY-5462, SCENARIO-VERIFY-5462.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5462_active_constraint_minimal_core_pdit_bridge_v496 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5462_active_constraint_minimal_core_pdit_bridge_v496.py "
    "-q --no-cov"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5462_active_constraint_minimal_core_pdit_bridge_v496.py "
    "-m pytest "
    "tests/python/test_experiment_5462_active_constraint_minimal_core_pdit_bridge_v496.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report "
    "--include=python/carnot/experiment_5462_active_constraint_minimal_core_pdit_bridge_v496.py "
    "--fail-under=100"
)
FULL_SUITE_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
E2E_COMMAND = (
    "ops/e2e-test-plan.md review: Exp5462 is a deterministic solver fixture; "
    "no live Ising training, PyO3 round-trip, KV260, or CSL e2e path applies"
)


def test_req_verify_5462_spec_declares_minimal_core_pdit_contract() -> None:
    """REQ-VERIFY-5462: OpenSpec anchors the V496 bridge contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5462") : spec.index("### REQ-VERIFY-5433")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5462",
        "SCENARIO-VERIFY-5462",
        str(mod.RESULT_RELATIVE_PATH),
        "assignment/QAP-style p-dit",
        "minimal core",
        "p-bit binary",
        "p-dit multi-state",
        "unrestricted exact solve",
        "hardware timing-ratio receipt",
        "scripts/research_conductor.py",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_req_verify_5462_fixtures_cover_sat_csp_lns_assignment_and_pdit() -> None:
    """REQ-VERIFY-5462: fixtures cover exact SAT/CSP/LNS and p-dit assignment."""

    fixtures = mod.build_bridge_fixtures()
    density = mod.density_before_after(fixtures)

    assert len(fixtures) == mod.EXPECTED_FIXTURE_COUNT
    assert mod.constraint_family_counts(fixtures) == {
        "assignment": 1,
        "csp": 1,
        "lns": 1,
        "sat": 2,
    }
    assert {fixture.source_module for fixture in fixtures} >= {
        "experiment_5407_pbit_qubo_active_constraint_stress_v492",
        "experiment_5433_active_constraint_diversity_lns_v494",
        "pdit_assignment_fixture",
    }
    assert mod.pdit_variable_count(fixtures) >= 3
    assert all(fixture.pdit_control_names for fixture in fixtures)
    assert all(fixture.pdit_samples for fixture in fixtures)
    assert density["mean_before"] > density["mean_after"]
    assert density["mean_restored_sparsity_delta"] > 0
    assert set(density["by_fixture"]) == {fixture.fixture_id for fixture in fixtures}

    assignment = next(fixture for fixture in fixtures if fixture.constraint_family == "assignment")
    baseline = mod.solve_exact(assignment, ())
    assert baseline.status == "sat"
    assert baseline.solution == assignment.expected_solution
    assert baseline.objective_value == 0
    assert len(assignment.assignment_domain) == len(assignment.variables)


def test_scenario_verify_5462_rows_keep_solver_authoritative() -> None:
    """SCENARIO-VERIFY-5462: advisory rows preserve unrestricted exact outcomes."""

    diagnostic = mod.run_diagnostic()
    rows = diagnostic["row_records"]
    by_source = {
        source: [row for row in rows if row["assumption_source"] == source]
        for source in mod.ASSUMPTION_SOURCES
    }

    assert diagnostic["fixture_count"] == mod.EXPECTED_FIXTURE_COUNT
    assert diagnostic["assumption_source_counts"] == {
        "active_constraint": mod.EXPECTED_FIXTURE_COUNT,
        "pbit_binary": mod.EXPECTED_FIXTURE_COUNT,
        "pdit_multistate": mod.EXPECTED_FIXTURE_COUNT,
    }
    assert all(by_source.values())
    assert all(row["solver_authoritative"] is True for row in rows)
    assert all(row["accepted_without_verification"] is False for row in rows)
    assert all(row["final_matches_exact"] is True for row in rows)
    assert all(row["solution_valid"] is True for row in rows)
    assert all(row["unsafe_false_accept"] is False for row in rows)
    assert all(row["hardware_speedup_claim"] is False for row in rows)
    assert any(row["assumption_decision"] == "rejected" for row in rows)
    assert any(row["assumption_decision"] == "overwritten" for row in rows)
    assert any(row["fallback_used"] is True for row in rows)
    assert any(row["minimal_core_ids"] for row in rows)
    assert all(row["assumption_decision"] == "accepted" for row in by_source["active_constraint"])
    assert any(row["constraint_family"] == "assignment" for row in by_source["pdit_multistate"])
    assert diagnostic["pdit_variable_count"] >= 3
    assert diagnostic["minimal_core_count"] >= 3
    assert diagnostic["fallback_completeness_rate"] == pytest.approx(1.0)
    assert diagnostic["rejected_assumption_count"] > 0
    assert diagnostic["overwritten_assumption_count"] > 0
    assert diagnostic["unsafe_false_accepts"] == 0
    assert diagnostic["solver_work_delta"] > 0
    assert diagnostic["minimal_core_pbit_bridge_ready"] is True


def test_scenario_verify_5462_wrong_assumptions_are_rescued_and_diagnosed() -> None:
    """SCENARIO-VERIFY-5462: bad p-bit and p-dit assumptions cannot create false claims."""

    fixtures = {fixture.fixture_id: fixture for fixture in mod.build_bridge_fixtures()}

    rejected = mod.evaluate_fixture_source(fixtures["sat_false_basin_rescue"], "pbit_binary")
    assert rejected["assumption_decision"] == "rejected"
    assert rejected["fallback_used"] is True
    assert rejected["assumption_attempt_status"] == "unsat"
    assert rejected["baseline_status"] == "sat"
    assert rejected["final_status"] == rejected["baseline_status"]
    assert rejected["final_matches_exact"] is True
    assert rejected["unsafe_false_accept"] is False
    assert rejected["minimal_core_assumptions"] == ["!x1"]
    assert all(
        item["without_core_assumption_matches_exact"] for item in rejected["minimal_core_evidence"]
    )

    overwritten = mod.evaluate_fixture_source(
        fixtures["assignment_pdit_tradeoff"], "pdit_multistate"
    )
    assert overwritten["assumption_decision"] == "overwritten"
    assert overwritten["fallback_used"] is True
    assert overwritten["assumption_attempt_status"] == "sat"
    assert overwritten["solution_valid"] is True
    assert overwritten["final_solution"] == overwritten["baseline_solution"]
    assert overwritten["assumption_solution"] != overwritten["baseline_solution"]
    assert overwritten["minimal_core_assumptions"] == ["ana=test"]
    assert overwritten["unsafe_false_accept"] is False


def test_req_verify_5462_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5462: artifact exposes required fields and claim boundaries."""

    tests_run = [
        {"command": TEST_COMMAND, "outcome": "passed"},
        {"command": COVERAGE_COMMAND, "outcome": "passed"},
        {"command": COVERAGE_REPORT_COMMAND, "outcome": "passed"},
        {"command": FULL_SUITE_COMMAND, "outcome": "passed"},
        {"command": SPEC_COVERAGE_COMMAND, "outcome": "passed"},
        {"command": E2E_COMMAND, "outcome": "not_applicable"},
    ]
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["fixture_count"] == mod.EXPECTED_FIXTURE_COUNT
    assert artifact["solver_authoritative"] is True
    assert artifact["fallback_completeness_rate"] == pytest.approx(1.0)
    assert artifact["pdit_variable_count"] >= 3
    assert artifact["minimal_core_count"] >= 3
    assert artifact["rejected_assumption_count"] > 0
    assert artifact["solver_work_delta"] > 0
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["minimal_core_pbit_bridge_ready"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == tests_run
    assert artifact["research_conductor_modified"] is False


def test_req_verify_5462_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5462: checked-in JSON is stable under deterministic replay."""

    checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = mod.build_artifact(tests_run=checked_in["tests_run"])

    assert checked_in == replay
    assert checked_in["minimal_core_pbit_bridge_ready"] is True
    assert checked_in["hardware_speedup_claim"] is False
    mod.validate_artifact(checked_in)


def test_req_verify_5462_validation_rejects_unsafe_schema_drift() -> None:
    """REQ-VERIFY-5462: validation fails closed on authority and schema drift."""

    artifact = mod.build_artifact(tests_run=[{"command": TEST_COMMAND, "outcome": "passed"}])
    mod.validate_artifact(artifact)

    missing = deepcopy(artifact)
    missing.pop("minimal_core_count")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "hardware_sampler"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_hardware = deepcopy(artifact)
    bad_hardware["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(bad_hardware)

    bad_unsafe = deepcopy(artifact)
    bad_unsafe["unsafe_false_accepts"] = 1
    with pytest.raises(ValueError, match="unsafe_false_accepts"):
        mod.validate_artifact(bad_unsafe)

    bad_pdit = deepcopy(artifact)
    bad_pdit["pdit_variable_count"] = 0
    with pytest.raises(ValueError, match="pdit_variable_count"):
        mod.validate_artifact(bad_pdit)

    bad_core_count = deepcopy(artifact)
    bad_core_count["minimal_core_count"] = 0
    with pytest.raises(ValueError, match="minimal_core_count"):
        mod.validate_artifact(bad_core_count)

    bad_density = deepcopy(artifact)
    first_fixture = next(iter(bad_density["density_before_after"]["by_fixture"]))
    bad_density["density_before_after"]["by_fixture"][first_fixture]["after"] = 1.1
    with pytest.raises(ValueError, match="density_before_after"):
        mod.validate_artifact(bad_density)

    bad_authority = deepcopy(artifact)
    bad_authority["row_records"][0]["final_matches_exact"] = False
    with pytest.raises(ValueError, match="fallback_completeness_rate"):
        mod.validate_artifact(bad_authority)

    bad_core = deepcopy(artifact)
    bad_core["row_records"][0]["minimal_core_ids"] = ["unrelated:core"]
    with pytest.raises(ValueError, match="minimal_core"):
        mod.validate_artifact(bad_core)

    bad_conductor = deepcopy(artifact)
    bad_conductor["research_conductor_modified"] = True
    with pytest.raises(ValueError, match="research_conductor.py"):
        mod.validate_artifact(bad_conductor)


def test_req_verify_5462_blockers_and_helper_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5462: blockers expose missing utility and malformed inputs."""

    fixture = mod.build_bridge_fixtures()[0]
    assert mod.run_diagnostic(row_overrides=lambda rows: rows)["row_count"] == (
        mod.EXPECTED_FIXTURE_COUNT * len(mod.ASSUMPTION_SOURCES)
    )

    no_tests = mod.build_artifact()
    assert no_tests["minimal_core_pbit_bridge_ready"] is False
    assert "tests_not_recorded" in no_tests["readiness_blockers"]
    mod.validate_artifact(no_tests)

    string_test = mod.build_artifact(tests_run=[TEST_COMMAND])
    assert string_test["tests_run"] == [{"command": TEST_COMMAND, "outcome": "passed"}]

    with pytest.raises(ValueError, match="assumption_source"):
        mod.evaluate_fixture_source(fixture, "bogus")

    assignment = next(
        item
        for item in mod.build_bridge_fixtures()
        if item.fixture_id == "assignment_pdit_tradeoff"
    )
    assert mod.minimal_core_for_assumptions(assignment, ("ana=pack",)) == ((), (), [])
    skipped, _, skipped_evidence = mod.minimal_core_for_assumptions(
        assignment,
        ("cy=ship", "ana=test"),
    )
    assert skipped == ("ana=test",)
    assert skipped_evidence[0]["without_core_assumption_matches_exact"] is True
    assert mod._pdit_value_to_assumption(fixture, "x1", "unknown") == ""

    monkeypatch.setattr(mod, "_attempt_disagrees", lambda _candidate, _baseline: True)
    forced_core, forced_ids, forced_evidence = mod.minimal_core_for_assumptions(
        fixture,
        (),
    )
    assert forced_core == ()
    assert forced_ids == ()
    assert forced_evidence == []

    bad_summary = deepcopy(mod.run_diagnostic())
    bad_summary["fixture_count"] = 1
    bad_summary["constraint_family_counts"] = {"sat": 1}
    bad_summary["assumption_source_counts"] = {"active_constraint": 1}
    bad_summary["pdit_variable_count"] = 0
    bad_summary["minimal_core_count"] = 0
    bad_summary["density_before_after"] = {"mean_before": 0.1, "mean_after": 0.2}
    bad_summary["solver_authoritative"] = False
    bad_summary["fallback_completeness_rate"] = 0.5
    bad_summary["rejected_assumption_count"] = 0
    bad_summary["solver_work_delta"] = -1
    bad_summary["unsafe_false_accepts"] = 1
    assert mod.readiness_blockers(bad_summary) == [
        "fixture_count_mismatch",
        "constraint_family_missing",
        "assumption_source_coverage_mismatch",
        "pdit_variables_missing",
        "minimal_cores_missing",
        "density_restoration_not_measured",
        "solver_not_authoritative",
        "fallback_completeness_incomplete",
        "no_rejected_assumptions",
        "solver_work_not_reduced",
        "unsafe_false_accepts_present",
    ]

    unsat_metrics = mod.SolveMetrics(
        status="unsat",
        solution=None,
        objective_value=None,
        conflicts=0,
        propagations=0,
        iterations=0,
    )
    assert mod._solution_valid_for_final(fixture, unsat_metrics) is False
    with pytest.raises(ValueError, match="best_score"):
        mod._require_int(None, "best_score")
    assert mod._rate(1, 0) == 0.0

    cli_path = tmp_path / mod.RESULT_RELATIVE_PATH
    assert mod._main(["--output", str(cli_path), "--test-run", TEST_COMMAND]) == 0
    cli_artifact = json.loads(cli_path.read_text(encoding="utf-8"))
    assert cli_artifact["tests_run"] == [{"command": TEST_COMMAND, "outcome": "passed"}]
