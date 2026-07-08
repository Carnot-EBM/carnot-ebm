"""Tests for Exp5448 active-constraint p-bit sparsity bridge.

Spec refs: REQ-VERIFY-5448, SCENARIO-VERIFY-5448.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5448_active_constraint_pbit_sparsity_bridge_v495 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5448_active_constraint_pbit_sparsity_bridge_v495.py "
    "-q --no-cov"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5448_active_constraint_pbit_sparsity_bridge_v495.py "
    "-m pytest tests/python/test_experiment_5448_active_constraint_pbit_sparsity_bridge_v495.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report "
    "--include=python/carnot/experiment_5448_active_constraint_pbit_sparsity_bridge_v495.py "
    "--fail-under=100"
)
FULL_SUITE_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
E2E_COMMAND = (
    "ops/e2e-test-plan.md review: Exp5448 is a deterministic solver fixture; "
    "no live Ising training, PyO3 round-trip, KV260, or CSL e2e path applies"
)


def test_req_verify_5448_spec_declares_bridge_contract() -> None:
    """REQ-VERIFY-5448: OpenSpec anchors the sparsity bridge artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5448") : spec.index("### REQ-VERIFY-5433")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5448",
        "SCENARIO-VERIFY-5448",
        str(exp.RESULT_RELATIVE_PATH),
        "bounded SAT, CSP, and LNS fixtures",
        "temporary assumptions",
        "reject or overwrite",
        "unrestricted exact solve",
        "density before restoration",
        "density after restoration",
        "hardware_speedup_claim",
        "deterministic_solver_pbit_assumption_fixture",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in normalized


def test_req_verify_5448_fixtures_cover_sat_csp_lns_and_density() -> None:
    """REQ-VERIFY-5448: fixtures cover bounded families and restored sparsity."""

    fixtures = exp.build_bridge_fixtures()
    density = exp.density_before_after(fixtures)

    assert len(fixtures) == exp.EXPECTED_FIXTURE_COUNT
    assert exp.constraint_family_counts(fixtures) == {"csp": 1, "lns": 1, "sat": 2}
    assert {fixture.source_module for fixture in fixtures} >= {
        "experiment_5407_pbit_qubo_active_constraint_stress_v492",
        "experiment_5433_active_constraint_diversity_lns_v494",
    }
    assert density["mean_before"] > density["mean_after"]
    assert density["mean_restored_sparsity_delta"] > 0
    assert set(density["by_fixture"]) == {fixture.fixture_id for fixture in fixtures}
    for fixture in fixtures:
        row = density["by_fixture"][fixture.fixture_id]
        assert 0.0 <= row["after"] <= row["before"] <= 1.0
        assert row["possible_edges"] > 0
        assert row["before_edges"] >= row["after_edges"]


def test_scenario_verify_5448_rows_keep_solver_authoritative() -> None:
    """SCENARIO-VERIFY-5448: all assumption rows preserve exact outcomes."""

    diagnostic = exp.run_diagnostic()
    rows = diagnostic["row_records"]
    by_source = {
        source: [row for row in rows if row["assumption_source"] == source]
        for source in exp.ASSUMPTION_SOURCES
    }

    assert diagnostic["fixture_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert diagnostic["constraint_family_counts"] == {"csp": 1, "lns": 1, "sat": 2}
    assert diagnostic["assumption_source_counts"] == {
        "active_constraint": exp.EXPECTED_FIXTURE_COUNT,
        "pbit_consensus": exp.EXPECTED_FIXTURE_COUNT,
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
    assert all(row["assumption_decision"] == "accepted" for row in by_source["active_constraint"])
    assert diagnostic["fallback_completeness_rate"] == pytest.approx(1.0)
    assert diagnostic["rejected_assumption_count"] > 0
    assert diagnostic["overwritten_assumption_count"] > 0
    assert diagnostic["unsafe_false_accepts"] == 0
    assert diagnostic["solver_work_delta"] > 0
    assert diagnostic["pbit_assumption_bridge_ready"] is True


def test_scenario_verify_5448_wrong_assumptions_cannot_create_false_labels() -> None:
    """SCENARIO-VERIFY-5448: bad p-bit assumptions are rescued or overwritten."""

    fixtures = {fixture.fixture_id: fixture for fixture in exp.build_bridge_fixtures()}

    rejected = exp.evaluate_fixture_source(fixtures["sat_false_basin_rescue"], "pbit_consensus")
    assert rejected["assumption_decision"] == "rejected"
    assert rejected["fallback_used"] is True
    assert rejected["assumption_attempt_status"] == "unsat"
    assert rejected["baseline_status"] == "sat"
    assert rejected["final_status"] == rejected["baseline_status"]
    assert rejected["final_matches_exact"] is True
    assert rejected["unsafe_false_accept"] is False

    overwritten = exp.evaluate_fixture_source(
        fixtures["lns_valid_but_suboptimal"], "pbit_consensus"
    )
    assert overwritten["assumption_decision"] == "overwritten"
    assert overwritten["fallback_used"] is True
    assert overwritten["assumption_attempt_status"] == "sat"
    assert overwritten["solution_valid"] is True
    assert overwritten["final_solution"] == overwritten["baseline_solution"]
    assert overwritten["assumption_solution"] != overwritten["baseline_solution"]
    assert overwritten["unsafe_false_accept"] is False


def test_req_verify_5448_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5448: artifact exposes required fields and claim boundaries."""

    tests_run = [
        {"command": TEST_COMMAND, "outcome": "passed"},
        {"command": COVERAGE_COMMAND, "outcome": "passed"},
        {"command": COVERAGE_REPORT_COMMAND, "outcome": "passed"},
        {"command": FULL_SUITE_COMMAND, "outcome": "passed"},
        {"command": SPEC_COVERAGE_COMMAND, "outcome": "passed"},
        {"command": E2E_COMMAND, "outcome": "not_applicable"},
    ]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["fixture_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert artifact["solver_authoritative"] is True
    assert artifact["fallback_completeness_rate"] == pytest.approx(1.0)
    assert artifact["rejected_assumption_count"] > 0
    assert artifact["overwritten_assumption_count"] > 0
    assert artifact["solver_work_delta"] > 0
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["pbit_assumption_bridge_ready"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == tests_run


def test_req_verify_5448_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5448: checked-in JSON is stable under deterministic replay."""

    checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(tests_run=checked_in["tests_run"])

    assert checked_in == replay
    assert checked_in["pbit_assumption_bridge_ready"] is True
    assert checked_in["hardware_speedup_claim"] is False
    exp.validate_artifact(checked_in)


def test_req_verify_5448_validation_rejects_unsafe_schema_drift() -> None:
    """REQ-VERIFY-5448: validation fails closed on unsafe aggregate drift."""

    artifact = exp.build_artifact(tests_run=[{"command": TEST_COMMAND, "outcome": "passed"}])
    exp.validate_artifact(artifact)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "hardware_sampler"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_hardware = deepcopy(artifact)
    bad_hardware["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        exp.validate_artifact(bad_hardware)

    bad_unsafe = deepcopy(artifact)
    bad_unsafe["unsafe_false_accepts"] = 1
    with pytest.raises(ValueError, match="unsafe_false_accepts"):
        exp.validate_artifact(bad_unsafe)

    bad_density = deepcopy(artifact)
    first_fixture = next(iter(bad_density["density_before_after"]["by_fixture"]))
    bad_density["density_before_after"]["by_fixture"][first_fixture]["after"] = 1.1
    with pytest.raises(ValueError, match="density_before_after"):
        exp.validate_artifact(bad_density)

    bad_authority = deepcopy(artifact)
    bad_authority["row_records"][0]["final_matches_exact"] = False
    with pytest.raises(ValueError, match="fallback_completeness_rate"):
        exp.validate_artifact(bad_authority)


def test_req_verify_5448_blockers_and_helper_guards(tmp_path: Path) -> None:
    """REQ-VERIFY-5448: blockers expose missing utility and malformed inputs."""

    fixture = exp.build_bridge_fixtures()[0]
    assert exp.run_diagnostic(row_overrides=lambda rows: rows)["row_count"] == (
        exp.EXPECTED_FIXTURE_COUNT * len(exp.ASSUMPTION_SOURCES)
    )

    no_tests = exp.build_artifact()
    assert no_tests["pbit_assumption_bridge_ready"] is False
    assert "tests_not_recorded" in no_tests["readiness_blockers"]
    exp.validate_artifact(no_tests)

    string_test = exp.build_artifact(tests_run=[TEST_COMMAND])
    assert string_test["tests_run"] == [{"command": TEST_COMMAND, "outcome": "passed"}]

    with pytest.raises(ValueError, match="assumption_source"):
        exp.evaluate_fixture_source(fixture, "bogus")

    bad_summary = deepcopy(exp.run_diagnostic())
    bad_summary["fixture_count"] = 1
    bad_summary["constraint_family_counts"] = {"sat": 1}
    bad_summary["assumption_source_counts"] = {"active_constraint": 1}
    bad_summary["density_before_after"] = {"mean_before": 0.1, "mean_after": 0.2}
    bad_summary["solver_authoritative"] = False
    bad_summary["fallback_completeness_rate"] = 0.5
    bad_summary["rejected_assumption_count"] = 0
    bad_summary["overwritten_assumption_count"] = 0
    bad_summary["solver_work_delta"] = -1
    bad_summary["unsafe_false_accepts"] = 1
    assert exp.readiness_blockers(bad_summary) == [
        "fixture_count_mismatch",
        "constraint_family_missing",
        "assumption_source_coverage_mismatch",
        "density_restoration_not_measured",
        "solver_not_authoritative",
        "fallback_completeness_incomplete",
        "no_rejected_assumptions",
        "no_overwritten_assumptions",
        "solver_work_not_reduced",
        "unsafe_false_accepts_present",
    ]

    unsat_metrics = exp.SolveMetrics(
        status="unsat",
        solution=None,
        objective_value=None,
        conflicts=0,
        propagations=0,
        iterations=0,
    )
    assert exp._solution_valid_for_final(fixture, unsat_metrics) is False
    with pytest.raises(ValueError, match="best_score"):
        exp._require_int(None, "best_score")
    assert exp._rate(1, 0) == 0.0

    cli_path = tmp_path / exp.RESULT_RELATIVE_PATH
    assert exp._main(["--output", str(cli_path), "--test-run", TEST_COMMAND]) == 0
    cli_artifact = json.loads(cli_path.read_text(encoding="utf-8"))
    assert cli_artifact["tests_run"] == [{"command": TEST_COMMAND, "outcome": "passed"}]
