"""Tests for Exp5371 CPU p-bit boundary-exchange schedule diagnostic.

Spec refs: REQ-VERIFY-5371, SCENARIO-VERIFY-5371.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5371_pbit_boundary_exchange_schedule_v489 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_verify_5371_spec_declares_boundary_exchange_contract() -> None:
    """REQ-VERIFY-5371: OpenSpec anchors boundary-exchange cadence telemetry."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5371") : spec.index("### REQ-VERIFY-5345")
    ]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5371",
        "SCENARIO-VERIFY-5371",
        str(exp.RESULT_RELATIVE_PATH),
        "monolithic CPU baseline",
        "stale-boundary exchange",
        "frequent-boundary exchange",
        "eta_values",
        "false_accept_count=0",
        "simulation_only=true",
        "hardware_speedup_claim=false",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_verify_5371_builds_boundary_plans_from_exp5359_fixtures() -> None:
    """REQ-VERIFY-5371: boundary plans reuse Exp5359 p-bit schedule fixtures."""

    instances = exp.build_boundary_instances()
    plans = exp.build_boundary_exchange_plans()

    assert len(instances) == exp.EXPECTED_FIXTURE_COUNT
    assert {instance.source_experiment for instance in instances} == {"exp5292", "exp5299"}
    assert all(instance.hardware_execution is False for instance in instances)
    assert exp.ETA_VALUES == (0.25, 0.5, 1.0)
    assert tuple(plan.exchange_mode for plan in plans) == (
        "monolithic_baseline",
        "stale_boundary_exchange",
        "stale_boundary_exchange",
        "frequent_boundary_exchange",
    )
    assert [plan.eta for plan in plans[1:]] == list(exp.ETA_VALUES)
    assert [plan.boundary_exchange_period for plan in plans[1:]] == [4, 2, 1]
    assert plans[0].eta is None


def test_scenario_verify_5371_measures_cadence_against_monolithic_baseline() -> None:
    """SCENARIO-VERIFY-5371: eta cadence rows are compared with monolithic."""

    diagnostic = exp.run_boundary_diagnostic()
    rows = diagnostic["boundary_exchange_results"]

    assert diagnostic["fixture_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert diagnostic["eta_values"] == list(exp.ETA_VALUES)
    assert diagnostic["schedule_modes_measured"] == [
        "monolithic_baseline",
        "stale_boundary_exchange",
        "frequent_boundary_exchange",
    ]
    assert diagnostic["baseline_comparison_present"] is True
    assert diagnostic["timing_ratios_present"] is True
    assert diagnostic["false_accept_count"] == 0
    assert diagnostic["boundary_exchange_schedule_ready"] is True
    assert diagnostic["eta_threshold_estimate"] == 1.0
    assert diagnostic["eta_summaries"]["0.25"]["conflict_delta_vs_monolithic"] < 0
    assert diagnostic["eta_summaries"]["1.0"]["conflict_delta_vs_monolithic"] > 0
    assert diagnostic["convergence_delta_vs_monolithic"] > 0
    assert diagnostic["conflict_delta_vs_monolithic"] > 0
    assert diagnostic["misleading_class_harm_rate"] > 0

    monolithic_rows = [row for row in rows if row["exchange_mode"] == "monolithic_baseline"]
    boundary_rows = [row for row in rows if row["exchange_mode"] != "monolithic_baseline"]
    assert len(monolithic_rows) == exp.EXPECTED_FIXTURE_COUNT
    assert len(boundary_rows) == exp.EXPECTED_FIXTURE_COUNT * len(exp.ETA_VALUES)
    assert all(row["solver_authoritative"] is True for row in rows)
    assert all(row["simulation_only"] is True for row in rows)
    assert all(row["hardware_speedup_claim"] is False for row in rows)
    assert all(row["false_accept"] is False for row in rows)
    assert all(isinstance(row["boundary_variables"], list) for row in rows)
    assert all(row["boundary_exchange_period"] >= 1 for row in boundary_rows)
    assert any(row["stale_boundary_reads"] > 0 for row in boundary_rows)
    assert any(
        row["energy_monotonicity_violations"] > 0
        for row in rows
        if row["eta"] == 0.25
    )


def test_req_verify_5371_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5371: artifact exposes required cadence fields."""

    tests_run = [{"command": "unit exp5371", "outcome": "passed"}]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["boundary_exchange_schedule_ready"] is True
    assert artifact["simulation_only"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["fixture_count"] == exp.EXPECTED_FIXTURE_COUNT
    assert artifact["eta_values"] == list(exp.ETA_VALUES)
    assert artifact["eta_threshold_estimate"] == 1.0
    assert artifact["convergence_delta_vs_monolithic"] > 0
    assert artifact["conflict_delta_vs_monolithic"] > 0
    assert artifact["energy_monotonicity_violation_count"] > 0
    assert artifact["false_accept_count"] == 0
    assert artifact["tests_run"] == tests_run
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert "no hardware speedup claim" in artifact["honest_verdict"]


def test_req_verify_5371_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5371: checked-in result is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(tests_run=result["tests_run"])

    assert result == replay
    assert result["status"] == "complete"
    assert result["boundary_exchange_schedule_ready"] is True
    assert result["simulation_only"] is True
    assert result["hardware_speedup_claim"] is False
    assert result["false_accept_count"] == 0
    exp.validate_artifact(result)


def test_req_verify_5371_validation_rejects_unsafe_or_unmeasured_drift() -> None:
    """REQ-VERIFY-5371: validation fails closed on safety and schema drift."""

    artifact = exp.build_artifact(
        tests_run=[{"command": "unit exp5371", "outcome": "passed"}]
    )

    bad_status = deepcopy(artifact)
    bad_status["status"] = "done"
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_status)

    bad_simulation = deepcopy(artifact)
    bad_simulation["simulation_only"] = False
    with pytest.raises(ValueError, match="simulation_only"):
        exp.validate_artifact(bad_simulation)

    bad_hardware = deepcopy(artifact)
    bad_hardware["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        exp.validate_artifact(bad_hardware)

    bad_eta = deepcopy(artifact)
    bad_eta["eta_values"] = [1.0]
    with pytest.raises(ValueError, match="eta_values"):
        exp.validate_artifact(bad_eta)

    bad_accept = deepcopy(artifact)
    bad_accept["false_accept_count"] = 1
    with pytest.raises(ValueError, match="false_accept_count"):
        exp.validate_artifact(bad_accept)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    bad_tests["boundary_exchange_schedule_ready"] = True
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)
