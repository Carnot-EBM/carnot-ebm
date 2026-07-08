"""Tests for Exp5384 p-bit boundary/overwrite joint diagnostic.

Spec refs: REQ-VERIFY-5384, SCENARIO-VERIFY-5384.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5384_pbit_boundary_overwrite_joint_diagnostic_v490 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_verify_5384_spec_declares_joint_diagnostic_contract() -> None:
    """REQ-VERIFY-5384: OpenSpec anchors the joint boundary/overwrite contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5384") : spec.index("### REQ-VERIFY-5371")]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5384",
        "SCENARIO-VERIFY-5384",
        str(exp.RESULT_RELATIVE_PATH),
        "shared fixture count",
        "monolithic p-bit",
        "boundary-exchange",
        "overwrite-guided",
        "pbit_boundary_overwrite_ready",
        "unsafe_false_accepts=0",
        "simulation_only=true",
        "hardware_speedup_claim=false",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_verify_5384_reuses_only_shared_pbit_and_overwrite_fixtures() -> None:
    """REQ-VERIFY-5384: joint fixtures are the p-bit/overwrite intersection."""

    sources = exp.load_joint_sources()
    links = exp.build_shared_fixture_links(sources)

    assert sources["source_readiness"] == {
        "exp5371_boundary_exchange_schedule_ready": True,
        "exp5383_overwrite_guidance_scale_ready": True,
    }
    assert str(exp.exp5371.RESULT_RELATIVE_PATH) in sources["source_artifacts"]
    assert str(exp.exp5383.RESULT_RELATIVE_PATH) in sources["source_artifacts"]
    assert sources["shared_fixture_ids"] == [
        "exp5292_aligned_factor_sat",
        "exp5292_misleading_factor_sat",
        "exp5292_neutral_factor_sat",
    ]
    assert len(links) == exp.EXPECTED_SHARED_FIXTURE_COUNT
    assert all(link.fixture_id.startswith("exp5292_") for link in links)
    assert all(len(link.boundary_rows) == len(exp.ETA_VALUES) for link in links)
    assert all(len(link.overwrite_rows) == len(exp.HINT_CLASS_NAMES) for link in links)


def test_scenario_verify_5384_joins_boundary_eta_with_overwrite_safety() -> None:
    """SCENARIO-VERIFY-5384: boundary deltas are paired with overwrite safety."""

    diagnostic = exp.run_joint_diagnostic()
    rows = diagnostic["joint_results"]

    assert diagnostic["fixture_count"] == exp.EXPECTED_SHARED_FIXTURE_COUNT
    assert diagnostic["eta_values"] == list(exp.ETA_VALUES)
    assert diagnostic["eta_threshold_estimate"] == 1.0
    assert diagnostic["solver_overwrite_enabled"] is True
    assert diagnostic["post_projection_validity_rate"] == pytest.approx(1.0)
    assert diagnostic["fallback_completeness_rate"] == pytest.approx(1.0)
    assert diagnostic["unsafe_false_accepts"] == 0
    assert diagnostic["pbit_boundary_overwrite_ready"] is True
    assert diagnostic["conflict_delta_vs_monolithic"] > 0
    assert diagnostic["convergence_delta_vs_monolithic"] > 0
    assert diagnostic["comparison_variants"] == [
        "monolithic_pbit",
        "boundary_exchange",
        "overwrite_guided",
    ]
    assert diagnostic["eta_summaries"]["0.25"]["conflict_delta_vs_monolithic"] < 0
    assert diagnostic["eta_summaries"]["1.0"]["conflict_delta_vs_monolithic"] > 0

    assert len(rows) == (
        exp.EXPECTED_SHARED_FIXTURE_COUNT * len(exp.ETA_VALUES) * len(exp.HINT_CLASS_NAMES)
    )
    assert {row["eta"] for row in rows} == set(exp.ETA_VALUES)
    assert {row["hint_class"] for row in rows} == set(exp.HINT_CLASS_NAMES)
    assert all(row["variant"] == "overwrite_guided" for row in rows)
    assert all(row["solver_overwrite_enabled"] is True for row in rows)
    assert all(row["projection_valid"] is True for row in rows)
    assert all(row["fallback_complete"] is True for row in rows)
    assert all(row["unsafe_false_accept"] is False for row in rows)
    assert all(row["simulation_only"] is True for row in rows)
    assert all(row["hardware_speedup_claim"] is False for row in rows)


def test_req_verify_5384_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5384: artifact exposes required joint diagnostic fields."""

    tests_run = [{"command": "unit exp5384", "outcome": "passed"}]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["pbit_boundary_overwrite_ready"] is True
    assert artifact["simulation_only"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["fixture_count"] == exp.EXPECTED_SHARED_FIXTURE_COUNT
    assert artifact["eta_values"] == list(exp.ETA_VALUES)
    assert artifact["eta_threshold_estimate"] == 1.0
    assert artifact["solver_overwrite_enabled"] is True
    assert artifact["conflict_delta_vs_monolithic"] > 0
    assert artifact["convergence_delta_vs_monolithic"] > 0
    assert artifact["post_projection_validity_rate"] == pytest.approx(1.0)
    assert artifact["fallback_completeness_rate"] == pytest.approx(1.0)
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == tests_run
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert "no hardware speedup claim" in artifact["honest_verdict"]


def test_req_verify_5384_blocked_without_recorded_tests() -> None:
    """REQ-VERIFY-5384: missing tests produce honest_blocked, not readiness."""

    artifact = exp.build_artifact(tests_run=[])

    assert artifact["status"] == "honest_blocked"
    assert artifact["pbit_boundary_overwrite_ready"] is False
    assert artifact["simulation_only"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert "tests_not_recorded" in artifact["readiness_blockers"]
    assert artifact["honest_verdict"] == "blocked_pbit_boundary_overwrite_joint_not_ready"
    exp.validate_artifact(artifact)


def test_req_verify_5384_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5384: checked-in result is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(tests_run=result["tests_run"])

    assert result == replay
    assert result["status"] == "complete"
    assert result["pbit_boundary_overwrite_ready"] is True
    assert result["simulation_only"] is True
    assert result["hardware_speedup_claim"] is False
    assert result["unsafe_false_accepts"] == 0
    exp.validate_artifact(result)


def test_req_verify_5384_validation_rejects_unsafe_or_unmeasured_drift() -> None:
    """REQ-VERIFY-5384: validation fails closed on safety and schema drift."""

    artifact = exp.build_artifact(tests_run=[{"command": "unit exp5384", "outcome": "passed"}])

    bad_status = deepcopy(artifact)
    bad_status["status"] = "done"
    with pytest.raises(ValueError, match="status"):
        exp.validate_artifact(bad_status)

    bad_ready = deepcopy(artifact)
    bad_ready["pbit_boundary_overwrite_ready"] = "yes"
    with pytest.raises(ValueError, match="pbit_boundary_overwrite_ready"):
        exp.validate_artifact(bad_ready)

    bad_simulation = deepcopy(artifact)
    bad_simulation["simulation_only"] = False
    with pytest.raises(ValueError, match="simulation_only"):
        exp.validate_artifact(bad_simulation)

    bad_hardware = deepcopy(artifact)
    bad_hardware["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        exp.validate_artifact(bad_hardware)

    bad_accept = deepcopy(artifact)
    bad_accept["unsafe_false_accepts"] = 1
    with pytest.raises(ValueError, match="unsafe_false_accepts"):
        exp.validate_artifact(bad_accept)

    bad_validity = deepcopy(artifact)
    bad_validity["post_projection_validity_rate"] = 0.99
    with pytest.raises(ValueError, match="post_projection_validity_rate"):
        exp.validate_artifact(bad_validity)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = []
    bad_tests["pbit_boundary_overwrite_ready"] = True
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)

    assert (
        exp.eta_threshold_from_summaries(
            {
                "0.25": {
                    "conflict_delta_vs_monolithic": -1,
                    "convergence_delta_vs_monolithic": 1,
                    "unsafe_false_accepts": 0,
                }
            }
        )
        is None
    )
