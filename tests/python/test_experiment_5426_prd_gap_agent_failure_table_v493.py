"""Tests for Exp5426 .493 PRD gap and agent-failure table.

Spec refs: REQ-REPORT-5426, SCENARIO-REPORT-5426,
SCENARIO-REPORT-5426-MISSING-UPSTREAM.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5426_prd_gap_agent_failure_table_v493 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def _lane_names(rows: list[dict[str, object]]) -> list[str]:
    return [str(row["lane"]) for row in rows]


def _support_map(row: dict[str, object]) -> dict[tuple[str, str], object]:
    return {
        (str(field["artifact_path"]), str(field["field_name"])): field["value"]
        for field in row["supporting_fields"]  # type: ignore[index]
    }


def test_req_report_5426_spec_declares_gap_table_contract() -> None:
    """REQ-REPORT-5426: OpenSpec anchors the .493 gap table."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5426") :]

    for marker in (
        "REQ-REPORT-5426",
        "SCENARIO-REPORT-5426",
        "SCENARIO-REPORT-5426-MISSING-UPSTREAM",
        str(exp.RESULT_RELATIVE_PATH),
        "structured verification",
        "continuous self-learning",
        "solver guidance",
        "ARC live progress",
        "hardware",
        "certificates",
        "`research-roadmap.yaml`",
        "`scripts/research_conductor.py`",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f'principle "{principle}"' in section


def test_scenario_report_5426_available_artifacts_emit_capstone_ready_table() -> None:
    """SCENARIO-REPORT-5426: actual .493 artifacts classify every PRD lane."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5426", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["upstream_artifacts_read"] == [str(path) for path in exp.EXPECTED_ARTIFACTS]
    assert artifact["upstream_artifacts_missing"] == []
    assert artifact["prd_gap_table_ready"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["failure_taxonomy_counts"] == {
        "tool-use": 1,
        "planning": 3,
        "reasoning": 3,
        "measurement-access": 2,
        "calibration": 1,
        "live-environment": 2,
    }

    assert _lane_names(artifact["closed_lanes"]) == [  # type: ignore[arg-type]
        "structured_verification",
        "continuous_self_learning",
    ]
    assert _lane_names(artifact["partial_lanes"]) == [  # type: ignore[arg-type]
        "solver_guidance",
        "hardware",
        "certificates",
    ]
    assert _lane_names(artifact["blocked_lanes"]) == ["arc_live_progress"]  # type: ignore[arg-type]
    assert artifact["missing_lanes"] == []

    structured = artifact["closed_lanes"][0]
    structured_support = _support_map(structured)  # type: ignore[arg-type]
    assert structured_support[
        (
            "results/experiment_5417_risk_calibrated_sota_structured_panel_v493.json",
            "risk_calibrated_structured_panel_ready",
        )
    ] is True
    assert structured_support[
        (
            "results/experiment_5418_predictive_prefix_action_safety_v493.json",
            "prefix_gated_unreachable_tool_action_rate",
        )
    ] == 0.0
    assert structured["failure_taxonomy"] == ["tool-use", "calibration"]

    csl = artifact["closed_lanes"][1]
    csl_support = _support_map(csl)  # type: ignore[arg-type]
    assert csl_support[
        (
            "results/experiment_5421_evidence_reliance_csl_v493.json",
            "hidden_forgetting_detected",
        )
    ] is True
    assert csl_support[
        (
            "results/experiment_5422_csl_promotion_reliance_scale_v493.json",
            "promoted_fragment_count",
        )
    ] == 3

    solver = artifact["partial_lanes"][0]
    solver_support = _support_map(solver)  # type: ignore[arg-type]
    assert solver_support[
        (
            "results/experiment_5419_active_constraint_lns_scale_v493.json",
            "active_constraint_lns_scale_ready",
        )
    ] is True
    assert solver_support[
        ("results/experiment_5419_active_constraint_lns_scale_v493.json", "work_delta")
    ] == 234
    assert solver["classification_reason"] == "bounded_solver_guidance"

    arc = artifact["blocked_lanes"][0]
    arc_support = _support_map(arc)  # type: ignore[arg-type]
    assert arc_support[
        ("results/experiment_5423_arc_coex_landmark_levelup_v493.json", "status")
    ] == "honest_null"
    assert arc_support[
        ("results/experiment_5423_arc_coex_landmark_levelup_v493.json", "arc_new_level_banked")
    ] is False
    assert arc["classification_reason"] == "honest_null_no_new_level_banked"

    hardware = artifact["partial_lanes"][1]
    hardware_support = _support_map(hardware)  # type: ignore[arg-type]
    assert hardware_support[
        (
            "results/experiment_5424_hardware_comparable_timing_receipts_v493.json",
            "hardware_speedup_claim",
        )
    ] is False
    assert hardware_support[
        (
            "results/experiment_5424_hardware_comparable_timing_receipts_v493.json",
            "measurement_access_complete",
        )
    ] is True

    certificates = artifact["partial_lanes"][2]
    cert_support = _support_map(certificates)  # type: ignore[arg-type]
    assert cert_support[
        (
            "results/experiment_5425_kan_measurement_access_certificate_v493.json",
            "missing_evidence_detected",
        )
    ] is True
    assert cert_support[
        (
            "results/experiment_5425_kan_measurement_access_certificate_v493.json",
            "broad_kan_verification_claim",
        )
    ] is False

    transition_fields = {
        field["field_name"]: field["value"] for field in artifact["transition_context"]["supporting_fields"]
    }
    assert transition_fields["next_task_range"] == "exp5415-exp5427"
    assert transition_fields["honest_verdict"].startswith("complete:")


def test_req_report_5426_rows_preserve_exact_present_field_names() -> None:
    """REQ-REPORT-5426: present lane evidence names real artifact fields."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5426", "outcome": "passed"}],
    )
    rows = (
        artifact["closed_lanes"]
        + artifact["partial_lanes"]
        + artifact["blocked_lanes"]
        + artifact["missing_lanes"]
    )

    assert artifact["lane_order"] == list(exp.LANE_NAMES)
    for row in rows:
        assert set(exp.REQUIRED_LANE_FIELDS) <= set(row)
        assert row["lane"] in exp.LANE_NAMES
        assert row["classification"] in exp.LANE_CLASSIFICATIONS
        assert isinstance(row["artifact_paths"], list)
        assert isinstance(row["supporting_fields"], list)
        assert isinstance(row["missing_supporting_fields"], list)
        assert isinstance(row["claim_boundary"], str)
        assert isinstance(row["prd_refs"], list)
        assert isinstance(row["research_program_priorities"], list)
        assert set(row["failure_taxonomy"]) <= set(exp.FAILURE_TAXONOMY)
        if row["classification"] != "missing":
            assert row["supporting_fields"]
            assert row["missing_supporting_fields"] == []
            for field in row["supporting_fields"]:
                assert field["present"] is True
                assert field["artifact_path"] in row["artifact_paths"]
                assert field["field_name"]


def test_scenario_report_5426_missing_inputs_stay_missing(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5426-MISSING-UPSTREAM: absent inputs are not fabricated."""

    artifact = exp.build_artifact(
        root=tmp_path,
        tests_run=[{"command": "unit exp5426 missing", "outcome": "passed"}],
    )

    exp.validate_artifact(artifact)
    assert artifact["status"] == "blocked_missing_upstream"
    assert artifact["upstream_artifacts_read"] == []
    assert artifact["upstream_artifacts_missing"] == [str(path) for path in exp.EXPECTED_ARTIFACTS]
    assert artifact["closed_lanes"] == []
    assert artifact["partial_lanes"] == []
    assert artifact["blocked_lanes"] == []
    assert _lane_names(artifact["missing_lanes"]) == list(exp.LANE_NAMES)  # type: ignore[arg-type]
    assert artifact["prd_gap_table_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["failure_taxonomy_counts"] == {name: 0 for name in exp.FAILURE_TAXONOMY}

    for row in artifact["missing_lanes"]:
        assert row["classification"] == "missing"
        assert row["supporting_fields"] == []
        assert row["classification_reason"] == "missing_upstream_artifact"


def test_req_report_5426_run_writes_stable_json(tmp_path: Path) -> None:
    """REQ-REPORT-5426: run() writes the required deterministic artifact."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    tests_run = exp.default_tests_run()
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["spec_refs"] == list(exp.SPEC_REFS)
    assert artifact["tests_run"] == tests_run
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    exp.validate_artifact(artifact)


def test_req_report_5426_repository_artifact_matches_replay() -> None:
    """REQ-REPORT-5426: checked-in result is stable under deterministic replay."""

    checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(root=REPO, tests_run=checked_in["tests_run"])

    assert checked_in == replay
    assert checked_in["prd_gap_table_ready"] is True
    exp.validate_artifact(checked_in)


def test_req_report_5426_validation_rejects_schema_and_claim_drift() -> None:
    """REQ-REPORT-5426: validation fails closed on schema or claim drift."""

    artifact = exp.build_artifact(
        root=REPO,
        tests_run=[{"command": "unit exp5426", "outcome": "passed"}],
    )

    missing_required = deepcopy(artifact)
    missing_required.pop("upstream_artifacts_read")
    with pytest.raises(ValueError, match="upstream_artifacts_read"):
        exp.validate_artifact(missing_required)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "finished without terminal prefix"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_ready = deepcopy(artifact)
    bad_ready["prd_gap_table_ready"] = False
    with pytest.raises(ValueError, match="prd_gap_table_ready"):
        exp.validate_artifact(bad_ready)

    bad_counts = deepcopy(artifact)
    bad_counts["failure_taxonomy_counts"]["planning"] = -1
    with pytest.raises(ValueError, match="failure_taxonomy_counts"):
        exp.validate_artifact(bad_counts)

    bad_lane = deepcopy(artifact)
    bad_lane["closed_lanes"][0]["classification"] = "partial"
    with pytest.raises(ValueError, match="lane buckets"):
        exp.validate_artifact(bad_lane)

    bad_support = deepcopy(artifact)
    bad_support["closed_lanes"][0]["supporting_fields"][0]["present"] = False
    with pytest.raises(ValueError, match="supporting_fields"):
        exp.validate_artifact(bad_support)
