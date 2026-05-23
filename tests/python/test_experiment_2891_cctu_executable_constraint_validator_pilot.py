"""Tests for Exp 2891 CCTU executable constraint validator pilot.

Spec: REQ-VERIFY-2891, SCENARIO-VERIFY-2891.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import cctu_executable_constraint_validator_pilot as exp


def test_req_verify_2891_builds_local_cases_covering_cctu_categories() -> None:
    """REQ-VERIFY-2891: pilot cases cover resource, behavior, toolset, and response."""

    cases = exp.build_local_pilot_cases()
    categories = {case.category for case in cases}
    rows = [exp.validate_pilot_case(case) for case in cases]

    assert len(cases) == 5
    assert {"resource", "behavior", "toolset", "response"} <= categories
    assert len({case.case_id for case in cases}) == len(cases)
    assert all(not row["overall_passed"] for row in rows)

    by_category = {row["category"]: row for row in rows if row["category"] != "response_verifier"}
    assert by_category["resource"]["violations"][0]["category"] == "resource"
    assert by_category["toolset"]["violations"][0]["localized_to"] == "tool_call"
    assert by_category["behavior"]["violations"][0]["localized_to"] == "tool_result"
    assert by_category["response"]["violations"][0]["localized_to"] == "final_answer"

    for row in rows:
        assert row["input_checksum"].startswith("sha256:")
        assert row["validation_checksum"].startswith("sha256:")
        assert set(exp.STEP_IDS) == {step["step_id"] for step in row["step_results"]}
        assert row["executable_validation_used"] is True
        assert row["source_module"] == exp.CCTU_SOURCE_MODULE


def test_req_verify_2891_artifact_schema_and_claim_boundaries(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-2891: artifact writes required fields without benchmark claims."""

    output_path = tmp_path / exp.OUTPUT_FILENAME
    artifact = exp.write_experiment_artifact(
        exp.ExperimentConfig(
            output_path=output_path,
            started_at=10.0,
            clock=lambda: 12.5,
            tests_run=("focused pytest",),
        )
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert persisted == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["cctu_validator_ready"] is True
    assert artifact["n_cases"] == 5
    assert artifact["constraint_categories"] == [
        "behavior",
        "resource",
        "response",
        "response_verifier",
        "toolset",
    ]
    assert artifact["category_coverage"]["resource"] == {"passed": 0, "total": 1}
    assert artifact["category_coverage"]["response_verifier"] == {"passed": 0, "total": 1}
    assert artifact["executable_validation_used"] is True
    assert artifact["live_llm_called"] is False
    assert artifact["headline_metric_claim_made"] is False
    assert artifact["run_date"] == "20260523"
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["tests_run"] == ["focused pytest"]
    assert artifact["unsupported_categories"]["multi_turn_state"]["supported"] is False
    assert artifact["field_principles"]["headline_metric_claim_made"]
    assert all(row["violations"] or row["overall_passed"] for row in artifact["validation_rows"])


def test_req_verify_2891_checksum_and_coverage_helpers_are_deterministic() -> None:
    """REQ-VERIFY-2891: checksums and category coverage are stable JSON-derived values."""

    case = exp.build_local_pilot_cases()[0]
    first = exp.validate_pilot_case(case)
    second = exp.validate_pilot_case(case)

    assert first["input_checksum"] == second["input_checksum"]
    assert first["validation_checksum"] == second["validation_checksum"]
    assert exp.category_coverage([first, second]) == {case.category: {"passed": 0, "total": 2}}
    assert exp.category_coverage([{"category": "gold", "overall_passed": True}]) == {
        "gold": {"passed": 1, "total": 1}
    }
    with pytest.raises(ValueError, match="unknown source CCTU case"):
        exp._source_case_by_id("missing-case")
