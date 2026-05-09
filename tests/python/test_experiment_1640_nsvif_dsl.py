"""Tests for Exp 1640 NSVIF-style instruction-to-constraint DSL workflow.

Spec: REQ-VERIFY-1640, SCENARIO-VERIFY-1640.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_1640_nsvif_dsl as mod


def test_req_verify_1640_parses_nl_into_carnot_constraint_rows() -> None:
    """REQ-VERIFY-1640: NL instructions become schema-valid Carnot constraints."""

    case = mod.default_instruction_cases()[0]
    parsed = mod.parse_instruction_to_carnot_constraints(str(case["instruction"]))

    assert parsed["parser_success"] is True
    assert parsed["dsl_pack"]["schema_version"] == mod.dsl.DSL_SCHEMA_VERSION
    assert [constraint["metadata"]["nsvif_operator"] for constraint in parsed["constraints"]] == [
        "json_object",
        "json_has_keys",
        "contains",
        "not_contains",
        "max_words",
    ]
    assert parsed["constraints"][0]["constraint_type"] == "instruction_constraint"
    assert parsed["constraints"][0]["metadata"]["dsl_schema_version"] == mod.dsl.DSL_SCHEMA_VERSION
    assert parsed["constraints"][1]["metadata"]["value"] == ["answer", "confidence"]
    json.dumps(parsed["constraints"])


def test_req_verify_1640_compiles_python_validator_from_same_constraints() -> None:
    """REQ-VERIFY-1640: compiled Python validators accept good and reject bad rows."""

    case = mod.default_instruction_cases()[0]
    result = mod.evaluate_instruction_case(case)

    assert result["parser_success"] is True
    assert result["validator_compiled"] is True
    assert result["known_good"]["accepted"] is True
    assert result["known_bad"]["accepted"] is False
    assert {"c002-json_has_keys", "c003-contains", "c004-not_contains"} <= set(
        result["known_bad"]["failure_ids"]
    )
    assert result["constraint_count"] == 5
    assert result["carnot_constraint_count"] == 5


def test_req_verify_1640_fail_closed_for_unsupported_instruction() -> None:
    """REQ-VERIFY-1640: unsupported instructions do not report parser success."""

    case = {
        "case_id": "unsupported",
        "instruction": "Answer carefully.",
        "known_good": "careful answer",
        "known_bad": "bad answer",
    }

    result = mod.evaluate_instruction_case(case)
    artifact = mod.build_artifact(cases=[case], tests_run=[])

    assert result["parser_success"] is False
    assert result["validator_compiled"] is False
    assert "no supported constraints" in result["error"]
    assert artifact["status"] == "partial"
    assert artifact["parser_success"] is False
    assert artifact["validators_compiled"] == 0


def test_scenario_verify_1640_builds_complete_parser_success_artifact() -> None:
    """SCENARIO-VERIFY-1640: artifact records parser_success and validator metrics."""

    artifact = mod.build_artifact(
        cases=mod.default_instruction_cases(),
        tests_run=[".venv/bin/pytest tests/python/test_experiment_1640_nsvif_dsl.py -q"],
    )

    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["spec_traces"] == ["REQ-VERIFY-1640", "SCENARIO-VERIFY-1640"]
    assert artifact["parser_success"] is True
    assert artifact["instructions_tested"] == 4
    assert artifact["constraints_extracted"] == artifact["carnot_constraints_emitted"]
    assert artifact["validators_compiled"] == 4
    assert artifact["known_good_pass_rate"] == pytest.approx(1.0)
    assert artifact["known_bad_reject_rate"] == pytest.approx(1.0)
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["arbitrary_code_execution_path_introduced"] is False
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_1640_run_experiment_writes_json_deliverable(tmp_path: Path) -> None:
    """REQ-VERIFY-1640: run_experiment writes the required JSON deliverable."""

    output_path = tmp_path / "results" / "experiment_1640_nsvif_dsl.json"

    artifact = mod.run_experiment(
        output_path=output_path,
        tests_run=["focused"],
        cases=mod.default_instruction_cases(),
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["artifact_path"] == str(output_path)
    assert artifact["tests_run"] == ["focused"]


def test_req_verify_1640_artifact_validation_rejects_inconsistent_shapes() -> None:
    """REQ-VERIFY-1640: artifact validation catches missing and inconsistent fields."""

    artifact = mod.build_artifact(cases=mod.default_instruction_cases(), tests_run=[])

    missing = dict(artifact)
    del missing["parser_success"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)

    with pytest.raises(AssertionError, match="parser_success"):
        mod.validate_artifact(dict(artifact, parser_success=False))

    with pytest.raises(AssertionError, match="false_accept_rate"):
        mod.validate_artifact(dict(artifact, false_accept_rate=1.0))
