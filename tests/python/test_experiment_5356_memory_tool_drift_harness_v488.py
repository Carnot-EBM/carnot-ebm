"""Tests for Exp5356 memory-induced verifier/tool drift harness.

Spec refs: REQ-LEARN-5356, SCENARIO-LEARN-5356-CLEAN,
SCENARIO-LEARN-5356-DRIFT, SCENARIO-LEARN-5356-DEFLECT,
SCENARIO-LEARN-5356-CONTROLS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5356_memory_tool_drift_harness_v488 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_learn_5356_spec_declares_memory_tool_drift_contract() -> None:
    """REQ-LEARN-5356: OpenSpec anchors fields, variants, and scenarios."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5356") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5356",
        "SCENARIO-LEARN-5356-CLEAN",
        "SCENARIO-LEARN-5356-DRIFT",
        "SCENARIO-LEARN-5356-DEFLECT",
        "SCENARIO-LEARN-5356-CONTROLS",
        str(exp.RESULT_RELATIVE_PATH),
        "biased, stale, poisoned, irrelevant, counterfactual, no-op",
        "shuffled-control",
        "rollback variants SHALL restore the clean verifier",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_scenario_learn_5356_clean_memory_selects_correct_tools() -> None:
    """SCENARIO-LEARN-5356-CLEAN: clean memory selects declared choices."""

    tasks = exp.build_drift_tasks()
    clean = exp.evaluate_clean_selection(tasks)

    assert len(tasks) == exp.TASK_COUNT
    assert clean["clean_selection_accuracy"] == 1.0
    assert {row["task_id"] for row in clean["clean_rows"]} == {
        task.task_id for task in tasks
    }
    assert all(row["selected"] == row["expected_clean_selection"] for row in clean["clean_rows"])
    assert all(row["answer_correct"] is True for row in clean["clean_rows"])


def test_scenario_learn_5356_memory_variants_induce_and_measure_drift() -> None:
    """SCENARIO-LEARN-5356-DRIFT: biased/stale/poisoned rows drift."""

    tasks = exp.build_drift_tasks()
    cases = exp.build_memory_cases(tasks)
    audit = exp.evaluate_memory_tool_drift(tasks, cases)

    assert {case.memory_variant for case in cases} == set(exp.MEMORY_VARIANTS)
    assert audit["drift_case_count"] == len(cases)
    assert audit["induced_tool_drift_rate"] == 1.0
    assert audit["susceptible_parameter_count"] == len(audit["susceptible_parameters"])
    assert audit["susceptible_parameter_count"] >= 4

    drift_rows = [
        row
        for row in audit["case_rows"]
        if row["memory_variant"] in exp.DRIFT_INDUCING_VARIANTS
    ]
    assert drift_rows
    assert all(row["raw_drifted_from_clean"] for row in drift_rows)
    assert all(row["raw_answer_correct"] is False for row in drift_rows)
    assert all(
        {
            "verifier",
            "tool",
            "action",
            "parameters",
        }.issubset(row["raw_selected"])
        for row in drift_rows
    )


def test_scenario_learn_5356_counterfactual_rollback_and_controls_gate_ready() -> None:
    """SCENARIO-LEARN-5356-DEFLECT/CONTROLS: safety lanes restore clean."""

    artifact = exp.build_result_artifact(
        tests_run=[{"command": "unit memory tool drift", "outcome": "passed"}]
    )
    counterfactual_rows = [
        row
        for row in artifact["memory_case_rows"]
        if row["memory_variant"] == "counterfactual"
    ]
    poisoned_rows = [
        row for row in artifact["memory_case_rows"] if row["memory_variant"] == "poisoned"
    ]
    control_rows = [
        row
        for row in artifact["memory_case_rows"]
        if row["memory_variant"] in exp.CONTROL_VARIANTS
    ]

    assert artifact["counterfactual_memory_deflection_rate"] == 1.0
    assert artifact["rollback_recovery_rate"] == 1.0
    assert artifact["no_op_control_delta"] == 0.0
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["memory_tool_drift_ready"] is True
    assert all(row["counterfactual_deflected"] for row in counterfactual_rows)
    assert all(row["accepted_memory"] is False for row in poisoned_rows)
    assert all(row["guarded_selected"] == row["expected_clean_selection"] for row in poisoned_rows)
    assert all(not row["raw_drifted_from_clean"] for row in control_rows)
    assert all(row["rollback_restored_clean"] for row in artifact["rollback_rows"])


def test_req_learn_5356_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5356-7: run() writes the required terminal artifact."""

    tests_run = [
        {"command": "unit deterministic harness", "outcome": "passed"},
        {"command": "unit rollback checks", "outcome": "passed"},
    ]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == exp.MILESTONE
    assert artifact["status"]["value"] == "memory_tool_drift_ready"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert artifact["continuous_self_learning_target"] is True
    assert artifact["no_weight_mutation"] is True
    assert artifact["drift_case_count"] == len(exp.build_memory_cases(exp.build_drift_tasks()))
    assert artifact["clean_selection_accuracy"] == 1.0
    assert artifact["induced_tool_drift_rate"] > 0.0
    assert artifact["susceptible_parameter_count"] > 0
    assert artifact["counterfactual_memory_deflection_rate"] == 1.0
    assert artifact["rollback_recovery_rate"] == 1.0
    assert artifact["no_op_control_delta"] == 0.0
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["memory_tool_drift_ready"] is True
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5356_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5356: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_result_artifact(tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert result["memory_tool_drift_ready"] is True
    assert result["continuous_self_learning_target"] is True
    assert result["no_weight_mutation"] is True
    assert result["unsafe_false_accepts"] == 0
    exp.validate_artifact(result)


def test_req_learn_5356_blocked_when_tests_not_recorded() -> None:
    """REQ-LEARN-5356-7: ready gate stays false without test records."""

    artifact = exp.build_result_artifact(tests_run=[])

    assert artifact["status"]["value"] == "blocked_memory_tool_drift_gate"
    assert artifact["honest_verdict"]["value"].startswith(
        "blocked_memory_tool_drift_not_ready:"
    )
    assert "tests_not_recorded" in artifact["honest_verdict"]["value"]
    assert artifact["memory_tool_drift_ready"] is False
    assert artifact["unsafe_false_accepts"] == 0
    exp.validate_artifact(artifact)


def test_req_learn_5356_validation_rejects_schema_drift() -> None:
    """REQ-LEARN-5356-7: artifact validation rejects scalar and gate drift."""

    artifact = exp.build_result_artifact(
        tests_run=[{"command": "unit memory drift", "outcome": "passed"}]
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "deterministic_context_memory"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_learning = deepcopy(artifact)
    bad_learning["continuous_self_learning_target"] = {"value": True}
    with pytest.raises(ValueError, match="continuous_self_learning_target"):
        exp.validate_artifact(bad_learning)

    bad_weight = deepcopy(artifact)
    bad_weight["no_weight_mutation"] = False
    with pytest.raises(ValueError, match="no_weight_mutation"):
        exp.validate_artifact(bad_weight)

    bad_count = deepcopy(artifact)
    bad_count["drift_case_count"] = True
    with pytest.raises(ValueError, match="drift_case_count"):
        exp.validate_artifact(bad_count)

    bad_numeric = deepcopy(artifact)
    bad_numeric["induced_tool_drift_rate"] = {"value": 1.0}
    with pytest.raises(ValueError, match="induced_tool_drift_rate"):
        exp.validate_artifact(bad_numeric)

    bad_unsafe = deepcopy(artifact)
    bad_unsafe["unsafe_false_accepts"] = 1
    with pytest.raises(ValueError, match="unsafe_false_accepts"):
        exp.validate_artifact(bad_unsafe)

    bad_ready = deepcopy(artifact)
    bad_ready["memory_tool_drift_ready"] = "yes"
    with pytest.raises(ValueError, match="memory_tool_drift_ready"):
        exp.validate_artifact(bad_ready)

    bad_missing_tests = deepcopy(artifact)
    bad_missing_tests["tests_run"]["value"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_missing_tests)
