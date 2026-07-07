"""Tests for Exp5355 dependency-edge provenance self-learning.

Spec refs: REQ-LEARN-5355, SCENARIO-LEARN-5355-GRAPH,
SCENARIO-LEARN-5355-FAULTS, SCENARIO-LEARN-5355-METRICS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5355_dependency_provenance_self_learning_v488 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_learn_5355_spec_declares_dependency_provenance_contract() -> None:
    """REQ-LEARN-5355: OpenSpec anchors fields, faults, and scenarios."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5355") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5355",
        "SCENARIO-LEARN-5355-GRAPH",
        "SCENARIO-LEARN-5355-FAULTS",
        "SCENARIO-LEARN-5355-METRICS",
        str(exp.RESULT_RELATIVE_PATH),
        "positive, stale, poisoned, missing-edge, and cyclic-dependency",
        "Execution feedback SHALL be stored separately from memory hygiene",
        "Exp5342 only as excluded quarantined context",
        exp.INFERENCE_SUBSTRATE,
    ):
        assert marker in section

    for field, principle in exp.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_learn_5355_source_gate_reuses_only_clean_v487_fixtures() -> None:
    """REQ-LEARN-5355-1: Exp5340/5341 are reused; Exp5342 is quarantined."""

    gate = exp.confirm_source_fixture_readiness(root=REPO)

    assert gate["all_passed"] is True
    assert gate["utility_memory_ready"] is True
    assert gate["compressor_drift_fixture_ready"] is True
    assert gate["no_weight_mutation"] is True
    assert gate["quarantined_scaleup_reused"] is False
    assert gate["failed_gates"] == []
    assert str(exp.EXP5342_QUARANTINED_RELATIVE_PATH) in gate["excluded_artifacts"]


def test_scenario_learn_5355_graph_edges_reconstruct_decisions() -> None:
    """SCENARIO-LEARN-5355-GRAPH: accepted graph is replayable and acyclic."""

    cases = exp.build_dependency_cases()
    audit = exp.evaluate_dependency_provenance(cases)
    case_types = {case.case_type for case in cases}

    assert case_types == {"positive", "stale", "poisoned", "missing_edge", "cyclic"}
    assert audit["dependency_edge_count"] == sum(len(case.expected_edges) for case in cases)
    assert audit["graph_integrity"]["accepted_graph_acyclic"] is True
    assert audit["graph_integrity"]["final_expected_edges_present"] is True
    assert audit["graph_integrity"]["faulty_cases_quarantined"] is True
    assert audit["graph_integrity"]["graph_integrity_holds"] is True
    assert audit["point_in_time_reconstruction_rate"] == 1.0

    final_relations = {edge.relation for edge in audit["final_edges"]}
    assert {
        "context_informs_retrieval",
        "retrieval_routes_verifier",
        "retrieval_routes_tool",
        "verifier_affects_outcome",
        "tool_affects_outcome",
        "outcome_records_execution_feedback",
        "outcome_updates_memory_hygiene",
        "outcome_triggers_rollback",
    }.issubset(final_relations)


def test_scenario_learn_5355_missing_edges_and_cycles_are_quarantined() -> None:
    """SCENARIO-LEARN-5355-FAULTS: audit faults do not enter final state."""

    audit = exp.evaluate_dependency_provenance(exp.build_dependency_cases())
    missing = audit["graph_integrity"]["missing_edge_findings"]
    cycles = audit["graph_integrity"]["cycle_findings"]

    assert audit["dependency_edge_recall"] < 1.0
    assert audit["dependency_edge_precision"] < audit["dependency_edge_recall"]
    assert len(missing) == 1
    assert missing[0]["case_id"] == "dep-missing-edge"
    assert missing[0]["missing_edge"]["relation"] == "outcome_records_execution_feedback"
    assert cycles and cycles[0]["case_id"] == "dep-cyclic-dependency"
    assert audit["graph_integrity"]["spurious_edge_count"] == 2
    assert all(
        not case["accepted_into_final_graph"]
        for case in audit["fault_cases"]
        if case["case_type"] in {"missing_edge", "cyclic"}
    )


def test_scenario_learn_5355_metrics_keep_feedback_hygiene_and_efficiency_distinct() -> None:
    """SCENARIO-LEARN-5355-METRICS: aggregate fields are non-tautological."""

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit dependency graph", "outcome": "passed"}],
    )
    metric_values = {
        field: artifact[field] for field in exp.AGGREGATE_METRIC_FIELDS
    }

    assert artifact["execution_feedback_attribution_rate"] == 0.8
    assert artifact["memory_hygiene_delta"] > 0.0
    assert artifact["context_efficiency_delta"] > 0.0
    assert len(set(metric_values.values())) == len(metric_values)
    assert artifact["duplicated_metric_pairs"] == []
    assert artifact["dependency_provenance_ready"] is True
    assert all(
        "memory_hygiene" not in row
        for row in artifact["execution_feedback_rows"]
    )
    assert all(
        "execution_status" not in row
        for row in artifact["memory_hygiene_rows"]
    )


def test_req_learn_5355_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-LEARN-5355-6: run() writes the required terminal artifact."""

    tests_run = [
        {"command": "unit dependency graph", "outcome": "passed"},
        {"command": "unit rollback schema", "outcome": "passed"},
    ]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(root=REPO, result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert artifact["schema"] == exp.SCHEMA
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_ID
    assert artifact["milestone"]["value"] == exp.MILESTONE
    assert artifact["status"]["value"] == "dependency_provenance_ready"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert artifact["continuous_self_learning_target"] is True
    assert artifact["no_weight_mutation"] is True
    assert artifact["dependency_edge_count"] > 0
    assert artifact["dependency_edge_recall"] < 1.0
    assert artifact["dependency_edge_precision"] < artifact["dependency_edge_recall"]
    assert artifact["point_in_time_reconstruction_rate"] == 1.0
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["duplicated_metric_pairs"] == []
    assert artifact["dependency_provenance_ready"] is True
    assert artifact["tests_run"]["value"] == tests_run
    assert artifact["field_principles"] == exp.REQUIRED_FIELD_PRINCIPLES
    exp.validate_artifact(artifact)


def test_req_learn_5355_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-LEARN-5355: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_result_artifact(root=REPO, tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert result["dependency_provenance_ready"] is True
    assert result["continuous_self_learning_target"] is True
    assert result["no_weight_mutation"] is True
    assert result["unsafe_false_accepts"] == 0
    exp.validate_artifact(result)


def test_req_learn_5355_blocked_when_tests_not_recorded() -> None:
    """REQ-LEARN-5355-6: ready gate stays false without test records."""

    artifact = exp.build_result_artifact(root=REPO, tests_run=[])

    assert artifact["status"]["value"] == "blocked_dependency_provenance_gate"
    assert artifact["honest_verdict"]["value"].startswith(
        "blocked_dependency_provenance_not_ready:"
    )
    assert "tests_not_recorded" in artifact["honest_verdict"]["value"]
    assert artifact["dependency_provenance_ready"] is False
    assert artifact["unsafe_false_accepts"] == 0
    exp.validate_artifact(artifact)


def test_req_learn_5355_validation_rejects_schema_drift() -> None:
    """REQ-LEARN-5355-6: artifact validation rejects scalar and gate drift."""

    artifact = exp.build_result_artifact(
        root=REPO,
        tests_run=[{"command": "unit dependency provenance", "outcome": "passed"}],
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "deterministic_provenance_bound"
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
    bad_count["dependency_edge_count"] = True
    with pytest.raises(ValueError, match="dependency_edge_count"):
        exp.validate_artifact(bad_count)

    bad_numeric = deepcopy(artifact)
    bad_numeric["dependency_edge_recall"] = {"value": 1.0}
    with pytest.raises(ValueError, match="dependency_edge_recall"):
        exp.validate_artifact(bad_numeric)

    bad_duplicates = deepcopy(artifact)
    bad_duplicates["duplicated_metric_pairs"] = [
        {
            "left": "memory_hygiene_delta",
            "right": "context_efficiency_delta",
            "value": 0.25,
        }
    ]
    with pytest.raises(ValueError, match="duplicated_metric_pairs"):
        exp.validate_artifact(bad_duplicates)

    bad_unsafe = deepcopy(artifact)
    bad_unsafe["unsafe_false_accepts"] = 1
    with pytest.raises(ValueError, match="unsafe_false_accepts"):
        exp.validate_artifact(bad_unsafe)

    bad_ready = deepcopy(artifact)
    bad_ready["dependency_provenance_ready"] = "yes"
    with pytest.raises(ValueError, match="dependency_provenance_ready"):
        exp.validate_artifact(bad_ready)

    bad_missing_tests = deepcopy(artifact)
    bad_missing_tests["tests_run"]["value"] = []
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_missing_tests)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = [{"command": "lost principle", "outcome": "passed"}]
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)


def test_req_learn_5355_helper_edges_are_deterministic() -> None:
    """REQ-LEARN-5355: helper failure paths stay explicit and deterministic."""

    audit = exp.evaluate_dependency_provenance(exp.build_dependency_cases())
    source_gate = {"failed_gates": ["source_gate"], "all_passed": False}
    unsafe_audit = deepcopy(audit)
    unsafe_audit["unsafe_false_accepts"] = 1
    broken_graph_audit = deepcopy(audit)
    broken_graph_audit["graph_integrity"]["graph_integrity_holds"] = False
    duplicate_pairs = [
        {"left": "memory_hygiene_delta", "right": "context_efficiency_delta", "value": 0.25}
    ]

    assert "unsafe_false_accepts" in exp._honest_verdict(
        False,
        source_gate,
        unsafe_audit,
        [],
        [{"command": "x", "outcome": "passed"}],
    )
    assert "graph_integrity" in exp._honest_verdict(
        False,
        source_gate,
        broken_graph_audit,
        [],
        [{"command": "x", "outcome": "passed"}],
    )
    assert "duplicated_metric_pairs" in exp._honest_verdict(
        False,
        source_gate,
        audit,
        duplicate_pairs,
        [{"command": "x", "outcome": "passed"}],
    )
    assert exp._wrapped_value("plain") == "plain"
    assert exp._json_ready(Path("x")) == "x"
    assert exp._rate(1, 0) == 0.0
