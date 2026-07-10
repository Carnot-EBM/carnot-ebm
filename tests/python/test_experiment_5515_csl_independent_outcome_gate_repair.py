"""Tests for Exp5515 independent-outcome graph-memory gate repair.

Spec refs: REQ-LEARN-5515,
SCENARIO-LEARN-5515-INDEPENDENT-LABELS,
SCENARIO-LEARN-5515-GRAPH-CONTROLS,
SCENARIO-LEARN-5515-GATE-FIELDS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5515_csl_independent_outcome_gate_repair as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/self-learning/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5515_csl_independent_outcome_gate_repair.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5515_csl_independent_outcome_gate_repair.py "
    "-m pytest tests/python/test_experiment_5515_csl_independent_outcome_gate_repair.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5515_csl_independent_outcome_gate_repair.py "
    "--fail-under=100"
)
TESTS_RUN = [TEST_COMMAND, COVERAGE_COMMAND]


def _artifact() -> exp.JsonDict:
    return exp.build_artifact(root=REPO, tests_run=TESTS_RUN)


def test_req_learn_5515_spec_declares_independent_gate_repair_contract() -> None:
    """REQ-LEARN-5515: OpenSpec anchors labels, controls, and gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-LEARN-5515") :]

    for marker in (
        "REQ-LEARN-5515",
        "SCENARIO-LEARN-5515-INDEPENDENT-LABELS",
        "SCENARIO-LEARN-5515-GRAPH-CONTROLS",
        "SCENARIO-LEARN-5515-GATE-FIELDS",
        str(exp.RESULT_RELATIVE_PATH),
        exp.INFERENCE_SUBSTRATE,
        "independent label table",
        "stale-memory",
        "negative-transfer",
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_learn_5515_independent_labels_score_all_conditions() -> None:
    """SCENARIO-LEARN-5515-INDEPENDENT-LABELS: labels are not retrieval scores."""

    fixture = exp.build_stream_fixture()
    pre_graph = exp.initial_memory_graph()
    post_graph, updates = exp.apply_memory_updates(fixture, pre_graph)
    no_memory = exp.score_condition(fixture, pre_graph, condition="no_memory")
    stale_memory = exp.score_condition(fixture, pre_graph, condition="stale_memory")
    graph_memory = exp.score_condition(fixture, post_graph, condition="graph_memory")
    audit = exp.metric_independence_audit(fixture)

    label_ids = {row["label_id"] for row in fixture["heldout_labels"]}
    assert len(updates) == len(fixture["pre_memory_tasks"]) == 3
    assert exp.graph_hash(pre_graph) != exp.graph_hash(post_graph)
    assert no_memory["score"] == pytest.approx(0.0)
    assert stale_memory["score"] == pytest.approx(0.0)
    assert graph_memory["score"] == pytest.approx(1.0)
    assert audit["clean"] is True
    assert audit["label_source"] == exp.INDEPENDENT_LABEL_SOURCE

    for condition in (no_memory, stale_memory, graph_memory):
        assert {row["label_id"] for row in condition["row_results"]} == label_ids
        assert all(row["label_source"] == exp.INDEPENDENT_LABEL_SOURCE for row in condition["row_results"])
    assert all(
        "utility_score" not in label["label_id"] and label["source_kind"] == "independent_fixture"
        for label in fixture["heldout_labels"]
    )


def test_scenario_learn_5515_graph_controls_record_rejections() -> None:
    """SCENARIO-LEARN-5515-GRAPH-CONTROLS: unsafe memories are rejected."""

    artifact = _artifact()
    traces = artifact["retrieval_traces"]["graph_memory"]
    traces_by_task = {trace["task_id"]: trace for trace in traces}

    exp.validate_artifact(artifact)
    assert artifact["negative_transfer_rate"] == pytest.approx(0.0)
    assert artifact["stale_evidence_rejection_rate"] == pytest.approx(1.0)
    assert artifact["stale_evidence_cases"] == [
        {
            "node_id": "node5515-stale-db-restart",
            "task_id": "5515-heldout-db-timeout",
            "rejection_reason": "stale_evidence",
        }
    ]
    assert artifact["negative_transfer_cases"] == [
        {
            "node_id": "node5515-transfer-sql-offset",
            "task_id": "5515-heldout-api-pagination",
            "rejection_reason": "negative_transfer",
        }
    ]
    assert traces_by_task["5515-heldout-db-timeout"]["selected_action"] == "run-circuit-reset"
    assert traces_by_task["5515-heldout-db-timeout"]["rejected_node_ids_by_reason"] == {
        "stale_evidence": ["node5515-stale-db-restart"]
    }
    assert traces_by_task["5515-heldout-api-pagination"]["selected_action"] == "use-zero-index-bound"
    assert traces_by_task["5515-heldout-api-pagination"]["rejected_node_ids_by_reason"] == {
        "negative_transfer": ["node5515-transfer-sql-offset"]
    }
    assert traces_by_task["5515-heldout-access-policy"]["selected_action"] == "deny-escalation"


def test_scenario_learn_5515_gate_fields_are_top_level_and_stable(tmp_path: Path) -> None:
    """SCENARIO-LEARN-5515-GATE-FIELDS: run() writes resolvable gates."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    fixture_path = tmp_path / exp.STREAM_FIXTURE_RELATIVE_PATH
    artifact = exp.run(
        root=REPO,
        result_path=result_path,
        stream_fixture_path=fixture_path,
        tests_run=TESTS_RUN,
        write=True,
    )
    dry_run = exp.run(
        root=REPO,
        result_path=tmp_path / "dry-run.json",
        stream_fixture_path=fixture_path,
        tests_run=TESTS_RUN,
        write=False,
    )
    repo_result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    repo_replay = exp.build_artifact(root=REPO, tests_run=repo_result["tests_run"])

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert json.loads(fixture_path.read_text(encoding="utf-8")) == artifact["stream_fixture"]
    assert dry_run == artifact
    assert not (tmp_path / "dry-run.json").exists()
    assert repo_result == repo_replay
    for field in (
        "metric_independence_clean",
        "csl_experience_graph_ready",
        "csl_gate_fields_resolvable",
        "continuous_self_learning_evidence",
    ):
        assert artifact[field] is True
        assert field in artifact
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_learn_5515_validation_fails_closed_on_gate_or_metric_drift() -> None:
    """REQ-LEARN-5515-5/6/7: malformed evidence cannot pass gates."""

    artifact = _artifact()
    exp.validate_artifact(artifact)
    assert exp._resolve_output_path(REPO, exp.RESULT_RELATIVE_PATH) == REPO / exp.RESULT_RELATIVE_PATH
    assert exp._resolve_output_path(REPO, REPO / exp.RESULT_RELATIVE_PATH) == REPO / exp.RESULT_RELATIVE_PATH

    drift_cases = [
        ("inference_substrate", "verifier_ensemble_against_cached_candidates", "inference_substrate"),
        ("independent_label_source", "memory_utility_score", "independent_label_source"),
        ("pre_memory_hash", artifact["post_memory_hash"], "pre_memory_hash"),
        ("post_memory_hash", artifact["pre_memory_hash"], "post_memory_hash"),
        ("no_memory_score", 0.5, "heldout_delta"),
        ("stale_memory_score", 1.0, "csl_experience_graph_ready"),
        ("heldout_delta", 0.0, "heldout_delta"),
        ("negative_transfer_rate", 0.5, "negative_transfer_rate"),
        ("stale_evidence_rejection_rate", 0.5, "stale_evidence_rejection_rate"),
        ("metric_independence_clean", False, "metric_independence_clean"),
        ("csl_experience_graph_ready", False, "csl_experience_graph_ready"),
        ("csl_gate_fields_resolvable", False, "csl_gate_fields_resolvable"),
        ("continuous_self_learning_evidence", False, "continuous_self_learning_evidence"),
        ("honest_verdict", "done", "honest_verdict"),
        ("research_conductor_modified", True, "scripts/research_conductor.py"),
    ]
    for field, value, expected in drift_cases:
        bad = deepcopy(artifact)
        bad[field] = value
        with pytest.raises(ValueError, match=expected):
            exp.validate_artifact(bad)

    missing = deepcopy(artifact)
    missing.pop("csl_gate_fields_resolvable")
    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(missing)

    no_tests = deepcopy(artifact)
    no_tests["tests_run"] = []
    no_tests["csl_experience_graph_ready"] = False
    no_tests["continuous_self_learning_evidence"] = False
    no_tests["honest_verdict"] = "blocked: independent_outcome_graph_memory_not_ready"
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(no_tests)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.validate_artifact(bad_checksum)


def test_req_learn_5515_helper_edges_keep_exact_outcomes_explicit() -> None:
    """REQ-LEARN-5515-1/4: helper edges preserve independent outcomes."""

    fixture = exp.build_stream_fixture()
    task = fixture["heldout_tasks"][0]
    graph = {
        "schema": "edge-case",
        "nodes": [
            exp.memory_node(
                "node5515-uncached",
                "skill",
                "not-cached",
                domain=task["domain"],
                locality=task["locality"],
                tags=task["tags"],
                trust_score=0.99,
                version=1,
                success_count=1,
                description="Uncached action cannot steer a cached replay.",
            )
        ],
        "edges": [],
    }
    trace = exp.retrieve_memory(task, graph, enforce_controls=True)
    outcome = exp.exact_label_outcome(task, "not-cached")
    controls = exp.control_rates([trace])
    stale_memory = exp.score_condition(fixture, exp.initial_memory_graph(), condition="stale_memory")
    stale_controls = exp.control_rates(stale_memory["retrieval_traces"])

    assert trace["selected_action"] == task["no_memory_action"]
    assert trace["rejected_node_ids_by_reason"] == {"action_not_cached": ["node5515-uncached"]}
    assert outcome["accepted"] is False
    assert outcome["failure_reasons"] == ["selected_action_not_cached"]
    assert controls["stale_evidence_rejection_rate"] == pytest.approx(0.0)
    assert controls["negative_transfer_rate"] == pytest.approx(0.0)
    assert stale_controls["negative_transfer_rate"] == pytest.approx(1.0)
    assert exp._list_of_mappings({"not": "a-list"}) == []
    assert exp._string_list("not-a-list") == []
    assert exp._honest_verdict(False).startswith("blocked:")
