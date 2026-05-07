"""Tests for Exp 1501 deterministic plan-graph energy adapter.

Spec: REQ-VERIFY-1501, SCENARIO-VERIFY-1501.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verify import plan_graph_energy_adapter as exp


def test_req_verify_1501_converts_cctu_cases_to_directed_plan_graphs() -> None:
    """REQ-VERIFY-1501: CCTU tool-use cases become typed dependency graphs."""

    graphs = exp.convert_cctu_traces_to_plan_graphs(trace_limit=8)

    assert len(graphs) == 8
    assert len({graph.case_id for graph in graphs}) == 8
    for graph in graphs:
        node_types = {node.node_type for node in graph.nodes}
        edge_types = {edge.edge_type for edge in graph.edges}

        assert {"prompt", "tool_call", "tool_result", "final_answer"} <= node_types
        assert {"prompt_to_tool", "tool_to_result", "result_to_answer"} <= edge_types
        assert graph.expected_outputs["tool_name"]
        assert graph.expected_outputs["final_answer"]
        assert all(edge.source != edge.target for edge in graph.edges)
        assert all("step_index" in node.attributes for node in graph.nodes)
        assert all("dependency_required" in edge.attributes for edge in graph.edges)


def test_req_verify_1501_injects_faults_and_localizes_risky_elements() -> None:
    """REQ-VERIFY-1501: graph energy ranks injected dependency faults top-1."""

    original = exp.convert_cctu_traces_to_plan_graphs(trace_limit=1)[0]
    faults = exp.inject_dependency_faults(original)

    assert {fault.fault_type for fault in faults} == {
        "dangling_output",
        "missing_edge",
        "missing_intermediate",
        "wrong_ordering",
        "wrong_tool_input_type",
    }
    assert all(not node.attributes.get("fault_injected") for node in original.nodes)
    assert all(edge.attributes.get("present", True) for edge in original.edges)

    for fault in faults:
        evaluation = exp.evaluate_fault(fault)
        score = exp.score_graph_risk(fault.graph)

        assert score.energy > 0
        assert score.ranked_node_ids[0] == fault.target_node_id
        assert score.ranked_edge_ids[0] == fault.target_edge_id
        assert evaluation["node_localized_top1"] is True
        assert evaluation["edge_localized_top1"] is True
        assert evaluation["trained_gnn_used"] is False
        assert evaluation["graph_risk_energy"] == pytest.approx(score.energy)


def test_req_verify_1501_bootstrap_artifact_contains_required_fields(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1501: the adapter can write the mandatory in-progress artifact."""

    output_path = tmp_path / "experiment_1501.json"
    payload = exp.write_in_progress_artifact(output_path, run_date="20260507")
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload == persisted
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(payload)
    assert payload["status"] == "in_progress"
    assert payload["plan_graph_energy_ready"] is False
    assert payload["adapter_manifest_path"].endswith("plan_graph_energy_manifest_1501.jsonl")


def test_scenario_verify_1501_runner_writes_manifest_and_complete_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1501: the runner writes per-fault rows and beats baselines."""

    output_path = tmp_path / "experiment_1501.json"
    manifest_path = tmp_path / "plan_graph_energy_manifest_1501.jsonl"

    artifact = exp.run_plan_graph_energy_adapter(
        output_path=output_path,
        manifest_path=manifest_path,
        run_date="20260507",
        trace_limit=4,
        tests_run=["focused pytest"],
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact == persisted
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["plan_graph_energy_ready"] is True
    assert artifact["traces_converted"] == 4
    assert artifact["injected_graph_faults"] == 20
    assert artifact["node_localization_top1_rate"] == pytest.approx(1.0)
    assert artifact["edge_localization_top1_rate"] == pytest.approx(1.0)
    assert artifact["graph_energy_beats_baselines"] is True
    assert artifact["random_baseline_top1_rate"] < 1.0
    assert artifact["length_baseline_top1_rate"] < 1.0
    assert artifact["tests_run"] == ["focused pytest"]
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(rows) == artifact["injected_graph_faults"]
    assert rows[0]["trace_id"] == "cctu-1486-arith-001"
    assert rows[0]["nodes"]
    assert rows[0]["edges"]
    assert rows[0]["expected_outputs"]["tool_name"]
    assert rows[0]["scorer_type"] == "deterministic_graph_risk"
    assert rows[0]["trained_gnn_used"] is False


def test_req_verify_1501_metrics_handle_empty_rows() -> None:
    """REQ-VERIFY-1501: aggregate metrics fail closed before any fault rows exist."""

    metrics = exp.aggregate_localization_metrics([])

    assert metrics["node_localization_top1_rate"] == 0.0
    assert metrics["edge_localization_top1_rate"] == 0.0
    assert metrics["random_baseline_top1_rate"] == 0.0
    assert metrics["length_baseline_top1_rate"] == 0.0
    assert metrics["graph_energy_beats_baselines"] is False


def test_req_verify_1501_cli_and_defensive_branches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-1501: CLI and bounded helper branches stay deterministic."""

    assert exp.convert_cctu_traces_to_plan_graphs(trace_limit=0) == []
    assert len(exp.convert_cctu_traces_to_plan_graphs(trace_limit=100)) == 20
    assert exp._json_type(None) == "null"
    assert exp._json_type(True) == "bool"
    assert exp._json_type(object()).endswith("object")
    assert exp._rate([]) == 0.0

    empty_graph = exp.PlanGraph(
        case_id="empty",
        family="empty",
        nodes=(),
        edges=(),
        expected_outputs={},
    )
    assert exp._length_baseline_node_id(empty_graph) is None
    assert exp._length_baseline_edge_id(empty_graph) is None
    with pytest.raises(ValueError, match="missing edge"):
        exp._edge_by_id(empty_graph, "edge:missing")

    output_path = tmp_path / "cli_experiment_1501.json"
    manifest_path = tmp_path / "cli_manifest_1501.jsonl"
    rc = exp.main(
        [
            "--trace-limit",
            "1",
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
        ]
    )
    assert rc == 0
    assert "ready=True" in capsys.readouterr().out

    blocked_output = tmp_path / "blocked_experiment_1501.json"
    blocked_manifest = tmp_path / "blocked_manifest_1501.jsonl"
    blocked_rc = exp.main(
        [
            "--trace-limit",
            "0",
            "--output",
            str(blocked_output),
            "--manifest",
            str(blocked_manifest),
        ]
    )
    assert blocked_rc == 1
    assert json.loads(blocked_output.read_text(encoding="utf-8"))["status"] == "blocked"
