"""Tests for Exp 1510 deterministic plan-graph structural contract gate.

Spec: REQ-VERIFY-1510, SCENARIO-VERIFY-1510.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.verify import plan_graph_structural_contract_gate as exp


def test_req_verify_1510_contract_schema_covers_pre_execution_families() -> None:
    """REQ-VERIFY-1510: contracts cover prerequisite, path, order, object, and API rules."""

    contracts = exp.define_structural_contracts()
    schema_rows = [exp.contract_to_json(contract) for contract in contracts]

    assert [contract.contract_family for contract in contracts] == [
        "graph_prerequisites",
        "acquisition_path",
        "tool_ordering",
        "required_object_acquisition",
        "incompatible_api_use",
    ]
    assert all(row["contract_schema_version"] == exp.CONTRACT_SCHEMA_VERSION for row in schema_rows)
    assert schema_rows[0]["required_node_types"] == [
        "prompt",
        "tool_call",
        "tool_result",
        "final_answer",
    ]
    assert ["tool_call", "tool_result"] in schema_rows[1]["required_path_edges"]
    assert ["delete", "read_after_delete"] in schema_rows[-1]["incompatible_operations"]


def test_req_verify_1510_loads_exp1501_graphs_and_exp1509_events(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1510: gate inputs preserve Exp 1501 graph and Exp 1509 event provenance."""

    exp1501_artifact = tmp_path / "experiment_1501.json"
    exp1509_manifest = tmp_path / "events_1509.jsonl"
    _write_json(
        exp1501_artifact,
        {
            "status": "complete",
            "plan_graph_energy_ready": True,
            "selected_trace_ids": ["cctu-1486-arith-001"],
        },
    )
    _write_jsonl(
        exp1509_manifest,
        [
            _runtime_event("cctu-1486-arith-001", 1),
            _runtime_event("cctu-1486-arith-001", 2),
            _runtime_event("not-selected", 3),
        ],
    )

    gate_inputs = exp.load_gate_inputs(
        exp1501_artifact_path=exp1501_artifact,
        exp1509_event_manifest_path=exp1509_manifest,
        trace_limit=3,
    )
    contracts = exp.define_structural_contracts()
    rows = exp.evaluate_graph_contracts(
        gate_inputs.graphs[0],
        contracts,
        graph_label="known_good",
        graph_index=0,
        runtime_event_count=gate_inputs.runtime_event_counts_by_case["cctu-1486-arith-001"],
    )

    assert gate_inputs.blockers == []
    assert [graph.case_id for graph in gate_inputs.graphs] == ["cctu-1486-arith-001"]
    assert gate_inputs.runtime_events_loaded == 3
    assert gate_inputs.runtime_event_counts_by_case["cctu-1486-arith-001"] == 2
    assert all(row["detected_violation"] is False for row in rows)
    assert all(row["classifier_outcome"] == "true_negative" for row in rows)
    assert {row["runtime_event_count"] for row in rows} == {2}


def test_scenario_verify_1510_injected_violations_are_classified() -> None:
    """SCENARIO-VERIFY-1510: injected structural contract violations are rejected."""

    graph = exp.load_gate_inputs(trace_limit=1).graphs[0]
    contracts = exp.define_structural_contracts()
    known_good_rows = exp.evaluate_graph_contracts(
        graph,
        contracts,
        graph_label="known_good",
        graph_index=0,
    )
    violations = exp.inject_structural_contract_violations(graph)
    violation_rows = [
        row
        for violation_index, violation in enumerate(violations, start=1)
        for row in exp.evaluate_graph_contracts(
            violation.graph,
            contracts,
            graph_label="injected_violation",
            graph_index=violation_index,
            injected_violation=violation,
        )
    ]
    target_rows = [row for row in violation_rows if row["expected_violation"]]
    metrics = exp.aggregate_gate_metrics([*known_good_rows, *violation_rows])

    assert [violation.contract_family for violation in violations] == [
        "graph_prerequisites",
        "acquisition_path",
        "tool_ordering",
        "required_object_acquisition",
        "incompatible_api_use",
    ]
    assert len(target_rows) == len(violations)
    assert {row["classifier_outcome"] for row in target_rows} == {"true_positive"}
    assert {row["violation_code"] for row in target_rows} == {
        "missing_prerequisite_edge",
        "broken_acquisition_path",
        "tool_ordering_violation",
        "missing_required_object_acquisition",
        "incompatible_api_operations",
    }
    assert metrics["violations_injected"] == 5
    assert metrics["violations_detected"] == 5
    assert metrics["false_accept_rate"] == pytest.approx(0.0)
    assert metrics["false_reject_rate"] == pytest.approx(0.0)
    assert 0.0 <= metrics["random_baseline_detection_rate"] <= 1.0
    assert 0.0 <= metrics["length_baseline_detection_rate"] <= 1.0


def test_scenario_verify_1510_runner_writes_manifest_and_ready_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1510: runner writes per-contract rows and required artifact fields."""

    output_path = tmp_path / "experiment_1510.json"
    manifest_path = tmp_path / "contracts_1510.jsonl"
    exp1501_artifact = tmp_path / "experiment_1501.json"
    exp1509_manifest = tmp_path / "events_1509.jsonl"
    _write_json(
        exp1501_artifact,
        {
            "status": "complete",
            "plan_graph_energy_ready": True,
            "selected_trace_ids": ["cctu-1486-arith-001", "cctu-1486-table-001"],
        },
    )
    _write_jsonl(exp1509_manifest, [_runtime_event("cctu-1486-arith-001", 1)])

    artifact = exp.run_structural_contract_gate(
        output_path=output_path,
        manifest_path=manifest_path,
        exp1501_artifact_path=exp1501_artifact,
        exp1509_event_manifest_path=exp1509_manifest,
        trace_limit=4,
        tests_run=["focused pytest"],
    )
    manifest_rows = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["structural_contract_gate_ready"] is True
    assert artifact["plan_graphs_checked"] == 12
    assert artifact["contracts_defined"] == 5
    assert artifact["violations_injected"] == 10
    assert artifact["violations_detected"] == 10
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["false_reject_rate"] == pytest.approx(0.0)
    assert artifact["contract_manifest_path"] == str(manifest_path)
    assert artifact["blockers"] == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == ["focused pytest"]
    assert len(manifest_rows) == artifact["plan_graphs_checked"] * artifact["contracts_defined"]
    assert exp.validate_contract_manifest_row(manifest_rows[0]) == []
    assert {
        row["runtime_event_count"]
        for row in manifest_rows
        if row["case_id"] == "cctu-1486-arith-001"
    } == {1}


def test_req_verify_1510_bootstrap_blocked_cli_and_defensive_branches(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-1510: bootstrap, blocked runs, CLI, and schema errors are deterministic."""

    output_path = tmp_path / "bootstrap.json"
    payload = exp.write_in_progress_artifact(output_path, manifest_path=tmp_path / "rows.jsonl")
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
    assert payload["status"] == "in_progress"
    assert payload["structural_contract_gate_ready"] is False

    missing_inputs = exp.load_gate_inputs(
        exp1501_artifact_path=tmp_path / "missing_exp1501.json",
        exp1509_event_manifest_path=tmp_path / "missing_events.jsonl",
        trace_limit=0,
    )
    assert missing_inputs.graphs == []
    assert missing_inputs.blockers == [
        f"missing_exp1501_artifact:{tmp_path / 'missing_exp1501.json'}",
        f"missing_exp1509_event_manifest:{tmp_path / 'missing_events.jsonl'}",
        "no_plan_graphs_loaded",
    ]

    blocked_output = tmp_path / "blocked.json"
    blocked_manifest = tmp_path / "blocked.jsonl"
    blocked = exp.run_structural_contract_gate(
        output_path=blocked_output,
        manifest_path=blocked_manifest,
        exp1501_artifact_path=tmp_path / "missing_exp1501.json",
        exp1509_event_manifest_path=tmp_path / "missing_events.jsonl",
        trace_limit=0,
    )
    assert blocked["status"] == "blocked"
    assert blocked["structural_contract_gate_ready"] is False
    assert blocked["false_accept_rate"] is None
    assert blocked_manifest.read_text(encoding="utf-8") == ""

    invalid_row = dict(
        exp.evaluate_graph_contracts(
            exp.load_gate_inputs(trace_limit=1).graphs[0], exp.define_structural_contracts()
        )[0]
    )
    del invalid_row["contract_id"]
    assert "missing:contract_id" in exp.validate_contract_manifest_row(invalid_row)
    invalid_row["contract_id"] = "contract"
    invalid_row["contract_schema_version"] = "wrong"
    assert "invalid:contract_schema_version" in exp.validate_contract_manifest_row(invalid_row)
    assert exp.aggregate_gate_metrics([])["false_accept_rate"] is None

    not_ready_1501 = tmp_path / "not_ready_exp1501.json"
    empty_events = tmp_path / "empty_events.jsonl"
    _write_json(not_ready_1501, {"status": "complete", "plan_graph_energy_ready": False})
    _write_jsonl(empty_events, [])
    not_ready_inputs = exp.load_gate_inputs(
        exp1501_artifact_path=not_ready_1501,
        exp1509_event_manifest_path=empty_events,
        trace_limit=1,
    )
    assert f"exp1501_not_ready:{not_ready_1501}" in not_ready_inputs.blockers
    assert [graph.case_id for graph in not_ready_inputs.graphs] == ["cctu-1486-arith-001"]

    graph = exp.load_gate_inputs(trace_limit=1).graphs[0]
    no_prompt_graph = exp.PlanGraph(
        case_id=graph.case_id,
        family=graph.family,
        nodes=tuple(node for node in graph.nodes if node.node_type != "prompt"),
        edges=tuple(graph.edges),
        expected_outputs=graph.expected_outputs,
    )
    prerequisite_contract = exp.define_structural_contracts()[0]
    no_prompt_row = exp.evaluate_graph_contracts(no_prompt_graph, [prerequisite_contract])[0]
    assert no_prompt_row["violation_code"] == "missing_prerequisite_node"

    assert exp._classifier_outcome(True, False) == "false_accept"
    assert exp._classifier_outcome(False, True) == "false_reject"
    bare_edge = exp.PlanEdge(
        edge_id="edge:bare",
        source="missing",
        target="target",
        edge_type="bare",
        attributes={},
    )
    stepped_node = exp.PlanNode(
        node_id="source",
        node_type="source",
        label="source",
        value_type="source",
        attributes={"step_index": 7},
    )
    assert exp._edge_step(bare_edge, "source_step", None) == 0
    assert exp._edge_step(bare_edge, "source_step", stepped_node) == 7

    cli_output = tmp_path / "cli.json"
    cli_manifest = tmp_path / "cli.jsonl"
    rc = exp.main(
        [
            "--trace-limit",
            "1",
            "--output",
            str(cli_output),
            "--manifest",
            str(cli_manifest),
        ]
    )
    assert rc == 0
    assert "ready=True" in capsys.readouterr().out


def _runtime_event(case_id: str, replay_index: int) -> dict[str, Any]:
    return {
        "event_schema_version": "monitor-runtime-event/v1",
        "event_id": f"runtime-1509-{replay_index:06d}",
        "replay_index": replay_index,
        "source_experiment": "1509",
        "source_kind": "monitor",
        "source_path": "results/executable_monitor_events_1509.jsonl",
        "source_line": replay_index,
        "source_row_id": f"row-{replay_index}",
        "source_event_id": f"source-{replay_index}",
        "event_kind": "monitor_decision",
        "case_id": case_id,
        "family": "arithmetic",
        "token_offset": replay_index * 64,
        "validation_status": "pass",
        "verifier_false_accept": False,
        "linked_monitor_event_id": None,
        "link_status": "not_applicable",
        "provenance": {},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
