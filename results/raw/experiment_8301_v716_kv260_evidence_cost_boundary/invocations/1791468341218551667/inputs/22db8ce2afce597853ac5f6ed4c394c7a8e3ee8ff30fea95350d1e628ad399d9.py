"""REQ-REPORT-8301 / REQ-VERIFY-8301: scoped CPU evidence and real CLI."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import kv260_evidence_cost_boundary_8301 as h
from carnot.reporting import kv260_evidence_cost_execution_8301 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.primary_publication import publish_primary
import test_kv260_evidence_cost_boundary_8287 as previous
import test_kv260_evidence_cost_boundary_8273 as cli_tests


def data():
    """Private clocks exercise mechanics without granting natural study benefit."""
    value = previous.data()
    value.update(cpu_ready=True, cpu_binding={"invocation": "private"}, cpu_cost_rows=[])
    for arm in ["full", "scoped", "one_hop"]:
        value["cpu_cost_rows"].append(
            dict(
                graph_id="private-graph",
                topology="chain",
                arm=arm,
                repetition=0,
                total_ns=100,
                prefix_issue_ns=10,
                closure_ns=20,
                replay_ns=30,
                persistence_ns=25,
                scan_count=12,
                fallback_count=0,
            )
        )
    return value


def test_cpu_survives_missing_live():
    """SCENARIO-VERIFY-8301-BOUNDARY: live absence never erases CPU clocks."""
    value = data()
    value.update(requests=[], cold_costs=[])
    result = h.reduce(value, h.precision(value))
    assert result["scoped_cpu_boundary_ready_score"] == 1
    assert result["live_request_boundary_ready_score"] == 0
    assert result["verdict_class"] == "blocked"
    assert len(result["scoped_operation_rows"]) == 3
    assert result["cpu_graph_unit_count"] == 1
    assert result["independent_count"] == 0
    for row in result["scoped_operation_rows"]:
        assert row["host_overhead_ns"] == 15
        assert sum(row["disjoint_component_ns"].values()) == row["total_ns"]
        assert row["compatible_fraction"] == 0 and row["maximum_gain"] == 1
        assert row["counter_update_ns"] is None
        assert row["invocation_binding"] == value["cpu_binding"]
    assert result["ideal_whole_request_bound"]["maximum_gain"] is None
    assert all(r["maximum_gain"] == 1 for r in result["cpu_condition_bounds"])
    assert any(
        c["artifact_field"] == "complete_live_request_spans" for c in result["gate_check_summary"]
    )
    assert not result["nfr01_met"] and result["current_device_execution_count"] == 0
    assert result["generalized_learning_benefit_score"] == 0
    ops = {r["operation"]: r for r in result["operation_rows"]}
    assert ops["quadratic_ising"]["k_max"] == 5
    assert all(
        not ops[n]["existing_fabric_supported"]
        for n in [
            "graph_closure",
            "integer_counters",
            "typed_constraint_checks",
            "gaussian_head",
            "persistence",
        ]
    )
    value["cpu_ready"] = False
    assert h.reduce(value, [])["scoped_cpu_boundary_ready_score"] == 0
    value["cpu_ready"] = True
    value["cpu_cost_rows"][0]["total_ns"] = 1
    with pytest.raises(ValueError, match="cpu_span"):
        h.reduce(value, [])


def test_qualified_scopes_and_mechanics(monkeypatch):
    """REQ-VERIFY-8301: retain prior overlap, overflow and fallback assertions."""
    value = data()
    value["current"] = dict.fromkeys(["boundary", *h.CURRENT], True)
    value["requests"][0].update(workload="fit_concurrent", arm="fit")
    value["requests"].append(dict(value["requests"][0], request_id="q2"))
    value["cold_costs"][0]["workload"] = "fit_concurrent"
    result = h.reduce(value, h.precision(value))
    assert result["live_request_boundary_ready_score"] == 1
    assert result["verdict_class"] == "circular_positive"
    row = next(r for r in result["phase_cost_rows"] if r["source_cost_scope"] == "current_v716")
    assert row["request_work_sum_s"] == 2
    assert row["observed_parallel_makespan_s"] == pytest.approx(3.1)
    assert row["cold_start_count"] == 1 and row["sequential_latency_s"] is None
    assert any(r["overflow"] for r in result["fixed_point_rows"])
    assert all(r["final_action_changed"] == 0 for r in result["fixed_point_rows"])
    value["cpu_ready"] = False
    assert h.reduce(value, [])["honest_verdict"] == "complete_blocked_cpu"
    value["cpu_ready"] = True
    value["resources_available"] = False
    assert h.reduce(value, [])["scoped_cpu_boundary_ready_score"] == 0


def test_missing_and_authenticated_cpu(tmp_path, monkeypatch):
    """REQ-REPORT-8301: terminal custody and CPU gates do not depend on live gates."""
    monkeypatch.setattr(h.legacy.prior, "load", lambda root, raw: data())
    atomic_json(
        tmp_path / "results/experiment_8242_v712_independent_concurrent_service.json",
        dict(phase_spans=[]),
    )
    value = h.load(tmp_path, tmp_path / "raw")
    assert value["cpu_ready"] is False and not value["current"]["capture"]
    path = tmp_path / "results/experiment_8291_v716_dependency_scoped_admission.json"
    work_path = tmp_path / "cpu-primitives/work.json"
    manifest_path = tmp_path / "cpu-primitives/manifest.json"
    primitive_path = tmp_path / "cpu-primitives/cost.json"
    atomic_json(manifest_path, {})
    atomic_json(primitive_path, {"fixture": True})
    atomic_json(
        work_path,
        dict(
            manifest_path=str(manifest_path),
            manifest_sha256=reference(manifest_path)["sha256"],
            runs=[dict.fromkeys(["journal", "costs", "prefix"], reference(primitive_path))],
        ),
    )
    payload = dict(
        experiment_id=8291,
        task_id="exp8291-fixture",
        honest_verdict="complete_circular_positive_fixture",
        verdict_class="circular_positive",
        required_checks_passed=True,
        flagged_adversarial=False,
        raw_shard_hashes=[],
        soundness_ready_score=1,
        operation_cost_rows=data()["cpu_cost_rows"],
        invocation_argv=["private"],
        terminal_validation_sidecar_path=str(work_path.with_name("terminal.json")),
    )
    publish_primary(path, payload, lambda p: dict(passed=True))
    monkeypatch.setattr(h.cpu, "replay", lambda p: True)
    value = h.load(tmp_path, tmp_path / "raw2")
    assert value["cpu_ready"] and value["cpu_cost_rows"] == payload["operation_cost_rows"]
    assert h.verify_cpu(value) is None
    value["cpu_cost_rows"][0]["closure_ns"] += 1
    with pytest.raises(ValueError, match="cpu_cost_drift"):
        h.verify_cpu(value)
    monkeypatch.setattr(h.cpu, "replay", lambda p: False)
    value = h.load(tmp_path, tmp_path / "raw3")
    assert value["cpu_ready"] is False
    assert any(
        c["artifact_field"] == "primitive_replay" and not c["passed"] for c in value["checks"]
    )


def test_frozen_commands(tmp_path):
    """REQ-REPORT-8301: freeze actual argv, E2E, strict coverage and consumers."""
    plan = e.commands(tmp_path)
    assert {"coverage_combine", "e2e015", "e2e019", "affected_consumers"} <= {s.name for s in plan}
    assert len(e.validators(tmp_path / (h.NAME + ".json"))) == 3
    assert h.MODEL_SPECS == [] and all("8301" in p for p in e.OWNED)


def test_real_cli_and_rehashed_controls(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8301-CLI: real children reject negative and rehashed bytes."""
    monkeypatch.setattr(cli_tests, "h", h)
    monkeypatch.setattr(cli_tests, "e", e)
    monkeypatch.setattr(cli_tests, "data", data)
    cli_tests.test_real_cli_and_tamper(tmp_path)


def test_failure_paths(tmp_path, monkeypatch):
    """REQ-REPORT-8301: all readiness goes to zero on owned validation failure."""
    monkeypatch.setattr(cli_tests, "h", h)
    monkeypatch.setattr(cli_tests, "e", e)
    monkeypatch.setattr(cli_tests, "data", data)
    cli_tests.test_owned_and_terminal_failure_paths(tmp_path, monkeypatch)


def test_rehashed_cpu_and_failure_readiness(tmp_path):
    """SCENARIO-REPORT-8301-CLI: CPU summaries cannot replace their parent bytes."""
    source, output = tmp_path / "input.json", tmp_path / (h.NAME + ".json")
    value = data()
    atomic_json(source, value)
    monkey = pytest.MonkeyPatch()
    monkey.setattr(cli_tests, "h", h)
    try:
        result = cli_tests.run_cli("--input", source, "--output", output)
        assert result.returncode == 0, result.stdout + result.stderr
        original = json.loads(output.read_bytes())
        altered = deepcopy(original)
        altered["scoped_operation_rows"][0]["closure_ns"] += 1
        altered["reproducibility_checksum"] = e.checksum(altered)
        atomic_json(output, altered)
        assert cli_tests.run_cli("--cold-replay", output).returncode == 1
        altered = deepcopy(original)
        altered.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_owned_checks",
            required_checks_passed=False,
        )
        altered = e.normalize(altered)
        assert all(altered[k] == 0 for k in e.READINESS)
        altered["reproducibility_checksum"] = e.checksum(altered)
        atomic_json(output, altered)
        assert cli_tests.run_cli("--cold-replay", output).returncode == 0
        assert e.main(["--cold-replay", str(output)]) == 0
    finally:
        monkey.undo()
