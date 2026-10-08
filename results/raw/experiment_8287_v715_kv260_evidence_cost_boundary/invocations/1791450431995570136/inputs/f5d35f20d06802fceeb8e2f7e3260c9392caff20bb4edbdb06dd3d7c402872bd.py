"""REQ-REPORT-8273 / REQ-VERIFY-8273: private custody, costs and real CLI."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import kv260_evidence_cost_boundary_8273 as h
from carnot.reporting import kv260_evidence_cost_execution_8273 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.primary_publication import publish_primary
from test_kv260_evidence_cost_boundary_8258 import data as prior_data


def data():
    """Private clocks are mechanics fixtures and carry no natural acquisition claim."""
    value = prior_data()
    value.update(
        historical_requests=deepcopy(value["requests"]),
        historical_cold_costs=deepcopy(value["cold_costs"]),
        historical_service_spans=[],
        historical_cost_binding={"invocation": "private"},
    )
    return value


def run_cli(*argv):
    """Execute the actual runner outside this checkout with no ambient path."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, str(h.ROOT / h.CLI), *map(str, argv)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_current_and_historical_costs():
    """SCENARIO-VERIFY-8273-COSTS: absent current clocks never inherit f=0."""
    value = data()
    value["requests"] = []
    value["cold_costs"] = []
    result = h.reduce(value, h.precision(value))
    assert result["ideal_whole_request_bound"]["status"] == "unavailable"
    assert result["ideal_whole_request_bound"]["compatible_fraction"] is None
    assert result["historical_ideal_whole_request_bound"]["maximum_gain"] == 1
    assert result["kv260_boundary_ready_score"] == 1
    assert result["verdict_class"] == "blocked"
    assert all(r["source_cost_scope"] == "historical_v712" for r in result["phase_cost_rows"])
    value = data()
    value["current"] = dict.fromkeys(["boundary", *h.CURRENT], True)
    value["requests"][0].update(workload="fit_concurrent", arm="fit")
    value["requests"].append(dict(value["requests"][0], request_id="q2"))
    value["cold_costs"][0]["workload"] = "fit_concurrent"
    result = h.reduce(value, h.precision(value))
    current = [r for r in result["phase_cost_rows"] if r["source_cost_scope"] == "current_v714"]
    assert current[0]["request_work_sum_s"] == 2
    assert current[0]["observed_parallel_makespan_s"] == pytest.approx(3.1)
    assert result["ideal_whole_request_bound"]["maximum_gain"] == 1
    assert result["nfr01_met"] is False
    assert result["generalized_learning_benefit_score"] == 0
    assert all(r["final_action_changed"] == 0 for r in result["fixed_point_rows"])
    assert any(r["overflow"] for r in result["fixed_point_rows"])
    assert all(
        r["evidence_scope"] == "synthetic_numerical_mechanics" for r in result["fixed_point_rows"]
    )
    value["resources_available"] = False
    assert h.reduce(value, [])["kv260_boundary_ready_score"] == 0


def test_custody_and_missing_operands(tmp_path, monkeypatch):
    """REQ-REPORT-8273: independently authenticate current branches and historical bytes."""
    monkeypatch.setattr(h.prior, "load", lambda root, raw: data())
    atomic_json(
        tmp_path / "results/experiment_8242_v712_independent_concurrent_service.json",
        dict(phase_spans=[], invocation_clocks={}),
    )
    missing = h.load(tmp_path, tmp_path / "raw")
    assert not any(missing["current"].values())
    assert missing["requests"] == [] and missing["board"]["custody_valid"]
    primitive = tmp_path / "primitive.json"
    atomic_json(
        primitive,
        dict(
            schema=h.PRIMITIVE_SCHEMA,
            current_heads=data()["heads"],
            current_samples=data()["samples"],
            counter_states=[],
            requests=data()["requests"],
            cold_costs=data()["cold_costs"],
            service_spans=[],
        ),
    )
    for name, (eid, suffix) in h.SOURCES.items():
        if name in {"tune", "seal"}:
            sealed = json.loads(primitive.read_bytes())
            sealed["requests"] = [
                dict(r, request_id=name, workload=name + "_serial") for r in sealed["requests"]
            ]
            sealed["cold_costs"] = [
                dict(r, workload=name + "_serial") for r in sealed["cold_costs"]
            ]
            primitive = tmp_path / (name + ".json")
            atomic_json(primitive, sealed)
        path = tmp_path / "results" / f"experiment_{eid}_{suffix}.json"
        payload = dict(
            experiment_id=eid,
            task_id=f"exp{eid}-fixture",
            honest_verdict="complete_null_fixture",
            verdict_class="null",
            required_checks_passed=True,
            flagged_adversarial=False,
            raw_shard_hashes=[reference(primitive)],
            kv260_obligation=dict(historical=data()["board"]),
            boundary_primitives_reference=reference(primitive),
            invocation_clocks={},
            phase_spans=[],
        )
        publish_primary(path, payload, lambda p: dict(passed=True))
    loaded = h.load(tmp_path, tmp_path / "raw2")
    assert all(loaded["current"].values())
    assert loaded["current_heads"] and loaded["counter_states"] == []
    assert h.reduce(loaded, h.precision(loaded))["verdict_class"] == "null"
    boundary_path = tmp_path / "results/experiment_8244_v712_kv260_decision_boundary.json"
    payload = json.loads(boundary_path.read_bytes())
    publish_primary(
        boundary_path,
        dict(payload, kv260_obligation=dict(historical={})),
        lambda p: dict(passed=True),
    )
    assert not h.load(tmp_path, tmp_path / "rawboard")["current"]["boundary"]
    primitive.write_text('{"schema": "wrong"}')
    for name, (eid, suffix) in h.CURRENT.items():
        path = tmp_path / "results" / f"experiment_{eid}_{suffix}.json"
        payload = json.loads(path.read_bytes())
        payload.update(
            raw_shard_hashes=[reference(primitive)],
            boundary_primitives_reference=reference(primitive),
        )
        publish_primary(path, payload, lambda p: dict(passed=True))
    assert not h.load(tmp_path, tmp_path / "raw3")["current"]["heads"]


def test_real_cli_and_tamper(tmp_path):
    """SCENARIO-REPORT-8273-REPLAY: fresh children reject rehashed summaries."""
    source, output = tmp_path / "input.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, data())
    result = run_cli("--date", "20261008", "--input", source, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert run_cli("--cold-replay", output).returncode == 0
    original = json.loads(output.read_bytes())
    for key, replacement in [("config", {}), ("reproducibility_checksum", "bad")]:
        value = deepcopy(original)
        value[key] = replacement
        atomic_json(output, value)
        assert run_cli("--cold-replay", output).returncode == 1
    value = deepcopy(original)
    value["ideal_whole_request_bound"]["maximum_gain"] = 2
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert run_cli("--cold-replay", output).returncode == 1
    atomic_json(output, original)
    primitive = Path(original["primitive_reference"]["path"])
    atomic_json(primitive, dict(fixed_point_rows=[]))
    value = deepcopy(original)
    value["primitive_reference"] = reference(primitive)
    value["raw_shard_hashes"] = [reference(Path(r["path"])) for r in value["raw_shard_hashes"]]
    atomic_json(output, value)
    assert run_cli("--cold-replay", output).returncode == 1
    assert run_cli("--date", "20261007").returncode == 2
    assert (
        run_cli("--input", source, "--output", h.ROOT / "results" / (h.NAME + ".json")).returncode
        == 1
    )
    source.write_text("[]")
    assert run_cli("--input", source, "--output", output).returncode == 1


def test_owned_and_terminal_failure_paths(tmp_path, monkeypatch):
    """REQ-REPORT-8273: reuse unchanged assertions for honest failure publication."""
    import test_kv260_evidence_cost_boundary_8258 as qualified_tests

    monkeypatch.setattr(qualified_tests, "h", h)
    monkeypatch.setattr(qualified_tests, "e", e)
    monkeypatch.setattr(qualified_tests, "data", data)
    qualified_tests.test_execution_failure_paths(tmp_path, monkeypatch)


def test_frozen_validation_commands(tmp_path):
    """REQ-REPORT-8273: required coverage, consumer and E2E commands are frozen."""
    plan = e.commands(tmp_path)
    assert {"coverage_combine", "e2e015", "e2e019", "affected_consumers"} <= {s.name for s in plan}
    assert len(e.validators(tmp_path / (h.NAME + ".json"))) == 3
