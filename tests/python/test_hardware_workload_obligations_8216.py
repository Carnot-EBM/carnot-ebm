"""REQ-REPORT-8216, REQ-VERIFY-8216: independent evidence stays scoped."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.reporting import hardware_workload_obligations_8216 as h
from carnot.reporting import hardware_obligations_execution_8216 as cli
from carnot.reporting.current_work_receipt import atomic_json


def panel():
    """A frozen zero head gives exact ties without granting acceptance."""
    rows = [
        dict(unit_id=str(i), source_cluster_id=str(i), x=[x], historical_x=[x])
        for i, x in enumerate([-1.0, 0.0, 1.0, 2.0])
    ]
    geometry = dict(mean=[0.0], scale=[1.0], centers=[[0.0]], width=1.0)
    head = dict(
        arm="energy",
        weights=[0.0, 0.0],
        temperature=1.0,
        geometry=geometry,
        fit_ids=["0", "1", "2"],
    )
    baseline = dict(geometry=geometry, weights=[10.0, 0.0], calibration=[0.0, 1.0])
    return dict(
        heads=[head],
        samples=rows,
        baseline=baseline,
        boards=[],
        learning={},
        service={},
        branches=dict(action=True, learning=False, service=False),
        checks=[],
        references=[],
        cited=[],
        fixture=True,
    )


def test_precision_permissions_and_missing():
    """SCENARIO-VERIFY-8216-PRECISION: conversion cannot grant acceptance."""
    data = panel()
    data["samples"].append(
        dict(unit_id="missing", source_cluster_id="missing", x=None, historical_x=None)
    )
    rows = h.precision(data)
    assert len(rows) == 15
    assert sum(r["status"] == "excluded" for r in rows) == 3
    assert all(r["final_action"] != "accept" for r in rows if r["status"] == "completed")
    assert all(
        r["fallback"] for r in rows if r["status"] == "completed" and r["precision"] != "float64"
    )
    assert all(r["numerator"] == 0 for r in rows if r["status"] == "completed")
    data = panel()
    data["heads"][0]["weights"] = [-8.0, 1.0]
    data["baseline"]["weights"] = [-10.0, 0.0]
    assert any(r["final_action"] == "accept" for r in h.precision(data))
    data["samples"][0]["historical_x"] = None
    assert h.precision(data)[0]["status"] == "excluded"
    for arm in ["additive", "logistic"]:
        data["heads"][0]["arm"] = arm
        assert len(h.precision(data)) == 12


def service():
    """Small disjoint clocks let independent arithmetic verify the whole workload."""
    branch = dict(
        unit_id="q",
        arm="python_durable_batch",
        start_ns=0,
        conversion_end_ns=10,
        pending_end_ns=12,
        boundary_crossing_start_ns=12,
        boundary_crossing_end_ns=22,
        serialization_start_ns=22,
        serialization_end_ns=24,
        commit_start_ns=24,
        commit_end_ns=44,
        readout_start_ns=44,
        end_ns=50,
    )
    return dict(
        request_rows=[
            dict(
                request_id="q",
                source_cluster_id="s",
                status="completed",
                acquisition_ns=100,
                arms=[branch],
            )
        ],
        cold_start_cost_s=100e-9,
        shared_recording_and_checkpoint_overhead_s=20e-9,
    )


def test_costs_and_branch_independence(tmp_path):
    """SCENARIO-VERIFY-8216-COSTS: full acquisition dominates the removable kernel."""
    rows, bounds = h.service_costs(service())
    assert bounds[0]["total_ns"] == pytest.approx(270)
    assert bounds[0]["serial_fraction"] == pytest.approx(260 / 270)
    assert bounds[0]["maximum_speedup"] == pytest.approx(270 / 260)
    assert not bounds[0]["supports_100x"]
    assert rows[0]["transfer_ns"] == 10
    bad = service()
    bad["request_rows"][0]["arms"][0]["end_ns"] = 30
    with pytest.raises(ValueError, match="clock_partition"):
        h.service_costs(bad)
    data = panel()
    data["service"] = service()
    data["branches"]["service"] = True
    result = h.reduce(data, h.precision(data), [])
    assert result["hardware_boundary_ready_score"] == 1
    assert result["branch_readiness"]["learning"]["status"] == "blocked"
    assert result["workload_rows"]
    probe = h.storage_probe(b"actual bytes", tmp_path / "storage.bin", "learning")
    assert probe["bytes"] == 12 and probe["fsync_ns"] > 0 and probe["readout_ns"] > 0
    assert h.reduce(panel(), [], [])["verdict_class"] == "circular_positive"
    data["branches"] = dict(action=False, learning=False, service=False)
    assert h.reduce(data, [], [])["verdict_class"] == "blocked"
    mismatch = h.precision(panel())
    mismatch[0]["numerator"] = 1
    assert h.reduce(panel(), mismatch, [])["verdict_class"] == "disqualified"


def run_cli(tmp_path, *args):
    """The standalone child must work without the caller's checkout imports."""
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(h.ROOT / h.CLI), *map(str, args)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_private_cli_and_cold_tamper(tmp_path):
    """SCENARIO-REPORT-8216-CLI: real publication and replay reject tampered claims."""
    source = tmp_path / "fixture.json"
    atomic_json(source, panel())
    output = tmp_path / (h.NAME + ".json")
    result = run_cli(tmp_path, "--input", source, "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["MODEL_SPECS"] == []
    assert value["current_device_execution_count"] == 0
    assert run_cli(tmp_path, "--cold-replay", output).returncode == 0
    value["completed_count"] += 1
    atomic_json(output, value)
    assert run_cli(tmp_path, "--cold-replay", output).returncode == 1
    assert run_cli(tmp_path, "--date", "20261007").returncode == 2
    assert (
        run_cli(
            tmp_path, "--input", source, "--output", h.ROOT / "results" / output.name
        ).returncode
        == 1
    )
    missing = tmp_path / "missing"
    output.unlink()
    assert run_cli(tmp_path, "--root", missing, "--output", output).returncode == 0
    assert json.loads(output.read_text())["verdict_class"] == "blocked"


def test_authentication_and_validation_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8216: authenticate branches separately and preserve owned failures."""
    data = panel()
    data["fixture"] = False

    def authenticated(path, eid, score, raw, target):
        value = dict(
            experiment_id=eid,
            task_id=f"exp{eid}-{h.PRODUCERS[next(n for n, v in h.PRODUCERS.items() if v[0] == eid)][1].replace('_', '-')}",
            verifier_is_oracle=False,
        )
        if eid == 8208:
            heads = tmp_path / "heads.json"
            atomic_json(heads, dict(heads=data["heads"], baseline=data["baseline"]))
            samples = tmp_path / "samples.json"
            atomic_json(samples, dict(evidence=dict(rows=data["samples"])))
            value.update(
                frozen_heads_path=str(heads),
                frozen_heads_sha256=h.sha256_file(heads),
                measurement_reference=h.reference(samples),
            )
        elif eid == 8211:
            trajectory = tmp_path / "trajectory.json"
            atomic_json(trajectory, dict(states=[]))
            value.update(
                trajectory_path=str(trajectory), trajectory_sha256=h.sha256_file(trajectory)
            )
        else:
            value.update(service())
            primitive = tmp_path / "result.json"
            atomic_json(primitive, dict(work=dict(requests=value["request_rows"])))
            value["raw_shard_hashes"] = [h.reference(primitive)]
        return value, True

    monkeypatch.setattr(h, "authenticate", authenticated)
    monkeypatch.setattr(h, "PINS", dict.fromkeys(h.PINS))
    loaded = h.load(tmp_path, tmp_path / "raw")
    assert all(loaded["branches"].values())
    assert h.reduce(loaded, h.precision(loaded), [])["amdahl_bounds"][-1]["maximum_speedup"] is None
    monkeypatch.setattr(h, "authenticate", lambda *args: ({}, False))
    assert not any(h.load(tmp_path, tmp_path / "raw2")["branches"].values())
    private = tmp_path / "plan"
    private.mkdir()
    plan = cli.commands(private)
    assert any(s.name == "changed_module_coverage_report" for s in plan)
    assert any("--strict" in s.argv for s in plan)
    monkeypatch.setattr(cli.h, "load", lambda *args: data)
    monkeypatch.setattr(cli.h, "precision", lambda *args: [])
    monkeypatch.setattr(cli, "commands", lambda *args: [])
    monkeypatch.setattr(cli, "execute", lambda specs, raw: [dict(passed=False, normal_exit=True)])
    monkeypatch.setattr(
        cli, "publish_primary", lambda output, value, validator: dict(primary_path=str(output))
    )
    output = tmp_path / "normal" / (h.NAME + ".json")
    assert cli.main(["--output", str(output)]) == 1


def test_service_publication_contract(tmp_path):
    """REQ-REPORT-8216: the service producer's existing receipt authenticates bytes."""
    from carnot.reporting.primary_publication import publish_primary

    source = tmp_path / "results" / "experiment_8214_v709_prospective_service_measurement.json"
    raw = source.parent / "raw" / source.stem / "invocations" / "test"
    raw.mkdir(parents=True)
    value = dict(
        experiment_id=8214,
        task_id="exp8214-prospective-service-measurement",
        service_measurement_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        honest_verdict="complete_null_service",
        verdict_class="null",
        raw_path=str(raw),
    )
    publication = publish_primary(source, value, lambda _: dict(passed=True))
    atomic_json(raw / "publication_receipt.json", dict(publication=publication))
    data = dict(checks=[], references=[])
    _, valid = h.authenticate(
        source, 8214, "service_measurement_ready_score", tmp_path / "copies", data
    )
    assert valid
    atomic_json(
        raw / "publication_receipt.json", dict(publication=dict(publication, primary_sha256="bad"))
    )
    assert not h.authenticate(
        source, 8214, "service_measurement_ready_score", tmp_path / "copies2", data
    )[1]


def test_failed_service_costs_preserve_unknown():
    """SCENARIO-VERIFY-8216-COSTS: failed conversion costs cannot become zero."""
    value = service()
    value["request_rows"].append(
        dict(
            request_id="failed",
            source_cluster_id="f",
            status="failed",
            acquisition_ns=200,
            arms=[dict(arm="python_durable_batch", start_ns=100, end_ns=120)],
        )
    )
    rows, bounds = h.service_costs(value)
    assert rows[1]["fsync_ns"] is None
    assert bounds[0]["total_ns"] == pytest.approx(490)
    assert bounds[0]["unpartitioned_failed_downstream_ns"] == 20
    assert rows[1]["status"] == "failed"


def test_current_inputs_and_failure_operands(tmp_path, monkeypatch):
    """REQ-REPORT-8216: original board receipts and producer primitives remain immutable."""
    loaded = h.load(h.ROOT, tmp_path / "current")
    assert loaded["branches"]["action"]
    assert {b["board"] for b in loaded["boards"]} == {"KV260", "PolarFire", "GateMate"}
    assert loaded["branches"]["service"]
    rows, bounds = h.service_costs(loaded["service"])
    assert len(rows) == 48 and len(bounds) == 2
    assert all(b["maximum_speedup"] < 100 for b in bounds)
    monkeypatch.setattr(h, "PINS", dict.fromkeys(h.PINS))
    monkeypatch.setattr(
        h,
        "authenticate",
        lambda path, eid, score, raw, data: (
            dict(
                task_id=f"exp{eid}-{h.PRODUCERS[next(n for n, v in h.PRODUCERS.items() if v[0] == eid)][1].replace('_', '-')}"
            ),
            True,
        ),
    )
    broken = h.load(tmp_path / "absent", tmp_path / "broken")
    assert not any(broken["branches"].values())
    assert any(c["artifact_field"] == "primitive_contract" for c in broken["checks"])
    monkeypatch.setattr(Path, "read_bytes", lambda _: b"changed")
    with pytest.raises(ValueError, match="storage_parity"):
        h.storage_probe(b"expected", tmp_path / "store", "learning")


def test_owned_failure_and_replay_mutations(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8216-CLI: owned failures disqualify and rehashed claims fail replay."""
    data = panel()
    payload = tmp_path / "learning.json"
    payload.write_bytes(b"actual storage source")
    data["learning"] = dict(storage_source=h.reference(payload))
    data["branches"]["learning"] = True
    monkeypatch.setattr(h, "load", lambda *args: deepcopy(data))
    monkeypatch.setattr(cli, "commands", lambda _: [])

    def receipts(specs, raw):
        return [
            dict(
                name="control",
                passed=raw.name != "validation",
                normal_exit=True,
                exit_code=0 if raw.name != "validation" else 1,
            )
        ]

    monkeypatch.setattr(cli, "execute", receipts)
    output = tmp_path / (h.NAME + ".json")
    assert cli.main(["--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "disqualified"
    assert not value["required_checks_passed"]
    assert cli.replay(output)["passed"]
    original = deepcopy(value)
    for key, replacement, error in [
        ("config", {}, "configuration_drift"),
        ("reproducibility_checksum", "wrong", "checksum_drift"),
    ]:
        value = dict(original, **{key: replacement})
        atomic_json(output, value)
        with pytest.raises(ValueError, match=error):
            cli.replay(output)
    value = deepcopy(original)
    value["precision_rows"][0]["probability_reference"] += 0.1
    atomic_json(output, value)
    with pytest.raises(ValueError, match="precision_drift"):
        cli.replay(output)
    with monkeypatch.context() as scope:
        scope.setattr(h, "precision", lambda _: value["precision_rows"])
        with pytest.raises(ValueError, match="primitive_drift"):
            cli.replay(output)
    atomic_json(output, original)
    health_log = tmp_path / "health.log"
    health_log.write_bytes(b"real test-controlled health output")
    health = tmp_path / "health.json"
    atomic_json(
        health,
        dict(
            receipts=[
                dict(
                    stdout_path=str(health_log),
                    stderr_path=str(health_log),
                    stdout_sha256=h.sha256_file(health_log),
                    stderr_sha256=h.sha256_file(health_log),
                )
            ]
        ),
    )
    assert cli.main(["--output", str(output), "--repository-health-receipt", str(health)]) == 0
    monkeypatch.setattr(cli, "execute", lambda *args: [dict(passed=False, normal_exit=True)])
    assert cli.main(["--output", str(output)]) == 1


def test_learning_clock_obligation():
    """SCENARIO-VERIFY-8216-COSTS: absent original timers are exact operand failures."""
    data = panel()
    data["branches"]["learning"] = True
    value = h.reduce(data, h.precision(data), [])
    gate = value["cost_obligation_checks"][0]
    assert gate["upstream"] == "exp8211"
    assert gate["artifact_field"] == "trajectory.states.component_clocks"
    assert gate["observed"] is None and gate["passed"] is False
    assert value["hardware_boundary_ready_score"] == 1


def test_numeric_envelope_and_permission_gates():
    """SCENARIO-VERIFY-8216-PRECISION: failed bounds or permission cannot qualify."""
    data = panel()
    measured = h.precision(data)
    measured[0]["interval_radius"] = -1
    assert h.reduce(data, measured, [])["verdict_class"] == "disqualified"
    measured = h.precision(data)
    measured[0]["final_action"] = "accept"
    assert h.reduce(data, measured, [])["verdict_class"] == "disqualified"


def test_service_source_drift_and_missing_publication(tmp_path, monkeypatch):
    """REQ-REPORT-8216: a byte-bound summary cannot replace different primitive requests."""
    missing = tmp_path / "missing.json"
    data = dict(checks=[], references=[])
    assert not h.authenticate(missing, 8214, "service_measurement_ready_score", tmp_path, data)[1]
    source = tmp_path / "results" / "experiment_8214_v709_prospective_service_measurement.json"
    atomic_json(source, dict(raw_path=str(tmp_path / "absent")))
    assert not h.authenticate(
        source, 8214, "service_measurement_ready_score", tmp_path / "copies", data
    )[1]
    primitive = tmp_path / "result.json"
    atomic_json(primitive, dict(work=dict(requests=[])))
    value = dict(
        task_id="exp8214-prospective-service-measurement",
        raw_shard_hashes=[h.reference(primitive)],
        **service(),
    )
    monkeypatch.setattr(h, "PINS", dict(h.PINS, **{}))
    monkeypatch.setitem(h.PINS, 8214, h.sha256_file(source))
    monkeypatch.setattr(
        h,
        "authenticate",
        lambda path, eid, score, raw, target: (value, True) if eid == 8214 else ({}, False),
    )
    loaded = h.load(tmp_path, tmp_path / "drift")
    assert not loaded["branches"]["service"]
    assert any(c.get("observed") == "service_request_primitive_drift" for c in loaded["checks"])
