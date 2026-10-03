"""REQ-REPORT-8078: complete partition costs and current publication gates."""

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_8078_v699_feature_cache_core as e


def test_contract_and_external_operands(tmp_path):
    """SCENARIO-REPORT-8078-COST: authenticate matrix and retain exact failures."""
    data, failures = e.load_inputs(e.ROOT, tmp_path / "inputs")
    assert not failures
    assert len(data["service_matrix"]["rows"]) == 2520
    assert len(e.schedule(data)) == 1260
    assert e.load_inputs(tmp_path / "absent", tmp_path / "blocked")[1]
    broken = copy.deepcopy(data)
    broken["service_matrix"]["warmups"] = 4
    with pytest.raises(ValueError, match="service_matrix"):
        e.schedule(broken)


def test_loaded_quartets_and_censoring(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8078-PARITY/COST: actual E2E-003/004 transactions."""
    from test_guarded_transaction_8053 import data as fixture

    data = fixture.__wrapped__()
    data["cases"] = [data["cases"][0]]
    data["service_matrix"] = e.sealed.service_matrix()
    native, receipt = e.legacy.old.prior.load_library(data["library_reference"], tmp_path)
    plan = e.schedule(data)
    first = [
        r
        for r in plan
        if r["repetition"] in (-1, 0)
        and r["condition"] == data["cases"][0]["arm"]
        and r["transaction_class"] == data["cases"][0]["class"]
    ]
    monkeypatch.setattr(e, "schedule", lambda data: first)
    value = e.measure(data, native, tmp_path / "measured")
    assert value["completed_count"] == 12
    assert value["excluded_count"] == 12
    assert all(r["passed"] for r in value["parity_rows"])
    assert value["complete_pair_count"] == 3
    assert len(list((tmp_path / "measured/pairs").glob("*.json"))) == 6
    assert all(r["native_calls"] > 0 for r in value["rows"] if r["path_arm"] == "native")
    assert all(r["transaction_ns"] == sum(r["components"].values()) for r in value["rows"])
    assert all(r["source_bytes_preserved"] for r in value["all_miss_identity_rows"])
    candidate = e.base([])
    candidate.update(
        value, loaded_library_receipt=receipt, raw_directory=str(tmp_path / "measured")
    )
    candidate["raw_shard_hashes"] = [e.legacy.reference(tmp_path / "measured/observations.json")]
    path = tmp_path / "candidate.json"
    e.legacy.atomic_json(path, candidate)
    assert e.replay(path)["passed"]
    candidate["complete_workload_ratios"] = []
    e.legacy.atomic_json(path, candidate)
    with pytest.raises(ValueError, match="reduction_drift"):
        e.replay(path)
    monkeypatch.setitem(e.CONFIG, "budget_s", -1)
    censored = e.measure(data, native, tmp_path / "censored")
    assert censored["censored_count"] == 12
    assert censored["complete_pair_count"] == 0


def test_speed_gate_uses_every_lower_bound():
    """REQ-REPORT-8078: complete slow service has readiness without speed credit."""
    rows = [dict(mode=m, lower_95=2.0) for m in e.MODES]
    assert e.speed_gate(rows)
    assert not e.speed_gate([])
    rows[0]["lower_95"] = 0.94
    assert not e.speed_gate(rows)
    rows[0]["lower_95"] = 2
    rows[1]["lower_95"] = 1.2
    assert not e.speed_gate(rows)


def test_replay_mutations_and_failed_prerequisite(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8078-TERMINAL: altered cache and quartet evidence reject."""
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *args: dict(report=dict(passed=False)))
    assert any(f["field"] == "report.passed" for f in e.load_inputs(e.ROOT, tmp_path / "bad")[1])
    value = e.base([])
    value["raw_directory"] = str(tmp_path / "replay")
    path = tmp_path / "replay.json"
    e.legacy.atomic_json(path, value)
    pair = tmp_path / "replay/pairs/0000.json"
    e.legacy.atomic_json(pair, dict(rows=[]))
    with pytest.raises(ValueError, match="quartet_drift"):
        e.replay(path)
    pair.unlink()
    cache = e.legacy.c.FeatureCache(tmp_path / "replay/cache/test.sqlite")
    cache.get(
        dict(
            family_id="test",
            source_bytes=b"Water is wet.".hex(),
            answer_bytes=b"Water is wet.".hex(),
        )
    )
    cache.db.execute("UPDATE features SET digest='forged'")
    cache.db.commit()
    cache.close()
    with pytest.raises(ValueError, match="persisted_cache_drift"):
        e.replay(path)


def test_private_cli_and_publication_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8078-TERMINAL: real process CLI success, blocked and replay."""
    import runpy
    import sys

    from test_guarded_transaction_8053 import data as fixture

    data = fixture.__wrapped__()
    monkeypatch.chdir(tmp_path)
    data["service_matrix"] = e.sealed.service_matrix()
    source = tmp_path / "fixture.json"
    e.legacy.atomic_json(source, data)
    output = tmp_path / (e.NAME + ".json")
    before = e.legacy.NAME
    assert e.validation_plan(tmp_path / "validation")
    assert e.legacy.NAME == before
    assert (
        e.main(["--fixture-input", str(source), "--output", str(output), "--validation-worker"])
        == 0
    )
    assert e.main(["--cold-replay", str(output)]) == 0
    assert e.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "blocked"
    monkeypatch.setattr(sys, "argv", [e.SCRIPT, "--cold-replay", str(output)])
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(e.ROOT / e.SCRIPT), run_name="__main__")
    assert exited.value.code == 0
    for ready, pairs, repetitions, expected in [
        (0, 0, 30, 0),
        (1, 269, 30, 0),
        (1, 270, 29, 0),
        (1, 270, 30, 1),
    ]:
        value = e.base([])
        value.update(
            feature_service_ready_score=ready,
            complete_pair_count=pairs,
            complete_workload_ratios=[dict(paired_repetitions=repetitions)] * 18,
            field_principles={},
        )
        e.publish(output, value, lambda p: dict(passed=True))
        assert json.loads(output.read_text())["cache_core_ready_score"] == expected


def test_owned_manifest_and_independent_reducer(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8078-TERMINAL: freeze true child budgets and reject reductions."""
    spec = e.legacy.CommandSpec(
        "child",
        (
            str(e.ROOT / ".venv/bin/python"),
            "-c",
            "print('real child',flush=True)",
            "--raw",
            str(tmp_path),
        ),
        "measurement",
        960,
    )
    e.legacy.atomic_json(
        tmp_path / "methods.json", dict(measurement_command={}, dependency_hashes={})
    )
    receipts = e.run_commands(tmp_path, [spec], log_dir=tmp_path / "logs", heartbeat_s=1)
    assert receipts[0]["exit_code"] == 0
    assert (
        json.loads((tmp_path / "methods.json").read_text())["measurement_command"]["timeout_s"]
        == 1860
    )
    plan = e.sealed.service_matrix()
    monkeypatch.setattr(e.sealed, "service_matrix", lambda: dict(plan, repetitions=29))
    assert e.load_inputs(e.ROOT, tmp_path / "altered")[1]
    primary = json.loads((e.ROOT / "results/experiment_8072_v699_sealed_methods.json").read_text())
    primary["service_matrix"]["warmups"] = 4
    private = tmp_path / "mutated/results/experiment_8072_v699_sealed_methods.json"
    e.legacy.atomic_json(private, primary)
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *args: dict(report=dict(passed=True)))
    monkeypatch.setitem(e.ORIGINAL, "load_inputs", lambda *args: (dict(references=[]), []))
    assert (
        "sealed_service_matrix_drift"
        in e.load_inputs(tmp_path / "mutated", tmp_path / "drift")[1][0]["observed"]
    )
    monkeypatch.setitem(e.ORIGINAL, "replay", lambda path: dict(passed=True))
    value = e.base([])
    value["raw_directory"] = str(tmp_path / "empty")
    rows = [
        dict(
            mode="warm",
            condition="c",
            transaction_class="accepted",
            path_arm="python",
            status="completed",
            cached=cached,
            repetition=0,
            transaction_ns=cost,
        )
        for cached, cost in [(False, 20), (True, 10)]
    ]
    value["rows"] = rows
    value["complete_workload_ratios"] = e.reduce_rows(rows)
    value["complete_workload_ratios"][0]["numerator"] += 1
    path = tmp_path / "reduction.json"
    e.legacy.atomic_json(path, value)
    with pytest.raises(ValueError, match="independent_reduction_drift"):
        e.replay(path)
