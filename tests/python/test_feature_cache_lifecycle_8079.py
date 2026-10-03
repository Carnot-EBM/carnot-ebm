"""REQ-REPORT-8079: lifecycle evidence must include complete costs and restarts."""

import copy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_8079_v699_feature_cache_lifecycle as e


def test_contract_and_mutations(tmp_path):
    """SCENARIO-REPORT-8079-COST/PARITY: freeze operands and reject changed identity."""
    data, failures = e.load_inputs(e.ROOT, tmp_path / "inputs")
    assert not failures
    plan = e.schedule(data)
    assert len(plan) == 1260
    assert {r["mode"] for r in plan} == set(e.MODES)
    changed = copy.deepcopy(data)
    changed["service_matrix"]["repetitions"] = 29
    with pytest.raises(ValueError, match="service_matrix"):
        e.schedule(changed)
    assert e.load_inputs(tmp_path / "absent", tmp_path / "blocked")[1]
    assert all(r["passed"] for r in e.identity_checks(tmp_path / "identity"))


def test_loaded_lifecycle_quartets(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8079-COST/PARITY: real loaded calls and executable restart."""
    from test_guarded_transaction_8053 import data as fixture

    data = fixture.__wrapped__()
    data["cases"] = [data["cases"][0]]
    data["cases"][0].update(updates=[0, 1], labels=[0, 1])
    data["service_matrix"] = e.core.sealed.service_matrix()
    native, receipt = e.legacy.old.prior.load_library(data["library_reference"], tmp_path)
    full = e.schedule(data)
    selected = [
        r
        for r in full
        if r["repetition"] in (-1, 0)
        and r["condition"] == data["cases"][0]["arm"]
        and r["transaction_class"] == data["cases"][0]["class"]
    ]
    monkeypatch.setattr(e, "schedule", lambda data: selected)
    value = e.measure(data, native, tmp_path / "measured")
    assert value["completed_count"] == value["excluded_count"] == 12
    assert value["complete_pair_count"] == 3
    assert len(list((tmp_path / "measured/pairs").glob("*.json"))) == 6
    assert all(r["passed"] for r in value["parity_rows"])
    assert all(
        r["passed"] and r["parent_pid"] != r["child_pid"] for r in value["process_restart_rows"]
    )
    assert all(r["transaction_ns"] == sum(r["components"].values()) for r in value["rows"])
    assert all(r["native_calls"] > 0 for r in value["rows"] if r["path_arm"] == "native")
    assert value["population_rows"] and value["invalidation_rows"]
    for restart in (tmp_path / "measured/restart").glob("*.input.json"):
        assert e.restart_check(json.loads(restart.read_text()))["passed"]
    candidate = e.base([])
    candidate.update(
        value, loaded_library_receipt=receipt, raw_directory=str(tmp_path / "measured")
    )
    candidate["raw_shard_hashes"] = [e.legacy.reference(tmp_path / "measured/observations.json")]
    path = tmp_path / "candidate.json"
    e.legacy.atomic_json(path, candidate)
    assert e.replay(path)["passed"]
    real_replay = e.ORIGINAL["replay"]
    monkeypatch.setitem(e.ORIGINAL, "replay", lambda path: dict(passed=True))
    original = copy.deepcopy(candidate)
    for field, message in [
        ("population_rows", "lifecycle_evidence_drift"),
        ("complete_service_ratios", "population_reduction_drift"),
        ("interval", "service_interval_drift"),
        ("process_restart_rows", "restart_receipt_drift"),
        ("identity_check_rows", "identity_control_drift"),
        ("missing_restart", "restart_receipt_drift"),
    ]:
        broken = copy.deepcopy(original)
        if field == "population_rows":
            broken[field] = []
        elif field == "complete_service_ratios":
            broken[field][0]["complete_service_denominator"] += 1
        elif field == "interval":
            broken["complete_service_ratios"][0]["lower_95"] += 1
        elif field == "identity_check_rows":
            broken[field] = []
        elif field == "missing_restart":
            broken["process_restart_rows"] = []
        else:
            broken[field][0]["parent_pid"] = broken[field][0]["child_pid"]
        e.legacy.atomic_json(path, broken)
        if field != "population_rows":
            e.legacy.atomic_json(tmp_path / "measured/observations.json", broken)
        with pytest.raises(ValueError, match=message):
            e.replay(path)
    e.legacy.atomic_json(tmp_path / "measured/observations.json", value)
    monkeypatch.setitem(e.ORIGINAL, "replay", real_replay)
    candidate["complete_workload_ratios"] = []
    e.legacy.atomic_json(path, candidate)
    with pytest.raises(ValueError, match="reduction_drift"):
        e.replay(path)
    monkeypatch.setitem(e.CONFIG, "budget_s", -1)
    assert e.measure(data, native, tmp_path / "censored")["censored_count"] == 12
    monkeypatch.setitem(e.CONFIG, "budget_s", 1800)
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: [dict(passed=False)])
    with pytest.raises(ValueError, match="restart_child_failed"):
        e.measure(data, native, tmp_path / "failed_child")


def test_readiness_and_private_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8079-TERMINAL: no readiness for incomplete or unsafe cells."""
    monkeypatch.chdir(tmp_path)
    assert e.validation_plan(tmp_path / "validation")
    output = tmp_path / (e.NAME + ".json")
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
            identity_check_rows=[
                dict(operand=operand, passed=True)
                for operand in ("source_bytes", "answer_bytes", "config", "schema", "corruption")
            ],
            rows=[
                dict(
                    unit=str(i),
                    mode="restart",
                    status="completed",
                    path_arm="native",
                    checkpoint={},
                )
                for i in range(420)
            ],
            process_restart_rows=[
                dict(
                    unit=str(i),
                    passed=True,
                    parent_pid=1,
                    child_pid=2,
                    receipt=dict(passed=True, exit_code=0),
                    native_calls=1,
                    checkpoint={},
                )
                for i in range(420)
            ],
            required_checks_passed=True,
        )
        e.publish(output, value, lambda path: dict(passed=True))
        assert json.loads(output.read_text())["cache_lifecycle_ready_score"] == expected
    assert not e.speed_gate([])
    assert e.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "blocked"
    assert e.main(["--cold-replay", str(output)]) == 0
    for action in ("populate", "restart", "mutate"):
        assert e.main(["--cache-probe", action, "--cache", str(tmp_path / "cache.sqlite")]) == 0
    monkeypatch.setattr(sys, "argv", [e.SCRIPT, "--cold-replay", str(output)])
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(e.ROOT / e.SCRIPT), run_name="__main__")
    assert exited.value.code == 0
    health, first = tmp_path / "health.json", tmp_path / "first.log"
    log = tmp_path / "health.log"
    log.write_text("private repository health fixture")
    first.write_text("private tests first fixture")
    e.legacy.atomic_json(health, dict(log_path=str(log), passed=False))
    monkeypatch.setattr(e, "HEALTH_PATH", health)
    monkeypatch.setattr(e, "FIRST_PATH", first)
    value = e.base([])
    value.update(raw_directory=str(tmp_path / "published"), field_principles={})
    e.publish(output, value, lambda path: dict(passed=True))
    assert json.loads(output.read_text())["current_repository_health"]["passed"] is False


def test_restart_rejects_corruption(tmp_path):
    """SCENARIO-REPORT-8079-PARITY: child checks exact cache bytes and journal."""
    cache = e.legacy.c.FeatureCache(tmp_path / "cache.sqlite")
    cache.get(
        dict(
            family_id="private",
            source_bytes=b"Water is wet.".hex(),
            answer_bytes=b"Water is wet.".hex(),
        )
    )
    snapshot = e.cache_snapshot(cache)
    cache.close()
    payload = dict(
        cache=str(tmp_path / "cache.sqlite"),
        snapshot=snapshot,
        checkpoint=None,
        library=None,
        arm="python",
    )
    assert e.restart_check(payload)["passed"]
    payload["snapshot"] = []
    with pytest.raises(ValueError, match="restart_cache_drift"):
        e.restart_check(payload)
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    e.legacy.atomic_json(source, dict(payload, snapshot=snapshot, output=str(output)))
    assert e.main(["--restart-input", str(source)]) == 0
    assert json.loads(output.read_text())["passed"]


def test_restart_state_and_manifest_failures(tmp_path):
    """SCENARIO-REPORT-8079-TERMINAL: bind child budgets and reject altered state."""
    import sqlite3

    from test_guarded_transaction_8053 import data as fixture

    data = fixture.__wrapped__()
    native, receipt = e.legacy.old.prior.load_library(data["library_reference"], tmp_path)
    row = e.legacy.old.m.transaction(
        data, data["cases"][0], native, "native", tmp_path / "state.json"
    )
    payload = dict(
        cache=None, snapshot=[], checkpoint=row["checkpoint"], library=receipt, arm="native"
    )
    assert e.restart_check(payload)["passed"]
    file = Path(row["checkpoint"]["path"])
    state = json.loads(file.read_text())
    state["head"]["parameters"][0] += 1
    e.legacy.atomic_json(file, state)
    payload["checkpoint"] = e.legacy.reference(file)
    with pytest.raises(ValueError, match="restart_journal_drift"):
        e.restart_check(payload)
    with sqlite3.connect(file.with_suffix(".sqlite")) as db:
        db.execute("UPDATE events SET payload=?", (file.read_bytes(),))
    with pytest.raises(ValueError, match="restart_state_drift"):
        e.restart_check(payload)
    e.legacy.atomic_json(
        tmp_path / "methods.json", dict(measurement_command={}, dependency_hashes={})
    )
    spec = e.legacy.CommandSpec(
        "private_measurement",
        (
            str(e.ROOT / ".venv/bin/python"),
            "-c",
            "print('normal exit',flush=True)",
            "--raw",
            str(tmp_path),
        ),
        "measurement",
        960,
    )
    assert e.run_commands(tmp_path, [spec], log_dir=tmp_path / "logs")[0]["passed"]
    assert (
        json.loads((tmp_path / "methods.json").read_text())["measurement_command"]["timeout_s"]
        == 1860
    )
