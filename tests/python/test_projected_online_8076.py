"""REQ-REPORT-8076: private causal, recovery and publication evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import numpy as np
import pytest

from carnot import experiment_8076_v699_projected_online_learning as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import projected_online_8076 as m
from test_fresh_feedback_8064 import data


def test_trajectory_and_future_mutation(tmp_path):
    """SCENARIO-REPORT-8076-CAUSAL: released roles, equal budgets and clocks."""
    value = data()
    result = m.measure(value, tmp_path / "run")
    assert result == m.reduce(tmp_path / "run")
    assert len(result["issued_prediction_rows"]) == 1024
    assert len(result["final_head_seals"]) == 4
    assert result["constraint_addition_rows"] and result["constraint_eviction_rows"]
    assert result["projection_rows"] and result["raw_candidate_rows"]
    budgets = result["per_arm_gradient_label_operation_budgets"]
    assert {r["gradients"] for r in budgets if r["arm"] != "frozen"} == {12}
    assert len({r["label_operations"] for r in budgets if r["arm"] != "frozen"}) == 1
    assert all(r["release_slot"] == r["slot"] + 20 for r in result["feedback_release_rows"])
    assert all(r["receipt"]["role"] == "update" for r in result["constraint_addition_rows"])
    assert all(r["release_frontier"] < r["slot"] for r in result["candidate_commit_rows"])
    used = result["admission_consumption_rows"]
    assert len({(r["seed"], r["slot"]) for r in used}) == len(used)
    for candidate in result["candidate_commit_rows"]:
        assert all(i + 20 < candidate["slot"] for i in candidate["update_slots"])
    assert all(
        len({r["slot"] for r in result["durable_commit_rows"] if r["candidate_slot"] == slot}) == 1
        for slot in (64, 128, 192)
    )
    changed = deepcopy(value)
    changed["labels"]["110"] ^= 1
    other = m.measure(changed, tmp_path / "changed")
    for field in ("issued_prediction_rows", "candidate_commit_rows", "gradient_rows"):
        cutoff = 130 if field == "issued_prediction_rows" else 128
        assert [r for r in result[field] if r["slot"] <= cutoff] == [
            r for r in other[field] if r["slot"] <= cutoff
        ]
    assert m.measure(value, tmp_path / "run") == result
    with pytest.raises(ValueError, match="prefix_drift"):
        m.measure(changed, tmp_path / "run")
    changed["sources"][0]["q"] = 0.8
    with pytest.raises(ValueError, match="input_drift"):
        m.measure(changed, tmp_path / "run")
    with pytest.raises(TimeoutError, match="numerical_budget"):
        m.measure(value, tmp_path / "timeout", budget_s=-1)
    with sqlite3.connect(tmp_path / "run/seed-101/ledger.sqlite") as db:
        db.execute("DROP TRIGGER forbid_UPDATE")
        db.execute("UPDATE events SET payload='{}' WHERE seq=0")
    with pytest.raises(ValueError):
        m.reduce(tmp_path / "run")


def test_harmful_guard_and_reset():
    """SCENARIO-REPORT-8076-GUARD: harmful and infeasible endpoints earn no credit."""
    head = dict(parameters=[0.0], calibration=[0.0, 1.0])
    x = np.tile([-1.0, 1.0], 6).reshape(12, 1)
    y = np.tile([0, 1], 6)
    initial = np.zeros(1)
    good = np.ones(1)
    assert m.admit(head, initial, good, initial, x, y, [], False)["alpha"] == 1
    assert m.admit(head, initial, -20 * good, initial, x, y, [], False)["alpha"] == 0
    bad = -20 * good
    rows = [dict(normal=[1.0, 0.0], rhs=0.0)]
    result = m.admit(head, bad, -30 * good, initial, x, y, rows, True)
    assert result["alpha"] is None and result["fallback"] == "initial"
    assert result["parameters"] == [0.0]
    assert (
        m.admit(head, initial, good, initial, x, np.zeros(12), [], True)["fallback"] == "incumbent"
    )


def test_deferrals_and_label_contract(tmp_path):
    """SCENARIO-REPORT-8076-CAUSAL: incomplete blocks cannot buy replacement evidence."""
    value = data()
    value["sources"][0]["public_eligible"] = False
    value["labels"]["1"] = None
    for row in value["sources"]:
        if m.partition(row["source_cluster_id"]) == 0:
            value["labels"][row["family_id"]] = 0
    result = m.measure(value, tmp_path / "oneclass")
    assert result["excluded_count"] == 2
    assert all(r["alpha"] is None for r in result["durable_commit_rows"] if r["arm"] in m.GUARDED)
    for row in value["sources"]:
        row["eligible"] = False
    assert m.measure(value, tmp_path / "empty")["pending_update_rows"]
    value = data()
    for row in value["sources"]:
        if row["slot"] >= 65 and m.partition(row["source_cluster_id"]) == 0:
            row["eligible"] = False
    assert any(
        r["status"] == "censored"
        for r in m.measure(value, tmp_path / "censored")["pending_update_rows"]
    )
    value["labels"]["0"] = 2
    with pytest.raises(ValueError, match="label_contract"):
        m.measure(value, tmp_path / "bad")


def cli(cwd, *args, crash=None):
    """Use actual children outside checkout so imports and exits are exercised."""
    cwd.mkdir(parents=True, exist_ok=True)
    command = [str(e.ROOT / ".venv/bin/python")]
    config = os.environ.get("CARNOT_8076_COVERAGE_CONFIG")
    if config:
        command += ["-m", "coverage", "run", "--rcfile=" + config, "--parallel-mode"]
    command += [str(e.ROOT / e.CLI), *map(str, args)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    if crash:
        env["CARNOT_8076_CRASH_EVENT"] = crash
    print("8076 subprocess before", command, flush=True)
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    print(
        "8076 subprocess after",
        result.returncode,
        result.stdout[-1500:],
        result.stderr[-1500:],
        flush=True,
    )
    return result


def test_cli_publication_and_mutation(tmp_path):
    """SCENARIO-REPORT-8076-TERMINAL: real private success, blocked and cold routes."""
    src = tmp_path / "fixture.json"
    atomic_json(src, data())
    out = tmp_path / "success" / (e.NAME + ".json")
    assert cli(tmp_path / "cwd", "--fixture-input", src, "--fixture-output", out).returncode == 0
    value = json.loads(out.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["learning_trajectory_ready_score"] == 0
    assert cli(tmp_path / "cwd", "--cold-replay", out).returncode == 0
    assert cli(tmp_path / "cwd", "--fixture-output", out).returncode == 1
    value["issued_prediction_rows"][0]["probability"] = 0.99
    atomic_json(out, value)
    assert cli(tmp_path / "cwd", "--cold-replay", out).returncode == 1
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    assert (
        cli(tmp_path / "cwd", "--root", tmp_path / "absent", "--fixture-output", blocked).returncode
        == 0
    )
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    assert cli(tmp_path / "cwd", "--cold-replay", blocked).returncode == 0
    assert cli(tmp_path / "cwd", "--cold-replay", tmp_path / "absent").returncode == 1


@pytest.mark.parametrize("event", ["addition", "candidate", "consume", "commit"])
def test_subprocess_recovery(tmp_path, event):
    """SCENARIO-REPORT-8076-DURABLE: four real death boundaries resume exactly once."""
    value = data()
    src = tmp_path / "fixture.json"
    atomic_json(src, value)
    raw = tmp_path / "recovered"
    child = cli(
        tmp_path / "cwd", "--fixture-input", src, "--worker-output", raw / "work.json", crash=event
    )
    assert child.returncode == 73
    assert not (raw / "work.json").exists()
    assert (
        cli(
            tmp_path / "cwd", "--fixture-input", src, "--worker-output", raw / "work.json"
        ).returncode
        == 0
    )
    recovered = m.reduce(raw / "trajectory")
    clean = m.measure(value, tmp_path / "clean")
    for field in m.FIELDS.values():
        assert recovered[field] == clean[field]
    assert len({(r["seed"], r["slot"]) for r in recovered["admission_consumption_rows"]}) == len(
        recovered["admission_consumption_rows"]
    )


def test_prerequisites_and_build(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8076-TERMINAL: owned failure differs from external absence."""
    load = e.prior.load_inputs

    def small(root, raw):
        value, failures = load(root, raw)
        value["seeds"] = [101]
        return value, failures

    monkeypatch.setattr(e.prior, "load_inputs", small)
    work = e.worker(e.ROOT, tmp_path / "real")
    assert not work["failures"] and work["evidence"]["final_head_seals"]
    receipts = [dict(name="measurement_normal_exit", passed=True)]
    value = e.build(work, tmp_path / "real", receipts, {}, fixture=False)
    assert value["verdict_class"] == "disqualified"
    assert value["learning_trajectory_ready_score"] == 0
    coverage = {p: dict(missing_lines=0) for p in e.OWNED}
    plan = json.loads((tmp_path / "real/work.json").read_text())
    assert plan == work
    receipts = [
        dict(name=n, passed=True)
        for n in [
            "measurement_normal_exit",
            *[r["name"] for r in e.manifest(tmp_path) if r.get("classification") != "diagnostic"],
        ]
    ]
    value = e.build(work, tmp_path / "real", receipts, coverage, fixture=False)
    assert value["learning_trajectory_ready_score"] == 1
    assert value["generalized_learning_benefit_score"] == 0
    failed = deepcopy(work)
    failed["owned_failure"] = True
    assert (
        e.build(failed, tmp_path / "real", receipts, coverage, fixture=False)["verdict_class"]
        == "disqualified"
    )
    empty = e.worker(tmp_path / "missing", tmp_path / "blocked")
    assert empty["failures"]
    assert (
        e.build(empty, tmp_path / "blocked", receipts, coverage, fixture=False)["verdict_class"]
        == "blocked"
    )
    root = tmp_path / "root"
    (root / "results").mkdir(parents=True)
    for name in e.INPUTS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if name.endswith(".json") else "input")
    refs, failures = e.prerequisites(root, tmp_path / "invalid")
    assert refs and any(r["field"] == "terminal_validation_sidecar_path" for r in failures)
    monkeypatch.setattr(e, "run_check", lambda *a, **k: dict(passed=False))
    candidate = tmp_path / "real/terminal_candidate.json"
    atomic_json(candidate, value)
    assert e.terminal(candidate)["passed"] is False


def test_cold_operand_mutations(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8076-DURABLE: hashes, receipts and reductions fail closed."""
    src = tmp_path / "data.json"
    atomic_json(src, data())
    raw = tmp_path / "raw"
    work = e.worker(e.ROOT, raw, src)
    log = tmp_path / "log"
    log.write_text("private validation")
    receipts = [dict(name="private", passed=True, log_path=str(log), log_sha256=sha256_file(log))]
    atomic_json(raw / "validation.json", dict(receipts=receipts, coverage={}, fixture=True))
    value = e.build(work, raw, receipts, {}, fixture=True)
    out = tmp_path / "candidate.json"
    atomic_json(out, value)
    assert e.replay(out)
    for field in ("raw_shard_hashes", "code_config_hashes"):
        altered = deepcopy(value)
        if field == "raw_shard_hashes":
            altered[field][0]["sha256"] = "sha256:bad"
        else:
            altered[field][e.MODULE] = "sha256:bad"
        atomic_json(out, altered)
        assert not e.replay(out)
    atomic_json(out, value)
    log.write_text("changed validation")
    assert not e.replay(out)
    log.write_text("private validation")
    for field in ("rows", "retention_rows"):
        altered = deepcopy(work)
        altered["evidence"][field][0]["numerator"] = 99
        atomic_json(raw / "work.json", altered)
        assert not e.replay(out)
    atomic_json(raw / "work.json", work)
    monkeypatch.setattr(
        e, "read_bound_sidecar", lambda p, s: dict(primary_path=str(p), report=dict(passed=False))
    )
    assert any(
        r["field"] == "terminal.passed" for r in e.prerequisites(e.ROOT, tmp_path / "binding")[1]
    )
    monkeypatch.setattr(e, "ROOT", tmp_path / "absent")
    assert any(
        r["check"] == "required_tool" for r in e.prerequisites(e.ROOT, tmp_path / "tools")[1]
    )


def test_worker_failure_and_heartbeat(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8076-TERMINAL: numerical failures cannot publish readiness."""
    src = tmp_path / "data.json"
    atomic_json(src, data())
    original = m.measure

    def fail(*args, **kwargs):
        raise TimeoutError("private failure")

    monkeypatch.setattr(m, "measure", fail)
    work = e.worker(e.ROOT, tmp_path / "owned", src)
    assert work["owned_failure"] and work["failures"][0]["classification"] == "owned"
    monkeypatch.setattr(e, "run_check", lambda *a, **k: dict(passed=False, exit_code=7))
    assert (
        e.worker(e.ROOT, tmp_path / "external", src)["failures"][0]["check"] == "python_environment"
    )
    monkeypatch.setattr(m, "measure", original)
    clock = m.time.monotonic
    ticks = iter(range(10000))
    monkeypatch.setattr(m.time, "monotonic", lambda: clock() + next(ticks) * 31)
    assert m.measure(data(), tmp_path / "heartbeat", budget_s=1e9)["rows"]
    monkeypatch.setattr(m.time, "monotonic", clock)
    monkeypatch.setenv("CARNOT_8076_CRASH_EVENT", "issue")

    def exit_stub(code):
        raise RuntimeError(str(code))

    monkeypatch.setattr(m.os, "_exit", exit_stub)
    with pytest.raises(RuntimeError, match="73"):
        m.measure(data(), tmp_path / "exit")
    monkeypatch.delenv("CARNOT_8076_CRASH_EVENT")
    with sqlite3.connect(tmp_path / "heartbeat/seed-101/ledger.sqlite") as db:
        db.execute("INSERT INTO events(kind,payload) VALUES('issue','{}')")
    with pytest.raises(ValueError, match="trailing_events"):
        m.measure(data(), tmp_path / "heartbeat")


def test_owned_main_failed_child_and_validation(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8076-TERMINAL: a failed child yields terminal disqualification."""
    src = tmp_path / "data.json"
    atomic_json(src, data())
    actual_manifest = e.manifest

    def short_manifest(private):
        return [
            actual_manifest(private)[0],
            dict(name="coverage_json", argv=[], deadline_s=1, expected_exit=0),
        ]

    monkeypatch.setattr(e, "manifest", short_manifest)

    def check(root, spec, private, durable):
        log = private / (spec["name"] + ".log")
        log.write_text("private failed-child fixture")
        if spec["name"] == "coverage_json":
            atomic_json(
                private / "coverage.json",
                dict(files={str(e.ROOT / p): dict(summary=dict(missing_lines=0)) for p in e.OWNED}),
            )
        return dict(
            spec,
            passed=spec["name"] != "measurement_normal_exit",
            exit_code=1 if spec["name"] == "measurement_normal_exit" else 0,
            log_path=str(log),
            log_sha256=sha256_file(log),
        )

    monkeypatch.setattr(e, "run_check", check)
    monkeypatch.setattr(e, "terminal", lambda path: dict(passed=True))
    out = tmp_path / (e.NAME + ".json")
    assert e.main(["--fixture-input", str(src), "--output", str(out)]) == 0
    value = json.loads(out.read_text())
    assert (
        value["verdict_class"] == "disqualified" and value["learning_trajectory_ready_score"] == 0
    )
    assert value["gate_check_summary"]
    assert (
        e.main(
            [
                "--worker-output",
                str(tmp_path / "failure/work.json"),
                "--fixture-input",
                str(tmp_path / "missing"),
            ]
        )
        == 1
    )
