"""REQ-VERIFY-8143 / REQ-REPORT-8143: causal memory and private publication."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import delayed_energy_memory_8143 as e


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    raw = tmp_path_factory.mktemp("memory8143")
    return e.measure(e.ROOT, raw, fixture=True), raw


def test_natural_engine_and_retention(measured):
    """REQ-VERIFY-8143: every original unit and causal boundary stays visible."""
    work, raw = measured
    value = e.build(deepcopy(work), raw, [dict(name="owned", passed=True)])
    assert value["learning_trajectory_ready_score"] == 1
    assert value["continuous_self_learning_task"] is True
    assert value["MODEL_SPECS"] == [] and value["call_ledger"] == []
    assert value["verdict_class"] == "circular_positive"
    assert len(work["state_manifest"]) == 20
    assert len(work["retention_predictions"]) == 20
    assert {r["slot"] for r in work["rows"] if r["condition"] == "later_stream"} == set(
        range(65, 257)
    )
    assert all(
        r["status"] == "censored"
        for r in work["rows"]
        if r["condition"] == "later_stream" and r["slot"] > 236
    )
    path = raw / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    value["rows"][0]["numerator"] += 1
    atomic_json(path, value)
    assert not e.replay(path)


def test_block_and_disqualification(tmp_path, measured):
    """REQ-REPORT-8143: external operands and owned failures stay distinct."""
    work = e.measure(tmp_path, tmp_path / "raw")
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["verdict_class"] == "blocked"
    assert value["learning_trajectory_ready_score"] == 0
    assert any(not r["passed"] and r["expected"] is not None for r in value["gate_check_summary"])
    failed = e.build(measured[0], measured[1], [dict(passed=False)])
    assert failed["verdict_class"] == "disqualified"
    assert failed["learning_trajectory_ready_score"] == 0


def test_opaque_labels_and_restart(tmp_path):
    """REQ-VERIFY-8143: delayed feedback is decoded only after sealed issuance."""
    rows, labels = e.engine.fixture()
    path = tmp_path / "labels.json"
    atomic_json(path, dict(rows=[dict(r, y=y) for r, y in zip(rows, labels, strict=True)]))
    full = e.run_seed(rows, path, 101, tmp_path / "full")
    part = e.run_seed(rows, path, 101, tmp_path / "part", stop=128)
    resumed = e.run_seed(rows, path, 101, tmp_path / "resume", state=part)
    assert resumed == full
    assert max(full["released"]) == 236 and len(full["pending"]) == 20
    assert set(full["training"]).isdisjoint(full["used_admission"])
    vault = e.DelayedLabels(path, rows)
    with pytest.raises(ValueError, match="unsealed_release"):
        vault[0]
    assert len(vault) == 256
    assert e.head_diagnostics(full)["error_center"]["installed_capacity"] <= 28


def test_rehashed_mutation(measured):
    """REQ-REPORT-8143: headline tampering fails even with a new checksum."""
    work, raw = measured
    value = e.build(deepcopy(work), raw, [dict(passed=True)])
    value["reductions"] = {}
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    path = raw / "mutation.json"
    atomic_json(path, value)
    assert not e.replay(path)


def child(argv, cwd, expected=0):
    """Poll real child completion with bounded waits and visible pending counts."""
    import os
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["JAX_PLATFORMS"] = "cpu"
    if os.environ.get("COVERAGE_RCFILE"):
        env["COVERAGE_PROCESS_START"] = os.environ["COVERAGE_RCFILE"]
    e.progress("before_private_subprocess", 0, 1)
    with subprocess.Popen(
        argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as process:
        for _ in range(10):
            try:
                log, _ = process.communicate(timeout=30)
                break
            except subprocess.TimeoutExpired:
                e.progress("private_child_pending", 0, 1)
        else:
            process.kill()
            raise AssertionError("private child exceeded300s")
        e.progress("after_private_subprocess", 1, 0)
        assert process.returncode == expected, log
        return log


def test_private_cli_and_real_crash_restart(tmp_path):
    """SCENARIO-REPORT-8143: private success/block/mutation/replay and real crash."""
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]
    output = tmp_path / (e.NAME + ".json")
    child([*argv, "--fixture-output", str(output)], tmp_path)
    assert "replay_passed" in child([*argv, "--cold-replay", str(output)], tmp_path)
    blocked = tmp_path / "block" / (e.NAME + ".json")
    child([*argv, "--fixture-output", str(blocked), "--root", str(tmp_path / "missing")], tmp_path)
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    child([*argv, "--cold-replay", str(blocked)], tmp_path)
    value = json.loads(output.read_text())
    Path(value["state_manifest"][0]["state"]["path"]).write_text("{}")
    child([*argv, "--cold-replay", str(output)], tmp_path, expected=1)
    child([*argv, "--date", "20260101"], tmp_path, expected=2)
    rows, labels = e.engine.fixture()
    targets = tmp_path / "targets.json"
    atomic_json(targets, dict(rows=[dict(r, y=y) for r, y in zip(rows, labels, strict=True)]))
    inputs = tmp_path / "seed.json"
    atomic_json(inputs, dict(rows=rows, label_path=str(targets), seed=101))
    crash, resume, full = [tmp_path / k for k in ["crash", "resume", "full"]]
    child(
        [*argv, "--seed-input", str(inputs), "--seed-output", str(crash), "--crash-slot", "64"],
        tmp_path,
        expected=73,
    )
    checkpoint = crash / "crash_checkpoint.json"
    assert json.loads(checkpoint.read_text())["phase"] == "candidate"
    child(
        [
            *argv,
            "--seed-input",
            str(inputs),
            "--seed-output",
            str(resume),
            "--resume-state",
            str(checkpoint),
        ],
        tmp_path,
    )
    child([*argv, "--seed-input", str(inputs), "--seed-output", str(full)], tmp_path)
    assert json.loads((resume / "final_state.json").read_text()) == json.loads(
        (full / "final_state.json").read_text()
    )
    child(
        [
            *argv,
            "--worker-output",
            str(tmp_path / "worker" / "measurement.json"),
            "--root",
            str(tmp_path / "missing"),
        ],
        tmp_path,
    )


def test_authentication_and_natural_branch(tmp_path, measured, monkeypatch):
    """REQ-VERIFY-8143: exact qualification and historical imports authorize reuse."""
    b = e.engine.methods.Custody(tmp_path / "custody")
    real = e.authenticate(e.ROOT, tmp_path / "raw", b)
    assert real["stream_feature_manifest"] and not b.failures
    old = json.loads((e.ROOT / e.UPSTREAM).read_text())
    old["learning_protocol_ready_score"] = 0
    atomic_json(tmp_path / e.UPSTREAM, old)
    monkeypatch.setattr(
        e.engine.methods.historical, "terminal", lambda *a: dict(report=dict(passed=True))
    )
    with pytest.raises(ValueError):
        e.authenticate(
            tmp_path, tmp_path / "bad", e.engine.methods.Custody(tmp_path / "bad" / "inputs")
        )
    work, raw = measured
    monkeypatch.setattr(e, "authenticate", lambda *a: work["input_manifests"])

    def cached(rows, path, seed, seed_raw):
        seed_raw.mkdir(parents=True)
        original = raw / f"seed-{seed}"
        for name in ["final_state.json", "opened_labels.json"]:
            (seed_raw / name).write_bytes((original / name).read_bytes())
        return json.loads((seed_raw / "final_state.json").read_text())

    monkeypatch.setattr(e, "run_seed", cached)
    natural = e.measure(e.ROOT, tmp_path / "natural")
    assert natural["input_ready"] == 1
    assert e.build(natural, tmp_path / "natural", [dict(passed=True)])["verdict_class"] == "null"
    monkeypatch.setattr(e, "authenticate", lambda *a: (_ for _ in ()).throw(KeyError("operand")))
    assert e.measure(tmp_path, tmp_path / "exception")["input_ready"] == 0


def test_cpu_budget_and_replay_guards(tmp_path, measured, monkeypatch):
    """REQ-VERIFY-8143: bounded CPU execution and code/config/state mutation guards."""
    rows, targets = e.engine.fixture()
    path = tmp_path / "labels.json"
    atomic_json(path, dict(rows=[dict(r, y=y) for r, y in zip(rows, targets, strict=True)]))
    ticks = iter([0, 1201])
    with monkeypatch.context() as m:
        m.setattr(e.time, "monotonic", lambda: next(ticks))
        with pytest.raises(TimeoutError, match="cpu_science_budget"):
            e.run_seed(rows, path, 101, tmp_path / "budget")
    with monkeypatch.context() as m:
        m.setattr(e.os, "_exit", lambda code: (_ for _ in ()).throw(SystemExit(code)))
        with pytest.raises(SystemExit) as crashed:
            e.run_seed(rows, path, 101, tmp_path / "crash", crash_slot=1)
        assert crashed.value.code == 73
    ticks = iter([0, 0, 1201])
    with monkeypatch.context() as m:
        m.setattr(e.time, "monotonic", lambda: next(ticks))
        with pytest.raises(TimeoutError):
            e.measure(e.ROOT, tmp_path / "measure_budget", fixture=True)
    work, raw = measured
    base = e.build(work, raw, [dict(passed=True)])

    def cached_run(rows, labels, seed, **kwargs):
        labels.opened = json.loads((raw / f"seed-{seed}" / "opened_labels.json").read_text())
        return json.loads((raw / f"seed-{seed}" / "final_state.json").read_text())

    monkeypatch.setattr(e.engine, "run", cached_run)
    path = tmp_path / "replay.json"
    assert not e.replay(path)
    for field, change in [
        ("code_config_hashes", {e.MODULE: "sha256:wrong"}),
        ("learning_trajectory_ready_score", 0),
        ("retention_predictions", []),
        ("state_manifest", [dict(work["state_manifest"][0], diagnostics={})]),
    ]:
        value = deepcopy(base)
        value[field] = change
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        atomic_json(path, value)
        assert not e.replay(path)
    value = deepcopy(base)
    value["retention_predictions"][0]["rows"][0]["predictions"][e.engine.ARMS[0]] += 0.1
    value["retention_prediction_manifest"] = e.engine.shard(
        tmp_path / "retention_mutation", dict(rows=value["retention_predictions"])
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    assert not e.replay(path)
    value = deepcopy(base)
    value["rows"][0]["numerator"] += 0.1
    value["reductions"] = e.engine.historical.reductions(value["rows"])
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    assert not e.replay(path)


def test_restart_mismatch(tmp_path, monkeypatch):
    """REQ-VERIFY-8143: a failed fresh resume cannot earn trajectory readiness."""
    monkeypatch.setattr(e, "run_check", lambda *a, **kw: dict(passed=False, actual_exit=1))
    assert e.restart_check([], tmp_path / "labels.json", tmp_path, {})["passed"] is False


def test_supervisor_routes(tmp_path, monkeypatch, measured):
    """REQ-REPORT-8143: frozen normal checks precede publication or disqualification."""
    from carnot.reporting import delayed_energy_execution_8143 as runner

    work, _ = measured
    owned_check = runner.check
    monkeypatch.setattr(e, "replay", lambda p: True)
    monkeypatch.setattr(e, "measure", lambda *a, **kw: deepcopy(work))
    monkeypatch.setattr(
        runner.previous, "run_check", lambda *a, **kw: dict(passed=True, actual_exit=0)
    )
    assert runner.check(dict(name="normal"), tmp_path)["normal_exit"] is True

    def check(spec, private):
        if spec["name"] == "measurement":
            atomic_json(Path(spec["argv"][-1]), work)
        return dict(name=spec["name"], passed=True, actual_exit=0, normal_exit=True)

    monkeypatch.setattr(runner, "check", check)
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    assert runner.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["required_checks_passed"] is True
    assert runner.main(["--fixture-output", str(output), "--mutation"]) == 0
    with pytest.raises(SystemExit) as error:
        runner.main(["--fixture-output", str(e.ROOT / "results" / (e.NAME + ".json"))])
    assert error.value.code == 2
    failed = tmp_path / "failed" / (e.NAME + ".json")

    def failing(spec, private):
        row = check(spec, private)
        row["passed"] = spec["name"] != "strict_row_lint"
        return row

    monkeypatch.setattr(runner, "check", failing)
    assert runner.main(["--output", str(failed)]) == 0
    assert json.loads(failed.read_text())["learning_trajectory_ready_score"] == 0
    monkeypatch.setattr(
        runner.previous,
        "run_check",
        lambda *a, **kw: dict(passed=False, actual_exit=-9, timed_out=True),
    )
    assert owned_check(dict(name="crashed"), tmp_path)["normal_exit"] is False
