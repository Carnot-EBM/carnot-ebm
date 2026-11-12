"""REQ-REPORT-9031-V771; SCENARIO-REPORT-9031-CUSTODY: private B custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import graph_corpus_b_9031 as d
from carnot.reporting import graph_corpus_runner_9031 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_frozen_authority_and_roster(tmp_path, monkeypatch):
    """REQ-REPORT-9031-V771: current qualification never changes original B bytes."""
    evidence = d.authenticate(d.ROOT)
    assert not evidence["failures"]
    units = d.ordered_units(evidence["protocol"])
    assert canonical_hash(units) == d.ROSTER_PIN
    assert d.reduce(units, [d.source.censored(u, "private") for u in units])["role_counts"] == {
        "calibration": 64,
        "test": 64,
        "online": 384,
    }
    assert d.authenticate(tmp_path)["failures"][0]["observed"] == "missing_file"
    name, (pin, field) = next(iter(d.PINS.items()))
    path = tmp_path / name
    for observed in (0, True, None, "missing_field"):
        value = {field: observed, "required_checks_passed": True, "flagged_adversarial": False}
        if observed == "missing_field":
            value.pop(field)
        atomic_json(path, value)
        monkeypatch.setitem(d.PINS, name, (d.reference(path)["sha256"], field))
        assert any(
            not x["passed"]
            for x in d.authenticate(tmp_path)["checks"]
            if x["artifact_field"] == field
        )
    monkeypatch.setitem(d.PINS, name, (pin, field))
    assert any(
        x["check"] == "current_authority_bytes" for x in d.authenticate(tmp_path)["failures"]
    )


def test_prior_completed_identity_refusal(tmp_path):
    """SCENARIO-REPORT-9031-CUSTODY: retained matching responses prohibit a second call."""
    units = d.ordered_units(d.authenticate(d.ROOT)["protocol"])
    primary = tmp_path / "results/experiment_9017_graph_corpus_b.json"
    atomic_json(primary, dict(rows=[dict(unit_id=units[0]["unit_id"], status="completed")]))
    inventory = d.inventory(tmp_path, units, tmp_path / "retained")
    assert inventory["check"]["observed"] == 1 and not inventory["check"]["passed"]
    response = (
        tmp_path
        / "results/raw/experiment_9017_graph_corpus_b/invocations/1/runtime/collection/batch-00"
        / units[1]["unit_id"]
        / "response.json"
    )
    atomic_json(response, dict(raw_response={"id": "retained-first-response"}))
    assert d.inventory(tmp_path, units, tmp_path / "retained")["check"]["observed"] == 2


@pytest.fixture
def work(tmp_path):
    """REQ-REPORT-9031-V771: private snapshots retain all censored identities."""
    return r.freeze(d.ROOT, tmp_path / "raw", True)


def test_build_classes_and_failed_checks(work, monkeypatch):
    """REQ-REPORT-9031-V771: failed owned checks zero readiness; oracle success is circular."""
    validation = dict(receipts=[], coverage={}, scope={})
    value = d.build(work, validation, True)
    assert value["experiment_id"] == 9031 and value["shard_id"] == "B"
    assert value["inference_substrate_class"] == "no_model_load"
    assert value["completed_count"] == 0 and len(value["rows"]) == 512
    assert not d.qualified(validation)
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"
    monkeypatch.setattr(d, "qualified", lambda _: True)
    work["runtime"]["checks"] = [dict(check="changed_input", observed=1)]
    value = d.build(work, validation, False)
    assert value["honest_verdict"] == "complete_blocked_changed_input"
    assert value["repository_health_passed"] is False


def child(tmp_path, *args):
    """The real entrypoint must run with only its declared imports and private outputs."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    cmd = [sys.executable, "-u", str(d.ROOT / d.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        cmd[1:2] = ["-m", "coverage", "run"]
    print("before_private_subprocess", args, flush=True)
    result = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    print(
        "after_private_subprocess",
        result.returncode,
        result.stdout[-300:],
        result.stderr[-300:],
        flush=True,
    )
    return result


def test_private_cli_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-9031-CUSTODY: missing and changed inputs remain terminal blocks."""
    output = tmp_path / (d.NAME + ".json")
    assert child(tmp_path, "--date", "20261112", "--fixture-e2e", output).returncode == 0
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    pristine = deepcopy(value)
    value["completed_count"] = 512
    value["reproducibility_checksum"] = d.shared.pilot.checksum(value)
    atomic_json(output, value)
    assert child(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, pristine)
    raw = Path(pristine["raw_shard_hashes"][0]["path"])
    raw.write_bytes(raw.read_bytes() + b" ")
    assert child(tmp_path, "--cold-replay", output).returncode == 1
    missing = tmp_path / "missing" / (d.NAME + ".json")
    assert (
        child(
            tmp_path,
            "--date",
            "20261112",
            "--input-root",
            tmp_path / "absent",
            "--fixture-e2e",
            missing,
        ).returncode
        == 0
    )
    assert json.loads(missing.read_bytes())["verdict_class"] == "blocked"
    assert child(tmp_path, "--cold-replay", missing).returncode == 0
    altered_root = tmp_path / "altered-authority"
    atomic_json(altered_root / next(iter(d.PINS)), {"provenance_ready_score": 1})
    altered = tmp_path / "altered" / (d.NAME + ".json")
    assert (
        child(
            tmp_path, "--date", "20261112", "--input-root", altered_root, "--fixture-e2e", altered
        ).returncode
        == 0
    )
    assert any(
        x["check"] == "current_authority_bytes"
        and x["observed"] != "missing_file"
        and not x["passed"]
        for x in json.loads(altered.read_bytes())["gate_check_summary"]
    )
    assert child(tmp_path, "--cold-replay", altered).returncode == 0
    assert child(tmp_path, "--date", "wrong").returncode == 2
    assert (
        child(tmp_path, "--date", "20261112", "--fixture-e2e", d.ROOT / "bad.json").returncode == 2
    )


def test_runtime_dispatch_and_unknown_health(work, tmp_path, monkeypatch):
    """REQ-REPORT-9031-V771: only authenticated live work reaches B's leased collector."""
    calls = []

    def capture(units, raw, *, task_id):
        calls.append(task_id)
        assert len(units) == 512 and task_id == d.TASK
        return deepcopy(work["runtime"])

    monkeypatch.setattr(r.runtime, "run", capture)
    assert not r.freeze(d.ROOT, tmp_path / "live", False)["runtime"]["live"]
    assert calls == [d.TASK]
    assert not r.freeze(tmp_path / "missing", tmp_path / "blocked", False)["runtime"]["live"]
    assert calls == [d.TASK]
    monkeypatch.setattr(r.original_runtime, "preflight", lambda _: dict(private=True))
    assert r.preflight(tmp_path) == dict(private=True)

    def unknown(_):
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(r.original_runtime, "preflight", unknown)
    with pytest.raises(RuntimeError, match="gpu_health_unknown"):
        r.preflight(tmp_path)


def test_parent_reports_completed_requests(tmp_path, monkeypatch, capsys):
    """SCENARIO-REPORT-9031-CUSTODY: waiting never hides actual completed/pending counts."""
    atomic_json(
        tmp_path / "invocation_ledger.json",
        dict(rows=[dict(operation="generation", status="completed")]),
    )
    monkeypatch.setattr(r.runtime, "run", lambda *_, **__: dict(private=True))
    actual_wait = r.wait
    calls = []

    def wait_once(futures, *, timeout):
        calls.append(timeout)
        return (set(), set()) if len(calls) == 1 else actual_wait(futures, timeout=timeout)

    monkeypatch.setattr(r, "wait", wait_once)
    assert r.collect([{}, {}], tmp_path) == dict(private=True)
    assert "completed=1 pending=1" in capsys.readouterr().out


def test_qualification_and_roster_tamper(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9031-CUSTODY: no headline pass replaces exact scope and coverage."""
    monkeypatch.setattr(d.checks, "receipts_pass", lambda _: True)
    monkeypatch.setattr(d, "verify_measured", lambda *_: True)
    assert d.qualified(dict(scope={}, coverage={}))
    monkeypatch.setattr(d, "ROSTER_PIN", "changed")
    with pytest.raises(ValueError, match="B_roster_drift"):
        d.ordered_units(d.authenticate(d.ROOT)["protocol"])


def test_production_validation_and_scanners(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9031-CUSTODY: failed owned checks retain rejected bytes only."""
    commands = r.plan(tmp_path / "private")
    assert [c.name for c in commands] == d.checks.CHECKS
    direct = next(c for c in commands if c.name == "direct_consumers")
    assert all(p in direct.argv for p in r.DIRECT)
    assert all(Path(p).name != "test_graph_corpus_a_8990.py" for p in d.OWNED)
    monkeypatch.setattr(
        r, "plan", lambda _: [r.CommandSpec("private", ("true",), "current_scoped", 10)]
    )
    monkeypatch.setattr(r.checks, "run_check", lambda *_: dict(passed=True, normal_exit=True))
    monkeypatch.setattr(r, "freeze", lambda *_: deepcopy(work))
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(r, "measured", lambda *_: dict(complete=True, rows=[]))
    monkeypatch.setattr(d, "qualified", lambda _: True)
    published = []
    monkeypatch.setattr(
        r, "publish_primary", lambda output, value, validator: published.append(validator(output))
    )
    monkeypatch.setattr(r.validation_runner, "scanners", lambda *_: dict(passed=True))
    assert r.scanners(tmp_path / "candidate", tmp_path / "logs", {})["passed"]
    assert r.main(["--date", "20261112"]) == 0
    assert len(published) == 1
    monkeypatch.setattr(d, "qualified", lambda _: False)
    assert r.main(["--date", "20261112"]) == 1
    assert len(published) == 1


def test_oracle_positive_and_load_only(work, monkeypatch):
    """REQ-REPORT-9031-V771: actual current ledger tokens determine the inference class."""
    from test_matched_depth_control_8948 import response

    monkeypatch.setattr(d, "qualified", lambda _: True)
    monkeypatch.setattr(d, "raw_complete", lambda _: True)
    runtime = work["runtime"]
    runtime["rows"] = [
        d.source.label_response(u, response(u["expected"][0])) for u in work["units"]
    ]
    runtime["live"] = True
    runtime["ledger"] = [
        dict(
            operation="model_load",
            call_id="load",
            scope="current",
            status="completed",
            output_tokens=0,
        )
    ] + [
        dict(
            operation="generation",
            call_id=u["unit_id"],
            scope="current",
            status="completed",
            output_tokens=1,
        )
        for u in work["units"]
    ]
    runtime["counts"] = d.custody.ledger_counts(runtime["ledger"])
    validation = dict(receipts=[], coverage={}, scope={})
    value = d.build(work, validation, False)
    assert value["verdict_class"] == "circular_positive" and value["shard_ready_score"] == 1
    assert (
        value["inference_substrate_class"] == "model_bounded_generation"
        and value["generated_tokens"] == 512
    )
    assert value["independent_generalization_score"] == 0 and value["minimum_duration_s"] == 10
    runtime["ledger"] = runtime["ledger"][:1]
    runtime["counts"] = d.custody.ledger_counts(runtime["ledger"])
    runtime["live"] = False
    value = d.build(work, validation, False)
    assert (
        value["inference_substrate_class"] == "model_load_no_generation"
        and value["generated_tokens"] == 0
    )
    assert value["minimum_duration_s"] == 2 and value["shard_ready_score"] == 0
