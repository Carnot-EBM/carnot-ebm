"""REQ-REPORT-8065: private causal, statistical and process recovery controls."""

from copy import deepcopy
import json
import hashlib
import time
import os
from pathlib import Path
import sqlite3
import subprocess

import numpy as np
import pytest

from carnot import experiment_8065_v698_fresh_learning_audit as e
from carnot.verify import fresh_learning_audit_8065 as a
from carnot.verify import fresh_feedback_8064 as learner
from carnot.reporting.current_work_receipt import atomic_json
from test_fresh_feedback_8064 import data


def bundle(tmp_path):
    """Keep every artificial target and journal outside historical results."""
    d = data()
    raw = tmp_path / "trajectory"
    learner.measure(d, raw)
    return d, raw


def test_independent_reconstruction(tmp_path):
    """SCENARIO-REPORT-8065-CAUSAL: producer replay cannot substitute independent math."""
    d, raw = bundle(tmp_path)
    rebuilt = a.reconstruct(raw, d["labels"])
    original = learner.reduce(raw)
    for field in learner.FIELDS.values():
        assert rebuilt[field] == original[field]
    assert rebuilt["admission_reuse_count"] == 0
    assert rebuilt["rows"] == original["rows"]
    assert len(rebuilt["issued_prediction_rows"]) == 1024
    with pytest.raises(ValueError):
        a.reconstruct(raw, dict(d["labels"], **{"0": 1 - d["labels"]["0"]}))
    db = sqlite3.connect(raw / "seed-101/ledger.sqlite")
    db.execute("DROP TRIGGER forbid_UPDATE")
    db.execute("UPDATE events SET payload='{}' WHERE seq=0")
    db.commit()
    db.close()
    with pytest.raises((ValueError, KeyError)):
        a.reconstruct(raw, d["labels"])


def test_deferrals(tmp_path):
    """SCENARIO-REPORT-8065-CAUSAL: missing targets/classes do not buy extra evidence."""
    for case in ("empty", "censored", "class", "excluded"):
        d = data()
        for r in d["sources"]:
            if case == "empty" or (
                case == "censored" and r["slot"] >= 65 and a.role(r) == "admission"
            ):
                r["eligible"] = False
            if case == "class" and a.role(r) == "admission":
                d["labels"][r["family_id"]] = 0
        if case == "excluded":
            d["sources"][0]["public_eligible"] = False
            d["labels"]["1"] = None
        raw = tmp_path / case
        learner.measure(d, raw)
        assert a.reconstruct(raw, d["labels"])["rows"] == learner.reduce(raw)["rows"]


def test_statistics():
    """SCENARIO-REPORT-8065-SCIENCE: .02 margin, chronological masks and no seed inflation."""
    rows, retained = [], []
    for seed in (101, 102):
        for slot in range(121, 236):
            for arm in a.ARMS:
                fresh = arm == "fresh_admission"
                rows.append(
                    dict(
                        seed=seed,
                        arm=arm,
                        slot=slot,
                        source=str(slot),
                        y=slot % 2,
                        action="reject" if fresh else "escalate",
                        probability=0.8 if fresh else 0.2,
                        numerator=0.0 if fresh else 0.5,
                        denominator=1,
                        brier=0.1,
                        status="completed",
                        unit=str(slot),
                        exclusion_reason=None,
                    )
                )
        for slot in range(64):
            for arm in a.ARMS:
                retained.append(
                    dict(
                        seed=seed,
                        arm=arm,
                        slot=slot,
                        source=str(slot),
                        y=slot % 2,
                        numerator=0.5,
                        denominator=1,
                        brier=0.1,
                    )
                )
    result = a.comparisons(rows, retained)
    h = result["primary_hypothesis_results"][0]
    assert h["support_count"] == 115 and h["beneficial_changed_sources"] == 115
    assert h["qualified_benefit"] and h["capstone_family_p"] < 0.05
    assert [t["block_length"] for t in h["tests"]] == [32, 16, 64]
    assert all(t["slot_count"] == 256 and t["margin"] == 0.02 for t in h["tests"])
    bad = deepcopy(retained)
    bad[2]["brier"] = 100
    assert a.comparisons(rows, bad)["primary_hypothesis_results"][0]["capstone_family_p"] == 1
    assert a.comparisons([], retained)["primary_hypothesis_results"][0]["capstone_family_p"] == 1
    assert a.comparisons(rows, [])["primary_hypothesis_results"][0]["support_passed"] is False
    x = [float("nan")] * 256
    assert a.bootstrap(x, 32)["raw_p"] == 1
    assert a.bootstrap([0.02] * 256, 32)["raw_p"] == 1
    assert a.bootstrap([0.03] * 256, 32)["raw_p"] < 0.05


def test_recovery(tmp_path):
    """SCENARIO-REPORT-8065-RECOVERY: SIGKILL on both sides of all three durable boundaries."""
    d = data()
    rows = e.recovery(d, tmp_path / "private")
    assert len(rows) == 6
    assert all(r["passed"] and r["exit_code"] == -9 for r in rows)
    assert all(r["recovered_sha256"] == r["uninterrupted_sha256"] for r in rows)
    assert {r["event"] for r in rows} == {"candidate", "consume", "commit"}
    assert e.verify_recovery(tmp_path / "private") == rows
    receipt = tmp_path / "private/receipts.json"
    original = json.loads(receipt.read_text())
    changed = deepcopy(original)
    changed["rows"][0]["pending_ids"] = [-1]
    atomic_json(receipt, changed)
    with pytest.raises(ValueError, match="pending_ids"):
        e.verify_recovery(tmp_path / "private")
    atomic_json(receipt, original)


def cli(cwd, *args):
    """Invoke real script routes outside checkout and retain child coverage."""
    cwd.mkdir(parents=True, exist_ok=True)
    argv = [str(e.ROOT / ".venv/bin/python")]
    config = os.environ.get("CARNOT_8065_COVERAGE_CONFIG")
    if config:
        argv += ["-m", "coverage", "run", "--rcfile=" + config]
    argv += [str(e.ROOT / e.CLI), *map(str, args)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    print("8065 subprocess before", argv, flush=True)
    started = time.monotonic()
    result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    log = result.stdout + result.stderr
    path = cwd / (str(time.time_ns()) + ".log")
    path.write_text(log)
    print(
        "8065 subprocess after",
        json.dumps(
            dict(
                argv=argv,
                exit_code=result.returncode,
                duration_s=time.monotonic() - started,
                log_path=str(path),
                log_sha256="sha256:" + hashlib.sha256(log.encode()).hexdigest(),
                tail=log[-1000:],
            )
        ),
        flush=True,
    )
    return result


def test_cli(tmp_path):
    """SCENARIO-REPORT-8065-TERMINAL: real success, missing, mutation and repeat routes."""
    d, trajectory = bundle(tmp_path)
    d["trajectory"] = str(trajectory)
    source = tmp_path / "fixture.json"
    atomic_json(source, d)
    output = tmp_path / "success" / (e.NAME + ".json")
    result = cli(tmp_path / "cwd", "--fixture-input", source, "--fixture-output", output)
    assert result.returncode == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["learning_audit_ready_score"] == value["learning_benefit_score"] == 0
    assert cli(tmp_path / "cwd", "--cold-replay", output).returncode == 0
    assert (
        cli(tmp_path / "cwd", "--fixture-input", source, "--fixture-output", output).returncode == 1
    )
    value["rows"][0]["probability"] = 0.99
    atomic_json(output, value)
    assert cli(tmp_path / "cwd", "--cold-replay", output).returncode == 1
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    assert (
        cli(
            tmp_path / "cwd", "--root", tmp_path / "missing", "--fixture-output", blocked
        ).returncode
        == 0
    )
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    assert cli(tmp_path / "cwd", "--cold-replay", blocked).returncode == 0
    assert cli(tmp_path / "cwd", "--date", "invalid").returncode == 2
    # A privately owned bad source is disqualified, rather than retryable partial.
    d["trajectory"] = str(tmp_path / "absent")
    atomic_json(source, d)
    bad = tmp_path / "bad" / (e.NAME + ".json")
    assert cli(tmp_path / "cwd", "--fixture-input", source, "--fixture-output", bad).returncode == 0
    assert json.loads(bad.read_text())["verdict_class"] == "disqualified"


def test_retention_seal(tmp_path):
    """SCENARIO-REPORT-8065-CAUSAL: evaluator targets open after every prediction seal."""
    d, raw = bundle(tmp_path)
    result = a.reconstruct(raw, d["labels"])

    def targets():
        seal = json.loads((tmp_path / "retention_prediction_seal.json").read_text())
        assert not seal["labels_opened"] and len(seal["rows"]) == 256
        return d["retention_labels"]

    rows = a.retention(d, result["final_head_seals"], tmp_path, targets)
    assert all(r["denominator"] for r in rows)
    d["retention"][0]["public_eligible"] = False
    d["retention_labels"]["1"] = None
    rows = a.retention(d, result["final_head_seals"], tmp_path, targets)
    assert sum(r["denominator"] == 0 for r in rows) == 8


def test_guards():
    """SCENARIO-REPORT-8065-CAUSAL: independent guard catches harm and all-alpha rejection."""
    h = data()["head"]
    x = np.zeros((12, 110))
    x[:, 0] = np.tile([-1.0, 1.0], 6)
    y = np.tile([0, 1], 6)
    zero = np.zeros(110)
    good = zero.copy()
    good[0] = 1.0
    assert a.guard(h, zero, good, zero, x, y)[1] == 1
    checks, alpha = a.guard(h, zero, -good * 20, zero, x, y)
    assert alpha == 0 and any(r["reasons"] for r in checks)
    assert a.guard(h, -good * 20, -good * 30, good, x, y)[1] is None
    assert a.guard(h, zero, good, zero, x[:0], y[:0]) == ([], None)
    assert a.guard(h, zero, good, zero, x, np.zeros(12)) == ([], None)


def test_custody(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8065-TERMINAL: authenticate real bytes, then expose exact bad operands."""
    d, failures = e.load_inputs(e.ROOT, tmp_path / "real")
    assert not failures and Path(d["trajectory"]).is_dir()
    assert len(e.labels(d, "labels")) == 256
    original = e.read_bound_sidecar

    def altered(*args):
        result = original(*args)
        result["report"]["passed"] = False
        return result

    monkeypatch.setattr(e, "read_bound_sidecar", altered)
    d, failures = e.load_inputs(e.ROOT, tmp_path / "changed")
    assert any(r["field"] == "report.passed" and r["observed"] is False for r in failures)
    original_inputs = e.upstream.load_inputs

    def mismatch(*args):
        d, failures = original_inputs(*args)
        d["seeds"] = [0]
        return d, failures

    monkeypatch.setattr(e.upstream, "load_inputs", mismatch)
    _, failures = e.load_inputs(e.ROOT, tmp_path / "drift")
    assert any(r["field"] == "seeds" for r in failures)


def test_normal_validation(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8065-TERMINAL: owned check failures override external blocking."""
    monkeypatch.setattr(
        e,
        "manifest",
        lambda private: [
            dict(name="python_environment", argv=["python"], deadline_s=30, expected_exit=0)
        ],
    )
    monkeypatch.setattr(e, "terminal", lambda path: dict(passed=True))
    log = tmp_path / "log.txt"
    log.write_text("private environment control\n")
    for exit_code in (0, 1):

        def environment(*args):
            atomic_json(
                args[2] / "coverage.json",
                dict(files={p: dict(summary=dict(missing_lines=0)) for p in e.OWNED}),
            )
            return dict(
                name="python_environment",
                passed=exit_code == 0,
                exit_code=exit_code,
                expected_exit=0,
                log_path=str(log),
                log_sha256=e.sha256_file(log),
            )

        monkeypatch.setattr(e, "run_check", environment)
        output = tmp_path / str(exit_code) / (e.NAME + ".json")
        assert e.main(["--root", str(tmp_path / "missing"), "--output", str(output)]) == 0
        value = json.loads(output.read_text())
        assert value["verdict_class"] == ("blocked" if exit_code == 0 else "disqualified")
        assert value["learning_audit_ready_score"] == 0
        assert e.replay(output)["passed"]


def test_owned_verdicts(tmp_path):
    """SCENARIO-REPORT-8065-SCIENCE: valid negatives qualify audit, and failures erase credit."""
    d, trajectory = bundle(tmp_path)
    result = a.reconstruct(trajectory, d["labels"])
    result["retention_rows"] = a.retention(
        d, result["final_head_seals"], tmp_path, lambda: d["retention_labels"]
    )
    result.update(a.comparisons(result["later_source_rows"], result["retention_rows"]))
    result["recovery_rows"] = []
    atomic_json(tmp_path / "independent_reduction.json", result)
    receipt = dict(name="control", passed=True)
    coverage = {p: dict(summary=dict(missing_lines=0)) for p in e.OWNED}
    started = e.time.monotonic_ns()
    value = e.build(d, [], tmp_path, [receipt], coverage, False, started)
    assert value["verdict_class"] == "null" and value["learning_audit_ready_score"] == 1
    assert value["learning_benefit_score"] == 0
    h = result["primary_hypothesis_results"][0]
    h["qualified_benefit"] = True
    atomic_json(tmp_path / "independent_reduction.json", result)
    assert (
        e.build(d, [], tmp_path, [receipt], coverage, False, started)["verdict_class"] == "positive"
    )
    assert e.build(d, [], tmp_path, [], {}, False, started)["verdict_class"] == "disqualified"
    owned = e.failure(tmp_path / "absent", "reconstruction", True, False, owned=True)
    assert (
        e.build(d, [owned], tmp_path, [receipt], coverage, False, started)["verdict_class"]
        == "disqualified"
    )


def test_pending_end_and_label_contract(tmp_path):
    """SCENARIO-REPORT-8065-CAUSAL: unreleased blocks and invalid targets cannot disappear."""
    d = data()
    # After the second opportunity no admission labels arrive until after slot192.
    for r in d["sources"]:
        if 109 <= r["slot"] < 193 and a.role(r) == "admission":
            r["eligible"] = False
    raw = tmp_path / "end"
    learner.measure(d, raw)
    result = a.reconstruct(raw, d["labels"])
    assert any(r["reason"] == "next_attempt" for r in result["pending_update_rows"])
    # An unfinished block at end is recorded while all released targets stay valid.
    for r in d["sources"]:
        if r["slot"] >= 193 and a.role(r) == "admission":
            r["eligible"] = False
    raw = tmp_path / "tail"
    learner.measure(d, raw)
    assert any(
        r["reason"] == "stream_end" for r in a.reconstruct(raw, d["labels"])["pending_update_rows"]
    )
    d["labels"]["0"] = 2
    with pytest.raises(ValueError, match="label_contract"):
        a.reconstruct(raw, d["labels"])
