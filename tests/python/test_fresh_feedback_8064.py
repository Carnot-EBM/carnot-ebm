"""REQ-REPORT-8064: private causal trajectories and real publication routes."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

import numpy as np
import pytest

from carnot import experiment_8064_v698_fresh_feedback_learning as e
from carnot.verify import fresh_feedback_8064 as m
from carnot.reporting.current_work_receipt import atomic_json
from test_causal_online_8025 import fixture


def data():
    """Public admission roles have both artificial classes without selecting outcomes."""
    value = fixture(256)
    admission = update = 0
    for i, row in enumerate(value["sources"]):
        if e.prior.partition(row["source_cluster_id"]) == 0:
            value["labels"][str(i)] = admission % 2
            admission += 1
        else:
            value["labels"][str(i)] = update % 2
            update += 1
        row["eligible"] = True
    value["retention"] = deepcopy(value["sources"][:64])
    value["retention_labels"] = deepcopy(value["labels"])
    return value


def test_guard():
    """SCENARIO-REPORT-8064-GUARD: incumbent checks, damage, all-alpha rejection."""
    h = data()["head"]
    x = np.zeros((12, 110))
    x[:, 0] = np.tile([-1.0, 1.0], 6)
    y = np.tile([0, 1], 6)
    zero = np.zeros(110)
    good = zero.copy()
    good[0] = 1
    checks, alpha = m.guard(h, zero, good, zero, x, y)
    assert alpha == 1 and len(checks) == 5
    checks, alpha = m.guard(h, zero, -good * 20, zero, x, y)
    assert alpha == 0 and any(r["reasons"] for r in checks)
    checks, alpha = m.guard(h, -good * 20, -good * 30, good, x, y)
    assert alpha is None and all(not r["passed"] for r in checks)
    assert m.guard(h, zero, good, zero, x[:0], y[:0]) == ([], None)
    assert m.guard(h, zero, good, zero, x, np.zeros(12)) == ([], None)


def test_trajectory(tmp_path):
    """SCENARIO-REPORT-8064-CAUSAL: equal clocks, one-use rows, cold reductions."""
    v = data()
    result = m.measure(v, tmp_path / "run")
    assert result == m.reduce(tmp_path / "run")
    assert len(result["issued_prediction_rows"]) == 1024
    assert all(r["gradients"] <= 12 for r in result["update_budget_rows"])
    assert {r["gradients"] for r in result["update_budget_rows"] if r["arm"] != "frozen"} == {12}
    assert len({(r["seed"], r["source"]) for r in result["admission_consumption_rows"]}) == len(
        result["admission_consumption_rows"]
    )
    assert all(r["release_slot"] == r["slot"] + 20 for r in result["feedback_release_rows"])
    assert all(
        set(r["update_slots"]).isdisjoint(r["fresh_slots"]) for r in result["candidate_commit_rows"]
    )
    assert len(result["final_head_seals"]) == 4
    changed = deepcopy(v)
    changed["labels"]["110"] ^= 1
    other = m.measure(changed, tmp_path / "other")
    assert [r for r in result["issued_prediction_rows"] if r["slot"] <= 130] == [
        r for r in other["issued_prediction_rows"] if r["slot"] <= 130
    ]
    assert [r for r in result["candidate_commit_rows"] if r["slot"] <= 128] == [
        r for r in other["candidate_commit_rows"] if r["slot"] <= 128
    ]
    assert m.measure(v, tmp_path / "run") == result
    with pytest.raises(TimeoutError, match="numerical_budget"):
        m.measure(v, tmp_path / "timeout", budget_s=-1)


def test_deferrals(tmp_path):
    """SCENARIO-REPORT-8064-CAUSAL: exclusions and missing class never buy replacements."""
    v = data()
    v["sources"][0]["public_eligible"] = False
    v["sources"][0].pop("features")
    v["labels"]["1"] = None
    for r in v["sources"]:
        if e.prior.partition(r["source_cluster_id"]) == 0:
            v["labels"][r["family_id"]] = 0
    result = m.measure(v, tmp_path / "classes")
    assert result["excluded_count"] == 2
    assert all(
        r["alpha"] is None
        for r in result["durable_commit_rows"]
        if r["arm"] in ("reused_guard", "fresh_admission")
    )
    for r in v["sources"]:
        r["eligible"] = False
    result = m.measure(v, tmp_path / "empty")
    assert not result["candidate_commit_rows"] and result["pending_update_rows"]
    assert all(r["gradients"] == 0 for r in result["update_budget_rows"])
    v = data()
    for r in v["sources"]:
        if r["slot"] >= 65 and e.prior.partition(r["source_cluster_id"]) == 0:
            r["eligible"] = False
    result = m.measure(v, tmp_path / "censored")
    assert any(r["status"] == "censored" for r in result["pending_update_rows"])


def test_journal(tmp_path):
    """SCENARIO-REPORT-8064-DURABLE: immutable prefixes reject changed release order."""
    journal = m.Journal(tmp_path)
    journal.emit("release", dict(slot=0, y=1))
    journal.close()
    db = sqlite3.connect(tmp_path / "ledger.sqlite")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE events SET payload='{}'")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("DELETE FROM events")
    db.close()
    journal = m.Journal(tmp_path)
    with pytest.raises(ValueError, match="prefix_drift"):
        journal.emit("release", dict(slot=1, y=1))
    journal.close()


def cli(root, *args):
    """Run the real script outside checkout and collect actual child coverage."""
    root.mkdir(parents=True, exist_ok=True)
    config = os.environ.get("CARNOT_8064_COVERAGE_CONFIG")
    command = [str(e.ROOT / ".venv/bin/python")]
    if config:
        command += ["-m", "coverage", "run", "--rcfile=" + config]
    command += [str(e.ROOT / e.CLI), *map(str, args)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    print("8064 subprocess before", command, flush=True)
    started = time.monotonic()
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=90)
    log = result.stdout + result.stderr
    print(
        json.dumps(
            dict(
                argv=command,
                exit_code=result.returncode,
                duration_s=time.monotonic() - started,
                log_sha256="sha256:" + hashlib.sha256(log.encode()).hexdigest(),
                log=log,
            )
        ),
        flush=True,
    )
    return result


def test_cli(tmp_path):
    """SCENARIO-REPORT-8064-TERMINAL: success, blocked and changed bytes use real CLI."""
    src = tmp_path / "data.json"
    atomic_json(src, data())
    out = tmp_path / "success" / (e.NAME + ".json")
    result = cli(tmp_path / "cwd", "--fixture-input", src, "--fixture-output", out)
    assert result.returncode == 0
    value = json.loads(out.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["learning_trajectory_ready_score"] == 0
    assert cli(tmp_path / "cwd", "--cold-replay", out).returncode == 0
    assert cli(tmp_path / "cwd", "--fixture-input", src, "--fixture-output", out).returncode == 1
    value["issued_prediction_rows"][0]["probability"] = 0.99
    atomic_json(out, value)
    assert cli(tmp_path / "cwd", "--cold-replay", out).returncode == 1
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


@pytest.mark.parametrize(
    "event,after", [("candidate", False), ("candidate", True), ("commit", False), ("commit", True)]
)
def test_death_resume(tmp_path, event, after):
    """SCENARIO-REPORT-8064-DURABLE: process death on either side of a durable commit."""
    src = tmp_path / "data.json"
    atomic_json(src, data())
    raw = tmp_path / "raw"
    code = "from pathlib import Path; import json,os; from carnot.verify import fresh_feedback_8064 as m; original=m.Journal.emit\ndef emit(self, kind, row):\n if kind==os.environ['DEATH_EVENT'] and os.environ['DEATH_AFTER']=='0': os._exit(23)\n original(self,kind,row)\n if kind==os.environ['DEATH_EVENT']: os._exit(23)\nm.Journal.emit=emit\nm.measure(json.loads(Path(os.environ['DEATH_INPUT']).read_text()),Path(os.environ['DEATH_RAW']))"
    env = dict(
        os.environ,
        PYTHONPATH=str(e.ROOT / "python") + ":" + str(e.ROOT),
        DEATH_EVENT=event,
        DEATH_AFTER=str(int(after)),
        DEATH_INPUT=str(src),
        DEATH_RAW=str(raw),
        OPENBLAS_NUM_THREADS="1",
    )
    print("8064 death subprocess before", event, after, flush=True)
    child = subprocess.run(
        [str(e.ROOT / ".venv/bin/python"), "-c", code], cwd=tmp_path, env=env, timeout=60
    )
    print("8064 death subprocess after", child.returncode, flush=True)
    assert child.returncode == 23
    result = m.measure(data(), raw)
    assert result == m.reduce(raw)


def test_invalid_events(tmp_path):
    """SCENARIO-REPORT-8064-DURABLE: release duplication, order and unknown labels fail."""
    v = data()
    raw = tmp_path / "run"
    m.measure(v, raw)
    changed = deepcopy(v)
    changed["head"]["parameters"][0] = 1
    with pytest.raises(ValueError, match="input_drift"):
        m.measure(changed, raw)
    bad = deepcopy(v)
    bad["labels"]["0"] = "bad"
    with pytest.raises(ValueError, match="label_contract"):
        m.measure(bad, tmp_path / "label")
    admission = next(r["fresh_slots"][0] for r in m.reduce(raw)["candidate_commit_rows"])
    bad = deepcopy(v)
    bad["labels"][str(admission)] = None
    deferred = m.measure(bad, tmp_path / "admission")
    assert deferred["excluded_count"] == 1
    assert any(r["status"] == "censored" for r in deferred["pending_update_rows"])
    db = sqlite3.connect(raw / "seed-101" / "ledger.sqlite")
    db.execute("DROP TRIGGER forbid_UPDATE")
    seq, text = db.execute("SELECT seq,payload FROM events WHERE kind='release' LIMIT 1").fetchone()
    row = json.loads(text)
    row["release_slot"] += 1
    db.execute("UPDATE events SET payload=? WHERE seq=?", (json.dumps(row), seq))
    db.commit()
    with pytest.raises(ValueError, match="event_order_or_operand_drift"):
        m.reduce(raw)
    db.execute("UPDATE events SET payload=? WHERE seq=?", (text, seq))
    db.execute("INSERT INTO events(kind,payload) VALUES('release',?)", (text,))
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="event_order_or_operand_drift"):
        m.reduce(raw)


def test_inputs(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8064-TERMINAL: authenticate real bytes and exact failed operands."""
    v, failures = e.load_inputs(e.ROOT, tmp_path / "bound")
    assert not failures and len(v["sources"]) == 256 and len(v["retention"]) == 64
    assert len(v["head"]["parameters"]) == 110
    original = json.loads((e.ROOT / "results" / (e.prior.NAME + ".json")).read_text())
    monkeypatch.setattr(e.historical, "load_inputs", lambda *a: (dict(data(), references=[]), []))
    root = tmp_path / "private"
    p = root / "results" / (e.prior.NAME + ".json")
    report = p.parent / "raw" / e.prior.NAME / "report.json"
    side = root / "terminal.json"
    value = deepcopy(original)
    value["terminal_validation_sidecar_path"] = str(side)
    value["learning_protocol_ready_score"] = 0
    atomic_json(p, value)
    binding = dict(primary_path=str(p), primary_sha256=e.sha256_file(p), sidecar_path=str(report))
    atomic_json(side, dict(publication=binding))
    atomic_json(report, dict(binding, report=dict(passed=True)))
    _, failures = e.load_inputs(root, tmp_path / "failed")
    assert any(
        r["field"] == "learning_protocol_ready_score" and r["observed"] == 0 for r in failures
    )
    value["learning_protocol_ready_score"] = 1
    value["role_manifests"]["stream"] = original["role_manifests"]["retention"]
    atomic_json(p, value)
    binding["primary_sha256"] = e.sha256_file(p)
    atomic_json(side, dict(publication=binding))
    atomic_json(report, dict(binding, report=dict(passed=True)))
    _, failures = e.load_inputs(root, tmp_path / "drift")
    assert failures[-1]["observed"] == "original_stream_identity"


def test_main_owned_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8064-TERMINAL: real failed check and numerical death retire honestly."""
    v = data()
    for key, target in (("labels", "target_reference"), ("retention_labels", "retention_target")):
        path = tmp_path / (key + ".json")
        atomic_json(
            path, dict(rows=[dict(family_id=k, eligible_y=y) for k, y in v.pop(key).items()])
        )
        v[target] = e.reference(path)

    def manifest(private):
        atomic_json(private / "coverage.json", dict(files={}))
        return [
            dict(
                name="negative_check",
                argv=[str(e.ROOT / ".venv/bin/python"), "-c", "raise SystemExit(1)"],
                deadline_s=30,
                expected_exit=0,
            )
        ]

    monkeypatch.setattr(e, "manifest", manifest)
    monkeypatch.setattr(e, "load_inputs", lambda *a: (v, []))
    config = os.environ.get("CARNOT_8064_COVERAGE_CONFIG")
    monkeypatch.setenv("CARNOT_8064_COVERAGE_CONFIG", config or "")
    output = tmp_path / "checked" / (e.NAME + ".json")
    assert e.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    assert e.replay(output)["passed"]
    v["target_reference"]["sha256"] = "wrong"
    failed = tmp_path / "failed" / (e.NAME + ".json")
    assert e.main(["--output", str(failed)]) == 0
    assert json.loads(failed.read_text())["verdict_class"] == "disqualified"

    def missing_environment(private):
        specs = manifest(private)
        specs[0]["name"] = "python_environment"
        return specs

    monkeypatch.setattr(e, "manifest", missing_environment)
    blocked = tmp_path / "environment" / (e.NAME + ".json")
    assert e.main(["--output", str(blocked)]) == 0
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    if config:
        monkeypatch.setenv("CARNOT_8064_COVERAGE_CONFIG", config)
    else:
        monkeypatch.delenv("CARNOT_8064_COVERAGE_CONFIG", raising=False)


def test_cli_resume(tmp_path):
    """SCENARIO-REPORT-8064-DURABLE: the actual CLI resumes after a candidate commit death."""
    source = tmp_path / "data.json"
    atomic_json(source, data())
    output = tmp_path / "out" / (e.NAME + ".json")
    code = "import os; from carnot import experiment_8064_v698_fresh_feedback_learning as e; original=e.m.Journal.emit\ndef emit(self,kind,row):\n original(self,kind,row)\n if kind=='candidate': os._exit(23)\ne.m.Journal.emit=emit\ne.main(['--fixture-input',os.environ['INPUT'],'--fixture-output',os.environ['OUTPUT']])"
    env = dict(
        os.environ,
        PYTHONPATH=str(e.ROOT / "python") + ":" + str(e.ROOT),
        INPUT=str(source),
        OUTPUT=str(output),
        JAX_PLATFORMS="cpu",
        OPENBLAS_NUM_THREADS="1",
    )
    print("8064 CLI death before", flush=True)
    child = subprocess.run(
        [str(e.ROOT / ".venv/bin/python"), "-c", code], cwd=tmp_path, env=env, timeout=60
    )
    print("8064 CLI death after", child.returncode, flush=True)
    assert child.returncode == 23
    raw = next((output.parent / "raw" / e.NAME).glob("*/plan.json")).parent
    resumed = cli(
        tmp_path / "cwd", "--fixture-input", source, "--fixture-output", output, "--resume", raw
    )
    assert resumed.returncode == 0 and e.replay(output)["passed"]
    other = tmp_path / "scope" / (e.NAME + ".json")
    assert (
        cli(
            tmp_path / "cwd",
            "--fixture-output",
            other,
            "--root",
            tmp_path / "missing",
            "--resume",
            raw,
        ).returncode
        == 1
    )


def test_build_and_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8064-TERMINAL: failed checks disqualify and retention tampering fails."""
    v = data()
    raw = tmp_path / "raw"
    result = m.measure(v, raw / "trajectory")
    e.retention(v, result, raw)
    coverage = {p: dict(summary=dict(missing_lines=0)) for p in e.OWNED}
    receipt = dict(
        name="test",
        passed=True,
        expected_exit=0,
        exit_code=0,
        log_path=str(raw),
        log_sha256="unused",
    )
    good = e.build(v, [], raw, [receipt], coverage, False, e.time.monotonic_ns())
    assert good["learning_trajectory_ready_score"] == 1 and good["verdict_class"] == "null"
    bad = e.build(
        v,
        [],
        raw,
        [dict(receipt, passed=False, exit_code=1)],
        coverage,
        False,
        e.time.monotonic_ns(),
    )
    assert bad["verdict_class"] == "disqualified" and bad["gate_check_summary"]
    path = tmp_path / (e.NAME + ".json")
    atomic_json(
        raw / "plan.json", dict(data=v, failures=[], fixture=False, validation_manifest=["test"])
    )
    atomic_json(raw / "validation.json", dict(receipts=[receipt], coverage=coverage))
    good["raw_shard_hashes"] = [e.reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
    atomic_json(path, good)
    assert e.replay(path)["passed"]
    forged = deepcopy(good)
    forged["verifier_is_oracle"] = True
    atomic_json(path, forged)
    with pytest.raises(ValueError, match="validation_readiness_drift"):
        e.replay(path)
    forged = deepcopy(good)
    forged["learning_trajectory_ready_score"] = 0
    atomic_json(path, forged)
    with pytest.raises(ValueError, match="validation_readiness_drift"):
        e.replay(path)
    forged = deepcopy(good)
    r = forged["retention_rows"][0]
    r["y"] ^= 1
    r["numerator"] = m.loss(r["action"], r["y"])
    r["brier"] = (r["probability"] - r["y"]) ** 2
    atomic_json(path, forged)
    with pytest.raises(ValueError, match="retention_target_drift"):
        e.replay(path)
    for field in ("retention_rows", "final_head_seals", "learning_trajectory_ready_score"):
        changed = deepcopy(good)
        if field == "retention_rows":
            changed[field][0]["brier"] = -1
        elif field == "final_head_seals":
            changed[field][0]["head_hash"] = "wrong"
        else:
            changed["verdict_class"] = "blocked"
        atomic_json(path, changed)
        with pytest.raises(ValueError):
            e.replay(path)
    retained_path = raw / "retention_rows.json"
    original_retained = json.loads(retained_path.read_text())
    for field, bad in (("denominator", 0), ("brier", -1)):
        forged = deepcopy(good)
        forged["retention_rows"][0][field] = bad
        atomic_json(retained_path, dict(rows=forged["retention_rows"]))
        forged["raw_shard_hashes"] = [
            e.reference(Path(r["path"])) for r in good["raw_shard_hashes"]
        ]
        atomic_json(path, forged)
        with pytest.raises(ValueError):
            e.replay(path)
    atomic_json(retained_path, original_retained)
    changed = deepcopy(v)
    changed["labels"]["0"] ^= 1
    alternate = tmp_path / "alternate"
    altered = m.measure(changed, alternate / "trajectory")
    e.retention(changed, altered, alternate)
    atomic_json(
        alternate / "plan.json",
        dict(data=v, failures=[], fixture=False, validation_manifest=["test"]),
    )
    atomic_json(alternate / "validation.json", dict(receipts=[receipt], coverage=coverage))
    forged = e.build(v, [], alternate, [receipt], coverage, False, e.time.monotonic_ns())
    atomic_json(path, forged)
    with pytest.raises(ValueError, match="released_target_drift"):
        e.replay(path)
    sealpath = raw / "retention_predictions.json"
    seal = json.loads(sealpath.read_text())
    seal["retention_labels_opened"] = True
    atomic_json(sealpath, seal)
    good["raw_shard_hashes"] = [e.reference(Path(r["path"])) for r in good["raw_shard_hashes"]]
    atomic_json(path, good)
    with pytest.raises(ValueError, match="retention_seal_drift"):
        e.replay(path)
