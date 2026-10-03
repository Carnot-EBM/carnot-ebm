"""REQ-REPORT-8051: released-only guards and persistent, byte-bound trajectories."""

import copy
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import numpy as np
import pytest

from carnot import experiment_8051_v697_feedback_constrained_learning as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import feedback_constrained_8051 as m
from test_causal_online_8025 import fixture


def data(n=96):
    """Alternating private guard classes make acceptance testable without natural claims."""
    value = fixture(n)
    for i, row in enumerate(value["sources"]):
        row["source_cluster_id"] = str(i)
    return value


def test_guard():
    """SCENARIO-REPORT-8051-GUARD: reject damage, accept benefit and preserve no-ops."""
    head = data()["head"]
    x = np.zeros((8, 110))
    x[:, 0] = np.tile([-1.0, 1.0], 4)
    y = np.tile([0, 1], 4)
    initial = np.zeros(110)
    delta = initial.copy()
    delta[0] = 1
    good = m.guard(head, initial, delta, initial, x, y, "feedback_constrained")
    assert good["alpha"] == 1 and not good["reset"]
    bad = m.guard(head, initial, -delta * 10, initial, x, y, "feedback_constrained")
    assert bad["alpha"] == 0 and bad["rejected"]
    assert any("brier" in r["reasons"] for r in bad["diagnostics"])
    no_op = m.guard(head, initial, initial, initial, x, y, "feedback_constrained")
    assert no_op["alpha"] == 1
    reset = m.guard(head, -delta, initial, initial, x, y, "feedback_constrained")
    assert reset["reset"] and reset["parameters"] == initial.tolist()
    free = m.guard(head, initial, -delta, initial, x, y, "unconstrained")
    assert (
        free["alpha"] == 1
        and free["diagnostics"]
        == m.guard(head, initial, -delta, initial, x, y, "feedback_constrained")["diagnostics"]
    )
    for labels in (np.array([]), np.zeros(8)):
        assert (
            m.guard(
                head, initial, delta, initial, x[: len(labels)], labels, "feedback_constrained"
            )["status"]
            == "waiting_guard"
        )
    assert m.controls()["passed"]


def test_trajectory(tmp_path):
    """SCENARIO-REPORT-8051-CAUSAL: shared attempts exclude guards and unreleased labels."""
    value = data()
    value["labels"]["3"] = None
    value["sources"][5]["public_eligible"] = False
    raw = tmp_path / "first"
    result = m.measure(value, raw)
    assert m.reduce(raw) == result
    assert len(result["issue_release_rows"]) == 3 * 96
    assert result["censored_count"] == 20
    guards = {
        r["family_id"] for r in result["guard_partition_rows"] if r["feedback_role"] == "guard"
    }
    assert all(r["family_id"] not in guards for r in result["candidate_update_rows"])
    budgets = result["attempted_gradient_counts"]
    assert budgets[0]["count"] == budgets[1]["count"] > 0 and budgets[2]["count"] == 0
    changed = copy.deepcopy(value)
    changed["labels"]["60"] ^= 1
    other = m.measure(changed, tmp_path / "other")
    assert [
        (r["probability"], r["head_hash"]) for r in result["issue_release_rows"] if r["slot"] <= 80
    ] == [
        (r["probability"], r["head_hash"]) for r in other["issue_release_rows"] if r["slot"] <= 80
    ]
    for field in ("candidate_update_rows", "guard_check_rows"):
        assert [
            {k: v for k, v in r.items() if k != "hot_update_ns"}
            for r in result[field]
            if r["slot"] < 80
        ] == [
            {k: v for k, v in r.items() if k != "hot_update_ns"}
            for r in other[field]
            if r["slot"] < 80
        ]
    db = sqlite3.connect(raw / "ledger.sqlite")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            'INSERT INTO events(kind,identity,payload) SELECT kind,identity,payload FROM events WHERE kind="release" LIMIT 1'
        )
    db.close()
    with pytest.raises(TimeoutError, match="numerical_budget"):
        m.measure(data(), tmp_path / "timeout", budget_s=-1)
    db = sqlite3.connect(raw / "ledger.sqlite")
    seq, text = db.execute('SELECT seq,payload FROM events WHERE kind="issue" LIMIT 1').fetchone()
    row = json.loads(text)
    row["probability"] = 0.99
    db.execute("UPDATE events SET payload=? WHERE seq=?", (json.dumps(row), seq))
    db.commit()
    db.close()
    with pytest.raises(ValueError):
        m.reduce(raw)


def cli(root, *args):
    """Run a real guarded child with optional subprocess statement coverage."""
    root.mkdir(parents=True, exist_ok=True)
    command = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.SCRIPT), *map(str, args)]
    config = os.environ.get("CARNOT_8051_COVERAGE_CONFIG")
    if config:
        command[1:2] = ["-m", "coverage", "run", "--rcfile=" + config]
    print("8051 test subprocess before", command, flush=True)
    result = subprocess.run(
        command,
        cwd=root,
        env=dict(os.environ, PYTHONUNBUFFERED="1", CARNOT_EXPERIMENT_ARTIFACT_ROOT=str(root)),
        text=True,
        capture_output=True,
        timeout=120,
    )
    print("8051 test subprocess after exit", result.returncode, flush=True)
    return result


def test_cli(tmp_path):
    """SCENARIO-REPORT-8051-PUBLICATION: valid, blocked, null and tampered actual exits."""
    source = tmp_path / "fixture.json"
    atomic_json(source, data(64))
    run = tmp_path / "run"
    result = cli(run, "--fixture-input", source, "--validation-worker")
    assert result.returncode == 0, result.stdout + result.stderr
    output = run / (e.NAME + ".json")
    v = json.loads(output.read_text())
    assert v["verdict_class"] == "circular_positive"
    assert cli(run, "--cold-replay", output).returncode == 0
    assert e.terminal(output)["passed"]
    v["rows"][0]["brier"] = 999
    atomic_json(output, v)
    assert cli(run, "--cold-replay", output).returncode == 1
    assert cli(run, "--date", "20261001").returncode == 2
    blocked = tmp_path / "blocked"
    result = cli(blocked, "--root", tmp_path / "missing", "--validation-worker")
    assert result.returncode == 0, result.stdout + result.stderr
    b = json.loads((blocked / (e.NAME + ".json")).read_text())
    assert b["verdict_class"] == "blocked" and b["gate_check_summary"]
    with pytest.raises(ValueError, match="unsafe_readiness"):
        b["learning_trajectory_ready_score"] = 1
        atomic_json(blocked / (e.NAME + ".json"), b)
        e.replay(blocked / (e.NAME + ".json"))


def upstream_fixture(root):
    """Bind private originals with real sidecar hashes for the null CLI route."""
    from test_windowed_online_8038 import source_fixture

    value, _ = source_fixture(root)
    value["head"]["converged"] = True
    from carnot.verify.evidence_features_7980 import normalized

    for row in value["sources"]:
        row["source_cluster_id"] = normalized(bytes.fromhex(row["source_bytes"]))
    head_path = root / "head.json"
    atomic_json(head_path, value["head"])
    for identity, name, content in [
        (
            8020,
            "experiment_8020_v695_qualified_energy_fit",
            dict(energy_fit_ready_score=1, head_checkpoints=[e.reference(head_path)]),
        ),
        (
            8046,
            m.protocol.NAME,
            dict(
                learning_protocol_ready_score=1,
                qualified_head=value["head"],
                guard_partition_rows=m.partition_rows(value["sources"]),
            ),
        ),
    ]:
        path = root / "results" / (name + ".json")
        sidecar = root / f"terminal-{identity}.json"
        report = root / f"report-{identity}.json"
        atomic_json(
            path,
            dict(
                content,
                experiment_id=identity,
                flagged_adversarial=False,
                terminal_validation_sidecar_path=str(sidecar),
            ),
        )
        binding = dict(
            primary_path=str(path), primary_sha256=e.sha256_file(path), sidecar_path=str(report)
        )
        atomic_json(sidecar, dict(publication=binding))
        atomic_json(report, dict(binding, report=dict(passed=True)))
    return value


def test_inputs_null_and_contracts(tmp_path):
    """SCENARIO-REPORT-8051-PUBLICATION: authentic bytes enable the real null path."""
    root = tmp_path / "upstream"
    upstream_fixture(root)
    loaded, failures = e.load_inputs(root, tmp_path / "custody")
    assert not failures and len(loaded["sources"]) == 256
    result = cli(tmp_path / "null", "--root", root, "--validation-worker")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads((tmp_path / "null" / (e.NAME + ".json")).read_text())
    assert value["verdict_class"] == "null" and value["completed_count"] == 236
    path = root / "results" / ("experiment_8020_v695_qualified_energy_fit.json")
    atomic_json(path, {})
    assert e.load_inputs(root, tmp_path / "bad-contract")[1]


@pytest.mark.parametrize("mode", ["pass", "fail", "absent", "blocked"])
def test_validation_branches(tmp_path, monkeypatch, mode):
    """REQ-REPORT-8051: private validation outcomes alone govern readiness."""
    monkeypatch.setenv("CARNOT_EXPERIMENT_ARTIFACT_ROOT", str(tmp_path))
    failures = (
        []
        if mode != "blocked"
        else [
            dict(
                check_name="missing",
                artifact_field="missing",
                expected=True,
                observed=False,
                passed=False,
            )
        ]
    )
    monkeypatch.setattr(e, "load_inputs", lambda root, raw: (data(32), failures))
    monkeypatch.setattr(e, "terminal", e.replay)

    def validation(root, commands, **kwargs):
        config = Path(kwargs["extra_env"]["CARNOT_8051_COVERAGE_CONFIG"])
        if mode != "absent":
            atomic_json(
                config.parent / "coverage.json",
                dict(files={p: dict(summary=dict(missing_lines=0)) for p in e.OWNED}),
            )
        return [
            dict(scope="owned", passed=mode == "pass", exit_code=int(mode != "pass")),
            dict(scope="repository_health", passed=False, exit_code=1),
        ]

    monkeypatch.setattr(e, "run_commands", validation)
    assert e.main([]) == 0
    value = json.loads((tmp_path / (e.NAME + ".json")).read_text())
    assert value["learning_trajectory_ready_score"] == int(mode == "pass")
    assert value["verdict_class"] == (
        "null" if mode == "pass" else "blocked" if mode == "blocked" else "disqualified"
    )


def test_raw_failures_and_empty(tmp_path):
    """REQ-REPORT-8051: unknown journal events and unavailable targets fail closed."""
    assert m.measure(data(20), tmp_path / "empty")["eligible_count"] == 0
    value = data(48)
    value["labels"]["0"] = 2
    with pytest.raises(ValueError, match="label_contract"):
        m.measure(value, tmp_path / "label")
    raw = tmp_path / "unknown"
    m.measure(data(48), raw)
    db = sqlite3.connect(raw / "ledger.sqlite")
    db.execute("UPDATE events SET kind='unknown' WHERE seq=1")
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="unknown_event"):
        m.reduce(raw)


@pytest.mark.parametrize("mode", ["partition", "missing"])
def test_head_contract_failures(tmp_path, mode):
    """REQ-REPORT-8051: original bytes and sealed partition are mandatory operands."""
    root = tmp_path / "upstream"
    upstream_fixture(root)
    path = root / "results" / (m.protocol.NAME + ".json")
    value = json.loads(path.read_text())
    if mode == "partition":
        value["guard_partition_rows"] = []
    else:
        del value["qualified_head"]
    atomic_json(path, value)
    sidecar = Path(value["terminal_validation_sidecar_path"])
    binding = json.loads(sidecar.read_text())["publication"]
    binding["primary_sha256"] = e.sha256_file(path)
    atomic_json(sidecar, dict(publication=binding))
    atomic_json(Path(binding["sidecar_path"]), dict(binding, report=dict(passed=True)))
    _, failures = e.load_inputs(root, tmp_path / "custody")
    assert failures and not failures[0]["passed"]


def test_missing_local_resources(tmp_path, monkeypatch):
    """REQ-REPORT-8051: absent tools name their exact operands rather than measured zeros."""
    monkeypatch.setattr(e, "ROOT", tmp_path)
    _, failures = e.load_inputs(tmp_path, tmp_path / "custody")
    assert any(r["path"].endswith(".venv/bin/python") and r["observed"] is False for r in failures)
