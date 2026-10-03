"""REQ-REPORT-8052: independent equations must reject unsafe evidence."""

import copy
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import numpy as np
import pytest

from carnot import experiment_8052_v697_learning_benefit_audit as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import feedback_constrained_8051 as producer
from carnot.verify import learning_benefit_8052 as a
from test_feedback_constrained_8051 import data


def test_independent_replay(tmp_path):
    """SCENARIO-REPORT-8052-REPLAY: every event and checkpoint is reconstructed."""
    inputs = data(96)
    inputs["labels"]["3"] = None
    inputs["sources"][5]["public_eligible"] = False
    raw = tmp_path / "trajectory"
    producer.measure(inputs, raw)
    r = a.replay(raw, inputs["labels"])
    assert len(r["rows"]) == 288
    assert r["sample_size_budget"]["censored"] == 20
    assert r["numerical_agreement"]["candidate_count"] > 0
    assert a.replay(raw, inputs["labels"]) == r
    db = sqlite3.connect(raw / "seed-101/ledger.sqlite")
    seq, payload = db.execute(
        "select seq,payload from events where kind='issue' limit 1"
    ).fetchone()
    row = json.loads(payload)
    row["probability"] = 0.99
    db.execute("update events set payload=? where seq=?", (json.dumps(row), seq))
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="probability"):
        a.replay(raw, inputs["labels"])


def test_guard_equations():
    """SCENARIO-REPORT-8052-REPLAY: alpha and rollback use separate equations."""
    head = dict(calibration=[0.0, 1.0])
    theta = np.zeros(1)
    x = np.tile([[-1.0], [1.0]], (4, 1))
    y = np.tile([0, 1], 4)
    for delta in [np.ones(1), -10 * np.ones(1), theta]:
        for arm in a.ARMS[:2]:
            assert a.guard(head, theta, delta, theta, x, y, arm) == producer.guard(
                head, theta, delta, theta, x, y, arm
            )
    assert a.guard(head, -np.ones(1), theta, theta, x, y, a.ARMS[1])["reset"]
    assert a.guard(head, theta, theta, theta, x[:2], y[:2], a.ARMS[1])["status"] == "waiting_guard"


def science_rows(gain=True):
    """Private controls expose source pairing without giving natural benefit credit."""
    rows = []
    for arm in a.ARMS:
        for seed in [101, 102]:
            for slot in range(256):
                y = slot % 2
                c = 0.0 if gain and arm == a.ARMS[1] else 0.5
                rows.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        slot=slot,
                        source_cluster_id=str(slot),
                        y=y,
                        eligible=slot >= 40 and slot < 236,
                        feedback_role="guard" if slot % 4 == 0 else "update",
                        typed_cost=c,
                        brier=c,
                        action="reject" if arm == a.ARMS[1] else "escalate",
                        false_accept=0,
                        post_first_update=slot >= 40,
                    )
                )
    retention = [
        dict(
            arm=arm,
            seed=seed,
            source_cluster_id=str(slot),
            y=slot % 2,
            eligible=True,
            cost_drift=0.0,
            brier_drift=0.0,
        )
        for arm in a.ARMS
        for seed in [101, 102]
        for slot in range(64)
    ]
    return rows, retention


def test_science():
    """SCENARIO-REPORT-8052-SCIENCE: masks, support and safety remain separate."""
    rows, retained = science_rows()
    r = a.compare(rows, retained)
    assert r["benefit_ready_score"] == 1
    assert r["primary_hypothesis_results"][0]["slot_count"] == 256
    assert len(r["per_seed_false_accept_rows"]) == 4
    rows[0]["false_accept"] = 1
    for row in retained:
        if row["arm"] == a.ARMS[0]:
            row["cost_drift"] = 0.03
    assert a.compare(rows, retained)["benefit_ready_score"] == 0
    null, retained = science_rows(False)
    assert a.compare(null, retained)["benefit_ready_score"] == 0
    assert a.compare([], [])["support_passed"] is False


def cli(root, *args):
    """Real process exits qualify the CLI while keeping all scratch private."""
    root.mkdir(parents=True, exist_ok=True)
    command = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.SCRIPT), *map(str, args)]
    config = os.environ.get("CARNOT_8052_COVERAGE_CONFIG")
    if config:
        command[1:2] = [
            "-m",
            "coverage",
            "run",
            "--rcfile=" + config,
            "--data-file=" + str(Path(config).parent / ".coverage"),
        ]
    print("8052 subprocess before", command, flush=True)
    r = subprocess.run(
        command,
        cwd=root,
        env=dict(os.environ, PYTHONUNBUFFERED="1", CARNOT_EXPERIMENT_ARTIFACT_ROOT=str(root)),
        capture_output=True,
        text=True,
        timeout=240,
    )
    print("8052 subprocess after", r.returncode, flush=True)
    return r


def test_cli_blocked(tmp_path):
    """SCENARIO-REPORT-8052-TERMINAL: external absence is terminal blocked."""
    r = cli(tmp_path, "--root", tmp_path, "--validation-worker")
    assert r.returncode == 0, r.stdout + r.stderr
    output = tmp_path / (e.NAME + ".json")
    v = json.loads(output.read_text())
    assert v["verdict_class"] == "blocked"
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    v["generalized_learning_benefit_score"] = 1
    atomic_json(output, v)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1


def bundle_fixture(root, n=256):
    """Private released targets exercise publication without importing natural evidence."""
    inputs = data(n)
    raw = root / "trajectory"
    producer.measure(inputs, raw)
    target = root / "stream-target.json"
    atomic_json(
        target, dict(rows=[dict(family_id=k, eligible_y=v) for k, v in inputs["labels"].items()])
    )
    public = [
        dict(r, family_id="retention-" + str(i), source_cluster_id="retention-" + str(i))
        for i, r in enumerate(inputs["sources"][:64])
    ]
    vault = root / "retention-target.json"
    atomic_json(
        vault,
        dict(rows=[dict(family_id=r["family_id"], eligible_y=i % 2) for i, r in enumerate(public)]),
    )
    bundle = dict(
        trajectory=str(raw),
        target_reference=e.reference(target),
        retention_public=public,
        retention_target=e.reference(vault),
        references=[],
        gate_checks=[],
    )
    path = root / "bundle.json"
    atomic_json(path, bundle)
    return inputs, bundle, path


def test_cli_valid_control(tmp_path):
    """SCENARIO-REPORT-8052-TERMINAL: actual valid/null/tampered CLI and normal exits."""
    inputs, bundle, path = bundle_fixture(tmp_path)
    result = cli(tmp_path / "published", "--fixture-bundle", path, "--validation-worker")
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "published" / (e.NAME + ".json")
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["benefit_ready_score"] == 0
    assert len(value["recovery_rows"]) == 16
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value["rows"][0]["typed_cost"] = 99
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    source = tmp_path / "recovery.json"
    atomic_json(
        source, dict(head=inputs["head"], source=inputs["sources"][0], arm=a.ARMS[1], delta=0.0)
    )
    assert cli(tmp_path, "--recovery-worker", source).returncode == 1


def bind(root, identity, name, content):
    """Minimal byte-bound primary fixture uses the same terminal receipt contract."""
    path = root / "results" / (name + ".json")
    terminal = root / f"terminal-{identity}.json"
    report = root / f"report-{identity}.json"
    atomic_json(
        path,
        dict(
            content,
            experiment_id=identity,
            task_id=f"exp{identity}-private",
            flagged_adversarial=False,
            terminal_validation_sidecar_path=str(terminal),
        ),
    )
    binding = dict(
        primary_path=str(path), primary_sha256=e.sha256_file(path), sidecar_path=str(report)
    )
    atomic_json(terminal, dict(publication=binding))
    atomic_json(report, dict(binding, report=dict(passed=True)))
    return path


def upstream_audit_fixture(root):
    """Build guarded upstream contracts for the actual private null CLI route."""
    from test_feedback_constrained_8051 import upstream_fixture

    upstream_fixture(root)
    loaded, failures = e.upstream.load_inputs(root, root / "loaded")
    assert not failures
    producer.measure(loaded, root / "trajectory")
    public = [
        dict(r, family_id="retention-" + str(i), source_cluster_id="retention-" + str(i))
        for i, r in enumerate(loaded["sources"][:64])
    ]
    public_path = root / "retention-public.json"
    atomic_json(public_path, dict(rows=public))
    vault = root / "retention-target.json"
    atomic_json(
        vault,
        dict(rows=[dict(family_id=r["family_id"], eligible_y=i % 2) for i, r in enumerate(public)]),
    )
    path = root / "results" / (e.upstream.m.protocol.NAME + ".json")
    content = json.loads(path.read_text())
    content.update(
        role_manifests=dict(retention=e.reference(public_path)),
        historical_exposure=dict(artificial=True),
    )
    bind(root, 8046, e.upstream.m.protocol.NAME, content)
    path = root / "results" / (e.upstream.prior.upstream.NAME + ".json")
    content = json.loads(path.read_text())
    content["role_manifests"] = dict(
        evaluator=dict(stream=loaded["target_reference"], retention=e.reference(vault))
    )
    bind(root, 8032, e.upstream.prior.upstream.NAME, content)
    bind(
        root,
        8051,
        e.upstream.NAME,
        dict(
            learning_trajectory_ready_score=1,
            retained_labels_opened=False,
            trajectory_directory=str(root / "trajectory"),
            raw_shard_hashes=[
                e.reference(p) for p in (root / "trajectory").rglob("*") if p.is_file()
            ],
        ),
    )
    bind(
        root,
        8039,
        "experiment_8039_v696_learning_benefit_audit",
        dict(
            learning_audit_ready_score=1,
            honest_verdict="complete_null_private",
            primary_hypothesis_results=[],
        ),
    )


def test_inputs_and_null_cli(tmp_path):
    """SCENARIO-REPORT-8052-REPLAY: authenticated contracts permit a valid null audit."""
    root = tmp_path / "upstream"
    upstream_audit_fixture(root)
    bundle, failures = e.load_inputs(root, tmp_path / "custody")
    assert not failures and bundle["learner_access_boundary"]["retained_labels_opened"] is False
    result = cli(tmp_path / "null", "--root", root, "--validation-worker")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads((tmp_path / "null" / (e.NAME + ".json")).read_text())
    assert value["verdict_class"] == "null"
    primary = root / "results" / (e.upstream.NAME + ".json")
    content = json.loads(primary.read_text())
    content["learning_trajectory_ready_score"] = 0
    bind(root, 8051, e.upstream.NAME, content)
    assert e.load_inputs(root, tmp_path / "bad")[1]
    content["learning_trajectory_ready_score"] = 1
    content["retained_labels_opened"] = True
    bind(root, 8051, e.upstream.NAME, content)
    assert e.load_inputs(root, tmp_path / "access")[1]


@pytest.mark.parametrize("mode", ["passed", "failed", "absent", "blocked", "positive", "support"])
def test_validation_outcomes(tmp_path, monkeypatch, mode):
    """SCENARIO-REPORT-8052-TERMINAL: owned exits and coverage govern readiness."""
    _, bundle, _ = bundle_fixture(tmp_path, 64 if mode == "support" else 256)
    (tmp_path / "output").mkdir()
    monkeypatch.setenv("CARNOT_EXPERIMENT_ARTIFACT_ROOT", str(tmp_path / "output"))
    failures = [] if mode != "blocked" else [dict(passed=False)]
    monkeypatch.setattr(e, "load_inputs", lambda root, raw: (bundle, failures))

    def recover(head, scratch, durable, root, script):
        rows = [dict(passed=True)]
        atomic_json(durable / "rows.json", dict(rows=rows))
        return rows

    monkeypatch.setattr(e.recovery, "recover", recover)
    monkeypatch.setattr(e, "terminal", lambda path: dict(passed=True))
    if mode == "positive":
        original = e.a.compare

        def positive(rows, retained):
            return dict(original(rows, retained), benefit_ready_score=1)

        monkeypatch.setattr(e.a, "compare", positive)

    def commands(root, plan, **kwargs):
        config = Path(kwargs["extra_env"]["CARNOT_8052_COVERAGE_CONFIG"])
        if mode != "absent":
            atomic_json(
                config.parent / "coverage.json",
                dict(
                    files={
                        p: dict(summary=dict(missing_lines=0, num_statements=1)) for p in e.OWNED
                    }
                ),
            )
        return [
            dict(scope="owned", passed=mode != "failed", exit_code=0 if mode != "failed" else 1),
            dict(scope="repository_health", passed=False, exit_code=1),
        ]

    monkeypatch.setattr(e, "run_commands", commands)
    assert e.main(["--root", str(tmp_path)]) == 0
    value = json.loads((tmp_path / "output" / (e.NAME + ".json")).read_text())
    expected = (
        "blocked"
        if mode in ("blocked", "support")
        else "disqualified"
        if mode in ("failed", "absent")
        else "positive"
        if mode == "positive"
        else "null"
    )
    assert value["verdict_class"] == expected
    assert value["learning_audit_ready_score"] == int(mode in ("passed", "positive"))
    if mode == "passed":
        value["learning_audit_ready_score"] = 1
        value["acceptance_gate_results"]["owned_checks"] = False
        value["raw_shard_hashes"] = []
        value["code_config_hashes"] = []
        value["acceptance_gate_results"]["validity"] = False
        atomic_json(tmp_path / "bad.json", value)
        with pytest.raises(ValueError, match="readiness"):
            e.replay(tmp_path / "bad.json")


@pytest.mark.parametrize("mode", ["timeout", "missing"])
def test_recovery_failures(tmp_path, monkeypatch, mode):
    """SCENARIO-REPORT-8052-RECOVERY: absent boundaries close failed children."""
    from carnot.reporting import learning_recovery_8052 as recovery

    original = recovery.subprocess.Popen

    def popen(argv, **kwargs):
        if "--boundary" in argv and mode == "missing":
            argv = [str(e.ROOT / ".venv/bin/python"), "-u", "-c", 'print("NO_BOUNDARY",flush=True)']
        return original(argv, **kwargs)

    monkeypatch.setattr(recovery.subprocess, "Popen", popen)
    with pytest.raises((TimeoutError, ValueError), match="recovery_boundary"):
        recovery.recover(
            data()["head"],
            tmp_path / "scratch",
            tmp_path / "durable",
            e.ROOT,
            e.SCRIPT,
            boundary_timeout_s=-1 if mode == "timeout" else 60,
        )


def test_replay_event_and_budget_failures(tmp_path):
    """SCENARIO-REPORT-8052-REPLAY: original event errors and budget remain explicit."""
    inputs = data(48)
    raw = tmp_path / "trajectory"
    producer.measure(inputs, raw)
    with pytest.raises(ValueError, match="budget"):
        a.replay(raw, inputs["labels"], budget_s=-1)
    db = sqlite3.connect(raw / "seed-101/ledger.sqlite")
    db.execute("update events set kind='unknown' where seq=1")
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="unknown_event"):
        a.replay(raw, inputs["labels"])


def test_retention_unknown_and_contract(tmp_path):
    """SCENARIO-REPORT-8052-SCIENCE: unknown retention targets remain exclusions."""
    inputs, bundle, _ = bundle_fixture(tmp_path, 64)
    replayed = a.replay(Path(bundle["trajectory"]), inputs["labels"])
    vault = Path(bundle["retention_target"]["path"])
    targets = json.loads(vault.read_text())
    targets["rows"][0]["eligible_y"] = None
    atomic_json(vault, targets)
    bundle["retention_target"] = e.reference(vault)
    rows = a.retention(bundle, replayed, tmp_path / "seal.json")
    assert not rows[0]["eligible"]
    assert a.compare(replayed["rows"], [])["retention_passed"] is False
    targets["rows"][0]["eligible_y"] = 3
    atomic_json(vault, targets)
    bundle["retention_target"] = e.reference(vault)
    with pytest.raises(ValueError, match="retention_target"):
        a.retention(bundle, replayed, tmp_path / "seal2.json")


def test_missing_resources(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8052-TERMINAL: named absent tools are exact failed operands."""
    monkeypatch.setattr(e, "ROOT", tmp_path)
    _, failures = e.load_inputs(tmp_path, tmp_path / "raw")
    assert any(
        r["artifact_field"] == "resource_exists" and r["observed"] is False for r in failures
    )


def test_worker_stops_and_idempotence(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8052-RECOVERY: stop requests and restart custody are checked."""
    from carnot.reporting import learning_recovery_8052 as recovery

    calls = []
    monkeypatch.setattr(recovery.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    inputs = data()
    source = tmp_path / "worker.json"
    atomic_json(
        source, dict(head=inputs["head"], source=inputs["sources"][0], arm=a.ARMS[1], delta=10.0)
    )
    for stage in recovery.STAGES:
        for side in ("before", "after"):
            directory = tmp_path / stage / side
            value = recovery.worker(source, directory, stage + "/" + side)
            assert value["exactly_once"]
            assert recovery.worker(source, directory, "none") == value
    assert len(calls) == 8 and all(sig == recovery.signal.SIGSTOP for _, sig in calls)
    value = json.loads(source.read_text())
    value["delta"] = 0.0
    atomic_json(source, value)
    with pytest.raises(ValueError, match="recovery_stage"):
        recovery.worker(source, tmp_path / "commit" / "after", "none")
