"""REQ-REPORT-8039: independent arithmetic and private process evidence."""

import copy
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import numpy as np
import pytest

from carnot import experiment_8039_v696_learning_benefit_audit as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import learning_benefit_8039 as a
from carnot.verify import windowed_online_8038 as producer
from test_causal_online_8025 import fixture


def bundle(tmp_path, n=96):
    """Private primitive rows exercise the real writer without natural credit."""
    data = fixture(n)
    data["labels"]["3"] = None
    data["sources"][5]["public_eligible"] = False
    producer.measure(data, tmp_path / "trajectory")
    target = tmp_path / "retention.json"
    public = copy.deepcopy(fixture(64)["sources"])
    public[5]["public_eligible"] = False
    atomic_json(
        target,
        dict(rows=[dict(family_id=r["family_id"], eligible_y=int(r["slot"] % 2)) for r in public]),
    )
    protocol = tmp_path / "audit_protocol.json"
    atomic_json(protocol, dict(config=a.CONFIG, frozen=True, identity=8039))
    return dict(
        trajectory=str(tmp_path / "trajectory"),
        retention_public=public,
        fit_public=data["sources"],
        retention_target=e.reference(target),
        audit_protocol=e.reference(protocol),
    )


def cli(tmp_path, *args):
    """Run actual script branches with artifact guard and owned coverage."""
    command = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.SCRIPT), *map(str, args)]
    config = os.environ.get("CARNOT_8039_COVERAGE_CONFIG")
    if config:
        command[1:2] = [
            "-m",
            "coverage",
            "run",
            "--rcfile=" + config,
            "--data-file=" + str(Path(config).parent / ".coverage"),
        ]
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        JAX_PLATFORMS="cpu",
        CARNOT_EXPERIMENT_ARTIFACT_ROOT=str(tmp_path),
    )
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=90
    )


def test_replay_and_retention(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8039-REPLAY, SCENARIO-REPORT-8039-RETENTION."""
    b = bundle(tmp_path)
    monkeypatch.setattr(producer, "reduce", lambda *args: pytest.fail("producer reducer called"))
    r = a.replay_trajectory(Path(b["trajectory"]))
    assert r["numerical_agreement"]["coefficient_max_error"] <= 1e-10
    assert r["numerical_agreement"]["probability_max_error"] <= 1e-10
    assert len(r["independent_replay_rows"]) == 48
    assert len(r["rows"]) == 384
    assert r["sample_size_budget"]["censored"] == 20
    assert r["sample_size_budget"]["excluded"] == 2
    seal = tmp_path / "prediction_seal.json"
    retention = a.retention(b, r, seal)
    assert json.loads(seal.read_text())["retention_labels_opened"] is False
    assert retention["retention_support"]["eligible"] == 63
    assert all(x["passed"] for x in retention["prefreeze_access_checks"])
    result = a.compare(r["rows"], retention["retention_rows"])
    assert result["primary_hypothesis_results"][0]["hypothesis"] == "H3"
    assert result["effective_independent_streams"] == 1
    assert result["learning_benefit_score"] == 0
    assert all(x["margin"] == 0.02 for x in result["primary_hypothesis_results"])
    assert a.bootstrap([float("nan")] * 3, 32)["raw_p"] == 1
    positive = a.bootstrap([0.1] * 192, 32)
    assert positive["gain"] == pytest.approx(0.1) and positive["raw_p"] < 0.05
    assert a.bootstrap([0.02] * 192, 16)["raw_p"] == 1
    assert a.controls()["passed"]


@pytest.mark.parametrize("mode", ["update", "order", "duplicate", "issue", "missing_final"])
def test_primitive_mutations(tmp_path, mode):
    """SCENARIO-REPORT-8039-REPLAY: changed bytes cannot masquerade as evidence."""
    b = bundle(tmp_path)
    raw = Path(b["trajectory"])
    db = sqlite3.connect(raw / "ledger.sqlite")
    if mode == "duplicate":
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO events(kind,identity,payload) SELECT kind,identity,payload FROM events LIMIT 1"
            )
    elif mode == "missing_final":
        db.execute("DELETE FROM events WHERE kind='final'")
    else:
        kind = "commit" if mode == "update" else "release" if mode == "order" else "issue"
        seq, text = db.execute(
            "SELECT seq,payload FROM events WHERE kind=? LIMIT 1", (kind,)
        ).fetchone()
        row = json.loads(text)
        if mode == "update":
            row["gradients"][0]["after_coefficients"][0] += 0.5
        elif mode == "order":
            row["origin_slot"] = 1
        else:
            row["probability"] += 0.5
        db.execute("UPDATE events SET payload=? WHERE seq=?", (json.dumps(row), seq))
    db.commit()
    db.close()
    if mode != "duplicate":
        # Remove only this private seal's hash protection to test arithmetic rejection too.
        atomic_json(raw / "seal.json", dict(sealed=True, references=[]))
        with pytest.raises(ValueError):
            a.replay_trajectory(raw)


def test_future_labels(tmp_path):
    """SCENARIO-REPORT-8039-REPLAY: a future label cannot change earlier issues."""
    data = fixture(128)
    producer.measure(data, tmp_path / "first")
    first = a.replay_trajectory(tmp_path / "first")
    selected = next(
        r
        for r in first["independent_replay_rows"]
        if int(r["family_id"]) >= 20 and r["arm"] == "recent64"
    )
    origin = int(selected["family_id"])
    data["labels"][str(origin)] ^= 1
    producer.measure(data, tmp_path / "second")
    second = a.replay_trajectory(tmp_path / "second")
    one, two = first["rows"], second["rows"]
    assert [(r["probability"], r["action"]) for r in one if r["slot"] <= origin + 20] == [
        (r["probability"], r["action"]) for r in two if r["slot"] <= origin + 20
    ]
    assert first["final_states"] != second["final_states"]


def test_private_cli_and_recovery(tmp_path):
    """SCENARIO-REPORT-8039-RECOVERY, SCENARIO-REPORT-8039-PUBLICATION."""
    b = bundle(tmp_path)
    src = tmp_path / "bundle.json"
    atomic_json(src, b)
    result = cli(tmp_path, "--fixture-bundle", src, "--validation-worker")
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / (e.NAME + ".json")
    v = json.loads(output.read_text())
    assert v["verdict_class"] == "circular_positive" and v["learning_audit_ready_score"] == 0
    assert len(v["recovery_rows"]) == 6 and all(r["passed"] for r in v["recovery_rows"])
    assert all(
        Path(r["path"]).is_relative_to(tmp_path / "raw" / e.NAME) for r in v["raw_shard_hashes"]
    )
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    assert e.terminal(output)["passed"]
    v["later_loss_rows"][0]["cost"] += 1
    atomic_json(output, v)
    bad = cli(tmp_path, "--cold-replay", output)
    assert bad.returncode == 1 and "reduction_drift" in bad.stdout
    assert cli(tmp_path, "--date", "20261001").returncode == 2


def test_external_block(tmp_path):
    """SCENARIO-REPORT-8039-PUBLICATION: external absence is terminal blocked."""
    result = cli(tmp_path, "--root", tmp_path / "absent", "--validation-worker")
    assert result.returncode == 0, result.stdout + result.stderr
    v = json.loads((tmp_path / (e.NAME + ".json")).read_text())
    assert v["verdict_class"] == "blocked" and v["learning_audit_ready_score"] == 0
    assert v["gate_check_summary"][0]["observed"] is False


def test_prefreeze_access(tmp_path):
    """SCENARIO-REPORT-8039-RETENTION: invalid freeze rejects before vault access."""
    b = bundle(tmp_path)
    replayed = a.replay_trajectory(Path(b["trajectory"]))
    protocol = Path(b["audit_protocol"]["path"])
    atomic_json(protocol, dict(config=a.CONFIG, frozen=False, identity=8039))
    b["audit_protocol"] = e.reference(protocol)
    b["retention_target"] = dict(path=str(tmp_path / "never_opened"), sha256="missing")
    with pytest.raises(ValueError, match="protocol_before_targets"):
        a.retention(b, replayed, tmp_path / "seal.json")


def test_admission_contracts(tmp_path):
    """REQ-REPORT-8039: exact missing and prefreeze operands remain terminal."""
    from test_windowed_online_8038 import source_fixture, bind

    root = tmp_path / "upstream"
    data, path = source_fixture(root)
    v = json.loads(path.read_text())
    protocol = Path(v["methods_reference"]["path"])
    p = json.loads(protocol.read_text())
    p["frozen_at_ns"] = 1
    atomic_json(protocol, p)
    v.update(
        task_id="exp8032-sealed-methods",
        evaluator_access_log=[dict(protocol_frozen_before_access=True, opened_at_ns=2)],
        methods_reference=e.reference(protocol),
    )
    v["role_manifests"]["public"] = {
        role: e.reference(root / "public.json") for role in ("fit", "retention")
    }
    v["role_manifests"]["evaluator"]["retention"] = e.reference(root / "target.json")
    atomic_json(path, v)
    bind(root)
    measured = producer.measure(data, root / "trajectory")
    learner_path = root / "results" / (e.producer.NAME + ".json")
    learner_terminal = root / "learner_terminal.json"
    learner_sidecar = root / "learner_sidecar.json"
    l = dict(
        experiment_id=8038,
        task_id=e.producer.TASK,
        learning_trajectory_ready_score=1,
        flagged_adversarial=False,
        terminal_validation_sidecar_path=str(learner_terminal),
        retained_labels_opened=False,
        cited_upstream_artifacts=[e.reference(protocol)],
        trajectory_directory=str(root / "trajectory"),
        trajectory_seal=e.reference(root / "trajectory/seal.json"),
        checkpoint_references=measured["checkpoint_references"],
        repository_health=[],
    )

    def seal_learner():
        atomic_json(learner_path, l)
        binding = dict(
            primary_path=str(learner_path),
            primary_sha256=e.sha256_file(learner_path),
            sidecar_path=str(learner_sidecar),
        )
        atomic_json(learner_terminal, dict(publication=binding))
        atomic_json(learner_sidecar, dict(binding, report=dict(passed=True)))

    seal_learner()
    admitted, failed = e.load_inputs(root, tmp_path / "admitted")
    assert not failed and admitted["gate_checks"]
    l.pop("learning_trajectory_ready_score")
    seal_learner()
    assert (
        e.load_inputs(root, tmp_path / "missing_ready")[1][0]["observed"]
        == "MISSING_CONTRACT_FIELD"
    )
    l["learning_trajectory_ready_score"] = 1
    l.pop("retained_labels_opened")
    seal_learner()
    assert (
        e.load_inputs(root, tmp_path / "missing_contract")[1][0]["artifact_field"]
        == "retained_labels_opened"
    )
    l["retained_labels_opened"] = False
    seal_learner()
    v["evaluator_access_log"][0]["opened_at_ns"] = 0
    atomic_json(path, v)
    bind(root)
    assert (
        e.load_inputs(root, tmp_path / "early")[1][0]["artifact_field"]
        == "protocol_before_evaluator_access"
    )


@pytest.mark.parametrize("mode", ["pass", "fail", "absent", "positive"])
def test_owned_validation_branches(tmp_path, monkeypatch, mode):
    """SCENARIO-REPORT-8039-PUBLICATION: health failures do not replace owned checks."""
    b = bundle(tmp_path)
    monkeypatch.setenv("CARNOT_EXPERIMENT_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(e, "load_inputs", lambda root, raw: (copy.deepcopy(b), []))
    monkeypatch.setattr(e, "terminal", e.replay)
    monkeypatch.setattr(e, "recover", lambda head, scratch, durable: fake_recovery(durable))
    if mode == "positive":
        original = a.compare
        monkeypatch.setattr(
            a, "compare", lambda *args: dict(original(*args), learning_benefit_score=1)
        )

    def run(root, commands, **kwargs):
        scratch = Path(kwargs["extra_env"]["CARNOT_8039_COVERAGE_CONFIG"]).parent
        if mode != "absent":
            atomic_json(
                scratch / "coverage.json",
                dict(
                    files={
                        p: dict(summary=dict(num_statements=1, missing_lines=int(mode == "fail")))
                        for p in e.OWNED
                    }
                ),
            )
        return [
            dict(scope="owned", passed=mode in ("pass", "positive")),
            dict(scope="repository_health", passed=False),
        ]

    monkeypatch.setattr(e, "run_commands", run)
    assert e.main([]) == 0
    v = json.loads((tmp_path / (e.NAME + ".json")).read_text())
    assert v["learning_audit_ready_score"] == int(mode in ("pass", "positive"))
    assert v["verdict_class"] == (
        "positive" if mode == "positive" else "null" if mode == "pass" else "disqualified"
    )
    if mode == "pass":
        v["acceptance_gate_results"]["owned_checks"] = False
        atomic_json(tmp_path / (e.NAME + ".json"), v)
        with pytest.raises(ValueError, match="unsafe_readiness"):
            e.replay(tmp_path / (e.NAME + ".json"))


def fake_recovery(durable):
    """Unit branch controls supplement the separate real CLI recovery test."""
    rows = [dict(passed=True)]
    atomic_json(durable / "recovery_rows.json", dict(rows=rows))
    return rows


def test_long_replay_and_unrecognized_event(tmp_path):
    """SCENARIO-REPORT-8039-REPLAY: long runs report progress and honor the cap."""
    data = fixture(300)
    producer.measure(data, tmp_path / "trajectory")
    r = a.replay_trajectory(tmp_path / "trajectory")
    assert len(r["independent_replay_rows"]) == 3 * 64
    db = sqlite3.connect(tmp_path / "trajectory/ledger.sqlite")
    db.execute("UPDATE events SET kind='unrecognized' WHERE kind='issue' AND seq=1")
    db.commit()
    db.close()
    atomic_json(tmp_path / "trajectory/seal.json", dict(sealed=True, references=[]))
    with pytest.raises(ValueError, match="event_kind"):
        a.replay_trajectory(tmp_path / "trajectory")


def test_store_cli_contract(tmp_path):
    """SCENARIO-REPORT-8039-RECOVERY: actual worker CLI validates its directory."""
    src = tmp_path / "worker.json"
    data = fixture()
    atomic_json(
        src,
        dict(
            head=data["head"],
            arm="recent64",
            seed=101,
            next_source=data["sources"][1],
            releases=[dict(source=data["sources"][0], family_id="private", y=1)],
        ),
    )
    result = cli(tmp_path, "--store-worker", src, "--store-dir", tmp_path / "store")
    assert result.returncode == 0, result.stdout + result.stderr
    assert cli(tmp_path, "--store-worker", src).returncode == 1


@pytest.mark.parametrize("mode", ["timeout", "missing"])
def test_recovery_child_failure(tmp_path, monkeypatch, mode):
    """SCENARIO-REPORT-8039-RECOVERY: failed owned children are closed and rejected."""
    scratch = tmp_path / "private"
    scratch.mkdir()
    original = subprocess.Popen

    def failing(argv, **kwargs):
        if "--boundary" in argv and mode == "missing":
            argv = [str(e.ROOT / ".venv/bin/python"), "-u", "-c", 'print("NO_COMMIT", flush=True)']
        return original(argv, **kwargs)

    monkeypatch.setattr(e.subprocess, "Popen", failing)
    with pytest.raises((TimeoutError, ValueError), match="commit_boundary"):
        e.recover(
            fixture()["head"],
            scratch,
            tmp_path / "durable",
            boundary_timeout_s=-1 if mode == "timeout" else 60,
        )


def test_conditional_positive_and_missing_support():
    """SCENARIO-REPORT-8039-INFERENCE: slot means and scientific gates are distinct."""
    rows, retained = [], []
    for arm in a.ARMS:
        for seed in (101, 102):
            for slot in range(256):
                y = slot % 2
                c = 0.0 if arm == "recent64" or y else 1.0
                rows.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        slot=slot,
                        source_cluster_id=str(slot),
                        y=y,
                        eligibility=True,
                        post_first_update=slot >= 36,
                        cost=c,
                        action="accept" if arm == "recent64" and not y else "reject",
                        false_accept=0,
                    )
                )
            for slot in range(64):
                retained.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        source_cluster_id=str(slot),
                        y=slot % 2,
                        eligibility=True,
                        cost_drift=0.0,
                        brier_drift=0.0,
                    )
                )
    r = a.compare(rows, retained)
    assert r["learning_benefit_score"] == 1
    assert r["primary_hypothesis_results"][0]["gain"] == 0.5
    for row in rows:
        row["eligibility"] = False
    assert a.compare(rows, [])["learning_benefit_score"] == 0
