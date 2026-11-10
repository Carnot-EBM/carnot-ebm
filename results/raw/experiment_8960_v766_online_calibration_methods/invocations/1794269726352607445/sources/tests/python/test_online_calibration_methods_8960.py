"""REQ-REPORT-8960 / SCENARIO-REPORT-8960-CONTRACT: causal calibration methods."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.memory import online_calibration_8960 as a
from carnot.reporting import online_calibration_methods_8960 as m
from carnot.reporting import online_calibration_runner_8960 as r


def fixture(tmp_path):
    """REQ-REPORT-8960: constructed probabilities carry no real error evidence."""
    path = tmp_path / "labels.json"
    m.atomic_json(path, dict(labels=[0, 1, 1, 0] * 8))
    return path, [0.2, 0.7, 0.3, 0.8] * 8, [(0.1, 0), (0.9, 1)] * 32


def test_equations_and_trust_boundaries(tmp_path):
    """REQ-REPORT-8960: independently calculate one block step and exact clipping."""
    z = [math.log(p / (1 - p)) for p in (0.2, 0.7, 0.3, 0.8)]
    y = [0, 1, 1, 0]
    residual = [p - v for p, v in zip((0.2, 0.7, 0.3, 0.8), y)]
    expected = [1 - 0.05 * sum(x * e for x, e in zip(z, residual)) / 4, -0.05 * sum(residual) / 4]
    assert a.step([1, 0], z, y, "adaptive") == pytest.approx(expected)
    residual2 = [1 / (1 + math.exp(-(2 * x + 0.5))) - v for x, v in zip(z, y)]
    expected2 = [
        2 - 0.05 * (sum(x * e for x, e in zip(z, residual2)) / 4 + 0.02),
        0.5 - 0.05 * (sum(residual2) / 4 + 0.01),
    ]
    assert a.step([2, 0.5], z, y, "adaptive") == pytest.approx(expected2)
    assert a.step([4, -2], [0.5] * 4, [1] * 4, "adaptive")[0] == 4
    assert a.step([1, 2], [0.0] * 4, [1] * 4, "adaptive")[1] == 2
    assert a.step([1, 0], z, y, "bias_only")[0] == 1
    assert a.step([1, 0], z, y, "frozen") == [1, 0]
    assert a.logit(0) == pytest.approx(math.log(1e-6 / (1 - 1e-6)))
    assert a.logit(1) == pytest.approx(math.log((1 - 1e-6) / 1e-6))
    assert a.probability([1, 0], 0) == 0.5
    assert a.step([4, 2], [1000.0] * 4, [1] * 4, "adaptive")[0] <= 4
    assert a.step([0.25, -2], [1000.0] * 4, [0] * 4, "adaptive") == [0.25, -2]
    for p in (-1, 2, float("nan"), True):
        with pytest.raises(ValueError):
            a.logit(p)
    for arm, xs, ys in (
        ("bad", z, y),
        ("adaptive", z[:1], y),
        ("adaptive", z, [True] * 4),
        ("adaptive", [float("inf")] * 4, y),
    ):
        with pytest.raises(ValueError):
            a.step([1, 0], xs, ys, arm)
    guard = [(0.5, 0)] * 64
    assert a.guard([1, -0.1], guard)["passed"]
    assert not a.guard([1, 0.1], guard)["passed"]
    checked = a.guard([1, -0.001], [(0.5, 1)] * 64)
    assert checked["brier_increase"] < 0.005 and checked["error_increase"] == 1
    assert not checked["passed"]
    with pytest.raises(ValueError):
        a.guard([1, 0], guard[:1])


def test_broker_seals_and_exact_restart(tmp_path):
    """SCENARIO-REPORT-8960-CONTRACT: future truth cannot affect a sealed prefix."""
    path, p, guard = fixture(tmp_path)
    full = a.trajectory(tmp_path / "full", p, path, guard)
    resumed = a.trajectory(tmp_path / "resumed", p, path, guard, restart=True)
    assert full == resumed
    assert len(full["rows"]) == 128 and full["pending"] == list(range(25, 33))
    assert full["numerical_training_calls"] == 21
    assert [x["query"] for x in full["feedback_rows"]] == list(range(8, 33, 4))
    assert full["feedback_rows"][0]["ordinals"] == [1, 2, 3, 4]
    assert full["feedback_rows"][0]["permuted_labels"] == [1, 0, 0, 1]
    assert full["params"]["frozen"] == [1, 0]
    assert full["params"]["adaptive"] != [1, 0]
    assert full["params"]["adaptive"] != full["params"]["permuted"]
    assert full["params"]["bias_only"][0] == 1
    m.atomic_json(path, dict(labels=[0, 1, 1, 0] * 3 + [1, 0, 0, 1] * 5))
    altered = a.trajectory(tmp_path / "altered", p, path, guard)
    assert full["rows"][:80] == altered["rows"][:80]
    broker = a.LabelBroker(path)
    with pytest.raises(PermissionError, match="retention"):
        broker.release(8, tmp_path / "full/state.json", role="retention")
    with pytest.raises(ValueError, match="seal"):
        broker.release(8, tmp_path / "missing")
    assert broker.release(7, tmp_path / "missing") == []
    seal = json.loads((tmp_path / "full/state.json").read_text())
    m.atomic_json(tmp_path / "bad-seal.json", dict(seal, query=7))
    with pytest.raises(ValueError, match="seal"):
        broker.release(8, tmp_path / "bad-seal.json")
    m.atomic_json(path, dict(labels=[True] * 32))
    with pytest.raises(ValueError, match="label"):
        broker.release(32, tmp_path / "full/state.json")
    for probs in (p[:1], [float("nan")] * 32):
        with pytest.raises(ValueError):
            a.trajectory(tmp_path / "invalid", probs, path, guard)


def test_protocol_and_private_controls(tmp_path):
    """REQ-REPORT-8960: all nine bounds are frozen before future responses exist."""
    evidence = m.authenticate(m.ROOT)
    assert not evidence["failures"]
    plan = m.protocol(evidence)
    assert len(plan["streams"]) == 24
    assert all(len(s["unit_ids"]) == 32 for s in plan["streams"])
    assert len({u for s in plan["streams"] for u in s["unit_ids"]}) == 768
    assert len(plan["retention_ids"]) == 128 and len(plan["guard_manifest"]["rows"]) == 64
    ledger = plan["multiplicity_ledger"]
    assert len(ledger["bounds"]) == 9 and ledger["draws"] == 10000
    assert ledger["alpha_per_bound"] == 0.01 / 9
    assert ledger["shared_retention_panel"] is True
    result = m.development(tmp_path)
    assert all(c["passed"] for c in result["control_rows"])
    assert result["independent_count"] == 0
    assert result["numerical_training_calls"] == 63


def test_external_diagnostics(tmp_path, monkeypatch):
    """REQ-REPORT-8960: absent evidence differs from zero, missing field and rejected bytes."""
    assert {r["check"] for r in m.authenticate(tmp_path)["failures"]} == {"missing_file"}
    name, pin = next(iter(m.PINS.items()))
    target = tmp_path / "results" / (name + ".json")
    m.atomic_json(target, {})
    assert m.authenticate(tmp_path)["failures"][0]["check"] == "rejected_producer"
    monkeypatch.setitem(m.PINS, name, m.sha256_file(target))
    assert m.authenticate(tmp_path)["failures"][0]["check"] == "missing_field"
    m.atomic_json(target, dict(experiment_id=8959, relational_methods_ready_score=0))
    monkeypatch.setitem(m.PINS, name, m.sha256_file(target))
    assert m.authenticate(tmp_path)["failures"][0]["check"] == "observed_zero"


def test_private_cli_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-8960-CONTRACT: real CLI publication, cold replay and semantic refusal."""
    output = tmp_path / "results" / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    def child(*args):
        print("[8960-test] child_before", args, flush=True)
        result = subprocess.run(
            [sys.executable, "-u", str(m.ROOT / m.CLI), *map(str, args)],
            cwd="/tmp",
            env=env,
            timeout=180,
            check=False,
        )
        print("[8960-test] child_after", result.returncode, flush=True)
        return result.returncode

    assert child("--date", "20261109", "--fixture-e2e", output) == 0
    assert child("--cold-replay", output) == 0
    value = json.loads(output.read_text())
    for key in ("rows", "online_methods_ready_score", "adapter_equations"):
        bad = deepcopy(value)
        bad[key] = [] if isinstance(bad[key], list) else 99
        bad["reproducibility_checksum"] = m.checksum(bad)
        m.atomic_json(tmp_path / "bad.json", bad)
        assert child("--cold-replay", tmp_path / "bad.json") == 1
    assert child("--date", "wrong") == 2
    assert child("--date", "20261109", "--fixture-e2e", m.ROOT / "bad.json") == 2
    assert not r.replay(tmp_path / "absent")["passed"]


def test_qualification_and_failed_owned_controls(tmp_path, monkeypatch):
    """REQ-REPORT-8960: failed owned checks or controls cannot grant readiness."""
    from carnot.reporting import v756_branch_qualification as coverage

    work = r.freeze(m.ROOT, tmp_path / "work")
    validation = dict(
        receipts=[dict(name=n, passed=True, exit_code=0, timed_out=False) for n in m.CHECKS],
        coverage={},
    )
    assert not m.qualified(validation)
    monkeypatch.setattr(coverage, "verify_measured", lambda *_: True)
    assert m.qualified(validation)
    value = m.build(work, validation, False)
    assert value["online_methods_ready_score"] == 1 and value["verdict_class"] == "null"
    work["result"]["control_rows"][0]["passed"] = False
    assert m.build(work, validation, False)["verdict_class"] == "disqualified"
    validation["receipts"][0]["passed"] = False
    assert not m.qualified(validation)
    assert m.build(work, validation, False)["online_methods_ready_score"] == 0
    assert m.build(work, validation, False)["verdict_class"] == "disqualified"
    monkeypatch.setattr(
        m,
        "authenticate",
        lambda _: dict(failures=[dict(check="missing_file")], references=[], relational={}),
    )
    blocked = r.freeze(m.ROOT, tmp_path / "blocked")
    assert blocked["result"] == {}
    assert m.build(blocked, dict(receipts=[], coverage={}), True)["verdict_class"] == "blocked"


def test_supervised_production_paths_are_private(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8960-CONTRACT: publication waits for owned checks; global debt is separate."""
    from carnot.reporting import matched_corpus_runner_8949 as shared
    from carnot.reporting import v756_branch_qualification as coverage

    work = r.freeze(m.ROOT, tmp_path / "original")
    private = tmp_path / "validation"
    private.mkdir()
    (private / ".coverage").write_bytes(b"private unit fixture")
    monkeypatch.setattr(r, "freeze", lambda *_: deepcopy(work))
    monkeypatch.setattr(r.tempfile, "mkdtemp", lambda **_: str(private))
    monkeypatch.setattr(shared, "validation_plan", lambda *_a, **_kw: [])
    monkeypatch.setattr(coverage, "measured", lambda *_: {})
    monkeypatch.setattr(r, "run_commands", lambda *_a, **_kw: [])
    monkeypatch.setattr(m, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="owned_checks_failed"):
        r.main(["--date", "20261109"])
    published = []
    monkeypatch.setattr(m, "qualified", lambda _: True)
    monkeypatch.setattr(
        r, "publish_primary", lambda output, value, validator: published.append(value) or {}
    )
    assert r.main(["--date", "20261109"]) == 0
    assert published[0]["online_methods_ready_score"] == 1
    assert published[0]["global_test_debt"]["affects_owned_readiness"] is False


def test_custody_and_rebuilt_mechanics_refuse_forgery(tmp_path, monkeypatch):
    """REQ-REPORT-8960: repairing content hashes cannot qualify invented mechanics."""
    raw = tmp_path / "raw"
    work = r.freeze(m.ROOT, raw)
    validation = dict(receipts=[], coverage={})
    m.atomic_json(raw / "work.json", work)
    m.atomic_json(raw / "validation.json", validation)
    work["raw_references"] += [m.reference(raw / "work.json"), m.reference(raw / "validation.json")]
    value = m.build(work, validation, True)
    path = tmp_path / "artifact.json"
    m.atomic_json(path, value)
    assert r.replay(path)["passed"]
    bad = deepcopy(value)
    bad["reproducibility_checksum"] = "sha256:bad"
    m.atomic_json(path, bad)
    assert r.replay(path)["reason"] == "checksum_drift"
    bad = deepcopy(value)
    bad["source_artifact_hashes"][0]["sha256"] = "bad"
    bad["reproducibility_checksum"] = m.checksum(bad)
    m.atomic_json(path, bad)
    assert r.replay(path)["reason"] == "retained_bytes_drift"
    bad = deepcopy(value)
    bad["code_config_hashes"][0]["source_path"] = str(tmp_path / "missing")
    bad["reproducibility_checksum"] = m.checksum(bad)
    m.atomic_json(path, bad)
    assert r.replay(path)["reason"] == "executed_bytes_drift"
    stored = json.loads((raw / "work.json").read_text())
    stored["result"]["numerical_training_calls"] += 1
    m.atomic_json(raw / "work.json", stored)
    stored["raw_references"] += [
        m.reference(raw / "work.json"),
        m.reference(raw / "validation.json"),
    ]
    m.atomic_json(path, m.build(stored, validation, True))
    assert r.replay(path)["reason"] == "mechanics_or_protocol_drift"
    monkeypatch.setattr(r, "validate_current_work_receipt", lambda _: ["invalid_current_time"])
    assert r.replay(path)["reason"] == "compute_scope"


def test_numeric_seal_and_guard_failures(tmp_path):
    """REQ-REPORT-8960: numeric adapters have no file operands and cannot bypass guards."""
    import inspect

    assert set(inspect.signature(a.step).parameters) == {"params", "zs", "ys", "arm"}
    path, p, guard = fixture(tmp_path)
    state = a.trajectory(tmp_path / "run", p, path, guard)
    state["prediction_hash"] = "wrong"
    m.atomic_json(tmp_path / "seal.json", state)
    with pytest.raises(ValueError, match="prediction_seal"):
        a.LabelBroker(path).release(32, tmp_path / "seal.json")
    run = a.trajectory(tmp_path / "rejected", p, path, [(0.75, 1)] * 64)
    assert any(not row["passed"] for row in run["update_rows"])
    assert all(row["before"] == row["after"] for row in run["update_rows"] if not row["passed"])


def test_original_operand_drift_and_opaque_history(tmp_path, monkeypatch):
    """REQ-REPORT-8960: current operands authenticate; retired audit bytes remain opaque."""
    name, _ = next(iter(m.PINS.items()))
    path = tmp_path / "results" / (name + ".json")
    value = dict(
        relational_methods_ready_score=1,
        required_checks_passed=True,
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
    )
    operand = tmp_path / "operand"
    value["raw_shard_hashes"] = [dict(path=str(operand), sha256="sha256:expected")]
    m.atomic_json(path, value)
    monkeypatch.setitem(m.PINS, name, m.sha256_file(path))
    assert m.authenticate(tmp_path)["failures"][0]["check"] == "missing_file"
    operand.write_text("changed original bytes")
    assert m.authenticate(tmp_path)["failures"][0]["check"] == "rejected_producer"
    value["raw_shard_hashes"] = []
    m.atomic_json(path, value)
    monkeypatch.setitem(m.PINS, name, m.sha256_file(path))
    old = list(m.PINS)[1]
    history = tmp_path / "results" / (old + ".json")
    m.atomic_json(history, dict(required_checks_passed=False, retired_gate=0))
    monkeypatch.setitem(m.PINS, old, m.sha256_file(history))
    assert not m.authenticate(tmp_path)["failures"]
    value.pop("raw_shard_hashes")
    m.atomic_json(path, value)
    monkeypatch.setitem(m.PINS, name, m.sha256_file(path))
    failure = m.authenticate(tmp_path)["failures"][0]
    assert failure["check"] == "missing_field"
    assert failure["artifact_field"] == "raw_shard_hashes" and failure["observed"] is None
