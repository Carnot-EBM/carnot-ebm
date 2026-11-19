"""REQ-REPORT-8988-CAUSAL: private controls qualify mechanics, never future benefit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest

from carnot.memory import online_spline_8988 as a
from carnot.reporting import online_spline_methods_8988 as d
from carnot.reporting import online_spline_runner_8988 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

pytestmark = pytest.mark.memory_watchdog_skip


def test_gradient_clipping_anchor_and_masks():
    """SCENARIO-REPORT-8988-CAUSAL: analytic ensemble gradients match finite differences."""
    theta = np.array([[0.2, -0.3], [0.1, 0.4], [-0.2, 0.5]])
    initial = theta - 0.1
    phi = np.array([[1.0, 0.4], [1.0, -0.2], [1.0, 0.7]])
    candidate, acc, info = a.step(theta, np.zeros_like(theta), initial, phi, 1, False)

    def loss(t):
        p = a.probability(t, phi)
        return -np.log(p) + 0.005 * np.sum((t[:, 1:] - initial[:, 1:]) ** 2)

    gradient = np.zeros_like(theta)
    for idx in np.ndindex(theta.shape):
        delta = np.zeros_like(theta)
        delta[idx] = 1e-6
        gradient[idx] = (loss(theta + delta) - loss(theta - delta)) / 2e-6
    assert np.allclose(info["gradient"], gradient, atol=1e-9)
    assert np.allclose(acc, gradient**2)
    assert np.allclose(candidate, theta - 0.05 * gradient / np.sqrt(acc + 1e-8))
    clipped = a.step(theta, acc, initial, phi * 100, 0, False)[2]
    assert clipped["unclipped_norm"] > 1 and clipped["clipped_norm"] <= 1 + 1e-14
    bias, bias_a, _ = a.step(theta, acc * 0, initial, phi, 0, True)
    assert np.array_equal(bias[:, 1:], theta[:, 1:]) and not bias_a[:, 1:].any()
    with pytest.raises(ValueError, match="update_operands"):
        a.step(theta, acc, initial, phi, 2, False)


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-8988-CAUSAL: immutable offline fits and the original64 guard rows."""
    return runner.freeze(d.ROOT, tmp_path_factory.mktemp("8988-work"))


def test_actual_methods_controls(work):
    """SCENARIO-REPORT-8988-CAUSAL: every fixture supplies primitive, retained evidence."""
    assert not work["evidence"]["failures"] and not work["owned_failure"]
    result = work["result"]
    assert all(r["passed"] for r in result["causal_fixture_rows"] + result["crash_restart_rows"])
    assert result["crash_restart_rows"][0]["killed_returncode"] == -9
    assert result["summary_recomputation"]["passed"]
    assert len(d.protocol()["ten_bound_contract"]["bounds"]) == 10
    assert result["operation_counts"]["labels_released_per_stream"] == 28
    assert result["operation_counts"]["pending_per_stream"] == 4


def test_missing_and_typed_gate(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8988-CAUSAL: external incompleteness is blocked, never partial."""
    evidence = d.authenticate(tmp_path, tmp_path / "raw")
    assert any(r["observed"] == "missing_file" for r in evidence["failures"])
    path = tmp_path / d.INPUT
    atomic_json(path, dict(fitted_states_ready_score=True, required_checks_passed=True))
    monkeypatch.setattr(d, "INPUT_PIN", d.reference(path)["sha256"])
    evidence = d.authenticate(tmp_path, tmp_path / "typed")
    assert any(r["observed"] is True and r["expected"] == 1 for r in evidence["failures"])


def test_resume_duplicate_causal_and_reject(work, tmp_path):
    """SCENARIO-REPORT-8988-CAUSAL: rejection rolls back A but consumes labels once."""
    request = deepcopy(work["result"]["fixture_request"])
    state = a.trajectory(request, tmp_path / "first", stop=16)
    original = deepcopy(state)
    assert not a.feedback(state, request, [], []) and state == original
    resumed = a.trajectory(request, tmp_path / "first", resume=True)
    full = a.trajectory(request, tmp_path / "full")
    assert resumed == full
    changed = deepcopy(request)
    labels = json.loads(Path(request["label_path"]).read_text())["labels"]
    changed["label_path"] = str(tmp_path / "sentinel-labels.json")
    atomic_json(
        Path(changed["label_path"]), dict(labels=labels[:12] + [1 - y for y in labels[12:]])
    )
    sentinel = a.trajectory(changed, tmp_path / "sentinel", stop=16)
    assert sentinel["rows"] == original["rows"]
    bad = deepcopy(state)
    bad["prediction_hash"] = "bad"
    atomic_json(tmp_path / "first/state.json", bad)
    with pytest.raises(ValueError, match="resume_state"):
        a.trajectory(request, tmp_path / "first", resume=True)
    with pytest.raises(ValueError, match="stream_roster"):
        a.trajectory({}, tmp_path)


def test_broker_trust_boundaries(work, tmp_path):
    """SCENARIO-REPORT-8988-CAUSAL: malformed truth and retention access fail closed."""
    request = work["result"]["fixture_request"]
    state = a.trajectory(request, tmp_path / "state", stop=16)
    seal = tmp_path / "state/seals/16.json"
    assert len(a.release(16, seal, Path(request["label_path"]))) == 4
    with pytest.raises(PermissionError):
        a.release(16, seal, Path(request["label_path"]), role="retention")
    with pytest.raises(ValueError, match="guard_roster"):
        a.guard([[0]], [[0]], [], [])
    with pytest.raises(ValueError, match="feedback_budget"):
        a.feedback(state, request, [13, 14, 15, 16], [0] * 4)
    data = json.loads(seal.read_text())
    data["prediction_hash"] = "tampered"
    atomic_json(seal, data)
    with pytest.raises(ValueError, match="prediction_seal"):
        a.release(16, seal, Path(request["label_path"]))
    data["prediction_hash"] = canonical_hash(data["rows"])
    atomic_json(seal, data)
    atomic_json(tmp_path / "bad-labels.json", dict(labels=[None] * 32))
    with pytest.raises(ValueError, match="online_label_schema"):
        a.release(16, seal, tmp_path / "bad-labels.json")


def test_readiness_requires_owned_checks(work, monkeypatch):
    """REQ-REPORT-8988-CAUSAL: signal and collection gates never control methods readiness."""
    monkeypatch.setattr(d.qualification, "qualified", lambda *args: True)
    w = deepcopy(work)
    w["evidence"]["producer"]["collection_ready_score"] = 0
    w["evidence"]["producer"]["development_signal_gate"] = 0
    validation = dict(receipts=[], coverage={})
    value = d.build(w, validation, False)
    assert value["online_methods_ready_score"] == 1
    assert value["independent_count"] == value["independent_generalization_score"] == 0
    assert not value["MODEL_SPECS"] and not value["model_invoked"]
    assert d.build(w, validation, True)["online_methods_ready_score"] == 0
    w["owned_failure"] = "failed_control"
    assert d.build(w, validation, False)["verdict_class"] == "disqualified"
    monkeypatch.setattr(d.qualification, "qualified", lambda *args: False)
    assert d.build(work, validation, False)["online_methods_ready_score"] == 0
    w["evidence"]["failures"] = [dict(check="missing_file")]
    assert d.build(w, validation, False)["verdict_class"] == "blocked"


def test_private_foreign_cli_success_missing_tamper_and_replay(tmp_path):
    """SCENARIO-REPORT-8988-CAUSAL: actual CLI runs outside checkout without PYTHONPATH."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    output = tmp_path / (d.NAME + ".json")

    def child(*args):
        print("before_private_child", args, flush=True)
        process = subprocess.Popen(
            [sys.executable, "-u", str(d.ROOT / d.CLI), *args],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while True:
            try:
                stdout, _ = process.communicate(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print("private_child completed=0 pending=1", flush=True)
        print(stdout, "after_private_child", process.returncode, flush=True)
        return process.returncode

    assert child("--date", "wrong") == 2
    assert child("--date", "20261111", "--fixture-e2e", str(output)) == 0
    value = json.loads(output.read_text())
    assert value["fixture_only"] and not value["online_methods_ready_score"]
    assert child("--cold-replay", str(output)) == 0
    changed = deepcopy(value)
    changed["rows"][0]["value"] += 0.1
    changed["reproducibility_checksum"] = d.checksum(changed)
    atomic_json(output, changed)
    assert child("--cold-replay", str(output)) == 1
    atomic_json(output, value)
    ref = value["source_artifact_hashes"][0]
    saved = Path(ref["path"]).read_bytes()
    Path(ref["path"]).write_text("{}")
    assert child("--cold-replay", str(output)) == 1
    Path(ref["path"]).write_bytes(saved)
    env["CARNOT_8986_INPUT_ROOT"] = str(tmp_path / "absent")
    blocked = tmp_path / "blocked" / output.name
    assert child("--date", "20261111", "--fixture-e2e", str(blocked)) == 0
    b = json.loads(blocked.read_text())
    assert b["verdict_class"] == "blocked" and b["online_methods_ready_score"] == 0
    assert child("--cold-replay", str(blocked)) == 0


def test_cold_replay_semantic_controls(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8988-CAUSAL: coordinated aggregate rehash cannot replace pinned inputs."""
    validation = dict(receipts=[], coverage={})
    w = deepcopy(work)
    raw = Path(w["raw"])

    def artifact(w):
        atomic_json(raw / "work.json", w)
        atomic_json(raw / "validation.json", validation)
        w["raw_references"] = [d.reference(raw / "work.json"), d.reference(raw / "validation.json")]
        with runner.binding():
            value = runner.base.artifact(w, validation, True)
        atomic_json(tmp_path / "candidate.json", value)
        return value

    value = artifact(w)
    assert runner.replay(tmp_path / "candidate.json")["passed"]
    bad = deepcopy(value)
    bad["code_config_hashes"][0]["source_path"] = str(tmp_path / "absent.py")
    atomic_json(tmp_path / "candidate.json", bad)
    assert runner.replay(tmp_path / "candidate.json")["reason"] == "executed_bytes_drift"
    w["evidence"]["producer"]["fitted_states_ready_score"] = 0
    artifact(w)
    assert runner.replay(tmp_path / "candidate.json")["reason"] == "pinned_operand_drift"
    monkeypatch.setattr(d, "evaluate", lambda *args, **kwargs: {})
    w = deepcopy(work)
    artifact(w)
    assert runner.replay(tmp_path / "candidate.json")["reason"] == "reduction_drift"
    assert not runner.replay(tmp_path / "absent")["passed"]


def test_fixture_replay_rejects_operand_state_and_reduction(work, tmp_path):
    """SCENARIO-REPORT-8988-CAUSAL: replay reconstructs predictions from the fitted states."""
    from carnot.reporting.online_spline_fixtures_8988 import fixtures

    raw = tmp_path
    result = deepcopy(work["result"])
    atomic_json(raw / "fixtures.json", result)
    assert fixtures(work["evidence"], raw, True) == result
    for field, reason in [
        ("fixture_request", "fixture_operand"),
        ("crash_restart_rows", "trajectory"),
        ("summary_recomputation", "primitive_reduction"),
    ]:
        changed = deepcopy(result)
        if field == "fixture_request":
            changed[field]["temperature"] = 2
        elif field == "crash_restart_rows":
            changed[field][0]["final_state_sha256"] = "bad"
        else:
            changed[field]["prediction_sum"] += 1
        atomic_json(raw / "fixtures.json", changed)
        with pytest.raises(ValueError, match=reason):
            fixtures(work["evidence"], raw, True)
    evidence = deepcopy(work["evidence"])
    ref = evidence["producer"]["trained_state_reference"]
    saved = next(r for r in evidence["references"] if r.get("source_path") == ref["path"])
    saved["path"] = str(tmp_path / "absent")
    with pytest.raises(ValueError, match="retained_operand_drift"):
        d.load(evidence, ref)


def test_freeze_failure_and_plan(tmp_path, monkeypatch):
    """REQ-REPORT-8988-CAUSAL: owned measurement failure is disqualified and retained."""

    def refuse(*args, **kwargs):
        raise ValueError("private_refusal")

    monkeypatch.setattr(d, "evaluate", refuse)
    failed = runner.freeze(d.ROOT, tmp_path / "failed")
    assert failed["owned_failure"] == "private_refusal"
    commands = runner.validation_plan(tmp_path)
    assert set(d.OWNED) <= set(arg for c in commands for arg in c.argv)
    assert all(
        "--noconftest" in c.argv
        for c in commands
        if c.name in {"owned_unit_child", "e2e018", "e2e021"}
    )
    monkeypatch.setattr(runner.base, "main", lambda argv: 17)
    assert runner.main([]) == 17


def test_worker_normal_midpoint_exit(work, tmp_path):
    """SCENARIO-REPORT-8988-CAUSAL: a normal child also flushes coverage for its wait path."""
    atomic_json(tmp_path / "request.json", work["result"]["fixture_request"])
    print("before_normal_midpoint_child", flush=True)
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "carnot.memory.online_spline_8988",
            str(tmp_path / "request.json"),
            str(tmp_path / "state"),
            "--stop16",
        ],
        input="\n",
        text=True,
        capture_output=True,
        timeout=60,
    )
    print("after_normal_midpoint_child", completed.returncode, flush=True)
    assert completed.returncode == 0 and "READY16" in completed.stdout


def test_failed_child_wait_heartbeat_and_cleanup(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8988-CAUSAL: a stalled or dead child cannot qualify recovery."""
    import io
    from types import SimpleNamespace
    from carnot.reporting import online_spline_fixtures_8988 as f

    polls = iter([None, 1])
    times = iter([0, 0, 20, 20, 20])
    process = SimpleNamespace(
        poll=lambda: next(polls), kill=lambda: None, wait=lambda **kw: 0, stdin=io.StringIO()
    )
    monkeypatch.setattr(f.subprocess, "Popen", lambda *args, **kw: process)
    monkeypatch.setattr(
        f, "time", SimpleNamespace(monotonic=lambda: next(times), sleep=lambda seconds: None)
    )
    with pytest.raises(ValueError, match="child_midpoint_failed"):
        f.crash(work["result"]["fixture_request"], tmp_path)
    assert process.stdin.closed


def test_rehashed_no_signal_rows_still_fail_replay(work, tmp_path):
    """SCENARIO-REPORT-8988-CAUSAL: independent replay detects coordinated row and sum tamper."""
    from carnot.reporting.online_spline_fixtures_8988 import fixtures

    result = deepcopy(work["result"])
    row = next(r for r in result["rows"] if r["group_id"] == "no_signal")
    row["value"] += 0.1
    row["numerator"] = row["value"]
    result["summary_recomputation"]["prediction_sum"] = sum(r["value"] for r in result["rows"])
    atomic_json(tmp_path / "fixtures.json", result)
    with pytest.raises(ValueError, match="primitive_rows_drift"):
        fixtures(work["evidence"], tmp_path, True)
