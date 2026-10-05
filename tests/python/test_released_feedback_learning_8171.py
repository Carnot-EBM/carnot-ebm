"""REQ-VERIFY-8171 / REQ-REPORT-8171: preserve causal clocks and publication."""

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import released_feedback_learning_8171 as e
from test_delayed_energy_memory_8143 import child


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    raw = tmp_path_factory.mktemp("released8171")
    return e.measure(e.ROOT, raw, fixture=True), raw


def test_trajectory_and_rehashed_tamper(measured, tmp_path):
    """SCENARIO-VERIFY-8171-CAUSAL: independent replay rejects fabricated changes."""
    work, raw = measured
    value = e.build(work, raw, [dict(passed=True, normal_exit=True)])
    assert value["learning_trajectory_ready_score"] == 1
    assert len(value["trajectory_manifest"]) == 20
    assert value["MODEL_SPECS"] == value["call_ledger"] == []
    assert value["independent_generalization_score"] == 0
    assert value["generalized_learning_benefit_score"] == 0
    assert value["trained_head_specs"][0]["maximum_centers"] == 24
    assert value["hardware_receipt"]["actual_loaded"]
    assert value["hardware_receipt"]["update_operations"] > 0
    assert {r["slot"] for r in value["event_rows"] if r["kind"] == "commit_candidate"} == {64, 144}
    assert all(r["passed"] for r in value["reference_rows"])
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    for field, replacement in [
        ("future_prediction_rows", []),
        ("rows", []),
        ("experiment_id", 8143),
    ]:
        bad = deepcopy(value)
        bad[field] = replacement
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        assert not e.replay(path), field
    path.write_text("{}")
    assert not e.replay(path)
    bad = deepcopy(value)
    bad["protocol_sha256"] = "sha256:wrong"
    atomic_json(path, bad)
    assert not e.replay(path)
    bad = deepcopy(value)
    bad["reproducibility_checksum"] = "bad"
    atomic_json(path, bad)
    assert not e.replay(path)
    failed = e.build(work, raw, [dict(passed=True, normal_exit=False)])
    assert failed["verdict_class"] == "disqualified"
    assert failed["learning_trajectory_ready_score"] == 0


def test_gate_and_external_block(tmp_path):
    """REQ-VERIFY-8171: the sole qualification gate cannot borrow H1 readiness."""
    raw = tmp_path / "blocked"
    work = e.measure(tmp_path, raw)
    value = e.build(work, raw, [dict(passed=True, normal_exit=True)])
    assert value["verdict_class"] == "blocked"
    assert value["learning_trajectory_ready_score"] == 0
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert any(not r["passed"] for r in value["gate_check_summary"])
    root = tmp_path / "gate"
    atomic_json(root / e.UPSTREAM, dict(learning_protocol_ready_score=0))
    work = e.measure(root, root / "raw")
    failed = next(r for r in work["gate_check_summary"] if not r["passed"])
    assert failed["artifact_field"] == "learning_protocol_ready_score"
    assert failed["observed"] == 0 and failed["expected"] == 1


def test_real_causal_negative_paths(tmp_path):
    """SCENARIO-VERIFY-8171-CAUSAL: future labels, duplicate release and overflow."""
    rows, targets = e.schedule.fixture("positive")
    labels = tmp_path / "labels.json"
    atomic_json(labels, dict(rows=[dict(r, y=y) for r, y in zip(rows, targets, strict=True)]))
    state = e.run_seed(rows, labels, 101, tmp_path / "full")
    assert max(state["released"]) == 236 and len(state["pending"]) == 20
    assert set(state["training"]).isdisjoint(state["used_admission"])
    assert e.schedule.reference(state, rows, targets)["passed"]
    assert e.schedule.exposure(state, rows)["usable_changed_later_count"] >= 32
    changed = list(targets)
    changed[200:] = [1 - y for y in changed[200:]]
    with e.arithmetic({}):
        other = e.schedule.run(rows, changed, 101)
    assert other["issued"][:220] == state["issued"][:220]
    overflow = e.schedule.run(rows, targets, 101, capacity=8)
    assert overflow["lost"] and not overflow["released"]
    assert e.schedule.reference(overflow, rows, targets)["overflow_count"] > 0
    saved = {}

    def halt(kind, current):
        if current["cursor"] == 64:
            saved.update(deepcopy(current))
            raise RuntimeError("crash_before_release")

    with pytest.raises(RuntimeError, match="crash_before_release"):
        e.schedule.run(rows, targets, 101, seal=halt)
    resumed = e.schedule.run(rows, targets, 101, state=deepcopy(saved))
    assert resumed == e.schedule.run(rows, targets, 101)
    saved["consumed"].append(44)
    with pytest.raises(ValueError, match="reused_label"):
        e.schedule.run(rows, targets, 101, state=saved)


def test_direct_private_cli(tmp_path):
    """SCENARIO-REPORT-8171-CLI: actual script success/block/tamper/cold replay."""
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]
    output = tmp_path / (e.NAME + ".json")
    child([*argv, "--fixture-output", str(output)], tmp_path)
    assert "replay_passed" in child([*argv, "--cold-replay", str(output)], tmp_path)
    blocked = tmp_path / "blocked" / output.name
    child([*argv, "--fixture-output", str(blocked), "--root", str(tmp_path / "absent")], tmp_path)
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    child([*argv, "--cold-replay", str(blocked)], tmp_path)
    output.write_text("{}")
    child([*argv, "--cold-replay", str(output)], tmp_path, expected=1)
    child([*argv, "--fixture-output", str(e.ROOT / "results" / output.name)], tmp_path, expected=2)


def test_manifest_and_supervisor(tmp_path, monkeypatch, measured):
    """REQ-REPORT-8171: frozen validation, truthful normal exits and failure verdict."""
    from carnot.reporting import released_feedback_execution_8171 as runner

    specs = runner.manifest(tmp_path, tmp_path / "candidate.json")
    assert (
        "--fail-under=100"
        in next(s for s in specs["commands"] if s["name"] == "coverage_report")["argv"]
    )
    for spec in specs["commands"]:
        if spec["name"] in ["strict_mypy", "ruff_check", "ruff_format", "spec_coverage"]:
            assert not any("::" in a for a in spec["argv"])
    monkeypatch.setattr(runner, "run_check", lambda *a, **k: dict(actual_exit=-9, passed=False))
    assert not runner.check(dict(name="killed"), tmp_path, tmp_path)["normal_exit"]
    monkeypatch.setattr(e, "measure", lambda *a, **k: deepcopy(measured[0]))
    assert runner.main(["--worker-output", str(tmp_path / "worker.json")]) == 0
    assert os.environ["PYTHONUNBUFFERED"] == "1"

    def check(spec, private, raw):
        if spec["name"] == "measurement":
            target = Path(spec["argv"][spec["argv"].index("--worker-output") + 1])
            atomic_json(target, measured[0])
        return dict(
            name=spec["name"], passed=spec["name"] != "adversarial_verify", normal_exit=True
        )

    monkeypatch.setattr(runner, "check", check)
    captured = []

    def publish(output, value, validate):
        captured.append(value)
        return dict(output=str(output), validation=validate(output))

    monkeypatch.setattr(runner, "publish_primary", publish)
    monkeypatch.setattr(e, "replay", lambda p: True)
    assert runner.main(["--output", str(tmp_path / (e.NAME + ".json"))]) == 0
    assert captured[-1]["verdict_class"] == "disqualified"
    assert captured[-1]["learning_trajectory_ready_score"] == 0


def test_natural_custody_and_binding_failure(tmp_path, measured, monkeypatch):
    """REQ-VERIFY-8171: actual source masks and loaded-binding parity are required."""
    binder = e.engine.methods.Custody(tmp_path / "inputs")
    upstream = e.authenticate(e.ROOT, tmp_path, binder)
    assert len(upstream["original_slot_mask"]["stream"]) == 256
    assert len(upstream["original_slot_mask"]["retention"]) == 64
    assert all(r["passed"] for r in binder.checks)
    base = deepcopy(measured[0])
    base["fixture_mode"] = False
    monkeypatch.setattr(e.previous, "measure", lambda *a, **k: deepcopy(base))
    assert e.measure(e.ROOT, tmp_path / "natural")["historical_model_provenance"]
    monkeypatch.setattr(e, "ORIGINAL_DESIGN", lambda *a: 0)
    rows, labels = e.schedule.fixture("positive")
    with pytest.raises(ValueError, match="loaded_binding_parity"), e.arithmetic({}):
        e.schedule.run(rows, labels, 101)


def test_crash_seal_and_seed_dispatch(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8171-CAUSAL: cover abrupt exit and saved-state dispatch."""
    from carnot.reporting import released_feedback_execution_8171 as runner

    rows, targets = e.schedule.fixture("positive")
    labels = tmp_path / "labels.json"
    atomic_json(labels, dict(rows=[dict(r, y=y) for r, y in zip(rows, targets, strict=True)]))

    def abrupt(code):
        raise SystemExit(code)

    monkeypatch.setattr(e.os, "_exit", abrupt)
    with pytest.raises(SystemExit) as stopped:
        e.run_seed(rows, labels, 101, tmp_path / "crash", crash_slot=64)
    assert stopped.value.code == 73
    saved = tmp_path / "crash/crash_checkpoint.json"
    checkpoint = json.loads(saved.read_text())
    assert checkpoint["phase"] == "release" and checkpoint["cursor"] == 64
    assert 44 not in checkpoint["consumed"]
    inputs = tmp_path / "input.json"
    atomic_json(inputs, dict(rows=rows, label_path=str(labels), seed=101))
    calls = []
    monkeypatch.setattr(e, "run_seed", lambda *a, **k: calls.append(k))
    assert (
        runner.main(
            [
                "--seed-input",
                str(inputs),
                "--seed-output",
                str(tmp_path / "resume"),
                "--resume-state",
                str(saved),
            ]
        )
        == 0
    )
    assert calls[0]["state"] == checkpoint


def test_exact_numerical_metadata(tmp_path):
    """REQ-VERIFY-8171: reported numerical authority names the actual V705 clock."""
    work = e.measure(tmp_path, tmp_path / "raw")
    assert work["numerical_protocol"] == e.schedule.protocol()
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, e.build(work, tmp_path / "raw", [dict(passed=True, normal_exit=True)]))
    assert e.replay(path)
