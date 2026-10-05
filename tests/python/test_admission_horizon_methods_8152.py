"""REQ-VERIFY-8152 / REQ-REPORT-8152: causal readiness has no natural credit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import admission_horizon_methods_8152 as e


@pytest.mark.parametrize("case", ["positive", "rejected", "late_label", "overflow"])
def test_scalar_events(case):
    """SCENARIO-VERIFY-8152-REFERENCE: every boundary follows a second clock."""
    rows, labels = e.fixture(case)
    state = e.run(rows, labels, 101, capacity=8 if case == "overflow" else 32)
    assert e.reference(state, rows, labels)["passed"]
    if case == "positive":
        exposure = e.exposure(state, rows)
        assert exposure["installation_slot"] <= 208
        assert exposure["usable_changed_later_count"] >= 32
        assert [len(state["arms"][a]["centers"]) for a in e.engine.ARMS] == [16, 24, 24, 24]
        commits = [r for r in state["events"] if r["kind"] == "commit_candidate"]
        assert [r["slot"] for r in commits] == [64, 144]
        assert (
            commits[1]["candidates"]["fixed_public_center"]["base"]["centers"][-4:]
            == state["reserved"][4:8]
        )
    elif case == "rejected":
        assert all(
            not any(r["steps"].values()) for r in state["events"] if r["kind"] == "admit_once"
        )
    elif case == "late_label":
        assert [r["slot"] for r in state["events"] if r["kind"] == "defer_candidate"] == [144, 224]
    else:
        assert state["lost"] and not state["released"]


def test_old_expiry_and_crash():
    """SCENARIO-VERIFY-8152-REFERENCE: old expiry fails; durable resume is exact."""
    rows, labels = e.fixture("positive")
    old = e.engine.run(rows, labels, 101)
    assert not any(any(r["steps"].values()) for r in old["events"] if r["kind"] == "admit_once")
    baseline = e.run(rows, labels, 101)
    saved = []

    def seal(kind, state):
        if kind == "durable_commit" and state["cursor"] == 144:
            saved.append(deepcopy(state))
            raise RuntimeError("fixture_crash")

    with pytest.raises(RuntimeError, match="fixture_crash"):
        e.run(rows, labels, 101, seal=seal)
    assert e.run(rows, labels, 101, state=saved[0]) == baseline
    with pytest.raises(ValueError, match="baseline_hash"):
        e.run(rows, labels, 101, state=dict(saved[0], baseline_hash="tampered"))
    with pytest.raises(ValueError, match="evaluator_slots"):
        e.run(rows, labels[:-1], 101)


def test_tamper_events_and_missing():
    """SCENARIO-VERIFY-8152-CUSTODY: masks remain; forged arithmetic is rejected."""
    rows, labels = e.fixture("positive")
    rows[230]["values"] = None
    labels[230] = None
    state = e.run(rows, labels, 101)
    assert e.reference(state, rows, labels)["passed"]
    changed = deepcopy(state)
    changed["issued"][200]["predictions"]["error_center"] += 0.1
    with pytest.raises(ValueError):
        e.reference(changed, rows, labels)
    changed = deepcopy(state)
    changed["events"][-1]["pending"] = []
    with pytest.raises(ValueError):
        e.reference(changed, rows, labels)


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """One private full runner receipt avoids counting test repeats as sources."""
    raw = tmp_path_factory.mktemp("horizon8152")
    return e.measure(e.ROOT, raw, fixture_mode=True), raw


def test_artifact_and_disqualification(measured):
    """REQ-REPORT-8152: readiness requires passed owned checks, with zero benefit."""
    work, raw = measured
    receipts = [dict(name="private", passed=True, normal_exit=True)]
    value = e.build(work, raw, receipts)
    assert value["learning_protocol_ready_score"] == value["future_exposure_fixture_score"] == 1
    assert value["verdict_class"] == "circular_positive" and value["verifier_is_oracle"]
    assert value["MODEL_SPECS"] == value["call_ledger"] == []
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert not any(value["model_invocation_counts"].values())
    assert e.build(work, raw, [dict(passed=False)])["verdict_class"] == "disqualified"
    output = raw / "candidate.json"
    atomic_json(output, value)
    assert e.replay(output)
    value = deepcopy(value)
    value["rows"][0]["numerator"] += 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert not e.replay(output)


def test_script_path_private_routes(tmp_path):
    """SCENARIO-REPORT-8152-CLI: success/block/tamper/replay outside checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONUNBUFFERED"] = "1"
    if os.environ.get("COVERAGE_RCFILE"):
        env["COVERAGE_PROCESS_START"] = os.environ["COVERAGE_RCFILE"]
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]

    def call(*args, expected=0):
        e.progress("before_private_subprocess")
        log = tmp_path / (str(time.time_ns()) + ".log")
        started = time.monotonic()
        with log.open("w") as stream:
            child = subprocess.Popen(
                [*argv, *args], cwd=tmp_path, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            while child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    e.progress("private_child_wait", 0, 1)
                    if time.monotonic() - started > 240:
                        child.kill()
                        child.wait()
                        pytest.fail(log.read_text())
        e.progress("after_private_subprocess", int(child.returncode == expected))
        assert child.returncode == expected, log.read_text()

    output = tmp_path / (e.NAME + ".json")
    call("--fixture-output", str(output))
    call("--cold-replay", str(output))
    blocked = tmp_path / "blocked" / output.name
    call("--fixture-output", str(blocked), "--root", str(tmp_path / "missing"))
    value = json.loads(blocked.read_text())
    assert value["verdict_class"] == "blocked" and value["learning_protocol_ready_score"] == 0
    assert next(r for r in value["gate_check_summary"] if not r["passed"])["observed"] is False
    call("--cold-replay", str(blocked))
    output.write_text("{}")
    call("--cold-replay", str(output), expected=1)
    call("--fixture-output", str(e.ROOT / "results" / output.name), expected=2)


def test_real_custody_and_historical_events(tmp_path):
    """SCENARIO-VERIFY-8152-CUSTODY: direct qualified captures retain historical clocks."""
    work = e.measure(e.ROOT, tmp_path)
    assert work["input_ready"] == 1, [r for r in work["gate_check_summary"] if not r["passed"]]
    assert len(work["historical_installation_rows"]) == 60
    assert all(
        r["installation_slot"] == 248 and r["usable_resolved_later_slots"] == []
        for r in work["historical_installation_rows"]
        if r["commitment_slot"] == 192
    )
    assert (
        work["original_slot_mask"]["stream"]
        == json.loads(
            (e.ROOT / "results/experiment_8102_v701_learning_stream_capture.json").read_text()
        )["original_slot_mask"]["stream"]
    )


def test_supervisor_and_failure_routes(tmp_path, monkeypatch, measured):
    """REQ-REPORT-8152: receipt control is isolated; real CLI tests own execution."""
    from carnot.reporting import admission_horizon_execution_8152 as runner

    original = e.measure
    work = deepcopy(measured[0])
    monkeypatch.setattr(e, "measure", lambda *a, **kw: deepcopy(work))

    def check(spec, private, raw):
        if spec["name"] == "measurement":
            target = Path(spec["argv"][spec["argv"].index("--worker-output") + 1])
            atomic_json(target, work)
        return dict(
            name=spec["name"],
            passed=spec["name"] != "adversarial_verify",
            actual_exit=1 if spec["name"] == "adversarial_verify" else 0,
            normal_exit=True,
        )

    monkeypatch.setattr(runner, "check", check)
    monkeypatch.setattr(
        runner,
        "publish_primary",
        lambda output, value, validate: dict(
            output=str(output), verdict=value["verdict_class"], validation=validate(output)
        ),
    )
    monkeypatch.setattr(e, "replay", lambda p: True)
    assert runner.main(["--output", str(tmp_path / (e.NAME + ".json"))]) == 0
    assert runner.main(["--worker-output", str(tmp_path / "worker.json")]) == 0
    monkeypatch.setattr(e, "measure", original)
    monkeypatch.setattr(
        e, "authenticate", lambda *a: (_ for _ in ()).throw(ValueError("bad_primitive"))
    )
    failed = original(e.ROOT, tmp_path / "bad")
    assert (
        failed["input_ready"] == 0
        and failed["gate_check_summary"][-1]["observed"] == "bad_primitive"
    )


def test_check_receipt_and_protocol_hash(tmp_path, monkeypatch):
    """REQ-REPORT-8152: abnormal exits and mutable protocol bytes cannot qualify."""
    from carnot.reporting import admission_horizon_execution_8152 as runner

    monkeypatch.setattr(
        runner, "run_check", lambda *a, **kw: dict(actual_exit=-9, timed_out=True, passed=False)
    )
    assert runner.check(dict(name="killed"), tmp_path, tmp_path)["normal_exit"] is False
    monkeypatch.setattr(e, "PROTOCOL_HASH", "sha256:wrong")
    with pytest.raises(ValueError, match="protocol_hash"):
        e.protocol()


def test_unchanged_label_guards():
    """REQ-VERIFY-8152: scheduling changes retain original label and one-use guards."""
    rows, labels = e.fixture("positive")
    labels[0] = 2
    with pytest.raises(ValueError, match="evaluator_label"):
        e.run(rows, labels, 101)
    rows, labels = e.fixture("positive")
    checkpoint = e.engine.genesis(rows, 101)
    checkpoint.update(cursor=21, phase="release", pending=[1], consumed=[1])
    with pytest.raises(ValueError, match="reused_label"):
        e.run(rows, labels, 101, state=checkpoint)


def test_rehashed_metadata_and_restart(measured, tmp_path):
    """SCENARIO-VERIFY-8152-CUSTODY: a new checksum cannot authorize forged evidence."""
    work, raw = measured
    base = e.build(work, raw, [dict(passed=True)])
    for field, changed in [
        ("protocol_sha256", "sha256:wrong"),
        ("restart_fixture", dict(work["restart_fixture"], passed=False)),
        ("old_expiry_fixture", dict(work["old_expiry_fixture"], passed=False)),
    ]:
        value = deepcopy(base)
        value[field] = changed
        value.pop("reproducibility_checksum")
        value["reproducibility_checksum"] = canonical_hash(value)
        path = tmp_path / (field + ".json")
        atomic_json(path, value)
        assert not e.replay(path)


def test_capture_floor_operand(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8152-CUSTODY: source-floor failure names actual224/223 counts."""
    methods = e.engine.methods
    value = json.loads((e.ROOT / methods.STREAM).read_text())
    direct = {k: value[k] for k in ["stream_feature_manifest", "retention_feature_manifest"]}
    monkeypatch.setattr(methods, "authenticate_stream", lambda *a: direct)
    original = methods.read_ref

    def read(binder, ref):
        result = original(binder, ref)
        if ref == value["stream_feature_manifest"]:
            result = deepcopy(result)
            keep = {r["slot"] for r in result["rows"] if r["values"] is not None}
            keep = set(sorted(keep)[:223])
            for row in result["rows"]:
                if row["slot"] not in keep:
                    row["values"] = None
        return result

    monkeypatch.setattr(methods, "read_ref", read)
    binder = methods.Custody(tmp_path)
    with pytest.raises(ValueError, match="stream_usable_sources_floor"):
        e.authenticate(e.ROOT, tmp_path, binder)
    check = binder.failures[-1]
    assert check["op"] == ">=" and check["expected"] == 224 and check["observed"] < 224
    assert check["observed"] == 223


@pytest.mark.parametrize("kind", ["code", "raw", "headline", "old", "restart"])
def test_bound_tamper(kind, measured, tmp_path):
    """SCENARIO-VERIFY-8152-CUSTODY: independent equations reject rehashed private copies."""
    work, raw = measured
    value = deepcopy(e.build(work, raw, [dict(passed=True)]))
    if kind == "code":
        value["code_config_hashes"][e.MODULE] = "sha256:wrong"
    elif kind == "raw":
        value["raw_shard_hashes"][0]["sha256"] = "sha256:wrong"
    elif kind == "headline":
        value["rows"][0]["numerator"] += 0.1
        value["reductions"] = e.engine.historical.reductions(value["rows"])
    else:
        fixture = value["old_expiry_fixture" if kind == "old" else "restart_fixture"]
        field = "transcript" if kind == "old" else "checkpoint"
        state = json.loads(Path(fixture[field]["path"]).read_text())
        if kind == "old":
            state["cursor"] = 0
        else:
            state["arms"]["error_center"]["intercept"] += 1
        path = tmp_path / "changed_state.json"
        atomic_json(path, state)
        fixture[field] = dict(path=str(path), sha256=sha256_file(path))
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    output = tmp_path / "candidate.json"
    atomic_json(output, value)
    assert not e.replay(output)
