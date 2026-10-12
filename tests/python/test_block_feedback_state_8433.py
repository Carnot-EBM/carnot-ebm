"""REQ-VERIFY-8433, REQ-REPORT-8433: independent bounded recovery evidence."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest
from scipy.interpolate import BSpline
from scipy.special import expit

from carnot.verify import block_feedback_state_8433 as e


@pytest.mark.parametrize("capacity", [8, 16])
@pytest.mark.parametrize("policy", ["fifo", "uniform"])
def test_blocks_capacity_causality_and_duplicate_delivery(tmp_path, capacity, policy):
    """SCENARIO-VERIFY-8433-RECOVERY: all siblings precede any released label."""
    store = e.Store(tmp_path, e.config(capacity, policy))
    store.initialize()
    with pytest.raises(ValueError, match="unsealed"):
        store.release(0, {})
    for block in range(5):
        before = store.read()
        sealed = store.seal(block, e.questions(block))
        state = store.read()
        assert len(state["issued"]) == 60 * (block + 1)
        assert len(state["pending"]) == 3 * capacity
        assert all("y" not in r for r in state["pending"])
        assert state["head"] == before["head"]
        assert store.seal(block, e.questions(block)) == sealed
        with pytest.raises(ValueError, match="release_roster"):
            store.release(block, {})
        labels = e.feedback(state)
        ack = store.release(block, labels)
        final = store.read()
        assert store.release(block, labels) == ack
        assert store.seal(block, e.questions(block)) == sealed
        assert store.read() == final
        assert final["cursor"] == 20 * (block + 1)
        assert final["version"] == 2 * (block + 1)
        assert not final["pending"]
    assert e.replay_state(store.read(), store.configuration)
    assert store.read()["head"] != e.head()


def test_independent_basis_nonzero_and_frozen(tmp_path):
    """SCENARIO-VERIFY-8433-RECOVERY: independent arithmetic catches no-op learning."""
    h = e.head()
    x = np.asarray(e.questions(0)[0]["answers"][0]["x"])
    knots = [0.0] * 4 + [i / 9 for i in range(1, 9)] + [1.0] * 4
    basis = np.asarray([BSpline(knots, np.eye(12)[k], 3)(v) for v in x for k in range(12)] + [1.0])
    theta = np.asarray(h["parameters"]) * h["decay_scale"]
    z = np.log(x[0] / (1 - x[0])) + basis @ theta
    expected = theta - 0.01 * (
        basis * (expit(z / h["temperature"]) - 1) / h["temperature"] + 0.002 * theta
    )
    changed, _ = e.optimizer.update(h, x, 1, 0.01)
    np.testing.assert_allclose(e.optimizer.parameters(changed), expected, rtol=0, atol=1e-12)
    assert np.max(np.abs(expected - theta)) > 1e-6
    result = e.drive(tmp_path, e.config(8, "uniform", frozen=True, shuffled=True))
    assert result["head"] == h
    assert result["parameter_updates"] == 0
    assert result["pending_high_water"] == 24


@pytest.mark.parametrize("field", ["coefficient", "cursor", "release", "rng"])
def test_self_consistently_rehashed_corruption(tmp_path, field):
    """SCENARIO-VERIFY-8433-RECOVERY: an outer checksum is not a semantic proof."""
    cfg = e.config(8, "uniform")
    state = e.drive(tmp_path, cfg)
    changed = deepcopy(state)
    if field == "coefficient":
        changed["head"]["parameters"][0] += 0.2
    elif field == "cursor":
        changed["cursor"] -= 20
    elif field == "release":
        changed["applied_feedback_ids"][0] = "forged"
    else:
        changed["admission_rng"][1][0] ^= 1
    assert not e.replay_state(changed, cfg)
    e.atomic_json(tmp_path / "tamper.json", dict(state=changed, sha256=e.canonical_hash(changed)))
    assert json.loads((tmp_path / "tamper.json").read_text())["sha256"] == e.canonical_hash(changed)


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    """REQ-REPORT-8433: reuse actual child evidence within this test invocation."""
    from carnot.reporting import block_checkpoint_qualification_8433 as q

    parent = tmp_path_factory.mktemp("8433-evidence")
    return q.measure(parent / "raw", parent / "scratch"), parent / "raw"


def test_real_crashes_and_future_decisions(evidence):
    """SCENARIO-VERIFY-8433-VALIDATION: exits are real children, including lost delivery."""
    work, _ = evidence
    assert len(work["rows"]) == 40
    assert all(r["passed"] for r in work["rows"])
    assert len(work["crash_rows"]) == 36
    assert work["nonzero_update_controls"]["passed"]
    assert all(r["passed"] for r in work["rng_resume_checks"])
    assert work["rng_resume_checks"][-1]["feedback_rng_changed"]
    assert all(r["future_label_reads"] == 0 for r in work["label_access_rows"])


def test_cli_modes_and_private_publication(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8433-CLI: dated CLI and cold rejection run independently."""
    from carnot.reporting import block_checkpoint_runner_8433 as runner
    from carnot.reporting import block_checkpoint_qualification_8433 as q

    assert runner.main(["--deliberate-error"]) == 7
    assert runner.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    config_path = tmp_path / "config.json"
    e.atomic_json(config_path, e.config())
    worker = tmp_path / "worker"
    assert runner.main(["--worker", str(worker), "--configuration", str(config_path)]) == 0
    final = worker / "final.json"
    assert runner.main(["--state-replay", str(final), "--configuration", str(config_path)]) == 0
    envelope = tmp_path / "envelope.json"
    value = json.loads(final.read_bytes())
    e.atomic_json(envelope, dict(state=value, sha256=e.canonical_hash(value)))
    assert runner.main(["--state-replay", str(envelope), "--configuration", str(config_path)]) == 0
    e.atomic_json(envelope, dict(state=value, sha256="forged"))
    assert runner.main(["--state-replay", str(envelope), "--configuration", str(config_path)]) == 1
    assert (
        runner.main(
            ["--state-replay", str(tmp_path / "missing"), "--configuration", str(config_path)]
        )
        == 1
    )
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261011"])
    manifest = q.manifest(tmp_path / "plan")
    assert any(r["name"] == "private_E2E020" for r in manifest)
    assert "_exit" in (tmp_path / "plan/coverage.ini").read_text()


def test_store_rejects_unsafe_inputs_and_torn_snapshots(tmp_path):
    """SCENARIO-VERIFY-8433-RECOVERY: custody, capacity and atomic reader failures."""
    with pytest.raises(ValueError, match="configuration"):
        e.config(20, "fifo")
    cfg = e.config()
    invalid = deepcopy(cfg)
    invalid["head"]["parameters"] = []
    with pytest.raises(ValueError, match="head_schema"):
        e.initial(invalid)
    store = e.Store(tmp_path, cfg)
    store.initialize()
    store.initialize()
    start = store.read()
    assert not e.replay_state(dict(start, version=-1), cfg)
    assert not e.replay_state(dict(start, version="1"), cfg)
    with pytest.raises(ValueError, match="unsealed_evaluator"):
        e.feedback(start)
    with pytest.raises(ValueError, match="seal_order"):
        store.seal(1, e.questions(1))
    with pytest.raises(ValueError, match="block_roster"):
        store.seal(0, [])
    bad = e.questions(0)
    bad[0]["answers"][0]["y"] = 1
    with pytest.raises(ValueError, match="answer_roster"):
        store.seal(0, bad)
    seen = []

    def inspect(phase):
        snapshot = store.read()
        seen.append((phase, snapshot["version"]))
        assert snapshot == start or snapshot["phase"] == "sealed"

    store.seal(0, e.questions(0), inspect)
    assert seen == [
        ("temporary_write", 0),
        ("file_fsync", 0),
        ("atomic_replace", 1),
        ("acknowledgment", 1),
    ]
    sealed = store.read()
    with pytest.raises(ValueError, match="seal_order"):
        e.seal(sealed, cfg, 0, e.questions(0))
    with pytest.raises(ValueError, match="prediction_identity"):
        e.seal(dict(start, issued=sealed["issued"]), cfg, 0, e.questions(0))
    with pytest.raises(ValueError, match="duplicate_conflict"):
        store.seal(0, e.questions(1))
    labels = e.feedback(sealed)
    with pytest.raises(ValueError, match="release_roster"):
        store.release(0, dict(labels, future_answer=1))
    poisoned = deepcopy(sealed)
    poisoned["applied_feedback_ids"] = ["feedback:0:" + sealed["pending"][0]["id"]]
    with pytest.raises(ValueError, match="duplicate_feedback"):
        e.release(poisoned, cfg, 0, labels)
    store.release(0, labels)
    with pytest.raises(ValueError, match="duplicate_conflict"):
        store.release(0, dict.fromkeys(labels, 1))
    pointer = json.loads((tmp_path / "current.json").read_bytes())
    e.atomic_json(tmp_path / "current.json", dict(pointer, file="../outside.json"))
    with pytest.raises(ValueError, match="pointer_path"):
        store.read()
    e.atomic_json(tmp_path / "current.json", pointer)
    envelope_path = tmp_path / pointer["file"]
    original = json.loads(envelope_path.read_bytes())
    e.atomic_json(envelope_path, dict(original, sha256="invalid"))
    with pytest.raises(ValueError, match="state_checksum"):
        store.read()
    for field, value, error in [
        ("schema", {}, "state_schema"),
        ("pending", sealed["pending"] * 4, "pending_capacity"),
    ]:
        changed = dict(original["state"], **{field: value})
        digest = e.canonical_hash(changed)
        e.atomic_json(envelope_path, dict(state=changed, sha256=digest))
        e.atomic_json(tmp_path / "current.json", dict(pointer, sha256=digest))
        with pytest.raises(ValueError, match=error):
            store.read()
    (tmp_path / "current.json").unlink()
    with pytest.raises(FileNotFoundError, match="missing_durable_pointer"):
        store.initialize()


def test_two_live_process_readers(tmp_path):
    """SCENARIO-VERIFY-8433-RECOVERY: readers repeatedly observe complete versions."""
    cfg = e.config(8, "uniform")
    store = e.Store(tmp_path / "store", cfg)
    store.initialize()
    config_path = tmp_path / "config.json"
    e.atomic_json(config_path, cfg)
    source = """import json,sys,time
from pathlib import Path
from carnot.verify.block_feedback_state_8433 import Store
p=Path(sys.argv[1]); s=Store(p/'store',json.loads((p/'config.json').read_bytes()))
print('reader_ready',flush=True)
n=0
while not (p/'stop').exists():
 v=s.read(); assert v['version']==v['cursor']//10+(v['phase']=='sealed')
 assert len(v['pending']) in (0,24); n+=1; time.sleep(.002)
print(n,flush=True)
"""
    readers = [
        subprocess.Popen(
            [sys.executable, "-u", "-c", source, str(tmp_path)], stdout=subprocess.PIPE, text=True
        )
        for _ in range(2)
    ]
    for reader in readers:
        assert reader.stdout.readline().strip() == "reader_ready"
    e.drive(tmp_path / "store", cfg)
    (tmp_path / "stop").touch()
    for reader in readers:
        assert int(reader.communicate(timeout=60)[0]) > 0
        assert reader.returncode == 0


def test_primitive_replay_and_forged_receipts(evidence, tmp_path):
    """SCENARIO-REPORT-8433-CLI: reductions need independently saved child operands."""
    from carnot.reporting import block_checkpoint_qualification_8433 as q
    from carnot.reporting.v709_execution import child

    work, raw = evidence
    receipt = child(
        "test_receipt", [sys.executable, "-c", "print('actual evidence')"], tmp_path / "logs"
    )
    output = tmp_path / (q.NAME + ".json")
    value = q.build(work, [receipt], raw, output)
    assert value["pending_high_water"] == {"8": 24, "16": 48}
    e.atomic_json(output, value)
    assert q.replay(output)
    for field in ("block_store_ready_score", "checkpoint_schema_hash", "pending_high_water"):
        changed = deepcopy(value)
        changed[field] = 999
        changed["reproducibility_checksum"] = q.checksum(changed)
        e.atomic_json(output, changed)
        assert not q.replay(output)
    changed = deepcopy(value)
    changed["validation_receipts"][0]["exit_code"] = 73
    changed["reproducibility_checksum"] = q.checksum(changed)
    e.atomic_json(output, changed)
    assert not q.replay(output)
    changed = deepcopy(value)
    changed["validation_receipts"][0]["passed"] = False
    changed["reproducibility_checksum"] = q.checksum(changed)
    e.atomic_json(output, changed)
    assert not q.replay(output)
    e.atomic_json(output, value)
    log = Path(receipt["stdout_path"])
    prior = log.read_bytes()
    log.write_bytes(b"tampered")
    assert not q.replay(output)
    log.write_bytes(prior)


def test_blocked_resources_and_private_run(evidence, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8433-VALIDATION: external absence and owned failure stay distinct."""
    from carnot.reporting import block_checkpoint_qualification_8433 as q
    from carnot.reporting import block_checkpoint_runner_8433 as runner

    work, _ = evidence
    blocked = q.measure(
        tmp_path / "blocked-raw", tmp_path / "blocked-private", tmp_path / "absent-root"
    )
    assert len(blocked["rows"]) == 40
    assert all(r["missing_reason"] for r in blocked["rows"])
    assert blocked["failures"]

    def reuse(raw, private, root):
        copy = deepcopy(work)
        e.atomic_json(raw / "measurement.json", copy)
        return copy

    monkeypatch.setattr(q, "measure", reuse)
    monkeypatch.setattr(
        q,
        "manifest",
        lambda private: [
            dict(
                name="private_unit_control",
                argv=[sys.executable, "-c", "print('actual control')"],
                expected=0,
                deadline=30,
                scope="owned",
            )
        ],
    )
    import coverage

    private = tmp_path / "private"
    private.mkdir()
    cov = coverage.Coverage(config_file=False, data_file=str(private / ".coverage.unit"))
    cov.start()
    q.independent_update()
    cov.stop()
    cov.save()
    cov.json_report(outfile=str(private / "coverage.json"))
    output = tmp_path / (q.NAME + ".json")
    assert runner.run(q.ROOT, output, private) == 0
    assert q.replay(output)
    assert json.loads(output.read_bytes())["block_store_ready_score"] == 0

    def reuse_blocked(raw, private, root):
        copy = deepcopy(blocked)
        e.atomic_json(raw / "measurement.json", copy)
        return copy

    monkeypatch.setattr(q, "measure", reuse_blocked)
    second = tmp_path / "blocked-output" / (q.NAME + ".json")
    assert runner.run(q.ROOT, second, tmp_path / "blocked-run") == 0
    assert q.replay(second)
    called = []
    monkeypatch.setattr(runner, "run", lambda *args: called.append(args) or 0)
    assert runner.main(["--output", str(output)]) == 0
    assert called


@pytest.mark.parametrize(
    "mutation", ["update", "row", "crash", "frozen", "authority", "roster", "byte_hash", "shuffled"]
)
def test_rehashed_primitive_control_failures(evidence, tmp_path, mutation):
    """SCENARIO-REPORT-8433-CLI: repaired aggregate seals cannot repair primitive meaning."""
    from carnot.reporting import block_checkpoint_qualification_8433 as q

    original, _ = evidence
    work = deepcopy(original)
    if mutation == "update":
        work["nonzero_update_controls"]["max_error"] = 1.0
    elif mutation == "row":
        work["rows"][0]["state_hash"] = "forged"
    elif mutation == "crash":
        work["crash_rows"][0]["committed_version"] = 99
    elif mutation == "frozen":
        primitive = q.read(work["frozen_controls"][0]["primitive"])
        primitive["state"]["head"]["parameters"][0] += 1
        target = tmp_path / "frozen.json"
        e.atomic_json(target, primitive)
        work["frozen_controls"][0]["primitive"] = q.reference(target)
    elif mutation == "authority":
        work["authority"]["activated"] = False
    elif mutation == "roster":
        work["rows"].pop()
    elif mutation == "byte_hash":
        work["code_refs"][0]["sha256"] = "forged"
    else:
        ref = next(
            r for r in work["primitive_files"] if "/shuffled/baseline/final.json" in r["path"]
        )
        primitive = q.read(ref)
        primitive["feedback_rng"][1][0] ^= 1
        target = tmp_path / "shuffled/baseline/final.json"
        e.atomic_json(target, primitive)
        ref.update(q.reference(target))
    e.atomic_json(tmp_path / "measurement.json", work)
    value = q.build(work, [], tmp_path, tmp_path / (q.NAME + ".json"))
    target = tmp_path / "candidate.json"
    e.atomic_json(target, value)
    assert not q.replay(target)


def test_required_resources_and_schema_rejection(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8433-VALIDATION: unavailable tools are explicit external gates."""
    from carnot.reporting import block_checkpoint_qualification_8433 as q

    monkeypatch.setattr(q.shutil, "which", lambda path: None)
    work = q.measure(tmp_path / "raw", tmp_path / "private", tmp_path / "absent")
    assert any(r.get("field") == "private_resources" for r in work["failures"])
    with monkeypatch.context() as scoped:
        scoped.setattr(q, "read", lambda ref: dict(methods_ready_score=0))
        work = q.measure(tmp_path / "bad-methods", tmp_path / "bad-private")
    assert any(r.get("field") == "methods_ready_score" for r in work["failures"])


def test_owned_child_failure_keeps_intended_rows(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8433-VALIDATION: failed workers must produce honest accounting."""
    from carnot.reporting import block_checkpoint_qualification_8433 as q
    from carnot.reporting.v709_execution import child

    receipt = child(
        "failed_worker",
        [sys.executable, "-c", "raise SystemExit(7)"],
        tmp_path / "failed",
        expected=0,
    )
    monkeypatch.setattr(q, "child", lambda *args, **kwargs: receipt)
    work = q.measure(tmp_path / "raw", tmp_path / "private")
    assert len(work["rows"]) == 40
    assert all(r["status"] == "failed" for r in work["rows"])
    assert all(r["passed"] is False for r in work["rows"])
    value = q.build(work, [receipt], tmp_path / "raw", tmp_path / (q.NAME + ".json"))
    assert value["verdict_class"] == "disqualified"
    assert value["block_store_ready_score"] == 0
    output = tmp_path / (q.NAME + ".json")
    e.atomic_json(output, value)
    assert q.replay(output)


def test_disk_backing_is_measured(tmp_path, monkeypatch):
    """REQ-REPORT-8433: a pathname alone cannot establish durable disk custody."""
    from carnot.reporting import block_checkpoint_qualification_8433 as q

    assert q.filesystem_type(Path("/var/tmp")) != "unknown"
    with monkeypatch.context() as scoped:
        scoped.setattr(
            Path,
            "read_text",
            lambda *args, **kwargs: (
                "1 0 8:0 / / rw - xfs /dev/private rw\n2 1 0:1 / /tmp rw - tmpfs tmpfs rw\n"
            ),
        )
        assert q.filesystem_type(Path("/var/tmp")) == "xfs"
        assert q.filesystem_type(Path("/tmp")) == "tmpfs"
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_text", lambda *args, **kwargs: "")
        assert q.filesystem_type(tmp_path) == "unknown"


def test_deadline_reserves_validation(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8433-VALIDATION: measurement stops before the validation reserve."""
    from types import SimpleNamespace
    from carnot.reporting import block_checkpoint_qualification_8433 as q

    ticks = iter([0] + [3901 * 10**9] * 200)
    monkeypatch.setattr(q, "time", SimpleNamespace(monotonic_ns=lambda: next(ticks)))
    work = q.measure(tmp_path / "raw", tmp_path / "private")
    assert len(work["rows"]) == 40
    assert any(r.get("check") == "measurement_deadline_s" for r in work["failures"])


def test_upstream_and_scanner_receipt_negative_controls(evidence, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8433-CLI: field and scanner claims need independent operands."""
    from carnot.reporting import block_checkpoint_qualification_8433 as q
    from carnot.reporting.v709_execution import child

    work, raw = evidence
    output = tmp_path / (q.NAME + ".json")
    value = q.build(work, [], raw, output)
    e.atomic_json(output, value)
    original_read = q.read
    with monkeypatch.context() as scoped:
        scoped.setattr(
            q,
            "read",
            lambda ref: (
                dict(methods_ready_score=0)
                if ref["path"] == str(q.ROOT / q.METHODS_PATH)
                else original_read(ref)
            ),
        )
        assert not q.replay(output)
    receipt = child(
        "adversarial",
        [sys.executable, "-u", "scripts/adversarial_verify.py", "--json", str(output)],
        tmp_path / "scanner",
    )
    changed = deepcopy(work)
    changed["adversarial_findings"] = [dict(check="deliberately_forged")]
    e.atomic_json(tmp_path / "measurement.json", changed)
    e.atomic_json(output, q.build(changed, [receipt], tmp_path, output))
    assert not q.replay(output)
