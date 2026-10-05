"""REQ-VERIFY-8159 / REQ-REPORT-8159: complete batches retain durable decisions."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess

import numpy as np
import pytest

from carnot.verify import durable_batch_8159 as e
from carnot.reporting import durable_batch_execution_8159 as runner


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """REQ-VERIFY-8159: read original natural inputs without changing their storage."""
    return e.inputs(e.ROOT, tmp_path_factory.mktemp("8159-inputs"))


@pytest.fixture(scope="module")
def native(data):
    """REQ-VERIFY-8159: exercise the loaded library rather than a substitute."""
    return e.old.prior.old.host.load_binding(data)[0]


@pytest.fixture
def small():
    """REQ-VERIFY-8159: small private repetitions test transport without speed credit."""
    return dict(e.CONFIG, batches=[1, 8], repetitions=1)


def test_store_atomicity_identity_and_corruption(tmp_path):
    """SCENARIO-VERIFY-8159: stable identities deduplicate and reject changed content."""
    path = tmp_path / "store.json"
    s = e.Store(path, "frozen-head")
    row = dict(
        request_id="a",
        input_hash="hash",
        source_cluster_id="source",
        values=[1],
        probability=0.4,
        action="escalate",
        head_hash="frozen-head",
    )
    first = s.commit([row])
    assert first[0]["sequence"] == 1 and s.commit([row]) == first
    assert e.Store(path, "frozen-head").state == s.state
    with pytest.raises(ValueError, match="request_id_input_hash"):
        s.commit([dict(row, input_hash="changed")])
    with pytest.raises(ValueError, match="duplicate_batch_id"):
        s.commit([row, row])
    with pytest.raises(ValueError, match="store_custody"):
        e.Store(path, "other-head")
    value = json.loads(path.read_text())
    value["sha256"] = "wrong"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="store_custody"):
        e.Store(path, "frozen-head")


@pytest.mark.parametrize("condition", ["cold", "warm", "restart"])
@pytest.mark.parametrize("arrival", ["all_at_once", "cadence_4ms"])
def test_complete_transactions_and_queue(data, native, tmp_path, condition, arrival):
    """REQ-VERIFY-8159 / E2E-003/004: parity, durable return and actual wait receipts."""
    slot = dict(unit_id="private", batch=8, repetition=0, condition=condition, arrival=arrival)
    rows = [e.transaction(data, native, slot, arm, tmp_path / arm) for arm in e.ARMS]
    assert e.parity(rows)
    for row in rows:
        assert row["duration_ns"] >= row["components"]["arithmetic_ns"] > 0
        assert len(row["requests"]) == 8 and row["commit_count"] >= 1
        state = e.Store(Path(row["store_path"]), e.canonical_hash(data["head"])).state
        assert len(state["records"]) == 8
        for r in row["requests"]:
            assert r["enqueue_ns"] <= r["execution_start_ns"] <= r["execution_end_ns"]
            assert r["execution_end_ns"] <= r["serialization_start_ns"] <= r["serialization_end_ns"]
            assert (
                r["serialization_end_ns"]
                <= r["commit_start_ns"]
                <= r["commit_end_ns"]
                <= r["response_ns"]
            )
            assert r["latency_ns"] == r["response_ns"] - r["enqueue_ns"]
            assert r["queue_ns"] == r["execution_start_ns"] - r["enqueue_ns"]
            expected = e.old.engine.scalar_probability(data["head"], data["geometry"], r["values"])
            assert abs(r["probability"] - expected) < 1e-10
        if arrival == "cadence_4ms":
            assert (
                rows[-1]["requests"][-1]["enqueue_ns"] - rows[-1]["requests"][0]["enqueue_ns"]
                >= 28_000_000
            )
    assert rows[0]["commit_count"] == 8
    assert rows[-1]["commit_count"] == 1


def test_inputs_reduction_and_replay(data, native, small, tmp_path):
    """REQ-REPORT-8159: recompute rates and intervals and reject changed custody."""
    assert data["ready"] and e.MODEL_SPECS == []
    assert not e.inputs(tmp_path / "absent", tmp_path / "blocked")["ready"]
    raw = tmp_path / "raw"
    work = e.measure(data, native, raw, small)
    reduced = e.reduce_rows(work)
    assert reduced["passed"] and reduced["completed_count"] == 12
    assert all(len(r["ratios"]) == 1 for r in reduced["intervals"])
    receipts = [dict(passed=True, normal_exit=True)]
    value = e.build(data, work, raw, receipts, "20261005", 1.0, False)
    assert value["host_batch_ready_score"] == 1
    assert value["rejected_update_cost"] is None and not value["call_ledger"]
    path = tmp_path / "candidate.json"
    e.atomic_json(path, value)
    assert e.replay(path)
    changed = deepcopy(value)
    changed["rows"][0]["numerator"] += 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    e.atomic_json(path, changed)
    assert not e.replay(path)
    changed = deepcopy(value)
    changed["source_artifact_hashes"][0]["sha256"] = "wrong"
    changed["reproducibility_checksum"] = e.checksum(changed)
    e.atomic_json(path, changed)
    assert not e.replay(path)
    e.atomic_json(path, value)
    Path(work["pairs"][0]["arms"][0]["store_path"]).write_text("{}")
    assert not e.replay(path) and not e.replay(tmp_path / "absent.json")
    assert (
        e.build(data, work, raw, [dict(passed=False)], "20261005", 1, False)["verdict_class"]
        == "disqualified"
    )
    bad = deepcopy(work)
    bad["pairs"][0]["arms"][1]["requests"][0]["action"] = "wrong"
    assert not e.reduce_rows(bad)["passed"]


def test_crashes_deduplication(data, native, tmp_path):
    """SCENARIO-VERIFY-8159: actual child exits preserve acknowledged results once."""
    rows = e.durability(data, native, tmp_path)
    assert len(rows) == 2
    assert all(r["passed"] and r["exactly_once"] and r["causal_order_preserved"] for r in rows)
    assert [r["crash_exit"] for r in rows] == [71, 72]
    assert [r["unacknowledged_present_before_retry"] for r in rows] == [0, 8]


def test_script_routes(tmp_path):
    """SCENARIO-REPORT-8159: success, block, tamper and cold replay outside checkout."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    command = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]
    output = tmp_path / (e.NAME + ".json")
    for extra, expected in [
        (["--private-small", "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--date", "invalid"], 2),
        (["--cold-replay", str(tmp_path / "absent")], 1),
    ]:
        result = subprocess.run(
            command + extra, cwd=tmp_path, env=env, capture_output=True, timeout=90
        )
        assert result.returncode == expected, result.stdout.decode() + result.stderr.decode()
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive" and value["host_batch_ready_score"] == 0
    value["host_batch_ready_score"] = 1
    e.atomic_json(output, value)
    assert (
        subprocess.run(
            command + ["--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        ).returncode
        == 1
    )
    blocked = tmp_path / "blocked" / output.name
    result = subprocess.run(
        command + ["--private-small", "--root", str(tmp_path / "absent"), "--output", str(blocked)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"


def test_orchestration_failures_and_preservation(data, tmp_path, monkeypatch):
    """REQ-REPORT-8159: owned failure zeroes readiness and previous bytes survive."""
    monkeypatch.setattr(e, "CONFIG", dict(e.CONFIG, batches=[1], repetitions=1))
    monkeypatch.setattr(
        runner,
        "execute",
        lambda commands, raw, expected=0: [
            dict(name=c.name, passed=True, normal_exit=True) for c in commands
        ],
    )
    path = tmp_path / (e.NAME + ".json")
    assert runner.main(["--output", str(path)]) == 0
    assert runner.main(["--output", str(path)]) == 0
    assert list(tmp_path.glob("raw/**/preserved_primary.json"))
    assert runner.main(["--cold-replay", str(path)]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    with pytest.raises(SystemExit):
        runner.main(["--private-small"])
    monkeypatch.setattr(
        runner,
        "execute",
        lambda commands, raw, expected=0: [
            dict(name=c.name, passed=False, normal_exit=True) for c in commands
        ],
    )
    assert runner.main(["--output", str(tmp_path / "failed" / path.name)]) == 1
    monkeypatch.setattr(runner, "main", lambda: 0)
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(e.ROOT / e.CLI), run_name="__main__")
    assert raised.value.code == 0


def test_rehashed_causal_corruption_and_source_identity(data, native, tmp_path):
    """SCENARIO-VERIFY-8159: rehashing cannot legitimize broken order or substituted inputs."""
    raw = tmp_path / "raw"
    config = dict(
        e.CONFIG, batches=[1], repetitions=1, conditions=["cold"], arrivals=["all_at_once"]
    )
    work = e.measure(data, native, raw, config)
    value = e.build(data, work, raw, [dict(passed=True, normal_exit=True)], "20261005", 1, False)
    path = tmp_path / "candidate.json"
    store_path = Path(work["pairs"][0]["arms"][0]["store_path"])
    envelope = json.loads(store_path.read_text())
    changed = deepcopy(envelope)
    changed["state"]["records"][0]["previous_hash"] = "wrong"
    changed["sha256"] = e.canonical_hash(changed["state"])
    e.atomic_json(store_path, changed)
    with pytest.raises(ValueError, match="store_custody"):
        e.Store(store_path, e.canonical_hash(data["head"]))
    e.atomic_json(store_path, envelope)
    changed_work = deepcopy(work)
    for arm in changed_work["pairs"][0]["arms"]:
        arm["requests"][0]["input_hash"] = "substituted_input"
    e.atomic_json(raw / "primitive_rows.json", changed_work)
    value["raw_shard_hashes"] = [e.reference(Path(r["path"])) for r in value["raw_shard_hashes"]]
    value["primitive_rows"] = e.reference(raw / "primitive_rows.json")
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(path, value)
    assert not e.replay(path)


def test_private_crash_hooks_and_wait_limit(data, native, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8159: failure controls exercise both commit sides and deadline flushing."""

    def injected_exit(code):
        raise RuntimeError(str(code))

    monkeypatch.setattr(e.os, "_exit", injected_exit)
    row = dict(request_id="private", input_hash="same")
    store = e.Store(tmp_path / "crash.json", "head")
    for point, code in [("before_commit", "71"), ("after_commit", "72")]:
        with pytest.raises(RuntimeError, match=code):
            store.commit([row], point)
    slot = dict(unit_id="deadline", batch=32, repetition=0, condition="cold", arrival="cadence_4ms")
    result = e.transaction(data, native, slot, e.ARMS[-1], tmp_path / "deadline")
    assert result["commit_count"] >= 4 and len(result["requests"]) == 32
    result = e.transaction(
        data,
        native,
        dict(slot, batch=8, arrival="all_at_once"),
        e.ARMS[-1],
        tmp_path / "zero_wait",
        dict(e.CONFIG, maximum_wait_ns=0),
    )
    assert result["commit_count"] == 8
    manifest = tmp_path / "worker.json"
    e.atomic_json(manifest, dict(data=data, slot=dict(slot, batch=1), raw=str(tmp_path / "worker")))
    assert runner.main(["--worker", str(manifest)]) == 0


def test_replay_failure_controls(data, native, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8159: independent checks reject even a corrupted shared reducer."""
    malformed = tmp_path / "malformed"
    e.atomic_json(malformed / "results/experiment_8145_v704_natural_service_cost.json", {})
    assert not e.inputs(malformed, tmp_path / "bad_inputs")["ready"]
    config = dict(
        e.CONFIG, batches=[1], repetitions=1, conditions=["cold"], arrivals=["all_at_once"]
    )
    raw = tmp_path / "raw"
    original = e.measure(data, native, raw, config)
    receipts = [dict(passed=True, normal_exit=True)]
    value = e.build(data, original, raw, receipts, "20261005", 1, False)
    assert (
        e.build(data, original, raw, receipts, "20261005", 1, True)["verdict_class"]
        == "circular_positive"
    )
    path = tmp_path / "candidate.json"
    for kind in ["checksum", "code", "log", "config"]:
        changed = deepcopy(value)
        if kind == "code":
            changed["code_config_hashes"][e.OWNED[0]] = "wrong"
        elif kind == "log":
            log = tmp_path / "validation.log"
            log.write_text("changed")
            changed["validation_receipts"] = [dict(log_path=str(log), log_sha256="wrong")]
        elif kind == "config":
            changed["measurement_config"]["seed"] += 1
        changed["reproducibility_checksum"] = "wrong" if kind == "checksum" else e.checksum(changed)
        e.atomic_json(path, changed)
        assert not e.replay(path)
    for kind in ["store_hash", "probability", "committed"]:
        work = deepcopy(original)
        for arm in work["pairs"][0]["arms"]:
            if kind == "store_hash":
                arm["store_sha256"] = "wrong"
            elif kind == "probability":
                arm["requests"][0]["probability"] += 0.1
            else:
                arm["requests"][0]["previous_hash"] = "wrong"
        e.atomic_json(raw / "primitive_rows.json", work)
        changed = e.build(data, work, raw, receipts, "20261005", 1, False)
        e.atomic_json(path, changed)
        assert not e.replay(path)
    e.atomic_json(raw / "primitive_rows.json", original)
    value = e.build(data, original, raw, receipts, "20261005", 1, False)
    latency_path = Path(value["request_latency_rows"]["path"])
    saved = json.loads(latency_path.read_text())
    e.atomic_json(latency_path, dict(rows=[]))
    changed = deepcopy(value)
    changed["request_latency_rows"] = e.reference(latency_path)
    changed["raw_shard_hashes"] = [
        e.reference(Path(r["path"])) for r in changed["raw_shard_hashes"]
    ]
    changed["reproducibility_checksum"] = e.checksum(changed)
    e.atomic_json(path, changed)
    assert not e.replay(path)
    e.atomic_json(latency_path, saved)
    value = e.build(data, original, raw, receipts, "20261005", 1, False)
    reduced = e.reduce_rows(original)
    for field in ["p50_request_latency_ns", "complete_batch_throughput"]:
        changed = deepcopy(value)
        changed["natural_batch_rows"][0][field] += 1
        changed["reproducibility_checksum"] = e.checksum(changed)
        bad_reduction = dict(reduced, summaries=changed["natural_batch_rows"])
        with monkeypatch.context() as patch:
            patch.setattr(e, "reduce_rows", lambda work: bad_reduction)
            e.atomic_json(path, changed)
            assert not e.replay(path)
    blocked_raw = tmp_path / "blocked_raw"
    blocked = e.inputs(tmp_path / "absent", blocked_raw)
    empty = dict(config=config, pairs=[], durability=[])
    e.atomic_json(blocked_raw / "primitive_rows.json", empty)
    blocked_value = e.build(blocked, empty, blocked_raw, receipts, "20261005", 1, False)
    e.atomic_json(path, blocked_value)
    assert e.replay(path)
    assert not e.reduce_rows(empty)["passed"]
    store_path = Path(original["pairs"][0]["arms"][0]["store_path"])
    envelope = json.loads(store_path.read_text())
    envelope["state"]["causal_hash"] = "wrong"
    envelope["sha256"] = e.canonical_hash(envelope["state"])
    e.atomic_json(store_path, envelope)
    with pytest.raises(ValueError, match="store_custody"):
        e.Store(store_path, e.canonical_hash(data["head"]))
