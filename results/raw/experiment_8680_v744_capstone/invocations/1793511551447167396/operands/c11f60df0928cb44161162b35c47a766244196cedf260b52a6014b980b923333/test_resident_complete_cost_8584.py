"""REQ-REPORT-8584 / REQ-VERIFY-8584: complete clocks keep native and lifecycle effects separate."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import resident_complete_cost_8584 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def test_registered_roster_and_original_inputs(tmp_path):
    """SCENARIO-REPORT-8584-COST: fixed paired cells retain the original operands."""
    bundle = e.admit(e.ROOT)
    assert len(bundle["requests"]) == 64
    assert bundle["dependencies_changed"] == []
    roster = e.roster()
    assert len(roster) == 240
    assert {r["length"] for r in roster} == {1, 4, 16}
    assert {r["batch"] for r in roster} == {1, 8}
    assert roster[0]["arm"] == "cold_python"
    with pytest.raises(e.GateFailure):
        e.admit(tmp_path)


def test_paired_bootstrap_and_benefit_boundary():
    """REQ-REPORT-8584: repeat resampling cannot invent independent machines."""
    assert e.interval([2.0] * 10, [1.0] * 10) == [2.0, 2.0]
    assert e.benefit([1.01, 1.1], 1.05, True) == (1, 0)
    assert e.benefit([10.0, 11.0], 1.05, True) == (1, 1)
    for ci, latency, clean in [
        (None, None, True),
        ([1, 2], 1, True),
        ([2, 3], 1.06, True),
        ([2, 3], 1, False),
    ]:
        assert e.benefit(ci, latency, clean) == (0, 0)


def test_real_batched_children_and_rejections(tmp_path):
    """REQ-VERIFY-8584: actual native and Python peers retain startup and failures."""
    bundle = e.admit(e.ROOT)
    for arm in ["cold_python", "resident_python", "cold_native", "resident_native"]:
        slot = dict(arm=arm, batch=8, length=1, repeat=0)
        row = e.session(bundle, slot, tmp_path / arm)
        assert row["exit_code"] == 0
        assert len(row["requests"]) == 64
        assert row["requests"][0]["latency_s"] >= row["startup_s"]
        assert row["peak_rss_kib"] > 0
        assert sum(r["duration_s"] for r in row["operations"]) == pytest.approx(row["duration_s"])
    receipt = child(
        "bad_worker",
        [sys.executable, str(e.ROOT / e.CLI), "--worker", str(tmp_path / "missing")],
        tmp_path / "logs",
        expected=1,
    )
    assert receipt["passed"]


def test_missing_gate_and_owned_failures(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8584-REPLAY: absent external bytes differ from owned failure."""
    monkeypatch.setattr(
        e, "admit", lambda root: (_ for _ in ()).throw(e.GateFailure(root, "missing", 1, None))
    )
    work = e.measure(e.ROOT, tmp_path / "raw")
    assert work["gate_check_summary"][0]["observed"] is None
    value = e.artifact(work, [], tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked"
    failed = e.artifact(work, [dict(name="owned", passed=False)], tmp_path / (e.NAME + ".json"))
    assert failed["verdict_class"] == "disqualified"
    assert not e.replay(tmp_path / "absent")


def test_cold_replay_and_rehashed_tampering(tmp_path):
    """SCENARIO-VERIFY-8584-REPLAY: immutable source rows govern repaired envelopes."""
    work_path = Path(__import__("os").environ.get("CARNOT_8584_MEASUREMENT", tmp_path / "absent"))
    if not work_path.exists():
        bundle = e.admit(e.ROOT)
        work = e.empty_work(tmp_path / "raw")
        work["bundle"] = bundle
        atomic_json(tmp_path / "raw" / "measurement.json", work)
    else:
        work = json.loads(work_path.read_bytes())
        work["raw"] = str(tmp_path / "raw")
        atomic_json(tmp_path / "raw" / "measurement.json", work)
    output = tmp_path / (e.NAME + ".json")
    value = e.artifact(work, [], output)
    atomic_json(output, value)
    assert e.replay(output)
    changed = deepcopy(value)
    changed["resident_cost_ready_score"] = 1 - value["resident_cost_ready_score"]
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(output, changed)
    assert not e.replay(output)


def test_measure_checkpoint_failures_and_budget(tmp_path, monkeypatch):
    """REQ-VERIFY-8584: successful checkpoints, failed children and budget censoring stay distinct."""
    import time
    from types import SimpleNamespace

    bundle = e.admit(e.ROOT)
    original = e.session
    slots = [dict(arm="cold_python", batch=8, length=1, repeat=i) for i in range(4)]
    monkeypatch.setattr(e, "roster", lambda: slots)

    def run_session(bundle, slot, raw, **options):
        if slot["repeat"] == 1:
            raise ValueError("owned_child_failure")
        if slot["repeat"] == 2:
            raise TimeoutError("owned_child_deadline")
        return original(bundle, slot, raw, **options)

    monkeypatch.setattr(e, "session", run_session)
    work = e.measure(e.ROOT, tmp_path / "measured")
    assert [r["status"] for r in work["rows"]] == ["completed", "failed", "censored", "completed"]
    reduced = e.reduce(work)
    assert len(reduced["paired_session_rows"]) == 2
    assert len(reduced["censoring_rows"]) == 2
    for clocks in [[0, 2401, 2402, 2403, 2404, 2405], [0, 1, 2401, 2402, 2403, 2404, 2405]]:
        iterator = iter(clocks)
        monkeypatch.setattr(
            e,
            "time",
            SimpleNamespace(monotonic=lambda: next(iterator), monotonic_ns=time.monotonic_ns),
        )
        bounded = e.measure(e.ROOT, tmp_path / str(clocks[1]))
        assert all(r["status"] == "censored" for r in bounded["rows"])


def test_admission_drift_and_unavailable_inputs(tmp_path, monkeypatch):
    """REQ-REPORT-8584: changed qualified dependencies and absent resources cannot inherit parity."""
    original = e.sha256_file
    monkeypatch.setattr(
        e,
        "sha256_file",
        lambda path: (
            "sha256:changed" if path.name == "resident_measure_8572.py" else original(path)
        ),
    )
    with pytest.raises(e.GateFailure):
        e.admit(e.ROOT)
    monkeypatch.setattr(e, "admit", lambda root: (_ for _ in ()).throw(OSError("unavailable")))
    work = e.measure(e.ROOT, tmp_path / "raw")
    assert work["gate_check_summary"][0]["field"] == "available_original_inputs"


def test_replay_rejects_rehashed_bundle(tmp_path):
    """SCENARIO-VERIFY-8584-REPLAY: repaired summaries cannot replace pinned scientific operands."""
    bundle = e.admit(e.ROOT)
    work = e.empty_work(tmp_path / "raw")
    work["bundle"] = deepcopy(bundle)
    work["bundle"]["head"]["temperature"] *= 2
    atomic_json(tmp_path / "raw" / "measurement.json", work)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.artifact(work, [], output))
    assert not e.replay(output)


def test_real_worker_negative_frames(tmp_path):
    """REQ-VERIFY-8584: malformed and expired batch requests terminate genuine private children."""
    from carnot.reporting import resident_measure_8572 as peer
    from unittest.mock import patch

    bundle = e.admit(e.ROOT)
    config = dict(head=bundle["head"], extension=bundle["extension"])
    for kind in ["config", "duplicate", "expired", "batch", "json"]:
        with patch.object(peer, "CLI", str(e.ROOT / e.CLI)):
            process = peer.Session(config, tmp_path / kind, "python")
        request = peer.w.envelope("r", process.digest, dict(rows=bundle["requests"][:8]))
        if kind == "config":
            request["config_hash"] = "wrong"
        elif kind == "expired":
            request["deadline_ns"] = 0
        elif kind == "batch":
            request["row"]["rows"] = []
        elif kind == "duplicate":
            assert process.request(dict(rows=bundle["requests"][:8]), "r")["status"] == "ok"
        if kind == "json":
            __import__("os").write(process.input, b"\x00\x00\x00\x01{")
        else:
            peer.w.write_frame(process.input, request)
        receipt = process.close(shutdown=False)
        assert receipt["exit_code"] == 1


def test_deadline_reaps_real_session(tmp_path):
    """REQ-VERIFY-8584: an exhausted measurement budget reaps its actual private child."""
    import time

    bundle = e.admit(e.ROOT)
    with pytest.raises(TimeoutError):
        e.session(
            bundle,
            dict(arm="resident_python", batch=8, length=1, repeat=0),
            tmp_path / "deadline",
            deadline=time.monotonic(),
        )


def test_worker_saturation_without_hidden_requests(tmp_path, monkeypatch):
    """REQ-VERIFY-8584: the bounded worker fails when its request budget has no shutdown slot."""
    from carnot.reporting import resident_complete_worker_8584 as worker

    bundle = e.admit(e.ROOT)
    config = dict(head=bundle["head"], extension=bundle["extension"])
    monkeypatch.setattr(worker, "range", lambda n: range(64), raising=False)
    monkeypatch.setattr(worker.w, "configuration", lambda *args: (config, None, None))
    monkeypatch.setattr(
        worker.w,
        "read_frame",
        lambda *args: json.dumps(
            worker.w.envelope("a", "digest", dict(rows=bundle["requests"][:1]))
        ).encode(),
    )
    identifiers = iter(range(64))
    monkeypatch.setattr(
        worker.w,
        "read_frame",
        lambda *args: json.dumps(
            worker.w.envelope(str(next(identifiers)), "digest", dict(rows=bundle["requests"][:1]))
        ).encode(),
    )
    captured = []
    monkeypatch.setattr(worker.w, "write_frame", lambda fd, value, *args: captured.append(value))
    assert worker.serve(tmp_path / "config", "digest", "python") == 1
    assert captured[-1]["status"] == "ok"


def test_real_private_cli_and_owned_failure(tmp_path):
    """SCENARIO-VERIFY-8584-REPLAY: a genuine outside-checkout CLI never earns unrun validation."""
    from carnot.reporting import resident_complete_runner_8584 as runner

    work = e.empty_work(tmp_path / "source")
    work["first_cell_forecast"] = dict(observed_s=1.0, projected_total_s=60.0)
    work["gate_check_summary"] = [e.GateFailure(tmp_path / "missing", "input", 1, None).row]
    measurement = tmp_path / "source" / "measurement.json"
    atomic_json(measurement, work)
    output = tmp_path / (e.NAME + ".json")
    receipt = child(
        "private_cli",
        [
            sys.executable,
            str(e.ROOT / e.CLI),
            "--date",
            "20261016",
            "--private-output",
            str(output),
            "--resume-measurement",
            str(measurement),
        ],
        tmp_path / "logs",
        deadline=90,
    )
    assert receipt["passed"]
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    with pytest.raises(ValueError, match="not_ready"):
        e.consume(output)
    assert runner.main(["--date", "20261015"]) == 2
    assert runner.main(["--private-output", str(e.ROOT / "results" / (e.NAME + ".json"))]) == 2
    assert runner.main(["--cold-replay", str(tmp_path / "missing")]) == 1


def test_live_deadline_and_error_cleanup(tmp_path, monkeypatch):
    """REQ-VERIFY-8584: a deadline inside a session and failed responses reap the real peer."""
    import time
    from types import SimpleNamespace
    from carnot.reporting import resident_measure_8572 as peer

    bundle = e.admit(e.ROOT)
    clock = iter([0.0, 100.0])
    monkeypatch.setattr(
        e, "time", SimpleNamespace(monotonic_ns=time.monotonic_ns, monotonic=lambda: next(clock))
    )
    with pytest.raises(TimeoutError):
        e.session(
            bundle,
            dict(arm="resident_python", batch=8, length=1, repeat=0),
            tmp_path / "deadline",
            deadline=100,
        )
    assert (
        json.loads((tmp_path / "deadline" / "censored_session.json").read_bytes())["exit_code"] < 0
    )
    monkeypatch.setattr(e, "time", time)
    monkeypatch.setattr(peer.Session, "request", lambda *args: dict(status="error"))
    with pytest.raises(ValueError, match="worker_response"):
        e.session(
            bundle, dict(arm="resident_python", batch=8, length=1, repeat=0), tmp_path / "error"
        )
    assert (tmp_path / "error" / "censored_session.json").is_file()


def test_immutable_roster_and_operation_tampering(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8584-REPLAY: rehashed duplicates and redistributed operation shares fail."""
    bundle = e.admit(e.ROOT)
    slot = dict(arm="cold_python", batch=8, length=1, repeat=0)
    process = e.session(bundle, slot, tmp_path / "process")
    primitive = tmp_path / "primitive.json"
    atomic_json(primitive, dict(slot=slot, sessions=[process]))
    row = dict(
        slot,
        unit_id="cell",
        status="completed",
        primitive_reference=e.reference(primitive),
        missing_reason=None,
    )
    work = e.empty_work(tmp_path / "work")
    work.update(bundle=bundle, rows=[row])
    monkeypatch.setattr(e, "roster", lambda: [slot])
    assert len(e.reduce(work)["per_request_rows"]) == 64
    work["rows"] = [row, row]
    with pytest.raises(e.GateFailure):
        e.reduce(work)
    work["rows"] = [row]
    process["operations"][0]["duration_s"] += 0.00001
    process["operations"][1]["duration_s"] -= 0.00001
    atomic_json(primitive, dict(slot=slot, sessions=[process]))
    row["primitive_reference"] = e.reference(primitive)
    with pytest.raises(e.GateFailure, match="operation_clock_reconstruction"):
        e.reduce(work)


def test_orchestration_and_downstream_reader(tmp_path, monkeypatch):
    """REQ-VERIFY-8584: bounded orchestration preserves external blockage and reader readiness."""
    from carnot.reporting import resident_complete_runner_8584 as runner

    work = e.empty_work(tmp_path / "source")
    work["gate_check_summary"] = [e.GateFailure(tmp_path / "missing", "input", 1, None).row]
    measurement = tmp_path / "source" / "measurement.json"
    atomic_json(measurement, work)
    monkeypatch.setattr(runner.validation, "validate", lambda *args: [])
    monkeypatch.setattr(runner.validation, "finish", lambda *args: [])
    output = tmp_path / (e.NAME + ".json")
    assert runner.run(output, measurement=measurement) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    value["resident_cost_ready_score"] = 1
    atomic_json(output, value)
    monkeypatch.setattr(e, "replay", lambda path: True)
    assert e.consume(output)["passed"]
    assert runner.main(["--consume", str(output)]) == 0
    monkeypatch.setattr(e, "measure", lambda root, raw: dict(work, raw=str(raw)))
    assert runner.run(tmp_path / "second" / output.name, private=True) == 0
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: 0)
    assert runner.main([]) == 0


def test_validation_receipts_need_real_bytes(tmp_path):
    """REQ-VERIFY-8584: invented passing checks and changed logs cannot confer readiness."""
    receipt = child(
        "real_receipt",
        [sys.executable, "-c", 'print("actual validation child")'],
        tmp_path / "logs",
    )
    assert e.receipt_bytes({}, [receipt])
    assert not e.receipt_bytes({}, [dict(name="cold", passed=True)])
    work = dict(
        exact_union_receipt=dict(
            passed=True, collected=["test::node"], passed_node_ids=["test::node"]
        ),
        python_coverage=dict(
            totals=dict(percent_covered=100, missing_lines=0, covered_lines=1, num_statements=1)
        ),
    )
    assert e.receipt_bytes(
        work,
        [
            dict(name="exact_disjoint_node_union", passed=True),
            dict(name="owned_python_coverage_100", passed=True),
        ],
    )
    assert not e.receipt_bytes({}, [dict(name="owned_python_coverage_100", passed=True)])
    Path(receipt["stdout_path"]).write_text("changed")
    assert not e.receipt_bytes({}, [receipt])
