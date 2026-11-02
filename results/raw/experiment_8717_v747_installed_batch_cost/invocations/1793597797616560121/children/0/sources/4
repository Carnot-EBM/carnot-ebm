"""REQ-REPORT-8717 / REQ-VERIFY-8717: equal work and authentic complete clocks."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import installed_batch_cost_8717 as e
from carnot.reporting import installed_batch_cost_measure_8717 as m
from carnot.reporting import installed_batch_cost_runner_8717 as r
from carnot.reporting.v709_execution import child


def test_authority_and_roster(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8717-COST: the original protocol fixes every intended unit."""
    inputs = e.inputs(e.ROOT)
    assert inputs["passed"], inputs["gate_check_summary"]
    slots = e.roster(inputs["method"])
    assert len(slots) == 60 and sum(x["item_evaluations"] for x in slots) == 61440
    assert slots == e.roster(inputs["method"])
    assert len({x["unit_id"] for x in slots}) == 60
    assert all(slots[i]["repeat"] == slots[i + 1]["repeat"] for i in range(0, 60, 2))
    assert not e.inputs(tmp_path)["passed"]
    monkeypatch.setattr(e, "PIN", "sha256:changed")
    assert not e.inputs(e.ROOT)["passed"]


def test_process_intervals_and_nulls():
    """SCENARIO-REPORT-8717-COST: ten process repeats, never individual items, enter CIs."""
    rows = [
        dict(
            s,
            status="completed",
            duration_s=2.0 if s["arm"] == "python" else 1.0,
            steady_p95_ms=1.0,
            passed=True,
        )
        for s in e.roster(e.method())
    ]
    summary = e.aggregate(rows)
    assert summary["batching_benefit_score"] == summary["same_batch_10x_score"] == 0
    assert summary["paired_intervals"][0]["lower95"] == 1
    assert all(
        x["paired_repeats"] == 10 and x["bootstrap_draws"] == 10000
        for x in summary["paired_intervals"]
    )
    for row in rows:
        if row["arm"] == "native" and row["batch"] == 16:
            row["duration_s"] = 0.05
    assert e.aggregate(rows)["batching_benefit_score"] == 1
    assert e.aggregate(rows)["same_batch_10x_score"] == 1
    rows[0]["status"] = "censored"
    assert not e.aggregate(rows)["complete"]


def test_private_real_cli_and_replay(tmp_path):
    """SCENARIO-VERIFY-8717-CUSTODY: actual installed workers and fresh cold readers."""
    output = tmp_path / (e.NAME + ".json")
    args = [sys.executable, "-u", e.CLI, "--date", "20261102"]
    receipt = child(
        "private", args + ["--private-output", str(output)], tmp_path / "logs", deadline=420
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["completed_count"] == 60
    assert value["sample_size_budget"]["completed_item_evaluations"] == 61440
    assert all(x["passed"] for x in value["parity_rows"])
    authority = value["primitive_reference"]
    assert r.replay(output, authority)
    changed = deepcopy(value)
    changed["completed_count"] += 1
    e.atomic_json(output, changed)
    assert not r.replay(output, authority)
    e.atomic_json(output, value)
    assert not r.replay(output, None)
    assert not r.replay(tmp_path / "missing", authority)
    work = e.load(authority)
    assert e.build(work, output)["verdict_class"] == "null"
    work["private_control"] = False
    work["validation"] = dict(
        receipts=[dict(passed=True)],
        coverage=dict(complete=True),
        custody=dict(passed=True),
        raw_shard_hashes=[],
    )
    assert e.build(work, output)["batch_cost_ready_score"] == 1
    work["owned_error"] = "failed"
    assert e.build(work, output)["verdict_class"] == "disqualified"
    assert r.main(["--date", "wrong"]) == 2
    assert r.main(["--private-output", str(e.ROOT / "unsafe.json")]) == 2
    assert (
        r.main(
            ["--cold-replay", str(output), "--authority", authority["path"], authority["sha256"]]
        )
        == 0
    )


@pytest.fixture
def primitive(tmp_path):
    """SCENARIO-VERIFY-8717-CUSTODY: small real workers exercise failure boundaries."""
    inputs = e.inputs(e.ROOT)
    config = tmp_path / "config.json"
    e.atomic_json(
        config,
        dict(
            inputs["workload"]["configuration_template"], extension=inputs["workload"]["extension"]
        ),
    )
    slot = dict(
        unit_id="control",
        batch=16,
        panels=1,
        repeat=0,
        arm="python",
        status="censored",
        item_evaluations=64,
    )
    command = [sys.executable, "-u", str(e.ROOT / "scripts/experiments/resident_worker_8608.py")]
    value = m.session(
        command, config, slot, inputs["requests"], tmp_path, tmp_path / "worker", 2**63
    )
    return inputs, config, slot, command, value


def test_raw_rejections(primitive, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8717-COST: altered frames, identities and clocks never qualify."""
    inputs, config, slot, command, value = primitive
    for mode, message in [
        ("config", "configuration_or_worker_identity"),
        ("count", "request_count"),
        ("frame", "frame_operands"),
        ("parent", "parent_clocks"),
        ("worker", "worker_clocks"),
        ("complete", "complete_cost_clocks"),
    ]:
        changed = deepcopy(value)
        if mode == "config":
            changed["actual_exit"] = 1
        elif mode == "count":
            changed["observations"].pop()
        elif mode == "frame":
            changed["observations"][0]["request"]["rows"] = []
        elif mode == "parent":
            changed["observations"][0]["packed_ns"] = 0
        elif mode == "worker":
            changed["observations"][0]["response"]["spans"][0]["start_ns"] = 0
        else:
            changed["start_ns"] = changed["ready_ns"] + 1
        with pytest.raises(ValueError, match=message):
            m.reduce(changed, inputs)
    changed = deepcopy(value)
    changed["observations"][0]["response"]["responses"][0]["decision"] = "forged"
    assert not m.reduce(changed, inputs)["process_row"]["passed"]
    with pytest.raises(TimeoutError, match="admission"):
        m.session(command, config, slot, inputs["requests"], tmp_path, tmp_path / "early", 0)
    with pytest.raises(ValueError, match="worker_response"):
        m.session(
            command, config, slot, [dict(unit_id="invalid")], tmp_path, tmp_path / "reject", 2**63
        )
    failure = json.loads((tmp_path / "reject/failed_child.json").read_bytes())
    assert failure["observations"][0]["response"]["status"] == "error"
    assert not Path(f"/proc/{failure.get('pid', -1)}").exists()
    clock = iter([0, 0, 0, 100_000_000_000])
    monkeypatch.setattr(m.time, "monotonic_ns", lambda: next(clock, 100_000_000_000))
    with pytest.raises(TimeoutError, match="stop_deadline"):
        m.session(
            command,
            config,
            slot,
            inputs["requests"],
            tmp_path,
            tmp_path / "deadline",
            100_000_000_000,
        )


def test_measure_and_install_failures(primitive, tmp_path, monkeypatch):
    """REQ-VERIFY-8717: failed cells remain visible and block all benefit scores."""
    inputs, config, slot, command, value = primitive
    monkeypatch.setattr(m, "prepare", lambda *args: (command, config, []))
    monkeypatch.setattr(
        m, "session", lambda *args: (_ for _ in ()).throw(ValueError("broken_worker"))
    )
    measured = m.measure(tmp_path / "failed", tmp_path, inputs, 2**63)
    assert measured["process_rows"][0]["status"] == "failed"
    assert sum(x["status"] == "censored" for x in measured["process_rows"]) == 59

    def retained_session(*args):
        e.atomic_json(args[-2] / "session.json", value)
        return value

    monkeypatch.setattr(m, "session", retained_session)
    monkeypatch.setattr(
        m,
        "reduce",
        lambda *args: dict(
            process_row=dict(slot, status="completed", passed=False),
            request_rows=[],
            parity_rows=[],
            complete_cost_rows=[],
        ),
    )
    assert (
        m.measure(tmp_path / "parity", tmp_path, inputs, 2**63)["owned_error"] == "per_item_parity"
    )


def test_prepare_rejections(tmp_path, monkeypatch):
    """REQ-VERIFY-8717: unsuccessful installation and substituted extension fail closed."""
    inputs = e.inputs(e.ROOT)
    original = m.child

    def fake(name, args, raw, **kwargs):
        if name == "venv":
            (Path(args[-1]) / "lib/python3.12/site-packages").mkdir(parents=True)
        return dict(passed=False)

    monkeypatch.setattr(m, "child", fake)
    with pytest.raises(ValueError, match="installation_failed"):
        m.prepare(tmp_path / "raw", tmp_path, inputs)
    scratch = tmp_path / "second"
    monkeypatch.setattr(
        m,
        "child",
        lambda name, args, raw, **kwargs: dict(fake(name, args, raw, **kwargs), passed=True),
    )
    monkeypatch.setattr(
        m.shutil,
        "copyfile",
        lambda source, destination: Path(destination).write_bytes(b"wrong extension"),
    )
    with pytest.raises(ValueError, match="installed_extension_hash"):
        m.prepare(tmp_path / "raw", scratch, inputs)
    assert original is not None


def test_upstream_terminal_and_workload_rejections(monkeypatch):
    """SCENARIO-VERIFY-8717-CUSTODY: terminal acceptance and frozen operands are required."""
    original = e.load
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *args: dict(report=dict(passed=False)))
    assert not e.inputs(e.ROOT)["passed"]
    monkeypatch.undo()
    for mode in ["seal", "frozen", "bytes"]:

        def changed(ref):
            value = original(ref)
            if mode == "seal" and ref["path"].endswith("/work.json"):
                value["duration_s"] += 1
            if mode == "frozen" and ref["path"].endswith("installed_import_isolation.json"):
                value["complete_cost_protocol"]["frozen_installed_workload"]["method"] = {}
            return value

        monkeypatch.setattr(e, "load", changed)
        if mode == "bytes":
            ref = e.reference
            monkeypatch.setattr(
                e,
                "reference",
                lambda path: (
                    dict(ref(path), sha256="changed") if path.name == "head.json" else ref(path)
                ),
            )
        if mode == "frozen":
            auth = e.previous.authenticate

            def authenticated(*args, **kwargs):
                value = auth(*args, **kwargs)
                if (
                    isinstance(value.get("value"), dict)
                    and value["value"].get("experiment_id") == 8716
                ):
                    value["value"]["complete_cost_protocol"]["frozen_installed_workload"][
                        "method"
                    ] = {}
                return value

            monkeypatch.setattr(e.previous, "authenticate", authenticated)
        assert not e.inputs(e.ROOT)["passed"], mode
        monkeypatch.undo()


def test_runner_blocking_and_sealed_rejections(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8717-CUSTODY: external absence blocks and rehashing cannot repair it."""
    output = tmp_path / (e.NAME + ".json")
    assert r.run(output, root=tmp_path, private=True) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 60
    authority = value["primitive_reference"]
    original = e.load(authority)
    work_path = Path(authority["path"])

    def candidate(work):
        seal = tmp_path / "new-seal.json"
        e.atomic_json(seal, {k: v for k, v in work.items() if k != "seal_reference"})
        work["seal_reference"] = e.reference(seal)
        e.atomic_json(work_path, work)
        new = e.build(work, output)
        e.atomic_json(output, new)
        return new["primitive_reference"]

    changed = deepcopy(original)
    changed["duration_s"] += 1
    e.atomic_json(work_path, changed)
    value["primitive_reference"] = e.reference(work_path)
    e.atomic_json(output, value)
    assert not r.replay(output, value["primitive_reference"])
    for mode in ["input", "source", "measurement", "custody"]:
        work = deepcopy(original)
        if mode == "input":
            work["inputs"]["passed"] = True
        elif mode == "source":
            work["sources"][0]["sha256"] = "changed"
        elif mode == "measurement":
            work["measurement"]["request_rows"] = [dict(forged=True)]
        else:
            work["validation"]["raw_shard_hashes"] = [dict(path="unused", sha256="unused")]
            work["validation"]["custody"] = dict(retained_child_rows=[], node_ids=[], passed=True)
        auth = candidate(work)
        assert not r.replay(output, authority)
        assert not r.replay(output, auth), mode
    work = deepcopy(original)
    work["validation"]["raw_shard_hashes"] = [dict(path="unused", sha256="unused")]
    work["validation"]["custody"] = dict(retained_child_rows=[], node_ids=[], passed=True)
    monkeypatch.setattr(r.shared.retained, "aggregate", lambda *args: work["validation"]["custody"])
    assert r.replay(output, candidate(work))
    monkeypatch.setattr(r.shared, "validate", lambda *args: dict(checked=True))
    assert r.validate(tmp_path, tmp_path, []) == dict(checked=True)
    monkeypatch.setattr(r, "run", lambda *args, **kwargs: 0)
    assert r.main([]) == 0


def test_runner_owned_failures(tmp_path, monkeypatch):
    """REQ-VERIFY-8717: validation failure, source drift and measurement caps disqualify."""
    inputs = e.inputs(tmp_path)
    blank = m.measure(tmp_path / "unused", tmp_path, inputs, 2**63)
    monkeypatch.setattr(r, "publish_primary", lambda output, value, validator: dict(passed=True))
    good = dict(
        receipts=[dict(passed=True)],
        coverage=dict(complete=True),
        custody=dict(passed=True),
        raw_shard_hashes=[],
    )
    monkeypatch.setattr(r, "validate", lambda *args: good)
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls % 2 == 0:
            raise ValueError("prepare_failed")
        return deepcopy(blank)

    monkeypatch.setattr(m, "measure", fail)
    assert r.run(tmp_path / "a" / (e.NAME + ".json"), root=tmp_path) == 0

    def drift(*args):
        original = e.reference
        monkeypatch.setattr(
            e,
            "reference",
            lambda path: (
                dict(original(path), sha256="drift")
                if str(path).endswith(e.OWNED[0])
                else original(path)
            ),
        )
        return good

    monkeypatch.setattr(r, "validate", drift)
    monkeypatch.setattr(m, "measure", lambda *args, **kwargs: deepcopy(blank))
    assert r.run(tmp_path / "b" / (e.NAME + ".json"), root=tmp_path) == 0
    monkeypatch.undo()
    monkeypatch.setattr(r, "publish_primary", lambda output, value, validator: dict(passed=True))
    monkeypatch.setattr(r, "validate", lambda *args: good)
    pre = r.shared.shared.preconditions
    monkeypatch.setattr(
        r.shared.shared, "preconditions", lambda scratch: dict(pre(scratch), disk_backed=False)
    )
    monkeypatch.setattr(m, "measure", lambda *args, **kwargs: deepcopy(blank))
    clock = iter([0, 3601 * 10**9, 3602 * 10**9])
    monkeypatch.setattr(r.time, "monotonic_ns", lambda: next(clock, 3602 * 10**9))
    assert r.run(tmp_path / "c" / (e.NAME + ".json"), root=tmp_path) == 0


def test_scoped_coverage_and_heartbeat(tmp_path, monkeypatch):
    """REQ-VERIFY-8717: instrumentation excludes unrelated code and heartbeat reports waits."""
    config = tmp_path / "child.ini"
    config.write_text("[run]\nparallel=true\n")
    monkeypatch.setattr(r, "BASE_CHILD", lambda *args, **kwargs: dict(passed=True))
    assert r.scoped_child("control", ["python", "--rcfile=" + str(config)], tmp_path)["passed"]
    assert r.COVERAGE_INCLUDE in config.read_text()

    class Stop:
        calls = 0

        def wait(self, seconds):
            self.calls += 1
            return self.calls > 1

    r.heartbeat(Stop())


def test_gate_states_and_failed_process_evidence(tmp_path, monkeypatch, primitive):
    """SCENARIO-REPORT-8717-COST: absent fields, zero and flags have distinct gate rows."""
    inputs, config, slot, command, value = primitive
    original = e.previous.load(dict(path=str(e.ROOT / e.PRODUCER), sha256=e.PIN))
    producer = tmp_path / e.PRODUCER
    producer.parent.mkdir(parents=True)
    monkeypatch.setattr(e.previous, "inputs", lambda root: deepcopy(inputs))
    for mode, state in [
        ("missing", "missing_field"),
        ("zero", "observed_zero"),
        ("flagged", "observed"),
    ]:
        changed = deepcopy(original)
        if mode == "missing":
            del changed["batch_service_ready_score"]
        elif mode == "zero":
            changed["batch_service_ready_score"] = 0
        else:
            changed["flagged_adversarial"] = True
        e.atomic_json(producer, changed)
        monkeypatch.setattr(e, "PIN", e.reference(producer)["sha256"])
        checked = e.inputs(tmp_path)
        assert not checked["passed"]
        assert any(row["observed_state"] == state for row in checked["gate_check_summary"])
    monkeypatch.setattr(m, "prepare", lambda *args: (command, config, []))
    malformed = dict(inputs, requests=[dict(unit_id="invalid")])
    measured = m.measure(tmp_path / "failed-worker", tmp_path, malformed, 2**63)
    assert measured["process_rows"][0]["status"] == "failed"
    assert any(ref["path"].endswith("failed_child.json") for ref in measured["references"])
