"""REQ-VERIFY-8376 / REQ-REPORT-8376: direct serving preserves causal state."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import direct_atomic_state_8376 as s


@pytest.fixture
def trace():
    """SCENARIO-VERIFY-8376-CRASH: seal requests before observing recovery."""
    return s.trace(11, json.loads((s.ROOT / s.PROTOCOL).read_bytes())["head"])


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """SCENARIO-VERIFY-8376-CRASH: all controls reuse one sealed three-seed panel."""
    from carnot.reporting import direct_atomic_state_8376 as e

    directory = tmp_path_factory.mktemp("atomic-evidence")
    return e.measure(s.ROOT, directory / "raw", directory)


def test_frozen_optimizer_and_immutable_issue(trace):
    """REQ-VERIFY-8376: retries retain results and updates change future decisions."""
    state = s.initial(trace["head"])
    for event in trace["events"]:
        before = deepcopy(state["issued"])
        state, result = s.transition(state, event)
        assert all(state["issued"][key] == row for key, row in before.items())
        again, retry = s.transition(state, event)
        assert again == state and retry == result
    assert len(state["applied"]) == 16 and not state["pending"]
    assert state["release_cursor"] == 16
    assert state["head"]["coefficients"][:2] == trace["head"]["coefficients"][:2]
    assert state["head"]["coefficients"][2:] != trace["head"]["coefficients"][2:]
    assert s.fold(trace) == state


def test_crash_trace_contains_prior_acknowledged_updates(trace):
    """SCENARIO-VERIFY-8376-CRASH: recovery must retain earlier acknowledged learning."""
    assert s.fold(trace, trace["crash_event_index"])["applied"]


def test_reject_bad_events_and_heads(trace):
    """SCENARIO-VERIFY-8376-CONTROLS: malformed and stale input never commits."""
    state = s.fold(trace, 10)
    feedback = next(e for e in trace["events"] if e["kind"] == "feedback")
    for event in [
        {},
        dict(kind="other"),
        dict(trace["events"][0], x=[float("nan")] * 5),
        dict(trace["events"][0], x=[0] * 4),
        dict(trace["events"][0], id=""),
        dict(feedback, issued_version=-1),
        dict(feedback, clock=-1),
        dict(feedback, y=2),
        dict(feedback, issue_id="absent"),
    ]:
        with pytest.raises((ValueError, KeyError)):
            s.transition(state, event)
    final = s.fold(trace)
    with pytest.raises(ValueError, match="conflict"):
        s.transition(final, dict(feedback, y=1 - feedback["y"]))
    with pytest.raises(ValueError):
        s.transition(final, dict(feedback, id="new-feedback"))
    for head in [dict(trace["head"], table=[]), dict(trace["head"], temperature=0)]:
        with pytest.raises(ValueError):
            s.initial(head)


def test_missing_drift_and_interrupted_cleanup(trace, tmp_path):
    """REQ-VERIFY-8376: missing state and repaired checksums cannot create learning."""
    store = s.Store(tmp_path / "state", trace)
    with pytest.raises(FileNotFoundError):
        store.read()
    store.initialize()
    first = store.read()
    for event in trace["events"][:10]:
        store.apply(event)
    pinned = store.read()
    assert first != pinned
    (store.path / ".abandoned.tmp").write_bytes(b"incomplete")
    with pytest.raises(RuntimeError):
        store.cleanup(interrupt=True)
    assert store.read() == pinned
    assert store.cleanup() >= 0
    pointer = json.loads((store.path / "current.json").read_bytes())
    snapshot = store.path / pointer["file"]
    envelope = json.loads(snapshot.read_bytes())
    envelope["state"]["release_cursor"] += 1
    atomic_json(snapshot, envelope)
    with pytest.raises(ValueError):
        store.read()
    envelope["sha256"] = canonical_hash(envelope["state"])
    atomic_json(snapshot, envelope)
    pointer["sha256"] = envelope["sha256"]
    atomic_json(store.path / "current.json", pointer)
    with pytest.raises(ValueError, match="semantics"):
        store.read()


def test_real_kills_and_two_readers(work):
    """SCENARIO-VERIFY-8376-CRASH: real SIGKILL evidence includes exact recovery."""
    rows = [row for row in work["rows"] if row["seed"] == 11]
    assert len(rows) == 9
    assert all(row["passed"] for row in rows)
    assert {row["arm"] for row in rows} == {"uninterrupted", "restart", "two_readers"}
    assert all(row["actual_exit"] == -9 for row in rows if row["barrier"] != "none")
    assert all(row["recovery_mismatch_count"] == 0 for row in rows)
    assert all(row["mixed_version_read_count"] == 0 for row in rows)
    for row in rows:
        if row["kill_reference"]:
            marker = json.loads(Path(row["kill_reference"]["path"]).read_bytes())
            assert marker["pre_state"]["applied"]
            assert marker["acknowledged"]


def test_build_replay_and_fresh_cli(work, tmp_path):
    """SCENARIO-REPORT-8376-TERMINAL: rehashed aggregates fail cold recomputation."""
    from carnot.reporting import direct_atomic_state_8376 as e
    from carnot.reporting.v709_execution import child

    value = e.build(work, [dict(passed=True, scope="owned")])
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    assert value["MODEL_SPECS"] == [] and value["independent_count"] == 0
    assert e.build(work, [dict(passed=False, scope="owned")])["verdict_class"] == "disqualified"
    bad = deepcopy(value)
    bad["rows"][0]["recovery_mismatch_count"] = 1
    bad.pop("reproducibility_checksum")
    bad["reproducibility_checksum"] = canonical_hash(bad)
    atomic_json(path, bad)
    assert not e.replay(path)
    assert not e.replay(tmp_path / "missing")
    atomic_json(path, value)
    for name, args, expected in [
        ("valid", ["--cold-replay", str(path)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("error", ["--deliberate-error"], 1),
        ("date", ["--date", "20261009"], 2),
    ]:
        receipt = child(
            name,
            [sys.executable, "-u", str(s.ROOT / e.CLI)] + args,
            tmp_path / "logs",
            expected=expected,
            deadline=60,
        )
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()


def test_service_failure_boundaries(trace, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8376-CONTROLS: budgets and invalid input leave state intact."""
    store = s.Store(tmp_path, trace)
    store.initialize()
    for head in [dict(trace["head"], degree=2), dict(trace["head"], knots=[])]:
        with pytest.raises(ValueError, match="basis"):
            s.initial(head)
    state = store.read()
    with pytest.raises(ValueError, match="order"):
        s.transition(state, dict(trace["events"][0], slot=2))
    with pytest.raises(ValueError, match="trace_order"):
        store.apply(dict(trace["events"][0], x=[1, 0, 0, 0, 0]))
    monkeypatch.setattr(s, "MAX_EVENTS", 0)
    with pytest.raises(ValueError, match="budget"):
        store.apply(trace["events"][0])
    monkeypatch.setattr(s, "MAX_EVENTS", 128)
    monkeypatch.setattr(s, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="budget"):
        store.apply(trace["events"][0])
    with pytest.raises(ValueError, match="budget"):
        store.commit(s.initial(trace["head"]))
    atomic_json(tmp_path / "current.json", dict(file="../outside", sha256="wrong"))
    with pytest.raises(ValueError, match="pointer_path"):
        store.read()
    state = s.fold(trace, 9)
    feedback = trace["events"][9]
    with pytest.raises(ValueError, match="stale"):
        s.transition(state, dict(feedback, clock=0))
    state = s.initial(trace["head"])
    with pytest.raises(ValueError, match="features"):
        s.transition(state, dict(trace["events"][0], x=[0] * 4))


def test_explicit_missing_features_and_missing_durable_pointer(trace, tmp_path):
    """REQ-VERIFY-8376: explicit absence escalates; missing durable state never resets learning."""
    missing = deepcopy(trace)
    missing["events"][0]["x"] = None
    missing["events"][9]["y"] = None
    state = s.fold(missing)
    assert state["issued"]["issue-1"]["probability"] is None
    assert state["issued"]["issue-1"]["action"] == "escalate"
    store = s.Store(tmp_path, trace)
    store.initialize()
    (tmp_path / "current.json").unlink()
    with pytest.raises(FileNotFoundError):
        store.initialize()


def test_private_runner_publication_and_failure(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8376-TERMINAL: real validators check the private main route."""
    from carnot.reporting import direct_atomic_runner_8376 as runner

    frozen = runner.plan(tmp_path)
    assert frozen[-1]["scope"] == "global"
    assert any(command["name"] == "consumer_E2E018_020" for command in frozen)
    monkeypatch.setattr(
        runner,
        "plan",
        lambda private: [
            dict(
                name="actual_owned_child",
                argv=[sys.executable, "-c", "print('owned child completed')"],
                deadline_s=30,
                scope="owned",
            )
        ],
    )

    def private_work(root, raw, private):
        copied = deepcopy(work)
        copied["work_path"] = str(raw / "work.json")
        copied["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
        atomic_json(Path(copied["work_path"]), copied)
        (private / "coverage.ini").write_text("actual private command plan")
        return copied

    monkeypatch.setattr(runner.e, "measure", private_work)
    output = tmp_path / "results" / (runner.e.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"

    def broken(*args):
        raise ValueError("actual failure control")

    monkeypatch.setattr(runner.e, "measure", broken)
    assert runner.main(["--output", str(output)]) == 1
    with pytest.raises(SystemExit):
        runner.main(["--worker", "missing-state"])


def test_external_absence_and_changed_task(tmp_path, monkeypatch):
    """REQ-REPORT-8376: missing external bytes are blocked and never invented zeros."""
    from carnot.reporting import direct_atomic_state_8376 as e

    missing = e.measure(tmp_path / "absent", tmp_path / "raw", tmp_path)
    value = e.build(missing, [dict(passed=True)])
    assert value["verdict_class"] == "blocked"
    assert value["direct_state_ready_score"] == 0
    assert any(gate["observed"] is None for gate in value["gate_check_summary"])
    atomic_json(tmp_path / e.authority.ACTIVE, dict(tasks=[dict(id=e.TASK, MODEL_SPECS=["bad"])]))
    changed = e.measure(tmp_path, tmp_path / "changed", tmp_path)
    assert any(gate["check"] == "exact_task" for gate in changed["gate_check_summary"])
    path = tmp_path / "blocked.json"
    atomic_json(path, value)
    assert e.replay(path)


def test_repaired_hash_semantic_controls(work, tmp_path):
    """SCENARIO-REPORT-8376-TERMINAL: repaired bytes cannot forge primitive meaning."""
    from carnot.reporting import direct_atomic_state_8376 as e

    for index, mode in enumerate(
        (
            "state_hash",
            "final_state",
            "kill",
            "exit",
            "row_hash",
            "trace",
            "receipt",
            "primitive",
            "source_hash",
            "protocol",
            "unstarted",
            "acknowledgment",
            "reader",
            "latency",
            "arm",
            "marker_response",
            "marker_ack",
            "marker_reader",
        )
    ):
        directory = tmp_path / str(index)
        directory.mkdir()
        changed = deepcopy(work)
        row = changed["rows"][1]
        if mode == "state_hash":
            row["state_hash"] = "sha256:changed"
        elif mode in ("final_state", "kill", "trace"):
            key = {
                "final_state": "final_state_reference",
                "kill": "kill_reference",
                "trace": "trace_reference",
            }[mode]
            record = json.loads(Path(row[key]["path"]).read_bytes())
            if mode == "final_state":
                record["state"]["release_cursor"] += 1
            elif mode == "kill":
                record["visible_state"]["release_cursor"] += 1
            else:
                record["events"][0]["x"][0] += 0.1
            operand = directory / "changed-operand.json"
            atomic_json(operand, record)
            row[key] = e.reference(operand)
        elif mode == "exit":
            row["actual_exit"] = 0
        elif mode == "row_hash":
            row["kill_reference"]["sha256"] = "sha256:wrong"
        elif mode == "receipt":
            row["receipts"][0]["stderr_sha256"] = "sha256:wrong"
        elif mode == "source_hash":
            changed["source_artifact_hashes"][0]["sha256"] = "sha256:wrong"
        elif mode == "protocol":
            ref = next(
                r
                for r in changed["source_artifact_hashes"]
                if r.get("source_path", "").endswith(s.PROTOCOL)
            )
            operand = directory / "changed-protocol.json"
            operand.write_bytes(Path(ref["path"]).read_bytes() + b"\n")
            ref.update(e.reference(operand))
        elif mode == "unstarted":
            row["status"] = "unstarted"
        elif mode in ("acknowledgment", "reader"):
            record = json.loads(Path(row["final_state_reference"]["path"]).read_bytes())
            if mode == "acknowledgment":
                record["acknowledged"][0]["result"]["version"] += 1
            else:
                record["reader_rows"].append(dict(version=0, state_hash="sha256:wrong", mixed=0))
            operand = directory / "changed-response.json"
            atomic_json(operand, record)
            row["final_state_reference"] = e.reference(operand)
        elif mode == "latency":
            row["latency_s"] += 1
        elif mode == "arm":
            row["arm"] = "invented"
        elif mode.startswith("marker_"):
            record = json.loads(Path(row["kill_reference"]["path"]).read_bytes())
            if mode == "marker_response":
                record["result"]["version"] += 1
            elif mode == "marker_ack":
                record["acknowledged"][0]["result"]["version"] += 1
            else:
                record["reader_rows"].append(dict(version=0, state_hash="sha256:wrong", mixed=0))
            operand = directory / "changed-marker.json"
            atomic_json(operand, record)
            row["kill_reference"] = e.reference(operand)
        primitive = dict(rows=deepcopy(changed["rows"]), traces=changed["traces"])
        if mode == "primitive":
            primitive["rows"][0]["state_hash"] = "sha256:wrong"
        primitive_path = directory / "primitive.json"
        atomic_json(primitive_path, primitive)
        changed["primitive_reference"] = e.reference(primitive_path)
        changed["work_path"] = str(directory / "work.json")
        atomic_json(Path(changed["work_path"]), changed)
        candidate = directory / "candidate.json"
        atomic_json(candidate, e.build(changed, [dict(passed=True)]))
        assert not e.replay(candidate), mode
    valid = e.build(work, [dict(passed=True)])
    valid["reproducibility_checksum"] = "wrong"
    atomic_json(candidate, valid)
    assert not e.replay(candidate)


def test_worker_failure_controls(trace, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8376-CONTROLS: actual reader and retry faults stop acknowledgment."""
    from carnot.reporting import direct_atomic_state_8376 as e

    frozen = tmp_path / "trace.json"
    atomic_json(frozen, trace)
    path = tmp_path / "state"
    s.Store(path, trace).initialize()
    original = s.Store.apply
    calls = 0

    def wrong_retry(self, event, hook=lambda phase: None):
        nonlocal calls
        result = original(self, event, hook)
        calls += 1
        return dict(result, action="wrong") if calls == 2 else result

    monkeypatch.setattr(s.Store, "apply", wrong_retry)
    with pytest.raises(ValueError, match="retry_result"):
        e.worker(frozen, path, "none", False)
    original_read = s.Store.read
    calls = 0

    def duplicate(self, event, hook=lambda phase: None):
        nonlocal calls
        result = original(self, event, hook)
        calls += 1
        return result

    def drifted_read(self):
        result = original_read(self)
        if calls == 2:
            result["version"] += 1
        return result

    monkeypatch.setattr(s.Store, "apply", duplicate)
    monkeypatch.setattr(s.Store, "read", drifted_read)
    with pytest.raises(ValueError, match="duplicate_update"):
        e.worker(frozen, path, "none", False)
    monkeypatch.setattr(s.Store, "read", original_read)
    monkeypatch.setattr(s.Store, "apply", original)

    class NeverReady:
        def set(self):
            pass

        def wait(self, timeout):
            return False

    monkeypatch.setattr(e, "Event", NeverReady)
    with pytest.raises(TimeoutError, match="reader_deadline"):
        e.worker(frozen, path, "none", True)


def test_private_qualified_input_rejections(tmp_path, monkeypatch):
    """REQ-REPORT-8376: producer drift and missing inputs remain named external gates."""
    from carnot.reporting import direct_atomic_state_8376 as e

    protocol_path = tmp_path / s.PROTOCOL
    protocol = json.loads((s.ROOT / s.PROTOCOL).read_bytes())
    methods = json.loads((s.ROOT / e.authority.METHODS).read_bytes())
    atomic_json(protocol_path, protocol)
    methods_path = tmp_path / e.authority.METHODS
    atomic_json(methods_path, methods)
    monkeypatch.setattr(e.authority, "PROTOCOL_PIN", e.reference(protocol_path)["sha256"])
    monkeypatch.setattr(e.authority, "METHODS_PIN", e.reference(methods_path)["sha256"])
    missing = deepcopy(protocol)
    missing["checkpoint"]["path"] = str(tmp_path / "missing-head.json")
    atomic_json(protocol_path, missing)
    monkeypatch.setattr(e.authority, "PROTOCOL_PIN", e.reference(protocol_path)["sha256"])
    assert not e.measure(tmp_path, tmp_path / "missing", tmp_path)["input_ready"]
    atomic_json(protocol_path, protocol)
    monkeypatch.setattr(e.authority, "PROTOCOL_PIN", e.reference(protocol_path)["sha256"])
    bad = tmp_path / "bad-producer.json"
    atomic_json(bad, dict(local_kernel_ready_score=0))
    methods["reusable"]["kernel"].update(e.reference(bad))
    atomic_json(methods_path, methods)
    monkeypatch.setattr(e.authority, "METHODS_PIN", e.reference(methods_path)["sha256"])
    assert not e.measure(tmp_path, tmp_path / "zero", tmp_path)["input_ready"]
    monkeypatch.setattr(e.authority, "METHODS_PIN", "sha256:wrong")
    assert not e.measure(tmp_path, tmp_path / "methods", tmp_path)["input_ready"]
    checkpoint = json.loads(Path(protocol["checkpoint"]["path"]).read_bytes())
    checkpoint["heads"][0]["coefficients"][0] += 0.125
    head_path = tmp_path / "changed-head.json"
    atomic_json(head_path, checkpoint)
    protocol["checkpoint"] = e.reference(head_path)
    atomic_json(protocol_path, protocol)
    monkeypatch.setattr(e.authority, "PROTOCOL_PIN", e.reference(protocol_path)["sha256"])
    with pytest.raises(ValueError, match="frozen_head_drift"):
        e.measure(tmp_path, tmp_path / "head", tmp_path)


def test_owned_memory_precondition_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8376: a failed owned resource gate disqualifies even with external absence."""
    from carnot.reporting import direct_atomic_state_8376 as e

    original = Path.read_text

    def constrained(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return "MemAvailable: 1 kB\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", constrained)
    work = e.measure(tmp_path, tmp_path / "raw", tmp_path)
    assert not work["resource_preconditions_passed"]
    assert e.build(work, [dict(passed=True)])["verdict_class"] == "disqualified"
