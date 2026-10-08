"""REQ-VERIFY-8263 / REQ-REPORT-8263: qualify current protocol operands."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import focal_protocol_8263 as f
from carnot.verify import typed_admission_8263 as a
from carnot.verify import protocol_conformance_8263 as e


def public(source="Café is open. Café is open. It is not closed.", answer=None):
    answer = answer or " ".join(["Other sentence."] * 9 + ["Café is open, but only today."])
    return dict(source_bytes=source.encode().hex(), answer_bytes=answer.encode().hex())


def test_focal_request_and_token_selector():
    """SCENARIO-VERIFY-8263-REQUEST: selection ignores labels and keeps full bytes."""
    row = public()
    # A deliberately declared private tokenizer double tests adapter routing only.
    count = lambda text: 2 if text.strip() == "Café is open." else 3
    cached = [dict(sentence_index=9, p_unsupported=0.8, relation="E")]
    plan = f.views(row, cached, count)
    assert plan["selected_sentence_index"] == 0
    assert plan["control_sentence_index"] == 1
    assert plan == f.views(dict(row, human_label=1), cached, count)
    for view in plan["views"].values():
        request = view["request"]
        assert request["sentence_indices"] == [9] and request["max_tokens"] == 64
        assert json.loads(request["prompt"])["answer"].encode().hex() == row["answer_bytes"]
        assert request["seed"] == 7138250 and request["temperature"] == 0
        assert f.parse(view, "9|E|0.05|[0]", canonical_hash(request))["status"] == "completed"
        for bad in ["0|E|0.05|[0]", "9|E|0.05|[99]", "9|E|0.05|[0,0]", "", "9|B|0.50|[]\n"]:
            assert f.parse(view, bad, canonical_hash(request))["status"] == "escalated"
        with pytest.raises(ValueError, match="request_hash"):
            f.parse(view, "9|B|0.50|[]", "bad")
    assert f.views(public("One."), cached, count)["status"] == "unavailable"
    assert f.views(row, [], count)["status"] == "unavailable"
    empty = f.construct_view(row, [0, 1, 2])
    req = f.request(empty, 9, count)
    assert (
        f.parse(dict(empty, request=req), "9|B|0.50|[]", canonical_hash(req))["status"]
        == "completed"
    )
    for kwargs in [dict(context_tokens=1), dict(output_tokens=1), dict(output_tokens=65)]:
        with pytest.raises(ValueError, match="budget"):
            f.request(empty, 9, count, **kwargs)
    with pytest.raises(ValueError, match="focal"):
        f.request(empty, 90, count)
    with pytest.raises(ValueError, match="sentence_address"):
        f.construct_view(row, [99])


def test_typed_controls_and_rejections(tmp_path):
    """SCENARIO-VERIFY-8263-CAUSAL: causal typed actions and source-hash controls."""
    assert a.action(0.1, True) == a.action(0.5, True) == "escalate"
    assert a.action(0, False) == "escalate" and a.action(0, True) == "accept"
    assert a.action(1, False) == "reject"
    assert [a.cost(x, 1) for x in ["accept", "reject", "escalate"]] == [5, 0, 0.5]
    assert a.group(None) is None
    assert a.group(dict(relation="E", selected_delta=0.1, control_delta=-0.1)) == "100"
    assert a.group(dict(relation="B", selected_delta=0.2, control_delta=-0.2, y=1)) == "011"
    with pytest.raises(ValueError, match="features"):
        a.group(dict(relation="", selected_delta=0, control_delta=0))
    with pytest.raises(ValueError, match="probability"):
        a.action(float("nan"), True)
    roster = a.roster("natural")
    path = tmp_path / "journal"
    state = a.run(roster, path, stop=41)
    assert len(state["issued"]) == 41 and 40 in state["pending"]
    assert a.load(path) == state
    assert a.run(roster, path) == a.run(roster, tmp_path / "fresh")
    checkpoint = json.loads(path.with_suffix(".checkpoint.json").read_bytes())
    assert checkpoint["state_sha256"] == canonical_hash(a.load(path))
    assert checkpoint["pending"] == a.load(path)["pending"]
    before = deepcopy(state)
    for event, reason in [
        (dict(kind="release", record=dict(origin=0, now=8, label=0)), "duplicate"),
        (dict(kind="release", record=dict(origin=99, now=107, label=0)), "unissued"),
        (dict(kind="release", record=dict(origin=40, now=41, label=0)), "future"),
        (dict(kind="other", record={}), "event"),
    ]:
        with pytest.raises(ValueError, match=reason):
            a.transition(deepcopy(before), event)
    path.write_text("{")
    with pytest.raises(ValueError, match="partial"):
        a.load(path)
    positive = a.controls(tmp_path / "controls")
    assert positive["positive_control_passed"]
    assert all(
        r["actual_cost"] == a.cost(r["action"], r["label"])
        for r in positive["learnable_control_rows"]
    )
    assert positive["zero_admission_count"] == 0


def cli(tmp_path, *args):
    """SCENARIO-REPORT-8263-CLI: actual child entrypoint also supplies CLI coverage."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    print("before private8263 subprocess", flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    print("after private8263 subprocess", result.returncode, flush=True)
    return result


def test_hard_exit_pending_resume(tmp_path):
    """SCENARIO-VERIFY-8263-CAUSAL: genuine exits40/72 retain issue-before-release."""
    roster = tmp_path / "roster.json"
    rows = a.roster("natural")
    atomic_json(roster, rows)
    baseline = a.run(rows, tmp_path / "baseline")
    for slot in [40, 72]:
        path = tmp_path / str(slot)
        crash = cli(
            tmp_path, "--typed-roster", roster, "--typed-journal", path, "--crash-slot", slot
        )
        assert crash.returncode == 73
        saved = json.loads(path.with_suffix(".pending.json").read_bytes())
        assert saved["issued"][-1]["slot"] == slot and slot - 8 in saved["pending"]
        resumed = cli(tmp_path, "--typed-roster", roster, "--typed-journal", path)
        assert resumed.returncode == 0
        assert json.loads(path.with_suffix(".final.json").read_bytes()) == baseline
    altered = deepcopy(rows)
    altered[0]["role"] = "retention"
    with pytest.raises(ValueError, match="resume_roster"):
        a.run(altered, tmp_path / "baseline")
    altered = deepcopy(rows)
    altered[-1]["label"] ^= 1
    with pytest.raises(ValueError, match="resume_roster"):
        a.run(altered, tmp_path / "baseline")


def test_capture_real_peer(tmp_path):
    """SCENARIO-VERIFY-8263-CAPTURE: real wire requests, failure slots and resume."""
    from carnot.verify import focal_capture_8263 as c

    row = public()
    view = f.views(row, [dict(sentence_index=9, p_unsupported=0.8)], lambda _: 1)["views"][
        "original"
    ]
    slots = [
        dict(unit_id=str(i), role="fit", mode=mode, view=view)
        for i, mode in enumerate(["valid"] * 8 + ["partial", "duplicate", "role", "drop"])
    ]
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), "--scripted-peer"]
    path = tmp_path / "capture"
    result = c.capture(slots, path, c.PipePeer(argv))
    assert result["intended_count"] == 12 and result["completed_count"] == 8
    assert len(c.journal(path)) == 24 and path.with_suffix(".checkpoint.json").exists()
    assert c.capture(slots, path, c.PipePeer(argv))["rows"] == result["rows"]
    altered = deepcopy(slots)
    altered[0]["role"] = "evaluation"
    with pytest.raises(ValueError, match="resume_key"):
        c.capture(altered, path, c.PipePeer(argv))
    timeout = c.capture(
        [slots[0], dict(slots[0], mode="timeout")],
        tmp_path / "timeout",
        c.PipePeer(argv, deadline=0.1),
    )
    assert timeout["failed_count"] == 1 and "deadline" in timeout["rows"][1]["error"]
    events = c.journal(path)
    c.append(path, events[-1])
    with pytest.raises(ValueError, match="duplicate_reply"):
        c.capture(slots, path, c.PipePeer(argv))
    partial = tmp_path / "partial"
    c.append(
        partial, dict(kind="issue", slot=0, plan_hash=canonical_hash(slots), resume_key="wrong")
    )
    with pytest.raises(ValueError, match="role_drift"):
        c.capture(slots, partial, c.PipePeer(argv))


def test_private_cli_replay_and_blocking(tmp_path):
    """SCENARIO-REPORT-8263-CLI: private publication and rehashed headlines."""
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert (
        value["view_kernel_ready_score"] == 0
        and value["model_invocation_counts"] == e.ZERO_INVOCATION_COUNTS
    )
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value["view_kernel_ready_score"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    tamper = tmp_path / "tamper.json"
    atomic_json(tamper, value)
    assert cli(tmp_path, "--cold-replay", tamper).returncode == 1
    assert cli(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1
    assert cli(tmp_path, "--date", "20261007").returncode == 2
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results" / "private.json").returncode == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    work = e.measure(empty, tmp_path / "blocked")
    assert any(not r["passed"] for r in work["checks"])
    built = e.build(work, tmp_path / "blocked", [dict(name="private", passed=True)], fixture=True)
    assert built["verdict_class"] == "blocked" and built["view_kernel_ready_score"] == 0
    assert (
        e.build(work, tmp_path / "blocked", [dict(name="private", passed=False)], fixture=True)[
            "verdict_class"
        ]
        == "disqualified"
    )


def test_focal_actual_tokenizer_and_external_gates(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8263-REQUEST: actual embedded Unicode rankings differ."""
    work = e.measure(e.ROOT, tmp_path / "actual")
    assert all(r["passed"] for r in work["checks"])
    assert work["tokenizer"]["neural_weights_loaded"] is False
    plan = work["evidence"]["request_control_rows"][0]["plan"]
    assert plan["control_sentence_index"] == 2
    assert plan["diagnostics"][1]["embedded_tokens"] == 4
    assert plan["diagnostics"][2]["embedded_tokens"] == 5
    built = e.build(
        work,
        tmp_path / "actual",
        [dict(name=x + "_component", passed=True) for x in ["view", "admission"]],
        fixture=True,
    )
    assert built["view_kernel_ready_score"] == built["admission_kernel_ready_score"] == 1
    path = tmp_path / "actual.json"
    atomic_json(path, built)
    assert e.replay(path)
    with monkeypatch.context() as m:
        m.setattr(e, "authenticate", lambda *args: (_ for _ in ()).throw(ValueError("bad schema")))
        assert not e.measure(tmp_path, tmp_path / "bad")["checks"][-1]["passed"]
    checks = dict(checks=[])
    monkeypatch.setattr(e, "TOKENIZER_PATH", tmp_path / "missing.gguf")
    assert e.tokenizer(checks) == (None, {})
    monkeypatch.setattr(e, "sha256_file", lambda _: e.TOKENIZER_PIN)
    (tmp_path / "missing.gguf").write_bytes(b"private")
    import llama_cpp

    monkeypatch.setattr(
        llama_cpp, "Llama", lambda **kwargs: (_ for _ in ()).throw(ValueError("vocabulary missing"))
    )
    assert e.tokenizer(checks) == (None, {})


def test_typed_transition_negatives(tmp_path):
    """SCENARIO-VERIFY-8263-CAUSAL: invalid or missing rows cannot update groups."""
    rows = a.roster("natural")
    state = a.run(rows, tmp_path / "state", stop=17)
    record = deepcopy(state["issued"][-1])
    for change, reason in [
        (dict(slot=0), "order"),
        (dict(slot=17, role="wrong"), "order"),
        (dict(slot=17, predictions={}), "drift"),
    ]:
        with pytest.raises(ValueError, match=reason):
            a.transition(deepcopy(state), dict(kind="issue", record=dict(record, **change)))
    for label in [2, True]:
        if len(state["issued"]) == 17:
            row = {k: v for k, v in rows[17].items() if k != "label"}
            a.transition(state, dict(kind="issue", record=a.predict(state, row)))
        with pytest.raises(ValueError, match="label"):
            a.transition(
                deepcopy(state), dict(kind="release", record=dict(origin=9, now=17, label=label))
            )
    retention = a.initial()
    for t in range(9):
        row = dict(rows[t], role="retention")
        row.pop("label")
        a.transition(retention, dict(kind="issue", record=a.predict(retention, row)))
    with pytest.raises(ValueError, match="retention"):
        a.transition(retention, dict(kind="release", record=dict(origin=0, now=8, label=1)))
    # A missing row releases globally but leaves every group counter unchanged.
    rows = a.roster("zero")
    zero = a.run(rows, tmp_path / "zero")
    assert zero["global_counts"]["n"] == 96
    assert all(c["n"] == 0 for groups in zero["groups"].values() for c in groups.values())


def test_report_replay_negative_receipts(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8263-CLI: hash and rehashed primitive tampering fail."""
    work = e.measure(tmp_path, tmp_path / "raw", fixture=True)
    receipts = [dict(name="private", passed=True)]
    value = e.build(work, tmp_path / "raw", receipts, fixture=True)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert e.replay(path)
    bad = deepcopy(value)
    bad["reproducibility_checksum"] = "bad"
    atomic_json(path, bad)
    assert not e.replay(path)
    for field in ["code_config_hashes", "raw_shard_hashes"]:
        bad = deepcopy(value)
        bad[field][0]["sha256"] = "bad"
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        assert not e.replay(path)
    log = tmp_path / "log"
    log.write_text("actual")
    value["validation_receipts"][0].update(stdout_path=str(log), stdout_sha256="bad")
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    assert not e.replay(path)
    spec = dict(
        name="coverage_json",
        argv=[str(e.ROOT / ".venv/bin/python"), "-c", "print('actual child')"],
        deadline_s=10,
    )
    assert not e.run_check(e.ROOT, spec, tmp_path, tmp_path / "logs")["passed"]


def test_capture_peer_lifecycle_failures(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8263-CAPTURE: dropped and stubborn children are cleaned."""
    import io
    from unittest.mock import Mock
    import signal
    import sys
    from carnot.verify import focal_capture_8263 as c

    for mode, exit_code in [
        ("valid", 0),
        ("partial", 0),
        ("duplicate", 0),
        ("role", 0),
        ("drop", 17),
        ("timeout", 0),
    ]:
        wire = dict(mode=mode, request=dict(sentence_indices=[9]), resume_key="k", role="fit")
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(wire) + "\n"))
        monkeypatch.setattr(os, "read", lambda *args: b"x")
        assert c.peer() == exit_code
    peer = c.PipePeer([])
    peer.child = Mock()
    peer.child.poll.return_value = None
    peer.child.pid = 123
    peer.child.wait.side_effect = [subprocess.TimeoutExpired("private", 2), 0]
    killed = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    peer.__exit__()
    assert killed == [(123, signal.SIGTERM), (123, signal.SIGKILL)]


def test_report_rehashed_primitive_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8263-CLI: valid hashes cannot hide changed primitive science."""
    raw = tmp_path / "raw"
    work = e.measure(tmp_path, raw, fixture=True)
    receipts = [dict(name="private", passed=True)]
    path = tmp_path / "candidate.json"
    original = deepcopy(work)
    pending_original = (raw / "pending_40.json").read_bytes()
    cases = [
        "primitive",
        "roster",
        "request",
        "issues",
        "replies",
        "capture_key",
        "capture_parse",
        "pending_hash",
        "pending_state",
        "metadata",
    ]
    for case in cases:
        (raw / "pending_40.json").write_bytes(pending_original)
        current = deepcopy(original)
        primitive = deepcopy(current["evidence"])
        if case == "primitive":
            primitive["positive_control_passed"] = False
        elif case == "roster":
            primitive["rosters"]["natural"][0]["label"] ^= 1
        elif case == "request":
            primitive["request_control_rows"][0]["plan"]["control_sentence_index"] = 99
        elif case == "issues":
            primitive["capture_events"] = primitive["capture_events"][1:]
        elif case == "replies":
            primitive["capture"]["completed_count"] = 99
        elif case == "capture_key":
            primitive["capture_events"][1]["row"]["resume_key"] = "changed"
        elif case == "capture_parse":
            primitive["capture_events"][1]["row"]["transcript"] = "9|B|0.51|[]"
        elif case == "pending_hash":
            primitive["pending_state_hashes"][0] = "changed"
        elif case == "pending_state":
            pending = raw / "pending_40.json"
            saved = json.loads(pending.read_bytes())
            saved["pending"] = []
            atomic_json(pending, saved)
            primitive["pending_state_hashes"][0] = e.sha256_file(pending)
        else:
            current["fixture"] = False
            current["tokenizer"] = dict(metadata_sha256="wrong")
            monkeypatch.setattr(e, "tokenizer", lambda _: (None, {}))
        if case != "primitive":
            current["evidence"] = primitive
        atomic_json(raw / "primitive_evidence.json", primitive)
        atomic_json(raw / "measurement.json", current)
        value = e.build(current, raw, receipts, fixture=True)
        atomic_json(path, value)
        assert not e.replay(path), case
    spec = dict(
        name="coverage_json",
        argv=[str(e.ROOT / ".venv/bin/python"), "-c", "print('child')"],
        deadline_s=10,
    )
    monkeypatch.setattr(e.custody, "preserve", lambda *args: dict(private_adapter_test=True))
    assert e.run_check(e.ROOT, spec, tmp_path, tmp_path / "valid_logs")["passed"]


def test_admission_owned_failure_and_uncheckpointed_resume(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8263-CAUSAL: owned drift disqualifies and original keys bind."""
    rows = a.roster("natural")
    ledger = tmp_path / "single"
    row = {k: v for k, v in rows[0].items() if k != "label"}
    a.append(ledger, dict(kind="issue", record=a.predict(a.initial(), row)))
    altered = deepcopy(rows)
    altered[0]["original_source_sha256"] = "changed"
    with pytest.raises(ValueError, match="resume_roster"):
        a.run(altered, ledger)
    original_load = a.load
    monkeypatch.setattr(
        a, "load", lambda path: {} if path.name.startswith("restart-") else original_load(path)
    )
    work = e.measure(tmp_path, tmp_path / "drift", fixture=True)
    assert work["owned_failure"] == "hard_exit_resume_drift"
    value = e.build(work, tmp_path / "drift", [dict(name="private", passed=True)], fixture=True)
    assert value["verdict_class"] == "disqualified"
    assert value["view_kernel_ready_score"] == value["admission_kernel_ready_score"] == 0
