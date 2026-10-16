"""REQ-REPORT-8581, REQ-VERIFY-8581: qualify only explicitly bound new evidence."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import causal_accounting_8581 as e
from carnot.reporting import causal_accounting_runner_8581 as r
from carnot.reporting import causal_accounting_validation_8581 as v
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.discrete_count_proof import proven_counts
from carnot.reporting.v709_execution import child


def test_explicit_revision_and_original_counters(tmp_path):
    """SCENARIO-REPORT-8581-REVISION: alternate revisions cannot impersonate qualification."""
    binding, refs, failures = e.authority(e.ROOT)
    assert not failures and refs and binding["learning_inputs_ready_score"] == 1
    revision = binding["current_validator_reference"]
    accepted = e.accept_revision(revision)
    assert accepted["ready"]
    for field in ("code", "controls", "commands"):
        changed = deepcopy(accepted)
        changed[field] = []
        path = tmp_path / (field + ".json")
        atomic_json(path, changed)
        with pytest.raises(ValueError, match="unqualified_revision_reference"):
            e.accept_revision(e.old.reference(path))
    assert e.authority(tmp_path)[2][0]["observed"] is None
    events, _ = e.old.history()
    metrics, proof = e.previous.proof(events)
    assert metrics["admission_events"] == metrics["label_release_events"] == 2400
    assert metrics["applied_update_events"] == metrics["decay_scale_changes"] == 1800
    assert (
        len(
            proven_counts(
                dict(metrics, intended_count=2400, completed_count=2400, discrete_count_proof=proof)
            )
        )
        == 10
    )
    assert e.previous.authority(e.ROOT)[2]
    assert e.previous.VALIDATOR_PIN == e.ORIGINAL_VALIDATOR_PIN


def test_replay_and_gate_classes(tmp_path):
    """SCENARIO-VERIFY-8581-REPLAY: repaired envelopes cannot change source events or receipts."""
    work = r.prepare(tmp_path / "raw", tmp_path / "scratch", private=True)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.build(work, output))
    assert e.replay(output)
    assert not e.replay(tmp_path / "absent")
    value = json.loads(output.read_bytes())
    value["metrics"]["admission_events"] += 1
    value["reproducibility_checksum"] = e.old.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)
    work["owned_ready"] = True
    atomic_json(Path(work["raw"]) / "final.json", work)
    assert e.build(work, output)["verdict_class"] == "null"
    work["failures"] = [e.old.gate(tmp_path / "missing", "learning_inputs_ready_score", 1, None)]
    atomic_json(Path(work["raw"]) / "final.json", work)
    assert e.build(work, output)["verdict_class"] == "blocked"
    work["receipts"][0]["passed"] = False
    atomic_json(Path(work["raw"]) / "final.json", work)
    assert e.build(work, output)["verdict_class"] == "disqualified"
    atomic_json(output, e.build(work, output))
    assert not e.replay(output)


def test_real_private_controller_and_rejections(tmp_path):
    """SCENARIO-VERIFY-8581-REPLAY: genuine child exits bind publication and cold rejection."""
    output = tmp_path / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    receipt = child(
        "private", [*argv, "--private-output", str(output)], tmp_path / "logs", deadline=240
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["online_adapter_ready_score"] == 0
    assert value["verdict_class"] == "disqualified" and e.replay(output)
    assert r.terminal(output, tmp_path / "direct_terminal")["passed"]
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert terminal["selected_identity"]["passed"]
    assert not terminal["stream_consumer"]["passed"]
    changed = deepcopy(value)
    changed["required_checks_passed"] = True
    changed["reproducibility_checksum"] = e.old.checksum(changed)
    atomic_json(tmp_path / "tamper.json", changed)
    for name, args, expected in [
        ("cold", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("tamper", ["--cold-replay", str(tmp_path / "tamper.json")], 1),
        ("owned", ["--deliberate-error"], 7),
        ("date", ["--date", "wrong"], 2),
        ("unsafe", ["--private-output", str(e.ROOT / "results" / (e.NAME + ".json"))], 2),
    ]:
        assert child(name, [*argv, *args], tmp_path / "logs", expected=expected, deadline=90)[
            "passed"
        ]
    assert r.main(["--date", "wrong"]) == 2
    assert r.main(["--deliberate-error"]) == 7
    assert r.main(["--cold-replay", str(output)]) == 0


def test_validation_union_and_dispatch(tmp_path, monkeypatch):
    """REQ-VERIFY-8581: missing terminal nodes and uncovered statements cannot qualify."""
    legacy = v.previous
    work = dict(raw=str(tmp_path), receipts=[], owned_ready=False)
    node = e.TESTS[0] + "::test_explicit_revision_and_original_counters"
    deadlines = {}

    def fake_child(name, argv, logs, **kwargs):
        deadlines[name] = kwargs["deadline"]
        logs.mkdir(exist_ok=True)
        output = logs / (name + ".stdout")
        output.write_text(node + "\n")
        if name in {"owned", "e2e018", "e2e021"}:
            target = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--junitxml=")))
            target.write_text("<testsuites/>")
        if name == "coverage":
            atomic_json(
                tmp_path / "coverage.json",
                dict(files={p: dict(summary=dict(missing_lines=0)) for p in e.OWNED}),
            )
        return dict(name=name, passed=True, stdout_path=str(output))

    monkeypatch.setattr(legacy, "child", fake_child)
    v.validate(work)
    assert not work["union_passed"] and not v.ready(work)
    assert deadlines["owned"] == 900
    assert legacy.e is e.previous
    monkeypatch.setattr(
        legacy,
        "terminal_nodes",
        lambda p: [dict(node_id=node, outcome="passed")] if p.stem == "owned" else [],
    )
    work["receipts"] = []
    v.validate(work)
    assert v.ready(work)
    assert str(e.ROOT / e.CLI) in v.configuration(tmp_path).read_text()


def test_controller_dispatch_deadline_and_resource_gates(tmp_path, monkeypatch):
    """REQ-VERIFY-8581: production dispatch and required resource failures remain explicit."""
    work = r.prepare(tmp_path / "raw", tmp_path / "scratch", private=True)
    work["raw"] = str(tmp_path / "raw")
    monkeypatch.setattr(r, "prepare", lambda *a, **k: work)
    monkeypatch.setattr(v, "validate", lambda w: None)
    monkeypatch.setattr(r, "publish_primary", lambda *a: dict(sidecar_path="private"))
    monkeypatch.setattr(r.consumer, "stream_consumer", lambda *a: dict(passed=False))
    monkeypatch.setattr(r, "reader_receipt", lambda *a, **k: dict(passed=True))
    clock = iter([0.0, 3601.0, 3602.0, 3603.0])
    monkeypatch.setattr(r.time, "monotonic", lambda: next(clock))
    assert r.run(tmp_path / (e.NAME + ".json")) == 0
    assert work["failures"][-1]["field"] == "measurement_deadline"
    monkeypatch.setattr(r, "run", lambda *a, **k: 0)
    assert r.main([]) == 0


def test_current_dependency_failure_branches(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8581-REVISION: missing readiness or publication prevents admission."""
    state = dict(publication=True, validator=1, learning=1)

    def read(ref):
        return dict(
            terminal_validation_sidecar_path=str(tmp_path / "terminal"),
            publication=dict(sidecar_path="bound"),
            current_validator_ready_score=state["validator"],
            current_validator_reference={},
            learning_inputs_ready_score=state["learning"],
        )

    monkeypatch.setattr(e.old, "read", read)
    monkeypatch.setattr(e.old, "reference", lambda p: dict(path=str(p), sha256="fixture"))
    monkeypatch.setattr(e, "accept_revision", lambda ref: {})
    monkeypatch.setattr(
        e, "read_bound_sidecar", lambda *a: dict(report=dict(passed=state["publication"]))
    )
    for key, bad, error in [
        ("publication", False, "upstream_publication"),
        ("validator", 0, "current_validator_ready_score"),
        ("learning", 0, "learning_inputs_ready_score"),
    ]:
        state[key] = bad
        failures = e.authority(tmp_path)[2]
        assert any(
            f["observed"] == error or (f["field"] == error and f["observed"] == 0) for f in failures
        )
        state[key] = True if key == "publication" else 1


def test_rehashed_primitive_and_receipt_rejections(tmp_path):
    """SCENARIO-VERIFY-8581-REPLAY: rehashing all envelopes cannot invent numerical activity."""
    work = r.prepare(tmp_path / "raw", tmp_path / "scratch", private=True)
    output = tmp_path / (e.NAME + ".json")

    def seal():
        atomic_json(Path(work["raw"]) / "final.json", work)
        atomic_json(output, e.build(work, output))

    seal()
    value = json.loads(output.read_bytes())
    value["reproducibility_checksum"] = "wrong"
    atomic_json(output, value)
    assert not e.replay(output)
    seal()
    value = json.loads(output.read_bytes())
    value["code_config_hashes"][0]["sha256"] = "wrong"
    value["reproducibility_checksum"] = e.old.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)
    receipt = work["receipts"][0]
    original = deepcopy(receipt)
    receipt["stdout_sha256"] = "wrong"
    atomic_json(Path(receipt["log_path"]).with_suffix(".receipt.json"), receipt)
    seal()
    assert not e.replay(output)
    receipt.update(original)
    atomic_json(Path(receipt["log_path"]).with_suffix(".receipt.json"), receipt)
    events = e.previous.read(work["events"])
    events[0]["after"]["parameters"][0] = 3.0
    atomic_json(Path(work["events"]["path"]), events)
    work["events"] = e.old.reference(Path(work["events"]["path"]))
    seal()
    assert not e.replay(output)
