"""REQ-REPORT-8558 / REQ-VERIFY-8558: immutable candidates retain all evidence."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import nine_input_qualification_8558 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """SCENARIO-REPORT-8558-SNAPSHOTS: use the actual original loaded binary."""
    return e.measure(e.ROOT, tmp_path_factory.mktemp("native-8558"))


def test_snapshot_order_and_separate_versions(measured, tmp_path):
    """SCENARIO-REPORT-8558-SNAPSHOTS: stale reductions fail before repair."""
    work = deepcopy(measured)
    work = e.seal(work, tmp_path, "preliminary")
    output = tmp_path / (e.NAME + ".json")
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, e.artifact(work, [], output))
    assert e.replay(candidate)
    original = Path(work["raw"]) / "measurement.json"
    original_bytes = original.read_bytes()
    work["owned_failure"] = True
    with pytest.raises(ValueError, match="snapshot_drift"):
        e.artifact(work, [], output)
    terminal = e.seal(work, tmp_path, "terminal")
    assert terminal["raw"] != work["raw"]
    assert original.read_bytes() == original_bytes
    with pytest.raises(FileExistsError):
        e.seal(work, tmp_path, "preliminary")
    atomic_json(candidate, e.artifact(terminal, [], output))
    assert e.replay(candidate)
    assert json.loads(candidate.read_bytes())["verdict_class"] == "disqualified"


@pytest.mark.parametrize("mutation", ["checksum", "counter", "snapshot", "missing", "rows"])
def test_cold_replay_rejects_changed_operands(measured, tmp_path, mutation):
    """SCENARIO-REPORT-8558-SNAPSHOTS: repaired hashes cannot change observations."""
    work = e.seal(measured, tmp_path, "test")
    output = tmp_path / (e.NAME + ".json")
    value = e.artifact(work, [], output)
    if mutation == "checksum":
        value["reproducibility_checksum"] = "wrong"
    elif mutation == "counter":
        value["native_execution_count"] += 1
        value["reproducibility_checksum"] = e.legacy.checksum(value)
    elif mutation == "missing":
        value["primitive_reference"]["path"] += ".absent"
        value["reproducibility_checksum"] = e.legacy.checksum(value)
    else:
        primitive = Path(work["raw"]) / "measurement.json"
        primitive.chmod(0o644)
        changed = deepcopy(work)
        changed["rows"][0]["native_residual"] += 1
        atomic_json(primitive, changed)
        if mutation == "rows":
            value = e.artifact(changed, [], output)
    atomic_json(output, value)
    assert not e.replay(output)
    assert not e.replay(tmp_path / "absent")


def test_historical_failure_reproduces(measured, tmp_path):
    """SCENARIO-REPORT-8558-SNAPSHOTS: diagnosis must actually reproduce exit1."""
    receipt, diagnosis = e.reproduce(measured, tmp_path)
    assert receipt["passed"] and receipt["actual_exit"] == 1
    assert diagnosis["hypothesis_reproduced"]
    assert diagnosis["field_differences"]


def test_blocked_and_required_failure(tmp_path):
    """REQ-REPORT-8558: missing evidence blocks and owned failure disqualifies."""
    work = e.measure(tmp_path / "absent", tmp_path / "raw")
    work = e.seal(work, tmp_path, "blocked")
    output = tmp_path / (e.NAME + ".json")
    value = e.artifact(work, [], output)
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    atomic_json(output, value)
    with pytest.raises(ValueError, match="not_ready"):
        e.consume(output)
    assert (
        e.artifact(work, [dict(name="owned", passed=False)], output)["verdict_class"]
        == "disqualified"
    )


def test_actual_cli_rejections(measured, tmp_path):
    """SCENARIO-VERIFY-8558-CONTROLS: fresh children retain exit and log evidence."""
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    bundle = tmp_path / "bundle.json"
    atomic_json(
        bundle,
        dict(
            head=measured["head_reference"],
            tape=measured["request_tape_reference"],
            extension=measured["extension"],
        ),
    )
    for name, args, expected in [
        (
            "request",
            ["--request-bundle", str(bundle), "--request-output", str(tmp_path / "responses.json")],
            0,
        ),
        (
            "bad_request",
            [
                "--request-bundle",
                str(tmp_path / "absent"),
                "--request-output",
                str(tmp_path / "bad.json"),
            ],
            1,
        ),
        ("missing_output", ["--request-bundle", str(bundle)], 2),
        ("date", ["--date", "19000101"], 2),
        ("private_guard", ["--private-control"], 2),
        ("owned", ["--deliberate-error"], 7),
        ("absent", ["--cold-replay", str(tmp_path / "absent")], 1),
    ]:
        receipt = child(name, argv + args, tmp_path / "children", expected=expected, deadline=120)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    assert len(json.loads((tmp_path / "responses.json").read_bytes())["rows"]) == 1800


def test_runner_and_consumer_routes(measured, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8558-PUBLICATION: real publication binds separate versions."""
    from carnot.reporting import nine_input_runner_8558 as r

    with monkeypatch.context() as m:
        m.setattr(r.validation.previous, "historical", lambda *a, **kw: 7)
        assert r.main(["--sealed-tests", "fixture"]) == 7
    for control in [True, False]:
        with monkeypatch.context() as m:
            m.setattr(r.validation, "validate", lambda *a: [])
            m.setattr(r.validation, "finish", lambda *a: [])
            assert r.run(e.ROOT, tmp_path / str(control) / (e.NAME + ".json"), control) == 0
    output = tmp_path / "True" / (e.NAME + ".json")
    assert r.main(["--cold-replay", str(output)]) == 0
    assert r.main(["--consume", str(output)]) == 1
    with monkeypatch.context() as m:
        m.setattr(e, "replay", lambda p: True)
        value = json.loads(output.read_bytes())
        value.update(nine_input_native_ready_score=1, required_checks_passed=True)
        atomic_json(output, value)
        m.setattr(e.previous, "consume", lambda p: dict(ready=True))
        assert e.consume(output)["ready"]
        assert r.main(["--consume", str(output)]) == 0
    with monkeypatch.context() as m:
        m.setattr(r, "run", lambda *a: 0)
        assert r.main(["--private-control", "--output", str(tmp_path / "private.json")]) == 0


def test_original_binary_rejection_and_private_absence(tmp_path, monkeypatch):
    """REQ-REPORT-8558: changed build identity rejects and missing inputs stay blocked."""
    from carnot.reporting import nine_input_runner_8558 as r

    with monkeypatch.context() as m:
        original = e.inputs.reference
        m.setattr(e.inputs, "reference", lambda p: dict(original(p), sha256="changed"))
        with pytest.raises(ValueError, match="original_extension_hash"):
            e.measure(e.ROOT, tmp_path / "changed")
    with monkeypatch.context() as m:
        m.setattr(e.legacy.build, "FLAGS", "changed")
        work = e.measure(e.ROOT, tmp_path / "changed_flags")
        assert work["owned_failure"]
        assert work["failures"][-1]["field"] == "original_build_identity"
    assert r.run(tmp_path / "absent", tmp_path / "blocked" / (e.NAME + ".json"), True) == 0
    value = json.loads((tmp_path / "blocked" / (e.NAME + ".json")).read_bytes())
    assert value["verdict_class"] == "blocked"


def test_validation_adapters(tmp_path, monkeypatch):
    """REQ-VERIFY-8558: adapters invoke existing checks without changing their gates."""
    from carnot.reporting import nine_input_validation_8558 as v

    def historical(suite, argv, *, script=False):
        assert v.previous.e.CLI == e.CLI
        return 7

    monkeypatch.setattr(v.previous, "historical", historical)
    assert v.historical("fixtures", [], script=True) == 7
    monkeypatch.setattr(v.previous, "validate", lambda *a: [dict(passed=True)])
    assert v.validate({}, tmp_path) == [dict(passed=True)]
    binary = tmp_path / "original" / "nine_input_8533-test"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(v.previous, "finish", lambda *a: [dict(passed=True)])
    work = dict(extension=dict(path=str(binary.parent / "lib.so")))
    receipts = v.finish(work, tmp_path / "current")
    assert all(r["passed"] for r in receipts)


def test_complete_ready_and_failed_gates(measured, tmp_path):
    """REQ-REPORT-8558: every required receipt participates in readiness."""
    work = e.seal(measured, tmp_path, "ready")
    receipts = [dict(name=name, passed=True) for name in sorted(e.REQUIRED)]
    value = e.artifact(work, receipts, tmp_path / (e.NAME + ".json"))
    assert value["nine_input_native_ready_score"] == 1
    receipts[0]["passed"] = False
    assert e.artifact(work, receipts, tmp_path / "out.json")["nine_input_native_ready_score"] == 0
