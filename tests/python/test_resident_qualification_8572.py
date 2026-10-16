"""REQ-REPORT-8572 / REQ-VERIFY-8572: real workers preserve immutable decisions."""

from copy import deepcopy
import json
from pathlib import Path
import struct
import time

import pytest

from carnot.reporting import resident_qualification_8572 as e
from carnot.reporting import resident_worker_8572 as w
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def bundle():
    """SCENARIO-REPORT-8572-PARITY: authenticate original producer bytes."""
    return e.admit(e.ROOT)


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """SCENARIO-REPORT-8572-PARITY: one genuine fixed workload supplies immutable primitives."""
    return e.measure(e.ROOT, tmp_path_factory.mktemp("measured8572"))


def test_original_operands_and_absence(bundle, tmp_path):
    """REQ-REPORT-8572: exact immutable roster retains original authority."""
    assert len(bundle["answers"]) == 900
    assert len(bundle["controls"]) == 274
    assert bundle["prototype_ready_score"] == 1
    assert bundle["protocol"]["primary"] == dict(batch=1, session_panels=16)
    with pytest.raises(e.GateFailure):
        e.admit(tmp_path)


def test_real_panel_resident_parity_and_clocks(bundle, measured):
    """SCENARIO-REPORT-8572-PARITY: actual extension and four arms agree."""
    primitive = measured["primitive"]
    assert len(primitive["parity_rows"]) == 900
    assert len(primitive["numerical_control_rows"]) == 274
    assert all(r["passed"] for r in primitive["rows"])
    assert primitive["clock_partition_receipt"]["passed"]
    assert primitive["loaded_extension_hash"] == bundle["extension"]["sha256"]
    assert e.replay_primitive(primitive, bundle)
    changed = deepcopy(primitive)
    changed["parity_rows"][0]["passed"] = False
    assert not e.replay_primitive(changed, bundle)


def test_worker_rejections_and_recovery(measured):
    """REQ-VERIFY-8572: failure controls use actual subprocess pipes."""
    rows = measured["primitive"]["failure_control_rows"]
    assert {r["kind"] for r in rows} >= {
        "malformed",
        "oversize",
        "eof_mid_frame",
        "crash",
        "deadline",
        "full_queue",
        "duplicate_id",
        "shutdown",
        "configuration_change",
    }
    assert all(r["passed"] for r in rows)
    assert all(r.get("missing_reason") is None for r in rows)


def test_decide_validation_and_numerical_controls(bundle):
    """REQ-VERIFY-8572: malformed requests never expose a preceding answer."""
    config = dict(head=bundle["head"], extension=bundle["extension"])
    row = bundle["answers"][0]
    request = w.envelope("a", "hash", row)
    seen = set()
    assert w.decide(request, config, "hash", None, seen)["status"] == "ok"
    for value in [
        request,
        [],
        dict(request, request_id="b", config_hash="wrong"),
        dict(request, request_id="c", deadline_ns=0),
        dict(request, request_id="d", row={}),
        dict(request, request_id="e", operation="unknown"),
    ]:
        response = w.decide(value, config, "hash", None, seen)
        assert response["status"] == "error" and "response" not in response
    shutdown = dict(w.envelope("stop", "hash", {}), operation="shutdown")
    assert w.decide(shutdown, config, "hash", None, seen)["status"] == "shutdown"
    assert (
        w.decide(
            dict(request, request_id="z", deadline_ns=time.monotonic_ns() + 10**9),
            config,
            "hash",
            None,
            seen,
        )["status"]
        == "ok"
    )


def test_snapshot_replay_consumer_and_tampering(bundle, measured, tmp_path):
    """SCENARIO-VERIFY-8572-REPLAY: repaired hashes never repair meaning."""
    work = deepcopy(measured)
    sealed = e.seal(work, tmp_path, "preliminary")
    output = tmp_path / (e.NAME + ".json")
    receipts = [dict(name=n, passed=True) for n in e.REQUIRED]
    value = e.artifact(sealed, receipts, output)
    atomic_json(output, value)
    assert e.replay(output)
    assert e.consume(output)["passed"]
    with pytest.raises(FileExistsError):
        e.seal(work, tmp_path, "preliminary")
    changed = deepcopy(value)
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(output, changed)
    assert not e.replay(output)
    assert not e.replay(tmp_path / "absent")
    failed = e.artifact(sealed, receipts + [dict(name="owned", passed=False)], output)
    assert failed["verdict_class"] == "disqualified"
    atomic_json(output, failed)
    with pytest.raises(ValueError, match="not_ready"):
        e.consume(output)


def test_worker_startup_and_terminal_errors(bundle, tmp_path, monkeypatch):
    """REQ-VERIFY-8572: genuine children reject startup, EOF, malformed config and limits."""
    import os
    import subprocess
    import sys

    from carnot.reporting import resident_measure_8572 as m

    path = tmp_path / "config.json"
    config = dict(head=bundle["head"], extension=bundle["extension"])
    atomic_json(path, config)
    digest = e.reference(path)["sha256"]
    for args in [
        ["--worker", str(path), "--config-hash", "wrong"],
        ["--worker", str(tmp_path / "absent")],
        ["--panel", str(tmp_path / "absent"), "--worker-output", str(tmp_path / "out")],
        ["--date", "20261015"],
        ["--private-control"],
    ]:
        receipt = subprocess.run(
            [sys.executable, "-u", str(e.ROOT / e.CLI), *args], capture_output=True, timeout=30
        )
        assert receipt.returncode != 0
    peer = m.Session(config, tmp_path / "eof", "python")
    receipt = peer.close(shutdown=False)
    assert receipt["exit_code"] == 0
    bad = deepcopy(config)
    bad["extension"]["sha256"] = "bad"
    atomic_json(path, bad)
    assert w.serve(path, e.reference(path)["sha256"], "native") == 1
    with pytest.raises(ValueError, match="immutable_config"):
        w.configuration(path, "wrong", "bad")
    atomic_json(path, config)

    class WrongExtension:
        __file__ = str(path)

    monkeypatch.setattr(w, "load", lambda path: WrongExtension())
    with pytest.raises(ValueError, match="loaded_extension"):
        w.configuration(path, digest, "native")
    code = (
        "from pathlib import Path; from carnot.reporting import resident_worker_8572 as w; "
        "w.MAX_REQUESTS=1; raise SystemExit(w.serve(Path(__import__('sys').argv[1]),"
        "__import__('sys').argv[2],'python'))"
    )
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", code, str(path), digest],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert json.loads(w.read_frame(process.stdout.fileno(), 30))["status"] == "ready"
        w.write_frame(process.stdin.fileno(), w.envelope("one", digest, bundle["answers"][0]))
        assert json.loads(w.read_frame(process.stdout.fileno(), 10))["status"] == "ok"
        assert process.wait(timeout=10) == 1
    finally:
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()


def test_evaluation_deadline_and_bad_operands(bundle, monkeypatch):
    """REQ-VERIFY-8572: late arithmetic is an error even when computation returned a value."""
    config = dict(head=bundle["head"], extension=bundle["extension"])
    original = w.evaluate

    def slow(head, row, native):
        result = original(head, row, native)
        time.sleep(0.02)
        return result

    monkeypatch.setattr(w, "evaluate", slow)
    value = dict(
        w.envelope("late", "hash", bundle["answers"][0]),
        deadline_ns=time.monotonic_ns() + 10_000_000,
    )
    assert w.decide(value, config, "hash", None, set())["error"] == "evaluation_deadline"
    with pytest.raises(ValueError, match="probability_input"):
        original(bundle["head"], dict(bundle["answers"][0], q=float("nan")), None)
    with pytest.raises(ValueError, match="probability_input"):
        original(bundle["head"], dict(bundle["answers"][0], q=2), None)


def test_blocked_owned_errors_and_snapshot_drift(bundle, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8572-REPLAY: owned errors disqualify while missing upstream bytes block."""
    work = e.measure(tmp_path / "missing", tmp_path / "blocked")
    output = tmp_path / (e.NAME + ".json")
    sealed = e.seal(work, tmp_path, "blocked")
    value = e.artifact(sealed, [], output)
    assert value["verdict_class"] == "blocked"
    atomic_json(output, value)
    assert e.replay(output)
    changed = deepcopy(sealed)
    changed["work"]["duration_s"] += 1
    with pytest.raises(ValueError, match="snapshot_drift"):
        e.artifact(changed, [], output)

    def broken(root):
        raise ValueError("owned_math_failure")

    monkeypatch.setattr(e, "admit", broken)
    work = e.measure(e.ROOT, tmp_path / "owned")
    assert work["owned_failure"] == "owned_math_failure"
    sealed = e.seal(work, tmp_path, "owned")
    assert e.artifact(sealed, [], output)["verdict_class"] == "disqualified"


def test_runner_private_publication_and_dispatch(tmp_path, monkeypatch):
    """REQ-VERIFY-8572: actual private CLI publishes a bound blocked primary and rejects tampering."""
    import subprocess
    import sys
    from carnot.reporting import resident_runner_8572 as runner
    from carnot.reporting.primary_publication import read_bound_sidecar

    output = tmp_path / (e.NAME + ".json")
    process = subprocess.run(
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
            "--private-control",
        ],
        capture_output=True,
        timeout=90,
    )
    assert process.returncode == 0, process.stderr.decode()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert read_bound_sidecar(output, Path(side["publication"]["sidecar_path"]))["report"]["passed"]
    assert runner.main(["--deliberate-error"]) == 7
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    monkeypatch.setattr(runner, "run", lambda root, output, control: 0)
    assert runner.main(["--output", str(output)]) == 0


def test_scoped_controller_retains_exact_nodes_and_failure(tmp_path, monkeypatch):
    """REQ-VERIFY-8572: a real scoped child retains its union and incomplete coverage failure."""
    from carnot.reporting import resident_validation_8572 as validation

    suite = e.TEST + "::test_decide_validation_and_numerical_controls"
    monkeypatch.setattr(validation, "SUITES", [suite])
    work = {}
    receipts = validation.validate(work, tmp_path / "small-controller")
    assert all(r["passed"] for r in receipts)
    assert work["exact_union_receipt"]["collected"] == [suite]
    checks = validation.finish(work, tmp_path / "small-controller")
    assert not next(r for r in checks if r["name"] == "owned_python_coverage_100")["passed"]
    assert any(r["name"] == "coverage_report" and r["exit_code"] == 2 for r in checks)


def test_runner_acceptance_and_owned_failure(measured, tmp_path, monkeypatch):
    """REQ-VERIFY-8572: private publication exercises both readiness and failed owned checks."""
    from carnot.reporting import resident_runner_8572 as runner

    def cached(root, raw):
        value = deepcopy(measured)
        value["raw"] = str(raw)
        return value

    monkeypatch.setattr(e, "measure", cached)
    # These fixture receipts exercise publication branches; production uses real collected checks.
    monkeypatch.setattr(
        runner.validation,
        "validate",
        lambda work, raw: [dict(name=n, passed=True) for n in e.REQUIRED],
    )
    monkeypatch.setattr(runner.validation, "finish", lambda work, raw: [])
    output = tmp_path / "ready" / (e.NAME + ".json")
    assert runner.run(e.ROOT, output) == 0
    value = json.loads(output.read_bytes())
    assert value["resident_inference_ready_score"] == 1
    assert value["cost_consumer_receipt"]["passed"]
    side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert side["actual_cost_consumer"]["passed"]
    assert side["selected_identity"]["passed"]

    def broken(root, raw):
        value = cached(root, raw)
        value["owned_failure"] = "deliberate_private_owned_failure"
        return value

    monkeypatch.setattr(e, "measure", broken)
    output = tmp_path / "failed" / (e.NAME + ".json")
    assert runner.run(e.ROOT, output, control=True) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"


def test_real_child_cleanup_and_rejected_panel(bundle, tmp_path, monkeypatch):
    """REQ-VERIFY-8572: a rejected handshake and broken request cannot leave a live peer."""
    from carnot.reporting import resident_measure_8572 as m

    fake = tmp_path / "wrong_peer.py"
    fake.write_text(
        "import os,struct,time\np=b'{}'\nos.write(1,struct.pack('!I',len(p))+p)\ntime.sleep(30)\n"
    )
    original_cli = m.CLI
    monkeypatch.setattr(m, "CLI", str(fake))
    with pytest.raises(ValueError, match="session_ready"):
        m.Session(
            dict(head=bundle["head"], extension=bundle["extension"]),
            tmp_path / "handshake",
            "python",
        )
    monkeypatch.setattr(m, "CLI", original_cli)
    original_roster = m.roster(bundle)
    monkeypatch.setattr(m, "roster", lambda bundle: [original_roster[2]])

    def broken(self, row, identifier):
        raise ValueError("private_request_failure")

    monkeypatch.setattr(m.Session, "request", broken)
    with pytest.raises(ValueError, match="private_request_failure"):
        m.execute(bundle, tmp_path / "broken_request")
    monkeypatch.setattr(m, "roster", lambda bundle: [original_roster[0]])
    monkeypatch.setattr(m, "child", lambda *args, **kwargs: dict(passed=False))
    with pytest.raises(ValueError, match="panel_child_failed"):
        m.execute(bundle, tmp_path / "broken_panel")


def test_rehashed_primitive_and_clock_tampering(measured, tmp_path):
    """SCENARIO-VERIFY-8572-REPLAY: semantic replay rejects independently rehashed counterfeit responses."""
    work = deepcopy(measured)
    work["primitive"]["sessions"][0]["observations"][0]["response"]["response"]["probability"] += (
        0.01
    )
    sealed = e.seal(work, tmp_path, "forged")
    output = tmp_path / (e.NAME + ".json")
    value = e.artifact(sealed, [], output)
    atomic_json(output, value)
    assert not e.replay(output)
    good = e.seal(measured, tmp_path, "good")
    value = e.artifact(good, [], output)
    value["reproducibility_checksum"] = "wrong"
    atomic_json(output, value)
    assert not e.replay(output)
    primitive = deepcopy(measured["primitive"])
    primitive["sessions"][0]["exit_code"] = 1
    assert not e.replay_primitive(primitive, measured["bundle"])
    from carnot.reporting import resident_measure_8572 as m

    receipt = deepcopy(measured["primitive"]["sessions"][0])
    receipt["end_ns"] = receipt["start_ns"]
    assert not m.partition(receipt, "panel")["passed"]


def test_native_numerical_fallback_and_owned_gate(bundle, tmp_path, monkeypatch):
    """REQ-VERIFY-8572: the actual extension counts numerical fallback and owned failures."""
    from carnot.verify.native_nine_input_8533 import load

    native = load(Path(bundle["extension"]["path"]))
    head = dict(bundle["head"], parameters=[0.0] * 109, decay_scale=1.0, temperature=1.0)
    low, high = head["scaler"]["minimum"][0], head["scaler"]["maximum"][0]
    row = dict(
        unit_id="numerical-threshold",
        kind="numerical",
        decay_multiplier=1.0,
        scaled_features=[(0.2 - low) / (high - low), *([0.5] * 8)],
    )
    response, _ = w.evaluate(head, row, native)
    assert response["boundary_fallback"]
    assert abs(response["probability"] - 0.2) < 1e-12
    monkeypatch.setattr(e, "admit", lambda root: bundle)

    def failed(bundle, raw):
        e.require(raw, "private_owned_gate", True, False)

    monkeypatch.setattr(e, "execute", failed)
    work = e.measure(e.ROOT, tmp_path / "failed")
    assert work["owned_failure"] == "private_owned_gate"
