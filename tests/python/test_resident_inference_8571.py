"""REQ-REPORT-8571 / REQ-VERIFY-8571: lifecycle changes preserve numerical work."""

from copy import deepcopy
import io
import json
import os
from pathlib import Path
import struct
import time

import pytest

from carnot.reporting import resident_inference_8571 as e
from carnot.reporting import resident_worker_8571 as w
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def bundle():
    """SCENARIO-REPORT-8571-PROTOTYPE: original producer hashes supply authority."""
    return e.admit(e.ROOT)


def test_original_parity_cost_and_missing(bundle, tmp_path):
    """REQ-REPORT-8571: preserve the complete-cost null and all109 coefficients."""
    assert len(bundle["head"]["parameters"]) == 109
    assert bundle["historical_cost_null"]["ratio"] == 1.0101493212248978
    assert bundle["historical_cost_null"]["ci95"][0] < 1
    with pytest.raises(e.GateFailure) as error:
        e.admit(tmp_path)
    assert error.value.row["observed"] is None


def test_protocol_rejects_without_stale_response(bundle):
    """REQ-VERIFY-8571: rejected requests never carry a preceding response."""
    config = dict(head=bundle["head"], extension=bundle["extension"])
    request = w.envelope("a", "hash", bundle["requests"][0])
    state = set()
    valid = w.decide(request, config, "hash", None, state)
    assert valid["status"] == "ok" and json.loads(valid["response"])["probability"] >= 0
    for changed in [
        dict(request),
        dict(request, request_id="b", config_hash="wrong"),
        dict(request, request_id="c", deadline_ns=0),
        [],
        dict(request, request_id="d", row={}),
        dict(request, request_id="e", row=dict(request["row"], q=float("nan"))),
    ]:
        response = w.decide(changed, config, "hash", None, state)
        assert response["status"] == "error" and "response" not in response
    shutdown = dict(
        request_id="shutdown",
        config_hash="hash",
        operation="shutdown",
        deadline_ns=time.monotonic_ns() + 10**9,
    )
    assert w.decide(shutdown, config, "hash", None, state)["status"] == "shutdown"


def test_framing_eof_oversize_and_deadline():
    """REQ-VERIFY-8571: the byte boundary has bounded size and wait time."""
    reader, writer = os.pipe()
    try:
        w.write_frame(writer, dict(a=1))
        assert json.loads(w.read_frame(reader, 1)) == {"a": 1}
        with pytest.raises(TimeoutError):
            w.read_frame(reader, 0.001)
        os.write(writer, struct.pack("!I", w.MAX_FRAME + 1))
        with pytest.raises(ValueError, match="oversized"):
            w.read_frame(reader, 1)
        os.write(writer, b"x")
        os.close(writer)
        with pytest.raises(ValueError, match="truncated"):
            w.read_frame(reader, 1)
        assert w.read_frame(reader, 1) is None
    finally:
        os.close(reader)
    with pytest.raises(ValueError, match="oversized"):
        w.write_frame(-1, {"x": "x" * w.MAX_FRAME})


def test_eight_requests_real_processes_and_restarts(bundle, tmp_path):
    """SCENARIO-REPORT-8571-PROTOTYPE: real native/Python children restart cleanly."""
    result = e.prototype(bundle, tmp_path)
    assert len(result["prototype_request_rows"]) == 8
    assert all(r["passed"] for r in result["prototype_request_rows"])
    assert len({r["pid"] for r in result["lifecycle_receipts"]}) == 4
    assert all(r["exit_code"] == 0 and r["clean_shutdown"] for r in result["lifecycle_receipts"])
    assert sum(r["response"]["status"] == "error" for r in result["prototype_request_rows"]) == 2
    valid = [
        r for r in result["prototype_request_rows"] if r["session"] == 0 and r["kind"] == "valid"
    ]
    assert valid[0]["request"]["row"] != valid[1]["request"]["row"]
    assert e.replay_primitive(result, bundle)
    tamper = deepcopy(result)
    tamper["prototype_request_rows"][0]["response"]["response"] = "{}"
    assert not e.replay_primitive(tamper, bundle)


def test_artifact_replay_and_owned_failure(bundle, tmp_path):
    """SCENARIO-VERIFY-8571-REPLAY: checksum repair cannot grant semantic authority."""
    primitive = e.prototype(bundle, tmp_path / "primitive")
    work = dict(
        bundle=bundle,
        primitive=primitive,
        raw=str(tmp_path),
        gate_check_summary=[],
        preconditions_checked={},
        duration_s=1.0,
        phase_spans=[],
    )
    atomic_json(tmp_path / "work.json", work)
    receipts = [dict(name=name, passed=True) for name in e.REQUIRED]
    value = e.artifact(work, receipts, tmp_path / (e.NAME + ".json"))
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate)
    assert value["prototype_ready_score"] == 1 and value["verdict_class"] == "null"
    repeated = [dict(name="cold", passed=False), *receipts]
    assert e.artifact(work, repeated, tmp_path / (e.NAME + ".json"))["prototype_ready_score"] == 0
    forged = deepcopy(work)
    forged["primitive"]["prototype_request_rows"][0]["response"]["response"] = "{}"
    atomic_json(tmp_path / "forged_work.json", forged)
    changed = dict(value, work_reference=e.reference(tmp_path / "forged_work.json"))
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(candidate, changed)
    assert not e.replay(candidate)
    changed = deepcopy(value)
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(candidate, changed)
    assert not e.replay(candidate)
    value["reproducibility_checksum"] = "wrong"
    atomic_json(candidate, value)
    assert not e.replay(candidate)
    assert not e.replay(tmp_path / "absent")
    receipts.append(dict(name="actual_failure", passed=False))
    failed = e.artifact(work, receipts, tmp_path / (e.NAME + ".json"))
    assert failed["verdict_class"] == "disqualified" and failed["prototype_ready_score"] == 0
    blocked = dict(work, bundle=None, primitive=None, gate_check_summary=[dict(field="missing")])
    assert e.artifact(blocked, [], tmp_path / (e.NAME + ".json"))["verdict_class"] == "blocked"


def test_worker_cli_eof_rejections_and_date(bundle, tmp_path):
    """REQ-VERIFY-8571: genuine children reject bad startup and unsafe byte streams."""
    import subprocess
    import sys

    config = tmp_path / "config.json"
    atomic_json(config, dict(head=bundle["head"], extension=bundle["extension"]))
    digest = e.reference(config)["sha256"]
    base = [
        sys.executable,
        "-u",
        str(e.ROOT / e.CLI),
        "--worker",
        str(config),
        "--config-hash",
        digest,
    ]
    for payload, expected in [(b"", 0), (b"\0\1\0\1", 1), (b"x", 1), (b"\0\0\0\0", 1)]:
        result = subprocess.run(base, input=payload, capture_output=True, timeout=40)
        assert result.returncode == expected, result.stderr
    result = subprocess.run([*base, "--config-hash", "changed"], capture_output=True, timeout=40)
    assert result.returncode == 1
    bad = dict(head=bundle["head"], extension=dict(bundle["extension"], sha256="wrong"))
    atomic_json(config, bad)
    result = subprocess.run(
        [*base, "--config-hash", e.reference(config)["sha256"], "--arm", "native"],
        capture_output=True,
        timeout=40,
    )
    assert result.returncode == 1
    result = subprocess.run(
        [sys.executable, str(e.ROOT / e.CLI), "--date", "20261015"], capture_output=True, timeout=40
    )
    assert result.returncode == 2


def test_private_publication_and_consumer(tmp_path):
    """REQ-VERIFY-8571: real CLI publication binds bytes even in unqualified control mode."""
    import subprocess
    import sys

    output = tmp_path / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    result = subprocess.run(
        [*argv, "--private-control", "--output", str(output)], capture_output=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(output.read_bytes())
    assert value["prototype_ready_score"] == 0
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert terminal["publication"]["primary_sha256"] == e.reference(output)["sha256"]
    assert terminal["selected_identity"]["gate_path"] == str(output)
    result = subprocess.run([*argv, "--cold-replay", str(output)], capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([*argv, "--private-control"], capture_output=True, timeout=30)
    assert result.returncode == 2


def test_validation_scope_adapter_restores_originals(tmp_path, monkeypatch):
    """REQ-VERIFY-8571: adapting owned scope cannot permanently alter historical controllers."""
    from carnot.reporting import resident_validation_8571 as v

    original = v.legacy.e

    def verify(work, raw):
        assert v.legacy.e is e and raw == tmp_path
        assert v.legacy.SUITES == v.SUITES
        assert len(v.SUITES) == 3
        return [dict(name="scope", passed=True)]

    monkeypatch.setattr(v.legacy, "validate", verify)
    monkeypatch.setattr(v.legacy, "finish", lambda work, raw: [dict(name="finish", passed=True)])
    assert v.validate({}, tmp_path)[0]["passed"]
    assert v.finish({}, tmp_path)[0]["passed"]
    assert v.legacy.e is original


def test_typed_validation_and_evaluation_deadline(bundle, monkeypatch):
    """REQ-VERIFY-8571: invalid IDs, operations and late computations return fresh errors."""
    config = dict(head=bundle["head"], extension=bundle["extension"])
    base = w.envelope("valid", "hash", bundle["requests"][0])
    for request in [
        dict(base, request_id=""),
        dict(base, request_id="x" * 65),
        dict(base, operation="unknown"),
        dict(base, row=dict(base["row"], public=[])),
        dict(base, row=dict(base["row"], public={})),
        dict(base, row=dict(base["row"], q="0.5")),
    ]:
        assert w.decide(request, config, "hash", None, set())["status"] == "error"
    ticks = iter([1, 1, 30_000_000_001])
    from types import SimpleNamespace

    monkeypatch.setattr(w, "time", SimpleNamespace(monotonic_ns=lambda: next(ticks)))
    request = dict(base, deadline_ns=20_000_000_000)
    assert w.decide(request, config, "hash", None, set())["error"] == "evaluation_deadline"


def test_request_limit_configuration_change_and_idle_child(bundle, tmp_path):
    """REQ-VERIFY-8571: real children bound session length, idle peers and mutable files."""
    import subprocess
    import sys

    config = tmp_path / "config.json"
    atomic_json(config, dict(head=bundle["head"], extension=bundle["extension"]))
    digest = e.reference(config)["sha256"]
    argv = [
        sys.executable,
        "-u",
        str(e.ROOT / e.CLI),
        "--worker",
        str(config),
        "--config-hash",
        digest,
    ]
    parts = []
    for i in range(64):
        value = w.envelope(str(i), digest, bundle["requests"][0])
        encoded = json.dumps(value).encode()
        parts.append(struct.pack("!I", len(encoded)) + encoded)
    result = subprocess.run(argv, input=b"".join(parts), capture_output=True, timeout=50)
    assert result.returncode == 1 and b"request_limit" in result.stderr
    for mutation in ["config", "idle"]:
        with subprocess.Popen(
            [*argv, "--timeout", "0.05"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process:
            assert json.loads(w.read_frame(process.stdout.fileno(), 30))["status"] == "ready"
            if mutation == "config":
                atomic_json(
                    config, dict(head=bundle["head"], extension=bundle["extension"], changed=True)
                )
                w.write_frame(
                    process.stdin.fileno(), w.envelope("mutated", digest, bundle["requests"][0])
                )
            assert process.wait(timeout=10) == 1
            assert w.read_frame(process.stdout.fileno(), 1) is None
            if mutation == "config":
                atomic_json(config, dict(head=bundle["head"], extension=bundle["extension"]))


def test_invalid_child_handshake_is_killed(bundle, tmp_path, monkeypatch):
    """REQ-VERIFY-8571: a broken resident child cannot outlive its failed parent session."""
    fake = tmp_path / "broken_peer.py"
    fake.write_text(
        'import os,struct,time\np=b"{}"\nos.write(1,struct.pack("!I",len(p))+p)\ntime.sleep(30)\n'
    )
    monkeypatch.setattr(e, "CLI", str(fake))
    with pytest.raises(ValueError, match="session_ready"):
        e.prototype(bundle, tmp_path / "bad")


def test_precondition_gate_and_cli_dispatch(tmp_path, monkeypatch):
    """REQ-VERIFY-8571: absent originals block; explicit CLI routes retain real exit codes."""
    from carnot.reporting import resident_runner_8571 as runner

    work = runner.measure(tmp_path / "missing", tmp_path / "raw")
    assert work["bundle"] is None and work["gate_check_summary"][0]["observed"] is None
    assert work["preconditions_checked"]["free_disk_bytes"] > 0
    assert runner.main(["--deliberate-error"]) == 7
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    monkeypatch.setattr(pytest, "main", lambda arguments: 0)
    assert (
        runner.main(
            [
                "--sealed-tests",
                "unused",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--collect-only",
                "-q",
                e.TEST,
            ]
        )
        == 0
    )
    monkeypatch.setattr(runner, "run", lambda root, output, control: 0)
    assert runner.main(["--output", str(tmp_path / "private.json")]) == 0


def test_replay_primitive_rejects_rehashed_operands(bundle, tmp_path):
    """SCENARIO-VERIFY-8571-REPLAY: independent reconstruction rejects forged ready evidence."""
    primitive = e.prototype(bundle, tmp_path / "prototype")
    mutations = [
        lambda p: p["prototype_request_rows"].pop(),
        lambda p: p["prototype_request_rows"][0].update(unit_id="foreign"),
        lambda p: p["prototype_request_rows"][0].update(arm="native"),
        lambda p: p["prototype_request_rows"][0].update(passed=False),
        lambda p: p["prototype_request_rows"][1]["response"].update(response="stale"),
        lambda p: p["lifecycle_receipts"][0].update(exit_code=1),
        lambda p: p["lifecycle_receipts"][0]["stderr_reference"].update(sha256="changed"),
        lambda p: p["immutable_config_reference"].update(path=str(tmp_path / "missing")),
    ]
    for mutate in mutations:
        changed = deepcopy(primitive)
        mutate(changed)
        assert not e.replay_primitive(changed, bundle)
    config = Path(primitive["immutable_config_reference"]["path"])
    value = json.loads(config.read_bytes())
    value["head"]["temperature"] *= 2
    atomic_json(config, value)
    primitive["immutable_config_reference"] = e.reference(config)
    assert not e.replay_primitive(primitive, bundle)


def test_stalled_pipe_write_is_bounded():
    """REQ-VERIFY-8571: a peer that stops reading cannot block a writer indefinitely."""
    reader, writer = os.pipe()
    os.set_blocking(writer, False)
    try:
        with pytest.raises(BlockingIOError):
            while True:
                os.write(writer, b"x" * 4096)
        with pytest.raises(TimeoutError, match="write_deadline"):
            w.write_frame(writer, dict(status="reply"), timeout=0.001)
    finally:
        os.close(reader)
        os.close(writer)


def test_owned_measurement_failure_is_explicit(tmp_path, monkeypatch):
    """REQ-VERIFY-8571: an owned arithmetic exception is a failed check, never a missing upstream."""
    from carnot.reporting import resident_runner_8571 as runner

    def broken(root):
        raise ValueError("owned_evaluator_failure")

    monkeypatch.setattr(e, "admit", broken)
    work = runner.measure(e.ROOT, tmp_path / "raw")
    assert work["owned_measurement_failure"] == "owned_evaluator_failure"
    monkeypatch.setattr(runner, "measure", lambda root, raw: dict(work, raw=str(raw)))
    original = runner.measure

    def persist(root, raw):
        value = original(root, raw)
        atomic_json(raw / "work.json", value)
        return value

    monkeypatch.setattr(runner, "measure", persist)
    output = tmp_path / (e.NAME + ".json")
    assert runner.run(e.ROOT, output, control=True) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert not value["prototype_ready_score"]


def test_owned_validation_failure_blocks_readiness(tmp_path, monkeypatch):
    """REQ-VERIFY-8571: real publication retains a failed owned validation receipt."""
    from carnot.reporting import resident_runner_8571 as runner

    monkeypatch.setattr(
        runner.validation, "validate", lambda work, raw: [dict(name="failed_owned", passed=False)]
    )
    monkeypatch.setattr(runner.validation, "finish", lambda work, raw: [])
    output = tmp_path / (e.NAME + ".json")
    assert runner.run(e.ROOT, output) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["prototype_ready_score"] == 0
    assert any(
        r["name"] == "failed_owned" and not r["passed"] for r in value["validation_receipts"]
    )
