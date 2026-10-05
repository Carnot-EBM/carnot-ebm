"""REQ-VERIFY-8165 / REQ-REPORT-8165: qualify methods without natural credit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import learning_qualification_8165 as e


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Keep the full measured fixture private so it cannot change historical results."""
    raw = tmp_path_factory.mktemp("qualification8165")
    return e.measure(e.ROOT, raw, fixture_mode=True), raw


def test_qualification_and_negative_replay(measured, tmp_path):
    """SCENARIO-VERIFY-8165-REPLAY: rehashing does not authorize forged evidence."""
    work, raw = measured
    value = e.build(work, raw, [dict(passed=True, normal_exit=True)])
    assert value["experiment_id"] == 8165 and value["verdict_class"] == "circular_positive"
    assert value["learning_protocol_ready_score"] == value["future_exposure_fixture_score"] == 1
    assert Path(value["protocol_path"]).read_bytes() == (e.ROOT / e.legacy.PROTOCOL).read_bytes()
    assert value["protocol_sha256"] == e.legacy.PROTOCOL_HASH
    assert value["MODEL_SPECS"] == value["call_ledger"] == []
    assert not any(value["model_invocation_counts"].values())
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert value["fixture_rows"] == value["event_reference_rows"]
    assert e.build(work, raw, [dict(passed=False)])["verdict_class"] == "disqualified"
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    for field, changed in [
        ("reproducibility_checksum", "malformed"),
        ("protocol_sha256", "sha256:wrong"),
        ("experiment_id", 8152),
        ("fixture_rows", []),
        ("rows", [dict(value["rows"][0], numerator=42), *value["rows"][1:]]),
    ]:
        forged = deepcopy(value)
        forged[field] = changed
        if field != "reproducibility_checksum":
            forged.pop("reproducibility_checksum")
            forged["reproducibility_checksum"] = canonical_hash(forged)
        atomic_json(path, forged)
        assert not e.replay(path), field
    path.write_text("{}")
    assert not e.replay(path)


def test_real_custody(tmp_path):
    """REQ-VERIFY-8165: retain original custody and the disqualified predecessor."""
    work = e.measure(e.ROOT, tmp_path)
    assert work["input_ready"] == 1, [r for r in work["gate_check_summary"] if not r["passed"]]
    original = json.loads((e.ROOT / e.legacy.engine.historical.UPSTREAM).read_text())
    assert work["original_slot_mask"] == original["original_slot_mask"]
    assert len(work["original_slot_mask"]["stream"]) == 256
    assert len(work["original_slot_mask"]["retention"]) == 64
    assert work["predecessor_disposition"]["verdict_class"] == "disqualified"
    assert work["predecessor_disposition"]["failed_checks"] == ["coverage_report"]
    assert {r["experiment_id"] for r in work["cited_upstream_artifacts"]} >= {8111, 8152}


def test_direct_script_private_routes(tmp_path):
    """SCENARIO-REPORT-8165-CLI: real success/block/tamper/replay run outside checkout."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    if os.environ.get("COVERAGE_RCFILE"):
        env["COVERAGE_PROCESS_START"] = os.environ["COVERAGE_RCFILE"]
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]

    def call(*args, expected=0):
        e.progress("before_private_subprocess")
        log = tmp_path / (str(time.time_ns()) + ".log")
        with log.open("w") as stream:
            child = subprocess.Popen(
                [*argv, *args], cwd=tmp_path, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
            started = time.monotonic()
            while child.poll() is None:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    e.progress("private_child_wait", 0, 1)
                    if time.monotonic() - started > 240:
                        child.kill()
                        child.wait()
                        pytest.fail(log.read_text())
        e.progress("after_private_subprocess", int(child.returncode == expected))
        assert child.returncode == expected, log.read_text()

    output = tmp_path / (e.NAME + ".json")
    call("--fixture-output", str(output))
    call("--cold-replay", str(output))
    blocked = tmp_path / "blocked" / output.name
    call("--fixture-output", str(blocked), "--root", str(tmp_path / "missing"))
    value = json.loads(blocked.read_text())
    assert value["verdict_class"] == "blocked" and value["learning_protocol_ready_score"] == 0
    assert next(r for r in value["gate_check_summary"] if not r["passed"])["observed"] is False
    call("--cold-replay", str(blocked))
    output.write_text("{}")
    call("--cold-replay", str(output), expected=1)
    call("--fixture-output", str(e.ROOT / "results" / output.name), expected=2)


def test_supervisor_and_manifest(tmp_path, monkeypatch, measured):
    """REQ-REPORT-8165: frozen checks measure all owned statements and real paths."""
    from carnot.reporting import learning_qualification_execution_8165 as runner

    work = deepcopy(measured[0])
    specs = runner.manifest(tmp_path, tmp_path / "candidate.json")
    unit = next(r for r in specs["commands"] if r["name"] == "owned_unit_and_private_CLI")
    assert e.TEST in unit["argv"]
    assert e.legacy.TEST + "::test_exact_checksum_rejection" in unit["argv"]
    config = (tmp_path / "coverage.ini").read_text()
    assert all(str(e.ROOT / p) in config for p in runner.OWNED)
    assert e.legacy.MODULE not in config
    for spec in specs["commands"]:
        if spec["name"] in {"ruff_check", "ruff_format", "strict_mypy", "spec_coverage"}:
            assert not any("::" in arg for arg in spec["argv"])
    assert (
        "--fail-under=100"
        in next(r for r in specs["commands"] if r["name"] == "coverage_report")["argv"]
    )
    monkeypatch.setattr(
        runner, "run_check", lambda *a, **kw: dict(actual_exit=-9, timed_out=True, passed=False)
    )
    assert not runner.check(dict(name="killed"), tmp_path, tmp_path)["normal_exit"]
    monkeypatch.setattr(e, "measure", lambda *a, **kw: deepcopy(work))

    def check(spec, private, raw):
        if spec["name"] == "measurement":
            target = Path(spec["argv"][spec["argv"].index("--worker-output") + 1])
            atomic_json(target, work)
        return dict(
            name=spec["name"],
            passed=spec["name"] != "adversarial_verify",
            actual_exit=1 if spec["name"] == "adversarial_verify" else 0,
            normal_exit=True,
        )

    monkeypatch.setattr(runner, "check", check)
    monkeypatch.setattr(e, "replay", lambda p: True)
    captured = []

    def publish(output, value, validate):
        captured.append(value)
        return dict(output=str(output), validation=validate(output))

    monkeypatch.setattr(runner, "publish_primary", publish)
    assert runner.main(["--output", str(tmp_path / (e.NAME + ".json"))]) == 0
    assert captured[-1]["verdict_class"] == "disqualified"
    assert captured[-1]["learning_protocol_ready_score"] == 0
    assert runner.main(["--worker-output", str(tmp_path / "worker.json")]) == 0
