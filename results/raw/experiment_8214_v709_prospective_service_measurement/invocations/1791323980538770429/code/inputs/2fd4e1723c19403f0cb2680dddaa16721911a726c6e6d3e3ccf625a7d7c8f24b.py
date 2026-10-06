"""REQ-VERIFY-8214 / REQ-REPORT-8214: current work needs primitive custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.verify import prospective_service_8214 as e


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """SCENARIO-VERIFY-8214-MEASUREMENT: preserve the original scheduled sources."""
    value = e.inputs(e.ROOT, tmp_path_factory.mktemp("inputs-8214"))
    assert value["ready"] and len(value["schedule"]["rows"]) == 24
    return value


@pytest.fixture(scope="module")
def measured(data, tmp_path_factory):
    """SCENARIO-VERIFY-8214-MEASUREMENT: measure real storage once for primitive mutations."""
    return e.measure(data, e.FixtureRuntime(), tmp_path_factory.mktemp("pairs-8214"))


def test_measure_and_reduction(data, measured, tmp_path):
    """SCENARIO-VERIFY-8214-MEASUREMENT: real durable branches share acquired bytes."""
    work = measured
    reduced = e.reduce(work)
    assert reduced["independent_count"] == 21 and reduced["completed_count"] == 21
    assert reduced["paired_ratio_ci95"]["sources"] == 21
    assert reduced["complete_workload_totals"][e.ARMS[0]] > work["startup_s"]
    assert reduced["amdahl_ceiling"]["maximum_speedup"] >= 1
    assert e.validate_work(data, work)
    changed = deepcopy(work)
    changed["requests"][0]["arms"][0]["probability"] += 0.1
    assert not e.validate_work(data, changed)
    assert e.reduce({})["paired_ratio_ci95"] is None


@pytest.mark.parametrize("mode", ["error", "malformed", "truncated", "parity"])
def test_failures(data, tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8214-MEASUREMENT: failures never become manufactured features."""
    runtime = e.FixtureRuntime(mode)
    if mode == "parity":
        old = e.prior.commit_group

        def wrong(*args):
            old(*args)
            if args[1][0]["arm"] == e.ARMS[1]:
                args[1][0]["probability"] += 0.1

        monkeypatch.setattr(e.prior, "commit_group", wrong)
    value = e.measure(data, runtime, tmp_path)
    assert e.reduce(value)["failed_count"] == 24
    assert e.reduce(value)["independent_count"] == 0
    assert len(value["requests"]) == 24


def test_deadline_and_build(data, tmp_path):
    """REQ-REPORT-8214: insufficient support blocks; failed owned validation disqualifies."""
    work = e.measure(data, e.FixtureRuntime(), tmp_path / "deadline", deadline_s=0)
    assert e.reduce(work)["censored_count"] == 24
    value = e.build(data, {"work": work}, tmp_path, [{"passed": True}], 1, False)
    assert value["honest_verdict"] == "complete_blocked_support"
    assert value["service_measurement_ready_score"] == 0
    value = e.build(data, {}, tmp_path, [{"passed": False}], 1, False)
    assert value["verdict_class"] == "disqualified"
    blocked = e.inputs(tmp_path / "missing", tmp_path / "blocked")
    assert not blocked["ready"]
    value = e.build(blocked, {}, tmp_path, [{"passed": True}], 1, False)
    assert value["verdict_class"] == "blocked"


def cli(*args):
    """SCENARIO-REPORT-8214-CLI: invoke the executable from outside the checkout."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, str(e.ROOT / e.CLI), *map(str, args)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_cli(tmp_path):
    """SCENARIO-REPORT-8214-CLI: fixture bytes replay, changed claims fail."""
    output = tmp_path / (e.NAME + ".json")
    result = cli("--fixture-e2e", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0
    assert value["nfr01_met"] is False
    assert cli("--cold-replay", output).returncode == 0
    value["completed_count"] -= 1
    value["reproducibility_checksum"] = e.checksum(value)
    output.write_text(json.dumps(value))
    assert cli("--cold-replay", output).returncode == 1
    assert cli("--date", "20000101").returncode == 2
    assert cli("--fixture-e2e", e.ROOT / "results" / (e.NAME + ".json")).returncode == 2
    missing = tmp_path / "missing" / (e.NAME + ".json")
    assert cli("--root", tmp_path / "absent", "--fixture-e2e", missing).returncode == 0
    assert json.loads(missing.read_text())["verdict_class"] == "blocked"
    assert not e.replay(tmp_path / "absent.json")


def test_live_adapter_and_primary_paths(data, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8214-CLI: exercise supervisor adapter without model credit."""
    from types import SimpleNamespace
    from carnot.reporting import prospective_service_execution_8214 as runner

    legacy = e.prior.shared.prior.qualified.legacy
    template = "private scripted template"
    private_data = deepcopy(data)
    private_data["identity"]["chat_template_sha256"] = e.key(template)
    monkeypatch.setattr(runner, "get_json", lambda _: {"chat_template": template})

    def supervised(plan, raw, private):
        slots = legacy.capture.freeze({})
        assert len(slots) == 24
        assert legacy.load_public({}) == {}
        assert legacy.TASK == e.TASK
        runtime = e.FixtureRuntime()
        runtime.port = 1
        rows = legacy.capture.capture(
            slots, runtime, raw / "slots", "fixture", started=e.time.monotonic()
        )
        assert len(rows) == 24
        return dict(
            model_loads_completed=1,
            model_loads_attempted=1,
            model_identity_receipt={"duration_s": 1},
            native_binary={"sha256": data["identity"]["runtime_sha256"]},
            checks=[],
        )

    monkeypatch.setattr(legacy, "live_capture", supervised)
    result = runner.live(private_data, tmp_path / "adapter", tmp_path)
    assert result["work"]["startup_s"] >= 1
    value = e.build(private_data, result, tmp_path, [{"passed": True}], 20, False)
    assert value["verdict_class"] == "blocked"
    result["runtime_receipts"] = [dict(started_monotonic_ns=0, ended_monotonic_ns=11_000_000_000)]
    assert (
        e.build(private_data, result, tmp_path, [{"passed": True}], 20, False)[
            "service_measurement_ready_score"
        ]
        == 1
    )
    changed = deepcopy(private_data)
    changed["identity"]["runtime_sha256"] = "absent"
    assert not runner.live(changed, tmp_path, tmp_path)["checks"][0]["passed"]
    changed = deepcopy(private_data)
    changed["identity"]["chat_template_sha256"] = "changed"
    with pytest.raises(ValueError, match="served_chat_template"):
        runner.live(changed, tmp_path / "bad-template", tmp_path)
    assert e.validate_work(
        data, e.measure(data, e.FixtureRuntime(), tmp_path / "censored", deadline_s=0)
    )


def test_main_owned_failure_and_validator_rejection(tmp_path, monkeypatch):
    """REQ-REPORT-8214: owned failure and auditor rejection cannot publish readiness."""
    from carnot.reporting import prospective_service_execution_8214 as runner

    original = runner.execute
    monkeypatch.setattr(runner, "validation_plan", lambda p: [])

    def failed(plan, raw):
        if raw.name == "terminal":
            return original(plan, raw)
        if raw.name == "resources":
            return []
        return [{"passed": False}]

    monkeypatch.setattr(runner, "execute", failed)
    output = tmp_path / "disqualified" / (e.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    monkeypatch.setattr(runner, "execute", lambda plan, raw: [{"passed": False}])
    assert runner.main(["--fixture-e2e", str(tmp_path / "rejected" / (e.NAME + ".json"))]) == 1


def test_main_normal_flow_without_live_resource_credit(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8214-CLI: normal routing preserves an honest support block."""
    from carnot.reporting import prospective_service_execution_8214 as runner

    original = runner.execute
    monkeypatch.setattr(runner, "validation_plan", lambda p: [])
    monkeypatch.setattr(runner, "live", lambda *args: {})
    monkeypatch.setattr(
        runner,
        "execute",
        lambda plan, raw: (
            original(plan, raw)
            if raw.name == "terminal"
            else []
            if raw.name == "resources"
            else [{"passed": True}]
        ),
    )
    output = tmp_path / (e.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "blocked"


def test_missing_slots_and_parity_gate(data, measured, tmp_path):
    """REQ-REPORT-8214: missing slots remain explicit; one native mismatch forbids readiness."""
    value = e.build(data, {}, tmp_path, [{"passed": True}], 1, False)
    assert len(value["rows"]) == 24 and value["excluded_count"] == 24
    work = deepcopy(measured)
    next(r for r in work["requests"] if r["status"] == "completed")["native_parity_failed"] = True
    result = dict(
        work=work,
        model_loads_completed=1,
        runtime_receipts=[dict(started_monotonic_ns=0, ended_monotonic_ns=11_000_000_000)],
    )
    value = e.build(data, result, tmp_path, [{"passed": True}], 20, False)
    assert (
        value["verdict_class"] == "disqualified" and value["service_measurement_ready_score"] == 0
    )


def test_primitive_mutations(data, measured, tmp_path):
    """SCENARIO-VERIFY-8214-MEASUREMENT: rehashed clocks and typed state need independent checks."""
    for mutation in ("envelope", "terminal", "state", "clock"):
        changed = deepcopy(measured)
        row = next(r for r in changed["requests"] if r["status"] == "completed")
        if mutation == "envelope":
            row["envelope"]["condition"] = "changed"
        elif mutation == "terminal":
            row["terminal"]["result"] = {"changed": True}
        elif mutation == "state":
            row["arms"][0]["state"]["records"][0]["action"] = "changed"
        else:
            row["arms"][0]["conversion_end_ns"] = row["arms"][0]["start_ns"] - 1
        assert not e.validate_work(data, changed)
    raw = tmp_path / "raw"
    raw.mkdir()
    result = {"work": measured}
    e.atomic_json(raw / "result.json", result)
    artifact = e.build(data, result, raw, [{"passed": True}], 1, True)
    artifact["raw_shard_hashes"] = [e.reference(raw / "result.json")]
    output = tmp_path / "tamper.json"
    artifact["reproducibility_checksum"] = "wrong"
    output.write_text(json.dumps(artifact))
    assert not e.replay(output)
    artifact["raw_shard_hashes"][0]["sha256"] = "changed"
    artifact["reproducibility_checksum"] = e.checksum(artifact)
    output.write_text(json.dumps(artifact))
    assert not e.replay(output)
    changed = deepcopy(measured)
    changed["requests"][0]["envelope"]["condition"] = "changed"
    e.atomic_json(raw / "result.json", {"work": changed})
    artifact["raw_shard_hashes"] = [e.reference(raw / "result.json")]
    artifact["reproducibility_checksum"] = e.checksum(artifact)
    output.write_text(json.dumps(artifact))
    assert not e.replay(output)


def test_unavailable_recorder_configuration(data, tmp_path, monkeypatch):
    """REQ-REPORT-8214: missing and zero recorder readiness remain different operands."""
    root = tmp_path / "root"
    primary = root / "results/experiment_8213_v709_prospective_request_recorder.json"
    primary.parent.mkdir(parents=True)
    original = deepcopy(data)
    original["input_path"] = str(tmp_path / "input.json")
    monkeypatch.setattr(e.qualified, "inputs", lambda *args: deepcopy(original))
    for observed in (None, 0):
        primary.write_text(json.dumps(dict(request_recorder_ready_score=observed)))
        value = e.inputs(root, tmp_path / "raw")
        assert not value["ready"]
        assert (
            next(c for c in value["checks"] if c["check"] == "request_recorder_ready_score")[
                "observed"
            ]
            == observed
        )
