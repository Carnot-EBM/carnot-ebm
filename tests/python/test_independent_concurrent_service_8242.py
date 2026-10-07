"""REQ-VERIFY-8242 / REQ-REPORT-8242: all fixture writes stay in private scratch."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import independent_concurrent_service_8242 as report
from carnot.verify import independent_concurrent_service_8242 as measure
from carnot.verify import concurrency_canary_8227 as e


def protocol():
    """REQ-VERIFY-8242: read frozen obligations without borrowing their responses."""
    return json.loads((e.ROOT / report.BINDINGS).read_text())


def work_fixture():
    """SCENARIO-VERIFY-8242-MEASURE: repeated sweeps retain only24 source identities."""
    rows = []
    for i, original in enumerate(protocol()["benchmark"]):
        row = deepcopy(original)
        row.update(
            request_id=f"8242-{i}",
            status="completed",
            response_id=f"response-{i}",
            latency_s=2.0 if row["arm"] == "serial" else 1.0,
            acquisition_s=0.9,
            scoring_s=0.01,
            service_s=0.1,
            result=dict(
                usage=dict(prompt_tokens=10, completion_tokens=4),
                choices=[dict(message=dict(content="0|E|0.1|[0]"))],
            ),
        )
        rows.append(row)
    return dict(
        rows=rows,
        workloads=[
            dict(workload=w, warm_s=24.0, cold_s=30.0, startup_s=4.0, shutdown_s=2.0)
            for w in protocol()["launch_order"]
        ],
        loads=[dict(completed=True)] * 4,
        checks=[],
        phase_spans=[],
        telemetry=[],
    )


def test_reduction_support_and_completion_safety():
    """SCENARIO-VERIFY-8242-MEASURE: a shorter arm with fewer completions cannot win."""
    work = work_fixture()
    value = measure.reduce(work)
    assert value["completed_count"] == 96 and value["independent_count"] == 24
    assert value["latency_intervals"]["supported_sources"] == 24
    assert value["latency_intervals"]["mean_paired_delta_s"] == 1
    assert value["throughput_population_interval"] is None
    assert value["acquisition_fraction"] == pytest.approx(0.9)
    assert value["scoring_speedup_upper_bound"] == pytest.approx(1 / 0.99)
    assert value["nfr01_met"] is None
    for row in work["rows"]:
        if row["arm"] == "concurrent" and row["source_cluster_id"] in {
            r["source_cluster_id"] for r in work["rows"][:5]
        }:
            row["status"] = "error"
    value = measure.reduce(work)
    assert value["latency_intervals"] is None and not value["improvement_qualified"]
    assert value["failed_count"] == 10
    assert measure.reduce(dict(rows=[], workloads=[]))["acquisition_fraction"] is None


def test_blocked_and_owned_failure(tmp_path):
    """REQ-REPORT-8242: missing evidence earns zero calls; owned failure disqualifies."""
    data = report.inputs(tmp_path, tmp_path / "authenticated")
    assert not data["ready"] and data["checks"][0]["observed"] is None
    data["protocol"] = protocol()
    value = report.build(data, {}, tmp_path, [dict(passed=True, normal_exit=True)], 1)
    assert value["verdict_class"] == "blocked" and value["intended_count"] == 96
    assert value["inference_substrate_class"] == "no_model_load"
    assert value["model_invocation_counts"]["generation_calls"] == 0
    assert value["concurrent_service_ready_score"] == 0
    assert len(value["field_principles"]) >= len(value) - 1
    value = report.build(data, {}, tmp_path, [dict(passed=False, normal_exit=True)], 1)
    assert value["verdict_class"] == "disqualified"
    assert not report.replay(tmp_path / "missing.json")


def cli(*args):
    """SCENARIO-REPORT-8242-CLI: real children import this checkout from private cwd."""
    return subprocess.run(
        [sys.executable, str(e.ROOT / report.CLI), *map(str, args)],
        cwd="/tmp",
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_private_cli_replay_and_tampering(tmp_path):
    """SCENARIO-REPORT-8242-CLI: blocked publication and rehashed negative controls."""
    output = tmp_path / (report.NAME + ".json")
    child = cli("--root", tmp_path / "absent", "--output", output)
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert cli("--cold-replay", output).returncode == 0
    changed = deepcopy(value)
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = report.checksum(changed)
    e.atomic_json(output, changed)
    assert cli("--cold-replay", output).returncode == 1
    assert cli("--date", "20261006").returncode == 2
    assert cli("--fixture-e2e", e.ROOT / "results/bad.json").returncode == 2


@pytest.fixture(scope="module")
def authenticated(tmp_path_factory):
    """REQ-VERIFY-8242: use private copies of real qualified operands for native joins."""
    data = report.inputs(e.ROOT, tmp_path_factory.mktemp("authenticated-8242"))
    assert data["ready"], data["checks"]
    return data


def test_authenticated_inputs_and_native_persistence(authenticated, tmp_path):
    """SCENARIO-VERIFY-8242-MEASURE: unchanged Rust and real fsync commit this request."""
    data = authenticated
    row = data["protocol"]["benchmark"][1]
    native, _ = measure.service.prior.shared.host.old.prior.old.host.load_binding(data)
    response = measure.service.FixtureRuntime().generate(row["envelope"]["payload"])
    branch = measure.persist(data, row, response, native, tmp_path)
    assert branch["state"]["records"][0]["request_id"] == row["request_id"]
    assert branch["stages"]["commit_fsync_ns"] > 0


@pytest.mark.parametrize(
    "mode",
    ["valid", "lease", "load", "slots", "cleanup", "deadline", "generate", "service", "payload"],
)
def test_measurement_obligations(tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8242-MEASURE: real queues retain failures and each cold workload."""
    from test_concurrency_canary_8227 import fake_live

    p, resources = fake_live(tmp_path, monkeypatch, mode)
    runtime = measure.runtime
    original_execute = runtime.execute

    def execute(commands, raw):
        raw.mkdir(parents=True, exist_ok=True)
        return original_execute(commands, raw)

    monkeypatch.setattr(runtime, "execute", execute)
    monkeypatch.setattr(runtime, "command", lambda *a: ["fixture", "--port", "100"])
    monkeypatch.setattr(
        measure.service.prior.shared.host.old.prior.old.host,
        "load_binding",
        lambda data: (None, dict(fixture=True)),
    )
    monkeypatch.setattr(
        measure, "persist", lambda *a: dict(start_ns=1, end_ns=2, stages=dict(scoring_ns=1))
    )
    if mode == "generate":
        monkeypatch.setattr(runtime, "generate", lambda *a: {})
    if mode == "service":

        def fail_persist(*args):
            raise ValueError("private invalid normalization")

        monkeypatch.setattr(measure, "persist", fail_persist)
    if mode == "payload":
        acquire = e.acquire
        monkeypatch.setattr(
            e,
            "acquire",
            lambda rows, generate, *a, **kw: acquire(
                rows, lambda payload, slot: generate(dict(payload, changed=True), slot), *a, **kw
            ),
        )
    data = dict(protocol=p)
    work = measure.measure(
        data, resources, tmp_path / "measurement", cap_s=0 if mode == "deadline" else 3000
    )
    assert len(work["rows"]) == 96
    if mode == "valid":
        assert measure.reduce(work)["completed_count"] == 96
        assert len(work["loads"]) == len(work["workloads"]) == 4
        assert len({r["request_id"] for r in work["rows"]}) == 96
        assert (
            report.build(
                dict(data, ready=True, checks=[]),
                work,
                tmp_path,
                [dict(passed=True, normal_exit=True)],
                20,
            )["model_invocation_counts"]["generation_calls"]
            == 96
        )
        replay_data = dict(data, ready=True, checks=[], refs=[], code=[])
        value = report.build(replay_data, work, tmp_path, [dict(passed=True, normal_exit=True)], 20)
        e.atomic_json(tmp_path / "data.json", replay_data)
        e.atomic_json(tmp_path / "work.json", work)
        value["reproducibility_checksum"] = report.checksum(value)
        candidate = tmp_path / (report.NAME + ".json")
        e.atomic_json(candidate, value)
        assert report.replay(candidate)
        original_result = deepcopy(work["rows"][0]["result"])
        work["rows"][0]["result"]["id"] = "tampered"
        e.atomic_json(tmp_path / "work.json", work)
        assert not report.replay(candidate)
        work["rows"][0]["result"] = original_result
        original_latency = work["rows"][0]["latency_s"]
        work["rows"][0]["latency_s"] += 1
        e.atomic_json(tmp_path / "work.json", work)
        assert not report.replay(candidate)
        work["rows"][0]["latency_s"] = original_latency
        work["rows"][0]["clocks"]["issue"] += 1
        e.atomic_json(tmp_path / "work.json", work)
        assert not report.replay(candidate)
    elif mode in {"lease", "load", "slots", "cleanup"}:
        assert any(not c["passed"] for c in work["checks"])
    else:
        assert measure.reduce(work)["completed_count"] == 0


@pytest.mark.parametrize("mode", ["valid", "resource", "owned", "terminal", "fixture"])
def test_owned_orchestration(tmp_path, monkeypatch, mode):
    """REQ-REPORT-8242: no resource acquisition precedes owned checks; failed publication stays private."""
    from carnot.reporting import independent_concurrent_execution_8242 as run

    data = dict(ready=True, protocol=protocol(), checks=[], refs=[], code=[])
    monkeypatch.setattr(report, "inputs", lambda *a: deepcopy(data))
    sequence = []

    def plan(private):
        e.atomic_json(
            private / "coverage.json", dict(totals=dict(num_statements=1, covered_lines=1))
        )
        return [report.recorded.CommandSpec("owned", ("true",), "owned", 5)]

    def execute(commands, raw):
        sequence.extend(c.name for c in commands)
        return [
            dict(
                name=c.name,
                normal_exit=True,
                passed=not (
                    mode == "owned"
                    and c.scope == "owned"
                    or mode == "terminal"
                    and c.scope == "terminal"
                ),
            )
            for c in commands
        ]

    def preflight(*args):
        assert "owned" in sequence
        return dict(checks=[report.operand("gpu", tmp_path, True, mode != "resource")], receipts=[])

    monkeypatch.setattr(report, "validation_plan", plan)
    monkeypatch.setattr(run, "execute", execute)
    monkeypatch.setattr(run.runtime, "preflight", preflight)
    monkeypatch.setattr(report.measured, "measure", lambda *a: work_fixture())
    output = tmp_path / (report.NAME + ".json")
    e.atomic_json(output, dict(experiment_id=8242, task_id=report.TASK))
    argv = ["--fixture-e2e" if mode == "fixture" else "--output", str(output)]
    assert run.main(argv) == (1 if mode == "terminal" else 0)
    if mode != "terminal":
        value = json.loads(output.read_text())
        assert value["coverage_statement_counts"]["covered_lines"] == 1
        assert value["verdict_class"] == (
            "disqualified" if mode == "owned" else "blocked" if mode == "resource" else "null"
        )


@pytest.mark.parametrize("mode", ["valid", "schema", "hash", "terminal", "missing", "envelope"])
def test_private_authentication(tmp_path, monkeypatch, mode):
    """REQ-VERIFY-8242: changed external operands block with their exact original gate."""
    p = protocol()
    path = tmp_path / "protocol.json"
    if mode == "schema":
        p["schema"] = "changed"
    if mode == "envelope":
        p["benchmark"][0]["envelope"]["payload"]["seed"] += 1
    e.atomic_json(path, p)
    out, err = tmp_path / "out.log", tmp_path / "err.log"
    out.write_text("exit zero")
    err.write_text("")
    receipts = [
        dict(
            passed=True,
            normal_exit=True,
            stdout_path=str(out),
            stderr_path=str(err),
            stdout_sha256=e.sha256_file(out),
            stderr_sha256=e.sha256_file(err),
        )
    ]
    terminal_path = tmp_path / "terminal.json"
    primary = tmp_path / report.UPSTREAM
    value = dict(
        experiment_id=8236,
        concurrent_canary_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        concurrent_protocol_path=str(path),
        concurrent_protocol_sha256=e.sha256_file(path),
        terminal_validation_sidecar_path=str(terminal_path),
        raw_shard_hashes=[],
        code_config_hashes=[],
    )
    e.atomic_json(primary, value)
    digest = e.sha256_file(primary)
    e.atomic_json(terminal_path, dict(receipts=receipts, candidate_sha256=digest))
    bound = primary.parent / "raw" / primary.stem / "validators" / (digest.split(":")[1] + ".json")
    e.atomic_json(bound, dict(primary_sha256=digest, report=dict(receipts=receipts)))
    monkeypatch.setattr(report, "PIN", digest)
    monkeypatch.setattr(report.recorded, "inputs", lambda *a: dict(ready=True, checks=[], refs=[]))
    if mode == "hash":
        out.write_text("changed exit")
    if mode == "terminal":
        e.atomic_json(terminal_path, dict(receipts=[], candidate_sha256=digest))
    if mode == "missing":
        path.unlink()
    data = report.inputs(tmp_path, tmp_path / "custody")
    assert data["ready"] == (mode == "valid")
    if mode != "valid":
        assert any(not c["passed"] for c in data["checks"])


def test_replay_primitive_hash_and_checksum(tmp_path):
    """SCENARIO-REPORT-8242-CLI: primitive drift and bad checksums are separate failures."""
    output = tmp_path / (report.NAME + ".json")
    child = cli("--root", tmp_path / "missing", "--output", output)
    assert child.returncode == 0, child.stderr
    original = json.loads(output.read_text())
    wrong = dict(original, reproducibility_checksum="wrong")
    e.atomic_json(output, wrong)
    assert not report.replay(output)
    e.atomic_json(output, original)
    Path(original["replay_inputs"]["work_path"]).write_text("changed")
    assert not report.replay(output)


def test_private_reduction_replay(tmp_path):
    """REQ-REPORT-8242: private reducer controls do not receive live model credit."""
    data = dict(ready=True, protocol=protocol(), checks=[], refs=[], code=[])
    work = work_fixture()
    e.atomic_json(tmp_path / "data.json", data)
    e.atomic_json(tmp_path / "work.json", work)
    value = report.build(data, work, tmp_path, [dict(passed=True, normal_exit=True)], 20)
    value["reproducibility_checksum"] = report.checksum(value)
    output = tmp_path / (report.NAME + ".json")
    e.atomic_json(output, value)
    assert report.replay(output)
    assert value["model_invocation_counts"]["generation_calls"] == 0
