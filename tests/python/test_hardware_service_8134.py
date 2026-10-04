"""REQ-REPORT-8134, REQ-VERIFY-8134: private cost, precision and custody checks."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from carnot.reporting import hardware_service_8134 as h
from carnot import experiment_8134_v703_hardware_service_boundary as cli
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting import hardware_service_inputs_8134 as inputs
from carnot.reporting.evidence_features_custody_7980 import reference


def fixture():
    return dict(
        fixture=True,
        boards=[
            dict(board=n, custody_valid=True, processor_class=c[0], k_max=c[1], blocker=c[2])
            for n, c in h.prior.CONTRACTS.items()
        ],
        systems=[],
        checks=[],
        references=[],
        pairs=[
            dict(
                unit_id="p",
                batch=1,
                condition="warm",
                status="completed",
                arms=[
                    dict(
                        arm="python_batch",
                        full_latency_ns=1000,
                        arithmetic_ns=10,
                        residual_ns=0,
                        source_cluster_ids=["source"],
                        components={
                            "arithmetic_and_boundary_ns": 10,
                            "memory_ns": 690,
                            "durable_write_ns": 300,
                        },
                    )
                ],
            )
        ],
        modeled=[
            dict(
                unit_id="p",
                arm="python_batch",
                matched=True,
                historical_acquisition_s=1,
                historical_load_s=2,
                no_reuse_acquisition_s=4,
            )
        ],
        host_qualified=True,
        complete_service=False,
        cited=[],
    )


def test_cost_bounds_and_missing_operands():
    """REQ-VERIFY-8134: outer ceilings cannot spend missing costs as zero."""
    data = fixture()
    rows = h.cost_rows(data)
    assert rows[0]["outer_ceiling"] == pytest.approx(1000 / 990)
    assert all(
        r["target_100x"] == "ruled_out_even_by_outer_ceiling"
        for r in rows
        if r["status"] == "completed"
    )
    assert rows[-1]["status"] == "excluded"
    data["pairs"][0]["arms"][0]["components"]["memory_ns"] = None
    assert h.cost_rows(data)[0]["status"] == "excluded"
    data = fixture()
    data["pairs"][0]["arms"][0]["residual_ns"] = 1
    assert h.cost_rows(data)[0]["status"] == "excluded"
    data = fixture()
    data["modeled"] = []
    assert h.cost_rows(data)[1]["status"] == "excluded"
    data = fixture()
    data["pairs"][0]["arms"][0]["arithmetic_ns"] = 999
    data["pairs"][0]["arms"][0]["components"] = dict(arithmetic_and_boundary_ns=999, other=1)
    assert h.cost_rows(data)[0]["target_100x"] == "not_established"


def test_operations_precision_and_board_independence():
    """REQ-VERIFY-8134: byte counts and guarded decisions retain fixture scope."""
    data = fixture()
    result = h.reduce(data)
    assert result["verdict_class"] == "blocked"
    assert result["hardware_boundary_ready_score"] == 1
    assert len(result["workload_operation_rows"]) == 78
    r = result["workload_operation_rows"][0]
    assert r["distance_coordinate_terms"] == 144
    assert r["pending_event_capacity"] == 32
    assert r["pending_event_numeric_bytes"] == 32 * (9 * 8 + 8 + 8)
    data["boards"][0]["k_max"] = 6
    result = h.reduce(data)
    assert not result["board_rows"][0]["custody_valid"]
    assert result["board_rows"][1]["custody_valid"]
    data["host_qualified"] = False
    assert h.reduce(data)["measured_workload_bound_ready_score"] == 0
    data = fixture()
    data["complete_service"] = True
    assert h.reduce(data)["verdict_class"] == "circular_positive"
    data["fixture"] = False
    assert h.reduce(data)["verdict_class"] == "null"
    system = dict(
        seed=1,
        state=dict(
            geometry=dict(mean=[0] * 9, std=[1] * 9, sigma=1),
            centers=[dict(x=[0] * 9) for _ in range(16)],
            coefficients=[0] * 17,
        ),
        x=[[0] * 9, [5] * 9],
    )
    data["systems"] = [system]
    result = h.reduce(data)
    assert all(r["action_disagreements"] == 0 for r in result["quantization_rows"])
    assert all(all(r["fallback_flags"]) for r in result["quantization_rows"])


def invoke(args, tmp, expected=0):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    prefix = [str(cli.ROOT / ".venv/bin/python")]
    cfg = env.get("CARNOT_8134_COVERAGE_CONFIG")
    if cfg:
        prefix += ["-m", "coverage", "run", "--parallel-mode", "--rcfile=" + cfg]
    began = time.monotonic()
    result = subprocess.run(
        [*prefix, str(cli.ROOT / cli.SCRIPT), *args],
        cwd=tmp,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    receipt_path = env.get("CARNOT_8134_CLI_RECEIPTS")
    if receipt_path:
        path = Path(receipt_path)
        rows = json.loads(path.read_bytes())["rows"] if path.exists() else []
        log = path.with_name(f"private-cli-{len(rows)}.log")
        log.write_text(result.stdout + result.stderr)
        rows.append(
            dict(
                command_argv=result.args,
                expected_exit=expected,
                exit_code=result.returncode,
                normal_exit=result.returncode >= 0,
                duration_s=time.monotonic() - began,
                log_sha256=reference(log)["sha256"],
                transcript=result.stdout + result.stderr,
            )
        )
        atomic_json(path, dict(rows=rows))
    return result


def test_private_cli_success_block_mutation_and_replay(tmp_path):
    """SCENARIO-REPORT-8134: outside-checkout CLI, block and cold replay routes."""
    source = tmp_path / "input.json"
    output = tmp_path / (cli.NAME + ".json")
    data = fixture()
    data["complete_service"] = True
    atomic_json(source, data)
    r = invoke(["--input", str(source), "--output", str(output)], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert invoke(["--cold-replay", str(output)], tmp_path).returncode == 0
    value = json.loads(output.read_bytes())
    value["amdahl_bounds"][0]["outer_ceiling"] = 2
    atomic_json(output, value)
    assert invoke(["--cold-replay", str(output)], tmp_path, 1).returncode == 1
    data = fixture()
    atomic_json(source, data)
    assert invoke(["--input", str(source), "--output", str(output)], tmp_path).returncode == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    data["boards"][0]["k_max"] = 6
    atomic_json(source, data)
    assert invoke(["--input", str(source), "--output", str(output)], tmp_path).returncode == 0
    assert not json.loads(output.read_bytes())["board_rows"][0]["custody_valid"]
    source.write_text("broken")
    assert invoke(["--input", str(source), "--output", str(output)], tmp_path, 1).returncode == 1


def test_input_custody_missing_and_pinned(tmp_path, monkeypatch):
    """REQ-REPORT-8134: original pins survive missing service and hash mutations."""
    monkeypatch.setattr(inputs.custody, "RESOURCES", [])
    assert not inputs.load(tmp_path, tmp_path / "empty")["host_qualified"]
    data = fixture()
    source = tmp_path / "board.json"
    transcript = tmp_path / "transcript.json"
    atomic_json(transcript, dict(board="original"))
    atomic_json(
        source,
        dict(
            kv260_terminal_transcript_path=str(transcript),
            kv260_terminal_transcript_sha256=reference(transcript)["sha256"],
        ),
    )
    data["boards"][0].update(source_path=str(source), source_hash=reference(source)["sha256"])
    replay = tmp_path / "replay.json"
    atomic_json(replay, dict(systems=[]))
    primitive = tmp_path / "primitive.json"
    atomic_json(primitive, dict(pairs=data["pairs"]))
    modeled = tmp_path / "modeled.json"
    atomic_json(modeled, data["modeled"])

    def auth(path, eid, field, raw, ledger):
        return dict(
            board_rows=data["boards"],
            replay_input_reference=reference(replay),
            component_cost_rows=reference(primitive),
            modeled_acquisition_bounds=reference(modeled),
            complete_service_ready_score=1,
        ), True

    monkeypatch.setattr(inputs.prior, "authenticate", auth)
    loaded = inputs.load(tmp_path, tmp_path / "valid")
    assert loaded["host_qualified"]
    assert loaded["boards"][0]["custody_valid"]
    source.write_text("{}")
    primitive.write_text("{}")
    # Pin the original expected bytes so changed receipts fail before parsing.
    oldref = dict(path=str(primitive), sha256="sha256:" + "0" * 64)

    def changed(path, eid, field, raw, ledger):
        value, valid = auth(path, eid, field, raw, ledger)
        value["component_cost_rows"] = oldref
        value["replay_input_reference"] = oldref
        return value, valid

    monkeypatch.setattr(inputs.prior, "authenticate", changed)
    loaded = inputs.load(tmp_path, tmp_path / "changed")
    assert not loaded["boards"][0]["custody_valid"]
    assert not loaded["host_qualified"]


def test_frozen_plan_production_and_owned_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8134: normal measurement, replay custody and owned checks gate publication."""
    specs = cli.commands(tmp_path / "plan")
    assert any(s.name == "full_pytest" for s in specs)
    assert any(s.name == "strict_mypy" for s in specs)
    assert len(cli.terminal_commands(tmp_path / "candidate.json")) == 3
    output = tmp_path / (cli.NAME + ".json")
    monkeypatch.setattr(cli.inputs, "load", lambda *_: fixture())
    monkeypatch.setattr(cli, "commands", lambda _: [])
    mode = {"fail": False, "drift": False, "owned": False}

    def run(root, specs, **kwargs):
        if specs and specs[0].scope == "measurement":
            args = list(specs[0].argv)
            value = h.reduce(fixture())
            if mode["drift"]:
                value["independent_count"] = 1
            atomic_json(Path(args[args.index("--output") + 1]), value)
        return [
            dict(
                name=s.name,
                scope=s.scope,
                passed=not mode["fail"],
                exit_code=int(mode["fail"]),
                command_argv=list(s.argv),
                duration_s=0.001,
            )
            for s in specs
        ]

    monkeypatch.setattr(cli, "run_commands", run)
    assert cli.main(["--output", str(output)]) == 0
    assert cli.replay(output)["passed"]
    value = json.loads(output.read_bytes())
    value["required_checks_passed"] = False
    atomic_json(output, value)
    with pytest.raises(ValueError, match="validation_receipt_drift"):
        cli.replay(output)
    mode["fail"] = True
    assert not cli.terminal(tmp_path / "candidate.json")["passed"]
    assert cli.main(["--output", str(output)]) == 1
    mode.update(fail=False, drift=True)
    assert cli.main(["--output", str(output)]) == 1
    mode.update(drift=False)
    worker = tmp_path / "worker_input.json"
    atomic_json(worker, fixture())
    assert cli.main(["--worker-input", str(worker), "--output", str(tmp_path / "worker.json")]) == 0
    # A failed owned check must publish only a disqualified, zero-readiness result.
    monkeypatch.setattr(cli, "commands", lambda _: [specs[0]])

    def owned_failure(root, plan, **kwargs):
        rows = run(root, plan, **kwargs)
        if plan and plan[0].scope != "measurement":
            rows[0].update(passed=False, exit_code=1)
        return rows

    monkeypatch.setattr(cli, "run_commands", owned_failure)
    monkeypatch.setattr(cli, "terminal", cli.replay)
    assert cli.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert (
        value["hardware_boundary_ready_score"] == value["measured_workload_bound_ready_score"] == 0
    )
