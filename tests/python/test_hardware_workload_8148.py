"""REQ-REPORT-8148, REQ-VERIFY-8148: arithmetic ceilings keep unknown costs."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from carnot.reporting import hardware_workload_8148 as h
from carnot.reporting import hardware_workload_inputs_8148 as inputs
from carnot import experiment_8148_v704_hardware_workload_boundary as cli
from carnot.reporting.current_work_receipt import atomic_json


def fixture():
    state = dict(
        geometry=dict(mean=[0] * 9, std=[1] * 9, sigma=1),
        centers=[dict(x=[0] * 9) for _ in range(16)],
        coefficients=[0] * 17,
    )
    arm = dict(
        arm="cpu",
        full_latency_ns=1000,
        arithmetic_ns=10,
        residual_ns=0,
        components=dict(arithmetic_and_boundary_ns=10, storage_ns=990),
        durable_state=state,
        values=[[0] * 9],
        source_cluster_ids=["source"],
    )
    return dict(
        fixture=True,
        boards=[
            dict(board=n, custody_valid=True, processor_class=c[0], k_max=c[1], blocker=c[2])
            for n, c in h.prior.CONTRACTS.items()
        ],
        checks=[],
        references=[],
        cited=[],
        systems=[],
        branches={
            "exp8145": dict(
                qualified=True,
                score=1,
                pairs=[dict(unit_id="unit", condition="warm", status="completed", arms=[arm])],
            ),
            "exp8146": dict(qualified=False, score=0, pairs=[]),
        },
    )


def test_independent_bounds_and_missing_costs():
    """SCENARIO-REPORT-8148-1: service failure preserves host and boards."""
    data = fixture()
    v = h.reduce(data)
    assert v["verdict_class"] == "blocked"
    assert v["hardware_boundary_ready_score"] == v["measured_workload_bound_ready_score"] == 1
    row = v["natural_workload_rows"][0]
    assert row["outer_ceiling"] == pytest.approx(1 / (1 - 0.01))
    assert row["exact_arithmetic_only_ceiling"] is None
    assert row["retained_components"]["storage_ns"] == 990
    assert row["distance_evaluations"] == 16
    assert row["distance_coordinate_terms"] == 144
    assert row["total_bytes_moved"] is None
    assert v["gate_check_summary"][-1]["observed"] == 0
    assert v["quantization_rows"] and not v["fixture_quantization_rows"]
    assert all(r["action_disagreements"] == 0 for r in v["quantization_rows"])
    data["branches"]["exp8145"]["pairs"][0]["arms"][0]["components"]["storage_ns"] = None
    missing = h.reduce(data)
    assert missing["measured_workload_bound_ready_score"] == 0
    assert any(
        c["artifact_field"].endswith("storage_ns") and c["observed"] is None
        for c in missing["gate_check_summary"]
    )
    data = fixture()
    data["boards"][0]["k_max"] = 6
    v = h.reduce(data)
    assert v["hardware_boundary_ready_score"] == 0
    assert v["board_rows"][1]["custody_valid"]
    assert v["measured_workload_bound_ready_score"] == 1


def test_qualified_branch_fixture_and_null():
    """REQ-VERIFY-8148: branch qualification and fixture truth are explicit."""
    data = fixture()
    data["branches"]["exp8146"] = deepcopy(data["branches"]["exp8145"])
    assert h.reduce(data)["verdict_class"] == "circular_positive"
    data["fixture"] = False
    assert h.reduce(data)["verdict_class"] == "null"
    arm = data["branches"]["exp8145"]["pairs"][0]["arms"][0]
    arm["arithmetic_only_ns"] = 5
    assert h.reduce(data)["natural_workload_rows"][0][
        "exact_arithmetic_only_ceiling"
    ] == pytest.approx(1000 / 995)
    data["systems"] = [dict(seed=1, state=arm["durable_state"], x=[[0] * 9])]
    arm.pop("values")
    data["branches"]["exp8146"]["pairs"] = []
    assert h.reduce(data)["fixture_quantization_rows"]


def test_natural_frozen_offset(tmp_path):
    """REQ-VERIFY-8148: natural decision margins include the frozen base logit."""
    data = fixture()
    arm = data["branches"]["exp8145"]["pairs"][0]["arms"][0]
    arm["values"][0][0] = -13.815509557963773
    v = h.reduce(data)
    for row in v["quantization_rows"]:
        assert row["reference_probabilities"][0] == pytest.approx(1e-6)
        assert row["reference_actions"] == ["accept"]
        assert row["issued_actions"] == ["accept"]


def invoke(args, tmp, expected=0):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    prefix = [str(cli.ROOT / ".venv/bin/python")]
    cfg = env.get("CARNOT_8148_COVERAGE_CONFIG")
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
    path = (
        Path(env["CARNOT_8148_CLI_RECEIPTS"])
        if env.get("CARNOT_8148_CLI_RECEIPTS")
        else tmp / "cli_receipts.json"
    )
    rows = json.loads(path.read_bytes())["rows"] if path.exists() else []
    log = path.with_name(f"private-cli-{len(rows)}.log")
    log.write_text(result.stdout + result.stderr)
    rows.append(
        dict(
            command_argv=result.args,
            expected_exit=expected,
            actual_exit=result.returncode,
            normal_exit=result.returncode >= 0,
            duration_s=time.monotonic() - began,
            log_sha256=cli.reference(log)["sha256"],
            transcript=result.stdout + result.stderr,
        )
    )
    atomic_json(path, dict(rows=rows))
    return result


def test_private_cli_and_replay(tmp_path):
    """SCENARIO-REPORT-8148-2: real outside-checkout CLI and mutation exits."""
    source = tmp_path / "input.json"
    output = tmp_path / (cli.NAME + ".json")
    data = fixture()
    data["branches"]["exp8146"] = deepcopy(data["branches"]["exp8145"])
    atomic_json(source, data)
    result = invoke(["--input", str(source), "--output", str(output)], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert invoke(["--cold-replay", str(output)], tmp_path).returncode == 0
    value = json.loads(output.read_bytes())
    value["amdahl_bounds"][0]["outer_ceiling"] = 2
    atomic_json(output, value)
    assert invoke(["--cold-replay", str(output)], tmp_path, 1).returncode == 1
    data = fixture()
    data["boards"][0]["k_max"] = 6
    atomic_json(source, data)
    assert invoke(["--input", str(source), "--output", str(output)], tmp_path).returncode == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["board_rows"][1]["custody_valid"]
    source.write_text("broken")
    assert invoke(["--input", str(source), "--output", str(output)], tmp_path, 1).returncode == 1


def test_missing_upstreams(tmp_path, monkeypatch):
    """REQ-REPORT-8148: missing external inputs name each failed operand."""
    monkeypatch.setattr(inputs.custody, "RESOURCES", [])
    data = inputs.load(tmp_path, tmp_path / "raw")
    assert not any(b["qualified"] for b in data["branches"].values())
    assert len(data["boards"]) == 3
    assert h.reduce(data)["verdict_class"] == "blocked"


def test_authenticated_inputs_and_changed_board(tmp_path, monkeypatch):
    """REQ-REPORT-8148: authenticate each pin and never erase other boards."""
    monkeypatch.setattr(inputs.custody, "RESOURCES", [])
    data = fixture()
    transcript = tmp_path / "dispatch.json"
    atomic_json(transcript, dict(original=True))
    board_path = tmp_path / "board.json"
    atomic_json(
        board_path,
        dict(
            kv260_terminal_transcript_path=str(transcript),
            kv260_terminal_transcript_sha256=cli.reference(transcript)["sha256"],
            gate_check_summary=dict(observed_latest_receipt_date="20260913"),
        ),
    )
    for board in data["boards"]:
        board.update(source_path=str(board_path), source_hash=cli.reference(board_path)["sha256"])
    primitive = tmp_path / "primitive_rows.json"
    atomic_json(primitive, dict(pairs=data["branches"]["exp8145"]["pairs"], updates=[]))
    monkeypatch.setattr(inputs.custody, "RESOURCES", ["board.json"])

    def authenticate(path, eid, field, raw, ledger):
        return dict(
            board_rows=data["boards"],
            raw_shard_hashes=[cli.reference(primitive)],
            primitive_rows=cli.reference(primitive),
            **{field: 1},
        ), True

    monkeypatch.setattr(inputs.prior, "authenticate", authenticate)
    loaded = inputs.load(tmp_path, tmp_path / "saved")
    assert all(b["custody_valid"] for b in loaded["boards"])
    assert loaded["branches"]["exp8145"]["qualified"]
    board_path.write_text("{}")
    loaded = inputs.load(tmp_path, tmp_path / "changed")
    assert not loaded["boards"][0]["custody_valid"]
    manifest = tmp_path / "ops" / "exclusion_manifest.yaml"
    manifest.parent.mkdir()
    manifest.write_text("retired:\n- experiment_id: 8145\n")
    assert not inputs.load(tmp_path, tmp_path / "retired")["branches"]["exp8145"]["qualified"]
    primitive.write_text("{}")
    assert not inputs.load(tmp_path, tmp_path / "empty")["branches"]["exp8146"]["qualified"]


def test_production_validation_and_replay_guards(tmp_path, monkeypatch):
    """REQ-REPORT-8148: owned failures disqualify; health stays a separate receipt."""
    specs = cli.commands(tmp_path / "plan")
    assert any(s.name == "full_pytest" for s in specs)
    assert len(cli.terminal_commands(tmp_path / "candidate.json")) == 3
    simple = cli.CommandSpec(
        "normal",
        (str(cli.ROOT / ".venv/bin/python"), "-c", "print('complete', flush=True)"),
        "owned",
        60,
    )
    assert cli.execute([simple], tmp_path / "normal")[0]["normal_exit"]
    data = fixture()
    data["fixture"] = False
    monkeypatch.setattr(cli.inputs, "load", lambda *_: data)
    monkeypatch.setattr(
        cli,
        "commands",
        lambda _: [simple, cli.CommandSpec("health", simple.argv, "repository_health", 60)],
    )
    mode = {"failed": False}

    def run(root, plan, **kwargs):
        log = kwargs["log_dir"] / "receipt.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("normal exit\n")
        return [
            dict(
                name=s.name,
                scope=s.scope,
                passed=not (mode["failed"] and s.scope == "owned"),
                exit_code=int(mode["failed"] and s.scope == "owned"),
                timed_out=False,
                command_argv=list(s.argv),
                duration_s=0.001,
                log_path=str(log),
                log_sha256=cli.reference(log)["sha256"],
            )
            for s in plan
        ]

    monkeypatch.setattr(cli, "run_commands", run)
    output = tmp_path / (cli.NAME + ".json")
    assert cli.main(["--output", str(output)]) == 0
    assert cli.replay(output)["passed"]
    value = json.loads(output.read_bytes())
    value["required_checks_passed"] = False
    atomic_json(output, value)
    with pytest.raises(ValueError, match="validation_receipt_drift"):
        cli.replay(output)
    mode["failed"] = True
    health_path = Path(value["replay_input_reference"]["path"]).parent / "validation_receipts.json"
    assert cli.main(["--output", str(output), "--health-receipt", str(health_path)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert (
        value["hardware_boundary_ready_score"] == value["measured_workload_bound_ready_score"] == 0
    )
    assert cli.replay(output)["passed"]
    source = tmp_path / "input.json"
    atomic_json(source, fixture())
    assert (
        cli.main(
            ["--input", str(source), "--output", str(cli.ROOT / "results" / (cli.NAME + ".json"))]
        )
        == 1
    )
