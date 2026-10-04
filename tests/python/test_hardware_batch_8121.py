"""REQ-REPORT-8121, REQ-VERIFY-8121: private traffic and publication controls."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from carnot.reporting import hardware_batch_8121 as h
from carnot.reporting import hardware_batch_inputs_8121 as inputs
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot import experiment_8121_v702_hardware_batch_boundary as cli


def fixture():
    return dict(
        fixture=True,
        systems=[],
        boards=[
            dict(board=n, custody_valid=True, processor_class=c[0], k_max=c[1], blocker=c[2])
            for n, c in h.prior.CONTRACTS.items()
        ],
        batches=[1, 8, 32, 128],
        costs=[],
        updates=[],
        checks=[],
        references=[],
        numerical_available=True,
        batch_available=False,
        touches_available=False,
    )


def cost():
    return dict(
        unit_id="one",
        source_cluster_id="one",
        arm="cpu",
        total_ns=1000,
        radial_ns=10,
        acquisition_ns=200,
        transfer_ns=50,
        fallback_ns=10,
        persistence_ns=100,
        other_ns=630,
    )


def test_traffic_bounds():
    """REQ-VERIFY-8121: packed byte ceilings charge fallback and dense distances."""
    row = h.traffic(8, 16, 12, 8, "operation_count_fixture")
    assert row["center_storage_bytes"] == 216
    assert row["coefficient_storage_bytes"] == 26
    assert row["distance_evaluations"] == 128
    assert row["distance_coordinate_terms"] == 1152
    assert row["fallback_distance_evaluations"] == 128
    assert row["resident_transfer_bytes"] == 8 * (9 * 8 + 8)
    assert row["streaming_transfer_bytes"] >= row["resident_transfer_bytes"]
    for args in [(0, 16, 8, 0), (1, 0, 8, 0), (1, 16, 7, 0), (1, 16, 8, 2)]:
        with pytest.raises(ValueError):
            h.traffic(*args, "fixture")


def test_complete_service_ceiling():
    """REQ-VERIFY-8121: incomplete costs are unavailable, never free."""
    value = h.service([cost()])
    assert value["conditional_speedup_bound"] == pytest.approx(1000 / 990)
    assert value["target_100x"] == "infeasible_under_measured_costs"
    high = cost()
    high.update(total_ns=100000, radial_ns=99010)
    assert h.service([high])["target_100x"] == "feasible_only_if_device_costs_fit_remaining_budget"
    assert h.service([])["status"] == "unavailable"
    for key in ["acquisition_ns", "transfer_ns", "fallback_ns", "persistence_ns"]:
        bad = cost()
        bad[key] = None
        assert h.service([bad])["status"] == "unavailable"
    bad = cost()
    bad["total_ns"] = 999
    assert h.service([bad])["status"] == "unavailable"
    assert h.service([cost(), cost()])["status"] == "unavailable"


def test_independent_rows_and_optional_blocks():
    """SCENARIO-VERIFY-8121-BOUNDARY: missing boards leave other rows usable."""
    data = fixture()
    value = h.reduce(data)
    assert value["verdict_class"] == "circular_positive"
    assert len(value["operation_count_rows"]) == 24
    assert value["hardware_execution"] is False
    assert value["conditional_speedup_bound"] is None
    assert all(r["verdict_class"] == "blocked" for r in value["subinput_rows"])
    assert value["independent_generalization_score"] == 0
    data["fixture"] = False
    assert h.reduce(data)["verdict_class"] == "null"
    data["boards"][0]["k_max"] = 6
    value = h.reduce(data)
    assert value["board_rows"][0]["status"] == "blocked"
    assert value["board_rows"][1]["status"] == "completed"
    assert value["hardware_boundary_ready_score"] == 1
    data["numerical_available"] = False
    assert h.reduce(data)["honest_verdict"].startswith("complete_blocked_")
    data["checks"] = [dict(check="missing_exact_path", passed=False)]
    assert h.reduce(data)["honest_verdict"] == "complete_blocked_missing_exact_path"


def test_load_missing_and_authenticated(tmp_path, monkeypatch):
    """REQ-REPORT-8121: sidecar and exact primitive hashes gate subinputs."""
    raw = tmp_path / "raw"
    raw.mkdir()
    assert not inputs.load(tmp_path, raw)["numerical_available"]
    data = fixture()
    source = tmp_path / "board.json"
    atomic_json(source, dict(workload="original", run_date="20260913"))
    board = data["boards"][0]
    board.update(source_path=str(source), source_hash=sha256_file(source))
    replay = tmp_path / "original.json"
    atomic_json(replay, dict(systems=[]))
    primitive = tmp_path / "primitive_rows.json"
    atomic_json(
        primitive,
        dict(
            whole_service_cost_rows=[cost()],
            rows=[
                dict(touched_coefficients=[0, 1], before_weights=[1], after_weights=[2], steps=4)
            ],
        ),
    )

    def auth(path, eid, field, raw, ledger):
        return dict(
            board_rows=[board],
            replay_input_reference=dict(path=str(replay), sha256=sha256_file(replay)),
            code_config_hashes=[],
            measurement_config=dict(batches=[2]),
            complete_service_ready_score=1,
            raw_shard_hashes=[dict(path=str(primitive), sha256=sha256_file(primitive))],
        ), True

    monkeypatch.setattr(inputs.prior, "authenticate", auth)
    monkeypatch.setattr(inputs, "RESOURCES", [])
    loaded = inputs.load(tmp_path, raw)
    assert loaded["batches"] == [2]
    assert loaded["costs"] == [cost()]
    assert loaded["updates"][0]["coefficient_touches"] == 2
    assert loaded["boards"][0]["custody_valid"]
    source.write_text("{}")
    loaded = inputs.load(tmp_path, raw)
    assert not loaded["boards"][0]["custody_valid"]
    assert len(loaded["boards"]) == 3


def invoke(args, tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    cfg = env.get("CARNOT_8121_COVERAGE_CONFIG")
    prefix = [str(cli.ROOT / ".venv/bin/python")]
    if cfg:
        prefix += ["-m", "coverage", "run", "--parallel-mode", "--rcfile=" + cfg]
    started = time.monotonic()
    result = subprocess.run(
        [*prefix, str(cli.ROOT / cli.SCRIPT), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    receipt_path = env.get("CARNOT_8121_CLI_RECEIPTS")
    if receipt_path:
        path = Path(receipt_path)
        rows = json.loads(path.read_text())["rows"] if path.exists() else []
        log = path.with_name(f"cli-{len(rows)}.log")
        log.write_text(result.stdout + result.stderr)
        rows.append(
            dict(
                command_argv=result.args,
                exit_code=result.returncode,
                normal_exit=result.returncode >= 0,
                duration_s=time.monotonic() - started,
                log_sha256=sha256_file(log),
                transcript=result.stdout + result.stderr,
                route=args[0],
            )
        )
        atomic_json(path, dict(rows=rows))
    return result


def test_private_cli_success_blocked_mutation_cold_replay(tmp_path):
    """SCENARIO-REPORT-8121-CLI: real script paths work outside the checkout."""
    source = tmp_path / "fixture.json"
    output = tmp_path / (cli.NAME + ".json")
    atomic_json(source, fixture())
    result = invoke(["--input", str(source), "--output", str(output)], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["MODEL_SPECS"] == [] and value["required_checks_passed"]
    assert invoke(["--cold-replay", str(output)], tmp_path).returncode == 0
    value["operation_count_rows"][0]["center_storage_bytes"] += 1
    atomic_json(output, value)
    assert invoke(["--cold-replay", str(output)], tmp_path).returncode == 1
    source.write_text("broken")
    assert invoke(["--input", str(source), "--output", str(output)], tmp_path).returncode == 1
    blocked = fixture()
    blocked["numerical_available"] = False
    atomic_json(source, blocked)
    assert invoke(["--input", str(source), "--output", str(output)], tmp_path).returncode == 0
    assert json.loads(output.read_text())["verdict_class"] == "blocked"


def test_plan_and_production_routes(tmp_path, monkeypatch):
    """REQ-REPORT-8121: frozen owned argv and normal worker exits gate readiness."""
    assert any(s.name == "full_pytest" for s in cli.commands(tmp_path))
    assert any(s.name == "strict_mypy" for s in cli.commands(tmp_path))
    assert len(cli.terminal_commands(tmp_path / "candidate.json")) == 3
    output = tmp_path / (cli.NAME + ".json")
    monkeypatch.setattr(cli.inputs, "load", lambda *_: fixture())
    monkeypatch.setattr(cli, "commands", lambda _: [])

    def run(root, specs, **kwargs):
        if specs and specs[0].scope == "measurement":
            args = list(specs[0].argv)
            atomic_json(Path(args[args.index("--output") + 1]), h.reduce(fixture()))
        return [
            dict(
                name=s.name,
                scope=s.scope,
                passed=True,
                exit_code=0,
                command_argv=list(s.argv),
                duration_s=0.001,
            )
            for s in specs
        ]

    monkeypatch.setattr(cli, "run_commands", run)
    assert cli.main(["--output", str(output)]) == 0
    assert cli.replay(output)["passed"]
    monkeypatch.setattr(cli, "commands", lambda _: cli.terminal_commands(output))
    monkeypatch.setattr(
        cli,
        "run_commands",
        lambda *a, **kw: [dict(name="failed", scope="owned", passed=False, exit_code=1)],
    )
    assert cli.main(["--output", str(output)]) == 1
    atomic_json(tmp_path / "worker_input.json", fixture())
    assert (
        cli.main(
            [
                "--worker-input",
                str(tmp_path / "worker_input.json"),
                "--output",
                str(tmp_path / "worker.json"),
            ]
        )
        == 0
    )


def test_numeric_replay_and_failed_envelope(monkeypatch):
    """REQ-VERIFY-8121: analytic bounds and failed numerical checks are exercised."""
    data = fixture()
    data["systems"] = [
        dict(
            seed=i,
            x=[[0.0] * 9],
            state=dict(
                geometry=dict(mean=[0.0] * 9, std=[1.0] * 9, sigma=2.0),
                centers=[dict(x=[0.0] * 9)],
                coefficients=[0.0, 0.0],
            ),
        )
        for i in range(16)
    ]
    value = h.reduce(data)
    assert len(value["precision_rows"]) == 48
    assert value["precision_summary"]["passed"]
    monkeypatch.setattr(h.prior, "summarize", lambda _: dict(passed=False))
    assert h.reduce(data)["verdict_class"] == "disqualified"


def test_owned_failure_and_replay_receipt_drift(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8121-CLI: failed owned checks force zero readiness."""
    output = tmp_path / (cli.NAME + ".json")
    monkeypatch.setattr(cli.inputs, "load", lambda *_: fixture())
    monkeypatch.setattr(cli, "commands", lambda _: cli.terminal_commands(output))

    def run(root, specs, **kw):
        measurement = specs and specs[0].scope == "measurement"
        if measurement:
            args = list(specs[0].argv)
            atomic_json(Path(args[args.index("--output") + 1]), h.reduce(fixture()))
        return [
            dict(
                name=s.name,
                scope=s.scope,
                passed=bool(measurement or s.scope == "terminal"),
                command_argv=list(s.argv),
                exit_code=0,
            )
            for s in specs
        ]

    monkeypatch.setattr(cli, "run_commands", run)
    monkeypatch.setattr(cli, "commands", lambda _: [cli.CommandSpec("failed", (), "owned", 1)])
    assert cli.main(["--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "disqualified" and value["hardware_boundary_ready_score"] == 0
    assert cli.replay(output)["passed"]
    value["validation_receipts"] = []
    atomic_json(output, value)
    with pytest.raises(ValueError, match="validation_receipt_drift"):
        cli.replay(output)
    original = deepcopy(h.reduce(fixture()))
    original["hardware_execution"] = True

    def forged(root, specs, **kw):
        args = list(specs[0].argv)
        atomic_json(Path(args[args.index("--output") + 1]), original)
        return [dict(passed=True)]

    monkeypatch.setattr(cli, "run_commands", forged)
    assert cli.main(["--output", str(output)]) == 1


def test_transcript_code_and_manifest_mutations(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8121-CLI: board transcripts and frozen bytes are binding."""
    source = tmp_path / "original.json"
    output = tmp_path / (cli.NAME + ".json")
    transcript = tmp_path / "transcript.json"
    atomic_json(transcript, dict(workload="unchanged"))
    atomic_json(
        source,
        dict(
            kv260_terminal_transcript_path=str(transcript),
            kv260_terminal_transcript_sha256=sha256_file(transcript).split(":")[1],
        ),
    )
    code = tmp_path / "code.py"
    code.write_text("unchanged")
    replay = tmp_path / "replay.json"
    atomic_json(replay, dict(systems=[]))
    board = fixture()["boards"][0]
    board.update(source_path=str(source), source_hash=sha256_file(source))

    def auth(path, eid, field, raw, ledger):
        return dict(
            board_rows=[board],
            code_config_hashes=[dict(path=str(code), sha256=sha256_file(code))],
            replay_input_reference=dict(path=str(replay), sha256=sha256_file(replay)),
            raw_shard_hashes=[],
        ), eid == 8108

    monkeypatch.setattr(inputs.prior, "authenticate", auth)
    monkeypatch.setattr(inputs, "RESOURCES", [])
    raw = tmp_path / "inputs"
    raw.mkdir()
    assert inputs.load(tmp_path, raw)["boards"][0]["custody_valid"]
    transcript.write_text("{}")
    assert not inputs.load(tmp_path, raw)["boards"][0]["custody_valid"]
    atomic_json(source, fixture())
    assert invoke(["--input", str(source), "--output", str(output)], tmp_path).returncode == 0
    value = json.loads(output.read_text())
    manifest = next(
        Path(r["path"])
        for r in value["raw_shard_hashes"]
        if Path(r["path"]).name == "validation_manifest.json"
    )
    manifest.write_text("{}")
    with pytest.raises(ValueError):
        cli.replay(output)
