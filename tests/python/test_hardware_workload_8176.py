"""REQ-REPORT-8176, REQ-VERIFY-8176: separate retained work and device custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess

import pytest

from carnot.reporting import hardware_workload_8176 as h
from carnot.reporting import hardware_boundary_execution_8176 as cli
from carnot.reporting import hardware_workload_inputs_8176 as inputs
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference
from test_hardware_workload_8162 import panel as previous_panel


def panel():
    """Use known private costs so inclusive timer mistakes change assertions."""
    data = previous_panel()
    data["branches"].pop("exp8160")
    data["branches"]["exp8173"] = dict(
        qualified=False, score=0, pairs=[], path="missing", hash=None
    )
    data["branches"]["exp8174"] = deepcopy(data["branches"]["exp8159"])
    data["branches"]["exp8174"]["pairs"][0]["arms"][0].pop("arithmetic_only_ns")
    data["branches"]["exp8174"]["pairs"][0]["arms"][0]["components"] = dict(
        acquisition_ns=50, arithmetic_ns=20, persistence_ns=25, queue_and_host_ns=5
    )
    data["startup_costs"] = dict(cold_load_ns=48, amortization_requests=48, startup_total_s=None)
    return data


def test_independent_scopes_and_receipts():
    """SCENARIO-REPORT-8176-INDEPENDENT: one blocked branch preserves useful scopes."""
    data = panel()
    value = h.reduce(data)
    assert value["honest_verdict"] == "complete_blocked_composition_replay_ready_score"
    assert (
        value["hardware_boundary_ready_score"] == value["measured_workload_bound_ready_score"] == 1
    )
    assert {r["upstream"] for r in value["workload_rows"]} == {"exp8159", "exp8173", "exp8174"}
    complete = next(r for r in value["workload_rows"] if r["upstream"] == "exp8174")
    assert complete["outer_ceiling"] == 1.25
    assert complete["exact_arithmetic_only_ceiling"] is None
    assert complete["retained_components"]["acquisition_ns"] == 50
    assert complete["distance_coordinate_terms"] == 9
    assert value["independent_count"] == 0
    assert {r["bits"] for r in value["fixture_quantization_rows"]} == {8, 12, 16, 64}
    assert all(r["measured_incremental_fallback_ns"] is None for r in value["fallback_cost_rows"])
    data["boards"][0]["k_max"] = 6
    changed = h.reduce(data)
    assert changed["hardware_boundary_ready_score"] == 0
    assert changed["board_rows"][1]["custody_valid"]
    assert changed["measured_workload_bound_ready_score"] == 1
    data["branches"]["exp8174"]["pairs"][0]["arms"][0]["components"]["acquisition_ns"] = None
    assert any(
        r["upstream"] == "exp8174" and r["status"] == "excluded"
        for r in h.reduce(data)["workload_rows"]
    )


def test_success_exact_arithmetic_and_missing_branch():
    """REQ-VERIFY-8176: pure arithmetic and mixed envelopes have separate fractions."""
    data = panel()
    data["branches"]["exp8173"] = dict(
        qualified=True, score=1, pairs=[], path="private", hash="private"
    )
    for board in data["boards"]:
        board["terminal_criterion_met"] = True
    data["fixture"] = False
    value = h.reduce(data)
    assert value["verdict_class"] == "null"
    assert value["independent_count"] == 1
    assert value["amdahl_bounds"][0]["required_arithmetic_fraction_for_100x"] == 0.99
    assert value["amdahl_bounds"][0]["exact_arithmetic_only_ceiling"] == pytest.approx(100 / 90)
    data["fixture"] = True
    assert h.reduce(data)["verdict_class"] == "circular_positive"
    data["branches"]["exp8174"].update(qualified=False, score=None, pairs=[])
    value = h.reduce(data)
    assert value["measured_workload_bound_ready_score"] == 1
    assert any(
        c["check"] == "complete_service_ready_score" and c["observed"] is None
        for c in value["gate_check_summary"]
    )


def invoke(args, tmp_path, expected=0):
    """Exercise the deployed CLI from private CWD with no inherited import path."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    prefix = [str(cli.ROOT / ".venv/bin/python")]
    if env.get("CARNOT_8176_COVERAGE_CONFIG"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + env["CARNOT_8176_COVERAGE_CONFIG"]]
    result = subprocess.run(
        prefix + [str(cli.ROOT / cli.SCRIPT), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    return result


def test_private_cli_block_tamper_and_cold_replay(tmp_path):
    """SCENARIO-VERIFY-8176-CLI: preserve original inputs and detect changed headlines."""
    source = tmp_path / "input.json"
    output = tmp_path / (cli.NAME + ".json")
    data = panel()
    atomic_json(source, data)
    invoke(["--input", str(source), "--output", str(output)], tmp_path)
    invoke(["--cold-replay", str(output)], tmp_path)
    value = json.loads(output.read_bytes())
    assert value["MODEL_SPECS"] == [] and value["required_checks_passed"]
    assert value["model_invocation_counts"]["model_loads_attempted"] == 0
    value["workload_rows"][0]["numerator"] += 1
    atomic_json(output, value)
    invoke(["--cold-replay", str(output)], tmp_path, 1)
    data["boards"][0]["custody_valid"] = False
    data["branches"]["exp8174"]["pairs"][0]["arms"][0]["components"]["acquisition_ns"] = None
    atomic_json(source, data)
    invoke(["--input", str(source), "--output", str(output)], tmp_path)
    assert json.loads(output.read_bytes())["hardware_boundary_ready_score"] == 0


def test_production_orchestration_and_owned_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8176: normal receipts permit publication; owned failure zeros readiness."""
    data = panel()
    monkeypatch.setattr(inputs, "load", lambda root, raw: deepcopy(data))
    failed = [False]

    def execute(specs, private, env=None):
        return [
            dict(
                name=s.name,
                passed=not (failed[0] and s.name == "focused_pytest"),
                normal_exit=True,
                duration_s=0.001,
                scope=s.scope,
            )
            for s in specs
        ]

    monkeypatch.setattr(cli, "execute", execute)
    output = tmp_path / (cli.NAME + ".json")
    assert cli.main(["--output", str(output)]) == 0
    assert cli.replay(output)["passed"]
    value = json.loads(output.read_text())
    value["validation_receipts"][0]["passed"] = False
    atomic_json(output, value)
    with pytest.raises(ValueError, match="validation_receipt_drift"):
        cli.replay(output)
    failed[0] = True
    assert cli.main(["--output", str(output)]) == 1
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "disqualified" and value["hardware_boundary_ready_score"] == 0
    assert cli.replay(output)["passed"]
    assert cli.main(["--cold-replay", str(output)]) == 0
    assert cli.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    assert cli.main(["--input", str(tmp_path / "absent")]) == 1
    monkeypatch.setattr(h, "reduce", lambda data: {})
    value["amdahl_bounds"][0]["outer_ceiling"] += 1
    atomic_json(output, value)
    with pytest.raises(ValueError, match="independent_ceiling_drift"):
        cli.replay(output)
    monkeypatch.setattr(cli, "main", lambda: 0)
    runpy.run_path(str(cli.ROOT / cli.SCRIPT), run_name="imported_cli")
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(cli.ROOT / cli.SCRIPT), run_name="__main__")
    assert exit_info.value.code == 0


def test_terminal_failure_and_missing_arithmetic(tmp_path, monkeypatch):
    """REQ-REPORT-8176: failed terminal checks never expose ready primary bytes."""
    data = panel()
    for branch in data["branches"].values():
        for pair in branch["pairs"]:
            for arm in pair["arms"]:
                arm["components"].pop("arithmetic_ns", None)
    assert h.reduce(data)["measured_workload_bound_ready_score"] == 0
    monkeypatch.setattr(inputs, "load", lambda root, raw: panel())

    def execute(specs, private, env=None):
        log = tmp_path / "receipt.log"
        log.write_text("private receipt\n")
        return [
            dict(
                name=s.name,
                passed=s.scope != "terminal",
                normal_exit=True,
                duration_s=0.001,
                scope=s.scope,
                log_path=str(log),
                log_sha256=reference(log)["sha256"],
            )
            for s in specs
        ]

    monkeypatch.setattr(cli, "execute", execute)
    output = tmp_path / (cli.NAME + ".json")
    assert cli.main(["--output", str(output)]) == 1
    assert not output.exists()
    failed = next((tmp_path / "raw").rglob("failed_terminal_candidate.json"))
    assert json.loads(failed.read_text())["verdict_class"] == "disqualified"


def test_authentication_and_missing_originals(tmp_path, monkeypatch):
    """REQ-REPORT-8176: authenticate original receipts, shards, stores and retirement."""
    missing = inputs.load(tmp_path, tmp_path / "missing")
    assert len(missing["boards"]) == 3 and not missing["branches"]["exp8159"]["qualified"]
    missing_input = tmp_path / "missing-input.json"
    missing_output = tmp_path / (cli.NAME + ".json")
    atomic_json(missing_input, missing)
    invoke(["--input", str(missing_input), "--output", str(missing_output)], tmp_path)
    assert json.loads(missing_output.read_bytes())["verdict_class"] == "blocked"
    private = panel()
    boards = private["boards"]
    for board in boards:
        name = board["board"]
        path = tmp_path / (name + ".json")
        transcript = tmp_path / (name + "-transcript.json")
        atomic_json(transcript, dict(private=True))
        receipt = dict(run_date="20260912")
        if name == "KV260":
            receipt.update(
                kv260_terminal_transcript_path=str(transcript),
                kv260_terminal_transcript_sha256=reference(transcript)["sha256"],
            )
        if name == "PolarFire":
            receipt.update(
                raw_dispatch_transcript_path=str(transcript),
                board_rows=[
                    dict(
                        board=name,
                        latest_receipt_hash=reference(transcript)["sha256"].split(":")[1],
                    )
                ],
            )
        atomic_json(path, receipt)
        board.update(source_path=str(path), source_hash=reference(path)["sha256"])
        monkeypatch.setitem(inputs.old.PINS, name, board["source_hash"])
    state = private["state"]
    upstream = tmp_path / "upstream.json"
    atomic_json(
        upstream,
        dict(
            head=dict(centers=state["centers"], intercept=0, weights=[0]),
            geometry=state["geometry"],
            trained_head_specs=[dict(kind="private")],
        ),
    )
    store = tmp_path / "store.json"
    atomic_json(store, dict(state=[]))
    pairs = deepcopy(private["branches"]["exp8159"]["pairs"])
    pairs[0]["arms"][0].update(store_path=str(store), store_sha256=reference(store)["sha256"])
    work = tmp_path / "work.json"
    atomic_json(
        work, dict(pairs=pairs, host_groups=pairs, startup_ns=32, warmups=[dict(acquisition_ns=32)])
    )
    values = {
        8162: dict(board_rows=boards),
        8159: dict(
            verdict_class="positive",
            primitive_rows=reference(work),
            input_data=reference(upstream),
            host_batch_ready_score=1,
        ),
        8173: dict(
            verdict_class="null",
            primitive_rows=reference(work),
            input_data=reference(upstream),
            composition_replay_ready_score=1,
        ),
    }
    monkeypatch.setattr(
        inputs.prior, "authenticate", lambda path, eid, field, raw, data: (values[eid], True)
    )
    snapshot = tmp_path / "snapshot.json"
    atomic_json(snapshot, dict(references=[reference(store)], state=state))
    values[8162]["replay_input_reference"] = reference(snapshot)
    request = deepcopy(private["branches"]["exp8159"]["pairs"][0]["arms"][0]["requests"][0])
    request.update(
        unit_id="complete",
        arm="cpu",
        condition="natural",
        status="completed",
        arrival_ns=0,
        response_ns=100,
        acquisition_end_ns=50,
        batch_start_ns=55,
        arithmetic_start_ns=55,
        arithmetic_end_ns=75,
        commit_start_ns=75,
        commit_end_ns=95,
        store=reference(store),
        evidence=[],
    )
    complete = tmp_path / "complete.json"
    atomic_json(complete, dict(requests=[request, dict(request, status="failed")]))
    composed = tmp_path / "composed.json"
    atomic_json(
        composed,
        dict(sources=[dict(value=dict(experiment_id=8160), work=json.loads(work.read_bytes()))]),
    )
    values[8173]["input_data"] = reference(composed)
    values[8174] = dict(
        verdict_class="null",
        complete_service_ready_score=1,
        primitive_rows=reference(complete),
        input_data=reference(upstream),
        startup_costs=dict(cold_load_ns=48, amortization_requests=48),
    )
    data = inputs.load(tmp_path, tmp_path / "valid")
    assert all(b["custody_valid"] for b in data["boards"])
    assert data["branches"]["exp8159"]["qualified"]
    assert data["branches"]["exp8173"]["startup_amortized_ns"] == 2
    assert data["state"]["coefficients"] == [0, 0]
    assert data["branches"]["exp8174"]["qualified"]
    assert data["branches"]["exp8174"]["pairs"][0]["arms"][0]["components"]["acquisition_ns"] == 50
    atomic_json(snapshot, dict(references=[], state={}))
    values[8162]["replay_input_reference"] = reference(snapshot)
    host_head = values[8159].pop("input_data")
    independent = inputs.load(tmp_path, tmp_path / "independent")
    assert independent["state"]["coefficients"] == [0, 0]
    values[8159]["input_data"] = host_head
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops/exclusion_manifest.yaml").write_text("retired:\n  - experiment_id: 8159\n")
    store.write_text("altered")
    (tmp_path / "KV260.json").write_text("altered")
    values[8173]["verdict_class"] = "disqualified"
    changed = inputs.load(tmp_path, tmp_path / "changed")
    assert not changed["boards"][0]["custody_valid"] and changed["boards"][1]["custody_valid"]
    assert not changed["branches"]["exp8159"]["qualified"]
    assert not changed["branches"]["exp8173"]["qualified"]
    source = tmp_path / "altered-input.json"
    output = tmp_path / (cli.NAME + ".json")
    atomic_json(source, changed)
    invoke(["--input", str(source), "--output", str(output)], tmp_path)
    assert json.loads(output.read_bytes())["hardware_boundary_ready_score"] == 0
    invoke(["--cold-replay", str(output)], tmp_path)


def test_configuration_and_wait_counts(tmp_path, capsys):
    """REQ-VERIFY-8176: cold config mutations fail and child waits report actual counts."""
    source = tmp_path / "input.json"
    output = tmp_path / (cli.NAME + ".json")
    atomic_json(source, panel())
    invoke(["--input", str(source), "--output", str(output)], tmp_path)
    value = json.loads(output.read_bytes())
    value["config"]["seed"] += 1
    atomic_json(output, value)
    with pytest.raises(ValueError, match="configuration_drift"):
        cli.replay(output)

    class Stop:
        remaining = 1

        def wait(self, seconds):
            self.remaining -= 1
            return self.remaining < 0

    cli.heartbeat(Stop(), 2, 5)
    assert "completed=2 pending=3" in capsys.readouterr().out
