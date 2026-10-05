"""REQ-REPORT-8162, REQ-VERIFY-8162: private costs cannot reopen a board."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess

import pytest

from carnot.reporting import hardware_workload_8162 as h
from carnot.reporting import hardware_boundary_execution_8162 as cli
from carnot.reporting import hardware_workload_inputs_8162 as inputs
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference


def panel():
    """Provide a declared private example with known arithmetic and margins."""
    state = dict(
        geometry=dict(mean=[0] * 9, std=[1] * 9, sigma=1),
        centers=[dict(x=[0] * 9)],
        coefficients=[0, 0],
    )
    request = dict(
        source_cluster_id="source",
        values=[-2.1972245773362196] + [0] * 8,
        queue_ns=5,
        latency_ns=100,
        probability=0.1,
        action="defer",
    )
    arm = dict(
        arm="cpu",
        duration_ns=100,
        arithmetic_only_ns=10,
        components=dict(
            arithmetic_ns=20, commit_write_fsync_ns=70, serialization_ns=5, residual_and_queue_ns=5
        ),
        requests=[request],
        durable_state_bytes=100,
        head_hash="private",
    )
    branch = dict(
        qualified=True,
        score=1,
        path="private",
        hash="private",
        pairs=[dict(unit_id="batch", condition="warm", status="completed", arms=[arm])],
    )
    return dict(
        fixture=True,
        checks=[],
        references=[],
        cited=[],
        trained_head_specs=[],
        boards=[
            dict(
                board=n,
                custody_valid=True,
                processor_class=c[0],
                k_max=c[1],
                blocker=c[2],
                terminal_criterion_met=False,
            )
            for n, c in h.prior.CONTRACTS.items()
        ],
        state=state,
        branches=dict(
            exp8159=branch,
            exp8160=dict(qualified=False, score=0, path="missing", hash=None, pairs=[]),
        ),
    )


def test_bounds_and_board_independence():
    """SCENARIO-VERIFY-8162: missing composition preserves host and custody."""
    data = panel()
    value = h.reduce(data)
    assert value["hardware_boundary_ready_score"] == 1
    assert value["measured_workload_bound_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"] == "complete_blocked_acquisition_composition_ready_score"
    row = value["workload_rows"][0]
    assert row["exact_arithmetic_only_ceiling"] == pytest.approx(100 / 90)
    assert row["outer_ceiling"] == pytest.approx(100 / 80)
    assert row["retained_components"]["commit_write_fsync_ns"] == 70
    assert row["total_bytes_moved"] is None and row["durable_state_bytes"] == 100
    assert row["distance_coordinate_terms"] == 9
    assert value["amdahl_bounds"][0]["required_retained_work_reduction_fraction"] == pytest.approx(
        79 / 80
    )
    assert all(r["action_disagreements"] == 0 for r in value["quantization_rows"])
    assert {r["bits"] for r in value["fixture_quantization_rows"]} == {8, 12, 16, 64}
    assert value["independent_count"] == 0
    data["boards"][0]["k_max"] = 6
    changed = h.reduce(data)
    assert changed["hardware_boundary_ready_score"] == 0
    assert changed["board_rows"][1]["custody_valid"]
    assert changed["measured_workload_bound_ready_score"] == 1
    data = panel()
    data["branches"]["exp8159"]["pairs"][0]["arms"][0]["components"]["serialization_ns"] = None
    bad = h.reduce(data)
    assert bad["measured_workload_bound_ready_score"] == 0
    assert any(c["observed"] is None and not c["passed"] for c in bad["gate_check_summary"])


def test_composition_and_unknown_arithmetic():
    """REQ-VERIFY-8162: request composition has a separate cost denominator."""
    data = panel()
    b = data["branches"]["exp8160"] = deepcopy(data["branches"]["exp8159"])
    b["captures"] = [dict(source_cluster_id="source", acquisition_ns=1000, status="completed")]
    b["startup_amortized_ns"] = 20
    b["pairs"][0]["arms"][0].update(duration_ns=100, arithmetic_only_ns=None, request_overhead_ns=2)
    value = h.reduce(data)
    composed = next(r for r in value["workload_rows"] if r["upstream"] == "exp8160")
    assert composed["numerator"] == 1122
    assert composed["exact_arithmetic_only_ceiling"] is None
    assert composed["retained_components"]["acquisition_ns"] == 1000
    data["fixture"] = False
    assert h.reduce(data)["independent_count"] == 1
    for board in data["boards"]:
        board["terminal_criterion_met"] = True
    assert h.reduce(data)["verdict_class"] == "null"
    data["fixture"] = True
    assert h.reduce(data)["verdict_class"] == "circular_positive"
    b["captures"][0]["acquisition_ns"] = None
    assert any(r["status"] == "excluded" for r in h.reduce(data)["workload_rows"])


def invoke(args, tmp_path, expected=0):
    """Run the actual CLI with child coverage and no ambient repository path."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    prefix = [str(cli.ROOT / ".venv/bin/python")]
    if env.get("CARNOT_8162_COVERAGE_CONFIG"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + env["CARNOT_8162_COVERAGE_CONFIG"]]
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


def test_private_cli_success_block_tamper_replay(tmp_path):
    """SCENARIO-REPORT-8162: success, missing cost and board mutations stay private."""
    data = panel()
    source = tmp_path / "input.json"
    output = tmp_path / (cli.NAME + ".json")
    atomic_json(source, data)
    invoke(["--input", str(source), "--output", str(output)], tmp_path)
    invoke(["--cold-replay", str(output)], tmp_path)
    value = json.loads(output.read_text())
    assert value["MODEL_SPECS"] == [] and value["required_checks_passed"]
    value["workload_rows"][0]["numerator"] += 1
    atomic_json(output, value)
    invoke(["--cold-replay", str(output)], tmp_path, 1)
    data["boards"][0]["custody_valid"] = False
    data["branches"]["exp8159"]["pairs"][0]["arms"][0]["components"]["serialization_ns"] = None
    atomic_json(source, data)
    invoke(["--input", str(source), "--output", str(output)], tmp_path)
    value = json.loads(output.read_text())
    assert value["hardware_boundary_ready_score"] == 0
    assert value["measured_workload_bound_ready_score"] == 0
    invoke(["--cold-replay", str(output)], tmp_path)
    invoke(["--date", "20261004"], tmp_path, 2)


def test_authentication_and_missing_originals(tmp_path, monkeypatch):
    """REQ-REPORT-8162: authenticate original receipts, shards, stores and retirement."""
    missing = inputs.load(tmp_path, tmp_path / "missing")
    assert len(missing["boards"]) == 3 and not missing["branches"]["exp8159"]["qualified"]
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
        monkeypatch.setitem(inputs.PINS, name, board["source_hash"])
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
        8148: dict(board_rows=boards),
        8159: dict(
            verdict_class="positive",
            primitive_rows=reference(work),
            input_data=reference(upstream),
            host_batch_ready_score=1,
        ),
        8160: dict(
            verdict_class="null",
            primitive_rows=reference(work),
            input_data=reference(upstream),
            acquisition_composition_ready_score=1,
        ),
    }
    monkeypatch.setattr(
        inputs.prior, "authenticate", lambda path, eid, field, raw, data: (values[eid], True)
    )
    data = inputs.load(tmp_path, tmp_path / "valid")
    assert all(b["custody_valid"] for b in data["boards"])
    assert data["branches"]["exp8159"]["qualified"]
    assert data["branches"]["exp8160"]["startup_amortized_ns"] == 2
    assert data["state"]["coefficients"] == [0, 0]
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops/exclusion_manifest.yaml").write_text("retired:\n  - experiment_id: 8159\n")
    store.write_text("altered")
    (tmp_path / "KV260.json").write_text("altered")
    values[8160]["verdict_class"] = "disqualified"
    changed = inputs.load(tmp_path, tmp_path / "changed")
    assert not changed["boards"][0]["custody_valid"] and changed["boards"][1]["custody_valid"]
    assert not changed["branches"]["exp8159"]["qualified"]
    assert not changed["branches"]["exp8160"]["qualified"]


def test_unknown_clocks_and_no_precision():
    """REQ-VERIFY-8162: missing clocks and ineligible inputs earn no arithmetic credit."""
    data = panel()
    source = data["branches"]["exp8160"] = deepcopy(data["branches"]["exp8159"])
    source.update(
        captures=[dict(source_cluster_id="source", acquisition_ns=1)], startup_amortized_ns=1
    )
    request = source["pairs"][0]["arms"][0]["requests"][0]
    request.update(response_ns=120, enqueue_ns=20)
    value = h.reduce(data)
    assert any(
        r["status"] == "completed" and r["upstream"] == "exp8160" for r in value["workload_rows"]
    )
    request.pop("response_ns")
    assert any(r["status"] == "excluded" for r in h.reduce(data)["workload_rows"])
    for branch in data["branches"].values():
        branch["qualified"] = False
    assert h.reduce(data)["measured_workload_bound_ready_score"] == 0
    assert h.precision(data) == ([], [])


def test_production_orchestration_and_owned_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8162: normal receipts permit publication; owned failure zeros readiness."""
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
    """REQ-REPORT-8162: failed terminal checks never expose ready primary bytes."""
    data = panel()
    data["branches"]["exp8159"]["pairs"][0]["arms"][0]["components"].pop("arithmetic_ns")
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
