"""REQ-REPORT-8577, REQ-VERIFY-8577: hardware accounting cannot create execution."""

from copy import deepcopy
import json
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import resident_hardware_boundary_8577 as e
from carnot.reporting import resident_hardware_evidence_8577 as source
from carnot.reporting.current_work_receipt import atomic_json


def test_missing_failure_and_rehashed_tamper(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8577-REPLAY: repaired envelopes do not replace original authority."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(tmp_path / "missing", raw, tmp_path)
    value = e.build(work, [dict(passed=True)], raw, output)
    assert value["verdict_class"] == "blocked"
    assert value["hardware_boundary_ready_score"] == 0
    assert value["physical_hardware_ready_score"] == 0
    assert len(value["operation_coverage_rows"]) == 50
    assert all(r["zero_cost_ceiling"] is None for r in value["conditional_ceiling_rows"])
    assert set(value) <= set(value["field_principles"])
    atomic_json(output, value)
    assert e.replay(output) and not e.replay(tmp_path / "absent")
    value["physical_hardware_ready_score"] = 1
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)
    changed = deepcopy(work)
    changed["evidence"]["history"]["qualified"] = True
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, [dict(passed=True)], raw, output))
    assert not e.replay(output)
    failed = e.build(work, [dict(name="owned", passed=False)], raw, output)
    assert failed["verdict_class"] == "disqualified"
    assert not failed["required_checks_passed"]


def test_original_scopes_and_external_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8577-BOUNDARY: cold scope never supplies resident clocks."""
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path)
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert work["evidence"]["history"]["qualified"]
    assert not value["qualified_cost_reference"]["qualified"]
    assert value["verdict_class"] == "blocked"
    assert value["hardware_boundary_ready_score"] == 1
    assert value["physical_hardware_ready_score"] == 0
    assert value["cold_boundary_reference"]["ceilings"][0]["zero_cost_ceiling"] == pytest.approx(
        1.0087464025904647
    )
    assert value["cold_boundary_reference"]["ceilings"][1]["zero_cost_ceiling"] == pytest.approx(
        1.0079812456487978
    )
    assert value["kv260_supported_operations"][0]["k_max"] == 5
    assert not value["polarfire_graduation"]["fpga_fabric_acceleration"]
    assert value["board_reopen_requirements"][2]["expected_idcode"] == "0x20000001"
    assert not value["acquisition_decision"]["recommended"]
    assert all(not r["device_supported"] for r in value["operation_coverage_rows"])
    assert all(r["resident_clock_ns"] is None for r in value["operation_coverage_rows"])
    assert value["model_invocation_counts"]["current_llm_invocations"] == 0
    with patch.object(source, "BOARD_PIN", "sha256:wrong"):
        assert not source.history(e.ROOT)["qualified"]
    with patch.object(source, "history", side_effect=lambda p: signal.raise_signal(signal.SIGALRM)):
        with pytest.raises(TimeoutError, match="evidence_scan_60_seconds"):
            e.operands(tmp_path)


def cell(session: int = 1, batch: int = 1) -> dict:
    """SCENARIO-REPORT-8577-BOUNDARY: synthetic clocks test algebra without science credit."""
    return dict(
        arm="resident_python",
        session_panels=session,
        batch=batch,
        repeat=0,
        wall_ns=1000,
        spans=[
            dict(operation=op, start_ns=i * 100, end_ns=(i + 1) * 100)
            for i, op in enumerate(source.OPERATIONS)
        ],
    )


def test_disjoint_cost_algebra() -> None:
    """SCENARIO-REPORT-8577-BOUNDARY: charge host and transfer once per complete session."""
    cells = [cell(), cell(4, 8)]
    rows = source.ceilings(dict(qualified=True, cells=cells))
    assert len(rows) == 4
    assert rows[0]["zero_cost_ceiling"] == pytest.approx(1000 / 700)
    assert rows[1]["zero_cost_ceiling"] == pytest.approx(1000 / 900)
    assert rows[0]["fixed_host_clock_ns"] == 700
    assert rows[0]["nfr01_boundary"] == "insufficient_for_NFR01"
    assert rows[2]["session_panels"] == 4 and rows[2]["batch"] == 8
    for change in ["overlap", "missing", "negative", "total", "duplicate", "bool"]:
        bad = deepcopy(cells)
        if change == "overlap":
            bad[0]["spans"][1]["start_ns"] = 0
        if change == "missing":
            bad[0]["spans"].pop()
        if change == "negative":
            bad[0]["spans"][0]["start_ns"] = -1
        if change == "total":
            bad[0]["wall_ns"] = 900
        if change == "duplicate":
            bad.append(deepcopy(bad[0]))
        if change == "bool":
            bad[0]["spans"][0]["start_ns"] = False
        with pytest.raises(ValueError, match="complete_disjoint"):
            source.ceilings(dict(qualified=True, cells=bad))


def test_private_cli_and_manifest(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8577-REPLAY: genuine children cover publication and CLI rejection."""
    from carnot.reporting import resident_hardware_runner_8577 as runner

    output = tmp_path / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for args, expected in [
        (["--private-e2e", "--root", str(tmp_path / "absent"), "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "absent")], 1),
        (["--deliberate-error"], 7),
        (["--private-e2e"], 2),
        (["--date", "20261015"], 2),
    ]:
        result = subprocess.run(argv + args, capture_output=True, timeout=120, check=False)
        assert result.returncode == expected, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert terminal["consumer"]["passed"]
    assert all(r["passed"] for r in terminal["validation_receipts"])
    plan = runner.manifest(tmp_path / "plan")
    for row in [r for r in plan if "original_node_ids" in r]:
        nodes = [
            n
            for r in plan
            if r.get("suite") == row["suite"]
            for n in r.get("collected_node_ids", [])
        ]
        assert nodes == row["original_node_ids"]
        assert len(nodes) == len(set(nodes))
    assert all("--no-cov" in r["argv"] for r in plan if "collected_node_ids" in r)


def test_qualified_cost_admission(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8577-REPLAY: even a green summary needs bound complete primitives."""
    p = tmp_path / source.COST
    p.parent.mkdir(parents=True)
    primitive = tmp_path / "costs.json"
    cells = [
        dict(cell(s, b), arm=a, repeat=r)
        for a in ["resident_python", "resident_native"]
        for s in [1, 4, 16]
        for b in [1, 8]
        for r in range(10)
    ]
    atomic_json(primitive, dict(cells=cells))
    parity_path = tmp_path / "results/experiment_8572_v736_resident_inference_qualification.json"
    parity_side = parity_path.parent / "raw" / parity_path.stem / "validator.json"
    parity = dict(
        experiment_id=8572,
        resident_inference_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        terminal_validation_sidecar_path=str(tmp_path / "parity_terminal.json"),
    )
    atomic_json(parity_path, parity)
    atomic_json(
        parity_side,
        dict(primary_sha256=e.reference(parity_path)["sha256"], report=dict(passed=True)),
    )
    atomic_json(
        tmp_path / "parity_terminal.json", dict(publication=dict(sidecar_path=str(parity_side)))
    )
    value = dict(
        experiment_id=8573,
        task_id="exp8573-resident-complete-cost",
        run_date="20261016",
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        required_checks_passed=True,
        flagged_adversarial=False,
        resident_complete_cost_ready_score=1,
        resident_inference_ready_score=1,
        resident_parity_reference=e.reference(parity_path),
        resident_operation_clocks_reference=e.reference(primitive),
        terminal_validation_sidecar_path=str(tmp_path / "terminal.json"),
    )
    atomic_json(p, value)
    side = p.parent / "raw" / p.stem / "validator.json"
    atomic_json(side, dict(primary_sha256=e.reference(p)["sha256"], report=dict(passed=True)))
    atomic_json(tmp_path / "terminal.json", dict(publication=dict(sidecar_path=str(side))))
    assert source.costs(tmp_path)["qualified"]
    with patch.object(source, "read_bound_sidecar", return_value=dict(report=dict(passed=False))):
        assert not source.costs(tmp_path)["qualified"]
    with patch.object(
        source,
        "read_bound_sidecar",
        side_effect=[dict(report=dict(passed=True)), dict(report=dict(passed=False))],
    ):
        assert not source.costs(tmp_path)["qualified"]
    atomic_json(primitive, dict(cells=cells[:-1]))
    value["resident_operation_clocks_reference"] = e.reference(primitive)
    atomic_json(p, value)
    atomic_json(side, dict(primary_sha256=e.reference(p)["sha256"], report=dict(passed=True)))
    assert not source.costs(tmp_path)["qualified"]
    changed = deepcopy(value)
    changed["resident_inference_ready_score"] = 0
    atomic_json(p, changed)
    assert not source.costs(tmp_path)["qualified"]
    atomic_json(p, value)
    primitive.write_text('{"cells": []}')
    assert not source.costs(tmp_path)["qualified"]


def test_rejected_original_and_terminal_evidence(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8577-REPLAY: green names cannot admit failed producer seals."""
    board = tmp_path / source.BOARD
    atomic_json(board, dict(required_checks_passed=False, flagged_adversarial=False))
    with patch.object(source, "BOARD_PIN", e.reference(board)["sha256"]):
        assert not source.history(tmp_path)["qualified"]
    with patch.object(source, "read_bound_sidecar", return_value=dict(report=dict(passed=False))):
        assert not source.history(e.ROOT)["qualified"]
    with pytest.raises(ValueError, match="complete_disjoint"):
        source.reconcile([])
