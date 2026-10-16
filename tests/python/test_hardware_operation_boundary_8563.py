"""REQ-REPORT-8563, REQ-VERIFY-8563: scope accounting grants no physical readiness."""

from copy import deepcopy
import json
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import hardware_operation_boundary_8563 as e
from carnot.reporting import hardware_operation_evidence_8563 as source
from carnot.reporting.current_work_receipt import atomic_json


def test_missing_owned_failure_and_rehashed_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8563-REPLAY: missing evidence differs from failed owned checks."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(tmp_path / "absent", raw, tmp_path)
    value = e.build(work, [dict(passed=True)], raw, output)
    assert value["verdict_class"] == "blocked"
    assert value["hardware_boundary_ready_score"] == 0
    assert value["physical_hardware_ready_score"] == 0
    assert len(value["operation_coverage_rows"]) == 45
    assert all(r["cost_ns"] is None for r in value["operation_coverage_rows"])
    assert all(r["zero_cost_ceiling"] is None for r in value["conditional_ceiling_rows"])
    assert value["model_invocation_counts"]["current_llm_invocations"] == 0
    assert set(value) <= set(value["field_principles"])
    atomic_json(output, value)
    assert e.replay(output) and not e.replay(tmp_path / "absent.json")
    value["physical_hardware_ready_score"] = 1
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)
    changed = deepcopy(work)
    changed["evidence"]["history"]["qualified"] = True
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, [dict(passed=True)], raw, output))
    assert not e.replay(output)
    assert e.build(work, [dict(passed=False)], raw, output)["verdict_class"] == "disqualified"


def test_original_scopes_costs_and_ceiling(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8563-BOUNDARY: qualified original costs retain complete fixed work."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw, tmp_path)
    value = e.build(work, [dict(passed=True)], raw, tmp_path / (e.NAME + ".json"))
    assert value["hardware_boundary_ready_score"] == 1
    assert value["physical_hardware_ready_score"] == 0
    assert value["run_date"] == "20261016"
    assert value["qualified_cost_reference"]["qualified"]
    assert len(value["conditional_ceiling_rows"]) == 8
    assert all(r["zero_cost_ceiling"] < 10 for r in value["conditional_ceiling_rows"])
    assert all(
        r["nfr01_boundary"] == "insufficient_for_NFR01" for r in value["conditional_ceiling_rows"]
    )
    assert value["last_authenticated_idcode"] == "0xffffffff"
    assert value["kv260_supported_operations"][0]["k_max"] == 5
    assert not value["polarfire_graduation"]["fpga_fabric_acceleration"]
    assert len(value["missing_source_hashes"]) == 2
    assert all(not r["device_supported"] for r in value["operation_coverage_rows"])
    assert all(
        r["cost_ns"] is None
        for r in value["operation_coverage_rows"]
        if r["operation"] == "host_device_transfer"
    )
    assert value["vendor_exclusion_rows"][0]["excluded_operations"] == [
        "dense_final_logits",
        "inter_device_movement",
    ]
    assert all(not r["local_measurement"] for r in value["literature_hypotheses"])
    assert len(value["board_reopen_requirements"]) == 5
    gate = next(r for r in value["board_reopen_requirements"] if r["board"] == "GateMate")
    assert gate["expected_idcode"] == "0x20000001"
    assert "dated_physical_cable_port_power_evidence" in gate["required"]
    with patch.object(source, "HISTORY_PIN", "sha256:wrong"):
        assert not source.history(e.ROOT)["qualified"]
    with patch.object(source, "read_bound_sidecar", return_value=dict(report=dict(passed=False))):
        assert not source.history(e.ROOT)["qualified"]
        assert not source.costs(e.ROOT)["qualified"]
    history = tmp_path / source.HISTORY
    atomic_json(
        history, dict(json.loads((e.ROOT / source.HISTORY).read_bytes()), run_date="20261014")
    )
    with patch.object(source, "HISTORY_PIN", e.reference(history)["sha256"]):
        assert not source.history(tmp_path)["qualified"]


def test_cost_rejection_paths(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8563-REPLAY: repaired hashes cannot replace cost qualification."""
    assert source.costs(tmp_path)["checks"][0]["observed"] is None
    original = json.loads((e.ROOT / source.COST).read_bytes())
    path = tmp_path / source.COST
    for updates in [{}, dict(run_date="20261015"), dict(complete_native_cost_ready_score=0)]:
        atomic_json(path, dict(original, **updates))
        with patch.object(source, "COST_PIN", e.reference(path)["sha256"]):
            assert not source.costs(tmp_path)["qualified"]
    with patch.object(source, "COST_PIN", "sha256:wrong"):
        assert not source.costs(e.ROOT)["qualified"]
    with patch.object(source.native, "consume", side_effect=ValueError("rehashed primitive")):
        assert not source.costs(e.ROOT)["qualified"]
    with patch.object(source.native, "consume", return_value=dict(ready=True)):
        assert not source.costs(e.ROOT)["qualified"]
    cost = source.costs(e.ROOT)
    with pytest.raises(ValueError, match="reconciled_complete_costs"):
        source.reconcile([])
    cells = deepcopy(cost["cells"])
    cells[0]["seconds"]["startup"] = -1
    with pytest.raises(ValueError, match="reconciled_complete_costs"):
        source.reconcile(cells)
    cells = deepcopy(cost["cells"])
    cells[0]["overlap_s"] = 0.01
    with pytest.raises(ValueError, match="reconciled_complete_costs"):
        source.reconcile(cells)
    cells = deepcopy(cost["cells"])
    cells[0]["wall_s"] += 1
    with pytest.raises(ValueError, match="reconciled_complete_costs"):
        source.reconcile(cells)
    cells = deepcopy(cost["cells"])
    for cell in cells:
        cell["seconds"]["binding_residual"] = 100
        cell["wall_s"] = sum(cell["seconds"].values())
    cost["cells"] = cells
    assert source.ceilings(cost)[0]["nfr01_boundary"] == "conditional_only"
    with patch.object(source, "reconcile", side_effect=ValueError("bad scope")):
        assert not source.costs(e.ROOT)["qualified"]


def test_private_cli(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8563-CLI: actual children check publication and terminal rejection."""
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


def test_manifest_and_timeout(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8563-CLI: original node unions and real scan deadlines stay bounded."""
    from carnot.reporting import hardware_operation_runner_8563 as runner

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
    with patch.object(source, "history", side_effect=lambda p: signal.raise_signal(signal.SIGALRM)):
        with pytest.raises(TimeoutError, match="evidence_scan_60_seconds"):
            e.operands(tmp_path)


def test_terminal_union(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8563-CLI: actual node outcomes and log seals decide union acceptance."""
    node = e.TEST + "::test_original_scopes_costs_and_ceiling"
    log = tmp_path / "nodes.stdout"
    log.write_text(node + " PASSED [100%]\n")
    atomic_json(
        tmp_path / "execution_manifest.json",
        dict(
            commands=[
                dict(name="collection", original_node_ids=[node]),
                dict(name="owned", collected_node_ids=[node]),
            ]
        ),
    )
    receipts = [
        dict(
            name="owned",
            passed=True,
            stdout_path=str(log),
            stdout_sha256=e.reference(log)["sha256"],
        )
    ]
    assert e.node_union(tmp_path, receipts)["passed"]
    log.write_text(node + " FAILED [100%]\n")
    assert not e.node_union(tmp_path, receipts)["passed"]
