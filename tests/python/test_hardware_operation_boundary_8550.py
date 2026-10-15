"""REQ-REPORT-8550, REQ-VERIFY-8550: complete scope grants no physical execution."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import hardware_operation_boundary_8550 as e
from carnot.reporting import hardware_operation_evidence_8550 as source
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary


def fixture_cost(tmp_path: Path) -> Path:
    """SCENARIO-REPORT-8550-BOUNDARY: private clocks test gates without inventing measurements."""
    spans = [
        dict(operation=op, start_ns=i * 10, end_ns=(i + 1) * 10)
        for i, op in enumerate(source.OPERATIONS)
    ]
    primitive = tmp_path / "spans.json"
    atomic_json(
        primitive,
        dict(
            spans=spans, complete_clock_ns=90, input_count=9, coefficient_count=109, reconciled=True
        ),
    )
    terminal = tmp_path / "terminal.json"
    value = dict(
        experiment_id=8546,
        task_id="exp8546-complete-native-cost",
        run_date="20261015",
        honest_verdict="complete_null_private_cost",
        verdict_class="null",
        complete_native_cost_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        terminal_validation_sidecar_path=str(terminal),
        disjoint_complete_spans_reference=e.reference(primitive),
    )
    path = tmp_path / source.COST
    atomic_json(
        terminal, dict(publication=publish_primary(path, value, lambda p: dict(passed=True)))
    )
    return path


def test_missing_replay_and_owned_failure(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8550-BOUNDARY: missing differs from zero and owned failures disqualify."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(tmp_path / "absent", raw, tmp_path)
    value = e.build(work, [dict(passed=True)], raw, output)
    assert value["verdict_class"] == "blocked"
    assert value["physical_hardware_ready_score"] == 0
    assert value["hardware_boundary_ready_score"] == 0
    assert len(value["operation_coverage_rows"]) == 45
    assert len({r["operation"] for r in value["operation_coverage_rows"]}) == 9
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


def test_original_and_vendor_scope(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8550-BOUNDARY: independent original pins retain board and vendor exclusions."""
    work = e.measure(e.ROOT, tmp_path / "raw", tmp_path)
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["hardware_boundary_ready_score"] == 1
    assert value["last_authenticated_idcode"] == "0xffffffff"
    assert len(value["missing_source_hashes"]) == 2
    assert value["kv260_supported_operations"][0]["k_max"] == 5
    assert not value["polarfire_graduation"]["fpga_fabric_acceleration"]
    assert all(not r["device_supported"] for r in value["operation_coverage_rows"])
    assert value["vendor_exclusion_rows"][0]["excluded_operations"] == [
        "dense_final_logits",
        "inter_device_movement",
    ]
    assert all(r["local_measurement"] is False for r in value["literature_hypotheses"])
    assert len(value["board_reopen_requirements"]) == 5
    with patch.object(source, "HISTORY_PIN", "sha256:wrong"):
        assert not source.history(e.ROOT)["qualified"]


def test_cost_validation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8550-BOUNDARY: only bound, complete, disjoint original costs qualify."""
    assert source.costs(tmp_path)["checks"][0]["observed"] is None
    path = fixture_cost(tmp_path)
    cost = source.costs(tmp_path)
    assert cost["qualified"]
    assert cost["conditional_ceiling"]["ceiling"] == 1.5
    with patch.object(source, "read_bound_sidecar", return_value=dict(report=dict(passed=False))):
        assert not source.costs(tmp_path)["qualified"]
    path.write_text(path.read_text() + " ")
    assert not source.costs(tmp_path)["qualified"]
    fixture_cost(tmp_path)
    value = json.loads(path.read_bytes())
    value["complete_native_cost_ready_score"] = 0
    atomic_json(path, value)
    assert source.costs(tmp_path)["checks"][0]["observed"] == 0
    spans = cost["spans"]
    for bad in [
        [],
        spans[:-1],
        [dict(r, start_ns=0) for r in spans],
        [dict(r, end_ns=True) for r in spans],
        [dict(r, start_ns=0, end_ns=0) for r in spans],
    ]:
        with pytest.raises(ValueError):
            source.ceiling(bad)
    fixture_cost(tmp_path)
    with patch.object(source, "ceiling", side_effect=ValueError("invalid clocks")):
        assert not source.costs(tmp_path)["qualified"]
    work = e.measure(tmp_path, tmp_path / "raw", tmp_path)
    value = e.build(work, [dict(passed=True)], tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["operation_coverage_rows"][0]["cost_ns"] == 10
    assert value["conditional_ceiling_rows"][0]["zero_cost_ceiling"] == 1.5
    assert value["conditional_ceiling_rows"][0]["nfr01_boundary"] == "insufficient_for_NFR01"


def test_private_cli(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8550-CLI: real children publish privately and reject unsafe invocations."""
    output = tmp_path / (e.NAME + ".json")
    cli = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for args, expected in [
        (["--private-e2e", "--root", str(tmp_path / "absent"), "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "absent.json")], 1),
        (["--deliberate-error"], 7),
        (["--private-e2e"], 2),
        (["--date", "20261014"], 2),
    ]:
        child = subprocess.run(cli + args, capture_output=True, timeout=180, check=False)
        assert child.returncode == expected, child.stdout + child.stderr
    value = json.loads(output.read_bytes())
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert terminal["consumer"]["passed"]
    assert all(r["passed"] for r in terminal["validation_receipts"])


def test_manifest_and_bound(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8550-CLI: original node unions are preserved and scan deadlines are real."""
    import signal
    from carnot.reporting import hardware_operation_runner_8550 as runner

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


def test_primitive_metadata_and_nfr_gate(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8550-BOUNDARY: repaired primitive hashes cannot conceal wrong cost scope."""
    fixture_cost(tmp_path)
    with patch.object(source, "read_bound_sidecar", return_value=dict(report=dict(passed=True))):
        primitive = tmp_path / "spans.json"
        original = json.loads(primitive.read_bytes())
        for updates in [
            dict(input_count=5),
            dict(coefficient_count=108),
            dict(reconciled=False),
            dict(complete_clock_ns=91),
        ]:
            atomic_json(primitive, dict(original, **updates))
            path = tmp_path / source.COST
            value = json.loads(path.read_bytes())
            value["disjoint_complete_spans_reference"] = e.reference(primitive)
            atomic_json(path, value)
            assert not source.costs(tmp_path)["qualified"]
    fixture_cost(tmp_path)
    cost = source.costs(tmp_path)
    clocks = deepcopy(cost["spans"])
    cursor = 0
    for row in clocks:
        size = 1000 if row["operation"] == "cubic_residual" else 1
        row.update(start_ns=cursor, end_ns=cursor + size)
        cursor += size
    cost.update(spans=clocks, conditional_ceiling=source.ceiling(clocks))
    assert source.ceilings(cost)[0]["nfr01_boundary"] == "conditional_only"


def test_original_date_and_terminal_rejection(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8550-BOUNDARY: an envelope repair cannot replace dates or terminal approval."""
    path = tmp_path / source.HISTORY
    original = json.loads((e.ROOT / source.HISTORY).read_bytes())
    for updates in [dict(run_date="20261014"), dict(required_checks_passed=False), {}]:
        atomic_json(path, dict(original, **updates))
        with patch.object(source, "HISTORY_PIN", e.reference(path)["sha256"]):
            with patch.object(
                source, "read_bound_sidecar", return_value=dict(report=dict(passed=False))
            ):
                assert not source.history(tmp_path)["qualified"]
