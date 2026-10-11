"""REQ-REPORT-8413 / REQ-VERIFY-8413: board custody grants no new acceleration."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import board_operation_boundary_8413 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_map_and_optional_costs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8413-BOUNDARY: absent transfer clocks are not zero costs."""
    mapped, fraction, missing = e.operation_map([])
    assert fraction is None and missing == e.OPERATIONS
    assert all(r["cost_ns"] is None for r in mapped)
    rows = [dict(operation=op, cost_ns=1) for op in e.OPERATIONS]
    assert e.operation_map(rows)[1] == 0
    for bad in [-1, True, float("nan")]:
        with pytest.raises(ValueError):
            e.operation_map([dict(operation=e.OPERATIONS[0], cost_ns=bad)])
    work = dict(checks=[], refs=[])
    assert e.cost_branch(tmp_path / "missing.json", tmp_path, work)["disposition"] == "unavailable"
    path = tmp_path / (e.SOURCES[1] + ".json")
    atomic_json(path, dict(status="blocked", honest_verdict="blocked_gate_check_failed"))
    assert e.cost_branch(path, tmp_path, work)["disposition"] == "unavailable"


def test_measure_replay_tamper_and_failure(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8413-REPLAY: rehashing cannot authorize changed board claims."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(e.ROOT, raw)
    value = e.build(work, [dict(passed=True)], raw, output)
    assert value["board_reader_ready_score"] == value["current_contract_ready_score"] == 1
    assert value["experiment_id"] == 8413 and value["run_date"] == "20261011"
    assert value["polarfire_workload_validated"]
    assert value["polarfire_execution_scope"] == "board_local_Linux_CPU_only"
    assert value["eligible_operation_count"] == value["device_command_count"] == 0
    assert value["kv260_k_max"] == 5 and value["MODEL_SPECS"] == []
    assert value["eligible_cost_fraction"] is None and not value["hardware_purchase_required"]
    assert set(value) <= set(value["field_principles"])
    atomic_json(output, value)
    assert e.replay(output)
    for key in ["board_reader_ready_score", "device_command_count", "polarfire_execution_scope"]:
        changed = deepcopy(value)
        changed[key] = "tampered"
        changed["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        atomic_json(output, changed)
        assert not e.replay(output)
    atomic_json(output, value)
    source = Path(work["refs"][-1]["path"])
    source.write_bytes(b"changed source")
    assert not e.replay(output)
    assert e.build(work, [dict(passed=False)], raw, output)["verdict_class"] == "disqualified"


def test_missing_and_pins(tmp_path: Path) -> None:
    """REQ-REPORT-8413: unchanged absence blocks and forged terminal pins fail."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(tmp_path / "absent", raw)
    value = e.build(work, [dict(passed=True)], raw, output)
    assert value["verdict_class"] == "blocked" and value["board_reader_ready_score"] == 0
    atomic_json(output, value)
    assert e.replay(output)
    assert not e.replay(tmp_path / "missing")
    with patch.object(e, "READER_PIN", "wrong"):
        assert not e.operands(e.ROOT, tmp_path / "wrong")["reader_ready"]
    original_probe = e.base.probe

    def drifted(path: Path, raw: Path, work: dict, field: str | None) -> dict | None:
        result = original_probe(path, raw, work, field)
        if path == e.ROOT / e.READER and result is not None:
            result = dict(result, polarfire_graduation={})
        return result

    with patch.object(e.base, "probe", drifted):
        assert not e.operands(e.ROOT, tmp_path / "dispatch-drift")["reader_ready"]


def test_real_cli_and_owned_failure(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8413-CONTROLS: genuine children bind private terminal bytes."""
    from carnot.reporting import board_operation_runner_8413 as runner
    from carnot.reporting.primary_publication import reader_receipt

    cli = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    output = tmp_path / (e.NAME + ".json")
    for args, expected in [
        (["--private-e2e", "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "missing")], 1),
        (["--deliberate-error"], 1),
        (["--private-e2e"], 2),
        (["--date", "20261010"], 2),
    ]:
        result = subprocess.run([*cli, *args], capture_output=True, text=True, timeout=180)
        assert result.returncode == expected, result.stdout + result.stderr
    assert reader_receipt(e.TASK, tmp_path, field="board_reader_ready_score")["passed"]
    historical = tmp_path / (e.prior.NAME + ".json")
    for args, expected in [
        (["--private-e2e", "--output", str(historical)], 0),
        (["--cold-replay", str(historical)], 0),
    ]:
        result = subprocess.run(
            [*cli, "--historical-argv", str(e.ROOT / e.prior.CLI), *args],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == expected, result.stdout + result.stderr
    assert reader_receipt(e.prior.TASK, tmp_path, field="board_reader_ready_score")["passed"]
    with pytest.raises(ValueError, match="historical_cli_identity"):
        runner.historical_main(["wrong-script"])
    ran = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "coverage",
            "run",
            "--parallel-mode",
            "--data-file=" + str(tmp_path / ".coverage"),
            str(e.ROOT / e.CLI),
            "--deliberate-error",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert ran.returncode == 1 and list(tmp_path.glob(".coverage*"))
    plan = runner.manifest(tmp_path)
    assert any("E2E018" in r["name"] for r in plan)
    failed = [
        dict(
            name="deliberate_error",
            argv=[sys.executable, "-u", "-c", "raise SystemExit(3)"],
            deadline=10,
            expected=0,
            scope="owned",
        )
    ]
    with patch.object(runner, "manifest", return_value=failed):
        assert runner.run(tmp_path / "missing", output, tmp_path) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and not value["board_reader_ready_score"]
    assert value["validation_receipts"][0]["exit_code"] == 3


def test_cost_authentication_and_reduction_failures(tmp_path: Path) -> None:
    """REQ-REPORT-8413: typed private controls reject unqualified and altered costs."""
    path = tmp_path / (e.SOURCES[0] + ".json")
    raw = tmp_path / "raw" / path.stem
    raw.mkdir(parents=True)
    cell = dict(workload="natural", batch=1, predictions_per_update=8)
    summaries = [
        dict(
            cell=cell,
            repeat=i,
            repeat_id=str(i),
            stage_ns=dict(input_validation=1),
            serialized_bytes=10,
        )
        for i in range(1, 6)
    ]
    refs = []
    for i in range(1, 6):
        primitive = raw / f"primitive{i}.json"
        atomic_json(primitive, dict(repeat_id=str(i), start_ns=1, end_ns=3))
        refs.append(dict(e.base.reference(primitive), warmup=False))
    source = raw / "work.json"
    atomic_json(source, dict(bundle={}, results=refs))
    terminal, sidecar = raw / "terminal.json", raw / "bound.json"
    value = dict(
        experiment_id=8407,
        task_id="exp8407-python-transaction-cost",
        milestone=e.MILESTONE,
        honest_verdict="complete_null_cost",
        verdict_class="null",
        required_checks_passed=True,
        flagged_adversarial=False,
        python_cost_ready_score=1,
        raw_shard_hashes=refs,
        terminal_validation_sidecar_path=str(terminal),
        work_reference=e.base.reference(source),
        primary_cell=dict(cell=cell),
        repeat_rows=summaries,
    )

    def check(changed: dict) -> dict:
        atomic_json(path, changed)
        publication = dict(
            primary_path=str(path),
            primary_sha256=e.base.sha256_file(path),
            sidecar_path=str(sidecar),
        )
        atomic_json(terminal, publication)
        atomic_json(sidecar, dict(publication, report=dict(passed=True)))
        with (
            patch.object(
                e.costs.k,
                "reduce_repeat",
                side_effect=lambda bundle, p: summaries[int(p["repeat_id"]) - 1],
            ),
            patch.object(e, "validate_primary"),
        ):
            return e.cost_branch(path, raw / "sealed", dict(checks=[], refs=[]))

    assert check(value)["disposition"] == "qualified_primary_cell"
    for key, replacement in [
        ("required_checks_passed", False),
        ("verdict_class", "blocked"),
        ("milestone", "wrong"),
        ("repeat_rows", summaries[:-1]),
        ("repeat_rows", [dict(r, serialized_bytes=11) for r in summaries]),
    ]:
        changed = dict(value, **{key: replacement})
        assert check(changed)["disposition"] == "unavailable"
