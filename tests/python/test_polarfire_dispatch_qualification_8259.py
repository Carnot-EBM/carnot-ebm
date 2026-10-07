"""REQ-REPORT-8259 and REQ-VERIFY-8259: private tests give no hardware credit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from carnot.reporting import polarfire_dispatch_qualification_8259 as q
from carnot.reporting import polarfire_state_dispatch_8245 as d
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference
from test_polarfire_state_dispatch_8245 import panel, producer


def invoke(*args):
    """An outside-checkout child tests the imports and terminal publication together."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(d.ROOT / q.CLI), *map(str, args)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_authenticated_schema_is_first_failed_operand(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8259-SCHEMA: valid bytes cannot authorize a new schema."""
    data = panel()
    data["state"]["schema_version"] = 2
    primary = producer(tmp_path, 8240, data)
    monkeypatch.setitem(d.PINS, 8240, reference(primary)["sha256"])
    loaded = d.load(tmp_path, tmp_path / "private-primitives")
    failure = next(i for i, row in enumerate(loaded["checks"]) if not row["passed"])
    assert all(row["passed"] for row in loaded["checks"][:failure])
    row = loaded["checks"][failure]
    assert row["artifact_field"] == "required_input_schema_and_hash"
    assert row["observed"] == "state_version"
    assert row["path"] == str(primary)
    assert loaded["state"] is None
    assert row in loaded["learner_block"]


def test_unchanged_manifest_and_current_commands(tmp_path):
    """REQ-REPORT-8259: no legacy command or coverage requirement is weakened."""
    old = q.BASE_COMMANDS(tmp_path / "legacy")
    plan = q.commands(tmp_path)
    assert plan[: len(old)] == old
    assert {"legacy_coverage_json", "current_coverage_json", "e2e015", "e2e019"} <= {
        s.name for s in plan
    }
    assert any(q.MODULE in s.argv for s in plan if "mypy" in s.name)
    assert q.CLI in next(s for s in plan if s.name == "current_changed_module_mypy").argv
    assert all(s.timeout_s <= 180 for s in plan)


def test_private_cli_and_rehashed_tampering(tmp_path):
    """SCENARIO-REPORT-8259-CLI: fresh replay rejects altered reductions and clocks."""
    source, output = tmp_path / "fixture.json", tmp_path / (q.NAME + ".json")
    atomic_json(source, panel())
    run = invoke("--input", source, "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    original = json.loads(output.read_bytes())
    assert original["experiment_id"] == 8259
    assert original["task_id"] == "exp8259-polarfire-dispatch-qualification"
    assert original["milestone"] == "2026.10.713"
    assert original["MODEL_SPECS"] == [] and original["current_model_calls"] == 0
    assert original["current_device_execution_count"] == 0
    assert original["independent_generalization_score"] == 0
    assert original["generalized_learning_benefit_score"] == 0
    assert original["coverage_statement_counts"]["measured"] is False
    assert original["transfer_seconds"] is None and original["board_cpu_seconds"] is None
    assert invoke("--cold-replay", output).returncode == 0
    failed = deepcopy(original)
    failed["validation_receipts"].append(
        dict(
            name="owned_failure",
            passed=False,
            actual_exit=1,
            expected_exit=0,
            stdout_path=str(source),
        )
    )
    qualified = q.qualify(failed)
    assert qualified["gate_check_summary"][-1]["artifact_field"] == "owned_failure.actual_exit"
    assert qualified["gate_check_summary"][-1]["observed"] == 1
    for key, changed in [
        ("transfer_seconds", 99),
        ("completed_count", 999),
        ("coverage_statement_counts", {}),
        ("experiment_id", 8245),
    ]:
        value = deepcopy(original)
        value[key] = changed
        value["reproducibility_checksum"] = q.old.checksum(value)
        atomic_json(output, value)
        assert invoke("--cold-replay", output).returncode == 1
    atomic_json(output, original)
    packet = Path(original["packet_path"])
    changed = json.loads(packet.read_bytes())
    changed["schema"] = "bad"
    atomic_json(packet, changed)
    assert invoke("--cold-replay", output).returncode == 1
    assert invoke("--evaluate", packet).returncode == 1
    assert invoke("--date", "bad").returncode == 2
    assert invoke("--input", tmp_path / "absent", "--output", output).returncode == 1
    assert invoke("--input", source, "--output", d.ROOT / "results" / output.name).returncode == 1
    data = panel()
    data.update(state=None, queries=[], checks=[d.operand("state", source, True, False)])
    atomic_json(source, data)
    assert invoke("--input", source, "--output", output).returncode == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    assert invoke("--cold-replay", output).returncode == 0


def test_primitive_coverage_and_clocks(tmp_path):
    """REQ-VERIFY-8259: measured counts and times come from authenticated receipts."""
    stream = tmp_path / "coverage.json"
    atomic_json(stream, {"totals": {"num_statements": 10, "covered_lines": 10, "missing_lines": 0}})
    receipt = dict(
        name="current_coverage_json",
        stdout_path=str(stream),
        stdout_sha256=reference(stream)["sha256"],
    )
    value = dict(validation_receipts=[receipt], board_reference=reference(stream))
    assert q.coverage_counts(value)["covered"] == 10
    timing = tmp_path / "board.stderr"
    atomic_json(timing, dict(board_cpu_seconds=0.25))
    atomic_json(
        stream,
        dict(
            receipts=[
                dict(name="board_transfer", duration_s=2),
                dict(
                    name="board_evaluate",
                    duration_s=3,
                    stderr_path=str(timing),
                    stderr_sha256=reference(timing)["sha256"],
                ),
            ]
        ),
    )
    value["board_reference"] = reference(stream)
    assert q.board_clocks(value) == {"transfer_seconds": 2, "board_cpu_seconds": 0.25}
    stream.write_text("tampered")
    with pytest.raises(ValueError):
        q.board_clocks(value)


def test_board_timer_keeps_existing_evaluator(tmp_path, monkeypatch):
    """REQ-VERIFY-8259: CPU time is measured on the board around the existing receiver."""
    plans = []
    monkeypatch.setattr(q, "BASE_EXECUTE", lambda plan, raw: plans.extend(plan) or [])
    spec = q.CommandSpec(
        "board_evaluate",
        (*d.SSH, "python3 -u /tmp/carnot8245-test/evaluator.py /tmp/carnot8245-test/packet.json"),
        "board",
        60,
    )
    assert q.execute([spec], tmp_path) == []
    assert plans[0].argv[:-1] == d.SSH
    assert "process_time" in plans[0].argv[-1]
    assert "runpy.run_path" in plans[0].argv[-1]
    assert "packet.json" in plans[0].argv[-1]
    assert plans[0].timeout_s == 60
    data = panel()
    d.packet(data, tmp_path)
    spec = q.CommandSpec(
        "board_evaluate",
        (*d.SSH, f"python3 -u {tmp_path}/evaluator.py {tmp_path}/packet.json"),
        "board",
        60,
    )
    q.execute([spec], tmp_path)
    child = subprocess.run(
        shlex.split(plans[-1].argv[-1]), capture_output=True, text=True, timeout=60
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert json.loads(child.stdout) == d.expected(data)
    assert json.loads(child.stderr)["board_cpu_seconds"] >= 0


def test_history_and_natural_state_only(tmp_path, monkeypatch):
    """REQ-REPORT-8259: historical failures survive; static substitution is forbidden."""
    history = q.historical(tmp_path / "history")
    assert history["reproduced"]
    assert history["exp8245_primary"]["sha256"] == q.HISTORICAL_PIN
    assert history["exp8245_failed_receipts"]
    monkeypatch.setattr(q, "BASE_LOAD", lambda root, raw: panel())
    loaded = q.load(tmp_path, tmp_path / "raw")
    assert loaded["state"] is None and loaded["queries"] == []
    assert loaded["checks"][-1]["artifact_field"] == "state_origin.kind"
