"""REQ-REPORT-8202 / REQ-VERIFY-8202: authenticated delta and private custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import sys

import pytest

from carnot.reporting import arc_supervisor_v708_frontier as task
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_8202_v708_arc_supervisor_frontier as cli
from test_arc_supervisor_qualification_8001 import live_episode
from test_arc_supervisor_refinement_7936 import producer


def test_preconditions_and_missing_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8202: the actual missing operand blocks rather than inventing history."""
    monkeypatch.setattr(task, "STATE", tmp_path / "missing-state.json")
    checked = task.inputs()
    assert checked["registry_precheck"]["ar25"] == 8
    assert checked["prior"]["experiment_id"] == 8189
    assert [r["artifact_field"] for r in checked["failures"]] == ["live_state_is_file"]
    assert checked["failures"][0]["observed"] is False
    atomic_json(task.STATE, {})
    assert not task.inputs()["failures"]
    monkeypatch.setitem(task.scope.PINNED, str(task.SIDECAR), "sha256:wrong")
    assert any(r["artifact_field"] == "sha256" for r in task.inputs()["failures"])


def test_clock_identity_and_unchanged_bytes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8202-FRONTIER: a file mtime cannot suppress a fresh event."""
    event = live_episode()
    event["trajectory_supervisor"]["redirects"] *= 2
    source = producer(tmp_path, [event])
    os.utime(source, (1, 1))
    checked = dict(prior={"finished_at": "2000-01-01T00:00:00Z"}, inventory={})
    first = task.scan(tmp_path, [source], checked, tmp_path / "one", current_date="20261006")
    assert len(first["new_event_rows"]) == 1
    assert all(k in first["rows"][0] for k in ("condition", "metric", "numerator", "denominator"))
    second = task.scan(
        tmp_path,
        [source],
        dict(prior=checked["prior"], inventory=first),
        tmp_path / "two",
        current_date="20261006",
    )
    assert not second["new_event_rows"]
    checked["prior"]["finished_at"] = "2026-10-07T00:00:00Z"
    assert not task.scan(tmp_path, [source], checked, tmp_path / "old", current_date="20261006")[
        "new_event_rows"
    ]


@pytest.mark.parametrize("state", ["null", "blocked", "disqualified"])
def test_readiness_and_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    """SCENARIO-VERIFY-8202-COLD: rehash logs and independently reject headline drift."""
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    value = task.scan(
        tmp_path, [], dict(prior={}, inventory={}), tmp_path / "scan", current_date="20261006"
    )
    inventory = tmp_path / "inventory.json"
    atomic_json(inventory, value)
    log = tmp_path / "validation.log"
    log.write_text("owned validation")
    value.update(
        verdict_class=state,
        gate_check_summary=[],
        source_artifact_hashes={str(inventory): sha256_file(inventory)},
        validation_receipts=[
            dict(
                name="cli_replay",
                classification="required",
                passed=True,
                normal_exit=True,
                log_path=str(log),
                log_sha256=sha256_file(log),
            )
        ],
    )
    if state == "blocked":
        value["gate_check_summary"] = [
            task.runner.previous.operand(
                tmp_path / "missing", "live_state_is_file", True, False, None
            )
        ]
    value.update(task.artifact_fields(value, {}))
    assert value["arc_evidence_ready_score"] == int(state == "null")
    assert value["new_level_solves_claimed"] is False
    assert value["arm_recommendations"] == []
    assert value["new_outcome_count"] == 0
    assert not task.replay(value)
    changed = deepcopy(value)
    changed["new_outcome_count"] = 1
    assert "new_outcome_count" in task.replay(changed)
    log.write_text("tampered")
    assert "validation_log_sha256" in task.replay(value)
    log.write_text("owned validation")
    inventory.write_text("tampered")
    assert any(e.startswith("source_sha256:") for e in task.replay(value))
    fixture = dict(task.runner.previous.reduce([]), rows=[])
    assert not task.replay(fixture)


def test_frozen_private_cli_and_entrypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-8202: private success, missing input, tamper and cold replay exit normally."""
    specs = task.commands(tmp_path)
    assert any(s["name"] == "full_python_suite" for s in specs)
    assert all("::" not in a for s in specs if s["name"] == "spec_coverage" for a in s["argv"])
    for spec in specs:
        if spec["name"].startswith("cli_"):
            row = task.runner.previous.run(spec, tmp_path, tmp_path / "logs")
            assert row["passed"], row["output_tail"]
    monkeypatch.setattr(cli, "run", lambda *a, **k: 0)
    assert cli.main([]) == 0
    monkeypatch.setattr(task.scope, "execute", lambda *a: 0)
    monkeypatch.setattr(sys, "argv", [str(task.scope.ROOT / task.CLI)])
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(task.scope.ROOT / task.CLI), run_name="__main__")
    assert exited.value.code == 0
    monkeypatch.setattr(task.runner, "execute", lambda *a: 0)
    assert task.execute(tmp_path / "unused", tmp_path) == 0


def test_terminal_and_observational_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8202: a single resolved outcome cannot recommend an arm."""
    source = producer(tmp_path, [live_episode()])
    value = task.scan(
        tmp_path,
        [source],
        dict(prior={"finished_at": "2000-01-01T00:00:00Z"}, inventory={}),
        tmp_path / "scan",
        current_date="20261006",
    )
    value.update(
        verdict_class="null",
        gate_check_summary=[],
        source_artifact_hashes={},
        validation_receipts=[],
    )
    value.update(task.artifact_fields(value, {}))
    assert value["new_outcome_count"] == 1
    assert value["per_arm_results"]["drop_goal_bias"]["resolved_by_levelup"] == 1
    assert value["arm_recommendations"] == []
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    monkeypatch.setattr(task.previous, "terminal", lambda *a: dict(passed=True, reports=[]))
    assert task.terminal(candidate, tmp_path, tmp_path)["passed"]
    assert json.loads(candidate.read_text())["new_level_solves_claimed"] is False
