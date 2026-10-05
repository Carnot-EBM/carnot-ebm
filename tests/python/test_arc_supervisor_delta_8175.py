"""REQ-REPORT-8175 / REQ-VERIFY-8175: reuse authority and retain honest nulls."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot.reporting import arc_supervisor_v706_delta as task
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_8175_v706_arc_supervisor_delta as cli
from test_arc_supervisor_qualification_8001 import live_episode
from test_arc_supervisor_refinement_7936 import producer


def test_authenticate_reader_and_missing_operand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8175-FRONTIER: a mismatch cannot renew reader authority."""
    checked = task.inputs()
    assert not checked["failures"]
    assert checked["prior"]["experiment_id"] == 8161
    assert checked["registry_precheck"]["ar25"] == 8
    assert len(checked["prior"]["historical_required_failures"]) >= 2
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(task.scope, "PRIOR", missing)
    monkeypatch.setitem(task.scope.PINNED, str(missing), "sha256:" + "0" * 64)
    assert task.inputs()["failures"][0]["observed"] == "missing"


def test_changed_reader_hash_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8175: saved hashes are expected operands, never refreshed."""
    original = task.runner.inputs

    def changed(scope: object) -> dict:
        checked = original(scope)
        checked["prior"] = deepcopy(checked["prior"])
        checked["prior"]["reader_receipt"]["current_code_hashes"][task.baseline.MODULE] = (
            "sha256:wrong"
        )
        return checked

    monkeypatch.setattr(task.runner, "inputs", changed)
    failed = task.inputs()["failures"]
    assert any(r["expected"] == "sha256:wrong" and r["observed"] != r["expected"] for r in failed)


def test_frontier_duplicate_and_old_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8175-FRONTIER: event identity survives duplicate receipts."""
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    original_output = task.baseline.scope.OUTPUT
    event = live_episode()
    event["trajectory_supervisor"]["redirects"] *= 2
    event["trajectory_supervisor"]["stagnations_unredirected"] = 3
    source = producer(tmp_path, [event])
    checked = dict(prior={"finished_at": "2000-01-01T00:00:00Z"}, inventory={})
    first = task.scan(tmp_path, [source], checked, tmp_path / "first", current_date="20261005")
    assert len(first["new_event_rows"]) == 1
    assert len(first["per_game_results"]["g1"]) == 1
    assert first["per_game_results"]["g1"][0]["resolved_by_levelup"] is True
    assert (
        first["rows"][0]["source_events"][0]["trajectory_supervisor"]["stagnations_unredirected"]
        == 3
    )
    assert task.baseline.scope.OUTPUT == original_output
    second = task.scan(
        tmp_path,
        [source],
        dict(prior=checked["prior"], inventory=first),
        tmp_path / "second",
        current_date="20261005",
    )
    assert not second["new_event_rows"]
    checked["prior"]["finished_at"] = "2026-10-01T00:00:00Z"
    assert not task.scan(tmp_path, [source], checked, tmp_path / "old", current_date="20261005")[
        "new_event_rows"
    ]


@pytest.mark.parametrize("state", ["null", "blocked", "disqualified"])
def test_readiness_and_terminal_class(tmp_path: Path, state: str) -> None:
    """REQ-REPORT-8175: readiness never claims scientific benefit or solve credit."""
    value = dict(
        task.runner.previous.reduce([]),
        rows=[],
        verdict_class=state,
        validation_receipts=[],
        source_artifact_hashes={},
        gate_check_summary=[],
    )
    if state == "blocked":
        value["gate_check_summary"] = [
            task.runner.previous.operand(
                tmp_path / "missing", "sha256", "expected", "missing", None
            )
        ]
    assert task.artifact_fields(value, {})["supervisor_reader_ready_score"] == 0
    value["reader_conformance_rows"] = task.baseline.baseline.conformance(tmp_path / "controls")
    value["validation_receipts"] = [
        dict(name="cli_replay", classification="required", passed=True, normal_exit=True)
    ]
    fields = task.artifact_fields(value, {})
    assert fields["supervisor_reader_ready_score"] == int(state == "null")
    assert fields["new_outcome_ready_score"] == 0
    assert fields["arm_recommendations"] == []
    assert fields["new_solve_claim"] is False
    assert (
        fields["independent_generalization_score"]
        == fields["generalized_learning_benefit_score"]
        == 0
    )
    if state == "blocked":
        assert fields["honest_verdict"] == "complete_blocked_authenticate_sha256"


def test_cli_and_frozen_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8175-COLD: success, block, tamper and replay use script paths."""
    specs = task.commands(tmp_path)
    assert any(
        s["name"] == "full_python_suite" and s["classification"] == "repository_health"
        for s in specs
    )
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
    assert task.execute(tmp_path / "unused.json", tmp_path) == 0


def test_cold_headlines_sources_and_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-8175-COLD: independently recompute and reject byte mutations."""
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    checked = task.inputs()
    value = task.scan(tmp_path, [], checked, tmp_path / "scan", current_date="20261005")
    inventory = tmp_path / "raw" / task.scope.OUTPUT.stem / "receipt_inventory.json"
    atomic_json(inventory, value)
    value.update(
        source_artifact_hashes={
            **value["scan_source_hashes"],
            str(inventory): sha256_file(inventory),
        },
        verdict_class="null",
        gate_check_summary=[],
        validation_receipts=[
            dict(name="cli_replay", classification="required", passed=True, normal_exit=True)
        ],
    )
    value.update(task.artifact_fields(value, checked))
    assert not task.replay(value)
    assert value["honest_verdict"] == "complete_null_no_new_outcomes"
    assert value["event_frontier"]["after_finished_at"] == checked["prior"]["finished_at"]
    changed = deepcopy(value)
    changed["supervisor_reader_ready_score"] = 0
    assert "supervisor_reader_ready_score" in task.replay(changed)
    changed = deepcopy(value)
    changed["source_artifact_hashes"][str(inventory)] = "sha256:wrong"
    assert any(e.startswith("source_sha256:") for e in task.replay(changed))
    log = tmp_path / "validation.log"
    log.write_text("original validation output")
    value["validation_receipts"][0].update(log_path=str(log), log_sha256=sha256_file(log))
    log.write_text("tampered validation output")
    assert "validation_log_sha256" in task.replay(value)
    log.write_text("original validation output")
    original = inventory.read_bytes()
    atomic_json(inventory, dict(rows=[], invented=True))
    assert any(e.startswith("source_sha256:") for e in task.replay(value))
    inventory.write_bytes(original)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    monkeypatch.setattr(task.runner.previous, "terminal", lambda *a: dict(passed=True, reports=[]))
    assert task.terminal(candidate, tmp_path / "private", tmp_path / "durable")["passed"]
    fixture = dict(task.runner.previous.reduce([]), rows=[])
    assert not task.replay(fixture)


def test_descriptive_new_event_has_no_priority_credit(tmp_path: Path) -> None:
    """REQ-REPORT-8175: an observed outcome remains descriptive below30/5."""
    source = producer(tmp_path, [live_episode()])
    value = task.scan(
        tmp_path,
        [source],
        dict(prior={"finished_at": "2000-01-01T00:00:00Z"}, inventory={}),
        tmp_path / "scan",
        current_date="20261005",
    )
    value.update(
        verdict_class="null",
        gate_check_summary=[],
        source_artifact_hashes={},
        validation_receipts=[
            dict(name="cli_replay", classification="required", passed=True, normal_exit=True)
        ],
    )
    fields = task.artifact_fields(value, {})
    assert fields["new_outcome_ready_score"] == 1
    assert fields["honest_verdict"] == "complete_null_descriptive_new_outcomes"
    assert fields["solve_provenance"] == "live_agent_self_discovery"
    assert fields["arm_recommendations"] == []
    assert fields["acceptance_gates"]["priority_change_minimum_resolved"] == 30
    assert fields["acceptance_gates"]["priority_change_minimum_games"] == 5


def test_changed_frontier_is_terminal_operand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8175-FRONTIER: changed custody cannot become a new frontier."""
    frontier = tmp_path / "frontier.json"
    frontier.write_text('{"rows": []}')
    monkeypatch.setattr(task.scope, "INVENTORY", frontier)
    monkeypatch.setitem(task.scope.PINNED, str(frontier), "sha256:" + "0" * 64)
    failed = task.inputs()["failures"]
    assert any(
        r["path"] == str(frontier) and r["observed"] == sha256_file(frontier) for r in failed
    )
    assert all(r["passed"] is False for r in failed)
