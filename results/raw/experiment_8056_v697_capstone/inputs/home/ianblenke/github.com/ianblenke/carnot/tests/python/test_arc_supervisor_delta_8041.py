"""REQ-REPORT-8041: bind refinement to new authenticated live event evidence."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from typing import Any

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting import arc_supervisor_v696_delta as task
from scripts.experiments import experiment_8041_v696_arc_supervisor_delta as cli
from test_arc_supervisor_qualification_8001 import live_episode
from test_arc_supervisor_refinement_7936 import producer


def checked() -> dict[str, Any]:
    """A fixed authenticated boundary lets fixtures exercise new event clocks."""
    return dict(prior={"finished_at": "2026-09-30T10:00:00Z"}, inventory={})


def test_authority_and_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8041-FRONTIER: missing authority blocks by exact operand."""
    result = task.inputs()
    assert not result["failures"]
    assert task.scope.PRIOR_ID == 8028
    assert result["registry_precheck"]["ar25"] == 8
    specs = task.commands(tmp_path)
    assert any(s["name"] == "e2e_017" for s in specs)
    assert next(s for s in specs if s["name"] == "full_python_suite")["deadline_s"] == 120
    assert task.scope.ADDED == [task.MODULE, task.CLI]
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(task.scope, "PRIOR", missing)
    monkeypatch.setitem(task.scope.PINNED, str(missing), "sha256:" + "0" * 64)
    failure = task.inputs()["failures"][0]
    assert failure["path"] == str(missing)
    assert failure["artifact_field"] == "sha256" and failure["observed"] == "missing"
    assert failure["passed"] is False and failure["check"]


def test_new_events_and_retries(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8041-FRONTIER: clocks and hashes survive same-day retries."""
    source = producer(tmp_path, [live_episode()])
    first = task.scan(tmp_path, [source], checked(), tmp_path / "first", current_date="20261002")
    assert len(first["new_event_rows"]) == 1
    assert first["new_event_rows"][0]["frame_evidence_verified"] is True
    assert not first["scan_failures"]
    prior = checked()
    prior["inventory"] = first
    second = task.scan(tmp_path, [source], prior, tmp_path / "again", current_date="20261002")
    assert second["new_event_rows"] == []
    duplicate = deepcopy(live_episode())
    duplicate.update(receipt_id="retry", invocation_id="retry")
    duplicate["run_receipt"]["invocation_id"] = "retry"
    source = producer(tmp_path, [duplicate])
    retried = task.scan(tmp_path, [source], prior, tmp_path / "retry", current_date="20261002")
    assert not retried["new_event_rows"] and not retried["scan_failures"]


@pytest.mark.parametrize("kind", ["wrapper", "clock", "frame"])
def test_missing_authentication_blocks(tmp_path: Path, kind: str) -> None:
    """SCENARIO-REPORT-8041-FRONTIER: absence is a contract error, never a zero."""
    episode = live_episode()
    if kind == "wrapper":
        episode.pop("run_receipt")
    elif kind == "clock":
        episode.pop("event_timestamp")
    else:
        episode["run_receipt"]["frames_sha256"] = "sha256:wrong"
    source = producer(tmp_path, [episode])
    value = task.scan(tmp_path, [source], checked(), tmp_path / "private", current_date="20261002")
    assert not value["new_event_rows"]
    failure = value["scan_failures"][0]
    assert Path(failure["path"]).is_file()
    assert failure["sha256"] == sha256_file(Path(failure["path"]))
    assert failure["artifact_field"] and failure["passed"] is False
    assert failure["expected"] != failure["observed"]


def test_fields_and_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8041-SEAL: durable checkpoints and zero-work claims replay."""
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    inventory = tmp_path / "raw" / task.scope.OUTPUT.stem / "receipt_inventory.json"
    value = dict(task.runner.previous.reduce([]), rows=[], scan_source_hashes={})
    atomic_json(inventory, value)
    value["source_artifact_hashes"] = {str(inventory): sha256_file(inventory)}
    value.update(task.artifact_fields(value, {}))
    assert value["arc_delta_ready_score"] == 1
    assert value["event_frontier"]["upstream_id"] == "exp8028-arc-supervisor-delta"
    assert value["candidate_refinement"] is None and value["solve_provenance"] is None
    assert value["current_game_execution_count"] == value["current_model_invocation_count"] == 0
    assert value["substrate_declaration"]["MODEL_SPECS"] == []
    assert value["checkpoint_references"][0]["path"] == str(inventory)
    assert not task.replay(value)
    value["current_game_execution_count"] = 1
    assert "current_game_execution_count" in task.replay(value)
    value["current_game_execution_count"] = 0
    atomic_json(inventory, {"tampered": True})
    assert "checkpoint_sha256" in task.replay(value)
    value["verdict_class"] = "blocked"
    assert task.artifact_fields(value, {})["arc_delta_ready_score"] == 0


@pytest.mark.parametrize("count,games,supported", [(9, 3, False), (10, 2, False), (10, 3, True)])
def test_support_boundary(tmp_path: Path, count: int, games: int, supported: bool) -> None:
    """SCENARIO-REPORT-8041-SUPPORT: each arm needs completed multi-game support."""
    source = producer(tmp_path, [live_episode()])
    value = task.scan(tmp_path, [source], checked(), tmp_path / "private", current_date="20261002")
    template = value["new_event_rows"][0]
    rows = [
        dict(template, game=f"g{i % games}", receipt_id=str(i), invocation_id=str(i))
        for i in range(count)
    ]
    # A second censored arm cannot borrow the first arm's completed denominator.
    rows.append(dict(template, arm="reset_explorer", status="censored", game="g0"))
    value.update(task.runner.previous.reduce(rows), rows=rows)
    fields = task.artifact_fields(value, {})
    assert (fields["candidate_refinement"] is not None) is supported
    assert fields["sample_interpretation"] == (
        "directional_observational" if supported else "descriptive"
    )
    assert fields["per_game_arm_statistics"]["g0"]["reset_explorer"]["censored"] == 1
    assert fields["generalized_learning_benefit_score"] == 0
    if supported:
        assert fields["candidate_refinement"]["existing_arms"] == ["drop_goal_bias"]
        assert fields["candidate_refinement"]["causal_gain_claimed"] is False


def test_private_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8041-SEAL: direct empty/new/malformed and replay routes exit."""
    source = producer(tmp_path, [])
    output = tmp_path / "fixture.json"
    args = [
        task.CLI,
        "--date",
        "20261002",
        "--reduce-ledger",
        str(tmp_path),
        "--producer",
        str(source),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as executed:
        runpy.run_path(task.CLI, run_name="__main__")
    assert executed.value.code == 0
    assert cli.main(["--cold-replay", str(output)]) == 0
    assert cli.main(["--terminal-recheck", str(output)]) == 0
    # A new receipt without an authenticated frontier clock blocks in private CLI.
    source = producer(tmp_path, [live_episode()])
    assert cli.main(args[1:]) == 1
    assert json.loads(output.read_text())["gate_check_summary"]
    source.write_text("{broken")
    assert cli.main(args[1:]) == 1
    value = json.loads(output.read_text())
    value["identity_filter_count"] = 1
    atomic_json(output, value)
    assert cli.main(["--cold-replay", str(output)]) == 1
    with pytest.raises(SystemExit) as missing:
        cli.main(["--reduce-ledger", str(tmp_path)])
    assert missing.value.code == 2
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    atomic_json(task.scope.OUTPUT, {"private": True})
    monkeypatch.setattr(task, "terminal", lambda *_args: dict(passed=True, reports=[]))
    monkeypatch.setattr(task.runner, "execute", lambda _out, _private, scope: 0)
    assert cli.main([]) == 0
    sealed = json.loads(
        (tmp_path / "raw" / task.scope.OUTPUT.stem / "published_terminal_reports.json").read_text()
    )
    assert sealed["primary_sha256"] == sha256_file(task.scope.OUTPUT)
    assert sealed["report"]["passed"] is True


def test_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8041-SEAL: final scope cold-replays in a fresh private child."""
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, dict(task.runner.previous.reduce([]), rows=[]))
    monkeypatch.setattr(
        task.runner.previous, "terminal", lambda *_args: dict(passed=True, reports=[])
    )

    def run(spec: dict[str, Any], *_args: Any) -> dict[str, Any]:
        assert str(task.scope.ROOT / task.CLI) in spec["argv"]
        assert "PYTHONPATH" in spec["argv"]
        return dict(name=spec["name"], passed=True)

    monkeypatch.setattr(task.runner.previous, "run", run)
    assert task.terminal(candidate, tmp_path, tmp_path)["passed"]
