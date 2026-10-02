"""REQ-REPORT-8028: empty deltas carry checked custody, never new solve credit."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from typing import Any

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_8028_v695_arc_supervisor_delta as task
from test_arc_supervisor_qualification_8001 import live_episode
from test_arc_supervisor_refinement_7936 import producer


def test_authority_and_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8028-SEAL: pin frontier identity and the owned command scope."""
    checked = task.inputs()
    assert not checked["failures"]
    assert task.scope.PRIOR_ID == 8014
    assert checked["inventory"]["outcome_hashes"] == checked["prior"]["outcome_hashes"]
    assert checked["registry_precheck"]["ar25"] == 8
    assert all(r["passed"] for r in checked["checks"])
    specs = task.commands(tmp_path)
    assert any(s["name"] == "e2e_017" for s in specs)
    assert not any(s["name"].startswith("e2e_016") for s in specs)
    assert task.scope.ADDED == [task.scope.CLI]
    assert [s for s in specs if s["name"] == "full_python_suite"][0]["classification"] == (
        "repository_health"
    )
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(task.scope, "PRIOR", missing)
    monkeypatch.setitem(task.scope.PINNED, str(missing), "sha256:" + "0" * 64)
    assert task.inputs()["failures"]
    assert all(not r["passed"] for r in task.inputs()["failures"])


def test_frontier_and_unknown_chronology(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8028-DELTA: unseen bytes need authenticated chronology."""
    source = producer(tmp_path, [live_episode()])
    first = task.scan(
        tmp_path,
        [source],
        dict(prior={}, inventory={}),
        tmp_path / "first",
        current_date="20261002",
    )
    assert first["new_event_rows"] == []
    assert first["rows"][0]["reason"] == "unknown_chronology"
    assert first["rows"][0]["solve_provenance"] == "live_agent_self_discovery"
    second = task.scan(
        tmp_path,
        [source],
        dict(prior={}, inventory=first),
        tmp_path / "second",
        current_date="20261002",
    )
    assert second["new_event_rows"] == []
    duplicate = deepcopy(live_episode())
    duplicate.update(receipt_id="retry", invocation_id="retry")
    duplicate["run_receipt"]["invocation_id"] = "retry"
    source = producer(tmp_path, [duplicate])
    retried = task.scan(
        tmp_path,
        [source],
        dict(prior={}, inventory=first),
        tmp_path / "retry",
        current_date="20261002",
    )
    assert retried["new_event_rows"] == []
    assert retried["rows"][0]["reason"] == "duplicate_outcome_bytes"


def test_fields_and_durable_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8028-SEAL: recount null fields and durable primitive bytes."""
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    inventory = tmp_path / "raw" / task.scope.OUTPUT.stem / "receipt_inventory.json"
    value = dict(task.runner.previous.reduce([]), rows=[], scan_source_hashes={})
    atomic_json(inventory, value)
    value["source_artifact_hashes"] = {str(inventory): sha256_file(inventory)}
    value.update(task.artifact_fields(value, {}))
    assert value["honest_verdict"] == "complete_null_no_new_outcomes"
    assert value["frontier"]["upstream_id"] == "exp8014-arc-supervisor-delta"
    assert value["arc_delta_ready_score"] == 1
    assert value["solve_provenance"] is None
    assert value["current_game_runs"] == 0
    assert value["proposed_generalization_refinement"] is None
    assert value["checkpoint_references"][0]["path"] == str(inventory)
    assert not task.replay(value)
    value["arc_delta_ready_score"] = 0
    assert "arc_delta_ready_score" in task.replay(value)
    value["arc_delta_ready_score"] = 1
    atomic_json(inventory, {"tampered": True})
    assert "checkpoint_sha256" in task.replay(value)
    value["verdict_class"] = "blocked"
    assert task.artifact_fields(value, {})["arc_delta_ready_score"] == 0


def test_private_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8028-SEAL: direct script execution includes negative replay."""
    source = producer(tmp_path, [])
    output = tmp_path / "fixture.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            task.scope.CLI,
            "--date",
            "20261002",
            "--reduce-ledger",
            str(tmp_path),
            "--producer",
            str(source),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as executed:
        runpy.run_path(task.scope.CLI, run_name="__main__")
    assert executed.value.code == 0
    assert task.main(["--cold-replay", str(output)]) == 0
    assert task.main(["--terminal-recheck", str(output)]) == 0
    value = json.loads(output.read_text())
    value["identity_filter_count"] = 1
    atomic_json(output, value)
    assert task.main(["--cold-replay", str(output)]) == 1
    with pytest.raises(SystemExit) as missing:
        task.main(["--reduce-ledger", str(tmp_path)])
    assert missing.value.code == 2
    monkeypatch.setattr(task.runner, "execute", lambda _out, _private, scope: 0)
    assert task.main([]) == 0


def test_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8028-SEAL: use a fresh process on the exact candidate."""
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, dict(task.runner.previous.reduce([]), rows=[]))
    monkeypatch.setattr(
        task.runner.previous, "terminal", lambda *_args: dict(passed=True, reports=[])
    )

    def run(spec: dict[str, Any], *_args: Any) -> dict[str, Any]:
        assert str(task.scope.ROOT / task.scope.CLI) in spec["argv"]
        assert "PYTHONPATH" in spec["argv"]
        return dict(name=spec["name"], passed=True)

    monkeypatch.setattr(task.runner.previous, "run", run)
    assert task.terminal(candidate, tmp_path, tmp_path)["passed"]
