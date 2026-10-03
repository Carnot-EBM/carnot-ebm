"""REQ-REPORT-8014: keep a qualified empty delta terminal and evidence durable."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from typing import Any

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_8014_v694_arc_supervisor_delta as task
from test_arc_supervisor_qualification_8001 import live_episode
from test_arc_supervisor_refinement_7936 import producer


def test_authority_and_validation_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8014-SEAL: pin Exp8001 and cover only new statements."""
    checked = task.inputs()
    assert not checked["failures"]
    assert task.scope.PRIOR_ID == 8001
    assert checked["inventory"]["outcome_hashes"] == checked["prior"]["outcome_hashes"]
    assert checked["registry_precheck"]["ar25"] == 8
    specs = task.commands(tmp_path)
    assert any(s["name"] == "e2e_017" for s in specs)
    assert not any(s["name"].startswith("e2e_016") for s in specs)
    assert task.scope.ADDED == [task.scope.CLI]
    assert all("arc_supervisor_v690_delta.py" not in s["argv"] for s in specs)
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(task.scope, "PRIOR", missing)
    monkeypatch.setitem(task.scope.PINNED, str(missing), "sha256:" + "0" * 64)
    assert task.inputs()["failures"]


def test_delta_frontier(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8014-DELTA: content frontiers exclude old outcomes and retries."""
    source = producer(tmp_path, [live_episode()])
    checked = dict(prior={}, inventory={})
    first = task.scope.scan(
        tmp_path, [source], checked, tmp_path / "first", current_date="20261002"
    )
    assert first["identity_filter_count"] == 1
    second = task.scope.scan(
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
    retried = task.scope.scan(
        tmp_path,
        [source],
        dict(prior={}, inventory=first),
        tmp_path / "retry",
        current_date="20261002",
    )
    assert retried["new_event_rows"] == []
    assert retried["rows"][0]["reason"] == "duplicate_outcome_bytes"


def test_fields_and_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8014-SEAL: cold reduction rejects inflated null summaries."""
    inventory = tmp_path / "receipt_inventory.json"
    value = dict(
        task.runner.previous.reduce([]), rows=[], scan_source_hashes={}, source_artifact_hashes={}
    )
    atomic_json(inventory, value)
    value["source_artifact_hashes"][str(inventory)] = sha256_file(inventory)
    value.update(task.artifact_fields(value, {}))
    assert value["honest_verdict"] == "complete_null_no_new_outcomes"
    assert value["frontier"]["upstream_id"] == "exp8001-arc-supervisor-delta"
    assert value["genuine_headroom"]["natural_evidence_count"] == 0
    assert value["checkpoint_references"][0]["path"] == str(inventory)
    assert not task.replay(value)
    value["no_new_outcomes"] = False
    assert "no_new_outcomes" in task.replay(value)
    value["verdict_class"] = "blocked"
    assert task.artifact_fields(value, {})["honest_verdict"].startswith("complete_blocked")


def test_observational_proposal() -> None:
    """SCENARIO-REPORT-8014-DELTA: sufficient associations suggest a future control only."""
    rows = [
        dict(
            game=str(i % 3),
            arm="drop_goal_bias",
            receipt_id=str(i),
            invocation_id=str(i),
            seed=0,
            resolved_by_levelup=True,
            actions_to_levelup=3,
            stagnations_unredirected=0,
            chronology="unknown",
        )
        for i in range(8)
    ]
    value = dict(
        new_event_rows=rows, source_artifact_hashes={}, scan_source_hashes={}, receipt_inventory={}
    )
    fields = task.artifact_fields(value, {})
    assert (
        fields["proposed_generalization_refinement"]["action"]
        == "future_controlled_generalization_test"
    )
    assert fields["defaults_changed"] is False
    assert fields["causal_benefit_claimed"] is False
    assert len(fields["chronology_unknown_rows"]) == 8
    assert fields["per_game_arm_statistics"]["0"]["drop_goal_bias"]["resolved_by_levelup"] == 3
    value["new_event_rows"] = rows[:7]
    assert task.artifact_fields(value, {})["proposed_generalization_refinement"] is None


def test_private_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8014-SEAL: run the script entrypoint and private terminal branches."""
    source = producer(tmp_path, [])
    output = tmp_path / "fixture.json"
    args = [
        task.scope.CLI,
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
        runpy.run_path(task.scope.CLI, run_name="__main__")
    assert executed.value.code == 0
    assert task.main(["--cold-replay", str(output)]) == 0
    assert task.main(["--terminal-recheck", str(output)]) == 0
    with pytest.raises(SystemExit) as missing:
        task.main(["--reduce-ledger", str(tmp_path)])
    assert missing.value.code == 2
    with pytest.raises(SystemExit) as date:
        task.main(["--date", "20261001"])
    assert date.value.code == 2
    monkeypatch.setattr(
        task.runner, "execute", lambda _out, _private, scope: int(scope.EXPERIMENT_ID != 8014)
    )
    assert task.main([]) == 0


def test_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8014-SEAL: cold process uses this invocation and unchanged bytes."""
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
    assert json.loads(candidate.read_text())["new_event_rows"] == []
