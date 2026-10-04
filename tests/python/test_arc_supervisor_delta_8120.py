"""REQ-REPORT-8120 and REQ-VERIFY-8120: preserve qualified delta custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess

import pytest

from carnot.reporting import arc_supervisor_v702_delta as task
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_8120_v702_arc_supervisor_delta as cli
from test_arc_supervisor_qualification_8001 import live_episode
from test_arc_supervisor_refinement_7936 import producer


def test_frontier_authentication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8120-FRONTIER: exact custody includes the real sidecar."""
    checked = task.inputs()
    drift = checked["failures"]
    assert len(drift) == 1
    assert drift[0]["path"] == str(task.scope.ROOT / task.baseline.MODULE)
    assert drift[0]["expected"] != drift[0]["observed"]
    original = task.sha256_file
    monkeypatch.setattr(
        task,
        "sha256_file",
        lambda p: drift[0]["expected"] if p == Path(drift[0]["path"]) else original(p),
    )
    checked = task.inputs()
    assert not checked["failures"]
    assert checked["prior"]["experiment_id"] == 8107
    assert checked["registry_precheck"]["ar25"] == 8
    assert all(r["check"] and r["passed"] for r in checked["checks"])
    missing = tmp_path / "absent-terminal.json"
    monkeypatch.setattr(task, "SIDECAR", missing)
    monkeypatch.setitem(task.scope.PINNED, str(missing), "sha256:" + "0" * 64)
    assert any(
        r["path"] == str(missing) and r["observed"] == "missing" for r in task.inputs()["failures"]
    )


def test_code_and_frontier_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8120-FRONTIER: qualification cannot survive changed bytes."""
    original = task.sha256_file
    monkeypatch.setattr(
        task,
        "sha256_file",
        lambda p: "sha256:changed" if p == task.scope.ROOT / task.baseline.MODULE else original(p),
    )
    assert any(r["observed"] == "sha256:changed" for r in task.inputs()["failures"])


@pytest.mark.parametrize("state", ["null", "blocked", "disqualified"])
def test_terminal_readiness(tmp_path: Path, state: str) -> None:
    """REQ-REPORT-8120: empty science is complete, external failures are blocked."""
    value = dict(
        task.runner.previous.reduce([]),
        rows=[],
        verdict_class=state,
        validation_receipts=[],
        gate_check_summary=[],
    )
    if state == "blocked":
        value["gate_check_summary"] = [
            task.runner.previous.operand(
                tmp_path / "absent.json", "sha256", "sha256:expected", "missing", None
            )
        ]
    value.update(task.artifact_fields(value, {}))
    assert value["arc_delta_ready_score"] == int(state == "null")
    assert value["new_outcome_count"] == value["new_solve_credit"] == 0
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert value["call_ledger"] == []
    assert not task.replay(value)
    if state == "blocked":
        assert value["honest_verdict"] == "complete_blocked_absent"
        assert value["gate_check_summary"][0]["observed"] == "missing"
    if state == "null":
        assert value["honest_verdict"] == "complete_null_no_new_outcomes"
    changed = deepcopy(value)
    changed["frontier_hashes"]["prior_primary"] = "sha256:forged"
    assert "frontier_hashes" in task.replay(changed)
    del changed["new_outcome_count"]
    assert "new_outcome_count" in task.replay(changed)


def test_changed_receipt_and_durable_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-8120-COLD: retries, missing outcomes and tampering fail closed."""
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    source = producer(tmp_path, [live_episode()])
    checked = dict(prior={"finished_at": "2026-09-30T10:00:00Z"}, inventory={})
    value = task.scan(tmp_path, [source], checked, tmp_path / "first", current_date="20261004")
    assert value["rows"][0]["source_cluster_id"] == value["rows"][0]["game"]
    shard = tmp_path / "raw" / task.scope.OUTPUT.stem / "receipt_inventory.json"
    atomic_json(shard, value)
    value["source_artifact_hashes"] = {str(shard): sha256_file(shard)}
    value.update(task.artifact_fields(value, {}))
    assert value["new_outcome_count"] == 1
    assert value["redirect_rows"][0]["fired"] == value["redirect_rows"][0]["helped"] == 1
    assert value["solve_provenance"] == "live_agent_self_discovery"
    assert not task.replay(value)
    retry = task.scan(
        tmp_path,
        [source],
        dict(prior=checked["prior"], inventory=value),
        tmp_path / "retry",
        current_date="20261004",
    )
    assert not retry["new_event_rows"]
    duplicate = deepcopy(value)
    duplicate["rows"].append(deepcopy(value["rows"][0]))
    assert "duplicate_event_identity" in task.replay(duplicate)
    removed = deepcopy(value)
    removed["rows"] = []
    removed.update(task.runner.previous.reduce([]))
    removed.update(task.artifact_fields(removed, {}))
    assert "checkpoint_rows" in task.replay(removed)
    atomic_json(shard, {})
    assert "checkpoint_sha256" in task.replay(value)
    value["checkpoint_references"][0]["sha256"] = sha256_file(shard)
    assert "checkpoint_rows" in task.replay(value)
    bad = live_episode()
    del bad["trajectory_supervisor"]["redirects"][0]["resolved_by_levelup"]
    source = producer(tmp_path, [bad])
    assert task.scan(tmp_path, [source], checked, tmp_path / "bad", current_date="20261004")[
        "scan_failures"
    ]


def test_private_cli_and_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8120-VALIDATION: real CLI routes run outside the checkout."""
    specs = task.commands(tmp_path)
    assert any(s["name"] == "e2e_017" for s in specs)
    assert "tests/python/test_primary_publication_7928.py" in task.scope.CONSUMERS
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for spec in specs:
        if spec["name"].startswith("cli_"):
            result = subprocess.run(
                spec["argv"], cwd=tmp_path, env=env, timeout=30, capture_output=True, text=True
            )
            assert result.returncode == spec["expected_exit"], result.stdout + result.stderr
    monkeypatch.setattr(task.runner, "execute", lambda *args: 0)
    assert task.execute(tmp_path / "unused.json", tmp_path) == 0
    monkeypatch.setattr(cli, "run", lambda argv, scope: 0)
    assert cli.main([]) == 0
    monkeypatch.setattr(task.scope, "execute", lambda *args: 0)
    monkeypatch.setattr("sys.argv", [str(task.scope.ROOT / task.CLI), "--date", "20261004"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(task.scope.ROOT / task.CLI), run_name="__main__")
    assert stopped.value.code == 0


def test_terminal_validators(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8120-VALIDATION: archive the actual cold replay receipt."""
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, dict(task.runner.previous.reduce([]), rows=[]))
    monkeypatch.setattr(
        task.runner.previous, "terminal", lambda *args: dict(passed=True, reports=[])
    )
    report = task.terminal(candidate, tmp_path, tmp_path / "durable")
    assert report["passed"] and report["reports"][0]["exit_code"] == 0
    assert json.loads(candidate.read_text())["rows"] == []


def test_health_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8120-VALIDATION: preserve global failure without repeating it."""
    health = tmp_path / "health.json"
    monkeypatch.setattr(task, "HEALTH", health)
    assert any(s["name"] == "full_python_suite" for s in task.commands(tmp_path))
    assert "owned_repository_health" not in task.inputs()
    atomic_json(
        health,
        dict(
            checks=[
                dict(
                    name="full_python_suite",
                    passed=False,
                    classification="repository_health",
                    exit_code=-15,
                )
            ]
        ),
    )
    assert not any(s["name"] == "full_python_suite" for s in task.commands(tmp_path))
    checked = task.inputs()
    assert checked["owned_repository_health"][0]["reused"]
    assert checked["additional_source_hashes"][str(health)] == sha256_file(health)
