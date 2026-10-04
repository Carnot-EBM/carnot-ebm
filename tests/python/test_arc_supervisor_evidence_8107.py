"""REQ-REPORT-8107 and REQ-VERIFY-8107: qualify custody without inventing evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from carnot.reporting import arc_supervisor_v701_evidence as task
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_8107_v701_arc_supervisor_evidence as cli
from test_arc_supervisor_qualification_8001 import live_episode
from test_arc_supervisor_refinement_7936 import producer


def test_actual_reader_authentication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8107-AUTHENTICATION: resolve the import that really runs."""
    checked = task.inputs()
    assert not checked["failures"]
    assert checked["resolved_reader_path"] == str(task.scope.ROOT / task.baseline.MODULE)
    assert checked["registry_precheck"]["ar25"] == 8
    assert all(row["check"] and row["passed"] for row in checked["checks"])
    missing = tmp_path / "absent-sidecar.json"
    monkeypatch.setattr(task, "SIDECAR", missing)
    monkeypatch.setitem(task.scope.PINNED, str(missing), "sha256:" + "0" * 64)
    assert any(
        r["path"] == str(missing) and r["observed"] == "missing" for r in task.inputs()["failures"]
    )


def test_code_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8107-AUTHENTICATION: historical custody includes code bytes."""
    original = task.sha256_file
    monkeypatch.setattr(
        task,
        "sha256_file",
        lambda p: "sha256:changed" if p == task.scope.ROOT / task.baseline.MODULE else original(p),
    )
    assert any(r["observed"] == "sha256:changed" for r in task.inputs()["failures"])


def test_rows_retries_and_missing_outcomes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8107-REDUCTION: only new authenticated redirects qualify."""
    source = producer(tmp_path, [live_episode()])
    checked = dict(prior={"finished_at": "2026-09-30T10:00:00Z"}, inventory={})
    first = task.scan(tmp_path, [source], checked, tmp_path / "first", current_date="20261004")
    row = first["new_event_rows"][0]
    assert row["metric"] == "observed_levelup_after_redirect"
    assert row["numerator"] == row["denominator"] == 1
    second = task.scan(
        tmp_path,
        [source],
        dict(prior=checked["prior"], inventory=first),
        tmp_path / "second",
        current_date="20261004",
    )
    assert not second["new_event_rows"]
    bad = live_episode()
    del bad["trajectory_supervisor"]["redirects"][0]["resolved_by_levelup"]
    source = producer(tmp_path, [bad])
    assert task.scan(tmp_path, [source], checked, tmp_path / "bad", current_date="20261004")[
        "scan_failures"
    ]


@pytest.mark.parametrize("state", ["null", "blocked", "disqualified"])
def test_readiness_and_cold_reduction(tmp_path: Path, state: str) -> None:
    """SCENARIO-VERIFY-8107-COLD: empty data and readiness are independent."""
    value = dict(
        task.runner.previous.reduce([]),
        rows=[],
        verdict_class=state,
        validation_receipts=[],
        gate_check_summary=[
            dict(
                path=str(tmp_path / "absent.json"),
                artifact_field="sha256",
                expected="sha256:expected",
                observed="missing",
                sha256=None,
                upstream_id="fixture",
            )
        ]
        if state == "blocked"
        else [],
    )
    value.update(task.artifact_fields(value, {}))
    assert value["supervisor_reader_ready_score"] == int(state == "null")
    assert value["new_outcome_count"] == 0
    assert value["claim_scope"] == value["exposure_scope"] == 0
    assert value["solve_provenance"] == "not_applicable_no_solve_claim"
    if state == "blocked":
        assert value["gate_check_summary"][0]["field"] == "sha256"
        assert value["honest_verdict"] == "complete_blocked_absent"
    assert not task.replay(value)
    changed = deepcopy(value)
    changed["new_outcome_count"] = 99
    assert "new_outcome_count" in task.replay(changed)
    shard = tmp_path / "receipt_inventory.json"
    atomic_json(shard, {})
    value["checkpoint_references"] = [dict(path=str(shard), sha256="sha256:wrong")]
    assert "checkpoint_sha256" in task.replay(value)


def test_live_rows_and_owned_failure(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8107-REDUCTION: observed help does not earn solve credit."""
    source = producer(tmp_path, [live_episode()])
    value = task.scan(
        tmp_path,
        [source],
        dict(prior={"finished_at": "2026-09-30T10:00:00Z"}, inventory={}),
        tmp_path / "scan",
        current_date="20261004",
    )
    value.update(task.artifact_fields(value, {}))
    assert value["new_outcome_count"] == 1 and value["redirect_rows"][0]["helped"] == 1
    assert value["solve_provenance"] == "live_agent_self_discovery"
    assert value["new_solve_credit"] == 0
    assert not task.replay(value)
    value["validation_receipts"] = [dict(classification="required", passed=False)]
    assert task.artifact_fields(value, {})["supervisor_reader_ready_score"] == 0
    duplicate = deepcopy(value)
    duplicate["rows"].append(deepcopy(value["rows"][0]))
    assert task.replay(duplicate)


def test_manifest_external_routes_and_entrypoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8107-VALIDATION: execute real CLI routes outside checkout."""
    monkeypatch.setattr(task, "HEALTH", tmp_path / "absent-health.json")
    specs = task.commands(tmp_path)
    assert any(s["name"] == "e2e_017" for s in specs)
    assert next(s for s in specs if s["name"] == "full_python_suite")["deadline_s"] == 120
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for spec in specs:
        if spec["name"].startswith("cli_"):
            result = subprocess.run(
                spec["argv"], cwd=tmp_path, env=env, timeout=30, capture_output=True, text=True
            )
            assert result.returncode == spec["expected_exit"], result.stdout + result.stderr
    output = tmp_path / "success" / task.scope.OUTPUT.name
    assert cli.main(["--cold-replay", str(output)]) == 0
    monkeypatch.setattr(sys, "argv", [task.CLI, "--cold-replay", str(output)])
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(task.CLI, run_name="__main__")
    assert exited.value.code == 0


def test_terminal_and_execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8107-COLD: final candidate is checked before publication."""
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, dict(task.runner.previous.reduce([]), rows=[]))
    monkeypatch.setattr(task.runner.previous, "terminal", lambda *_: dict(passed=True, reports=[]))

    def run(spec: dict, *_: object) -> dict:
        assert str(task.scope.ROOT / task.CLI) in spec["argv"]
        return dict(passed=True, command_argv=spec["argv"])

    monkeypatch.setattr(task.runner.previous, "run", run)
    assert task.terminal(candidate, tmp_path, tmp_path / "raw")["passed"]
    monkeypatch.setattr(task.runner, "execute", lambda *_: 0)
    assert task.execute(tmp_path / task.scope.OUTPUT.name, tmp_path) == 0


def test_health_receipt_is_separate_and_not_repeated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8107-VALIDATION: reuse the failed whole-suite diagnostic once."""
    receipt = tmp_path / "repository_health.json"
    atomic_json(
        receipt,
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
    monkeypatch.setattr(task, "HEALTH", receipt)
    checked = task.inputs()
    assert not checked["failures"]
    assert checked["owned_repository_health"][0]["reused"] is True
    assert checked["additional_source_hashes"][str(receipt)] == sha256_file(receipt)
    assert not any(s["name"] == "full_python_suite" for s in task.commands(tmp_path))


def test_recomputed_mutations_cannot_replace_durable_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-8107-COLD: fresh summaries cannot hide duplicate or deleted events."""
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    source = producer(tmp_path, [live_episode()])
    value = task.scan(
        tmp_path,
        [source],
        dict(prior={"finished_at": "2026-09-30T10:00:00Z"}, inventory={}),
        tmp_path / "scan",
        current_date="20261004",
    )
    shard = tmp_path / "raw" / task.scope.OUTPUT.stem / "receipt_inventory.json"
    atomic_json(shard, value)
    value["source_artifact_hashes"] = {str(shard): sha256_file(shard)}
    value.update(task.artifact_fields(value, {}))
    assert not task.replay(value)
    duplicate = deepcopy(value)
    duplicate["rows"].append(deepcopy(value["rows"][0]))
    duplicate.update(task.runner.previous.reduce(duplicate["rows"]))
    duplicate.update(task.artifact_fields(duplicate, {}))
    assert "duplicate_event_identity" in task.replay(duplicate)
    removed = deepcopy(value)
    removed["rows"] = []
    removed.update(task.runner.previous.reduce([]))
    removed.update(task.artifact_fields(removed, {}))
    assert "checkpoint_rows" in task.replay(removed)
    for field in ("redirect_rows", "new_outcome_count"):
        missing = deepcopy(value)
        del missing[field]
        assert field in task.replay(missing)
