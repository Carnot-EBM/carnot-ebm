"""REQ-REPORT-8080: preserve the event boundary while validating this invocation."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import os
import subprocess
import sys
from typing import Any

import pytest

from carnot.reporting import arc_supervisor_v699_frontier as task
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_8080_v699_arc_supervisor_frontier as cli
from test_arc_supervisor_qualification_8001 import live_episode
from test_arc_supervisor_refinement_7936 import producer


def test_authenticated_authority_and_missing_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8080-FRONTIER: authority binds bytes, code and terminal checks."""
    checked = task.inputs()
    assert len(checked["failures"]) == 1
    assert checked["failures"][0]["path"].endswith(task.NAMED_PRIOR_CODE)
    assert checked["failures"][0]["observed"] == "missing"
    assert task.scope.PRIOR_ID == 8067 and task.scope.RUN_DATE == "20261003"
    assert all(row["check_name"] for row in checked["checks"])
    assert all(
        row["passed"]
        for row in checked["checks"]
        if not row["path"].endswith(task.NAMED_PRIOR_CODE)
    )
    assert checked["registry_precheck"]["ar25"] == 8
    specs = task.commands(tmp_path)
    assert any(row["name"] == "e2e_017" for row in specs)
    assert next(row for row in specs if row["name"] == "full_python_suite")["deadline_s"] == 120
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(task, "SIDECAR", missing)
    monkeypatch.setitem(task.scope.PINNED, str(missing), "sha256:" + "0" * 64)
    failure = task.inputs()["failures"][0]
    assert failure["path"] == str(missing) and failure["observed"] == "missing"
    assert failure["passed"] is False


def test_changed_scanner_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8080-FRONTIER: scanner code drift cannot remain qualified."""
    original = task.sha256_file
    monkeypatch.setattr(
        task,
        "sha256_file",
        lambda p: "sha256:changed" if p == task.scope.ROOT / task.AUTH_CODE[0] else original(p),
    )
    failure = task.inputs()["failures"][0]
    assert failure["artifact_field"] == "sha256" and failure["check_name"]


def test_frontier_retry_and_missing_auth(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8080-FRONTIER: retries and broken frames create no live credit."""
    checked = dict(prior={"finished_at": "2026-09-30T10:00:00Z"}, inventory={})
    source = producer(tmp_path, [live_episode()])
    first = task.scope.scan(
        tmp_path, [source], checked, tmp_path / "first", current_date="20261003"
    )
    assert len(first["new_event_rows"]) == 1
    checked["inventory"] = first
    second = task.scope.scan(
        tmp_path, [source], checked, tmp_path / "retry", current_date="20261003"
    )
    assert not second["new_event_rows"]
    episode = live_episode()
    episode["run_receipt"]["frames_sha256"] = "sha256:wrong"
    source = producer(tmp_path, [episode])
    bad = task.scope.scan(
        tmp_path, [source], dict(prior={}, inventory={}), tmp_path / "bad", current_date="20261003"
    )
    assert bad["scan_failures"] and not bad["new_event_rows"]


@pytest.mark.parametrize(
    "count,games,supported", [(0, 0, False), (9, 3, False), (10, 2, False), (10, 3, True)]
)
def test_reduction_and_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, games: int, supported: bool
) -> None:
    """SCENARIO-REPORT-8080-SELECTION: zero evidence and underpowered arms stay unchanged."""
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    source = producer(tmp_path, [live_episode()])
    value = task.scope.scan(
        tmp_path,
        [source],
        dict(prior={"finished_at": "2026-09-30T10:00:00Z"}, inventory={}),
        tmp_path / "private",
        current_date="20261003",
    )
    template = value["new_event_rows"][0]
    rows = [
        dict(template, game=f"g{i % games}", receipt_id=str(i), invocation_id=str(i))
        for i in range(count)
    ]
    value.update(task.runner.previous.reduce(rows), rows=rows, scan_source_hashes={})
    inventory = tmp_path / "raw" / task.scope.OUTPUT.stem / "receipt_inventory.json"
    atomic_json(inventory, value)
    value["source_artifact_hashes"] = {str(inventory): sha256_file(inventory)}
    value.update(task.artifact_fields(value, {}))
    assert value["event_frontier"]["upstream_id"] == "exp8067-arc-supervisor-frontier"
    assert (value["candidate_refinement"] is not None) is supported
    assert value["current_game_execution_count"] == value["current_model_invocation_count"] == 0
    assert value["generalized_learning_benefit_score"] == 0
    assert not task.replay(value)
    mutated = deepcopy(value)
    mutated["current_game_execution_count"] = 1
    assert "current_game_execution_count" in task.replay(mutated)
    atomic_json(inventory, {"tampered": True})
    assert "checkpoint_sha256" in task.replay(value)
    value.update(
        verdict_class="blocked",
        gate_check_summary=[
            dict(
                path="/tmp/absent.json",
                artifact_field="sha256",
                expected="hash",
                observed="missing",
                sha256=None,
                upstream_id="fixture",
            )
        ],
    )
    fields = task.artifact_fields(value, {})
    assert fields["arc_delta_ready_score"] == 0
    assert fields["honest_verdict"] == "complete_blocked_absent"


def test_cli_and_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8080-VALIDATION: real script exits and final-byte checks are reachable."""
    source = producer(tmp_path, [])
    output = tmp_path / "fixture.json"
    args = [
        task.CLI,
        "--date",
        "20261003",
        "--reduce-ledger",
        str(tmp_path),
        "--producer",
        str(source),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(task.CLI, run_name="__main__")
    assert exited.value.code == 0
    assert cli.main(["--cold-replay", str(output)]) == 0
    source.write_text("{broken")
    assert cli.main(args[1:]) == 1
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    atomic_json(task.scope.OUTPUT, dict(task.runner.previous.reduce([]), rows=[]))
    monkeypatch.setattr(task.runner, "execute", lambda *_args: 0)
    monkeypatch.setattr(
        task.runner.previous, "terminal", lambda *_args: dict(passed=True, reports=[])
    )

    def run(spec: dict[str, Any], *_args: Any) -> dict[str, Any]:
        assert str(task.scope.ROOT / task.CLI) in spec["argv"]
        return dict(passed=True, name=spec["name"])

    monkeypatch.setattr(task.runner.previous, "run", run)
    assert cli.main([]) == 0
    sealed = json.loads(
        (tmp_path / "raw" / task.scope.OUTPUT.stem / "published_terminal_reports.json").read_text()
    )
    assert sealed["primary_sha256"] == sha256_file(task.scope.OUTPUT)
    assert sealed["report"]["passed"] is True


def test_frontier_fields_and_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8080-VALIDATION: forged or missing observations fail replay."""
    source = producer(tmp_path, [live_episode()])
    value = task.scope.scan(
        tmp_path,
        [source],
        dict(prior={"finished_at": "2026-09-30T10:00:00Z"}, inventory={}),
        tmp_path / "private",
        current_date="20261003",
    )
    value.update(task.artifact_fields(value, {}))
    assert value["new_outcome_count"] == 1
    assert value["per_game_results"]
    assert not task.replay(value)
    for key, change in [
        ("new_outcome_count", 2),
        ("current_frontier", {}),
        ("new_event_rows", []),
        ("identity_filter_count", 2),
    ]:
        mutated = deepcopy(value)
        mutated[key] = change
        assert task.replay(mutated), key
    duplicate = deepcopy(value)
    duplicate["rows"].append(deepcopy(duplicate["rows"][0]))
    assert task.replay(duplicate)
    missing = live_episode()
    del missing["trajectory_supervisor"]["redirects"][0]["resolved_by_levelup"]
    source = producer(tmp_path, [missing])
    rejected = task.scope.scan(
        tmp_path,
        [source],
        dict(prior={}, inventory={}),
        tmp_path / "missing",
        current_date="20261003",
    )
    assert rejected["scan_failures"] and not rejected["new_event_rows"]


def test_empty_fields_and_required_checks() -> None:
    """SCENARIO-REPORT-8080-FRONTIER: custody permits a cheap terminal null."""
    value = dict(task.runner.previous.reduce([]), rows=[], validation_receipts=[])
    fields = task.artifact_fields(value, {})
    assert fields["new_outcome_count"] == 0 and fields["refinement_proposal"] is None
    assert fields["pending_events"] == []
    assert fields["solve_provenance"] is None
    assert fields["prior_frontier"]["upstream_id"] == "exp8067-arc-supervisor-frontier"
    assert fields["required_checks_passed"] is True
    value.update(
        verdict_class="disqualified",
        validation_receipts=[
            dict(
                classification="required",
                passed=False,
            )
        ],
    )
    assert task.artifact_fields(value, {})["required_checks_passed"] is False


def test_pending_observations_are_preserved(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8080-FRONTIER: open and unknown events remain primitive evidence."""
    source = producer(tmp_path, [live_episode()])
    value = task.scope.scan(
        tmp_path,
        [source],
        dict(prior={"finished_at": "2026-09-30T10:00:00Z"}, inventory={}),
        tmp_path / "private",
        current_date="20261003",
    )
    row = value["rows"][0]
    open_row = dict(row, status="censored", receipt_id="open")
    unknown_row = dict(row, status="excluded", chronology="unknown", receipt_id="unknown")
    value["rows"].extend([open_row, unknown_row])
    assert task.artifact_fields(value, {})["pending_events"] == [open_row, unknown_row]


def test_readiness_is_a_prerequisite_gate() -> None:
    """SCENARIO-REPORT-8080-VALIDATION: a blocked binary gate is no perfect metric."""
    from scripts import adversarial_verify as verifier

    payload = dict(
        arc_delta_ready_score=0,
        arc_evidence_ready_score=0,
        verdict_class="blocked",
        gate_check_summary=[dict(observed="missing")],
    )
    flags: list[Any] = []
    verifier.check_implausible_perfect(payload, flags)
    assert not flags
    for change in (
        dict(verdict_class="null"),
        dict(arc_evidence_ready_score=1),
        dict(gate_check_summary=[]),
        dict(brier_loss=0),
    ):
        flags = []
        verifier.check_implausible_perfect(dict(payload, **change), flags)
        assert flags


def test_terminal_failure_operand_shape() -> None:
    """SCENARIO-REPORT-8080-VALIDATION: failure reports remain replayable evidence."""
    value = dict(
        task.runner.previous.reduce([]),
        rows=[],
        verdict_class="disqualified",
        gate_check_summary=[dict(check="terminal_validation", report=dict(passed=False))],
    )
    value.update(task.artifact_fields(value, {}))
    assert value["gate_check_summary"][0]["artifact_field"] == "terminal_validation"
    assert not task.replay(value)


def test_external_cli_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8080-VALIDATION: real external replay rejects three mutations."""
    source = producer(tmp_path, [live_episode()])
    value = task.scope.scan(
        tmp_path,
        [source],
        dict(prior={"finished_at": "2026-09-30T10:00:00Z"}, inventory={}),
        tmp_path / "private",
        current_date="20261003",
    )
    value.update(task.artifact_fields(value, {}))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for mutation in ("valid", "forged_firing", "duplicate_frontier", "missing_outcome"):
        candidate = deepcopy(value)
        if mutation == "forged_firing":
            candidate["identity_filter_count"] += 1
        elif mutation == "duplicate_frontier":
            candidate["rows"].append(deepcopy(candidate["rows"][0]))
        elif mutation == "missing_outcome":
            candidate["new_event_rows"] = []
        path = tmp_path / (mutation + ".json")
        atomic_json(path, candidate)
        result = subprocess.run(
            [
                str(task.scope.ROOT / ".venv/bin/python"),
                "-u",
                str(task.scope.ROOT / task.CLI),
                "--cold-replay",
                str(path),
            ],
            cwd=tmp_path,
            env=env,
            timeout=30,
            capture_output=True,
            text=True,
        )
        assert result.returncode == int(mutation != "valid"), result.stdout + result.stderr
        assert "cold_replay_errors" in result.stdout
