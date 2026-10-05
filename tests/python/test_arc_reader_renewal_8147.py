"""REQ-REPORT-8147 / REQ-VERIFY-8147: independent custody keeps failures visible."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot.reporting import arc_supervisor_v704_renewal as task
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_8147_v704_arc_reader_renewal as cli
from test_arc_supervisor_qualification_8001 import live_episode
from test_arc_supervisor_refinement_7936 import producer


def test_history_is_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8147-CUSTODY: old failed hashes remain failed history."""
    checked = task.inputs()
    assert not checked["failures"]
    assert len(checked["prior"]["historical_required_failures"]) >= 2
    failed = checked["prior"]["historical_required_failures"][-1]
    assert (
        failed["gate_check_summary"][0]["expected"] != failed["gate_check_summary"][0]["observed"]
    )
    assert checked["registry_precheck"]["ar25"] == 8
    missing = tmp_path / "absent.json"
    monkeypatch.setattr(task.scope, "PRIOR", missing)
    monkeypatch.setitem(task.scope.PINNED, str(missing), "sha256:" + "0" * 64)
    assert task.inputs()["failures"][0]["observed"] == "missing"


def test_controls_and_cold_assertions(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8147-COLD: original assertions bind independently sealed events."""
    controls = task.baseline.conformance(tmp_path / "controls")
    assert all(r["passed"] for r in controls)
    assert not task.control_errors(controls, tmp_path / "replay")
    forged = deepcopy(controls)
    forged[0]["source_events"][0]["trajectory_supervisor"]["redirects"][0]["actions_to_levelup"] = (
        99
    )
    assert task.control_errors(forged, tmp_path / "mutation")
    assert task.control_errors([], tmp_path / "missing")
    assert not task.replay(dict(task.runner.previous.reduce([]), rows=[]))


def test_frontier_deduplication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8147-FRONTIER: duplicates and old clocks supply no new credit."""
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    event = live_episode()
    event["trajectory_supervisor"]["stagnations_unredirected"] = 3
    event["trajectory_supervisor"]["redirects"] *= 2
    source = producer(tmp_path, [event])
    checked = dict(prior={"finished_at": "2000-01-01T00:00:00Z"}, inventory={})
    first = task.scan(tmp_path, [source], checked, tmp_path / "first", current_date="20261005")
    assert len(first["new_event_rows"]) == 1
    assert first["rows"][0]["source_cluster_id"] == "g1"
    assert (
        first["rows"][0]["source_events"][0]["trajectory_supervisor"]["stagnations_unredirected"]
        == 3
    )
    second = task.scan(
        tmp_path,
        [source],
        dict(prior=checked["prior"], inventory=first),
        tmp_path / "second",
        current_date="20261005",
    )
    assert second["new_event_rows"] == []
    checked["prior"]["finished_at"] = "2026-10-01T00:00:00Z"
    assert not task.scan(tmp_path, [source], checked, tmp_path / "old", current_date="20261005")[
        "new_event_rows"
    ]


@pytest.mark.parametrize("state", ["null", "blocked", "disqualified"])
def test_readiness_requires_normal_validation(tmp_path: Path, state: str) -> None:
    """REQ-REPORT-8147: readiness is separate from exposed natural benefit."""
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
    value["reader_conformance_rows"] = task.baseline.conformance(tmp_path / "controls")
    value["validation_receipts"] = [
        dict(name="cli_replay", classification="required", passed=True, normal_exit=True)
    ]
    fields = task.artifact_fields(value, {})
    assert fields["supervisor_reader_ready_score"] == int(state == "null")
    assert fields["new_outcome_ready_score"] == 0
    assert fields["new_solve_claim"] is False
    assert (
        fields["independent_generalization_score"]
        == fields["generalized_learning_benefit_score"]
        == 0
    )


def test_manifest_and_cli_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8147-COLD: private empty, missing, mutation and replay routes."""
    specs = task.commands(tmp_path)
    health = [s for s in specs if s["name"] == "full_python_suite"]
    assert all(s["classification"] == "repository_health" for s in health)
    assert health or task.HEALTH.is_file()
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


def test_cold_custody_and_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8147-COLD: primitive, assertion and current hash mutations fail."""
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    value = task.scan(
        tmp_path, [], dict(prior={}, inventory={}), tmp_path / "scan", current_date="20261005"
    )
    inventory = tmp_path / "raw" / task.scope.OUTPUT.stem / "receipt_inventory.json"
    atomic_json(inventory, value)
    original_inventory = inventory.read_bytes()
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
    value.update(task.artifact_fields(value, {}))
    assert not task.replay(value)
    assert value["honest_verdict"] == "complete_null_no_new_outcomes"
    changed = deepcopy(value)
    changed["supervisor_reader_ready_score"] = 0
    assert "supervisor_reader_ready_score" in task.replay(changed)
    changed = deepcopy(value)
    changed["source_artifact_hashes"][str(inventory)] = "sha256:wrong"
    assert any(e.startswith("source_sha256:") for e in task.replay(changed))
    atomic_json(inventory, dict(rows=[dict(status="malformed")]))
    assert "checkpoint_rows" in task.replay(value)
    inventory.write_bytes(original_inventory)
    transcript = next(Path(p) for p in value["reader_receipt"]["source_event_shards"])
    original = json.loads(transcript.read_text())
    atomic_json(transcript, dict(controls=[]))
    assert "control_evidence_rows" in task.replay(value)
    atomic_json(transcript, original)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    monkeypatch.setattr(task.runner.previous, "terminal", lambda *a: dict(passed=True, reports=[]))
    assert task.terminal(candidate, tmp_path / "private", tmp_path / "durable")["passed"]


def test_failed_controls_and_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8147-CUSTODY: failure remains an explicit operand."""
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    controls = task.baseline.conformance(tmp_path / "fixture")
    controls[0]["observed"] = 0
    monkeypatch.setattr(task.baseline, "conformance", lambda *a: controls)
    bad_frontier = dict(task.runner.previous.reduce([]), rows=[], identity_filter_count=1)
    value = task.scan(
        tmp_path,
        [],
        dict(prior={}, inventory=bad_frontier),
        tmp_path / "scan",
        current_date="20261005",
    )
    assert value["scan_failures"]
    monkeypatch.setitem(task.HISTORY, "absent.json", "0" * 64)
    assert any(r["observed"] == "missing" for r in task.inputs()["failures"])


def test_failed_inventory_stays_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8147-CUSTODY: old invalid manifests never gate fresh science."""
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    source = tmp_path / "results/experiment_failed.json"
    atomic_json(
        source,
        dict(
            run_date="20261004",
            verdict_class="blocked",
            source_artifact_hashes={
                str(tmp_path / "results/raw/missing-manifest.json"): "sha256:old_expected"
            },
        ),
    )
    monkeypatch.setitem(task.HISTORY, str(source.relative_to(tmp_path)), sha256_file(source)[7:])
    value = task.scan(
        tmp_path, [source], dict(prior={}, inventory={}), tmp_path / "scan", current_date="20261005"
    )
    assert not value["scan_failures"]
    assert value["new_event_rows"] == []
