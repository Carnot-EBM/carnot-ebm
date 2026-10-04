"""REQ-REPORT-8133 and REQ-VERIFY-8133: current behavior has new custody."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot.reporting import arc_supervisor_v703_frontier as task
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_8133_v703_arc_reader_frontier as cli
from test_arc_supervisor_qualification_8001 import live_episode
from test_arc_supervisor_refinement_7936 import producer


def test_current_custody(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8133-CONFORMANCE: drift history never becomes current custody."""
    checked = task.inputs()
    assert not checked["failures"]
    failure = checked["prior"]["historical_required_failures"][-1]["gate_check_summary"][0]
    assert failure["expected"] != failure["observed"]
    assert failure["path"].endswith("arc_supervisor_v701_evidence.py")
    assert checked["registry_precheck"]["ar25"] == 8
    missing = tmp_path / "absent.json"
    monkeypatch.setattr(task, "SIDECAR", missing)
    monkeypatch.setitem(task.scope.PINNED, str(missing), "sha256:" + "0" * 64)
    assert any(r["observed"] == "missing" for r in task.inputs()["failures"])


def test_controls_and_mutated_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8133-CONFORMANCE: exact private outcomes exercise current code."""
    rows = task.conformance(tmp_path)
    assert rows and all(r["passed"] for r in rows)
    assert {r["condition"] for r in rows} >= {"supported", "missing", "mutation", "zero"}
    assert all(r["claim_scope"] == "circular_positive" for r in rows)
    monkeypatch.setattr(
        task.baseline, "scan", lambda *a, **k: dict(task.runner.previous.reduce([]), rows=[])
    )
    assert not all(r["passed"] for r in task.conformance(tmp_path / "mutated"))


def test_frontier_rows_and_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8133-FRONTIER: games group independent evidence; repeats do not."""
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    source = producer(tmp_path, [live_episode()])
    checked = dict(prior={"finished_at": "2000-01-01T00:00:00Z"}, inventory={})
    first = task.scan(tmp_path, [source], checked, tmp_path / "first", current_date="20261004")
    assert len(first["new_event_rows"]) == 1
    assert first["rows"][0]["source_cluster_id"] == "g1"
    assert first["inventory_counts"]["new"] == 1
    second = task.scan(
        tmp_path,
        [source],
        dict(prior=checked["prior"], inventory=first),
        tmp_path / "retry",
        current_date="20261004",
    )
    assert second["new_event_rows"] == []
    bad = live_episode()
    del bad["trajectory_supervisor"]["redirects"][0]["actions_to_levelup"]
    source = producer(tmp_path, [bad])
    assert task.scan(tmp_path, [source], checked, tmp_path / "bad", current_date="20261004")[
        "scan_failures"
    ]
    monkeypatch.setattr(task, "conformance", lambda p: [dict(passed=False)])
    assert task.scan(tmp_path, [], checked, tmp_path / "failed", current_date="20261004")[
        "scan_failures"
    ]


@pytest.mark.parametrize("state", ["null", "blocked", "disqualified"])
def test_separate_readiness(tmp_path: Path, state: str) -> None:
    """REQ-REPORT-8133: reader qualification supplies no behavioral benefit."""
    value = dict(
        task.runner.previous.reduce([]),
        rows=[],
        verdict_class=state,
        validation_receipts=[],
        gate_check_summary=[],
        reader_conformance_rows=[dict(passed=True)],
    )
    if state == "blocked":
        value["gate_check_summary"] = [
            task.runner.previous.operand(
                tmp_path / "absent.json", "sha256", "expected", "missing", None
            )
        ]
    value.update(task.artifact_fields(value, {}))
    assert value["supervisor_reader_ready_score"] == int(state == "null")
    assert value["new_outcome_ready_score"] == 0
    assert value["new_solve_claim"] is False and value["arm_recommendations"] == []
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert not task.replay(value)
    if state == "null":
        assert value["honest_verdict"] == "complete_null_no_new_outcomes"
    if state == "blocked":
        assert value["honest_verdict"] == "complete_blocked_absent"
    changed = deepcopy(value)
    changed["new_outcome_ready_score"] = 1
    assert "new_outcome_ready_score" in task.replay(changed)
    del changed["new_solve_claim"]
    assert "new_solve_claim" in task.replay(changed)


def test_durable_cold_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8133-COLD: deleted rows and altered current code fail replay."""
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    source = producer(tmp_path, [live_episode()])
    value = task.scan(
        tmp_path,
        [source],
        dict(prior={"finished_at": "2000-01-01T00:00:00Z"}, inventory={}),
        tmp_path / "scan",
        current_date="20261004",
    )
    shard = tmp_path / "raw" / task.scope.OUTPUT.stem / "receipt_inventory.json"
    atomic_json(shard, value)
    value.update(
        source_artifact_hashes={str(shard): sha256_file(shard)},
        verdict_class="null",
        validation_receipts=[],
    )
    value.update(task.artifact_fields(value, {}))
    assert value["new_outcome_ready_score"] == 1
    assert not task.replay(value)
    forged = deepcopy(value)
    forged["reader_receipt"]["current_code_hashes"][task.baseline.MODULE] = "sha256:wrong"
    assert "reader_receipt" in task.replay(forged)
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
    value["validation_receipts"] = [dict(classification="required", passed=False)]
    assert task.artifact_fields(value, {})["supervisor_reader_ready_score"] == 0


def test_cli_manifest_and_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8133-FRONTIER: bounded real CLI controls have frozen argv."""
    specs = task.commands(tmp_path)
    assert any(s["name"] == "e2e_017" for s in specs)
    assert "tests/python/test_primary_publication_7928.py" in task.scope.CONSUMERS
    assert (
        next(s for s in specs if s["name"] == "full_python_suite")["classification"]
        == "repository_health"
    )
    for spec in specs:
        if spec["name"].startswith("cli_"):
            row = task.runner.previous.run(spec, tmp_path, tmp_path / "logs")
            assert row["passed"], row["output_tail"]
    monkeypatch.setattr(task.runner, "execute", lambda *a: 0)
    assert task.execute(tmp_path / "unused.json", tmp_path) == 0
    monkeypatch.setattr(cli, "run", lambda *a, **k: 0)
    assert cli.main([]) == 0
    monkeypatch.setattr(task.scope, "execute", lambda *a: 0)
    monkeypatch.setattr(sys, "argv", [str(task.scope.ROOT / task.CLI)])
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(task.scope.ROOT / task.CLI), run_name="__main__")
    assert exited.value.code == 0


def test_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8133-COLD: normal cold replay and validators precede publish."""
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, dict(task.runner.previous.reduce([]), rows=[]))
    monkeypatch.setattr(task.runner.previous, "terminal", lambda *a: dict(passed=True, reports=[]))
    assert task.terminal(candidate, tmp_path, tmp_path / "durable")["passed"]


def test_duplicate_and_old_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8133-FRONTIER: identities and clocks bound the new panel."""
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    event = live_episode()
    event["trajectory_supervisor"]["redirects"] *= 2
    source = producer(tmp_path, [event])
    checked = dict(prior={"finished_at": "2000-01-01T00:00:00Z"}, inventory={})
    value = task.scan(tmp_path, [source], checked, tmp_path / "duplicate", current_date="20261004")
    assert len(value["new_event_rows"]) == 1
    assert any(r.get("reason") == "duplicate_redirect_identity" for r in value["rows"])
    checked["prior"]["finished_at"] = "2026-10-01T00:00:00Z"
    value = task.scan(tmp_path, [source], checked, tmp_path / "old", current_date="20261004")
    assert value["new_event_rows"] == []
    checked["inventory"] = dict(task.runner.previous.reduce([]), rows=[], identity_filter_count=9)
    assert task.scan(tmp_path, [], checked, tmp_path / "bad-frontier", current_date="20261004")[
        "scan_failures"
    ]


def test_missing_control_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8133-COLD: recomputed claims cannot hide missing raw controls."""
    monkeypatch.setattr(task.scope, "OUTPUT", tmp_path / task.scope.OUTPUT.name)
    value = task.scan(
        tmp_path, [], dict(prior={}, inventory={}), tmp_path / "scan", current_date="20261004"
    )
    value.update(
        source_artifact_hashes=value["scan_source_hashes"],
        verdict_class="null",
        validation_receipts=[],
    )
    value.update(task.artifact_fields(value, {}))
    assert not task.replay(value)
    transcript = next(Path(p) for p in value["reader_receipt"]["source_event_shards"])
    original = json.loads(transcript.read_text())
    atomic_json(transcript, {})
    assert "control_evidence_sha256" in task.replay(value)
    value["source_artifact_hashes"][str(transcript)] = sha256_file(transcript)
    value.update(task.artifact_fields(value, {}))
    assert "control_evidence_rows" in task.replay(value)
    atomic_json(transcript, original)
    value["source_artifact_hashes"][str(transcript)] = sha256_file(transcript)
    value.update(task.artifact_fields(value, {}))
    transcript.unlink()
    assert "control_evidence_sha256" in task.replay(value)
