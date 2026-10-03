"""REQ-REPORT-7988 and REQ-VERIFY-7988 preserve historical qualification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import arc_supervisor_v692_delta as task
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7988_v692_arc_supervisor_delta as cli


def authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Private predecessors let failure tests preserve the original artifacts."""
    (tmp_path / "results").mkdir()
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops/exclusion_manifest.yaml").write_text("retired: []\n")
    for name in ("PRIOR", "INVENTORY", "HISTORY", "HISTORY_INVENTORY", "REGISTRY"):
        path = tmp_path / (name + (".yaml" if name == "REGISTRY" else ".json"))
        monkeypatch.setattr(task, name, path)
    atomic_json(task.INVENTORY, dict(receipt_inventory={}, seen_receipt_hashes={}))
    atomic_json(task.HISTORY_INVENTORY, dict(receipt_inventory={}, seen_receipt_hashes={}))
    base = dict(run_date="20261001", receipt_inventory={}, seen_receipt_hashes={})
    atomic_json(
        task.PRIOR,
        dict(
            base,
            experiment_id=7962,
            task_id="exp7962-arc-supervisor-delta",
            verdict_class="null",
            honest_verdict="complete_null_no_new_supervisor_outcomes",
            arc_evidence_ready_score=1,
            flagged_adversarial=False,
        ),
    )
    atomic_json(
        task.HISTORY,
        dict(
            base,
            experiment_id=7975,
            task_id="exp7975-arc-supervisor-delta",
            execution_date="20261001",
            verdict_class="disqualified",
            arc_evidence_ready_score=0,
            gate_check_summary=[dict(check="old_owned_failure")],
        ),
    )
    task.REGISTRY.write_text("games:\n  g1:\n    levels_reproduced: 2\n")
    monkeypatch.setattr(task, "ROOT", tmp_path)
    monkeypatch.setattr(
        task,
        "PINNED",
        {
            str(p): sha256_file(p)
            for p in (
                task.PRIOR,
                task.INVENTORY,
                task.HISTORY,
                task.HISTORY_INVENTORY,
                task.REGISTRY,
            )
        },
    )


# SCENARIO-REPORT-7988-FRONTIER: failure does not promote the retired primary.
def test_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authority(tmp_path, monkeypatch)
    checked = task.inputs()
    assert not checked["failures"]
    assert checked["registry_precheck"] == {"g1": 2}
    assert checked["prior"]["historical_required_failures"][-1]["verdict_class"] == "disqualified"
    for path, field, actual in (
        (task.PRIOR, "arc_evidence_ready_score", 0),
        (task.HISTORY, "verdict_class", "null"),
        (task.PRIOR, "experiment_id", 9),
    ):
        saved = path.read_bytes()
        value = json.loads(saved)
        value[field] = actual
        atomic_json(path, value)
        task.PINNED[str(path)] = sha256_file(path)
        assert any(r["artifact_field"] == field for r in task.inputs()["failures"])
        path.write_bytes(saved)
        task.PINNED[str(path)] = sha256_file(path)
    (tmp_path / "ops/exclusion_manifest.yaml").write_text("retired:\n- experiment_id: 7988\n")
    assert any(r["artifact_field"] == "retired" for r in task.inputs()["failures"])
    task.HISTORY_INVENTORY.unlink()
    assert any(r["observed"] == "missing" for r in task.inputs()["failures"])
    task.HISTORY.unlink()
    assert any(r["observed"] == "missing" for r in task.inputs()["failures"])


# SCENARIO-REPORT-7988-CLI: coverage scope covers only new code and the shared extension.
def test_commands(tmp_path: Path) -> None:
    specs = task.commands(tmp_path)
    assert {a for s in specs for a in s["argv"] if a.startswith("--include=")} == {
        "--include=" + task.INCLUDE
    }
    assert not any(s["name"].startswith("e2e_016") for s in specs)
    assert any(s["name"] == "full_python_suite" for s in specs)
    assert all(s["deadline_s"] <= 120 for s in specs)


# SCENARIO-REPORT-7988-ROWS and SCENARIO-VERIFY-7988-QUALIFICATION: no synthetic firing.
def test_fields_and_replay() -> None:
    value = dict(
        task.previous.reduce([]),
        rows=[],
        source_artifact_hashes={"code.py": "hash"},
        scan_source_hashes={},
        new_firing_count=0,
        new_helped_count=0,
        live_path_receipts=[],
    )
    value.update(task.artifact_fields(value, {}))
    assert value["empty_ledger_control"]["identity_filter_count"] == 0
    assert value["sample_interpretation"] == "descriptive"
    assert not task.replay(value)
    value["empty_ledger_control"]["identity_filter_count"] = 1
    assert "empty_ledger_control" in task.replay(value)
    rows = [
        dict(
            game="g1",
            arm="a",
            receipt_id="r",
            invocation_id="i",
            seed=0,
            resolved_by_levelup=True,
            actions_to_levelup=2,
            stagnations_unredirected=3,
        )
    ] * 2
    value["new_event_rows"] = rows
    fields = task.artifact_fields(value, {})
    assert fields["per_game_arm_statistics"]["g1"]["a"] == dict(
        fired=2, helped=2, actions_to_levelup=[2, 2], stagnations_unredirected=3
    )


# SCENARIO-REPORT-7988-CLI: actual callable CLI also rejects false cold summaries.
def test_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    producer = tmp_path / "producer.json"
    atomic_json(
        producer, dict(run_date="20261001", verdict_class="null", source_artifact_hashes={})
    )
    output = tmp_path / "success" / task.OUTPUT.name
    args = ["--reduce-ledger", str(tmp_path), "--producer", str(producer), "--output", str(output)]
    assert cli.main(args) == 0
    assert cli.main(["--cold-replay", str(output)]) == 0
    value = json.loads(output.read_text())
    value["identity_filter_count"] = 1
    atomic_json(output, value)
    assert cli.main(["--cold-replay", str(output)]) == 1
    producer.unlink()
    assert cli.main(args) == 1
    monkeypatch.setattr(task, "execute", lambda *_args: 0)
    assert cli.main([]) == 0


# SCENARIO-REPORT-7988-SEAL: the shared runner receives the new declared scope.
def test_execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def execute(output: Path, private: Path, scope: Any) -> int:
        assert scope.EXPERIMENT_ID == 7988
        return 0

    monkeypatch.setattr(task.runner, "execute", execute)
    assert task.execute(tmp_path / task.OUTPUT.name, tmp_path) == 0


# SCENARIO-REPORT-7988-SEAL: the owned replay child clears checkout import overrides.
def test_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task.previous, "terminal", lambda *_args: dict(passed=True, reports=[]))

    def run(spec: dict[str, Any], *_args: Any) -> dict[str, Any]:
        assert spec["argv"][:3] == ["env", "-u", "PYTHONPATH"]
        assert str(tmp_path) in spec["argv"]
        return dict(name=spec["name"], passed=True)

    monkeypatch.setattr(task.previous, "run", run)
    assert task.terminal(tmp_path / "primary.json", tmp_path, tmp_path)["passed"]
