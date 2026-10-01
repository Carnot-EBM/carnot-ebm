"""REQ-REPORT-7962: frontier custody prevents repeated scientific work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import arc_supervisor_v690_delta as task
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7962_v690_arc_supervisor_delta as cli


def authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Private bytes exercise authentication without altering research history."""
    (tmp_path / "results").mkdir()
    seen = {"raw:sha256:old": "sha256:old"}
    prior, inventory, registry = (
        tmp_path / n for n in ("prior.json", "inventory.json", "registry.yaml")
    )
    atomic_json(inventory, dict(receipt_inventory=seen, seen_receipt_hashes=seen))
    atomic_json(
        prior,
        dict(
            verdict_class="null",
            honest_verdict="complete_null_no_new_supervisor_outcomes",
            flagged_adversarial=False,
            arc_evidence_ready_score=1,
            receipt_inventory=seen,
            seen_receipt_hashes=seen,
            source_artifact_hashes={str(inventory): sha256_file(inventory)},
            historical_required_failures=[{"name": "old_failure"}],
            repository_health={"historical": True},
        ),
    )
    registry.write_text("games:\n  g1:\n    levels_reproduced: 2\n")
    for name, value in (
        ("ROOT", tmp_path),
        ("PRIOR", prior),
        ("INVENTORY", inventory),
        ("REGISTRY", registry),
    ):
        monkeypatch.setattr(task, name, value)
    monkeypatch.setattr(
        task, "PINNED", {str(p): sha256_file(p) for p in (prior, inventory, registry)}
    )


# SCENARIO-REPORT-7962-FRONTIER: authority mismatch is terminal, including failed thresholds.
def test_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authority(tmp_path, monkeypatch)
    assert task.inputs()["registry_precheck"] == {"g1": 2}
    assert not task.inputs()["failures"]
    prior = json.loads(task.PRIOR.read_text())
    prior["arc_evidence_ready_score"] = 0
    atomic_json(task.PRIOR, prior)
    task.PINNED[str(task.PRIOR)] = sha256_file(task.PRIOR)
    assert task.inputs()["failures"]
    task.REGISTRY.write_text("games:\n- game: g1\n  levels_reproduced: 3\n")
    task.PINNED[str(task.REGISTRY)] = sha256_file(task.REGISTRY)
    assert task.inputs()["registry_precheck"] == {"g1": 3}
    task.INVENTORY.unlink()
    assert any(r["observed"] == "missing" for r in task.inputs()["failures"])


# SCENARIO-REPORT-7962-CLI: all actual private publication routes retain separate paths.
def test_cli(tmp_path: Path) -> None:
    producer = tmp_path / "producer.json"
    atomic_json(
        producer, dict(run_date="20261001", verdict_class="null", source_artifact_hashes={})
    )
    output = tmp_path / "success" / task.OUTPUT.name
    args = [
        "--date",
        "20261001",
        "--reduce-ledger",
        str(tmp_path),
        "--producer",
        str(producer),
        "--output",
        str(output),
    ]
    assert cli.main(args) == 0
    assert (
        cli.main(["--cold-replay", str(output), "--output", str(tmp_path / "replay/report.json")])
        == 0
    )
    assert cli.main(["--terminal-recheck", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["fixture_claim_scope"] == "circular_positive"
    value["identity_filter_count"] = 1
    atomic_json(output, value)
    assert cli.main(["--cold-replay", str(output)]) == 1
    producer.unlink()
    assert cli.main(args) == 1
    with pytest.raises(SystemExit):
        cli.main(["--reduce-ledger", str(tmp_path)])
    with pytest.raises(SystemExit):
        cli.main(["--date", "20260930"])


# SCENARIO-REPORT-7962-SEAL: dates, deadlines and coverage scope freeze before science.
def test_commands(tmp_path: Path) -> None:
    specs = task.commands(tmp_path)
    assert {a for s in specs for a in s["argv"] if a.startswith("--include=")} == {
        "--include=" + task.INCLUDE
    }
    for s in specs:
        assert s["deadline_s"] <= 120
        if s["name"].startswith("e2e_016"):
            assert s["argv"][s["argv"].index("--date") + 1] == "20260929"
    assert any(s["classification"] == "repository_health" for s in specs)


# SCENARIO-REPORT-7962-SEAL: owned failures disqualify; external failure blocks.
@pytest.mark.parametrize(
    "state", ["null", "blocked", "failed", "coverage", "terminal", "new", "conflict"]
)
def test_execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    authority(tmp_path, monkeypatch)

    def commands(private: Path) -> list[dict[str, Any]]:
        atomic_json(
            private / "coverage.json",
            {
                "files": {
                    n: {"summary": {"num_statements": 1, "covered_lines": 1}, "missing_lines": []}
                    for n in task.ADDED
                }
            },
        )
        return [
            dict(
                name="unit",
                argv=["true"],
                expected_exit=0,
                expected_text=None,
                deadline_s=5,
                classification="required",
            )
        ]

    def run(spec: dict[str, Any], *_args: Any) -> dict[str, Any]:
        passed = state != "failed"
        return dict(
            spec,
            passed=passed,
            exit_code=int(not passed),
            command_argv=spec["argv"],
            log_path="private.log",
            log_sha256="sha256:private",
        )

    calls = 0

    def terminal(*_args: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return dict(passed=state != "terminal" or calls > 1, reports=[], replay_errors=[])

    monkeypatch.setattr(task, "commands", commands)
    monkeypatch.setattr(task.previous, "run", run)
    monkeypatch.setattr(task.previous, "terminal", terminal)
    monkeypatch.setattr(task.validation, "dependency_hashes", lambda *_args, **_kwargs: {})
    if state == "blocked":
        task.INVENTORY.unlink()
    if state == "coverage":
        monkeypatch.setattr(task.validation, "coverage_complete", lambda *_args, **_kwargs: False)
    if state == "new":
        original = task.previous.scan

        def changed(*args: Any, **kwargs: Any) -> dict[str, Any]:
            value = original(*args, **kwargs)
            value["identity_filter_count"] = 1
            return value

        monkeypatch.setattr(task.previous, "scan", changed)
    output = tmp_path / "results" / task.OUTPUT.name
    if state == "conflict":
        atomic_json(output.parent / "experiment_7962_conflict.json", {})
        with pytest.raises(ValueError, match="conflicting_primary"):
            task.execute(output, tmp_path / "private")
        return
    expected = state if state in {"null", "blocked"} else "disqualified"
    assert task.execute(output, tmp_path / "private") == int(expected != "null")
    value = json.loads(output.read_text())
    assert value["verdict_class"] == expected and value["experiment_id"] == 7962
    assert value["run_date"] == "20261001" and value["model_invocation_counts"] == 0
    assert value["new_level_solves_claimed"] == 0 and value["historical_required_failures"]
    receipt = json.loads(
        (output.parent / "raw" / output.stem / "primary_resolution_receipt.json").read_text()
    )
    assert receipt["passed"] and receipt["gate_sha256"] == sha256_file(output)
    monkeypatch.setattr(task, "execute", lambda *_args: 0)
    assert cli.main(["--output", str(output)]) == 0


# SCENARIO-REPORT-7962-FRONTIER: an unseen current-day event is eligible by identity.
def test_current_event(tmp_path: Path) -> None:
    root = tmp_path / "ledger"
    raw = root / "results/raw/live.json"
    episode = dict(
        game="g1",
        seed=7,
        invocation_id="new",
        receipt_id="new",
        event_sequence=9,
        sequence_scope="live-run",
        solve_provenance="live_agent_self_discovery",
        live_agent_provenance=dict(
            policy_class="E3AgentPolicy", agent_factory="make_carnot_agent", execution_mode="live"
        ),
        trajectory_supervisor=dict(
            enabled=True,
            mode="applied",
            redirects=[
                dict(
                    id="r1",
                    arm="drop_goal_bias",
                    fired=True,
                    resolved_by_levelup=False,
                    actions_to_levelup=None,
                )
            ],
        ),
    )
    atomic_json(raw, dict(rows=[episode]))
    producer = root / "results/experiment_7961_arc.json"
    atomic_json(
        producer,
        dict(
            run_date="20261001",
            verdict_class="null",
            source_artifact_hashes={str(raw): sha256_file(raw)},
        ),
    )
    value = task.previous.scan(
        root,
        [producer],
        dict(prior={}, inventory={}),
        tmp_path / "snapshot",
        current_date="20261001",
    )
    assert value["identity_filter_count"] == 1 and not task.previous.replay(value)
    assert value["per_game_results"]["g1"][0]["seed"] == 7
    assert value["refinement_decisions"][0]["decision"] == "unchanged"
