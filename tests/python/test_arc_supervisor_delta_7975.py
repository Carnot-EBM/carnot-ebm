"""REQ-REPORT-7975: frontier custody prevents repeated scientific work."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import arc_supervisor_v691_delta as task
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7975_v691_arc_supervisor_delta as cli


def authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Private bytes exercise authentication without altering research history."""
    (tmp_path / "results").mkdir()
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops/exclusion_manifest.yaml").write_text("retired: []\n")
    seen = {"raw:sha256:old": "sha256:old"}
    prior, inventory, registry = (
        tmp_path / n for n in ("prior.json", "inventory.json", "registry.yaml")
    )
    atomic_json(inventory, dict(receipt_inventory=seen, seen_receipt_hashes=seen))
    atomic_json(
        prior,
        dict(
            experiment_id=7962,
            task_id="exp7962-arc-supervisor-delta",
            run_date="20261001",
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


# SCENARIO-REPORT-7975-FRONTIER: authority mismatch is terminal, including failed thresholds.
def test_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authority(tmp_path, monkeypatch)
    assert task.inputs()["registry_precheck"] == {"g1": 2}
    assert not task.inputs()["failures"]
    history = tmp_path / "results/raw" / task.OUTPUT.stem / "prior_owned_attempt.json"
    atomic_json(
        history,
        dict(
            failed_required_checks=[dict(name="owned_old_unit", passed=False)],
            repository_health=[dict(name="full_python_suite", passed=False)],
            frozen_snapshots={},
        ),
    )
    checked = task.inputs()
    assert (
        checked["prior"]["historical_required_failures"][-1]["failed_required_checks"][0]["passed"]
        is False
    )
    assert checked["owned_repository_health"][0]["passed"] is False
    exclusion = tmp_path / "ops/exclusion_manifest.yaml"
    exclusion.write_text("retired:\n- experiment_id: 7962\n")
    assert any(r["artifact_field"] == "retired" for r in task.inputs()["failures"])
    exclusion.write_text("retired: []\n")
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


# SCENARIO-REPORT-7975-CLI: all actual private publication routes retain separate paths.
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


# SCENARIO-REPORT-7975-SEAL: dates, deadlines and coverage scope freeze before science.
def test_commands(tmp_path: Path) -> None:
    specs = task.commands(tmp_path)
    assert {a for s in specs for a in s["argv"] if a.startswith("--include=")} == {
        "--include=" + task.INCLUDE
    }
    for s in specs:
        assert s["deadline_s"] <= 120
        if s["name"].startswith("e2e_016"):
            assert s["argv"][s["argv"].index("--date") + 1] == "20260929"
        if s["name"] in {"affected_pytest", "unit_coverage", "full_python_suite"}:
            assert any(a.startswith("--basetemp=" + str(tmp_path)) for a in s["argv"])
    assert any(s["classification"] == "repository_health" for s in specs)
    assert any(s["name"] == "legacy_cli_success" for s in specs)


# SCENARIO-REPORT-7975-SEAL: a measured health run is retained without running it twice.
def test_health_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task, "ROOT", tmp_path)
    history = tmp_path / "results/raw" / task.OUTPUT.stem / "prior_owned_attempt.json"
    atomic_json(history, dict(repository_health=[dict(name="full_python_suite", passed=False)]))
    assert not any(s["name"] == "full_python_suite" for s in task.commands(tmp_path / "private"))


# SCENARIO-REPORT-7975-HEALTH-REUSE: history stays explicit and is never rerun.
def test_failed_health_receipt_is_planned_and_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority(tmp_path, monkeypatch)
    history = tmp_path / "results/raw" / task.OUTPUT.stem / "prior_owned_attempt.json"
    health = dict(
        name="full_python_suite",
        passed=False,
        exit_code=-15,
        timed_out=True,
        command_argv=["pytest", "tests/python", "-q"],
        classification="repository_health",
    )
    atomic_json(history, dict(repository_health=[health], frozen_snapshots={}))
    specs = task.commands(tmp_path / "planned")
    reused = [s for s in specs if s["classification"] == "repository_health"]
    assert len(reused) == 1
    assert reused[0]["reuse_receipt"] == health
    assert reused[0]["argv"] == health["command_argv"]
    assert reused[0]["reuse_source_path"] == str(history)
    assert reused[0]["reuse_source_sha256"] == sha256_file(history)

    def commands(private: Path) -> list[dict[str, Any]]:
        atomic_json(
            private / "coverage.json",
            dict(
                files={
                    n: dict(summary=dict(num_statements=1, covered_lines=1), missing_lines=[])
                    for n in task.ADDED
                }
            ),
        )
        return reused

    def unexpected_run(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        pytest.fail("A reused health receipt must not launch a child command")

    monkeypatch.setattr(task, "commands", commands)
    monkeypatch.setattr(task.previous, "run", unexpected_run)
    monkeypatch.setattr(
        task.previous, "terminal", lambda *_args: dict(passed=True, reports=[], replay_errors=[])
    )
    monkeypatch.setattr(task.validation, "dependency_hashes", lambda *_args, **_kwargs: {})
    output = tmp_path / "results" / task.OUTPUT.name
    assert task.execute(output, tmp_path / "private") == 0
    value = json.loads(output.read_text())
    receipt = value["validation_receipts"][0]
    assert receipt["passed"] is False and receipt["timed_out"] is True
    assert receipt["exit_code"] == -15 and receipt["reused"] is True
    assert receipt["classification"] == "repository_health"
    assert not value["observed_child_commands"]
    assert value["verdict_class"] == "null"
    assert value["repository_health"]["checks"][0]["passed"] is False


# SCENARIO-REPORT-7975-SEAL: new summary counters are checked against primitive rows.
def test_new_counters() -> None:
    value = task.previous.reduce([])
    value.update(rows=[], new_firing_count=1, new_helped_count=1, live_path_receipts=[{}])
    assert set(task.replay(value)) == {"new_firing_count", "new_helped_count", "live_path_receipts"}


# SCENARIO-REPORT-7975-SEAL: owned failures disqualify; external failure blocks.
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
    private = tmp_path.parent / ("private-" + state)
    if state == "conflict":
        atomic_json(output.parent / "experiment_7975_conflict.json", {})
        with pytest.raises(ValueError, match="conflicting_primary"):
            task.execute(output, private)
        return
    expected = state if state in {"null", "blocked"} else "disqualified"
    assert task.execute(output, private) == int(expected != "null")
    value = json.loads(output.read_text())
    assert value["verdict_class"] == expected and value["experiment_id"] == 7975
    assert value["run_date"] == "20261001" and value["model_invocation_counts"] == 0
    assert value["milestone"] == "2026.10.691"
    assert value["execution_date"] == "20261001"
    assert value["started_at"] <= value["finished_at"]
    assert value["new_firing_count"] == 0 and value["new_helped_count"] == 0
    assert value["scratch_root_receipt"]["outside_checkout"] is True
    assert value["live_path_receipts"] == []
    assert value["new_level_solves_claimed"] == 0 and value["historical_required_failures"]
    receipt = json.loads(
        (output.parent / "raw" / output.stem / "primary_resolution_receipt.json").read_text()
    )
    assert receipt["passed"] and receipt["gate_sha256"] == sha256_file(output)
    monkeypatch.setattr(task, "execute", lambda *_args: 0)
    assert cli.main(["--output", str(output)]) == 0


# SCENARIO-REPORT-7975-FRONTIER: an unseen current-day event is eligible by identity.
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
