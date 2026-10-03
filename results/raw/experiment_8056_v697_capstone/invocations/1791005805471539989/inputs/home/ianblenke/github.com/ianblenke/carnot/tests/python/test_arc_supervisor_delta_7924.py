"""REQ-REPORT-7924-V687: current evidence keeps historical failures and dates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import arc_supervisor_v687_delta as task
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from scripts.experiments import experiment_7924_v687_arc_supervisor_delta as cli


def authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Private authorities exercise custody without changing research results."""
    inventory = tmp_path / "inventory.json"
    atomic_json(inventory, {"baseline": {}, "current": {}, "producer_paths": []})
    prior = json.loads(task.previous.PRIOR.read_text())
    prior.update(receipt_inventory_path=str(inventory), cutoff_receipt_hashes={})
    prior["source_artifact_hashes"] = {str(inventory): sha256_file(inventory)}
    path = tmp_path / "prior.json"
    atomic_json(path, prior)
    history = tmp_path / "history.json"
    atomic_json(
        history,
        {
            "verdict_class": "disqualified",
            "rows": [],
            "source_artifact_hashes": {},
            "receipt_inventory_path": str(inventory),
            "validation_receipts": [{"name": "old_e2e", "passed": False}],
        },
    )
    monkeypatch.setattr(task.previous, "ROOT", tmp_path)
    monkeypatch.setattr(task.previous, "PRIOR", path)
    monkeypatch.setattr(task.previous, "EXPECTED_PRIOR", sha256_file(path))
    monkeypatch.setattr(task, "ROOT", tmp_path)
    monkeypatch.setattr(task, "HISTORY", history)
    monkeypatch.setattr(task, "EXPECTED_HISTORY", sha256_file(history))
    (tmp_path / "results").mkdir()


# SCENARIO-REPORT-7924-CUSTODY: disqualified headlines do not move the accepted cutoff.
def test_history_and_missing_ledgers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authority(tmp_path, monkeypatch)
    inputs = task.inputs()
    assert not inputs["failures"]
    assert task.audit(inputs)["new_outcome_count"] == 0
    assert inputs["history"]["verdict_class"] == "disqualified"
    task.HISTORY.write_text("{}")
    assert task.inputs()["failures"][0]["artifact_field"] == "sha256"
    task.HISTORY.unlink()
    assert task.inputs()["failures"][0]["observed"] == "missing"
    task.previous.PRIOR.unlink()
    assert task.inputs()["failures"]


# SCENARIO-REPORT-7924-CUSTODY: previously seen raw bytes never become new firings.
def test_seen_raw_and_counterfeit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authority(tmp_path, monkeypatch)
    raw = tmp_path / "results/raw/live/rows.json"
    episode = {
        "game": "g1",
        "seed": 1,
        "attempt": "a",
        "solve_provenance": "live_agent_self_discovery",
        "trajectory_supervisor": {
            "mode": "applied",
            "redirects": [
                {
                    "id": "a",
                    "arm": "drop_goal_bias",
                    "fired": True,
                    "resolved_by_levelup": True,
                    "actions_to_levelup": 3,
                }
            ],
        },
    }
    atomic_json(raw, {"rows": [episode]})
    producer = tmp_path / "results/experiment_9001_arc.json"
    atomic_json(
        producer,
        {
            "verdict_class": "null",
            "source_artifact_hashes": {str(raw.relative_to(tmp_path)): sha256_file(raw)},
        },
    )
    inputs = task.inputs()
    fresh = task.audit(inputs)
    assert fresh["new_outcome_count"] == 1
    assert task.replay(fresh) == []
    fresh["per_game_results"]["g1"]["eligible"] = 2
    assert task.replay(fresh)
    fresh["per_game_results"]["g1"]["eligible"] = 1
    fresh["per_game_results"]["g1"]["arms"]["drop_goal_bias"]["firings"] = 2
    assert task.replay(fresh)
    fresh["per_game_results"]["g1"]["arms"] = {}
    assert task.replay(fresh)
    inputs["history"]["source_artifact_hashes"] = {str(raw): sha256_file(raw)}
    assert task.audit(inputs)["new_outcome_count"] == 0
    inputs["history"]["source_artifact_hashes"] = {}
    inputs["history"]["rows"] = [
        {"source_sha256": sha256_file(raw), "event_id": "old", "content_sha256": "sha256:event"}
    ]
    assert task.audit(inputs)["new_outcome_count"] == 0
    inputs["history"]["rows"] = []
    episode["solve_provenance"] = "development_proxy"
    atomic_json(raw, {"rows": [episode]})
    atomic_json(
        producer,
        {
            "verdict_class": "null",
            "source_artifact_hashes": {str(raw.relative_to(tmp_path)): sha256_file(raw)},
        },
    )
    assert task.audit(inputs)["new_outcome_count"] == 0
    raw.unlink()
    assert task.inputs()["failures"][-1]["observed"] == "missing"


# SCENARIO-REPORT-7924-VALIDATION: both historical calls retain their fixture date.
def test_manifest_and_wrong_date(tmp_path: Path) -> None:
    rows = task.commands(tmp_path)
    assert all(row["deadline_s"] > 0 for row in rows)
    for row in rows:
        if row["name"].startswith("e2e_016") and "wrong_date" not in row["name"]:
            assert row["argv"][row["argv"].index("--date") + 1] == "20260929"
    wrong = next(row for row in rows if row["name"] == "e2e_016_wrong_date")
    assert wrong["expected_exit"] == 1 and wrong["expected_text"] == "run_date_mismatch"
    assert any(row["name"] == "cli_terminal_coverage" for row in rows)
    includes = [arg for row in rows for arg in row["argv"] if arg.startswith("--include=")]
    assert len(set(includes)) == 1 and task.MODULE in includes[0]


# SCENARIO-REPORT-7924-TERMINAL: zero rows cannot support fabricated headline counts.
def test_private_routes_and_replay(tmp_path: Path) -> None:
    producer, output = tmp_path / "producer.json", tmp_path / "delta.json"
    atomic_json(producer, {"source_artifact_hashes": {}, "verdict_class": "null"})
    args = ["--reduce-ledger", str(tmp_path), "--producer", str(producer)]
    with pytest.raises(SystemExit):
        cli.main(args)
    with pytest.raises(SystemExit):
        cli.main(["--date", "20260929"])
    assert cli.main([*args, "--output", str(output)]) == 0
    assert cli.main(["--cold-replay", str(output)]) == 0
    assert cli.main(["--terminal-recheck", str(output)]) == 0
    value = json.loads(output.read_text())
    for field in ("intended", "eligible", "started", "independent_n"):
        value["sample_size_budget"][field] = 1
        assert task.replay(value)
        value["sample_size_budget"][field] = 0
    value["recommendation_rows"] = [{"arm": "invented"}]
    assert task.replay(value)
    value["recommendation_rows"] = []
    for field in ("firings", "new_outcome_count", "new_level_solves", "new_live_outcome_count"):
        forged = dict(value, **{field: 1})
        assert task.replay(forged)
    value["sample_size_budget"]["completed"] = 1
    assert task.replay(value)
    value["sample_size_budget"]["completed"] = 0
    value["per_game_results"] = {"g1": {"eligible": 1}}
    assert task.replay(value)
    atomic_json(output, value)
    assert cli.main(["--cold-replay", str(output)]) == 1


# SCENARIO-REPORT-7924-TERMINAL: all terminal classes keep their actual validation evidence.
@pytest.mark.parametrize(
    "state", ["null", "positive", "blocked", "owned_failure", "coverage_failure", "resumed_null"]
)
def test_execute_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    authority(tmp_path, monkeypatch)
    real_commands = task.commands
    task.commands(tmp_path / "unused")
    private = tmp_path / "private"
    private.mkdir()

    def specs(directory: Path) -> list[dict[str, Any]]:
        atomic_json(
            directory / "coverage.json",
            {
                "files": {
                    name: {
                        "summary": {"num_statements": 1, "covered_lines": 1},
                        "missing_lines": [],
                    }
                    for name in (task.MODULE, task.CLI)
                }
            },
        )
        return [
            dict(
                name="unit",
                argv=["true"],
                classification="required",
                deadline_s=10,
                expected_exit=0,
                expected_text=None,
            )
        ]

    monkeypatch.setattr(task, "commands", specs)
    monkeypatch.setattr(task.validation, "dependency_hashes", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        task, "sha256_file", lambda path: sha256_file(path) if path.is_file() else "sha256:private"
    )
    if state == "blocked":
        task.HISTORY.unlink()
    if state == "positive":
        original = task.audit

        def positive(checked: dict[str, Any]) -> dict[str, Any]:
            value = original(checked)
            value.update(new_outcome_count=1)
            return value

        monkeypatch.setattr(task, "audit", positive)
        monkeypatch.setattr(task, "replay", lambda _: [])
    if state == "coverage_failure":
        monkeypatch.setattr(task.validation, "coverage_complete", lambda *_args, **_kwargs: False)

    def run(spec: dict[str, Any], *_args: Any) -> dict[str, Any]:
        passed = not (state == "owned_failure" and spec["name"] == "unit")
        return dict(
            spec,
            passed=passed,
            exit_code=int(not passed),
            command_argv=spec["argv"],
            log_path=str(private / "log"),
            log_sha256="sha256:private",
            output_tail="",
        )

    monkeypatch.setattr(task, "run", run)
    output = tmp_path / "output.json"
    resume = None
    if state == "resumed_null":
        resume = tmp_path / "prior-attempt.json"
        spec = dict(
            name="full_python_suite",
            argv=["true"],
            classification="repository_health",
            deadline_s=10,
            expected_exit=0,
            expected_text=None,
        )
        monkeypatch.setattr(
            task,
            "resume_receipt",
            lambda _: (
                dict(run(spec), reused=True),
                dict(sha256="sha256:private", required_failures=[]),
            ),
        )
        original_specs = specs
        monkeypatch.setattr(task, "commands", lambda directory: [*original_specs(directory), spec])
    code = task.execute(output, private, resume)
    value = json.loads(output.read_text())
    expected = (
        "disqualified" if "failure" in state else "null" if state == "resumed_null" else state
    )
    assert value["verdict_class"] == expected
    assert code == int(state not in {"null", "positive", "resumed_null"})
    assert value["experiment_id"] == 7924 and value["milestone"] == "2026.09.687"
    assert value["new_level_solves"] == 0 and value["MODEL_SPECS"] == []
    assert value["solve_provenance"] == [] and value["coverage_statement_counts"]
    assert sha256_file(output) == sha256_file(private / "terminal_candidate.json")
    monkeypatch.setattr(task, "commands", real_commands)
    if state == "null":
        monkeypatch.setattr(cli.task, "execute", lambda *_args: 0)
        assert cli.main(["--output", str(output)]) == 0


# SCENARIO-REPORT-7924-TERMINAL: a required validator failure zeros readiness before recheck.
@pytest.mark.parametrize("always_fail", [False, True])
def test_terminal_recheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, always_fail: bool
) -> None:
    calls = 0

    def run(spec: dict[str, Any], *_args: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        passed = calls > 2 and not always_fail
        return dict(spec, passed=passed, log_path="private.log", log_sha256="sha256:private")

    monkeypatch.setattr(task, "run", run)
    value = dict(flagged_adversarial=False, gate_check_summary=[], acceptance_gate_results={})
    if always_fail:
        with pytest.raises(ValueError, match="terminal_recheck_failed"):
            task.publish(value, tmp_path / "output.json", tmp_path, tmp_path / "durable")
    else:
        task.publish(value, tmp_path / "output.json", tmp_path, tmp_path / "durable")
        assert value["arc_delta_ready_score"] == 0 and not value["flagged_adversarial"]
        assert value["verdict_class"] == "disqualified"


# SCENARIO-REPORT-7924-VALIDATION: expected exit alone cannot qualify the wrong failure.
def test_expected_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        task.validation,
        "run_check",
        lambda *_args: {"passed": True, "output_tail": "other failure"},
    )
    assert not task.run({"expected_text": "run_date_mismatch"}, tmp_path, tmp_path)["passed"]
    assert task.run({"expected_text": None}, tmp_path, tmp_path)["passed"]


# SCENARIO-REPORT-7924-VALIDATION: command options precede positional coverage paths.
def test_combine_option_position(tmp_path: Path) -> None:
    command = next(row for row in task.commands(tmp_path) if row["name"] == "coverage_combine")[
        "argv"
    ]
    assert command.index("--keep") < command.index(str(tmp_path / "coverage.unit"))


# SCENARIO-REPORT-7924-VALIDATION: diagnostic reuse authenticates logs and retains owned failures.
def test_resume_receipt(tmp_path: Path) -> None:
    log = tmp_path / "full.log"
    log.write_text("collection debt\n")
    prior = tmp_path / "attempt.json"
    atomic_json(
        prior,
        {
            "experiment_id": 7924,
            "verdict_class": "disqualified",
            "gate_check_summary": [{"check": "coverage_combine"}],
            "validation_receipts": [
                dict(
                    name="full_python_suite",
                    classification="repository_health",
                    log_path=str(log),
                    log_sha256=sha256_file(log),
                )
            ],
        },
    )
    receipt, historical = task.resume_receipt(prior)
    assert receipt["reused"] and historical["sha256"] == sha256_file(prior)
    assert historical["required_failures"]
    log.write_text("changed")
    with pytest.raises(ValueError, match="repository_health_log_hash_mismatch"):
        task.resume_receipt(prior)
