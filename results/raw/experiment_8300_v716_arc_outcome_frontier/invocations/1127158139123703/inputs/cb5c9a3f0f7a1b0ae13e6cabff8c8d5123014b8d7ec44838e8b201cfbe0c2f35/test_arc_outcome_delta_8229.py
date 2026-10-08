"""REQ-REPORT-8229 / REQ-VERIFY-8229: preserve the environment evidence boundary."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.reporting import arc_outcome_delta_8229 as reader
from carnot.reporting import arc_outcome_execution_8229 as task
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary
from test_arc_authoritative_frontier_8215 import event, fixture


def frontier(tmp_path: Path, locator: Path, *, unchanged: bool = False) -> Path:
    """Seal a private historical primary so controls cannot replace real history."""
    path = tmp_path / reader.FRONTIER.name
    publish_primary(
        path,
        dict(
            experiment_id=8215,
            task_id="exp8215-arc-authoritative-frontier",
            schema="arc-authoritative-frontier-v709",
            honest_verdict="complete_null_no_new_outcomes",
            verdict_class="null",
            required_checks_passed=True,
            arc_delta_ready_score=1,
            locator_path=str(locator),
            current_frontier=dict(receipt_ids=[], event_ids=[]),
            source_artifact_hashes={},
            excluded_count=33,
            finished_at="2026-10-06T23:00:00Z",
            authority_signature=reader.signature(json.loads(locator.read_text()))
            if unchanged
            else {},
        ),
        lambda _: dict(passed=True),
    )
    return path


def current(
    game: str = "cd82", arm: str = "drop_goal_bias", resolved: bool = True
) -> dict[str, Any]:
    """A dated environment resolution tests progress without running a game."""
    value = event(game, arm, resolved)
    value["finished_at"] = "2026-10-07T00:01:00Z"
    return value


def test_unchanged_empty_missing_and_changed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8229-DELTA: missing is blocked and unchanged skips old reduction."""
    locator = fixture(tmp_path)
    prior = frontier(tmp_path, locator, unchanged=True)
    value = reader.inspect(locator, prior)
    assert value["unchanged_authority"] and value["new_outcome_count"] == 0
    assert value["historical_excluded_count"] == 33 and value["rows"] == []
    prior = frontier(tmp_path, locator)
    assert reader.inspect(locator, prior)["new_outcome_count"] == 0
    Path(json.loads(locator.read_text())["ledger"]["path"]).unlink()
    blocked = reader.inspect(locator, prior)
    assert blocked["failures"] and blocked["new_outcome_count"] == 0
    assert all(
        {"path", "sha256", "artifact_field", "op", "expected", "observed"} <= r.keys()
        for r in blocked["failures"]
    )
    assert reader.inspect(tmp_path / "missing.json", prior)["failures"]
    assert reader.inspect(locator, tmp_path / "missing_prior.json")["failures"]


def test_environment_metrics_and_overlap(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8229-DELTA: environment progress and game overlap govern support."""
    rows = [
        current(g, a, a == "drop_goal_bias")
        for g in ("cd82", "r11l", "ar25")
        for a in ("drop_goal_bias", "allow_reinduction")
    ]
    locator = fixture(tmp_path, rows)
    prior = frontier(tmp_path, locator)
    value = reader.inspect(locator, prior)
    assert not value["failures"] and value["new_outcome_count"] == 6
    assert value["completed_count"] == 3 and value["censored_count"] == 3
    assert value["independent_count"] == 3
    assert len(value["per_game_arm_rows"]) == 6
    assert len(value["descriptive_leave_one_game_out"]) == 3
    assert value["selection_recommendations"] == []
    assert value["selection_propensity"]["status"] == "missing"
    one_game = reader.summarize(value["rows"][:2], {})
    assert not one_game["descriptive_leave_one_game_out"]
    old = deepcopy(rows[0])
    old["seed"] = 8
    old["finished_at"] = "2026-10-06T12:00:00Z"
    locator = fixture(tmp_path, [old, rows[0], rows[0]])
    value = reader.inspect(locator, prior)
    assert value["new_outcome_count"] == 1
    assert value["excluded_count"] == 2


def test_original_provenance_and_frontier_duplicates(tmp_path: Path) -> None:
    """REQ-REPORT-8229: authenticated bytes must also name the original live provenance."""
    episode = current()
    episode["solve_provenance"] = "development_proxy"
    locator = fixture(tmp_path, [episode])
    prior = frontier(tmp_path, locator)
    value = reader.inspect(locator, prior)
    assert value["new_outcome_count"] == 0
    assert value["rows"][0]["reason"] == "non_live_provenance"
    episode["solve_provenance"] = "live_agent_self_discovery"
    locator = fixture(tmp_path, [episode])
    value = reader.inspect(locator, prior)
    historical = json.loads(prior.read_text())
    historical["current_frontier"]["event_ids"] = [value["rows"][0]["event_id"]]
    publish_primary(prior, historical, lambda _: dict(passed=True))
    assert reader.inspect(locator, prior)["rows"][0]["reason"] == "duplicate_frontier"
    missing = dict(value["rows"][0], stagnations_unredirected=None)
    assert (
        reader.summarize([missing], {})["per_game_arm_rows"][0]["stagnations_unredirected"][
            "missing_count"
        ]
        == 1
    )


def invoke(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """A child outside the checkout proves that the executable imports its own code."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [str(reader.ROOT / ".venv/bin/python"), "-u", str(reader.ROOT / task.CLI), *args],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_real_cli_replay_and_failure_paths(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8229-CLI / SCENARIO-VERIFY-8229-COLD: real private child controls."""
    locator = fixture(tmp_path, [current()])
    prior = frontier(tmp_path, locator)
    output = tmp_path / task.OUTPUT.name
    args = [
        "--locator",
        str(locator),
        "--frontier",
        str(prior),
        "--output",
        str(output),
        "--fixture-e2e",
    ]
    result = invoke(tmp_path, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["new_outcome_count"] == 0 and value["fixture_result"]["new_outcome_count"] == 1
    assert value["solve_provenance"] is None and value["credited_new_levels"] == 0
    assert invoke(tmp_path, "--cold-replay", str(output)).returncode == 0
    value["credited_new_levels"] = 1
    atomic_json(output, value)
    assert invoke(tmp_path, "--cold-replay", str(output)).returncode == 1
    assert invoke(tmp_path, "--date", "20261006").returncode == 2
    assert invoke(tmp_path, "--fixture-e2e").returncode == 2
    assert invoke(tmp_path, "--frontier", str(prior)).returncode == 2
    args[1] = str(tmp_path / "absent.json")
    result = invoke(tmp_path, *args)
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads(output.read_text())["verdict_class"] == "blocked"
    assert invoke(tmp_path, "--cold-replay", str(output)).returncode == 0


def test_owned_checks_coverage_and_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-8229: actual child failure, coverage and tampering govern readiness."""
    locator = fixture(tmp_path, [current()])
    prior = frontier(tmp_path, locator)
    private = tmp_path / "work"
    specs = task.commands(private)
    assert {"e2e_017", "e2e_023", "coverage_100", "full_python_suite"} <= {s["name"] for s in specs}
    assert (
        next(s for s in specs if s["name"] == "full_python_suite")["classification"]
        == "repository_health"
    )
    monkeypatch.setattr(
        task,
        "commands",
        lambda _: [
            dict(
                name="actual_child",
                argv=[str(reader.ROOT / ".venv/bin/python"), "-c", "print('real')"],
                deadline_s=10,
                expected_exit=0,
                classification="required",
            )
        ],
    )
    report = private / "coverage.json"
    atomic_json(
        report,
        dict(
            files={
                p: dict(summary=dict(num_statements=1, covered_lines=1), missing_lines=[])
                for p in task.OWNED
            }
        ),
    )
    output = tmp_path / task.OUTPUT.name
    assert task.execute(locator, prior, output, private) == 0
    value = json.loads(output.read_text())
    assert (
        value["new_outcome_count"] == 1 and value["solve_provenance"] == "live_agent_self_discovery"
    )
    assert not task.replay(value)
    altered_receipt = deepcopy(value)
    altered_receipt["validation_receipts"][0]["expected_exit"] = 9
    assert "validation_receipt:actual_child" in task.replay(altered_receipt)
    changes = [
        dict(new_outcome_count=99),
        dict(MODEL_SPECS=["fake"]),
        dict(verdict_class="positive"),
    ]
    for change in changes:
        altered = dict(value, **change)
        assert task.replay(altered)
    stream = Path(value["validation_receipts"][0]["stdout_path"])
    stream.write_text("changed")
    assert task.replay(value)
    primitive = Path(value["primitive_path"])
    altered = json.loads(primitive.read_text())
    altered["new_outcome_count"] = 99
    atomic_json(primitive, altered)
    value["raw_shard_hashes"][str(primitive)] = sha256_file(primitive)
    assert "primitive_reduction_drift" in task.replay(value)
    primitive.unlink()
    assert "missing_primitive" in task.replay(value)
    monkeypatch.setattr(
        task,
        "commands",
        lambda _: [
            dict(
                name="actual_failure",
                argv=[str(reader.ROOT / ".venv/bin/python"), "-c", "raise SystemExit(1)"],
                deadline_s=10,
                expected_exit=0,
                classification="required",
            )
        ],
    )
    assert task.execute(locator, prior, output, private) == 1
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "disqualified" and value["arc_delta_ready_score"] == 0
    monkeypatch.setattr(task, "coverage_complete", lambda *a, **k: False)
    assert task.execute(locator, prior, output, private) == 1
    monkeypatch.setattr(task, "terminal", lambda *a: dict(passed=False))
    with pytest.raises(ValueError, match="terminal_candidate_rejected"):
        task.execute(locator, prior, output, private)


@pytest.mark.parametrize(
    "mutation",
    ["json", "list", "oversized", "sidecar", "foreign", "prior_hash", "producer", "registry"],
)
def test_authority_operand_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """SCENARIO-REPORT-8229-DELTA: bad bytes and foreign identity retain exact failures."""
    locator = fixture(tmp_path)
    prior = frontier(tmp_path, locator)
    if mutation == "json":
        locator.write_text("{")
    elif mutation == "list":
        locator.write_text("[]")
    elif mutation == "oversized":
        with locator.open("wb") as stream:
            stream.truncate(33554433)
    elif mutation == "sidecar":
        for path in (prior.parent / "raw" / prior.stem / "validators").glob("*.json"):
            path.unlink()
    elif mutation == "prior_hash":
        monkeypatch.setattr(reader, "FRONTIER", prior)
    elif mutation == "registry":
        monkeypatch.setattr(reader.authority, "REGISTRY", tmp_path / "absent.yaml")
    else:
        value = json.loads(locator.read_text())
        if mutation == "foreign":
            value["task_id"] = "foreign"
        else:
            value["producer_code_hashes"][str(tmp_path / "absent.py")] = "sha256:bad"
        atomic_json(locator, value)
    assert reader.inspect(locator, prior)["failures"]


def test_named_preconditions_and_default_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8229-CLI: no locator uses producer discovery; named missing inputs block."""
    assert not task.preconditions(tmp_path)["failures"]
    monkeypatch.setattr(task, "ROOT", tmp_path)
    assert task.preconditions(tmp_path)["failures"]
    output = tmp_path / task.OUTPUT.name
    result = invoke(tmp_path, "--output", str(output), "--fixture-e2e")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_text())["honest_verdict"] == "complete_null_no_new_outcomes"
