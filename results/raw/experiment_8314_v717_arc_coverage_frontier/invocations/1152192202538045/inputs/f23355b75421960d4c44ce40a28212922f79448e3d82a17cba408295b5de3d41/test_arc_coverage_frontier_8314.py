"""REQ-REPORT-8314 / REQ-VERIFY-8314: real coverage, receipt and CLI controls."""

from copy import deepcopy
import json
from pathlib import Path
from types import FunctionType
from typing import Any

import pytest
import test_arc_outcome_frontier_8286 as previous

from carnot.reporting import arc_outcome_execution_8300 as historical
from carnot.reporting import arc_coverage_execution_8314 as task
from carnot.reporting import arc_coverage_frontier_8314 as reader
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def inherited(name: str) -> Any:
    """Keep existing authentication and CLI assertions against the current binding."""
    original = getattr(previous, name)
    reused = FunctionType(
        original.__code__, vars(previous) | globals(), original.__name__, original.__defaults__
    )
    reused.__kwdefaults__ = original.__kwdefaults__
    return reused


frontier = inherited("frontier")
inspect = inherited("inspect")
invoke = inherited("invoke")
test_cli = inherited("test_cli")
test_projection_and_object_tamper = inherited("test_projection_and_object_tamper")
test_current_date_boundary = inherited("test_current_date_boundary")
test_five_firing_floor = inherited("test_five_firing_floor")


@pytest.mark.parametrize("mutation", ["missing", "json", "sidecar", "code", "pin", "source"])
def test_authentication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    """SCENARIO-REPORT-8314-REPLAY: prior negative authority assertions remain active."""
    inherited("test_authentication")(tmp_path, monkeypatch, mutation)


def test_gap_regression(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8314-GAP: execute the actual previously missed statement."""
    old = json.loads(historical.OUTPUT.read_text())
    gap = task.historical_gap()
    assert gap["missing_lines"] == [212]
    assert gap["covered_lines"] == 84 and gap["num_statements"] == 85
    forged = dict(old, consumer_validation={"collection_equivalent": "forged"})
    assert "consumer_summary_drift" in historical.replay(forged)
    plan = task.commands(tmp_path)
    assert "full_python_suite" not in {s["name"] for s in plan}
    assert {"unit_child_coverage", "combine_coverage", "coverage_100", "e2e_017", "e2e_023"} <= {
        s["name"] for s in plan
    }
    assert (
        str(historical.ROOT / "python/carnot/reporting/arc_outcome_execution_8300.py")
        in (tmp_path / "coverage.ini").read_text()
    )
    assert "--keep" in next(s["argv"] for s in plan if s["name"] == "combine_coverage")
    combined = next(s["argv"] for s in plan if s["name"] == "combine_coverage")
    assert combined.index("--keep") < len(combined) - 1
    consumers = [
        s for s in plan if s["name"].startswith("consumer_") and s["name"] != "consumer_collection"
    ]
    assert [s["argv"][-2] for s in consumers] == historical.CONSUMERS
    assert all(s["deadline_s"] == 240 for s in consumers)
    assert sum(s["deadline_s"] for s in consumers) <= 1800


def test_cell_floor(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8314-CELLS: every game-arm cell must meet its own floor."""
    events = [
        previous.current(g, a, a == "drop_goal_bias")
        for g in ("cd82", "r11l", "ar25")
        for a in ("drop_goal_bias", "allow_reinduction")
        for _ in range(5)
    ]
    for index, row in enumerate(events):
        row["seed"] = index
    locator = previous.fixture(tmp_path, events)
    value = inspect(locator, frontier(tmp_path, locator))
    assert value["new_outcome_count"] == 30 and value["independent_count"] == 3
    assert len(value["descriptive_leave_one_game_out"]) == 3
    assert value["proposed_arm_change"]["live_priority_modified"] is False
    assert all(c["fired"] == 5 for c in value["per_game_arm_rows"])
    assert value["selection_propensity"]["status"] == "missing"
    thin = reader.summarize(value["rows"][1:], {})
    assert thin["proposed_arm_change"] is None and thin["descriptive_leave_one_game_out"] == []


def test_replay_and_missing_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8314-RECOVERY: real child failures and absent shards zero readiness."""
    locator = previous.fixture(tmp_path, [previous.current()])
    prior = frontier(tmp_path, locator)
    output = tmp_path / task.OUTPUT.name
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(
        task,
        "commands",
        lambda _: [
            dict(
                name="actual_failure",
                argv=[str(reader.ROOT / ".venv/bin/python"), "-u", "-c", "raise SystemExit(1)"],
                deadline_s=5,
                expected_exit=0,
                classification="required",
            )
        ],
    )
    assert task.execute(locator, prior, output, work) == 1
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "disqualified" and value["arc_reader_ready_score"] == 0
    assert value["new_outcome_count"] == 0 and value["owned_missing_statement_count"] > 0
    assert not task.replay(value)
    forged = deepcopy(value)
    forged["consumer_validation"]["collection_equivalent"] = True
    assert "consumer_summary_drift" in task.replay(forged)
    for change in [
        dict(arc_reader_ready_score=1),
        dict(credited_new_levels=1),
        dict(experiment_id=8300),
        dict(solve_provenance="offline_bfs"),
        dict(sample_size_budget={}),
    ]:
        assert task.replay(dict(value, **change))
    primitive = Path(value["primitive_path"])
    reduced = json.loads(primitive.read_text())
    reduced["new_outcome_count"] = 99
    atomic_json(primitive, reduced)
    value["raw_shard_hashes"][str(primitive)] = sha256_file(primitive)
    assert "primitive_reduction_drift" in task.replay(value)
    monkeypatch.setattr(task, "terminal", lambda *a: dict(passed=False))
    with pytest.raises(ValueError, match="terminal_candidate_rejected"):
        task.execute(locator, prior, output, work)
    assert not task.preconditions(tmp_path)["failures"]
    monkeypatch.setattr(task, "ROOT", tmp_path)
    assert task.preconditions(tmp_path)["failures"]


def test_consumer_identities(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8314-CONSUMERS: retain V716's real child assertions unchanged."""
    import test_arc_outcome_frontier_8300 as checked

    checked.test_child_equivalence(tmp_path)


def test_cached_preflight_and_shards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8314-RECOVERY: retain a real CLI shard and run each child once."""
    locator = previous.fixture(tmp_path)
    prior = frontier(tmp_path, locator, unchanged=True)
    work = tmp_path / "work"
    cov = work / "coverage"
    cov.mkdir(parents=True)
    config = work / "private.ini"
    config.write_text("[run]\nparallel=True\ndata_file=" + str(cov / ".coverage") + "\n")
    argv = [
        str(reader.ROOT / ".venv/bin/python"),
        "-u",
        "-m",
        "coverage",
        "run",
        "--rcfile=" + str(config),
        str(reader.ROOT / task.CLI),
        "--date",
        "invalid",
    ]
    monkeypatch.setattr(
        task,
        "commands",
        lambda _: [
            dict(
                name="unit_child_coverage",
                argv=argv,
                deadline_s=30,
                expected_exit=2,
                classification="required",
            )
        ],
    )
    output = tmp_path / task.OUTPUT.name
    assert task.execute(locator, prior, output, work) == 1
    value = json.loads(output.read_text())
    row = value["validation_receipts"][0]
    assert row["passed"] and row["exit_code"] == 2
    assert row["started_monotonic_ns"] < value["phase_spans"][1]["started_monotonic_ns"]
    assert Path(row["stdout_path"]).read_text().count("start no_model_load") == 1
    assert any("/coverage/.coverage." in p for p in value["raw_shard_hashes"])
    assert value["arc_reader_ready_score"] == 0 and not task.replay(value)
    assert "coverage_shard_drift" in task.replay(dict(value, coverage_shards=[]))
    forged = dict(value, acceptance_gates=dict(value["acceptance_gates"], owned_coverage_100=True))
    assert "owned_coverage_evidence_missing" in task.replay(forged)


def test_historical_authentication_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8314-REPLAY: a bound historical file still needs the right verdict."""
    from carnot.reporting.primary_publication import publish_primary

    locator = previous.fixture(tmp_path)
    prior = frontier(tmp_path, locator, unchanged=True)
    old = json.loads(historical.OUTPUT.read_text())
    altered = tmp_path / historical.OUTPUT.name
    old.update(verdict_class="null", honest_verdict="complete_null_forged")
    publish_primary(altered, old, lambda _: dict(passed=True))
    monkeypatch.setattr(historical, "OUTPUT", altered)
    with pytest.raises(ValueError, match="historical_coverage_authentication"):
        task.historical_gap()
    assert task.preconditions(tmp_path)["failures"]
    output = tmp_path / task.OUTPUT.name
    assert task.execute(locator, prior, output, tmp_path / "work", fixture=True) == 1
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and value["arc_reader_ready_score"] == 0
    assert value["gate_check_summary"] and not task.replay(value)
