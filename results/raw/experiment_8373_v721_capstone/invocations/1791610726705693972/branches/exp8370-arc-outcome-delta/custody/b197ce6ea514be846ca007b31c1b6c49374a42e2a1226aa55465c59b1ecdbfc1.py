"""REQ-REPORT-8370 / REQ-VERIFY-8370: private controls never supply live evidence."""

from copy import deepcopy
from pathlib import Path
import runpy
import signal
import sys

import pytest

from carnot.reporting import arc_outcome_delta_8370 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child
from test_arc_authoritative_frontier_8215 import fixture
from test_arc_supervisor_frontier_8243 import current


def test_delta_and_support(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8370-DELTA: keep exclusions and censoring outside support."""
    prior = e.authenticate(e.FRONTIER)
    rows = [
        current(g, a, True)
        for g in ("cd82", "r11l", "ar25", "ls20", "ft09", "ka59", "lp85", "wa30")
        for a in ("drop_goal_bias", "allow_reinduction")
        for _ in range(5)
    ]
    for index, row in enumerate(rows):
        row.update(seed=index, finished_at="2026-10-10T00:01:00Z")
    rows += [deepcopy(rows[0])]
    non_outcome = dict(game="cd82", seed=999, finished_at="2026-10-10T00:01:00Z")
    rows += [non_outcome]
    delta = e.inspect(fixture(tmp_path / "supported", rows), prior, tmp_path / "adapter")
    assert delta["new_outcome_count"] == 80 and delta["independent_count"] == 8
    assert len(delta["descriptive_leave_one_game_out"]) == 8
    assert delta["proposed_generalization_change"]["live_priority_modified"] is False
    assert delta["selection_propensity"] == dict(status="unknown", value=None)
    assert e.summarize(delta["rows"][10:], {})["proposed_generalization_change"] is None
    unresolved = [
        dict(r, helped=False, resolved_by_levelup=False, status="censored", actions_to_levelup=None)
        if r["arm"] == "allow_reinduction"
        else r
        for r in delta["rows"]
    ]
    assert e.summarize(unresolved, {})["descriptive_leave_one_game_out"] == []
    censored = current("cd82", "drop_goal_bias", False)
    censored.update(seed=900, finished_at="2026-10-10T00:01:00Z")
    old = deepcopy(rows[0])
    old.update(seed=901, finished_at=prior["finished_at"])
    proxy = deepcopy(rows[1])
    proxy.update(seed=902, solve_provenance="development_proxy")
    invalid = deepcopy(rows[2])
    invalid.update(seed=903, gateway_card_actions_by_level=[])
    small = e.inspect(
        fixture(tmp_path / "mixed", [censored, old, proxy, invalid]),
        prior,
        tmp_path / "mixed-adapter",
    )
    assert small["censored_count"] == 1 and small["excluded_count"] == 3
    assert small["proposed_generalization_change"] is None
    assert {r["solve_provenance"] for r in small["rows"]} == {
        "development_proxy",
        "live_agent_self_discovery",
    }


def test_measure_replay_and_tamper(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8370-EXECUTION: rehashing invented deltas cannot pass."""
    raw = tmp_path / "raw"
    work = e.measure(raw, tmp_path)
    value = e.build(work, [dict(passed=True)], raw, tmp_path / e.OUTPUT.name)
    assert not work["failures"]
    assert value["experiment_id"] == 8370 and value["run_date"] == "20261010"
    assert value["honest_verdict"] == "complete_null_no_supervisor_outcomes"
    assert value["arc_reader_ready_score"] == 1 and value["arc_outcome_support_score"] == 0
    assert value["frontier_before"] == value["frontier_after"]
    assert value["no_change_receipt"]["authenticated"]
    assert value["prior_frontier"]["sha256"] == e.PIN
    assert (
        value["MODEL_SPECS"] == []
        and value["current_game_calls"] == value["current_model_calls"] == 0
    )
    assert not value["solve_credit_claimed"] and e.replay(value)
    assert not e.replay(dict(value, solve_credit_claimed=True)) and not e.replay({})
    assert all(r["passed"] for r in e.controls(value, tmp_path / "controls"))
    forged = deepcopy(work)
    forged["delta"]["new_outcome_count"] += 1
    atomic_json(Path(value["work_reference"]["path"]), forged)
    value["work_reference"]["sha256"] = sha256_file(Path(value["work_reference"]["path"]))
    value["reproducibility_checksum"] = canonical_hash(forged)
    assert not e.replay(value)


@pytest.mark.parametrize("mutation", ["bytes", "terminal", "qualification", "code", "report"])
def test_authentication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    """SCENARIO-REPORT-8370-DELTA: exact qualified bytes include both terminals."""
    source = e.FRONTIER
    if mutation == "bytes":
        source = tmp_path / source.name
        source.write_text("{}")
    elif mutation == "terminal":
        monkeypatch.setattr(e, "read_bound_sidecar", lambda *a: {})
    elif mutation == "qualification":
        original = e.json_document
        monkeypatch.setattr(
            e, "json_document", lambda p: dict(original(p), arc_reader_ready_score=0)
        )
    elif mutation == "code":
        monkeypatch.setattr(e, "READERS", ["absent.py"])
    else:
        original = e.json_document
        monkeypatch.setattr(
            e,
            "json_document",
            lambda p: (
                dict(original(p), publication={})
                if p.name == "terminal_validation.json"
                else original(p)
            ),
        )
    with pytest.raises((OSError, ValueError, KeyError)):
        e.authenticate(source)


def test_missing_and_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8370-EXECUTION: external absence remains blocked, never null."""
    plan = e.plan(tmp_path)
    assert {"strict_mypy", "private_E2E018_consumers", "qualified_ARC_reader"} <= {
        s["name"] for s in plan
    }
    assert all(s["deadline"] <= 240 for s in plan)
    assert not any("tests/python" in s["argv"] for s in plan)
    monkeypatch.setattr(e, "FRONTIER", tmp_path / "missing.json")
    work = e.measure(tmp_path / "blocked", tmp_path)
    value = e.build(work, [dict(passed=True)], tmp_path / "blocked", tmp_path / e.OUTPUT.name)
    assert value["verdict_class"] == "blocked" and value["arc_reader_ready_score"] == 0
    assert value["gate_check_summary"][0]["observed"] is None and value["no_change_receipt"] is None
    assert e.replay(value)


def test_censored_no_change(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8370-DELTA: an unresolved redirect cannot imply completed support."""
    row = current("cd82", "drop_goal_bias", False)
    row["finished_at"] = "2026-10-10T00:01:00Z"
    raw = tmp_path / "raw"
    work = e.measure(raw, tmp_path)
    work["delta"] = e.inspect(
        fixture(tmp_path / "censored", [row]), e.authenticate(e.FRONTIER), tmp_path / "adapter"
    )
    work["fixture_claim_scope"] = True
    value = e.build(work, [dict(passed=True)], raw, tmp_path / e.OUTPUT.name)
    assert value["censored_count"] == 1 and value["verdict_class"] == "null"
    assert value["no_change_receipt"]["new_completed_outcome_count"] == 0
    assert value["arc_outcome_support_score"] == 0


def test_task_alarm_restored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8370-EXECUTION: an interrupted run must release its task alarm."""
    main = runpy.run_path(str(e.ROOT / e.CLI))["main"]

    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted owned work")

    monkeypatch.setattr(e, "run", interrupted)
    previous = signal.getitimer(signal.ITIMER_REAL)
    with pytest.raises(RuntimeError, match="interrupted owned work"):
        main(["--private-e2e", "--output", str(tmp_path / e.OUTPUT.name)])
    assert signal.getitimer(signal.ITIMER_REAL) == previous


@pytest.mark.parametrize("failure", ["scratch", "tools", "authority", "protocol"])
def test_precondition_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """SCENARIO-VERIFY-8370-EXECUTION: real resources and frozen authority fail closed."""
    if failure == "scratch":
        monkeypatch.setattr(e.shutil, "disk_usage", lambda p: type("Disk", (), {"free": 0})())
    elif failure == "authority":
        monkeypatch.setattr(
            e.authority, "authority", lambda *a: dict(activated=False, gate_check_summary=[])
        )
    elif failure == "protocol":
        monkeypatch.setattr(e.authority.legacy.base, "PIN", "invalid")
    failures = [dict(passed=False, path="missing-tool")] if failure == "tools" else []
    work = e.measure(tmp_path / failure, tmp_path, failures)
    assert work["failures"] and not work["delta"]["new_outcome_count"]


def test_cli_children_failure_and_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8370-EXECUTION: actual child failures clear owned readiness."""
    main = runpy.run_path(str(e.ROOT / e.CLI))["main"]
    for args in [["--date", "20261009"], ["--private-e2e", "--output", str(e.OUTPUT)]]:
        with pytest.raises(SystemExit):
            main(args)
    output = tmp_path / e.OUTPUT.name
    receipt = child(
        "cli8370",
        [sys.executable, "-u", str(e.ROOT / e.CLI), "--private-e2e", "--output", str(output)],
        tmp_path / "child",
        deadline=60,
    )
    assert receipt["passed"]
    value = e.json_document(output)
    assert value["fixture_claim_scope"] and value["arc_reader_ready_score"] == 0
    assert main(["--cold-replay", str(output)]) == 0
    atomic_json(output, dict(value, solve_credit_claimed=True))
    assert main(["--cold-replay", str(output)]) == 1
    monkeypatch.setattr(
        e,
        "plan",
        lambda p: [
            dict(
                name="owned_failure",
                argv=[sys.executable, "-u", "-c", "raise SystemExit(7)"],
                deadline=5,
                expected=0,
                scope="owned",
            )
        ],
    )
    assert e.run(tmp_path / "failed" / e.OUTPUT.name, tmp_path) == 1
    monkeypatch.setattr(e, "plan", lambda p: [])
    assert e.run(tmp_path / "recovery" / e.OUTPUT.name, tmp_path) == 0
    failed = e.json_document(tmp_path / "failed" / e.OUTPUT.name)
    assert failed["verdict_class"] == "disqualified" and failed["arc_reader_ready_score"] == 0
    timeout = child(
        "timeout",
        [sys.executable, "-u", "-c", "while True: pass"],
        tmp_path / "children",
        deadline=0.02,
        heartbeat=0.01,
    )
    assert timeout["timed_out"] and not timeout["passed"]
