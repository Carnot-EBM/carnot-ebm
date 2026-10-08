"""REQ-REPORT-8300 / REQ-VERIFY-8300: retain qualified assertions under current bindings.

The prior tests still execute their original assertions. Private globals route
their real CLI and replay calls to the new audit without changing historical tests.
"""

from pathlib import Path
from types import FunctionType
from typing import Any

import pytest
import test_arc_outcome_frontier_8286 as previous

from carnot.reporting import arc_outcome_execution_8300 as task
from carnot.reporting import arc_outcome_frontier_8300 as reader
from carnot.reporting.current_work_receipt import atomic_json


def inherited(name: str) -> Any:
    """Retain all prior assertions while running the actual current script path."""
    original = getattr(previous, name)
    reused = FunctionType(
        original.__code__, vars(previous) | globals(), argdefs=original.__defaults__
    )
    reused.__kwdefaults__ = original.__kwdefaults__
    return reused


frontier = inherited("frontier")
inspect = inherited("inspect")
invoke = inherited("invoke")
test_delta_and_support = inherited("test_delta_and_support")
test_cli = inherited("test_cli")
test_execution_checks = inherited("test_execution_checks")
test_projection_and_object_tamper = inherited("test_projection_and_object_tamper")
test_current_date_boundary = inherited("test_current_date_boundary")
test_five_firing_floor = inherited("test_five_firing_floor")


def test_authentication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8300-DELTA: every prior negative authority case still fails."""
    check = inherited("test_authentication")
    for mutation in ("missing", "json", "sidecar", "code", "pin", "source"):
        case = tmp_path / mutation
        case.mkdir()
        with monkeypatch.context() as context:
            check(case, context, mutation)


def test_split_plan(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8300-SPLIT: preserve seven files, deadlines and health scope."""
    plan = task.commands(tmp_path)
    consumers = [
        s for s in plan if s["name"].startswith("consumer_") and s["name"] != "consumer_collection"
    ]
    assert len(consumers) == 7
    assert [s["argv"][-2] for s in consumers] == task.CONSUMERS
    assert all(s["deadline_s"] == 240 and s["classification"] == "required" for s in consumers)
    assert all("-u" in s["argv"] and "-vv" in s["argv"] for s in consumers)
    assert sum(s["deadline_s"] for s in consumers) <= 1800
    collection = next(s for s in plan if s["name"] == "consumer_collection")
    assert all(p in collection["argv"] for p in task.CONSUMERS)
    assert (
        next(s for s in plan if s["name"] == "full_python_suite")["classification"]
        == "repository_health"
    )
    assert task.reader.FRONTIER == previous.reader.FRONTIER
    assert task.reader.TASK_ID == "exp8300-arc-outcome-frontier"


def test_child_equivalence(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8300-SPLIT: missing identities and actual failures cannot qualify."""
    py = str(reader.ROOT / ".venv/bin/python")
    identities = [task.CONSUMERS[0] + "::test_one", task.CONSUMERS[0] + "::test_two"]
    (tmp_path / "consumer_collection.stdout").write_text("\n".join(identities) + "\n")
    program = "print(" + repr("\n".join(p + " PASSED [100%]" for p in identities)) + ")"
    row = task.child("consumer_0", [py, "-u", "-c", program], tmp_path, deadline=5)
    assert row["passed"] and row["collection_equivalent"]
    assert row["collected_test_ids"] == identities == row["passed_test_ids"]
    assert row["normal_exit"] and not row["timed_out"]
    summary = task.consumer_summary([row])
    assert summary["collected_test_ids"] == identities == summary["passed_test_ids"]
    assert not summary["collection_equivalent"] and summary["elapsed_s"] < 1800
    row = task.child("consumer_0", [py, "-c", "print('nothing')"], tmp_path, deadline=5)
    assert not row["passed"] and not row["collection_equivalent"]
    row = task.child("consumer_0", [py, "-c", "raise SystemExit(1)"], tmp_path, deadline=5)
    assert not row["passed"] and row["exit_code"] == 1
    assert task.child("ordinary", [py, "-c", "print('ok')"], tmp_path, deadline=5)["passed"]
    atomic_json(tmp_path / "unused.json", {})


def test_consumer_summary_tamper(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8300-COLD: a forged collection summary fails real cold replay."""
    locator = previous.fixture(tmp_path)
    prior = frontier(tmp_path, locator, unchanged=True)
    output = tmp_path / task.OUTPUT.name
    result = invoke(
        tmp_path,
        "--locator",
        str(locator),
        "--frontier",
        str(prior),
        "--output",
        str(output),
        "--fixture-e2e",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = previous.json.loads(output.read_text())
    value["consumer_validation"]["collection_equivalent"] = True
    assert "consumer_summary_drift" in task.replay(value)
    atomic_json(output, value)
    replayed = invoke(tmp_path, "--cold-replay", str(output))
    assert replayed.returncode == 1 and "consumer_summary_drift" in replayed.stdout
