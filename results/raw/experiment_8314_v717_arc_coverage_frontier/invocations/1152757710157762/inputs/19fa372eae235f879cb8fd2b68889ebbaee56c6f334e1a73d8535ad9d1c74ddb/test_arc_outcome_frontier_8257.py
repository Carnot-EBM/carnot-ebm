"""REQ-REPORT-8257 / REQ-VERIFY-8257: authenticate before measuring new outcomes."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import arc_outcome_frontier_8257 as reader
from carnot.reporting import arc_outcome_execution_8257 as task
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary
from test_arc_authoritative_frontier_8215 import fixture
from test_arc_supervisor_frontier_8243 import current


def frontier(tmp_path: Path, locator: Path, *, unchanged: bool = False) -> Path:
    """Private history binds the actual reader bytes without replacing live evidence."""
    path = tmp_path / reader.FRONTIER.name
    publish_primary(
        path,
        dict(
            experiment_id=8243,
            task_id="exp8243-arc-supervisor-frontier",
            schema="arc-supervisor-frontier-v712",
            honest_verdict="complete_null_no_new_outcomes",
            verdict_class="null",
            required_checks_passed=True,
            arc_delta_ready_score=1,
            current_frontier=dict(receipt_ids=[], event_ids=[]),
            excluded_count=33,
            finished_at="2026-10-07T00:00:00Z",
            code_config_hashes={p: sha256_file(reader.ROOT / p) for p in reader.QUALIFIED_CODE},
            qualified_authority_signature=reader.qualified.signature(
                json.loads(locator.read_text())
            )
            if unchanged
            else {},
        ),
        lambda _: dict(passed=True),
    )
    return path


def inspect(locator: Path, prior: Path) -> dict[str, Any]:
    """Each private projection is scoped to the immutable history it adapts."""
    return reader.inspect(locator, prior, prior.parent / (sha256_file(prior)[7:] + "-adapter"))


def test_delta_and_support(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8257-DELTA / SCENARIO-ARC-WMTE-8257-JOIN: grounded overlap only."""
    locator = fixture(tmp_path)
    prior = frontier(tmp_path, locator, unchanged=True)
    empty = inspect(locator, prior)
    assert empty["unchanged_authority"] and empty["rows"] == []
    assert empty["receipt_frontier"]["after_finished_at"] == "2026-10-07T00:00:00Z"
    rows = [
        current(g, a, a == "drop_goal_bias")
        for g in ("cd82", "r11l", "ar25")
        for a in ("drop_goal_bias", "allow_reinduction")
        for _ in range(2)
    ]
    for index, row in enumerate(rows):
        row["seed"] = index
    locator = fixture(tmp_path, rows)
    prior = frontier(tmp_path, locator)
    value = inspect(locator, prior)
    assert not value["failures"] and value["new_outcome_count"] == 12
    assert value["independent_count"] == 3 and value["censored_count"] == 6
    assert len(value["per_game_arm_rows"]) == 6
    assert len(value["descriptive_leave_one_game_out"]) == 3
    assert value["proposed_arm_change"]["causal_superiority"] is False
    assert value["selection_recommendations"] == []
    small = reader.summarize(value["rows"][::2], {})
    assert small["proposed_arm_change"] is None and small["descriptive_leave_one_game_out"] == []
    historical = json.loads(prior.read_text())
    historical["current_frontier"]["event_ids"] = [value["rows"][0]["event_id"]]
    publish_primary(prior, historical, lambda _: dict(passed=True))
    assert inspect(locator, prior)["new_outcome_count"] == 11
    old = current()
    old["finished_at"] = "2026-10-06T23:59:59Z"
    old["gateway_card_actions_by_level"] = []
    locator = fixture(tmp_path, [old])
    value = inspect(locator, prior)
    assert value["new_outcome_count"] == 0


@pytest.mark.parametrize("mutation", ["missing", "json", "sidecar", "code", "pin", "source"])
def test_authentication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    """REQ-REPORT-8257: an unauthenticated operand stays missing rather than measured zero."""
    locator = fixture(tmp_path)
    prior = frontier(tmp_path, locator)
    if mutation == "missing":
        prior.unlink()
    elif mutation == "json":
        prior.write_text("{")
    elif mutation == "sidecar":
        for path in (prior.parent / "raw" / prior.stem / "validators").glob("*.json"):
            path.unlink()
    elif mutation == "pin":
        monkeypatch.setattr(reader, "FRONTIER", prior)
    elif mutation == "source":
        Path(json.loads(locator.read_text())["sources"][0]["path"]).unlink()
    else:
        value = json.loads(prior.read_text())
        value["code_config_hashes"] = {}
        publish_primary(prior, value, lambda _: dict(passed=True))
    value = reader.inspect(locator, prior, tmp_path / "adapter")
    assert value["failures"] and value["new_outcome_count"] == 0
    assert all(
        {"path", "sha256", "artifact_field", "op", "expected", "observed"} <= r.keys()
        for r in value["failures"]
    )


def invoke(tmp_path: Path, *args: str) -> Any:
    """Run the actual CLI outside the checkout so import and process statements count."""
    import os
    import subprocess

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


def test_cli(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8257-CLI / SCENARIO-VERIFY-8257-COLD: private real child replay."""
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
    assert value["proposed_arm_change"] is None and value["credited_new_levels"] == 0
    upstream = next(r for r in value["cited_upstream_artifacts"] if r["path"] == str(prior))
    assert {"finished_at", "current_frontier", "qualified_authority_signature"} <= set(
        upstream["fields_imported"]
    )
    assert invoke(tmp_path, "--cold-replay", str(output)).returncode == 0
    value["credited_new_levels"] = 1
    atomic_json(output, value)
    assert invoke(tmp_path, "--cold-replay", str(output)).returncode == 1
    assert invoke(tmp_path, "--date", "20261006").returncode == 2
    assert invoke(tmp_path, "--fixture-e2e").returncode == 2
    assert invoke(tmp_path, "--frontier", str(prior)).returncode == 2
    args[1] = str(tmp_path / "missing.json")
    result = invoke(tmp_path, *args)
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads(output.read_text())["verdict_class"] == "blocked"
    assert invoke(tmp_path, "--cold-replay", str(output)).returncode == 0
    result = invoke(tmp_path, "--output", str(output), "--fixture-e2e")
    assert result.returncode == 0, result.stdout + result.stderr


def test_execution_checks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-8257: actual exits and rehashed negative controls determine readiness."""
    locator = fixture(tmp_path, [current()])
    prior = frontier(tmp_path, locator)
    private = tmp_path / "work"
    specs = task.commands(private)
    assert {"e2e_015", "e2e_019", "coverage_100", "full_python_suite"} <= {s["name"] for s in specs}
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
    atomic_json(
        private / "coverage.json",
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
    assert value["new_outcome_count"] == 1 and not task.replay(value)
    for change in [
        dict(new_outcome_count=99),
        dict(MODEL_SPECS=["fake"]),
        dict(verdict_class="positive"),
    ]:
        assert task.replay(dict(value, **change))
    altered = deepcopy(value)
    altered["validation_receipts"][0]["expected_exit"] = 9
    assert "validation_receipt:actual_child" in task.replay(altered)
    Path(value["validation_receipts"][0]["stdout_path"]).write_text("tampered")
    assert task.replay(value)
    primitive = Path(value["primitive_path"])
    reduced = json.loads(primitive.read_text())
    reduced["new_outcome_count"] = 99
    atomic_json(primitive, reduced)
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
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    monkeypatch.setattr(task, "coverage_complete", lambda *a, **k: False)
    assert task.execute(locator, prior, output, private) == 1
    monkeypatch.setattr(task, "terminal", lambda *a: dict(passed=False))
    with pytest.raises(ValueError, match="terminal_candidate_rejected"):
        task.execute(locator, prior, output, private)
    assert not task.preconditions(tmp_path)["failures"]
    monkeypatch.setattr(task, "ROOT", tmp_path)
    assert task.preconditions(tmp_path)["failures"]


def test_projection_and_object_tamper(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8257-COLD: adapter bytes cannot acquire authority through rehashing."""
    locator = fixture(tmp_path)
    prior = frontier(tmp_path, locator, unchanged=True)
    adapter = tmp_path / "adapter"
    assert not reader.inspect(locator, prior, adapter)["failures"]
    projection = adapter / reader.qualified.FRONTIER.name
    value = json.loads(projection.read_text())
    value["finished_at"] = "2026-10-01T00:00:00Z"
    publish_primary(projection, value, lambda _: dict(passed=True))
    assert reader.inspect(locator, prior, adapter)["failures"]
    prior.write_text("[]")
    assert reader.inspect(locator, prior, adapter)["failures"]
