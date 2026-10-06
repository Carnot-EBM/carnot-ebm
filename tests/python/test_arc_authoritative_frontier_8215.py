"""REQ-REPORT-8215 / REQ-VERIFY-8215: producer authority and bounded CLI replay."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.agentic.arc_supervisor_refinement import empty_ledger, ingest_files
from carnot.reporting import arc_authoritative_frontier_8215 as reader
from carnot.reporting import arc_authoritative_execution_8215 as task
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def seal(path: Path, value: Any) -> dict[str, Any]:
    """Keep qualification inputs in private scratch, bound to exact bytes."""
    atomic_json(path, value)
    return dict(path=str(path), sha256=sha256_file(path))


def fixture(tmp_path: Path, events: list[dict[str, Any]] | None = None) -> Path:
    """A tiny producer-shaped corpus proves mechanics without running a game."""
    events = events or []
    source = tmp_path / "run.json"
    source_ref = seal(
        source,
        dict(
            experiment="arc_leaderboard_eval",
            policy="e3",
            random_seed=7,
            budget=100,
            per_game=events,
        ),
    )
    ledger = empty_ledger()
    ingest_files(ledger, [source], "2026-10-06T12:00:00Z")
    prior_ref = seal(
        tmp_path / "prior.json",
        dict(
            experiment_id=8189,
            supervisor_reader_ready_score=1,
            required_checks_passed=True,
            verdict_class="null",
            finished_at="2026-10-06T00:00:00Z",
        ),
    )
    inventory_ref = seal(
        tmp_path / "inventory.json", dict(rows=[], outcome_hashes=[], receipt_inventory={})
    )
    terminal_ref = seal(
        tmp_path / "terminal.json",
        dict(primary_sha256=prior_ref["sha256"], report=dict(passed=True)),
    )
    locator = dict(
        schema=reader.SCHEMA,
        task_id=reader.TASK_ID,
        ledger=seal(tmp_path / "ledger.json", ledger),
        prior=prior_ref,
        inventory=inventory_ref,
        terminal=terminal_ref,
        producer_code_hashes=reader.producer_hashes(),
        ledger_definition=list(reader.DEFAULT_LEDGER_PARTS),
        sources=[dict(source_ref, kind="eval", receipt_ids=list(ledger["entries"]))],
    )
    return Path(seal(tmp_path / "locator.json", locator)["path"])


def event(game: str = "cd82", arm: str = "drop_goal_bias", resolved: bool = True) -> dict[str, Any]:
    """Use existing native receipt fields, including episode-end censoring."""
    return dict(
        game=game,
        seed=7,
        finished_at="2026-10-06T12:00:00Z",
        solve_provenance="live_agent_self_discovery",
        trajectory_supervisor=dict(
            enabled=True,
            mode="applied",
            window=120,
            actions_observed=10,
            stagnations_unredirected=0,
            arm_outcomes={arm: dict(fired=1, helped=int(resolved))},
            redirects=[
                dict(
                    arm=arm,
                    action_index=1,
                    level=0,
                    resolved_by_levelup=resolved,
                    actions_to_levelup=3 if resolved else None,
                )
            ],
        ),
    )


def test_empty_and_default_authority(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8215-LOCATOR: empty is authenticated, default points at real producers."""
    path = fixture(tmp_path)
    value = reader.read_delta(path)
    assert not value["failures"] and value["rows"] == []
    assert value["new_outcome_count"] == 0
    locator = reader.discover()
    assert locator["ledger"]["path"].endswith("/ops/arc_supervisor_refinement_ledger.json")
    assert "arc_live_agent_state" not in json.dumps(locator)
    assert all(s["kind"] in {"eval", "harness"} for s in locator["sources"])
    assert locator["prior"]["sha256"] == reader.PINNED_PRIOR


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "hash",
        "foreign",
        "schema",
        "producer",
        "ledger_schema",
        "entries",
        "prior",
        "terminal",
        "receipt",
    ],
)
def test_authority_failures(tmp_path: Path, mutation: str) -> None:
    """REQ-VERIFY-8215: identify exact bad authority rather than inventing empty measurements."""
    path = fixture(tmp_path, [event()])
    value = json.loads(path.read_text())
    if mutation == "missing":
        Path(value["ledger"]["path"]).unlink()
    elif mutation == "hash":
        Path(value["sources"][0]["path"]).write_text("{}")
    elif mutation == "foreign":
        value["task_id"] = "exp8202-foreign"
    elif mutation == "schema":
        value["schema"] = "unknown"
    elif mutation == "producer":
        value["producer_code_hashes"][next(iter(value["producer_code_hashes"]))] = "sha256:bad"
    elif mutation in {"ledger_schema", "entries", "receipt"}:
        ledger = json.loads(Path(value["ledger"]["path"]).read_text())
        if mutation == "ledger_schema":
            ledger["schema"] = "foreign"
        elif mutation == "entries":
            ledger["entries"] = []
        else:
            next(iter(ledger["entries"].values()))["game"] = "foreign"
        value["ledger"] = seal(Path(value["ledger"]["path"]), ledger)
    elif mutation == "prior":
        prior = json.loads(Path(value["prior"]["path"]).read_text())
        prior["experiment_id"] = 8202
        value["prior"] = seal(Path(value["prior"]["path"]), prior)
    else:
        value["terminal"] = seal(Path(value["terminal"]["path"]), dict(report=dict(passed=False)))
    atomic_json(path, value)
    result = reader.read_delta(path)
    assert result["failures"] and result["new_outcome_count"] == 0
    assert all(
        {"path", "sha256", "artifact_field", "op", "expected", "observed"} <= r.keys()
        for r in result["failures"]
    )


def test_delta_censoring_resume_and_selection(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8215-SELECTION: duplicates, clocks and curated arms bound the proposal."""
    events = [event(), event(), event("r11l"), event("r11l", "allow_reinduction", False)]
    unknown = event("ar25")
    unknown.pop("finished_at")
    old = event("ft09")
    old["finished_at"] = "2026-10-05T12:00:00Z"
    foreign = event("wa30", "invented_arm")
    path = fixture(tmp_path, [*events, unknown, old, foreign])
    value = reader.read_delta(path)
    assert value["new_outcome_count"] == 3
    assert value["completed_count"] == 2 and value["censored_count"] == 1
    assert value["independent_count"] == 2
    assert value["per_arm_results"]["drop_goal_bias"]["helped"] == 2
    assert value["arm_selection_proposal"][0]["held_game"] == "cd82"
    assert value["arm_selection_proposal"][0]["selected_arm"] == "drop_goal_bias"
    assert {"duplicate_receipt", "chronology_absent", "before_frontier", "outcome_schema"} <= {
        r["reason"] for r in value["rows"]
    }
    locator = json.loads(path.read_text())
    inventory = dict(
        receipt_ids=value["current_frontier"]["receipt_ids"],
        event_ids=value["current_frontier"]["event_ids"],
    )
    locator["inventory"] = seal(tmp_path / "inventory.json", inventory)
    atomic_json(path, locator)
    assert reader.read_delta(path)["new_outcome_count"] == 0
    os.utime(Path(locator["sources"][0]["path"]), (1, 1))
    assert reader.read_delta(path)["new_outcome_count"] == 0


def invoke(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Exercise the real script from outside the checkout without ambient imports."""
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


def test_real_cli_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8215-CLI: candidates replay from primitives and reject rehashed drift."""
    locator = fixture(tmp_path, [event()])
    output = tmp_path / task.OUTPUT.name
    result = invoke(
        tmp_path,
        "--date",
        "20261006",
        "--locator",
        str(locator),
        "--output",
        str(output),
        "--fixture-e2e",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["new_outcome_count"] == 0 and value["fixture_result"]["new_outcome_count"] == 1
    assert value["MODEL_SPECS"] == [] and value["defaults_changed"] is False
    assert invoke(tmp_path, "--cold-replay", str(output)).returncode == 0
    value["new_outcome_count"] = 99
    atomic_json(output, value)
    assert invoke(tmp_path, "--cold-replay", str(output)).returncode == 1
    result = invoke(tmp_path, "--date", "20261005")
    assert result.returncode == 2
    result = invoke(tmp_path, "--fixture-e2e", "--output", str(task.OUTPUT))
    assert result.returncode == 2
    output = tmp_path / "blocked" / task.OUTPUT.name
    result = invoke(
        tmp_path,
        "--locator",
        str(tmp_path / "absent.json"),
        "--output",
        str(output),
        "--fixture-e2e",
    )
    assert result.returncode == 1 and json.loads(output.read_text())["verdict_class"] == "blocked"
    assert invoke(tmp_path, "--cold-replay", str(output)).returncode == 0


def test_frozen_checks_and_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8215: real command receipts and owned failures govern publication."""
    specs = task.commands(tmp_path)
    assert (
        next(s for s in specs if s["name"] == "full_python_suite")["classification"]
        == "repository_health"
    )
    assert "--files" in next(s for s in specs if s["name"] == "spec_coverage")["argv"]
    assert any(s["name"] == "e2e_017" for s in specs)
    locator = fixture(tmp_path)
    monkeypatch.setattr(task, "commands", lambda _: [])
    monkeypatch.setattr(task, "coverage_complete", lambda *a, **k: True)
    output = tmp_path / task.OUTPUT.name
    assert task.execute(locator, output, tmp_path / "work") == 0
    assert not task.replay(json.loads(output.read_text()))
    monkeypatch.setattr(task, "coverage_complete", lambda *a, **k: False)
    assert task.execute(locator, output, tmp_path / "work2") == 1
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    primitive = Path(json.loads(output.read_text())["primitive_path"])
    changed = json.loads(primitive.read_text())
    changed["new_outcome_count"] = 50
    atomic_json(primitive, changed)
    value = json.loads(output.read_text())
    value["raw_shard_hashes"][str(primitive)] = sha256_file(primitive)
    assert task.replay(value)


@pytest.mark.parametrize(
    "shape", ["malformed", "list", "oversized", "foreign_policy", "shadow", "harness"]
)
def test_native_schema_and_bounded_operands(tmp_path: Path, shape: str) -> None:
    """REQ-VERIFY-8215: malformed/oversized operands block; shadow evidence stays excluded."""
    path = fixture(tmp_path)
    loc = json.loads(path.read_text())
    source = Path(loc["sources"][0]["path"])
    if shape == "malformed":
        source.write_text("{")
    elif shape == "list":
        source.write_text("[]")
    elif shape == "oversized":
        with source.open("wb") as stream:
            stream.truncate(33554433)
    elif shape == "foreign_policy":
        seal(source, dict(experiment="arc_leaderboard_eval", policy="explorer", per_game=[]))
    elif shape == "shadow":
        row = event()
        row["trajectory_supervisor"]["mode"] = "shadow"
        seal(source, dict(experiment="arc_leaderboard_eval", policy="e3", per_game=[row]))
    else:
        row = event()
        row["arm"] = "S_llmon"
        seal(source, dict(rows=[row]))
        loc["sources"][0]["kind"] = "harness"
    loc["sources"][0]["sha256"] = sha256_file(source)
    atomic_json(path, loc)
    value = reader.read_delta(path)
    assert bool(value["failures"]) == (
        shape in {"malformed", "list", "oversized", "foreign_policy"}
    )
    assert value["new_outcome_count"] == int(shape == "harness")
    assert value["registry_precheck"]["ar25"] == 8


def test_replay_and_owned_receipt_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8215-COLD: real child logs and coverage are replayed, not inferred."""
    locator = fixture(tmp_path)
    private = tmp_path / "owned"
    private.mkdir()
    atomic_json(
        private / "coverage.json",
        dict(
            files={
                p: dict(summary=dict(num_statements=1, covered_lines=1), missing_lines=[])
                for p in task.OWNED
            }
        ),
    )
    monkeypatch.setattr(
        task,
        "commands",
        lambda _: [
            dict(
                name="owned_failure",
                argv=[
                    str(reader.ROOT / ".venv/bin/python"),
                    "-c",
                    'import sys; print("real failure"); sys.exit(3)',
                ],
                expected_exit=0,
                deadline_s=30,
                classification="required",
            )
        ],
    )
    output = tmp_path / task.OUTPUT.name
    assert task.execute(locator, output, private) == 1
    value = json.loads(output.read_text())
    assert (
        value["verdict_class"] == "disqualified"
        and value["validation_receipts"][0]["exit_code"] == 3
    )
    assert not task.replay(value)
    changed = deepcopy(value)
    changed["experiment_id"] = 0
    changed["reproducibility_checksum"] = "wrong"
    assert {"task_identity", "checksum"} <= set(task.replay(changed))
    changed = deepcopy(value)
    changed["validation_receipts"][0]["exit_code"] = 4
    assert "validation_receipt:owned_failure" in task.replay(changed)
    log = Path(value["validation_receipts"][0]["stderr_path"])
    log.write_text("changed")
    assert any(e.startswith("validation_stream:") for e in task.replay(value))
    primitive = Path(value["primitive_path"])
    primitive.unlink()
    assert "missing_primitive" in task.replay(value)
    monkeypatch.setattr(task, "terminal", lambda *a: dict(passed=False))
    with pytest.raises(ValueError, match="terminal_candidate_rejected"):
        task.execute(locator, tmp_path / "reject" / task.OUTPUT.name, private)


def test_discovery_missing_and_default_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8215: no invented default state and real default CLI qualification."""
    output = tmp_path / task.OUTPUT.name
    result = invoke(tmp_path, "--fixture-e2e", "--output", str(output))
    assert result.returncode in {0, 1}, result.stderr
    value = json.loads(output.read_text())
    assert value["new_outcome_count"] == 0
    assert invoke(tmp_path, "--cold-replay", str(output)).returncode == 0
    monkeypatch.setattr(reader, "ROOT", tmp_path)
    monkeypatch.setattr(reader, "producer_hashes", lambda: {})
    assert reader.discover()["ledger"]["sha256"] is None
    ledger = tmp_path.joinpath(*reader.DEFAULT_LEDGER_PARTS)
    ledger.parent.mkdir(exist_ok=True)
    ledger.write_text("{")
    assert reader.discover()["ledger"]["sha256"] == sha256_file(ledger)


def test_scan_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-8215: a bounded scan stops dependent reading on deadline expiry."""
    locator = fixture(tmp_path, [event()])
    ticks = iter([0.0, 121.0])
    monkeypatch.setattr(reader.time, "monotonic", lambda: next(ticks))
    value = reader.read_delta(locator)
    assert value["new_outcome_count"] == 0
    assert any(r["artifact_field"] == "scan_deadline_s" for r in value["failures"])


def test_branch_preconditions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8215: required inputs differ from optional siblings and scratch is writable."""
    value = task.preconditions(tmp_path)
    assert not value["failures"]
    assert any(
        r["artifact_field"] == "private_writable_storage" and r["passed"] for r in value["checks"]
    )
    monkeypatch.setattr(task, "ROOT", tmp_path)
    value = task.preconditions(tmp_path)
    assert value["failures"]
    assert all(r["required"] for r in value["failures"])
    assert any(
        not r["required"] and r["disposition"] == "absent_not_required" for r in value["checks"]
    )


def test_replay_execution_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-8215: rehashed primitive, terminal state and invocation claims cannot drift."""
    locator = fixture(tmp_path, [event()])
    monkeypatch.setattr(task, "commands", lambda _: [])
    monkeypatch.setattr(task, "coverage_complete", lambda *a, **k: True)
    output = tmp_path / task.OUTPUT.name
    assert task.execute(locator, output, tmp_path / "work") == 0
    value = json.loads(output.read_text())
    assert (
        value["new_outcome_count"] == 1
        and value["honest_verdict"] == "complete_null_descriptive_arm_selection"
    )
    changed = deepcopy(value)
    changed["verdict_class"] = "positive"
    changed["new_level_solves_claimed"] = 1
    assert {"terminal_state:verdict_class", "execution_claim:new_level_solves_claimed"} <= set(
        task.replay(changed)
    )
