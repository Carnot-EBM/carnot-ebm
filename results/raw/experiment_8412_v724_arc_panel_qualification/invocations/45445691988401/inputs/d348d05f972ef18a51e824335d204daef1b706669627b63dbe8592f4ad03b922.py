"""REQ-REPORT-8412 / REQ-VERIFY-8412: controls never supply research observations."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.reporting import arc_panel_qualification_8412 as p
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.v709_execution import child
from test_arc_supervisor_live_panel_8384 import Arcade, work


def test_scratch_parent_and_unrelated_current_authority(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8412-SCRATCH: frozen reads survive nonexistent private parents."""
    private = tmp_path / "nested" / "absent"
    calls = []

    def pytest_main(args):
        assert private.is_dir() and (private / "writes").is_dir()
        calls.append(args)
        assert (p.ROOT / p.current.ACTIVE).read_bytes() != b"unrelated-current-roadmap"
        return 0

    monkeypatch.setattr(pytest, "main", pytest_main)
    original = Path.open

    def unrelated(path, *args, **kwargs):
        if path == p.ROOT / p.current.ACTIVE:
            from io import BytesIO

            return BytesIO(b"unrelated-current-roadmap")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unrelated)
    assert p.historical_checks(private) == 0
    assert p.base.TEST in calls[0]


def test_preconditions_and_frozen_manifest(tmp_path):
    """REQ-VERIFY-8412: authenticate current complete task independently of history."""
    checks, hashes, resources = p.preconditions(tmp_path, p.ROOT)
    assert checks == [] and resources["task_sha256"] == p.TASK_PIN
    assert hashes[str(p.ROOT / p.PRIOR)] == p.PRIOR_PIN
    specs = p.plan(tmp_path / "new" / "parent")
    assert all(s["scope"] == "owned" for s in specs)
    assert any(s["name"] == "historical_wrapper_assertions" for s in specs)
    assert any(s["name"] == "coverage_report" for s in specs)
    assert all("tests/python'" not in s["argv"] for s in specs)
    checks, _, resources = p.preconditions(tmp_path, tmp_path)
    assert checks and resources["task_sha256"] is None


def test_hash_failure_and_bindings(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8412-REPLAY: historical substitution closes collection."""
    original = p.old.NAME
    with p.bindings():
        assert p.old.NAME == p.NAME and p.base.CLI == p.CLI
    assert p.old.NAME == original
    monkeypatch.setattr(p, "PRIOR_PIN", "sha256:invalid")
    checks, _, resources = p.preconditions(tmp_path, p.ROOT)
    assert any(c["check"] == "historical_receipt_hash" for c in checks)
    assert resources["task_sha256"] is None


def test_build_replay_and_cold_controls(tmp_path):
    """SCENARIO-REPORT-8412-REPLAY: valid and self-rehashed controls run in fresh children."""
    value = p.build(work(tmp_path), [], tmp_path, tmp_path / (p.NAME + ".json"))
    assert value["run_date"] == "20261011" and value["experiment_id"] == 8412
    assert value["milestone"] == "2026.10.724" and value["verdict_class"] == "blocked"
    assert value["arm_refinement_authorized"] is False and value["overlapping_game_count"] == 0
    assert value["firings_per_arm_cell"] == {}
    assert value["headline_solve_credit"] == value["generalized_learning_benefit_score"] == 0
    assert value["MODEL_SPECS"] == [] and value["model_invocation_counts"]["loads"] == 0
    assert set(value) <= set(value["field_principles"])
    assert p.replay(value) and all(r["passed"] for r in p.controls(value, tmp_path))
    changed = deepcopy(value)
    changed["arm_refinement_authorized"] = True
    assert not p.replay(changed) and not p.replay({})


def test_constructed_live_orchestration(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8412-PANEL: only real wrapper calls reduce constructed transport."""
    panel = p.old.freeze([], [])
    arcade = Arcade()
    arcade.available_environments = [SimpleNamespace(game_id=g) for g in panel["selected_games"]]
    monkeypatch.setattr(p.old, "sdk", lambda: arcade)
    monkeypatch.setattr(p, "plan", lambda private: [])
    monkeypatch.setattr(p.old, "execute", lambda specs, logs: [dict(passed=True, scope="owned")])

    def episode_child(name, argv, logs, **kwargs):
        unit = json.loads(Path(argv[argv.index("--episode") + 1]).read_bytes())
        raw = Path(argv[argv.index("--raw") + 1])
        p.old.runtime.episode(dict(unit, max_actions=4), arcade, raw)
        return dict(passed=True, scope="measurement", name=name)

    monkeypatch.setattr(p.old, "child", episode_child)
    output = tmp_path / (p.NAME + ".json")
    assert p.run(output, tmp_path / "private") == 0
    value = json.loads(output.read_bytes())
    assert value["completed_count"] == 8 and value["arc_panel_ready_score"] == 1
    assert value["independent_count"] == 2 and value["verdict_class"] == "null"
    assert all(r["action_count"] == 4 for r in value["rows"])
    assert p.replay(value)


def test_real_cli_consumers_and_failure_paths(tmp_path):
    """SCENARIO-VERIFY-8412-CONTROLS: real subprocess and reader paths retain blocked bytes."""
    output = tmp_path / (p.NAME + ".json")
    cli = [str(p.ROOT / ".venv/bin/python"), "-u", str(p.ROOT / p.CLI)]
    receipt = child(
        "private_cli",
        [*cli, "--private-e2e", "--output", str(output)],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"]
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and p.replay(value)
    read = reader_receipt(p.TASK, tmp_path, field="arc_panel_ready_score", expected=0)
    assert read["passed"] and read["gate_sha256"] == sha256_file(output)
    unit = tmp_path / "unit.json"
    atomic_json(
        unit,
        dict(
            episode_id="missing:11:off",
            game="absent-sdk-game",
            seed=11,
            arm="off",
            max_actions=256,
            max_seconds=180,
        ),
    )
    for name, args, expected in [
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        (
            "episode_absent",
            ["--episode", str(tmp_path / "absent"), "--raw", str(tmp_path / "episode")],
            1,
        ),
        ("episode", ["--episode", str(unit), "--raw", str(tmp_path / "episode")], 1),
        ("episode_no_raw", ["--episode", str(unit)], 2),
        ("unsafe_private", ["--private-e2e"], 2),
        ("historic_date", ["--date", "20261010"], 2),
    ]:
        assert child(name, [*cli, *args], tmp_path / "logs", expected=expected, deadline=60)[
            "passed"
        ]


def test_measured_firing_cells_remain_unauthorized(tmp_path):
    """REQ-REPORT-8412: firing support does not authorize a two-game refinement."""
    unit = dict(
        episode_id="fixture:11:on",
        game="fixture",
        seed=11,
        arm="on",
        max_actions=125,
        max_seconds=180,
    )
    episode_raw = tmp_path / "episode"
    terminal = p.runtime.episode(unit, Arcade(), episode_raw)
    row = p.base.reduce_episode(unit, terminal, episode_raw)
    assert row["supervisor_outcome_rows"]
    w = work(tmp_path)
    w.update(rows=[row], panel=dict(units=[unit], registry_precheck={}))
    value = p.build(w, [], tmp_path, tmp_path / (p.NAME + ".json"))
    assert value["firings_per_arm_cell"] and value["overlapping_game_count"] == 1
    assert value["arm_refinement_authorized"] is False
