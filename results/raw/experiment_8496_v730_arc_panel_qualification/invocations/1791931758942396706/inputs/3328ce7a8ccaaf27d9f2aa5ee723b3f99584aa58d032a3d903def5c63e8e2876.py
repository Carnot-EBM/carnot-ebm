"""REQ-REPORT-8468 / REQ-VERIFY-8468: fixtures qualify custody, never science."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.reporting import arc_generalization_panel_8468 as p
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.v709_execution import child
from test_arc_supervisor_live_panel_8384 import Arcade, work


def test_frozen_four_games(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8468-PANEL: shipped ordering precedes every observation."""
    source = p.evidence.producer(p.ROOT)
    roster = source["frozen_panel"]["roster"]
    panel = p.freeze(roster, [{"game": "tn36", "levels_reproduced": 7}])
    assert panel["selected_games"] == roster[:4]
    assert panel["descriptive_old_subset"] == roster[:2]
    assert len(panel["units"]) == 16
    assert all(u["max_actions"] == 512 and u["max_seconds"] == 120 for u in panel["units"])
    assert panel["registry_precheck"][roster[0]]["levels_reproduced"] == 7
    assert p.freeze(roster[::-1], [])["selected_games"] == roster[:4]
    assert p.freeze([], [])["missing_games"] == roster[:4]
    monkeypatch.setattr(
        p.evidence, "producer", lambda root: {"frozen_panel": {"roster": ["a", "b"]}}
    )
    with pytest.raises(ValueError, match="four_shipped_games"):
        p.freeze([], [])


def test_authority_resources_and_plan(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8468-BOUNDS: only real required operands gate CPU work."""
    checks, hashes, resources = p.preconditions(tmp_path, p.ROOT)
    assert checks == [] and resources["task_sha256"] == p.TASK_PIN
    assert resources["validation_reserve_s"] == 900
    assert resources["measurement_cap_s"] == 3600
    specs = p.plan(tmp_path / "plan")
    assert all(s["scope"] == "owned" for s in specs)
    assert any(s["name"] == "private_E2E011_017_wrapper" for s in specs)
    assert any(s["name"] == "private_E2E021" for s in specs)
    assert "--keep" in next(s["argv"] for s in specs if s["name"] == "coverage_combine")
    assert hashes[str(p.ROOT / p.evidence.CANDIDATE)] == p.evidence.CANDIDATE_PIN
    checks, _, resources = p.preconditions(tmp_path, tmp_path)
    assert checks and resources["task_sha256"] is None
    monkeypatch.setattr(
        p.evidence, "producer", lambda root: (_ for _ in ()).throw(ValueError("seal"))
    )
    checks, _, _ = p.preconditions(tmp_path, p.ROOT)
    assert any(c["check"] == "historical_producer" for c in checks)


def test_build_and_cold_controls(tmp_path):
    """SCENARIO-REPORT-8468-CUSTODY: self-rehashed headlines cannot manufacture observations."""
    w = work(tmp_path)
    w["panel"] = p.freeze([], [])
    value = p.build(w, [], tmp_path, tmp_path / (p.NAME + ".json"))
    assert value["intended_count"] == 16 and value["run_date"] == "20261013"
    assert value["verdict_class"] == "blocked" and value["arc_support_ready_score"] == 0
    assert value["new_level_solves"] == 0 and value["policy_model_calls"] == 0
    assert value["MODEL_SPECS"] == [] and value["independent_generalization_score"] == 0
    assert len(value["paired_progress_rows"]) == 8
    assert set(value) <= set(value["field_principles"])
    assert p.replay(value) and all(r["passed"] for r in p.controls(value, tmp_path))
    changed = deepcopy(value)
    changed["panel_execution_ready_score"] = 1
    assert not p.replay(changed) and not p.replay({})
    value["source_input_receipts"] = [{"snapshot_path": str(tmp_path / "absent"), "sha256": "bad"}]
    assert not p.replay(value)


def test_constructed_collection_and_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8468-PANEL: actual wrapper and paired seeds drive fixture transport."""
    panel = p.freeze([], [])
    arcade = Arcade()
    arcade.available_environments = [SimpleNamespace(game_id=g) for g in panel["roster"]]
    monkeypatch.setattr(p, "sdk", lambda: arcade)
    p.plan(tmp_path / "private")
    monkeypatch.setattr(p, "plan", lambda private: [])
    monkeypatch.setattr(p, "execute", lambda specs, logs: [dict(passed=True, scope="owned")])

    def episode_child(name, argv, logs, **kwargs):
        unit = json.loads(Path(argv[argv.index("--episode") + 1]).read_bytes())
        raw = Path(argv[argv.index("--raw") + 1])
        short = dict(unit, max_actions=4)
        p.runtime.episode(short, arcade, raw)
        return dict(passed=True, scope="measurement", name=name)

    monkeypatch.setattr(p, "child", episode_child)
    output = tmp_path / (p.NAME + ".json")
    p.plan(tmp_path / "private")
    assert p.run(output, tmp_path / "private") == 0
    value = json.loads(output.read_bytes())
    assert value["completed_count"] == 16 and value["panel_execution_ready_score"] == 1
    assert value["independent_count"] == 4 and value["arc_support_ready_score"] == 0
    assert len(value["paired_progress_rows"]) == 8
    assert all(row["peak_level_delta_on_minus_off"] == 0 for row in value["paired_progress_rows"])
    assert p.replay(value)
    read = reader_receipt(p.TASK, tmp_path, field="panel_execution_ready_score", expected=1)
    assert read["passed"] and read["gate_sha256"] == sha256_file(output)
    changed = deepcopy(value)
    changed["paired_progress_rows"][0]["peak_level_delta_on_minus_off"] = 1
    assert not p.replay(changed)


@pytest.mark.parametrize("mode", ["missing", "sdk", "deadline", "validation", "fixture"])
def test_orchestration_failures(tmp_path, monkeypatch, mode):
    """REQ-VERIFY-8468: absence blocks; owned failure disqualifies without invented rows."""
    panel = p.freeze([], [])
    arcade = Arcade()
    arcade.available_environments = [SimpleNamespace(game_id=g) for g in panel["roster"]]
    monkeypatch.setattr(p, "sdk", lambda: arcade)
    monkeypatch.setattr(p, "plan", lambda private: [])
    monkeypatch.setattr(
        p, "execute", lambda specs, logs: [dict(passed=mode != "validation", scope="owned")]
    )
    monkeypatch.setattr(p, "controls", lambda value, raw: [])
    monkeypatch.setattr(p, "publish", lambda value, work, output, raw: atomic_json(output, value))
    if mode == "missing":
        arcade.available_environments = []
    if mode == "sdk":
        monkeypatch.setattr(p, "sdk", lambda: (_ for _ in ()).throw(ValueError("SDK absent")))
    if mode == "deadline":
        ticks = iter([0, 0, 3601, 3602, 3603, 3604, 3605])
        monkeypatch.setattr(p.time, "monotonic", lambda: next(ticks, 3606))
    output = tmp_path / (p.NAME + ".json")
    assert p.run(output, tmp_path / "private", private_e2e=mode == "fixture") == 0
    value = json.loads(output.read_bytes())
    assert value["completed_count"] == 0 and value["panel_execution_ready_score"] == 0
    assert value["verdict_class"] == ("disqualified" if mode == "validation" else "blocked")


def test_real_cli_paths(tmp_path):
    """SCENARIO-REPORT-8468-CUSTODY: private CLI errors and publication execute in fresh processes."""
    cli = [str(p.ROOT / ".venv/bin/python"), "-u", str(p.ROOT / p.CLI)]
    output = tmp_path / (p.NAME + ".json")
    for name, args, expected in [
        ("private", ["--private-e2e", "--output", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("valid", ["--cold-replay", str(output)], 0),
        (
            "episode_absent",
            ["--episode", str(tmp_path / "absent"), "--raw", str(tmp_path / "ep")],
            1,
        ),
        ("episode_no_raw", ["--episode", str(output)], 2),
        ("bad_date", ["--date", "20261012"], 2),
        ("public_fixture", ["--private-e2e"], 2),
    ]:
        assert child(name, [*cli, *args], tmp_path / "cli", expected=expected, deadline=120)[
            "passed"
        ]
    unit = tmp_path / "unit.json"
    atomic_json(
        unit,
        dict(
            episode_id="absent:11:off",
            game="absent-sdk-game",
            seed=11,
            arm="off",
            max_actions=1,
            max_seconds=1,
        ),
    )
    assert child(
        "missing_game",
        [*cli, "--episode", str(unit), "--raw", str(tmp_path / "ep")],
        tmp_path / "cli",
        expected=1,
        deadline=60,
    )["passed"]


def test_missing_history_keeps_sixteen_units(tmp_path, monkeypatch):
    """REQ-REPORT-8468: absent original evidence blocks rather than crashing or shrinking n."""
    monkeypatch.setattr(
        p.evidence, "producer", lambda root: (_ for _ in ()).throw(ValueError("missing producer"))
    )
    monkeypatch.setattr(p, "plan", lambda private: [])
    monkeypatch.setattr(p, "execute", lambda specs, logs: [])
    monkeypatch.setattr(p, "controls", lambda value, raw: [])
    monkeypatch.setattr(p, "publish", lambda value, work, output, raw: atomic_json(output, value))
    output = tmp_path / (p.NAME + ".json")
    assert p.run(output, tmp_path / "private") == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and len(value["rows"]) == 16
    assert value["independent_count"] == 0


def test_independent_replay_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8468-CUSTODY: independent seals reject internally consistent replacements."""
    w = work(tmp_path)
    operand = tmp_path / "wrong-snapshot"
    operand.write_bytes(b"changed input")
    w["source_input_receipts"] = [{"snapshot_path": str(operand), "sha256": "sha256:wrong"}]
    value = p.build(w, [], tmp_path, tmp_path / (p.NAME + ".json"))
    assert not p.replay(value)
    w["source_input_receipts"] = []
    w["preconditions_checked"] = {"task_sha256": p.TASK_PIN}
    value = p.build(w, [], tmp_path, tmp_path / (p.NAME + ".json"))
    with monkeypatch.context() as m:
        m.setattr(
            p.authority, "authority", lambda *args: dict(activated=False, tasks=[dict(id=p.TASK)])
        )
        assert not p.replay(value)
    registry = []
    w["panel"] = p.freeze([], registry)
    value = p.build(w, [], tmp_path, tmp_path / (p.NAME + ".json"))
    with monkeypatch.context() as m:
        m.setattr(p, "freeze", lambda *args: {})
        assert not p.replay(value)


def test_failed_owned_episode(tmp_path, monkeypatch):
    """REQ-VERIFY-8468: killed owned children disqualify rather than becoming a scientific null."""
    panel = p.freeze([], [])
    arcade = Arcade()
    arcade.available_environments = [SimpleNamespace(game_id=g) for g in panel["roster"]]
    monkeypatch.setattr(p, "sdk", lambda: arcade)
    monkeypatch.setattr(p, "plan", lambda private: [])
    monkeypatch.setattr(p, "execute", lambda specs, logs: [dict(passed=True, scope="owned")])
    monkeypatch.setattr(p, "child", lambda *args, **kwargs: dict(passed=False, scope="measurement"))
    monkeypatch.setattr(p, "controls", lambda value, raw: [])
    monkeypatch.setattr(p, "publish", lambda value, work, output, raw: atomic_json(output, value))
    output = tmp_path / (p.NAME + ".json")
    assert p.run(output, tmp_path / "private") == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["failed_count"] == 0
    assert value["censored_count"] == 16 and not value["required_checks_passed"]


def test_real_successful_episode(tmp_path):
    """SCENARIO-VERIFY-8468-BOUNDS: one real SDK reset qualifies the successful thin child route."""
    game = p.freeze([], [])["selected_games"][0]
    unit = tmp_path / "unit.json"
    atomic_json(
        unit,
        dict(
            episode_id=f"{game}:11:off",
            game=game,
            seed=11,
            arm="off",
            max_actions=1,
            max_seconds=20,
        ),
    )
    receipt = child(
        "one_actual_reset",
        [
            str(p.ROOT / ".venv/bin/python"),
            "-u",
            str(p.ROOT / p.CLI),
            "--episode",
            str(unit),
            "--raw",
            str(tmp_path / "ep"),
        ],
        tmp_path / "logs",
        deadline=60,
    )
    assert receipt["passed"]
    terminal = json.loads((tmp_path / "ep/episode.json").read_bytes())
    assert terminal["llm_invocation_count"] == 0 and terminal["error"] is None


def test_missing_authority_task_is_rejected(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8468-CUSTODY: absent task identity closes replay without an exception leak."""
    w = work(tmp_path)
    w["preconditions_checked"] = {"task_sha256": p.TASK_PIN}
    value = p.build(w, [], tmp_path, tmp_path / (p.NAME + ".json"))
    monkeypatch.setattr(p.authority, "authority", lambda *args: dict(activated=True, tasks=[]))
    assert not p.replay(value)
