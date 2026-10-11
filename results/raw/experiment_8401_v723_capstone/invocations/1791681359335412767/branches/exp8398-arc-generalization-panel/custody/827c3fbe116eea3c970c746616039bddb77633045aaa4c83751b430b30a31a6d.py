"""REQ-REPORT-8384 and REQ-VERIFY-8384: fixtures test mechanics, never research rows."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.reporting import arc_supervisor_live_panel_8384 as p
from carnot.agentic import arc_live_panel_runtime_8384 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child


def test_schedule() -> None:
    """SCENARIO-REPORT-8384-LIVE: selection precedes outcomes and preserves exposed credit."""
    games = ["aa-v1", "bb-v1", "cc-v1"]
    panel = p.freeze(games, [{"game": "aa", "levels_reproduced": 4}])
    assert len(panel["units"]) == 8
    assert [u["seed"] for u in panel["units"]] == [11, 11, 22, 22] * 2
    assert [u["arm"] for u in panel["units"]] == ["off", "on"] * 4
    assert panel["registry_precheck"]["aa-v1"]["levels_reproduced"] == 4
    assert p.freeze(list(reversed(games)), [])["selected_games"] == panel["selected_games"]
    with pytest.raises(ValueError, match="two_sdk_games"):
        p.freeze([], [])


def test_tripwire() -> None:
    """SCENARIO-VERIFY-8384-TRIPWIRE: every dangerous seam rejects before model construction."""
    import builtins
    from carnot.agentic import arc_competition_agent as agent
    from carnot.agentic import arc_executable_world_model as world

    original = builtins.__import__
    with r.no_models() as attempts:
        for module in r.FORBIDDEN:
            with pytest.raises(RuntimeError, match="model_tripwire"):
                builtins.__import__(module)
        with pytest.raises(RuntimeError, match="model_tripwire"):
            world.LocalGGUFProposer()
        with pytest.raises(RuntimeError, match="model_tripwire"):
            agent.E3AgentPolicy._proposer(None)
        assert builtins.__import__("json") is json
    assert builtins.__import__ is original
    assert len(attempts) == len(r.FORBIDDEN) + 2


class Frame:
    """A constructed visible frame can qualify wrapper mechanics only."""

    def __init__(self, level: int = 0) -> None:
        self.frame = [[[0] * 8 for _ in range(8)]]
        self.levels_completed = level
        self.available_actions = [1, 2, 3, 4, 5, 6]
        self.state = "NOT_FINISHED"


class Arcade:
    def __init__(self, fail: bool = False, null: bool = False) -> None:
        self.fail, self.null = fail, null
        self.seeds: list[int] = []

    def open_scorecard(self) -> str:
        return "constructed-control"

    def make(self, game_id: str, *, seed: int, scorecard_id: str) -> "Arcade":
        self.seeds.append(seed)
        return self

    def reset(self) -> Frame:
        return Frame()

    def step(self, action: object, *, data: object = None) -> Frame | None:
        if self.fail:
            raise RuntimeError("constructed SDK error")
        return None if self.null else Frame()


def unit() -> dict:
    return dict(
        episode_id="aa:11:off", game="aa", seed=11, arm="off", max_actions=4, max_seconds=180
    )


def test_wrapper_and_receipt(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8384-LIVE: real production policy uses constructed transport only here."""
    arcade = Arcade()
    result = r.episode(unit(), arcade, tmp_path)
    assert arcade.seeds == [11]
    assert result["model_tripwire_passed"]
    assert result["actual_wrapper_path"]["policy"] == "E3AgentPolicy"
    row = p.reduce_episode(unit(), result, tmp_path)
    assert row["action_count"] == 4 and row["peak_level"] == 0
    assert row["headline_solve_credit"] == 0
    assert row["censoring"] == "action_limit"
    assert row["supervisor_receipt"]["mode"] == "shadow"
    stream = tmp_path / "steps.jsonl"
    original = stream.read_text()
    events = [json.loads(line) for line in original.splitlines()]
    events[0]["action_sha256"] = "invalid"
    stream.write_text("\n".join(json.dumps(event) for event in events))
    with pytest.raises(ValueError, match="action_hash"):
        p.reduce_episode(unit(), result, tmp_path)
    stream.write_text(original)
    changed = deepcopy(result)
    changed["supervisor_receipt"]["actions_observed"] += 1
    with pytest.raises(ValueError, match="supervisor_receipt"):
        p.reduce_episode(unit(), changed, tmp_path)


@pytest.mark.parametrize("fail,null", [(True, False), (False, True)])
def test_episode_errors(tmp_path: Path, fail: bool, null: bool) -> None:
    """SCENARIO-VERIFY-8384-CLI: SDK failures retain the already observed prefix."""
    result = r.episode(unit(), Arcade(fail, null), tmp_path)
    assert result["error"] and not result["model_tripwire_failed"]
    row = p.reduce_episode(unit(), result, tmp_path)
    assert row["action_count"] == 1 and row["status"] == "failed"


def test_preconditions_and_plan(tmp_path: Path) -> None:
    """REQ-VERIFY-8384: commands freeze private scratch, owned coverage and global health."""
    plan = p.plan(tmp_path)
    assert any(s["scope"] == "global" for s in plan)
    assert any(s["name"] == "coverage_report" for s in plan)
    checks, hashes, authority = p.preconditions(tmp_path, p.ROOT)
    assert hashes and authority["task_sha256"] == p.TASK_PIN
    assert any(c["check"] == "independent_design_contract_available" for c in checks)
    checks, _, _ = p.preconditions(tmp_path, tmp_path)
    assert any(c["observed"] is None for c in checks)


def work(tmp_path: Path) -> dict:
    """A sealed empty ledger exercises reporting without supplying research observations."""
    value = dict(
        panel=None,
        rows=[],
        failures=[
            dict(
                check="sdk_access",
                upstream="arc_agi",
                path="absent",
                hash=None,
                field="catalogue",
                operator="==",
                expected=True,
                observed=None,
                passed=False,
            )
        ],
        source_artifact_hashes={},
        raw_shard_hashes={},
        code_config_hashes={},
        preconditions_checked={},
        historical_model_provenance=[],
        phase_spans=[],
        duration_s=0.1,
        run_date="20261010",
    )
    atomic_json(tmp_path / "measurement.json", value)
    return value


def test_build_replay_and_controls(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8384-REPLAY: missing observations stay absent and cold controls are real."""
    w = work(tmp_path)
    value = p.build(w, [], tmp_path, tmp_path / (p.NAME + ".json"))
    assert value["verdict_class"] == "blocked" and value["intended_count"] == 8
    assert value["completed_count"] == 0 and value["arc_panel_ready_score"] == 0
    assert p.replay(value)
    results = p.controls(value, tmp_path)
    assert len(results) == 4 and all(row["passed"] for row in results)
    changed = deepcopy(value)
    changed["completed_count"] = 8
    assert not p.replay(changed)
    value["work_reference"]["sha256"] = "invalid"
    assert not p.replay(value)


def test_live_aggregation_and_tamper(tmp_path: Path) -> None:
    """REQ-REPORT-8384: pending events and census costs reduce from sealed primitive rows."""
    u = unit()
    u.update(arm="on", max_actions=125)
    result = r.episode(u, Arcade(), tmp_path / "episode")
    row = p.reduce_episode(u, result, tmp_path / "episode")
    assert row["supervisor_receipt"]["mode"] == "applied"
    assert row["pending_outcome_count"] == 0
    w = work(tmp_path)
    w.update(panel=dict(units=[u], registry_precheck={}), rows=[row], failures=[])
    atomic_json(tmp_path / "measurement.json", w)
    receipts = [dict(scope="owned", passed=True, name="test")]
    value = p.build(w, receipts, tmp_path, tmp_path / (p.NAME + ".json"))
    assert value["independent_count"] == 1 and value["headline_solve_credit"] == 0
    assert p.replay(value)
    assert all(receipt["passed"] for receipt in p.controls(value, tmp_path))
    w["rows"][0]["action_count"] += 1
    atomic_json(tmp_path / "measurement.json", w)
    value["work_reference"]["sha256"] = sha256_file(tmp_path / "measurement.json")
    value["reproducibility_checksum"] = canonical_hash(w)
    assert not p.replay(value)
    assert (
        p.build(w, [dict(scope="owned", passed=False)], tmp_path, tmp_path / (p.NAME + ".json"))[
            "verdict_class"
        ]
        == "disqualified"
    )


def test_cli(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8384-CLI: real script handles private empty evidence and argument errors."""
    cli = str(p.ROOT / p.CLI)
    for name, args, expected in [
        ("private", ["--private-e2e", "--output", str(tmp_path / (p.NAME + ".json"))], 0),
        ("bad_date", ["--date", "19990101"], 2),
        ("unsafe_private", ["--private-e2e"], 2),
        ("absent_replay", ["--cold-replay", str(tmp_path / "absent.json")], 1),
        ("bad_episode", ["--episode", str(tmp_path / "absent.json"), "--raw", str(tmp_path)], 1),
    ]:
        assert child(
            name,
            [p.sys.executable, "-u", cli, *args],
            tmp_path / "logs",
            expected=expected,
            deadline=60,
        )["passed"]


def test_run_private_and_sdk_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-8384: external SDK absence remains blocked and does not invent a panel."""
    monkeypatch.setattr(p, "plan", lambda private: [])
    monkeypatch.setattr(p, "preconditions", lambda private, root: ([], {}, {}))
    monkeypatch.setattr(p, "publish", lambda *args: None)
    monkeypatch.setattr(p, "controls", lambda *args: [])
    monkeypatch.setattr(p, "sdk", lambda: (_ for _ in ()).throw(ImportError("missing SDK")))
    output = tmp_path / (p.NAME + ".json")
    assert p.run(output, tmp_path / "private") == 0
    assert p.run(output, tmp_path / "private", private_e2e=True) == 0


def test_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8384-CLI: failed audit is retained in a checked disqualified terminal."""
    w = work(tmp_path)
    value = p.build(w, [], tmp_path, tmp_path / (p.NAME + ".json"))
    monkeypatch.setattr(p, "child", lambda *args, **kwargs: dict(passed=True, scope="owned"))
    monkeypatch.setattr(
        p, "audit", lambda *args: dict(passed=True, receipt=dict(passed=True), findings=[])
    )
    p.publish(value, w, tmp_path / (p.NAME + ".json"), tmp_path)
    monkeypatch.setattr(
        p,
        "audit",
        lambda *args: dict(
            passed=False,
            receipt=dict(passed=False, scope="owned"),
            findings=[dict(severity="warning")],
        ),
    )
    p.publish(value, w, tmp_path / (p.NAME + ".json"), tmp_path)
    terminal = json.loads((tmp_path / (p.NAME + ".json")).read_text())
    assert terminal["verdict_class"] == "disqualified" and terminal["adversarial_findings"]
    monkeypatch.setattr(
        p, "publish_primary", lambda *args: (_ for _ in ()).throw(ValueError("other"))
    )
    with pytest.raises(ValueError, match="other"):
        p.publish(value, w, tmp_path / (p.NAME + ".json"), tmp_path)


def test_supervisor_primitive_controls(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8384-REPLAY: constructed stagnation verifies receipt arithmetic only."""
    from dataclasses import asdict
    from carnot.agentic.arc_trajectory_supervisor import TrajectorySupervisor, TrajectorySnapshot

    supervisor = TrajectorySupervisor(window=120)
    snapshot = TrajectorySnapshot(0, False, False, 0, 0, False)
    rows = []
    for index in range(125):
        redirect = supervisor.observe(snapshot)
        action = dict(name="ACTION1", data={})
        observation = dict(pixels=[[[0]]], level=0, available_actions=[1])
        rows.append(
            dict(
                index=index + 1,
                action=action,
                action_sha256=canonical_hash(action),
                observation=observation,
                observation_sha256=canonical_hash(observation),
                progress=False,
                choose_action_s=0.001,
                sdk_transition_s=0.001,
                supervisor_events=[
                    dict(snapshot=asdict(snapshot), redirect=asdict(redirect) if redirect else None)
                ],
            )
        )
    stream = tmp_path / "steps.jsonl"
    stream.write_text("\n".join(json.dumps(row) for row in rows))
    terminal = dict(
        supervisor_receipt={},
        pending_supervisor_events=[],
        error=None,
        censoring="action_limit",
        model_tripwire_passed=True,
        actual_wrapper_path={},
    )
    reduced = p.reduce_episode(unit(), terminal, tmp_path)
    assert reduced["redirect_count"] == 1 and reduced["pending_outcome_count"] == 1
    for field, replacement, match in [
        ("observation_sha256", "invalid", "observation_hash"),
        ("progress", True, "progress_drift"),
        (
            "supervisor_events",
            [dict(snapshot=asdict(snapshot) | dict(level=1), redirect=None)],
            "supervisor_level",
        ),
        ("supervisor_events", [dict(snapshot=asdict(snapshot), redirect={})], "redirect_drift"),
    ]:
        changed = deepcopy(rows)
        changed[0][field] = replacement
        stream.write_text("\n".join(json.dumps(row) for row in changed))
        with pytest.raises(ValueError, match=match):
            p.reduce_episode(unit(), terminal, tmp_path)
    stream.write_text("\n".join(json.dumps(row) for row in rows))
    terminal["pending_supervisor_events"] = [dict(snapshot=asdict(snapshot), redirect=None)]
    assert (
        p.reduce_episode(unit(), terminal, tmp_path)["supervisor_receipt"]["actions_observed"]
        == 126
    )
    terminal["pending_supervisor_events"][0]["redirect"] = {}
    with pytest.raises(ValueError, match="pending_redirect_drift"):
        p.reduce_episode(unit(), terminal, tmp_path)
    assert p.reduce_episode(unit(), None, tmp_path)["status"] == "censored"


def test_runtime_time_and_policy_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8384-TRIPWIRE: bounded and failed mechanics never become observations."""
    u = unit() | dict(max_seconds=0)
    assert r.episode(u, Arcade(), tmp_path / "time")["censoring"] == "time_limit"
    monkeypatch.setattr(
        r.agent_module,
        "make_carnot_agent",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("broken constructor")),
    )
    assert r.episode(unit(), Arcade(), tmp_path / "policy")["error"]


def test_live_run_orchestration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-8384: mocked children qualify the eight-unit orchestration, with no publication."""
    from carnot.reporting.current_work_receipt import atomic_json

    arcade = SimpleNamespace(
        available_environments=[SimpleNamespace(game_id="aa"), SimpleNamespace(game_id="bb")]
    )
    monkeypatch.setattr(p, "sdk", lambda: arcade)
    monkeypatch.setattr(p, "plan", lambda private: [dict(name="owned", scope="owned")])
    monkeypatch.setattr(
        p,
        "execute",
        lambda specs, logs: [dict(passed=True, scope=s["scope"], name=s["name"]) for s in specs],
    )
    monkeypatch.setattr(
        p, "preconditions", lambda private, root: ([], {}, dict(task_sha256=p.TASK_PIN))
    )
    monkeypatch.setattr(p, "publish", lambda *args: None)
    monkeypatch.setattr(p, "controls", lambda *args: [])

    def measured_child(name: str, argv: list[str], logs: Path, **kwargs: object) -> dict:
        path = Path(argv[-1])
        path.mkdir(parents=True)
        # A missing terminal tests censoring, not an invented research observation.
        atomic_json(path / "boundary.json", dict(child="censored_control"))
        return dict(passed=True, scope="measurement", name=name)

    monkeypatch.setattr(p, "child", measured_child)
    assert p.run(tmp_path / (p.NAME + ".json"), tmp_path / "private") == 0
    assert len(list(tmp_path.rglob("boundary.json"))) == 8


def test_sdk_metadata_and_episode_cli(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8384-CLI: real SDK access and child error paths are independently exercised."""
    assert len(p.sdk().available_environments) >= 2
    operand = tmp_path / "unknown_game.json"
    atomic_json(operand, unit() | dict(game="missing-sdk-game"))
    for name, args, expected in [
        ("actual_sdk_error", ["--episode", str(operand), "--raw", str(tmp_path / "episode")], 1),
        ("missing_raw", ["--episode", str(operand)], 2),
    ]:
        receipt = child(
            name,
            [p.sys.executable, "-u", str(p.ROOT / p.CLI), *args],
            tmp_path / "logs",
            expected=expected,
            deadline=60,
        )
        assert receipt["passed"]


def test_guard_importlib_and_fromlist() -> None:
    """SCENARIO-VERIFY-8384-TRIPWIRE: importlib and fromlist cannot bypass a before-load guard."""
    import builtins
    import importlib

    with r.no_models():
        with pytest.raises(RuntimeError, match="model_tripwire"):
            importlib.import_module("llama_cpp")
        with pytest.raises(RuntimeError, match="model_tripwire"):
            builtins.__import__("carnot.agentic", fromlist=["arc_game_adapters"])
        assert importlib.import_module("json") is json


def test_swallowed_tripwire_and_induction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8384-TRIPWIRE: swallowed load attempts and actual induction both fail."""
    import builtins
    from carnot.agentic import arc_competition_agent as agent
    from contextlib import contextmanager, suppress
    import os

    @contextmanager
    def enabled_induction():
        monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "0")
        yield

    with monkeypatch.context() as local:
        local.setattr(r, "withheld_policy_inputs", enabled_induction)
        assert (
            "induction_not_disabled" in r.episode(unit(), Arcade(), tmp_path / "enabled")["error"]
        )
    original = agent.make_carnot_agent

    def factory(base: type, **kwargs: object) -> type:
        cls = original(base, **kwargs)
        choosing = cls.choose_action

        def choose(self: object, frames: list, latest: object) -> object:
            action = choosing(self, frames, latest)
            with suppress(RuntimeError):
                builtins.__import__("llama_cpp")
            return action

        cls.choose_action = choose
        return cls

    with monkeypatch.context() as local:
        local.setattr(agent, "make_carnot_agent", factory)
        assert (
            "model_tripwire_observed"
            in r.episode(unit(), Arcade(), tmp_path / "swallowed")["error"]
        )

    def induced(base: type, **kwargs: object) -> type:
        cls = original(base, **kwargs)
        choosing = cls.choose_action

        def choose(self: object, frames: list, latest: object) -> object:
            action = choosing(self, frames, latest)
            self._policy.induction_attempts.append(dict(skipped="actual_induction"))
            return action

        cls.choose_action = choose
        return cls

    with monkeypatch.context() as local:
        local.setattr(agent, "make_carnot_agent", induced)
        assert "induction_executed" in r.episode(unit(), Arcade(), tmp_path / "induced")["error"]
    assert os.environ.get("CARNOT_ARC_DISABLE_INDUCTION") == "0"


def test_replay_primitive_and_log_hashes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8384-REPLAY: checksums and child-log hashes remain part of cold custody."""
    w = work(tmp_path)
    primitive = tmp_path / "sealed"
    primitive.write_text("sealed bytes")
    w["raw_shard_hashes"][str(primitive)] = sha256_file(primitive)
    value = p.build(w, [], tmp_path, tmp_path / (p.NAME + ".json"))
    assert p.replay(value)
    value["reproducibility_checksum"] = "invalid"
    assert not p.replay(value)
    value["reproducibility_checksum"] = canonical_hash(w)
    primitive.write_text("changed bytes")
    assert not p.replay(value)
    primitive.write_text("sealed bytes")
    log = tmp_path / "stdout"
    log.write_text("observed output")
    value = p.build(
        w,
        [dict(passed=True, scope="owned", stdout_path=str(log), stdout_sha256=sha256_file(log))],
        tmp_path,
        tmp_path / (p.NAME + ".json"),
    )
    assert p.replay(value)
    log.write_text("substituted output")
    assert not p.replay(value)


def test_cached_game_memory_withheld(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8384: cached per-game transition memory is removed through the existing loader seam."""
    from carnot.agentic import arc_frame_change_predictor as numeric

    calls = []
    monkeypatch.setattr(
        numeric, "load_live_action_effect_scorer", lambda **kwargs: calls.append(kwargs) or None
    )
    result = r.episode(unit(), Arcade(), tmp_path)
    assert result["model_tripwire_passed"]
    assert calls and all(call["use_memory"] is False for call in calls)
    assert result["adapter_withheld"]["cached_action_effect_memory"] is True
