"""REQ-REPORT-7708 and REQ-ARC-WMTE-7708 CPU contract tests."""

from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from carnot.agentic import arc_generalization_runtime as runtime
from carnot.agentic import arc_strategy_router
from carnot import experiment_7708_v671_arc_generalization_runner as runner


class Frame:
    def __init__(self, level: int = 0) -> None:
        self.levels_completed = level
        self.frame = [[[0] * 8 for _ in range(8)]]
        self.available_actions = [1, 2, 3, 4, 5, 6]
        self.state = "NOT_FINISHED"


class Environment:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def reset(self) -> Frame:
        self.calls.append("reset")
        return Frame()

    def step(self, action: object, *, data: object = None) -> Frame:
        self.calls.append("step")
        if self.fail:
            raise RuntimeError("fixture SDK failure")
        return Frame(1)


class Arcade:
    def __init__(self, *, fail: bool = False) -> None:
        self.env = Environment(fail=fail)
        self.made: list[str] = []

    def open_scorecard(self) -> str:
        return "fixture-scorecard"

    def make(self, game: str, *, scorecard_id: str) -> Environment:
        self.made.append(game)
        assert scorecard_id == "fixture-scorecard"
        return self.env


def fake_factory(base_cls: type, *, cascade: bool, proposer: object) -> type:
    assert cascade and proposer is None

    class Agent(base_cls):
        def __init__(self, game_id: str) -> None:
            super().__init__(game_id)
            self._policy = SimpleNamespace(
                induced_goal=None, model_accepted=False, goal_fired=False
            )
            self.choices = 0

        def is_done(self, frames: list[Frame], latest: Frame | None) -> bool:
            return len(frames) >= 2

        def choose_action(self, frames: list[Frame], latest: Frame | None) -> object:
            self.choices += 1
            return SimpleNamespace(name="RESET" if not frames else "ACTION1", action_data=None)

    return Agent


def test_freeze_schedule_keeps_cleared_public_game() -> None:
    """SCENARIO-ARC-WMTE-7708-SCORED-PATH: registry credit cannot filter the roster."""
    registry = [
        {"game": "aa", "levels_reproduced": 5, "full_game_clear": True},
        {"game": "bb", "levels_reproduced": 2},
        {"game": "cc", "levels_reproduced": 0},
    ]
    schedule = runtime.freeze_schedule(["cc", "bb", "aa"], registry)
    expected = sorted(
        ("aa", "bb", "cc"),
        key=lambda game: (runtime.game_digest(game), game),
    )[:2]
    assert [row["game"] for row in schedule["rows"]] == expected
    assert all(row["adapter_withheld"] for row in schedule["rows"])
    assert schedule["registry_precheck"]["aa"]["full_game_clear"] is True
    assert schedule["new_solve_credit"] is False


def test_scored_runner_observes_sdk_transition() -> None:
    """SCENARIO-REPORT-7708-CPU: choose_action and SDK observe really execute."""
    arcade = Arcade()
    row = runtime.run_episode(
        {"game": "aa", "episode_id": "aa:fixture", "max_actions": 2},
        arcade,
        agent_factory=fake_factory,
    )
    assert arcade.made == ["aa"]
    assert arcade.env.calls == ["reset", "step"]
    assert row["counts"] == {"choose_action": 2, "sdk_transitions": 2, "observations": 2}
    assert row["raw_metrics"]["peak_level"] == 1
    assert row["solve_provenance"] == "development_proxy"
    assert row["new_solve_credit"] is False
    assert row["telemetry"][1]["sdk_transition"]["level_after"] == 1


def test_environment_exception_is_censored() -> None:
    """SCENARIO-ARC-WMTE-7708-SCORED-PATH: SDK exceptions stay in raw rows."""
    row = runtime.run_episode(
        {"game": "bb", "episode_id": "bb:fixture", "max_actions": 2},
        Arcade(fail=True),
        agent_factory=fake_factory,
    )
    assert row["censoring"] == "sdk_exception"
    assert row["counts"]["choose_action"] == 2
    assert row["counts"]["sdk_transitions"] == 1
    assert "fixture SDK failure" in row["error"]


def test_missing_sdk_blocks_with_exact_operand(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7708-BLOCKED: absent SDK is an external block."""
    checks, hashes = runner.collect_preconditions(tmp_path, sdk_roster=None)
    candidate = runner.build_artifact(
        checks=checks,
        hashes=hashes,
        schedule=None,
        rows=[],
        run_date="20260926",
        validation_receipts=[],
        duration_s=0.1,
    )
    assert candidate["honest_verdict"].startswith("complete_blocked_")
    assert candidate["verdict_class"] == "blocked"
    assert candidate["arc_runner_ready_score"] == 0
    sdk = next(
        row
        for row in candidate["gate_check_summary"]["failed_checks"]
        if row["check"] == "sdk_catalogue"
    )
    assert set(("upstream", "path", "field", "operator", "expected", "observed")) <= sdk.keys()
    assert candidate["MODEL_SPECS"] == []
    assert candidate["model_invoked"] is False


def test_fixture_success_is_circular_and_requires_validation(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7708-TERMINAL: a fixture cannot establish benefit."""
    root = Path(__file__).resolve().parents[2]
    checks, hashes = runner.collect_preconditions(root, sdk_roster=["aa", "bb"])
    schedule = runtime.freeze_schedule(["aa", "bb"], [])
    rows = [runtime.run_episode(unit, Arcade()) for unit in schedule["rows"]]
    from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES

    receipts = [
        {"name": name, "exit_code": 0, "log_sha256": "sha256:test"}
        for name in (*REQUIRED_CHECK_NAMES, "e2e_009", "e2e_009_smoke", "e2e_011", "e2e_013")
    ]
    candidate = runner.build_artifact(
        checks=checks,
        hashes=hashes,
        schedule=schedule,
        rows=rows,
        run_date="20260926",
        validation_receipts=receipts,
        duration_s=1.0,
    )
    assert candidate["verdict_class"] == "circular_positive"
    assert candidate["arc_runner_ready_score"] == 1
    assert candidate["verifier_is_oracle"] is True
    assert candidate["acceptance_gate_results"]["probability"]["passed"] is False
    candidate = runner.build_artifact(
        checks=checks,
        hashes=hashes,
        schedule=schedule,
        rows=rows,
        run_date="20260926",
        validation_receipts=[
            *receipts,
            {"name": "required", "exit_code": 1, "log_sha256": "sha256:test"},
        ],
        duration_s=1.0,
    )
    assert candidate["verdict_class"] == "disqualified"
    assert candidate["arc_runner_ready_score"] == 0


def test_fixture_preflight_does_not_require_coding_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-7708-TERMINAL: offline replay needs no coding session."""
    monkeypatch.delenv("CODEX_SESSION_ID", raising=False)
    root = Path(__file__).resolve().parents[2]
    checks, _ = runner.collect_preconditions(root, sdk_roster=["aa", "bb"])
    assert all(row["passed"] for row in checks)


def test_real_scored_policy_reaches_observation(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-7708: fixture transport drives the real E3 wrapper."""
    monkeypatch.setenv("CARNOT_ARC_DISABLE_INDUCTION", "1")
    monkeypatch.setenv("CARNOT_ARC_GOAL_PROBE_LOOP", "0")
    arcade = Arcade()
    row = runtime.run_episode(
        {"game": "zz", "episode_id": "zz:real-policy", "max_actions": 2},
        arcade,
    )
    assert row["policy_entry"]["factory"] == "make_carnot_agent"
    assert row["policy_entry"]["policy_class"] == "E3AgentPolicy"
    assert row["counts"]["choose_action"] >= 1
    assert row["counts"]["observations"] >= 1


def test_frozen_selection_needs_two_games() -> None:
    """SCENARIO-REPORT-7708-BLOCKED: one game cannot fake two groups."""
    with pytest.raises(ValueError, match="two_sdk_games_required"):
        runtime.freeze_schedule(["only"], [])


def test_no_adapter_import_or_evaluator_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-7708: the runner denies the two oracle shortcuts."""
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if "arc_game_adapters" in name or "GameAdapter" in name:
            raise AssertionError("adapter import")
        return original_import(name, *args, **kwargs)

    class LabelFrame(Frame):
        @property
        def evaluator_label(self) -> object:
            raise AssertionError("evaluator label read")

    class LabelEnvironment(Environment):
        def reset(self) -> LabelFrame:
            self.calls.append("reset")
            return LabelFrame()

        def step(self, action: object, *, data: object = None) -> LabelFrame:
            self.calls.append("step")
            return LabelFrame(1)

    arcade = Arcade()
    arcade.env = LabelEnvironment()
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    row = runtime.run_episode(
        {"game": "aa", "episode_id": "aa:no-leak", "max_actions": 2},
        arcade,
    )
    assert row["error"] is None
    assert row["counts"]["observations"] == 2


def test_runtime_time_and_policy_stop() -> None:
    """REQ-ARC-WMTE-7708: a CPU fixture has bounded action work."""
    base = {"game": "aa", "episode_id": "aa:bound", "max_actions": 3}
    timeout = runtime.run_episode({**base, "max_seconds": -1}, Arcade(), agent_factory=fake_factory)
    assert timeout["censoring"] == "time_limit"
    assert timeout["counts"]["choose_action"] == 0
    stopped = runtime.run_episode(base, Arcade(), agent_factory=fake_factory)
    assert stopped["censoring"] is None
    assert stopped["counts"]["choose_action"] == 2


def test_validation_scope_builds_private_e2e_commands(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7708-TERMINAL: E2E commands are frozen before a child starts."""
    from carnot.reporting import experiment_7303_validation_scope as scope

    seen: list[str] = []
    monkeypatch.setattr(scope, "build_scoped_commands", lambda *args, **kwargs: [])

    def run_commands(
        root: Path, commands: list[object], **kwargs: object
    ) -> list[dict[str, object]]:
        seen.extend(command.name for command in commands)
        assert kwargs["heartbeat_s"] == 45.0
        assert kwargs["extra_env"]["CARNOT_ARC_DISABLE_INDUCTION"] == "1"
        return [{"name": command.name, "exit_code": 0} for command in commands]

    monkeypatch.setattr(scope, "run_commands", run_commands)
    receipts = runner._validate(tmp_path, tmp_path / "private", 0.0)
    assert seen == ["e2e_009", "e2e_011", "e2e_013", "e2e_009_smoke"]
    assert len(receipts) == 4


def test_validation_places_temporary_outputs_outside_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7708-TERMINAL: E3 smoke and pytest temp avoid immutable results."""
    from carnot.reporting import experiment_7303_validation_scope as scope

    private = tmp_path / "results" / "raw" / "private_validation"
    captured: list[Path] = []

    def build_commands(root: Path, tests: object, modules: object, **kwargs: object) -> list:
        captured.append(Path(kwargs["basetemp"]))
        return []

    def run_commands(root: Path, commands: list, **kwargs: object) -> list[dict]:
        for command in commands:
            if command.name.startswith("e2e_0") and command.name != "e2e_009_smoke":
                temp_arg = next(arg for arg in command.argv if arg.startswith("--basetemp="))
                captured.append(Path(temp_arg.split("=", 1)[1]))
            elif command.name == "e2e_009_smoke":
                captured.append(Path(command.argv[command.argv.index("--output") + 1]))
        assert all(path.exists() or path.parent.exists() for path in captured)
        return [{"name": command.name, "exit_code": 0} for command in commands]

    monkeypatch.setattr(scope, "build_scoped_commands", build_commands)
    monkeypatch.setattr(scope, "run_commands", run_commands)
    runner._validate(tmp_path, private, 0.0)
    assert len(captured) == 5
    assert all(not path.resolve().is_relative_to(tmp_path / "results") for path in captured)


def test_terminal_reader_commands_bind_one_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7708-TERMINAL: cold and strict readers see one path."""
    from carnot.reporting import experiment_7303_validation_scope as scope

    candidate = tmp_path / "candidate.json"

    def run_commands(
        root: Path, commands: list[object], **kwargs: object
    ) -> list[dict[str, object]]:
        assert len(commands) == 3
        assert all(str(candidate) in command.argv for command in commands)
        return [{"name": command.name, "exit_code": 0} for command in commands]

    monkeypatch.setattr(scope, "run_commands", run_commands)
    receipts = runner._terminal_readers(tmp_path, candidate, tmp_path, 0.0)
    assert [row["name"] for row in receipts] == [
        "cold_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]


def test_cold_reader_recounts_and_rejects_credit(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7708-TERMINAL: persisted rows control the count."""
    import json

    path = tmp_path / "candidate.json"
    value = {
        "rows": [
            {
                "game": "aa",
                "counts": {"choose_action": 2, "observations": 2},
                "new_solve_credit": False,
            }
        ],
        "sample_size_budget": {"observed_independent_games": 1},
    }
    path.write_text(json.dumps(value))
    assert runner.cold_read(path)["observations"] == 2
    value["sample_size_budget"]["observed_independent_games"] = 2
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="raw_group_count_mismatch"):
        runner.cold_read(path)
    value["sample_size_budget"]["observed_independent_games"] = 1
    value["rows"][0]["new_solve_credit"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fixture_solve_credit"):
        runner.cold_read(path)


def _mock_receipts() -> list[dict[str, object]]:
    from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES

    return [
        {"name": name, "exit_code": 0, "log_sha256": "sha256:fixture"}
        for name in (*REQUIRED_CHECK_NAMES, "e2e_009", "e2e_009_smoke", "e2e_011", "e2e_013")
    ]


def test_complete_runner_scripted_transport(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-7708: the actual runner publishes a two-game CPU receipt."""
    registry = tmp_path / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir()
    registry.write_text("games: []\n")
    monkeypatch.setattr(runner, "_catalogue", lambda started: (["aa", "bb"], None, []))
    monkeypatch.setattr(
        runner,
        "collect_preconditions",
        lambda root, sdk_roster: (
            [runner.check("fixture", "test", "fixture", "ready", True, True)],
            {"producer_files": {}},
        ),
    )
    monkeypatch.setattr(runner, "_validate", lambda *args: _mock_receipts())
    monkeypatch.setattr(
        runner,
        "_terminal_readers",
        lambda *args: [{"name": "cold_reduction", "exit_code": 0}],
    )
    output = tmp_path / "result.json"
    candidate = runner.run_experiment(tmp_path, "20260926", output)
    assert candidate["verdict_class"] == "circular_positive"
    assert candidate["arc_runner_ready_score"] == 1
    assert len(candidate["rows"]) == 2
    assert all(row["policy_entry"]["policy_class"] == "E3AgentPolicy" for row in candidate["rows"])
    assert (tmp_path / runner.RAW / "schedule.json").is_file()
    assert (tmp_path / runner.RAW / "validation_receipts.json").is_file()
    assert output.is_file()


def test_sdk_missing_is_blocked_even_after_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7708-BLOCKED: no fake two-game schedule is written."""
    monkeypatch.setattr(runner, "_catalogue", lambda started: (None, "missing SDK", []))
    monkeypatch.setattr(
        runner,
        "collect_preconditions",
        lambda root, sdk_roster: (
            [runner.check("sdk_catalogue", "SDK", "environment_files", "present", True, False)],
            {"missing_custody": ["SDK"]},
        ),
    )
    monkeypatch.setattr(runner, "_validate", lambda *args: _mock_receipts())
    monkeypatch.setattr(runner, "_terminal_readers", lambda *args: [])
    result = runner.run_experiment(tmp_path, "20260926", tmp_path / "blocked.json")
    assert result["verdict_class"] == "blocked"
    assert result["arc_runner_ready_score"] == 0
    assert not (tmp_path / runner.RAW / "schedule.json").exists()


def test_terminal_failure_disqualifies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """SCENARIO-REPORT-7708-TERMINAL: an adversarial failure zeroes readiness."""
    registry = tmp_path / "ops" / "arc_solve_registry.yaml"
    registry.parent.mkdir()
    registry.write_text("games: []\n")
    monkeypatch.setattr(runner, "_catalogue", lambda started: (["aa", "bb"], None, []))
    monkeypatch.setattr(
        runner,
        "collect_preconditions",
        lambda root, sdk_roster: (
            [runner.check("fixture", "test", "fixture", "ready", True, True)],
            {},
        ),
    )
    monkeypatch.setattr(runner, "_validate", lambda *args: _mock_receipts())
    calls = 0

    def terminal(*args: object) -> list[dict[str, object]]:
        nonlocal calls
        calls += 1
        return [{"name": "adversarial_verify", "exit_code": 2 if calls == 1 else 0}]

    monkeypatch.setattr(runner, "_terminal_readers", terminal)
    result = runner.run_experiment(tmp_path, "20260926", tmp_path / "disqualified.json")
    assert calls == 2
    assert result["verdict_class"] == "disqualified"
    assert result["flagged_adversarial"] is True
    assert result["arc_runner_ready_score"] == 0


def test_catalogue_reports_resource_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7708-BLOCKED: SDK availability is measured, not assumed."""
    from carnot.agentic import arc_solver_kit

    fake = Arcade()
    monkeypatch.setattr(
        arc_solver_kit,
        "offline_arcade",
        lambda: SimpleNamespace(
            available_environments=[
                SimpleNamespace(game_id="aa-v1"),
                SimpleNamespace(game_id="bb-v1"),
            ],
            make=fake.make,
            open_scorecard=fake.open_scorecard,
        ),
    )
    assert runner._catalogue(0.0) == (["aa", "bb"], None, [])
    original_make = fake.make

    def mixed_make(game: str, *, scorecard_id: str) -> Environment:
        if game == "bb":
            raise RuntimeError("broken game")
        return original_make(game, scorecard_id=scorecard_id)

    monkeypatch.setattr(fake, "make", mixed_make)
    games, error, excluded = runner._catalogue(0.0)
    assert games == ["aa"] and error is None
    assert excluded[0]["game"] == "bb"

    def unavailable() -> object:
        raise RuntimeError("sdk unavailable")

    monkeypatch.setattr(arc_solver_kit, "offline_arcade", unavailable)
    games, error, excluded = runner._catalogue(0.0)
    assert games is None and "sdk unavailable" in error
    assert excluded == []


def test_catalogue_excludes_null_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7708-CPU: a game without a reset observation is not runnable."""
    from carnot.agentic import arc_solver_kit

    class NullReset(Environment):
        def reset(self) -> None:
            return None

    arcade = Arcade()
    monkeypatch.setattr(
        arcade,
        "make",
        lambda game, *, scorecard_id: NullReset() if game == "bb" else arcade.env,
    )
    monkeypatch.setattr(
        arc_solver_kit,
        "offline_arcade",
        lambda: SimpleNamespace(
            available_environments=[
                SimpleNamespace(game_id="aa-v1"),
                SimpleNamespace(game_id="bb-v1"),
            ],
            make=arcade.make,
            open_scorecard=arcade.open_scorecard,
        ),
    )
    games, error, excluded = runner._catalogue(0.0)
    assert games == ["aa"] and error is None
    assert excluded == [{"game": "bb", "reason": "RuntimeError: sdk_null_observation"}]


def test_partial_and_fixture_transport_guard(tmp_path: Path) -> None:
    """REQ-REPORT-7708: unfinished owned validation is the only partial case."""
    checks = [runner.check("ready", "fixture", "fixture", "present", True, True)]
    schedule = runtime.freeze_schedule(["aa", "bb"], [])
    rows = [runtime.run_episode(unit, Arcade()) for unit in schedule["rows"]]
    partial = runner.build_artifact(
        checks=checks,
        hashes={},
        schedule=schedule,
        rows=rows,
        run_date="20260926",
        validation_receipts=[],
        duration_s=1.0,
    )
    assert partial["verdict_class"] == "partial"
    invalid = runner.build_artifact(
        checks=checks,
        hashes={},
        schedule=schedule,
        rows=[],
        run_date="20260926",
        validation_receipts=[],
        duration_s=1.0,
    )
    assert invalid["verdict_class"] == "disqualified"
    with pytest.raises(ValueError, match="invalid_fixture_transport"):
        runner._FixtureArcade().make("", scorecard_id="wrong")


def test_cli_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-7708: the entrypoint supports separate cold reduction."""
    import json

    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps({"rows": [], "sample_size_budget": {"observed_independent_games": 0}})
    )
    assert runner.main(["--cold-read", str(candidate)]) == 0
    assert '"independent_games": 0' in capsys.readouterr().out
    called: list[object] = []
    monkeypatch.setattr(runner, "run_experiment", lambda *args: called.append(args))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    assert runner.main(["--date", "20260926", "--output", "out.json"]) == 0
    assert called[0][2] == tmp_path / "out.json"


def test_telemetry_flag_preserves_scored_fixture_actions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7708-CPU: passive tracing leaves the scored choice intact."""
    import random

    import numpy as np

    from carnot.agentic import arc_decision_telemetry as telemetry

    unit = {"game": "aa", "episode_id": "aa:parity", "max_actions": 3}
    monkeypatch.delenv(telemetry.TELEMETRY_ENV_FLAG, raising=False)
    random.seed(7708)
    np.random.seed(7708)
    off = runtime.run_episode(unit, Arcade())
    monkeypatch.setenv(telemetry.TELEMETRY_ENV_FLAG, "1")
    monkeypatch.setenv(telemetry.TELEMETRY_PATH_ENV, str(tmp_path / "telemetry.jsonl"))
    random.seed(7708)
    np.random.seed(7708)
    on = runtime.run_episode(unit, Arcade())
    trace = lambda row: [(step["action"], step["data"]) for step in row["telemetry"]]
    assert trace(on) == trace(off)
    assert on["counts"] == off["counts"]


def test_registry_route_is_withheld_from_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-WMTE-7708: known-game mechanic labels cannot route the agent."""

    def reject_registry(reg: object = None) -> object:
        assert reg == {"games": []}
        return reg

    monkeypatch.setattr(arc_strategy_router, "_load_registry", reject_registry)
    row = runtime.run_episode(
        {"game": "wa30", "episode_id": "wa30:no-registry", "max_actions": 2},
        Arcade(),
    )
    assert row["error"] is None


def test_observation_hash_binds_every_numpy_pixel() -> None:
    """SCENARIO-ARC-WMTE-7708-SCORED-PATH: hidden center pixels alter the receipt."""
    left = np.zeros((64, 64), dtype=np.uint8)
    right = left.copy()
    right[30, 30] = 1
    a = SimpleNamespace(frame=[left], levels_completed=0, available_actions=[])
    b = SimpleNamespace(frame=[right], levels_completed=0, available_actions=[])
    assert runtime._observation(a)["frame_sha256"] != runtime._observation(b)["frame_sha256"]


def test_observation_rejects_unsupported_visible_value() -> None:
    """SCENARIO-REPORT-7708-CPU: unsupported SDK pixels cannot enter a receipt."""
    frame = SimpleNamespace(frame=[{1, 2}], levels_completed=0, available_actions=[])
    with pytest.raises(TypeError, match="unsupported visible frame type: set"):
        runtime._observation(frame)


def test_sdk_none_frame_is_censored() -> None:
    """SCENARIO-REPORT-7708-CPU: SDK's swallowed failure cannot pass readiness."""

    class NoneEnvironment(Environment):
        def reset(self) -> None:
            return None

    arcade = Arcade()
    arcade.env = NoneEnvironment()
    row = runtime.run_episode(
        {"game": "aa", "episode_id": "aa:null-sdk", "max_actions": 2},
        arcade,
        agent_factory=fake_factory,
    )
    assert row["censoring"] == "sdk_exception"
    assert row["counts"]["observations"] == 0
    assert "sdk_null_observation" in row["error"]
