"""Spec: REQ-ARC-WMTE-7010, SCENARIO-ARC-WMTE-7010-A, SCENARIO-ARC-WMTE-7010-B,
SCENARIO-ARC-WMTE-7010-C, SCENARIO-ARC-WMTE-7010-D, SCENARIO-ARC-WMTE-7010-E.

The live-path eval writes an in-run heartbeat for the game in flight.

INCIDENT 2026-09-04. A single r11l game ran for 6h57m (24,998 s) and wrote nothing until it
ended. The per-game partial (REQ-ARC-WMTE-6850) banks a game only after it finishes, so
WITHIN a game an operator could not tell a level-up from a hang, and the game's wall clock
reached the record only as a line on stdout. Measured the same day: the classical path runs
272 actions in one second with the LLM off, so a stale file can only mean a generator call.

These tests drive the REAL `run_game` loop against scripted fakes, as
`test_arc_per_level_reset_attribution.py` does: the defect is in the control flow around the
loop. Every write and unlink is captured in memory; the one test that drives `main()` lets it
create the gitignored run-scoped directory, exactly as the REQ-6850 tests do.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import arc_leaderboard_eval as ale  # noqa: E402


class _Frame:
    def __init__(self, levels_completed: int) -> None:
        self.levels_completed = int(levels_completed)
        self.state = "NOT_FINISHED"
        self.available_actions: list = []


class _FakeEnv:
    """Replays a scripted level schedule: move index -> levels_completed after that move."""

    def __init__(self, schedule: dict[int, int]) -> None:
        self.schedule = dict(schedule)
        self.i = -1
        self.level = 0
        self.info = type("I", (), {"baseline_actions": []})()

    def _advance(self) -> _Frame:
        self.i += 1
        if self.i in self.schedule:
            self.level = int(self.schedule[self.i])
        return _Frame(self.level)

    def reset(self) -> _Frame:
        return self._advance()

    def step(self, action: Any, data: Any = None) -> _Frame:
        return self._advance()


class _FakePolicy:
    """Scripted moves, plus the two seams the heartbeat reads: an `induction_attempts` list
    that grows at a scripted move, and the `induction_progress_hook` attribute."""

    def __init__(self, n_moves: int, attempt_at: int | None = None) -> None:
        self.n_moves = int(n_moves)
        self.n = 0
        self.attempt_at = attempt_at
        self.induction_attempts: list[dict[str, Any]] = []
        self.induction_progress_hook: Any = None

    def is_done(self, frames: Any, latest: Any) -> bool:
        return self.n >= self.n_moves

    def next_move(self, frames: Any, latest: Any) -> tuple:
        if self.attempt_at is not None and self.n == self.attempt_at:
            self.induction_attempts.append(
                {
                    "reason": "stall",
                    "planned": False,
                    "skipped": "x",
                    "transition_count": 3,
                    "wall_s": 12.5,
                }
            )
        self.n += 1
        return "1", None


@pytest.fixture
def writes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, dict[str, Any]]]:
    captured: list[tuple[Path, dict[str, Any]]] = []
    monkeypatch.setattr(
        ale, "_write_json_atomic", lambda out, payload: captured.append((out, json.loads(payload)))
    )
    return captured


def _run(
    monkeypatch: pytest.MonkeyPatch,
    policy: _FakePolicy,
    schedule: dict[int, int],
    progress: Any,
    budget: int | None = None,
) -> dict[str, Any]:
    env = _FakeEnv(schedule)
    arcade = type(
        "A",
        (),
        {"open_scorecard": lambda self: "sc", "make": lambda self, g, scorecard_id=None: env},
    )()
    monkeypatch.setattr(ale.kit, "offline_arcade", lambda: arcade)
    return ale.run_game("fake", policy, budget=budget or policy.n_moves + 2, progress=progress)


def _events(writes: list[tuple[Path, dict[str, Any]]], path: Path) -> list[dict[str, Any]]:
    return [p for out, p in writes if out == path]


# --- SCENARIO-A: a level-up is on disk the moment it happens -------------------------------


def test_a_level_up_is_written_when_it_happens(
    writes: list[tuple[Path, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-7010-A: the heartbeat records the level-up event with its action count."""
    path = tmp_path / "x.progress.json"
    policy = _FakePolicy(n_moves=6)
    progress = ale.ProgressWriter(path, game="fake", game_index=1, games_planned=2, policy=policy)
    row = _run(monkeypatch, policy, {2: 1}, progress)

    ups = [e for e in _events(writes, path) if e["last_event"] == "level_up"]
    assert len(ups) == 1
    assert ups[0]["level"] == 1 and ups[0]["levels"] == 1
    assert ups[0]["level_up_actions"] == [3]  # three actions taken when the level flipped
    assert ups[0]["game"] == "fake" and ups[0]["game_index"] == 1 and ups[0]["games_planned"] == 2
    assert ups[0]["schema"] == ale.PROGRESS_SCHEMA
    # the final write closes the game and the row agrees
    assert _events(writes, path)[-1]["game_complete"] is True
    assert row["levels"] == 1


def test_the_writer_never_uses_the_partial_readers_key(
    writes: list[tuple[Path, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-7010-A: a heartbeat is not a result record. Readers of the partial
    files key on `complete`; a heartbeat carrying that key would be read as a banked game."""
    path = tmp_path / "x.progress.json"
    policy = _FakePolicy(n_moves=3)
    progress = ale.ProgressWriter(path, game="fake", game_index=1, games_planned=1, policy=policy)
    _run(monkeypatch, policy, {}, progress)
    for event in _events(writes, path):
        assert "complete" not in event
        assert "game_complete" in event


# --- SCENARIO-B: the file is never older than a period of search ---------------------------


def test_periodic_heartbeats_fire_every_period(
    writes: list[tuple[Path, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-7010-B: with no events, a write lands every PROGRESS_EVERY_STEPS."""
    path = tmp_path / "x.progress.json"
    n = ale.PROGRESS_EVERY_STEPS * 2 + 5
    policy = _FakePolicy(n_moves=n)
    progress = ale.ProgressWriter(path, game="fake", game_index=1, games_planned=1, policy=policy)
    _run(monkeypatch, policy, {}, progress, budget=n + 2)

    periodic = [e for e in _events(writes, path) if e["last_event"] == "periodic"]
    assert [e["loop_index"] for e in periodic] == [
        0,
        ale.PROGRESS_EVERY_STEPS,
        2 * ale.PROGRESS_EVERY_STEPS,
    ]
    assert periodic[-1]["actions"] == 2 * ale.PROGRESS_EVERY_STEPS + 1
    assert periodic[-1]["elapsed_s"] >= 0.0


# --- SCENARIO-C: induction is visible, in flight and when finished -------------------------


def test_a_new_induction_attempt_is_written_and_summarised(
    writes: list[tuple[Path, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-7010-C: the attempt count and the last attempt's wall time reach the
    file the step after the policy records them, with or without the hook."""
    path = tmp_path / "x.progress.json"
    policy = _FakePolicy(n_moves=8, attempt_at=4)
    progress = ale.ProgressWriter(path, game="fake", game_index=1, games_planned=1, policy=policy)
    row = _run(monkeypatch, policy, {}, progress)

    recorded = [e for e in _events(writes, path) if e["last_event"] == "induction_attempt_recorded"]
    assert len(recorded) == 1
    assert recorded[0]["induction_attempts_n"] == 1
    assert recorded[0]["last_induction"]["wall_s"] == 12.5
    assert recorded[0]["generator_wall_s"] == 12.5
    # ...and the row carries the same accounting (REQ-ARC-WMTE-7011 row fields)
    assert row["generator_wall_s"] == 12.5
    assert row["induction_attempt_wall_s"] == [12.5]


def test_the_policy_hook_marks_an_induce_in_flight(
    writes: list[tuple[Path, dict[str, Any]]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """SCENARIO-ARC-WMTE-7010-C: constructing the writer installs it as the policy's hook; a
    started event shows the induce as in flight, a finished event clears it and prints."""
    path = tmp_path / "x.progress.json"
    policy = _FakePolicy(n_moves=1)
    progress = ale.ProgressWriter(path, game="fake", game_index=1, games_planned=1, policy=policy)
    assert policy.induction_progress_hook == progress.on_induction_event

    policy.induction_progress_hook(
        "induction_started",
        {"attempt_index": 0, "reason": "stall", "transition_count": 25, "started_at": "t0"},
    )
    started = _events(writes, path)[-1]
    assert started["last_event"] == "induction_started"
    assert started["induction_in_flight"]["reason"] == "stall"
    assert started["induction_in_flight"]["transition_count"] == 25

    policy.induction_progress_hook(
        "induction_finished",
        {
            "attempt_index": 0,
            "reason": "stall",
            "planned": True,
            "skipped": "",
            "transition_count": 25,
            "wall_s": 1501.2,
            "started_at": "t0",
        },
    )
    finished = _events(writes, path)[-1]
    assert finished["last_event"] == "induction_finished"
    assert finished["induction_in_flight"] is None
    assert finished["last_induction"]["wall_s"] == 1501.2
    out = capsys.readouterr().out
    assert "induce#0" in out and "wall=1501s" in out


# --- SCENARIO-D: the row carries the game's wall clock -------------------------------------


def test_the_row_carries_wall_clock_fields(
    writes: list[tuple[Path, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-7010-D: `wall_s`, `started_at`, `finished_at` on every row."""
    policy = _FakePolicy(n_moves=3)
    progress = ale.ProgressWriter(
        tmp_path / "x.progress.json", game="fake", game_index=1, games_planned=1, policy=policy
    )
    row = _run(monkeypatch, policy, {}, progress)
    assert row["wall_s"] >= 0.0
    assert row["started_at"].endswith("+00:00") and row["finished_at"].endswith("+00:00")
    assert row["progress_write_errors"] == 0
    # a run without a writer still carries the clock, and says no writer was attached
    row2 = _run(monkeypatch, _FakePolicy(n_moves=3), {}, None)
    assert row2["wall_s"] >= 0.0 and row2["progress_write_errors"] is None


# --- SCENARIO-E: instrumentation never takes the run down ----------------------------------


def test_a_failing_writer_is_counted_not_raised(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-7010-E: a write that raises is counted in `write_errors`; the game
    finishes and its row names the count."""

    def _boom(out: Path, payload: str) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(ale, "_write_json_atomic", _boom)
    policy = _FakePolicy(n_moves=3)
    progress = ale.ProgressWriter(
        tmp_path / "x.progress.json", game="fake", game_index=1, games_planned=1, policy=policy
    )
    row = _run(monkeypatch, policy, {1: 1}, progress)
    assert row["levels"] == 1
    assert progress.write_errors >= 2  # at least the first step and the level-up
    assert row["progress_write_errors"] == progress.write_errors


def test_a_policy_without_the_seams_still_gets_a_heartbeat(
    writes: list[tuple[Path, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO-ARC-WMTE-7010-E: the tier-1 explorer policy has no `induction_attempts` and
    no hook; the writer must not assume either."""

    class _Bare:
        def __init__(self) -> None:
            self.n = 0

        def is_done(self, frames: Any, latest: Any) -> bool:
            return self.n >= 3

        def next_move(self, frames: Any, latest: Any) -> tuple:
            self.n += 1
            return "1", None

    path = tmp_path / "x.progress.json"
    env = _FakeEnv({})
    arcade = type(
        "A",
        (),
        {"open_scorecard": lambda self: "sc", "make": lambda self, g, scorecard_id=None: env},
    )()
    monkeypatch.setattr(ale.kit, "offline_arcade", lambda: arcade)
    policy = _Bare()
    progress = ale.ProgressWriter(path, game="fake", game_index=1, games_planned=1, policy=policy)
    ale.run_game("fake", policy, budget=5, progress=progress)
    last = _events(writes, path)[-1]
    assert last["induction_attempts_n"] == 0
    assert last["supervisor"]["enabled"] is False
    assert last["generator_channels"] == {"proposer": "absent"}
    assert progress.write_errors == 0


# --- the call site: main() wires a writer per game and removes the file at the end ---------


def test_main_wires_a_heartbeat_per_game_and_removes_it_at_the_end(
    writes: list[tuple[Path, dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7010: a writer nothing constructs is the defect this REQ exists for."""
    seen: list[Any] = []

    def _run_game(game: str, policy: Any, **kw: Any) -> dict[str, Any]:
        seen.append(kw.get("progress"))
        return {
            "game": game,
            "levels": 0,
            "reached": 0,
            "actions": 1,
            "efficiency": 0.0,
            "gap": None,
            "reset_replay_steps": 0,
            "forward_walk_hit_rate": 0.0,
        }

    gone: list[Path] = []
    monkeypatch.setattr(Path, "unlink", lambda self, missing_ok=False: gone.append(self))
    monkeypatch.setattr(ale, "run_game", _run_game)
    monkeypatch.setattr(ale, "_build_policy", lambda kind, game: object())
    # main() reads ops/arc_solve_registry.yaml under REPO, so REPO cannot be routed to tmp_path.
    # It also mkdirs the gitignored run-scoped directory, as the REQ-6850 tests already do;
    # every write and every unlink is captured, so no file is created or removed.
    monkeypatch.setattr(ale, "CLAIMED", ["gameA", "gameB"])
    monkeypatch.setattr(sys, "argv", ["arc_leaderboard_eval.py", "--only", "gameA,gameB"])
    assert ale.main() == 0

    assert len(seen) == 2 and all(isinstance(p, ale.ProgressWriter) for p in seen)
    assert [p.game_index for p in seen] == [1, 2] and seen[0].games_planned == 2
    assert seen[0].path is not None and seen[0].path.name.endswith(".progress.json")
    assert seen[0].path.parent.name == "arc_leaderboard_eval_runs"
    assert seen[0].path in gone


# --- REQ-ARC-WMTE-7031 on the heartbeat: the table running dry is visible in flight --------
# Lives here, not in test_arc_supervisor_exhaustion_20260905.py, because this file already pays
# the eval import at collection time; importing it inside a test trips the memory watchdog.


def test_scenario_7031_d_the_heartbeat_shows_the_table_running_dry() -> None:
    """SCENARIO-ARC-WMTE-7031-D: the progress record's supervisor block carries
    `stagnations_unredirected` and the number of window rows kept, from a receipt built by a
    real supervisor, so a reader sees the table run dry before the game ends."""
    from carnot.agentic.arc_trajectory_supervisor import TrajectorySnapshot, TrajectorySupervisor

    sup = TrajectorySupervisor(window=1)
    busy = TrajectorySnapshot(
        level=0,
        goal_bias_installed=False,
        induced=False,
        induction_attempts=0,
        new_transitions_since_induction=0,
        diversity_active=True,
    )
    for _ in range(3):
        assert sup.observe(busy) is None

    class _Policy:
        induction_attempts: list = []

        def trajectory_supervisor_diagnostics(self) -> dict:
            receipt = sup.receipt()
            receipt["mode"] = "applied"
            return receipt

    writer = ale.ProgressWriter(None, game="r11l", game_index=1, games_planned=1, policy=_Policy())
    block = writer.snapshot()["supervisor"]
    assert block["redirects_n"] == 0
    assert block["stagnations_unredirected"] == 3
    assert block["unredirected_windows_n"] == 3
