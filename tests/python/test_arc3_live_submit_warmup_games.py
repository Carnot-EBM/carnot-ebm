"""Regression test for the 2026-07-15 arc3_live_submit sc25 WARMUP_GAMES fix.

REQ: replay_live() must apply the same no-op-first-step warmup workaround that
arc3_replay_scorecard_metaharness.py's own replay_game() already applies for
WARMUP_GAMES (currently {"sc25"}), and that arc_solver_kit.reproduce() (THE
REPRODUCTION GATE that offline-certified sc25's L5 claim) requires via its
warmup_label parameter. Without it, every action in a WARMUP_GAMES trajectory
lands one position out of phase against the live game's actual state -- the
confirmed root cause of the 2026-07-15 "sc25 claimed L5 -> LIVE L-1 MISMATCH"
(a live 400 Bad Request on ACTION4, the 22nd banked action).

SCENARIO-LIVESUBMIT-WARMUP-1: a WARMUP_GAMES game gets an extra warmup env.step
                              (repeating actions[0]) BEFORE the real replay loop.
SCENARIO-LIVESUBMIT-WARMUP-2: a non-WARMUP_GAMES game is replayed unchanged (no
                              extra step) -- the fix must not regress every other
                              game.
SCENARIO-LIVESUBMIT-WARMUP-3: an empty action list or a dead reset (frame is
                              None) never crashes the warmup step.
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "arc3_live_submit_warmup", str(REPO / "scripts" / "arc3_live_submit.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _FakeFrame:
    def __init__(self, guid="g-1", levels=0):
        self.guid = guid
        self._levels = levels


class _FakeEnv:
    """Records every env.step() call (action id, data, reasoning) in order."""

    def __init__(self):
        self.steps: list[tuple[int, dict | None, dict]] = []
        self._dead = False

    def reset(self):
        return _FakeFrame()

    def step(self, action, data=None, reasoning=None):
        if self._dead:
            return None
        aid = int(str(action).rsplit("ACTION", 1)[-1])
        self.steps.append((aid, data, reasoning))
        return _FakeFrame(levels=len(self.steps))


class _FakeArcade:
    def __init__(self, env):
        self._env = env

    def make(self, gid, scorecard_id=None):
        return self._env


class _FakeMH:
    WARMUP_GAMES = {"sc25"}

    @staticmethod
    def normalize(a):
        aid = a.get("action")
        data = a.get("data")
        return (int(aid) if aid is not None else None), data


@pytest.fixture(autouse=True)
def _stub_frame_helpers(monkeypatch):
    """replay_live() imports these from real Carnot modules; stub them so this test
    stays a pure unit test of the warmup-step ordering, not an integration test of
    the frame/grid machinery (which is exercised elsewhere)."""
    drv = _load_driver()

    def _levels_completed(frame):
        return 0 if frame is None else getattr(frame, "_levels", 0)

    def _grid_of(frame):
        return [[0]]

    monkeypatch.setattr(
        "carnot.agentic.arc_agi3_live_adapter._levels_completed", _levels_completed, raising=False
    )
    monkeypatch.setattr("carnot.agentic.arc_agi3_world_model.grid_of", _grid_of, raising=False)
    return drv


def test_scenario_livesubmit_warmup_1_prepends_throwaway_step(_stub_frame_helpers):
    """SCENARIO-LIVESUBMIT-WARMUP-1: sc25 (a WARMUP_GAMES game) gets ONE extra
    env.step repeating actions[0] before the real replay begins, mirroring
    replay_game()'s WARMUP_GAMES handling."""
    drv = _stub_frame_helpers
    env = _FakeEnv()
    arcade = _FakeArcade(env)
    drv.resolve_game_id = lambda arcade, short: short  # type: ignore[method-assign]
    actions = [
        {"action": 6, "data": {"x": 29, "y": 49}},
        {"action": 3},
        {"action": 4},
    ]
    lvl, guid = drv.replay_live(arcade, "sc25", "sc-1", actions, _FakeMH(), corpus=None)
    # 1 warmup step (repeats actions[0]) + 3 real steps = 4 total env.step calls
    assert len(env.steps) == 4
    assert env.steps[0] == (6, {"x": 29, "y": 49}, {"policy": "warmup"})
    assert env.steps[1] == (6, {"x": 29, "y": 49}, {"policy": "offline_reproduced_replay"})
    assert env.steps[2] == (3, None, {"policy": "offline_reproduced_replay"})
    assert env.steps[3] == (4, None, {"policy": "offline_reproduced_replay"})
    assert lvl == 4  # _levels_completed reflects the total step count in this fake


def test_scenario_livesubmit_warmup_2_non_warmup_game_unchanged(_stub_frame_helpers):
    """SCENARIO-LIVESUBMIT-WARMUP-2: a game NOT in WARMUP_GAMES gets no extra step --
    the fix must not regress the other ~20 games the live driver replays."""
    drv = _stub_frame_helpers
    env = _FakeEnv()
    arcade = _FakeArcade(env)
    drv.resolve_game_id = lambda arcade, short: short  # type: ignore[method-assign]
    actions = [{"action": 6, "data": {"x": 1, "y": 2}}, {"action": 3}]
    drv.replay_live(arcade, "lp85", "sc-1", actions, _FakeMH(), corpus=None)
    assert len(env.steps) == 2  # no warmup step injected
    assert all(s[2] == {"policy": "offline_reproduced_replay"} for s in env.steps)


def test_scenario_livesubmit_warmup_3_empty_actions_and_dead_reset_safe(_stub_frame_helpers):
    """SCENARIO-LIVESUBMIT-WARMUP-3: an empty action list never crashes the warmup-
    step logic (no actions -> no warmup, no real steps). A reset() that returns None
    (dead env) also never crashes the NEW warmup-step guard specifically -- it is
    skipped (frame is not None guard) -- matching the pre-existing, unchanged
    behavior of the main replay loop, which still attempts its first env.step() even
    on a None reset (only breaking on a None RESULT from step(), not from reset())."""
    drv = _stub_frame_helpers
    env = _FakeEnv()
    arcade = _FakeArcade(env)
    drv.resolve_game_id = lambda arcade, short: short  # type: ignore[method-assign]
    lvl, _guid = drv.replay_live(arcade, "sc25", "sc-1", [], _FakeMH(), corpus=None)
    assert env.steps == []
    assert lvl == 0

    class _DeadEnv(_FakeEnv):
        def reset(self):
            return None

    dead_env = _DeadEnv()
    lvl2, _guid2 = drv.replay_live(
        _FakeArcade(dead_env),
        "sc25",
        "sc-1",
        [{"action": 6, "data": {"x": 1, "y": 1}}],
        _FakeMH(),
        corpus=None,
    )
    # the warmup step itself is skipped (guarded on frame is not None) -- only the
    # single real action from the main loop runs, unaffected by the warmup guard.
    assert dead_env.steps == [(6, {"x": 1, "y": 1}, {"policy": "offline_reproduced_replay"})]
    assert lvl2 == 1
