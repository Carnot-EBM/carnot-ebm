"""The recent-action-suffix state key must default OFF and must de-alias when ON.

THE DEFECT THIS GUARDS (exp6094 + the 2026-08-05 diagnosis). The adapter-free explorer's
node identity is a hash of the whole visible grid. On sc25 every root action is visually
inert on its FIRST application -- the game consumes it while hidden state advances -- so
every successor aliases into the root node, the root's untested list drains, and the search
terminates at 24 expansions / 1 "distinct" state, identically at 6000 and 30000 budgets.
A frontier collapse is a REPRESENTATION limit; no budget helps.

THE FIX UNDER TEST. `graph_explore_solve_v2(..., state_key_action_suffix_k=k)` (env flag
`CARNOT_ARC_STATE_KEY_SUFFIX_K`) appends the last k actions of the arriving path to the node
key -- the classic k-th-order remedy for a non-Markov observation. Generic: frames + the
agent's own actions only; no game ids.

WHAT IS ASSERTED, both directions, because either alone would be misleading:
  * with the flag OFF (the shipped default) keying is BYTE-IDENTICAL to the original single
    frame hash: on a hidden-state toy game the search still collapses to 1 state, so any
    future A/B against the default is interpretable;
  * with it ON (env flag or explicit parameter) the same toy game is SOLVED, because the
    visually-inert-but-state-advancing first action now creates a new frontier node instead
    of aliasing into its parent;
  * the explicit parameter beats the env flag (k=0 with the env set stays off), and garbage
    env values fail closed.

SCENARIO-ARC-GE-6110-STATE-KEY-SUFFIX-GATE (REQ-ARC-GE-6110)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from carnot.agentic.arc_graph_explore import graph_explore_solve_v2


class _HiddenStateToyEnv:
    """Minimal non-Markov-observation game, modelled on the measured sc25 root behaviour.

    Hidden counter c starts at 0. EVERY action advances c, but the visible frame is
    identical for c=0 and c=1 (the game "consumes" the first action invisibly, exactly what
    the 2026-08-05 sc25 probe measured: 8/8 root candidates IDENTICAL_TO_ROOT at one step,
    first visible change at depth 2). c=2 renders a distinct frame with levels_completed=1.
    """

    def __init__(self) -> None:
        self.c = 0

    def reset(self) -> Any:
        self.c = 0
        return self._frame()

    def _frame(self) -> Any:
        if self.c >= 2:
            return SimpleNamespace(
                frame=[[9]], levels_completed=1, available_actions=[1, 2], state=""
            )
        return SimpleNamespace(frame=[[0]], levels_completed=0, available_actions=[1, 2], state="")

    def step(self, action: Any, data: Any = None, reasoning: Any = None) -> Any:
        self.c += 1
        return self._frame()


@pytest.fixture(autouse=True)
def _shipped_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from the SHIPPED environment, so a stray export in the operator's
    shell cannot make a default-OFF assertion pass (or fail) for the wrong reason."""
    monkeypatch.delenv("CARNOT_ARC_STATE_KEY_SUFFIX_K", raising=False)


def _run(k: int | None, budget: int = 50) -> tuple[Any, int, dict]:
    stats: dict = {}
    traj, lvl = graph_explore_solve_v2(
        _HiddenStateToyEnv(),
        0,
        max_expansions=budget,
        max_depth=6,
        state_key_action_suffix_k=k,
        stats=stats,
    )
    return traj, lvl, stats


class TestDefaultOffReproducesTheCollapse:
    def test_default_collapses_to_one_state(self) -> None:
        """OFF direction: the shipped default keys on the bare frame hash, so the toy game's
        visually-inert first actions alias into the root and the frontier drains -- the
        exp6094 sc25 signature (1 state, expansions == the root's candidate count, no
        advance), NOT a budget wall."""
        traj, lvl, stats = _run(None)
        assert traj is None and lvl == 0
        assert stats["state_key_action_suffix_k"] == 0
        assert stats["states"] == 1
        assert stats["distinct_frames"] == 1
        assert stats["expansions"] == 2  # both root candidates tested, nothing else
        assert stats["expansions"] < stats["max_expansions"]  # frontier drain, not budget

    def test_garbage_env_value_fails_closed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CARNOT_ARC_STATE_KEY_SUFFIX_K", "banana")
        traj, _lvl, stats = _run(None)
        assert traj is None
        assert stats["state_key_action_suffix_k"] == 0

    def test_explicit_zero_beats_env_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The parameter is the authoritative switch: an experiment pinning k=0 as its
        control arm must get the control even under an operator's exported flag."""
        monkeypatch.setenv("CARNOT_ARC_STATE_KEY_SUFFIX_K", "1")
        traj, _lvl, stats = _run(0)
        assert traj is None
        assert stats["state_key_action_suffix_k"] == 0
        assert stats["states"] == 1


class TestSuffixKeyDealiases:
    def test_env_flag_enables_and_solves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ON direction (env flag): frame hash + last-1-action makes the consumed first
        action a NEW node, whose expansion reaches the visible win two steps deep."""
        monkeypatch.setenv("CARNOT_ARC_STATE_KEY_SUFFIX_K", "1")
        traj, lvl, stats = _run(None)
        assert traj is not None and lvl == 1
        assert stats["state_key_action_suffix_k"] == 1
        assert len(traj) == 2  # the true shortest path: consumed action, then the winner

    def test_explicit_parameter_enables_and_solves(self) -> None:
        traj, lvl, stats = _run(1)
        assert traj is not None and lvl == 1
        assert stats["state_key_action_suffix_k"] == 1

    def test_distinct_frames_counts_frames_not_suffix_splits(self) -> None:
        """The inflation-accounting stat: with the suffix on, `states` may exceed
        `distinct_frames`; the gap is the price paid for de-aliasing and must be visible
        to any A/B reading the stats."""
        _traj, _lvl, stats = _run(1)
        assert stats["distinct_frames"] <= stats["states"]
