"""REQ-ARC-ARM-6434: graph explorer uses certified suffixes only after alias evidence."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from carnot.agentic.arc_graph_explore import graph_explore_solve_v2


class _HiddenProgressEnv:
    """A tiny game where the first action changes hidden state but not pixels."""

    def __init__(self) -> None:
        self.counter = 0

    def reset(self) -> Any:
        self.counter = 0
        return self._frame()

    def _frame(self) -> Any:
        if self.counter >= 2:
            return SimpleNamespace(
                frame=[[9]], levels_completed=1, available_actions=[1, 2], state=""
            )
        return SimpleNamespace(frame=[[0]], levels_completed=0, available_actions=[1, 2], state="")

    def step(self, action: Any, data: Any = None, reasoning: Any = None) -> Any:
        self.counter += 1
        return self._frame()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CARNOT_ARC_STATE_KEY_SUFFIX_K", raising=False)
    monkeypatch.delenv("CARNOT_ARC_COLLISION_CERTIFIED_STATE_KEY_SUFFIX", raising=False)


def test_default_off_still_collapses_without_certificate() -> None:
    """SCENARIO-ARC-ARM-6434-NO-CERTIFICATE: shipped default does not extend keys."""
    stats: dict = {}
    traj, level = graph_explore_solve_v2(
        _HiddenProgressEnv(),
        0,
        max_expansions=30,
        max_depth=6,
        stats=stats,
    )

    assert traj is None and level == 0
    assert stats["state_key_collision_certified_suffix_enabled"] is False
    assert stats["state_key_collision_certificate_count"] == 0
    assert stats["state_key_effective_suffix_max_k"] == 0


def test_certified_opt_in_dealiases_after_collision() -> None:
    """SCENARIO-ARC-ARM-6434-CERTIFICATE: opt-in suffix solves the alias toy."""
    stats: dict = {}
    traj, level = graph_explore_solve_v2(
        _HiddenProgressEnv(),
        0,
        max_expansions=30,
        max_depth=6,
        collision_certified_state_key_suffix=True,
        stats=stats,
    )

    assert traj is not None and level == 1
    assert stats["state_key_action_suffix_k"] == 0
    assert stats["state_key_collision_certified_suffix_enabled"] is True
    assert stats["state_key_collision_certificate_count"] >= 1
    assert stats["state_key_effective_suffix_max_k"] == 1
    assert stats["state_key_collision_certificates"][0]["minimal_suffix_k"] == 1


def test_env_flag_enables_certified_route(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-ARM-6434-MATCHED-AB: treatment is an explicit opt-in flag."""
    monkeypatch.setenv("CARNOT_ARC_COLLISION_CERTIFIED_STATE_KEY_SUFFIX", "1")
    stats: dict = {}

    traj, level = graph_explore_solve_v2(
        _HiddenProgressEnv(),
        0,
        max_expansions=30,
        max_depth=6,
        stats=stats,
    )

    assert traj is not None and level == 1
    assert stats["state_key_collision_certified_suffix_enabled"] is True


def test_static_suffix_knob_remains_explicit_diagnostic_not_certified() -> None:
    """SCENARIO-ARC-ARM-6434-MATCHED-AB: the old suffix does not emit certificates."""
    stats: dict = {}

    traj, level = graph_explore_solve_v2(
        _HiddenProgressEnv(),
        0,
        max_expansions=30,
        max_depth=6,
        state_key_action_suffix_k=1,
        stats=stats,
    )

    assert traj is not None and level == 1
    assert stats["state_key_action_suffix_k"] == 1
    assert stats["state_key_collision_certified_suffix_enabled"] is False
    assert stats["state_key_collision_certificate_count"] == 0
