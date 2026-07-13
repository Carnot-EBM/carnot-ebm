"""Regression tests for rule-based live HUD-cell masking (E1, arXiv:2512.24156).

Why this file exists: a comparative gap analysis against the hidden-leaderboard
3rd-place solver ("just-explore") found that `StepwiseExplorer.hud_mask` had existed
as a constructor parameter since before this requirement (already consumed by
`_hash` to collapse masked cells out of node identity) but was never populated on
the live path — a ticking score/timer/step-counter HUD cell made every tick look
like a brand-new state to the live dedup. These tests pin the fix: a rule-based,
zero-action-cost mask (`_compute_hud_mask_from_frame`), computed at most once per
`StepwiseExplorer` instance, wired through `E3AgentPolicy` and `CarnotAgentPolicy`,
gated OFF by default pending offline validation.

Spec: REQ-ARC-WMTE-5583, SCENARIO-ARC-WMTE-5583-STATUS-BAR-CHANGE-DEDUPS,
SCENARIO-ARC-WMTE-5583-REAL-CHANGE-STILL-DISTINGUISHED,
SCENARIO-ARC-WMTE-5583-DEFAULT-OFF-PARITY.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_competition_agent import (
    SUBMITTED_AGENT_CONFIG,
    SUBMITTED_AUTO_HUD_MASK_ENABLED,
    CarnotAgentPolicy,
    E3AgentPolicy,
    StepwiseExplorer,
    _compute_hud_mask_from_frame,
)


class _FakeFrame:
    """Minimal stand-in for an arcengine frame: only .frame is read by grid_of."""

    def __init__(self, grid: np.ndarray) -> None:
        self.frame = grid
        self.state = "NOT_FINISHED"
        self.levels_completed = 0


def _status_bar_grid() -> np.ndarray:
    grid = np.zeros((20, 20), dtype=int)
    grid[0, :] = 16  # full-width, edge-touching, thin -> status-bar-like
    grid[10:12, 10:12] = 7  # a compact non-edge board blob, should NOT be masked
    return grid


def test_compute_hud_mask_from_frame_masks_status_bar_not_board_blob() -> None:
    """SCENARIO-ARC-WMTE-5583-STATUS-BAR-CHANGE-DEDUPS (part 1)."""

    mask = _compute_hud_mask_from_frame(_FakeFrame(_status_bar_grid()))

    assert mask is not None
    assert mask.shape == (20, 20)
    assert bool(mask[0, :].all())
    assert not bool(mask[10:12, 10:12].any())


def test_compute_hud_mask_from_frame_returns_none_for_none_frame() -> None:
    assert _compute_hud_mask_from_frame(None) is None


def test_compute_hud_mask_from_frame_returns_none_when_nothing_looks_like_a_status_bar() -> None:
    grid = np.zeros((10, 10), dtype=int)
    grid[3:5, 3:5] = 7  # a compact centered blob only; nothing edge-spanning
    assert _compute_hud_mask_from_frame(_FakeFrame(grid)) is None


def test_step_wise_explorer_auto_hud_mask_dedups_status_bar_only_change() -> None:
    """SCENARIO-ARC-WMTE-5583-STATUS-BAR-CHANGE-DEDUPS (part 2)."""

    explorer = StepwiseExplorer(auto_hud_mask=True)
    assert explorer.hud_mask is None
    assert explorer._hud_mask_attempted is False

    grid = _status_bar_grid()
    explorer._ingest(_FakeFrame(grid.copy()))
    first_hash = explorer.cur

    assert explorer.hud_mask is not None
    assert explorer._hud_mask_attempted is True

    # A ticking status-bar counter: only row 0's value changes (16 -> 17).
    ticked = grid.copy()
    ticked[0, :] = 17
    explorer._ingest(_FakeFrame(ticked))

    assert explorer.cur == first_hash


def test_step_wise_explorer_auto_hud_mask_still_distinguishes_real_board_change() -> None:
    """SCENARIO-ARC-WMTE-5583-REAL-CHANGE-STILL-DISTINGUISHED."""

    explorer = StepwiseExplorer(auto_hud_mask=True)
    grid = _status_bar_grid()
    explorer._ingest(_FakeFrame(grid.copy()))
    first_hash = explorer.cur

    changed = grid.copy()
    changed[15, 15] = 9  # a genuine, non-status-bar board change
    explorer._ingest(_FakeFrame(changed))

    assert explorer.cur != first_hash


def test_step_wise_explorer_auto_hud_mask_attempts_exactly_once() -> None:
    """A game with no status-bar-like blob must not retry mask discovery every frame."""

    explorer = StepwiseExplorer(auto_hud_mask=True)
    blank = np.zeros((10, 10), dtype=int)
    explorer._ingest(_FakeFrame(blank))

    assert explorer._hud_mask_attempted is True
    assert explorer.hud_mask is None  # no status bar in a blank frame

    # A second frame must not re-attempt (would be wasted compute, not a correctness
    # bug, but pins the "at most once" contract from REQ-ARC-WMTE-5583).
    explorer._ingest(_FakeFrame(_status_bar_grid()))
    assert explorer.hud_mask is None


def test_step_wise_explorer_explicit_hud_mask_disables_auto_detection() -> None:
    """An explicitly-passed static hud_mask must never be overridden by auto-detection."""

    explicit = np.zeros((20, 20), dtype=bool)
    explicit[5, 5] = True
    explorer = StepwiseExplorer(hud_mask=explicit, auto_hud_mask=True)

    assert explorer._hud_mask_attempted is True  # pre-armed, so auto-detect never fires
    explorer._ingest(_FakeFrame(_status_bar_grid()))

    assert explorer.hud_mask is explicit


def test_step_wise_explorer_auto_hud_mask_defaults_to_submitted_flag() -> None:
    """Tracks SUBMITTED_AUTO_HUD_MASK_ENABLED rather than a hardcoded literal -- see
    REQ-ARC-WMTE-5583's 2026-07-12 RESOLUTION for why the value is True as of that
    date (deferred to live-submission telemetry, not a hardcoded True/False pin)."""

    explorer = StepwiseExplorer()
    assert explorer.auto_hud_mask == SUBMITTED_AUTO_HUD_MASK_ENABLED


def test_e3_agent_policy_default_auto_hud_mask_matches_submitted_config() -> None:
    """SCENARIO-ARC-WMTE-5583-DEFAULT-PARITY."""

    pol = E3AgentPolicy("paritytest", proposer=None, value_head=lambda _f: 0.0)

    assert pol.explorer.auto_hud_mask == SUBMITTED_AGENT_CONFIG["auto_hud_mask_enabled"]
    assert pol.explorer.auto_hud_mask == SUBMITTED_AUTO_HUD_MASK_ENABLED


def test_e3_agent_policy_auto_hud_mask_can_be_opted_in() -> None:
    pol = E3AgentPolicy("paritytest", proposer=None, value_head=lambda _f: 0.0, auto_hud_mask=True)

    assert pol.explorer.auto_hud_mask is True


def test_carnot_agent_policy_threads_auto_hud_mask_to_explorer() -> None:
    pol = CarnotAgentPolicy("unclaimedgame", solutions={}, force_explore=True, auto_hud_mask=True)

    assert pol.explorer is not None
    assert pol.explorer.auto_hud_mask is True
