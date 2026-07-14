"""Tests for wiring ObjectHistorySaliencePrior into E3AgentPolicy/StepwiseExplorer's live
action_prior slot -- the deferred live-consuming mechanism task 10's own DONE note named.

Unlike task 9's InertClickSigPruner (which needed a brand-new rank_candidates call site),
`action_prior` is already a generic, externally-composable slot (see
arc_geometric_salience.GeometricSaliencePrior, the existing precedent for wrapping
ColorBlobSaliencePrior this way) that _ingest's OBSERVE hook and _candidates already consume
generically via hasattr checks -- so wiring this in needs no new hook call sites in
arc_competition_agent.py, only a new object_history_salience constructor param on
E3AgentPolicy that wraps whatever action_prior already resolved to.

Spec refs: REQ-ARC-FCP-5591-2, SCENARIO-ARC-FCP-5591-2-DEFAULT-OFF-PARITY,
SCENARIO-ARC-FCP-5591-2-OPT-IN-WRAPPING, SCENARIO-ARC-FCP-5591-2-INGEST-OBSERVES.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_color_blob_salience import ColorBlobSaliencePrior
from carnot.agentic.arc_competition_agent import (
    SUBMITTED_AGENT_CONFIG,
    SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED,
    E3AgentPolicy,
    StepwiseExplorer,
)
from carnot.agentic.arc_object_history_salience import ObjectHistorySaliencePrior


class _FakeFrame:
    """Minimal stand-in for an arcengine frame: only .frame is read by grid_of."""

    def __init__(self, grid: np.ndarray, *, levels_completed: int = 0) -> None:
        self.frame = grid
        self.state = "NOT_FINISHED"
        self.levels_completed = levels_completed


def test_scenario_5591_2_default_off_parity() -> None:
    """SCENARIO-ARC-FCP-5591-2-DEFAULT-OFF-PARITY: tracks
    SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED (currently False, pending a matched-budget A/B)
    rather than a hardcoded literal, and SUBMITTED_AGENT_CONFIG agrees -- the default action_prior
    is a plain, unwrapped ColorBlobSaliencePrior, byte-identical to before this task."""

    assert SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED is False
    assert (
        SUBMITTED_AGENT_CONFIG["object_history_salience_enabled"]
        is SUBMITTED_OBJECT_HISTORY_SALIENCE_ENABLED
    )

    pol = E3AgentPolicy("paritytest", proposer=None, value_head=lambda _f: 0.0)
    assert type(pol.explorer.action_prior) is ColorBlobSaliencePrior


def test_scenario_5591_2_opt_in_wraps_action_prior() -> None:
    """SCENARIO-ARC-FCP-5591-2-OPT-IN-WRAPPING: opting in wraps whatever action_prior already
    resolved to (the default ColorBlobSaliencePrior when none is externally supplied)."""

    pol = E3AgentPolicy(
        "paritytest", proposer=None, value_head=lambda _f: 0.0, object_history_salience=True
    )
    assert isinstance(pol.explorer.action_prior, ObjectHistorySaliencePrior)
    assert isinstance(pol.explorer.action_prior.base_prior, ColorBlobSaliencePrior)


def test_scenario_5591_2_opt_in_wraps_an_externally_supplied_action_prior() -> None:
    """Opting in wraps an EXTERNALLY-supplied action_prior too, not just the default --
    mirroring GeometricSaliencePrior's own compositional base_prior= contract."""

    custom_base = ColorBlobSaliencePrior(min_pixels=3)
    pol = E3AgentPolicy(
        "paritytest",
        proposer=None,
        value_head=lambda _f: 0.0,
        action_prior=custom_base,
        object_history_salience=True,
    )
    assert isinstance(pol.explorer.action_prior, ObjectHistorySaliencePrior)
    assert pol.explorer.action_prior.base_prior is custom_base


def test_scenario_5591_2_already_constructed_instance_passes_through() -> None:
    """Passing an already-constructed ObjectHistorySaliencePrior as action_prior directly
    (bypassing the object_history_salience flag entirely) works too, since action_prior is a
    generic externally-composable slot."""

    instance = ObjectHistorySaliencePrior(min_observations=7)
    explorer = StepwiseExplorer(action_prior=instance)
    assert explorer.action_prior is instance


def test_scenario_5591_2_ingest_observes_via_the_existing_generic_hook() -> None:
    """SCENARIO-ARC-FCP-5591-2-INGEST-OBSERVES: _ingest's pre-existing, generic
    hasattr(action_prior, "observe_transition") hook feeds ObjectHistorySaliencePrior real
    transitions with NO new hook code in arc_competition_agent.py -- the same site that already
    feeds dense_curiosity/controllable_novelty_policy/GeometricSaliencePrior/InertClickSigPruner."""

    prior = ObjectHistorySaliencePrior(min_observations=1)
    explorer = StepwiseExplorer(action_prior=prior)
    grid0 = np.zeros((10, 10), dtype=int)
    grid0[2:5, 2:5] = 5
    explorer._ingest(_FakeFrame(grid0.copy()))
    origin = explorer.cur

    explorer.awaiting = {
        "origin": origin,
        "action": 6,
        "data": {"x": 3, "y": 3},
        "grid": _FakeFrame(grid0.copy()),
        "level_before": int(explorer.best_level),
        "previous_frame": _FakeFrame(grid0.copy()),
    }
    grid1 = grid0.copy()
    grid1[3, 3] = 0
    explorer._ingest(_FakeFrame(grid1))

    assert prior.tracked_hash_count == 1


def test_scenario_5591_2_ingest_reset_clears_tally_on_level_up() -> None:
    """_ingest's existing hasattr(action_prior, "reset") level-up hook also reaches
    ObjectHistorySaliencePrior's reset(reset_to_prior=True) with no new wiring."""

    prior = ObjectHistorySaliencePrior(min_observations=1)
    explorer = StepwiseExplorer(action_prior=prior)
    grid0 = np.zeros((10, 10), dtype=int)
    grid0[2:5, 2:5] = 5
    explorer._ingest(_FakeFrame(grid0.copy()))
    origin = explorer.cur
    explorer.awaiting = {
        "origin": origin,
        "action": 6,
        "data": {"x": 3, "y": 3},
        "grid": _FakeFrame(grid0.copy()),
        "level_before": int(explorer.best_level),
        "previous_frame": _FakeFrame(grid0.copy()),
    }
    grid1 = grid0.copy()
    grid1[3, 3] = 0
    explorer._ingest(_FakeFrame(grid1))
    assert prior.tracked_hash_count == 1

    explorer.awaiting = {
        "origin": explorer.cur,
        "action": 6,
        "data": {"x": 3, "y": 3},
        "grid": _FakeFrame(grid1.copy()),
        "level_before": int(explorer.best_level),
        "previous_frame": _FakeFrame(grid1.copy()),
    }
    grid2 = grid1.copy()
    grid2[4, 4] = 0
    # A genuinely higher levels_completed (not a manually-bumped best_level) is what _ingest's
    # own level_increased detection (_level_of(latest) > previous_best_level) actually keys on.
    explorer._ingest(_FakeFrame(grid2, levels_completed=1))

    assert prior.tracked_hash_count == 0
