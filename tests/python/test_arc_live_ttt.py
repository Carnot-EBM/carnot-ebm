"""Unit tests for the live-TTT world model (LiveTTTWorldModel) -- 2026-06-21.

REQ: the live test-time-learning world model must (1) learn an exact transition table from played
transitions and reproduce them via its engine(grid, action, data)->grid contract; (2) recognize observed
win-states for the planner's is_level_complete; (3) score held-out transitions through the
WorldModelVerifier trust gate without crashing. These are the contracts the conductor wires into
E3AgentPolicy._induce_and_plan (replacing the failing LLM engine).

SCENARIO-TTT-1: an observed transition is reproduced EXACTLY by the engine (L0 exact table).
SCENARIO-TTT-2: a grid reached at a level-up is recognized as a win-state.
SCENARIO-TTT-3: the engine never crashes on an unseen (state, action) -- returns a same-shape grid.
SCENARIO-TTT-4: trust() returns a [0,1] held-out accuracy (the WorldModelVerifier gate input).
"""

import numpy as np

from carnot.agentic.arc_live_ttt import LiveTTTWorldModel, action_key, gated_engine_from_transitions


def test_scenario_ttt_1_engine_reproduces_observed_transition() -> None:
    """SCENARIO-TTT-1: L0 exact table reproduces a played click transition."""
    m = LiveTTTWorldModel("toy")
    g = np.zeros((5, 5), dtype=np.int16)
    ng = g.copy()
    ng[2, 3] = 7
    m.observe(g, 6, {"x": 3, "y": 2}, ng)
    assert np.array_equal(np.asarray(m.engine(g, 6, {"x": 3, "y": 2})), ng)


def test_scenario_ttt_2_win_state_recognized() -> None:
    """SCENARIO-TTT-2: a grid reached at a level-up is a win-state; an unrelated grid is not."""
    m = LiveTTTWorldModel("toy")
    g = np.zeros((4, 4), dtype=np.int16)
    win = g.copy()
    win[0, 0] = 1
    m.observe(g, 1, None, win, level_before=0, level_after=1)
    assert m.is_level_complete(win)
    assert not m.is_level_complete(g)


def test_scenario_ttt_3_engine_never_crashes_on_unseen() -> None:
    """SCENARIO-TTT-3: an unseen (state, action) returns a same-shape grid (identity fallback), no crash."""
    m = LiveTTTWorldModel("toy")
    g = np.zeros((4, 4), dtype=np.int16)
    out = np.asarray(m.engine(g, 3, None))  # nothing observed, no L1 fit -> identity
    assert out.shape == g.shape


def test_scenario_ttt_4_trust_is_a_unit_interval_accuracy() -> None:
    """SCENARIO-TTT-4: trust() returns a held-out accuracy in [0,1] and exact-reproduced held-outs score 1."""
    from carnot.agentic.arc_executable_world_model import Transition

    m = LiveTTTWorldModel("toy")
    g = np.zeros((4, 4), dtype=np.int16)
    ng = g.copy()
    ng[1, 1] = 2
    m.observe(g, 6, {"x": 1, "y": 1}, ng)
    held = [
        Transition(
            grid=g, action=6, data={"x": 1, "y": 1}, next_grid=ng, level_before=0, level_after=0
        )
    ]
    acc = m.trust(held)
    assert 0.0 <= acc <= 1.0
    assert acc == 1.0  # the engine reproduces this exact transition from its L0 table


def test_action_key_canonical_form() -> None:
    """action_key matches the live agent's _action_key: (6,x,y) for clicks, (id,) otherwise."""
    assert action_key(6, {"x": 5, "y": 9}) == (6, 5, 9)
    assert action_key(3, None) == (3,)
    assert action_key(6, None) == (6,)  # click without coords -> keyboard-style key


def test_req_arc_wmte_5157_gated_engine_accepts_redraw_prior_evidence() -> None:
    """SCENARIO-ARC-WMTE-5157-REDRAW-WARM-START: prior evidence seeds residual TTT."""

    from carnot.agentic.arc_executable_world_model import Transition

    grid = np.zeros((3, 3), dtype=np.int16)
    next_grid = grid.copy()
    next_grid[1, 1] = 4
    prior = [
        Transition(
            grid=grid,
            action=1,
            data=None,
            next_grid=next_grid,
            level_before=0,
            level_after=0,
        )
        for _ in range(8)
    ]

    cold_engine, _cold_done, cold_diag = gated_engine_from_transitions(
        "toy",
        [],
        holdout_frac=0.0,
        trust_threshold=0.0,
        dynamics_backend="dsl",
    )
    warm_engine, _warm_done, warm_diag = gated_engine_from_transitions(
        "toy",
        [],
        prior_transitions=prior,
        holdout_frac=0.0,
        trust_threshold=0.0,
        dynamics_backend="dsl",
    )

    assert cold_engine is None
    assert cold_diag["skip"] == "too_few_transitions"
    assert warm_engine is not None
    assert warm_diag["warm_start"] is True
    assert warm_diag["prior_transition_count"] == 8
    assert warm_diag["residual_adapter"] == "redraw_frozen_base_plus_target_residual"
    assert np.array_equal(np.asarray(warm_engine(grid, 1, None)), next_grid)
