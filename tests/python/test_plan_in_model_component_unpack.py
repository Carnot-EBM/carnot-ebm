"""Regression: plan_in_model's _model_candidates must not crash on the 5-tuple _components_detailed shape.

`_components_detailed` was widened from (cy,cx,area,color) to (cy,cx,area,color,is_grid_fallback) in the
GAP-ARC-BP35-CLICK-CANDIDATE-GENERATION-MISS fix (commit 2f0760307). That updated the arc_graph_explore
consumer defensively but MISSED _model_candidates, whose rigid `for cy,cx,_a,_c in comps` unpack then raised
ValueError on ANY grid with components -- silently disabling the entire plan_in_model world-model planning
tier for such games (e.g. tu93's 65 components). Spec: REQ-ARC-WMTE-5841.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_executable_world_model import _model_candidates


def test_model_candidates_handles_component_grid_without_crashing():
    # a grid with several distinct-color objects on a background -> real (5-tuple) components
    g = np.zeros((16, 16), dtype=np.int16)
    g[2:4, 2:4] = 3
    g[8:10, 10:12] = 5
    g[13, 13] = 7
    cands = _model_candidates(g)  # must NOT raise (the regression raised ValueError here)
    assert len(cands) >= 5  # the 5 directional/confirm actions always present
    assert any(c["action"] == 6 for c in cands)  # at least one click candidate from a component
    assert all(set(c) <= {"action", "data"} for c in cands)
    # click candidates carry integer x/y within the grid
    for c in cands:
        if c["action"] == 6 and c["data"] is not None:
            assert 0 <= int(c["data"]["x"]) < 16 and 0 <= int(c["data"]["y"]) < 16


def test_model_candidates_empty_on_blank_grid():
    cands = _model_candidates(np.zeros((8, 8), dtype=np.int16))
    assert [c["action"] for c in cands] == [1, 2, 3, 4, 5]  # no components -> only directional/confirm


def test_plan_in_model_keeps_goal_energy_and_diagnostics_params():
    # Regression guard for the REQ-ARC-WMTE-5845 nav goal-energy wiring: E3AgentPolicy._call_plan_in_model
    # only forwards goal_energy/diagnostics when the planner's signature accepts them
    # (_planner_accepts_goal_energy). If someone drops these params from plan_in_model, the nav best-first
    # override would silently degrade to plain BFS -- this catches that.
    import inspect

    from carnot.agentic.arc_executable_world_model import plan_in_model

    params = inspect.signature(plan_in_model).parameters
    assert "goal_energy" in params
    assert "diagnostics" in params
    assert "max_nodes" in params
